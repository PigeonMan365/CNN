#!/usr/bin/env python3
# training/dataset.py
"""
ByteImageDataset — loads 8-bit grayscale PNGs listed in conversion_log.csv.

Key features:
- Reads a filtered CSV (train.py handles mode filter) and builds an item list:
    items[i] = (rel_path, label:int, sha256:str)
- Returns (tensor, label, rel_path) where tensor is FloatTensor [1, H, W] normalized to 0..1.
- Optional SSD staging cache (FileLRU): copies source PNG from images_root (HDD or elsewhere)
  into a fast cache_root with LRU byte cap. Enabled via use_disk_cache.
- Optional in-RAM decoded tensor cache (TensorLRU): avoids re-decoding PNGs repeatedly.
- Absolute paths in CSV are honored; otherwise rel_path is resolved under images_root.

Constructor args (most are passed from train.py via config):
    csv_path: str
    images_root: str
    normalize: "01" | None
    use_disk_cache: bool = False
    cache_root: str = "cache"
    cache_max_bytes: str = "40GB"   # e.g., "10GB", "512MB"
    decode_cache_mem_mb: int = 0    # 0 disables RAM cache

Notes:
- Does not perform mode filtering; train.py writes a filtered CSV.
- Group information is not required here (train.py reads sha256 directly for splits).
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow is required. Install with: pip install pillow") from e

# Optional caching utilities.
# If utils/cache_io.py is not present yet, caching gracefully disables.
try:
    from utils.cache_io import FileLRU, TensorLRU, parse_bytes
except Exception:
    FileLRU = None
    TensorLRU = None
    parse_bytes = None


def _parse_label(v) -> int:
    s = str(v).strip().lower()
    if s in ("0", "benign"):
        return 0
    if s in ("1", "malware"):
        return 1
    # Be strict: dataset must be clean by now.
    raise ValueError(f"Unrecognized label in CSV: {v!r} (expected 0/1 or benign/malware)")


class ByteImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_root: str,
        normalize: Optional[str] = "01",
        use_disk_cache: bool = False,
        cache_root: str = "cache",
        cache_max_bytes: str = "40GB",
        decode_cache_mem_mb: int = 0,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.images_root = Path(images_root)
        self.normalize = (normalize or "01").lower() if normalize else None
        self.target_size = target_size  # (width, height) or None to use image as-is

        # Load CSV rows
        self.items: List[Tuple[str, int, str]] = []  # (rel_or_abs, label, sha256)
        self._load_csv()

        # SSD staging cache (LRU on-disk)
        self.use_disk_cache = bool(use_disk_cache) and FileLRU is not None and parse_bytes is not None
        self.file_cache: Optional[FileLRU] = None
        if self.use_disk_cache:
            self.file_cache = FileLRU(Path(cache_root), parse_bytes(cache_max_bytes))

        # In-RAM tensor cache (LRU by bytes)
        self.tensor_cache: Optional[TensorLRU] = None
        if TensorLRU is not None and int(decode_cache_mem_mb) > 0:
            self.tensor_cache = TensorLRU(int(decode_cache_mem_mb) * 1024 * 1024)

    # ------------------------ CSV ingest ------------------------

    def _load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        with self.csv_path.open("r", newline="") as f:
            r = csv.DictReader(f)
            required = {"rel_path", "label", "mode", "sha256"}
            if not required.issubset(set(r.fieldnames or [])):
                raise ValueError(
                    f"{self.csv_path} missing required columns. "
                    f"Found={r.fieldnames}, required={sorted(required)}"
                )
            for row in r:
                rel = row["rel_path"].strip()
                label = _parse_label(row["label"])
                sha = (row.get("sha256") or "").strip()
                self.items.append((rel, label, sha))

    # ------------------------ IO helpers ------------------------

    def _resolve_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return (self.images_root / p).resolve()

    def _stage_path_if_needed(self, p: Path) -> Path:
        if self.file_cache is None:
            return p
        return self.file_cache.stage(p)

    def _png_to_tensor(self, path: Path) -> torch.Tensor:
        # Build a cache key that invalidates when file changes or target_size changes.
        try:
            st = path.stat()
            key = (str(path), st.st_mtime_ns, st.st_size, self.normalize, self.target_size)
        except FileNotFoundError:
            key = (str(path), 0, 0, self.normalize, self.target_size)

        if self.tensor_cache is not None:
            t = self.tensor_cache.get(key)
            if t is not None:
                return t

        img = Image.open(path).convert("L")  # 8-bit grayscale
        
        # Resize if target_size is specified and image doesn't match
        if self.target_size is not None:
            target_w, target_h = self.target_size
            if img.size != (target_w, target_h):
                # Use LANCZOS resampling for quality (same as conversion uses)
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        arr = np.array(img, dtype=np.float32)
        if self.normalize == "01":
            arr = arr / 255.0
        t = torch.from_numpy(arr)[None, ...]  # [1,H,W]

        if self.tensor_cache is not None:
            self.tensor_cache.put(key, t)
        return t

    # ------------------------- Dataset API -------------------------

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel, label, sha = self.items[idx]
        p = self._resolve_path(rel)
        p = self._stage_path_if_needed(p)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        x = self._png_to_tensor(p)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, rel  # keep rel_path for logging/debug

