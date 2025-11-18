#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive converter & indexer for the project.

Features:
- Reads config.yaml:
    - paths.input_roots (list[str])
    - train_io.images_root (preferred) or paths.images_root (fallback)
- Ensures folder skeleton:
    dataset/output/{benign,malware}/{resize,truncate}, logs/, etc.
- Converts each input file found under <input_root>/{benign,malware}/
  into BOTH modes {resize, truncate} with deterministic filenames:
    images_root/<label>/<mode>/<sha256>.png
- Idempotent: skips conversion if the output PNG already exists.
- Rebuilds logs/conversion_log.csv by scanning images_root.

Preprocessing Methods:
- resize: Maps all bytes to grayscale with dynamic height,
  pads only the final row, then resizes to 256×256 using bilinear interpolation.
  Preserves all bytes and ensures uniform CNN input shape.
- truncate (entropy-aware): Divides file into 512-byte chunks, computes Shannon
  entropy per chunk, selects top-N highest-entropy chunks to fill 256×256 image.
  Preserves the most informative regions and avoids wasting space on low-entropy
  headers or padding.

- Usable as a script or library:
    - convert_file(path, mode, images_root, label) -> Path|None
    - rebuild_conversion_log(images_root, csv_path) -> int

CLI:
    python preprocessing/convert.py --config config.yaml
    python preprocessing/convert.py --rebuild-only
    python preprocessing/convert.py --skip-convert
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Iterable, Tuple, Optional

# --- Optional nice progress ---
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # fallback stub
        return x

# --- Pillow for PNG writing ---
try:
    from PIL import Image
except Exception:
    print("[convert] Missing Pillow. Install: pip install pillow", file=sys.stderr)
    raise

MODES = ("resize", "truncate")
LABELS = ("benign", "malware")
FIXED_PIXELS = 256 * 256  # 65,536

# ----------------- small utils -----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def load_yaml(path: str = "config.yaml") -> dict:
    try:
        import yaml
    except Exception:
        print("[convert] PyYAML not installed. pip install pyyaml", file=sys.stderr)
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[convert] WARNING: missing {path}; using built-in defaults.", file=sys.stderr)
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception as e:
        print(f"[convert] WARNING: failed to parse {path}: {e}", file=sys.stderr)
        return {}

# ----------------- byte -> image kernels -----------------

def _parse_target_size(value, default: tuple[int, int]) -> tuple[int, int]:
    """
    Parse target size from config value (list, string, or tuple).
    
    Args:
        value: Config value (e.g., [64, 64], "64,64", or (64, 64))
        default: Default tuple to return if value is invalid
    
    Returns:
        Tuple of (width, height) as integers
    """
    if value is None:
        return default
    
    # Handle list
    if isinstance(value, list):
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
        elif len(value) == 1:
            val = int(value[0])
            return (val, val)
    
    # Handle string like "64,64" or "64x64"
    if isinstance(value, str):
        value = value.strip()
        if "," in value:
            parts = value.split(",")
        elif "x" in value.lower():
            parts = value.lower().split("x")
        else:
            try:
                val = int(value)
                return (val, val)
            except ValueError:
                return default
        
        if len(parts) >= 2:
            return (int(parts[0].strip()), int(parts[1].strip()))
        elif len(parts) == 1:
            val = int(parts[0].strip())
            return (val, val)
    
    # Handle tuple
    if isinstance(value, tuple):
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
        elif len(value) == 1:
            val = int(value[0])
            return (val, val)
    
    return default

def bytes_to_img_resize(b: bytes, width: int = 256, target_size: tuple[int, int] | None = None) -> Image.Image:
    """
    Resize method: Map all bytes to grayscale pixels, then resize to target_size.
    
    Steps:
    1. Map all bytes to grayscale pixels (0-255) in byte order
    2. Use fixed width (256 pixels by default)
    3. Compute dynamic height: ceil(len(bytes) / width)
    4. Pad only the final row to complete the last line
    5. Resize the resulting image to target_size using bilinear interpolation
    
    Args:
        b: Raw byte stream
        width: Fixed image width for initial mapping (default 256)
        target_size: Target image size as (width, height) tuple. If None, reads from config.yaml
                     or defaults to (64, 64).
    
    Returns:
        PIL Image in grayscale mode, size target_size
        Note: Image is saved as 8-bit PNG; normalization to [0,1] occurs during dataset loading.
    """
    import math
    
    # If target_size not provided, try to read from config
    if target_size is None:
        try:
            cfg = load_yaml()
            training = cfg.get("training", {})
            resize_size = training.get("resize_target_size")
            if resize_size is not None:
                target_size = _parse_target_size(resize_size, (64, 64))
            else:
                target_size = (64, 64)
        except Exception:
            target_size = (64, 64)  # Fallback to default
    
    # Handle target_size as tuple or int (backward compatibility)
    if isinstance(target_size, (int, float)):
        target_size = (int(target_size), int(target_size))
    target_width, target_height = int(target_size[0]), int(target_size[1])
    
    n = len(b)
    if n == 0:
        # Empty file -> create empty image of target size
        return Image.new("L", (target_width, target_height), 0)
    
    # Compute dynamic height
    height = math.ceil(n / width)
    
    # Create image with dynamic height
    # Map bytes directly to pixels (each byte is already 0-255)
    # Pad only the final row to complete it (not the whole image)
    # Calculate how many bytes are in the last (incomplete) row
    bytes_in_full_rows = (height - 1) * width
    bytes_in_last_row = n - bytes_in_full_rows
    
    # Pad only the last row if it's incomplete
    if bytes_in_last_row < width:
        padding_needed = width - bytes_in_last_row
        padded_bytes = b + bytes(padding_needed)
    else:
        # Last row is complete or we have exactly the right number of pixels
        padded_bytes = b[:width * height]  # Ensure we don't exceed total_pixels
    
    # Create image from bytes
    img = Image.frombytes("L", (width, height), padded_bytes)
    
    # Resize to target_size using bilinear interpolation
    img_resized = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    
    return img_resized

def bytes_to_img_entropy_truncate(b: bytes, chunk_size: int = 512, target_size: tuple[int, int] | None = None) -> Image.Image:
    """
    Entropy-aware truncate: Select highest-entropy chunks, then map to target_size.
    
    Steps:
    1. Divide file into chunks (default 512 bytes per chunk)
    2. Compute Shannon entropy per chunk
    3. Select top-N chunks with highest entropy (enough to fill target_size[0] * target_size[1] bytes)
    4. Concatenate selected chunks and reshape to target_size image
    
    For files smaller than target_size[0] * target_size[1] bytes, still applies entropy selection among
    available chunks, then pads remainder with zeros.
    
    Args:
        b: Raw byte stream
        chunk_size: Size of each chunk for entropy calculation (default 512)
        target_size: Target image size as (width, height) tuple. If None, reads from config.yaml
                     or defaults to (256, 256).
    
    Returns:
        PIL Image in grayscale mode, size target_size
        Note: Image is saved as 8-bit PNG; normalization to [0,1] occurs during dataset loading.
    """
    import math
    import collections
    
    # If target_size not provided, try to read from config
    if target_size is None:
        try:
            cfg = load_yaml()
            training = cfg.get("training", {})
            truncate_size = training.get("truncate_target_size")
            if truncate_size is not None:
                target_size = _parse_target_size(truncate_size, (256, 256))
            else:
                target_size = (256, 256)
        except Exception:
            target_size = (256, 256)  # Fallback to default
    
    # Handle target_size as tuple or int (backward compatibility)
    if isinstance(target_size, (int, float)):
        target_size = (int(target_size), int(target_size))
    target_width, target_height = int(target_size[0]), int(target_size[1])
    
    target_bytes = target_width * target_height  # e.g., 65,536 for 256×256
    
    n = len(b)
    if n == 0:
        return Image.new("L", (target_width, target_height), 0)
    
    # For files smaller than target_bytes, take all available chunks
    if n <= target_bytes:
        # Still compute entropy for consistency, but take all chunks
        num_chunks = math.ceil(n / chunk_size) if chunk_size > 0 else 1
    else:
        num_chunks = math.ceil(n / chunk_size) if chunk_size > 0 else 1
    
    # Compute entropy per chunk
    chunk_entropies = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        chunk = b[start_idx:end_idx]
        
        if len(chunk) == 0:
            entropy = 0.0
        else:
            # Compute Shannon entropy: H = -Σ(p(x) * log2(p(x)))
            byte_counts = collections.Counter(chunk)
            entropy = 0.0
            chunk_len = len(chunk)
            for count in byte_counts.values():
                p = count / chunk_len
                if p > 0:
                    entropy -= p * math.log2(p)
        
        chunk_entropies.append((entropy, i, chunk))
    
    # Sort by entropy (descending), then by chunk index (for deterministic tie-breaking)
    chunk_entropies.sort(key=lambda x: (-x[0], x[1]))
    
    # Select top chunks to fill target_bytes
    selected_bytes = bytearray()
    bytes_needed = target_bytes
    
    for entropy, idx, chunk in chunk_entropies:
        if bytes_needed <= 0:
            break
        take = min(bytes_needed, len(chunk))
        selected_bytes.extend(chunk[:take])
        bytes_needed -= take
    
    # Pad remainder with zeros if needed
    if len(selected_bytes) < target_bytes:
        selected_bytes.extend(bytes(target_bytes - len(selected_bytes)))
    
    # Create image from selected bytes with target_size dimensions
    img = Image.frombytes("L", (target_width, target_height), bytes(selected_bytes[:target_bytes]))
    
    return img

def bytes_to_img_truncate(b: bytes, target_size: tuple[int, int] = (256, 256)) -> Image.Image:
    """
    Truncate mode: Now uses entropy-aware truncation instead of simple first-N-bytes.
    
    This upgrade preserves the most informative regions of the file by:
    1. Dividing file into 512-byte chunks
    2. Computing Shannon entropy per chunk
    3. Selecting top-N chunks with highest entropy (enough to fill target_size[0] * target_size[1] bytes)
    4. Concatenating selected chunks and mapping to target_size image
    
    This avoids wasting space on low-entropy headers or padding.
    
    Args:
        b: Raw byte stream
        target_size: Target image size as (width, height) tuple (default (256, 256))
    """
    return bytes_to_img_entropy_truncate(b, chunk_size=512, target_size=target_size)

# ----------------- public API -----------------

def convert_file(input_path: Path, mode: str, images_root: Path, label: str,
                 resize_target_size: tuple[int, int] = (64, 64),
                 truncate_target_size: tuple[int, int] = (256, 256)) -> Optional[Path]:
    """
    Convert one binary file into one PNG.

    Args:
        input_path: source binary/exe/etc.
        mode: 'resize' or 'truncate'
        images_root: base output directory (e.g., dataset/output)
        label: 'benign' or 'malware'
        resize_target_size: Target size for resize mode as (width, height) tuple
        truncate_target_size: Target size for truncate mode as (width, height) tuple

    Returns:
        Path to newly written PNG, or None if it already existed or args invalid.
    """
    mode = str(mode).lower()
    label = str(label).lower()
    if mode not in MODES or label not in LABELS:
        return None

    sha = sha256_file(input_path)
    out_dir = images_root / label / mode
    ensure_dir(out_dir)
    out_png = out_dir / f"{sha}.png"
    if out_png.exists():
        return None  # idempotent skip

    data = input_path.read_bytes()
    if mode == "truncate":
        img = bytes_to_img_truncate(data, target_size=truncate_target_size)
    else:
        img = bytes_to_img_resize(data, target_size=resize_target_size)
    img.save(out_png, format="PNG", optimize=False)
    return out_png

def rebuild_conversion_log(images_root: Path, csv_path: Path) -> int:
    """
    Rebuild logs/conversion_log.csv by scanning images_root.

    Writes header:
        rel_path,label,mode,sha256

    Returns:
        Number of rows written (excluding header).
    """
    ensure_dir(csv_path.parent)
    ensure_dir(images_root)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
    rows: list[Tuple[str, int, str, str]] = []

    # Walk the canonical layout: images_root/<label>/<mode>/*.png
    for cls, label in (("benign", 0), ("malware", 1)):
        for mode in MODES:
            base = images_root / cls / mode
            if not base.exists():
                continue
            # Collect files with a light progress bar
            files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            for p in tqdm(files, desc=f"index {cls}/{mode}", unit="file", leave=False):
                rel_path = p.relative_to(images_root).as_posix()
                # Trust filename when it's <sha>.png; if not, compute hash
                name = p.stem
                if len(name) == 64 and all(c in "0123456789abcdef" for c in name.lower()):
                    sha = name.lower()
                else:
                    sha = sha256_file(p)
                rows.append((rel_path, label, mode, sha))

    # Write CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "label", "mode", "sha256"])
        for rel_path, label, mode, sha in rows:
            w.writerow([rel_path, label, mode, sha])

    print(f"[convert] Rebuilt {csv_path} with {len(rows)} rows from {images_root}")
    return len(rows)

# --------------- high level runner ----------------

def _collect_inputs(input_roots: Iterable[Path]) -> list[tuple[Path, str]]:
    """
    Return a list of (file_path, label) pairs discovered under each input_root/{benign,malware}/
    """
    out: list[tuple[Path, str]] = []
    for in_root in input_roots:
        base = Path(in_root).resolve()
        for label in LABELS:
            src_dir = base / label
            if not src_dir.exists():
                continue
            for p in src_dir.iterdir():
                if p.is_file():
                    out.append((p, label))
    return out

def _print_input_summary(input_roots: Iterable[Path]) -> None:
    print("[convert] input file counts:")
    for in_root in input_roots:
        base = Path(in_root).resolve()
        for label in LABELS:
            src_dir = base / label
            n = sum(1 for f in src_dir.iterdir() if f.is_file()) if src_dir.exists() else 0
            print(f"  {label:7s} @ {in_root}: {n}")

def run_all(config_path: str = "config.yaml",
            rebuild_only: bool = False,
            skip_convert: bool = False,
            resize_target_size: Optional[tuple[int, int]] = None,
            truncate_target_size: Optional[tuple[int, int]] = None) -> None:
    """
    Main orchestration:
      - load config
      - ensure skeleton
      - (optional) convert inputs -> PNGs
      - rebuild conversion_log.csv
    
    Args:
        config_path: Path to config.yaml
        rebuild_only: Only rebuild CSV, don't convert
        skip_convert: Skip conversion but rebuild CSV
        resize_target_size: Override resize target size from config (tuple or None)
        truncate_target_size: Override truncate target size from config (tuple or None)
    """
    cfg = load_yaml(config_path)
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})
    training = cfg.get("training", {})

    # Parse target sizes from config (with CLI overrides taking precedence)
    resize_size = resize_target_size if resize_target_size is not None else _parse_target_size(
        training.get("resize_target_size"), (64, 64)
    )
    truncate_size = truncate_target_size if truncate_target_size is not None else _parse_target_size(
        training.get("truncate_target_size"), (256, 256)
    )

    # Resolve roots
    input_roots = paths.get("input_roots", ["dataset/input"])
    input_roots = [Path(r).resolve() for r in input_roots]
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    logs_root   = Path("logs").resolve()
    conv_csv    = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()

    # Ensure skeleton
    ensure_dir(logs_root)
    ensure_dir(images_root)
    for cls in LABELS:
        for mode in MODES:
            ensure_dir(images_root / cls / mode)

    if not rebuild_only and not skip_convert:
        _print_input_summary(input_roots)
        items = _collect_inputs(input_roots)
        if not items:
            print("[convert] No input files found under the configured input_roots.", file=sys.stderr)
        # Convert every file into BOTH modes; idempotent if already exists
        total_written = 0
        for (src, label) in tqdm(items, desc="converting", unit="file"):
            for mode in MODES:
                out = convert_file(src, mode, images_root, label,
                                 resize_target_size=resize_size,
                                 truncate_target_size=truncate_size)
                if out is not None:
                    total_written += 1
        print(f"[convert] Wrote {total_written} PNG(s) under {images_root}")
        print(f"[convert] Using target sizes: resize={resize_size}, truncate={truncate_size}")

    # Always rebuild CSV from disk to avoid duplicates/stale rows
    rebuild_conversion_log(images_root, conv_csv)

# ---------------- CLI ----------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Byte->image converter + CSV indexer")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="Only rebuild conversion_log.csv from images_root; do not convert new inputs.")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Skip conversion step (useful if another process created imgs), but rebuild CSV.")
    ap.add_argument("--resize-size", type=str, default=None,
                    help="Override resize target size (e.g., '64,64' or '64x64')")
    ap.add_argument("--truncate-size", type=str, default=None,
                    help="Override truncate target size (e.g., '256,256' or '256x256')")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    # Parse target sizes from CLI if provided
    resize_size = None
    truncate_size = None
    if args.resize_size:
        resize_size = _parse_target_size(args.resize_size, (64, 64))
    if args.truncate_size:
        truncate_size = _parse_target_size(args.truncate_size, (256, 256))
    
    run_all(config_path=args.config,
            rebuild_only=args.rebuild_only,
            skip_convert=args.skip_convert,
            resize_target_size=resize_size,
            truncate_target_size=truncate_size)

if __name__ == "__main__":
    main()

