#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive converter & indexer for the project.

Features:
- Reads config.yaml:
    - paths.input_roots (list[str])
    - train_io.images_root (preferred) or paths.images_root (fallback)
- Ensures folder skeleton:
    dataset/output/{compress,truncate}/{benign,malware}, logs/, etc.
- Converts each input file found under <input_root>/{benign,malware}/
  into BOTH modes {compress, truncate} with deterministic filenames:
    images_root/<mode>/<label>/<sha256>.png
- Idempotent: skips conversion if the output PNG already exists.
- Rebuilds logs/conversion_log.csv by scanning images_root.
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

MODES = ("compress", "truncate")
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

def bytes_to_img_truncate(b: bytes) -> Image.Image:
    """First 65,536 bytes -> 256x256; zero-pad if short."""
    buf = b[:FIXED_PIXELS]
    if len(buf) < FIXED_PIXELS:
        buf = buf + bytes(FIXED_PIXELS - len(buf))
    return Image.frombytes("L", (256, 256), buf)

def bytes_to_img_compress(b: bytes) -> Image.Image:
    """Bin-average bytes into 256x256."""
    import math
    n = len(b)
    if n == 0:
        return Image.frombytes("L", (256, 256), bytes(FIXED_PIXELS))
    out = bytearray(FIXED_PIXELS)
    bin_size = max(1, n // FIXED_PIXELS)
    bins = min(FIXED_PIXELS, math.ceil(n / bin_size))
    pos = 0
    for i in range(bins):
        s = b[pos:pos+bin_size]
        if s:
            out[i] = sum(s) // len(s)
        pos += bin_size
    return Image.frombytes("L", (256, 256), bytes(out))

# ----------------- public API -----------------

def convert_file(input_path: Path, mode: str, images_root: Path, label: str) -> Optional[Path]:
    """
    Convert one binary file into one PNG.

    Args:
        input_path: source binary/exe/etc.
        mode: 'compress' or 'truncate'
        images_root: base output directory (e.g., dataset/output)
        label: 'benign' or 'malware'

    Returns:
        Path to newly written PNG, or None if it already existed or args invalid.
    """
    mode = str(mode).lower()
    label = str(label).lower()
    if mode not in MODES or label not in LABELS:
        return None

    sha = sha256_file(input_path)
    out_dir = images_root / mode / label
    ensure_dir(out_dir)
    out_png = out_dir / f"{sha}.png"
    if out_png.exists():
        return None  # idempotent skip

    data = input_path.read_bytes()
    img = bytes_to_img_truncate(data) if mode == "truncate" else bytes_to_img_compress(data)
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

    # Walk the canonical layout: images_root/<mode>/<label>/*.png
    for mode in MODES:
        for cls, label in (("benign", 0), ("malware", 1)):
            base = images_root / mode / cls
            if not base.exists():
                continue
            # Collect files with a light progress bar
            files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            for p in tqdm(files, desc=f"index {mode}/{cls}", unit="file", leave=False):
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
            skip_convert: bool = False) -> None:
    """
    Main orchestration:
      - load config
      - ensure skeleton
      - (optional) convert inputs -> PNGs
      - rebuild conversion_log.csv
    """
    cfg = load_yaml(config_path)
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})

    # Resolve roots
    input_roots = paths.get("input_roots", ["dataset/input"])
    input_roots = [Path(r).resolve() for r in input_roots]
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    logs_root   = Path("logs").resolve()
    conv_csv    = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()

    # Ensure skeleton
    ensure_dir(logs_root)
    ensure_dir(images_root)
    for mode in MODES:
        for cls in LABELS:
            ensure_dir(images_root / mode / cls)

    if not rebuild_only and not skip_convert:
        _print_input_summary(input_roots)
        items = _collect_inputs(input_roots)
        if not items:
            print("[convert] No input files found under the configured input_roots.", file=sys.stderr)
        # Convert every file into BOTH modes; idempotent if already exists
        total_written = 0
        for (src, label) in tqdm(items, desc="converting", unit="file"):
            for mode in MODES:
                out = convert_file(src, mode, images_root, label)
                if out is not None:
                    total_written += 1
        print(f"[convert] Wrote {total_written} PNG(s) under {images_root}")

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
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    run_all(config_path=args.config,
            rebuild_only=args.rebuild_only,
            skip_convert=args.skip_convert)

if __name__ == "__main__":
    main()

