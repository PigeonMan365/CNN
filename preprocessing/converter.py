#!/usr/bin/env python3
"""
Primitive but deterministic byte->image converter for two modes:
- truncate : first 65,536 bytes -> 256x256 (zero-padded if short)
- compress : bin-average bytes into 256x256

Writes: images_root/<mode>/<label>/<sha256>.png (grayscale)
Exposes convert_file() for programmatic use by dual_convert.py.
"""

from pathlib import Path
import hashlib
import sys

try:
    from PIL import Image
except Exception:
    print("[converter] Missing Pillow. Install: pip install pillow", file=sys.stderr)
    raise

FIXED_PIXELS = 256 * 256  # 65,536
MODES = ("compress", "truncate")
LABELS = ("benign", "malware")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def bytes_to_img_truncate(b: bytes) -> Image.Image:
    buf = b[:FIXED_PIXELS]
    if len(buf) < FIXED_PIXELS:
        buf = buf + bytes(FIXED_PIXELS - len(buf))
    return Image.frombytes("L", (256, 256), buf)

def bytes_to_img_compress(b: bytes) -> Image.Image:
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

def convert_file(input_path: Path, mode: str, images_root: Path, label: str) -> Path | None:
    mode = mode.lower()
    label = label.lower()
    if mode not in MODES or label not in LABELS:
        return None

    sha = sha256_file(input_path)
    out_dir = images_root / mode / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{sha}.png"
    if out_png.exists():
        return None  # idempotent

    data = input_path.read_bytes()
    img = bytes_to_img_truncate(data) if mode == "truncate" else bytes_to_img_compress(data)
    img.save(out_png, format="PNG", optimize=False)
    return out_png
