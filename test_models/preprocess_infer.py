#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight preprocessing for inference.

Converts binary files from input/ into PNGs under output/{resize,truncate}/.
Uses same logic as training preprocessing but simplified for inference use.
"""

from __future__ import annotations

import hashlib
import math
import collections
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except ImportError:
    raise ImportError("Missing Pillow. Install: pip install pillow")

try:
    import yaml
except ImportError:
    raise ImportError("Missing PyYAML. Install: pip install pyyaml")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(p.read_text()) or {}


def sha256_file(p: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_target_size(value, default: tuple[int, int]) -> tuple[int, int]:
    """Parse target size from config value (list, string, or tuple)."""
    if value is None:
        return default
    
    if isinstance(value, list):
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
        elif len(value) == 1:
            val = int(value[0])
            return (val, val)
    
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
    
    if isinstance(value, tuple):
        if len(value) >= 2:
            return (int(value[0]), int(value[1]))
        elif len(value) == 1:
            val = int(value[0])
            return (val, val)
    
    return default


def bytes_to_img_resize(b: bytes, width: int = 256, target_size: tuple[int, int] = (64, 64)) -> Image.Image:
    """
    Resize method: Map all bytes to grayscale pixels, then resize to target_size.
    
    Args:
        b: Raw byte stream
        width: Fixed image width for initial mapping (default 256)
        target_size: Target image size as (width, height) tuple
    
    Returns:
        PIL Image in grayscale mode, size target_size
    """
    target_width, target_height = int(target_size[0]), int(target_size[1])
    
    n = len(b)
    if n == 0:
        return Image.new("L", (target_width, target_height), 0)
    
    # Compute dynamic height
    height = math.ceil(n / width)
    
    # Pad only the final row if incomplete
    bytes_in_full_rows = (height - 1) * width
    bytes_in_last_row = n - bytes_in_full_rows
    
    if bytes_in_last_row < width:
        padding_needed = width - bytes_in_last_row
        padded_bytes = b + bytes(padding_needed)
    else:
        padded_bytes = b[:width * height]
    
    # Create image from bytes
    img = Image.frombytes("L", (width, height), padded_bytes)
    
    # Resize to target_size using bilinear interpolation
    img_resized = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    
    return img_resized


def bytes_to_img_truncate(b: bytes, chunk_size: int = 512, target_size: tuple[int, int] = (256, 256)) -> Image.Image:
    """
    Entropy-aware truncate: Select highest-entropy chunks, then map to target_size.
    
    Args:
        b: Raw byte stream
        chunk_size: Size of each chunk for entropy calculation (default 512)
        target_size: Target image size as (width, height) tuple
    
    Returns:
        PIL Image in grayscale mode, size target_size
    """
    target_width, target_height = int(target_size[0]), int(target_size[1])
    target_bytes = target_width * target_height
    
    n = len(b)
    if n == 0:
        return Image.new("L", (target_width, target_height), 0)
    
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
    
    # Sort by entropy (descending), then by chunk index
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
    
    # Create image from selected bytes
    img = Image.frombytes("L", (target_width, target_height), bytes(selected_bytes[:target_bytes]))
    
    return img


def preprocess_file(input_path: Path, output_root: Path, 
                   resize_target_size: tuple[int, int] = (64, 64),
                   truncate_target_size: tuple[int, int] = (256, 256)) -> dict[str, Optional[Path]]:
    """
    Preprocess one file into both resize and truncate modes.
    
    Args:
        input_path: Source binary file
        output_root: Base output directory (e.g., output)
        resize_target_size: Target size for resize mode
        truncate_target_size: Target size for truncate mode
    
    Returns:
        Dict with keys 'resize' and 'truncate', values are Path to PNG or None if failed
    """
    results = {"resize": None, "truncate": None}
    
    try:
        data = input_path.read_bytes()
        sha = sha256_file(input_path)
        
        # Process resize mode
        resize_dir = output_root / "resize"
        resize_dir.mkdir(parents=True, exist_ok=True)
        resize_png = resize_dir / f"{sha}.png"
        if not resize_png.exists():
            img_resize = bytes_to_img_resize(data, width=256, target_size=resize_target_size)
            img_resize.save(resize_png, format="PNG", optimize=False)
        results["resize"] = resize_png
        
        # Process truncate mode
        truncate_dir = output_root / "truncate"
        truncate_dir.mkdir(parents=True, exist_ok=True)
        truncate_png = truncate_dir / f"{sha}.png"
        if not truncate_png.exists():
            img_truncate = bytes_to_img_truncate(data, chunk_size=512, target_size=truncate_target_size)
            img_truncate.save(truncate_png, format="PNG", optimize=False)
        results["truncate"] = truncate_png
        
    except Exception as e:
        print(f"[preprocess] Error processing {input_path}: {e}", file=__import__("sys").stderr)
    
    return results


def run_preprocessing(config_path: str = "config.yaml") -> int:
    """
    Main preprocessing function: scan input/ and convert all files.
    
    Returns:
        Number of files processed
    """
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    inference = cfg.get("inference", {})
    
    input_root = Path(paths.get("input_root", "input")).resolve()
    output_root = Path(paths.get("output_root", "output")).resolve()
    
    resize_size = _parse_target_size(
        inference.get("resize_target_size"), (64, 64)
    )
    truncate_size = _parse_target_size(
        inference.get("truncate_target_size"), (256, 256)
    )
    
    # Ensure output directories exist
    (output_root / "resize").mkdir(parents=True, exist_ok=True)
    (output_root / "truncate").mkdir(parents=True, exist_ok=True)
    
    # Collect input files (scan recursively, or check for benign/malware subdirs)
    input_files = []
    if input_root.exists():
        # Check for labeled subdirectories first
        for label_dir in ["benign", "malware"]:
            label_path = input_root / label_dir
            if label_path.exists():
                for f in label_path.rglob("*"):
                    if f.is_file():
                        input_files.append(f)
        
        # If no labeled subdirs, scan root directly
        if not input_files:
            for f in input_root.rglob("*"):
                if f.is_file():
                    input_files.append(f)
    
    if not input_files:
        print(f"[preprocess] No input files found in {input_root}")
        return 0
    
    print(f"[preprocess] Found {len(input_files)} file(s) to process")
    print(f"[preprocess] Using target sizes: resize={resize_size}, truncate={truncate_size}")
    
    # Process each file
    processed = 0
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(input_files, desc="preprocessing", unit="file")
    except ImportError:
        iterator = input_files
    
    for input_file in iterator:
        results = preprocess_file(input_file, output_root, resize_size, truncate_size)
        if results["resize"] or results["truncate"]:
            processed += 1
    
    print(f"[preprocess] Processed {processed} file(s) -> {output_root}")
    return processed


if __name__ == "__main__":
    import sys
    run_preprocessing()
    sys.exit(0)

