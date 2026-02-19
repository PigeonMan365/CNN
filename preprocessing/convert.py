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
  pads only the final row, then resizes to target_size (default 64×64).
  
  BYTE BUDGET: A 64×64 image contains 4,096 *pixels* (not literal bytes). Each pixel
  represents one byte value (0-255). Resizing is LOSSY - the full binary is compressed
  into fewer pixels. The target_size controls fidelity vs. efficiency trade-off.
  
  Improvements:
  - Progressive downsampling: Multi-stage resize reduces aliasing and preserves texture
  - Hybrid entropy + uniform sampling: Preserves both high-information regions and global structure
  - Configurable interpolation: Choose 'bilinear', 'bicubic', 'lanczos', or 'area'
  
- truncate (entropy-aware): Divides file into 512-byte chunks, computes Shannon
  entropy per chunk, selects top-N highest-entropy chunks to fill target_size (default 256×256) image.
  Can represent up to target_width × target_height bytes (e.g., 65,536 bytes for 256×256).
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
from typing import Iterable, Tuple, Optional, List

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
FIXED_PIXELS = 256 * 256  # 65,536 pixels (not bytes - each pixel represents one byte value)


def _compute_shannon_entropy(data: bytes) -> float:
    """Return Shannon entropy (bits) for the given byte sequence."""
    if not data:
        return 0.0
    import math
    import collections

    counts = collections.Counter(data)
    entropy = 0.0
    length = len(data)
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_frequency_score(data: bytes) -> float:
    """
    Lightweight byte-frequency dispersion metric.
    High values indicate diverse bytes, low values indicate repetition/padding.
    Returns a value in [0, 1].
    """
    if not data:
        return 0.0
    import collections

    counts = collections.Counter(data)
    length = len(data)
    freq_sum = 0.0
    for count in counts.values():
        p = count / length
        freq_sum += p * (1 - p)
    # Normalize: maximum occurs when all bytes equally likely.
    return min(1.0, freq_sum)


def _detect_file_type(data: bytes) -> str | None:
    """Best-effort detection of common executable formats."""
    if len(data) >= 2 and data[:2] == b"MZ":
        return "pe"
    if len(data) >= 4 and data[:4] == b"\x7fELF":
        return "elf"
    return None


def _detect_structural_chunk_indices(data: bytes, chunk_size: int, num_chunks: int) -> set[int]:
    """
    Identify chunk indices that contain critical structural data (headers, tables).
    These indices are always retained regardless of entropy.
    """
    indices: set[int] = {0}

    def add_offset(offset: int | None) -> None:
        if offset is None:
            return
        if 0 <= offset < len(data):
            idx = offset // chunk_size
            if 0 <= idx < num_chunks:
                indices.add(idx)

    file_type = _detect_file_type(data)
    if file_type == "pe":
        add_offset(0)
        if len(data) >= 0x40:
            pe_offset = int.from_bytes(data[0x3C:0x40], "little", signed=False)
            add_offset(pe_offset)
            if pe_offset + 0x18 <= len(data):
                opt_hdr_size = int.from_bytes(data[pe_offset + 0x14:pe_offset + 0x16], "little", signed=False)
                sec_table = pe_offset + 0x18 + opt_hdr_size
                add_offset(sec_table)
    elif file_type == "elf":
        add_offset(0)
        if len(data) >= 0x34:
            ei_class = data[4]
            if ei_class == 1:  # 32-bit
                e_phoff = int.from_bytes(data[28:32], "little", signed=False)
                e_shoff = int.from_bytes(data[32:36], "little", signed=False)
            elif ei_class == 2:  # 64-bit
                e_phoff = int.from_bytes(data[32:40], "little", signed=False)
                e_shoff = int.from_bytes(data[40:48], "little", signed=False)
            else:
                e_phoff = e_shoff = None
            add_offset(e_phoff)
            add_offset(e_shoff)
    return {idx for idx in indices if 0 <= idx < num_chunks}


def _entropy_weighted_allocation(
    chunk_infos: list[dict],
    target_bytes: int,
) -> bytearray:
    """
    Allocate bytes to chunks proportionally to their weights (entropy + bonuses).
    Ensures deterministic ordering.
    """
    if not chunk_infos:
        return bytearray()

    weights = [max(info["weight"], 1e-6) for info in chunk_infos]
    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = len(chunk_infos)
        weights = [1.0 for _ in chunk_infos]

    # Initial allocation (rounded)
    allocations = [0] * len(chunk_infos)
    remaining = target_bytes
    for idx, weight in enumerate(weights):
        quota = int(round(weight / total_weight * target_bytes))
        allocations[idx] = quota
        remaining -= quota

    # Adjust remainder to ensure sum equals target_bytes
    if remaining != 0:
        # Sort indices by descending weight for positive remainder, ascending otherwise
        order = sorted(range(len(chunk_infos)), key=lambda i: (-weights[i], chunk_infos[i]["index"]))
        if remaining < 0:
            order.reverse()
            remaining = abs(remaining)
        for idx in order:
            if remaining == 0:
                break
            allocations[idx] += 1 if remaining > 0 else -1
            remaining -= 1 if remaining > 0 else -1

    selected = bytearray()
    consumed = [0] * len(chunk_infos)
    for idx, info in enumerate(chunk_infos):
        quota = allocations[idx]
        if quota <= 0:
            continue
        take = min(len(info["data"]), quota, target_bytes - len(selected))
        if take > 0:
            selected.extend(info["data"][:take])
            consumed[idx] = take
        if len(selected) >= target_bytes:
            break

    # If we still fall short (due to rounding), append remaining chunks in weight order.
    if len(selected) < target_bytes:
        ordered_indices = sorted(range(len(chunk_infos)), key=lambda i: (-chunk_infos[i]["weight"], chunk_infos[i]["index"]))
        for idx in ordered_indices:
            info = chunk_infos[idx]
            needed = target_bytes - len(selected)
            if needed <= 0:
                break
            chunk_remaining = info["data"][consumed[idx]:]
            if not chunk_remaining:
                chunk_remaining = info["data"]
            take = min(len(chunk_remaining), needed)
            selected.extend(chunk_remaining[:take])
            consumed[idx] += take
    return selected


def _stratified_chunk_selection(
    chunk_infos: list[dict],
    target_bytes: int,
    structural_indices: set[int],
) -> bytearray:
    """
    Multi-entropy sampling: mix high, mid, and low entropy chunks plus structural regions.
    """
    selected = bytearray()
    bytes_needed = target_bytes
    added_indices: set[int] = set()

    def add_chunk(info: dict) -> None:
        nonlocal bytes_needed
        if bytes_needed <= 0 or info["index"] in added_indices:
            return
        take = min(len(info["data"]), bytes_needed)
        if take > 0:
            selected.extend(info["data"][:take])
            bytes_needed -= take
            added_indices.add(info["index"])

    # Always keep structural chunks first (headers, section tables, etc.)
    structural_infos = [
        info for info in chunk_infos if info["index"] in structural_indices
    ]
    structural_infos.sort(key=lambda x: x["index"])
    for info in structural_infos:
        add_chunk(info)
    if bytes_needed <= 0:
        return selected

    # Remaining chunks sorted by entropy
    remaining = [info for info in chunk_infos if info["index"] not in added_indices]
    if not remaining:
        return selected

    remaining.sort(key=lambda x: (-x["entropy"], x["index"]))
    n = len(remaining)
    top_count = max(1, n // 3)
    mid_count = max(1, (n - top_count) // 2)
    low_count = max(1, n - top_count - mid_count)

    top_group = remaining[:top_count]
    mid_group = remaining[top_count:top_count + mid_count]
    low_group = remaining[top_count + mid_count:]
    if not mid_group:
        mid_group = top_group
    if not low_group:
        low_group = mid_group

    groups = [top_group, mid_group, low_group]
    cycle = 0
    while bytes_needed > 0 and any(groups):
        group = groups[cycle % len(groups)]
        for info in group:
            add_chunk(info)
            if bytes_needed <= 0:
                break
        cycle += 1
        if cycle > len(chunk_infos) * 2:
            break

    if bytes_needed > 0:
        for info in remaining:
            add_chunk(info)
            if bytes_needed <= 0:
                break

    return selected

# ----------------- small utils -----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file_safe(p: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """Return SHA256 hex digest of file, or None if the file cannot be read (logs to stderr)."""
    try:
        return sha256_file(p, chunk_size=chunk_size)
    except OSError as e:
        print(f"[convert] SKIP (read error): {p} — {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[convert] SKIP (unexpected): {p} — {e}", file=sys.stderr)
        return None

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

def _parse_interpolation_method(method: str) -> Image.Resampling:
    """
    Parse interpolation method string to PIL Resampling enum.
    
    Args:
        method: One of 'bilinear', 'bicubic', 'lanczos', 'area'
    
    Returns:
        PIL Image.Resampling enum value
    """
    method_lower = method.lower().strip()
    mapping = {
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
        'area': Image.Resampling.BOX,  # AREA is called BOX in PIL
    }
    return mapping.get(method_lower, Image.Resampling.LANCZOS)

def bytes_to_img_resize(b: bytes, width: int = 256, target_size: tuple[int, int] | None = None,
                        resample: Image.Resampling | str | None = None,
                        entropy_hybrid: bool = False,
                        entropy_ratio: float = 0.6) -> Image.Image:
    """
    Resize method: Map all bytes to grayscale pixels, then resize to target_size.
    
    BYTE BUDGET CLARIFICATION:
    - A 64×64 image contains 4,096 *pixels* (not literal bytes)
    - Each pixel represents one byte value (0-255) from the original binary
    - Resizing is a LOSSY representation: the full binary is compressed into fewer pixels
    - The resize_target_size variable in config controls the fidelity vs. efficiency trade-off:
      * Larger sizes (256×256 = 65,536 pixels) preserve more information
      * Smaller sizes (64×64 = 4,096 pixels) are more efficient but lose detail
    - Information loss is inherent: you cannot represent N bytes in <N pixels without loss
    
    Steps:
    1. Map all bytes to grayscale pixels (0-255) in byte order
    2. Use fixed width (256 pixels by default)
    3. Compute dynamic height: ceil(len(bytes) / width)
    4. Pad only the final row to complete the last line
    5. Optionally apply hybrid entropy + uniform sampling (if entropy_hybrid=True)
    6. Resize to target_size using fractional progressive downsampling (always applied)
    
    Args:
        b: Raw byte stream
        width: Fixed image width for initial mapping (default 256)
        target_size: Target image size as (width, height) tuple. If None, reads from config.yaml
                     or defaults to (64, 64). This controls the lossy compression ratio.
        resample: Resampling method (PIL enum, string like 'lanczos', or None to read from config)
        entropy_hybrid: If True, use hybrid entropy + uniform sampling to preserve both
                       high-information regions and global structure
        entropy_ratio: Fraction of rows selected by entropy (0.0-1.0). Remaining rows are uniform.
                      Default 0.6 means 60% entropy-based, 40% uniform sampling.
    
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
    # BYTE BUDGET NOTE: We map bytes to pixels (each byte value 0-255 becomes one pixel)
    # The intermediate image has width×height pixels, where height = ceil(file_size / width)
    # This intermediate image will be downsampled to target_size, causing information loss
    # The resize_target_size variable in config controls this lossy compression ratio
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
    
    # Parse resampling method if string provided
    if isinstance(resample, str):
        resample = _parse_interpolation_method(resample)
    elif resample is None:
        # Read from config or default to LANCZOS
        try:
            cfg = load_yaml()
            training = cfg.get("training", {})
            method = training.get("resize_interpolation", "lanczos")
            resample = _parse_interpolation_method(method)
        except Exception:
            resample = Image.Resampling.LANCZOS
    
    # Hybrid entropy + uniform sampling: preserve both high-information regions and global structure
    if entropy_hybrid and height > target_height:
        import collections
        import numpy as np
        
        # Convert to numpy for processing
        img_array = np.array(img, dtype=np.uint8)
        
        # Compute Shannon entropy for each row
        row_entropies = []
        for row_idx in range(height):
            row = img_array[row_idx, :]
            byte_counts = collections.Counter(row.tolist())
            entropy = 0.0
            row_len = len(row)
            for count in byte_counts.values():
                p = count / row_len
                if p > 0:
                    entropy -= p * math.log2(p)
            row_entropies.append((entropy, row_idx))
        
        # Sort rows by entropy (descending)
        row_entropies.sort(key=lambda x: -x[0])
        
        # Hybrid selection: combine entropy-based and uniform sampling
        # Use configurable ratio (default 60% entropy, 40% uniform) to preserve both detail and structure
        # Clamp entropy_ratio to valid range [0.0, 1.0]
        entropy_ratio = max(0.0, min(1.0, entropy_ratio))
        uniform_ratio = 1.0 - entropy_ratio
        
        num_rows_needed = min(target_height, height)
        num_entropy_rows = int(num_rows_needed * entropy_ratio)
        num_uniform_rows = num_rows_needed - num_entropy_rows
        
        # Select high-entropy rows
        entropy_indices = [idx for _, idx in row_entropies[:num_entropy_rows]]
        
        # Select uniform baseline sample (evenly spaced across file)
        if num_uniform_rows > 0:
            uniform_indices = [int(i * height / num_uniform_rows) for i in range(num_uniform_rows)]
            # Remove duplicates and merge
            all_indices = sorted(set(entropy_indices + uniform_indices))
        else:
            all_indices = sorted(entropy_indices)
        
        # Ensure we have exactly target_height rows (or as close as possible)
        if len(all_indices) < num_rows_needed:
            # Fill remaining with highest entropy rows not yet selected
            remaining = [idx for _, idx in row_entropies if idx not in all_indices]
            all_indices.extend(remaining[:num_rows_needed - len(all_indices)])
            all_indices = sorted(all_indices[:num_rows_needed])
        elif len(all_indices) > num_rows_needed:
            all_indices = sorted(all_indices[:num_rows_needed])
        
        # Create new image with hybrid-selected rows
        selected_rows = img_array[all_indices, :]
        
        # If we have fewer rows than target, pad with zeros
        if len(selected_rows) < target_height:
            padding = np.zeros((target_height - len(selected_rows), width), dtype=np.uint8)
            selected_rows = np.vstack([selected_rows, padding])
        
        img = Image.fromarray(selected_rows, mode='L')
        # Update dimensions after row selection
        width, height = img.size
    
    # Fractional progressive downsampling: always apply to reduce aliasing and preserve texture
    # This is a lossy operation: we're compressing the full binary representation into fewer pixels
    # A 64×64 target = 4,096 pixels (not literal bytes), each pixel represents one byte value
    # Dynamically reduce dimensions by 1/2 or 1/4 until target size is reached
    if width > target_width or height > target_height:
        current_img = img
        current_w, current_h = width, height
        
        # Define intermediate stages using fractional downsampling (1/2 or 1/4 reduction)
        stages = []
        w, h = current_w, current_h
        
        while w > target_width or h > target_height:
            # Choose reduction factor: use 1/4 if we're more than 4x larger, else 1/2
            if w > target_width * 4 or h > target_height * 4:
                # Aggressive reduction: 1/4
                w = max(int(w * 0.25), target_width)
                h = max(int(h * 0.25), target_height)
            else:
                # Moderate reduction: 1/2
                w = max(int(w * 0.5), target_width)
                h = max(int(h * 0.5), target_height)
            
            # Only add stage if it's different from current and not already at target
            if (w != current_w or h != current_h) and (w > target_width or h > target_height):
                stages.append((w, h))
                current_w, current_h = w, h
        
        # Add final target size if not already included
        if not stages or stages[-1] != (target_width, target_height):
            stages.append((target_width, target_height))
        
        # Resize through each stage using the chosen interpolation method
        for stage_w, stage_h in stages:
            current_img = current_img.resize((stage_w, stage_h), resample)
        
        img_resized = current_img
    else:
        # Already at or below target size, no downsampling needed
        img_resized = img
    
    return img_resized

def bytes_to_img_entropy_truncate(
    b: bytes,
    chunk_size: int | None = None,
    target_size: tuple[int, int] | None = None,
    entropy_stratify: bool = True,
    entropy_weighted: bool = False,
    use_frequency: bool = False,
) -> Image.Image:
    """
    Entropy-aware truncate: select chunks using entropy + structural signals, then map to target_size.

    Improvements:
    - Adaptive chunk size based on config + file heuristics
    - Hybrid structural awareness (headers/section tables always included)
    - Multi-entropy sampling (top/mid/low) with optional entropy-weighted mapping
    - Optional byte-frequency features to augment entropy scoring
    """
    import math

    # Parse target size
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
            target_size = (256, 256)

    if isinstance(target_size, (int, float)):
        target_size = (int(target_size), int(target_size))
    target_width, target_height = int(target_size[0]), int(target_size[1])
    target_bytes = target_width * target_height

    n = len(b)
    if n == 0:
        return Image.new("L", (target_width, target_height), 0)

    # Determine chunk size (adaptive)
    base_chunk = chunk_size
    if base_chunk is None:
        try:
            cfg = load_yaml()
            training = cfg.get("training", {})
            base_chunk = int(training.get("truncate_chunk_size", 512))
        except Exception:
            base_chunk = 512
    base_chunk = max(64, int(base_chunk))

    file_type = _detect_file_type(b)
    adaptive_chunk = base_chunk
    if file_type == "pe" or file_type == "elf":
        adaptive_chunk = max(256, base_chunk)
    if n > 4 * target_bytes:
        adaptive_chunk = min(2048, adaptive_chunk * 2)
    elif n < target_bytes // 2:
        adaptive_chunk = max(256, adaptive_chunk // 2)

    chunk_size = adaptive_chunk
    num_chunks = max(1, math.ceil(n / chunk_size))
    structural_indices = _detect_structural_chunk_indices(b, chunk_size, num_chunks)

    chunk_infos: list[dict] = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        chunk = b[start_idx:end_idx]
        entropy = _compute_shannon_entropy(chunk)
        freq_bonus = _compute_frequency_score(chunk) if use_frequency else 0.0
        structural_bonus = 0.3 if i in structural_indices else 0.0
        weight = entropy + structural_bonus + (0.3 * freq_bonus)
        chunk_infos.append(
            {
                "index": i,
                "data": chunk,
                "entropy": entropy,
                "weight": weight,
            }
        )

    # Determine mean entropy to optionally fine-tune chunk size (high entropy -> smaller chunks)
    if chunk_infos:
        mean_entropy = sum(info["entropy"] for info in chunk_infos) / len(chunk_infos)
        if mean_entropy > 7.5 and chunk_size > 128:
            chunk_size = max(128, chunk_size // 2)
        elif mean_entropy < 5.0 and chunk_size < 2048:
            chunk_size = min(2048, chunk_size * 2)

    # Selection strategies
    if entropy_weighted:
        selected_bytes = _entropy_weighted_allocation(
            sorted(chunk_infos, key=lambda x: (-x["weight"], x["index"])),
            target_bytes,
        )
    else:
        if entropy_stratify:
            selected_bytes = _stratified_chunk_selection(
                sorted(chunk_infos, key=lambda x: (-x["entropy"], x["index"])),
                target_bytes,
                structural_indices,
            )
        else:
            # Simple highest-weight selection (with structural preference)
            selected_bytes = bytearray()
            structural_infos = [
                info for info in chunk_infos if info["index"] in structural_indices
            ]
            structural_infos.sort(key=lambda x: x["index"])
            for info in structural_infos:
                if len(selected_bytes) >= target_bytes:
                    break
                take = min(len(info["data"]), target_bytes - len(selected_bytes))
                selected_bytes.extend(info["data"][:take])

            if len(selected_bytes) < target_bytes:
                ordered = sorted(
                    [info for info in chunk_infos if info["index"] not in structural_indices],
                    key=lambda x: (-x["weight"], x["index"]),
                )
                for info in ordered:
                    if len(selected_bytes) >= target_bytes:
                        break
                    take = min(len(info["data"]), target_bytes - len(selected_bytes))
                    selected_bytes.extend(info["data"][:take])

    if len(selected_bytes) < target_bytes:
        selected_bytes.extend(bytes(target_bytes - len(selected_bytes)))
    elif len(selected_bytes) > target_bytes:
        selected_bytes = selected_bytes[:target_bytes]

    img = Image.frombytes("L", (target_width, target_height), bytes(selected_bytes))
    return img


def bytes_to_img_truncate(
    b: bytes,
    target_size: tuple[int, int] = (256, 256),
    chunk_size: int | None = None,
    entropy_stratify: bool = True,
    entropy_weighted: bool = False,
    use_frequency: bool = False,
) -> Image.Image:
    """
    Truncate mode with adaptive chunking + entropy/structure awareness.
    """
    return bytes_to_img_entropy_truncate(
        b,
        chunk_size=chunk_size,
        target_size=target_size,
        entropy_stratify=entropy_stratify,
        entropy_weighted=entropy_weighted,
        use_frequency=use_frequency,
    )

# ----------------- public API -----------------

def convert_file(input_path: Path, mode: str, images_root: Path, label: str,
                 resize_target_size: tuple[int, int] = (64, 64),
                 truncate_target_size: tuple[int, int] = (256, 256),
                 resize_resample: Image.Resampling | str | None = None,
                 resize_entropy_hybrid: bool | None = None,
                 resize_entropy_ratio: float | None = None,
                 truncate_chunk_size: int | None = None,
                 truncate_entropy_stratify: bool | None = None,
                 truncate_entropy_weighted: bool | None = None,
                 truncate_use_frequency: bool | None = None) -> Optional[Path]:
    """
    Convert one binary file into one PNG.

    Args:
        input_path: source binary/exe/etc.
        mode: 'resize' or 'truncate'
        images_root: base output directory (e.g., dataset/output)
        label: 'benign' or 'malware'
        resize_target_size: Target size for resize mode as (width, height) tuple
        truncate_target_size: Target size for truncate mode as (width, height) tuple
        resize_resample: Resampling method for resize (PIL enum, string, or None = read from config)
        resize_entropy_hybrid: Use hybrid entropy + uniform sampling (None = read from config or False)
        resize_entropy_ratio: Fraction of rows selected by entropy (None = read from config or 0.6)
        truncate_chunk_size: Chunk size for entropy computation (None = read from config or 512)
        truncate_entropy_stratify: Enable multi-entropy sampling (None = read from config or True)
        truncate_entropy_weighted: Enable entropy-weighted mapping (None = read from config or False)
        truncate_use_frequency: Use byte-frequency metrics in scoring (None = read from config or False)

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
        if (truncate_chunk_size is None or truncate_entropy_stratify is None or
                truncate_entropy_weighted is None or truncate_use_frequency is None):
            try:
                cfg = load_yaml()
                training = cfg.get("training", {})
                if truncate_chunk_size is None:
                    truncate_chunk_size = int(training.get("truncate_chunk_size", 512))
                if truncate_entropy_stratify is None:
                    truncate_entropy_stratify = bool(training.get("truncate_entropy_stratify", True))
                if truncate_entropy_weighted is None:
                    truncate_entropy_weighted = bool(training.get("truncate_entropy_weighted", False))
                if truncate_use_frequency is None:
                    truncate_use_frequency = bool(training.get("truncate_use_frequency", False))
            except Exception:
                if truncate_chunk_size is None:
                    truncate_chunk_size = 512
                if truncate_entropy_stratify is None:
                    truncate_entropy_stratify = True
                if truncate_entropy_weighted is None:
                    truncate_entropy_weighted = False
                if truncate_use_frequency is None:
                    truncate_use_frequency = False

        img = bytes_to_img_truncate(
            data,
            target_size=truncate_target_size,
            chunk_size=truncate_chunk_size,
            entropy_stratify=truncate_entropy_stratify,
            entropy_weighted=truncate_entropy_weighted,
            use_frequency=truncate_use_frequency,
        )
    else:
        # Read resize options from config if not provided (backward compatibility)
        if resize_resample is None or resize_entropy_hybrid is None or resize_entropy_ratio is None:
            try:
                cfg = load_yaml()
                training = cfg.get("training", {})
                if resize_resample is None:
                    method = training.get("resize_interpolation", "lanczos")
                    resize_resample = method  # Will be parsed by bytes_to_img_resize
                if resize_entropy_hybrid is None:
                    resize_entropy_hybrid = training.get("resize_entropy_hybrid", False)
                if resize_entropy_ratio is None:
                    resize_entropy_ratio = training.get("resize_entropy_ratio", 0.6)
            except Exception:
                if resize_resample is None:
                    resize_resample = "lanczos"
                if resize_entropy_hybrid is None:
                    resize_entropy_hybrid = False
                if resize_entropy_ratio is None:
                    resize_entropy_ratio = 0.6
        
        # Progressive downsampling is always applied (no config option needed)
        img = bytes_to_img_resize(data, target_size=resize_target_size,
                                 resample=resize_resample,
                                 entropy_hybrid=resize_entropy_hybrid,
                                 entropy_ratio=resize_entropy_ratio)
    # Enable PNG optimization for better compression (reduces file size significantly)
    # compress_level=6 is a good balance between compression and speed
    img.save(out_png, format="PNG", optimize=True, compress_level=6)
    return out_png

def load_done_sha256_from_log(csv_path: Path) -> set[str]:
    """
    Read conversion_log.csv and return the set of input file SHA256 hashes
    that are already recorded (i.e. already converted). Used to resume conversion.
    """
    out: set[str] = set()
    if not csv_path.exists():
        return out
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                sha = (row.get("sha256") or "").strip()
                if sha:
                    out.add(sha.lower())
    except Exception:
        pass
    return out


def append_rows_to_conversion_log(csv_path: Path, rows: list[Tuple[str, int, str, str]]) -> None:
    """
    Append rows to conversion_log.csv. If the file does not exist, write the header first.
    Each row is (rel_path, label, mode, sha256) with label 0=benign, 1=malware.
    """
    ensure_dir(csv_path.parent)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["rel_path", "label", "mode", "sha256"])
        for rel_path, label, mode, sha in rows:
            w.writerow([rel_path, label, mode, sha])


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


def _iter_inputs(input_roots: Iterable[Path]) -> Iterable[tuple[Path, str]]:
    """
    Yield (file_path, label) pairs under each input_root/{benign,malware}/ without building a full list.
    """
    for in_root in input_roots:
        base = Path(in_root).resolve()
        for label in LABELS:
            src_dir = base / label
            if not src_dir.exists():
                continue
            for p in src_dir.iterdir():
                if p.is_file():
                    yield (p, label)


def _already_converted(sha: str, label: str, images_root: Path) -> bool:
    """Return True if both resize and truncate PNGs exist for this input (no log read)."""
    base = images_root / label
    return (base / "resize" / f"{sha}.png").exists() and (base / "truncate" / f"{sha}.png").exists()

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

    resize_interpolation = training.get("resize_interpolation", "lanczos")
    resize_entropy_hybrid = bool(training.get("resize_entropy_hybrid", False))
    resize_entropy_ratio = float(training.get("resize_entropy_ratio", 0.6))

    truncate_chunk_size = int(training.get("truncate_chunk_size", 512))
    truncate_entropy_stratify = bool(training.get("truncate_entropy_stratify", True))
    truncate_entropy_weighted = bool(training.get("truncate_entropy_weighted", False))
    truncate_use_frequency = bool(training.get("truncate_use_frequency", False))

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
        # Progressive: count total first (lightweight pass, no hashing), then scan 1% → process → log, repeat
        total_inputs = sum(1 for _ in _iter_inputs(input_roots))
        if total_inputs == 0:
            print("[convert] No input files found under the configured input_roots.", file=sys.stderr)
        else:
            chunk_size = max(1, total_inputs // 100)
            total_written = 0
            total_skipped = 0
            total_failed: List[Tuple[Path, str, str]] = []  # (path, label, error)
            processed = 0
            batch: list[tuple[Path, str]] = []  # (path, label) for this 1% slice only

            def process_batch(
                batch_paths: list[tuple[Path, str]],
            ) -> tuple[list[Tuple[str, int, str, str]], int, int, List[Tuple[Path, str, str]]]:
                """Process one batch: hash, skip if done, convert. Returns (new_rows, written, skipped, failed)."""
                new_rows: list[Tuple[str, int, str, str]] = []
                written = 0
                skipped = 0
                failed: List[Tuple[Path, str, str]] = []  # (path, label, error_message)
                for (path, label) in batch_paths:
                    try:
                        sha = sha256_file(path)
                    except OSError as e:
                        failed.append((path, label, str(e)))
                        print(f"[convert] SKIP (read): {path} — {e}", file=sys.stderr)
                        continue
                    except Exception as e:
                        failed.append((path, label, str(e)))
                        print(f"[convert] SKIP (hash): {path} — {e}", file=sys.stderr)
                        continue
                    if _already_converted(sha, label, images_root):
                        skipped += 1
                        continue
                    label_int = 0 if label == "benign" else 1
                    try:
                        for mode in MODES:
                            out = convert_file(path, mode, images_root, label,
                                             resize_target_size=resize_size,
                                             truncate_target_size=truncate_size,
                                             resize_resample=resize_interpolation,
                                             resize_entropy_hybrid=resize_entropy_hybrid,
                                             resize_entropy_ratio=resize_entropy_ratio,
                                             truncate_chunk_size=truncate_chunk_size,
                                             truncate_entropy_stratify=truncate_entropy_stratify,
                                             truncate_entropy_weighted=truncate_entropy_weighted,
                                             truncate_use_frequency=truncate_use_frequency)
                            if out is not None:
                                written += 1
                                rel = out.relative_to(images_root).as_posix()
                                new_rows.append((rel, label_int, mode, sha))
                    except OSError as e:
                        failed.append((path, label, str(e)))
                        print(f"[convert] SKIP (convert): {path} — {e}", file=sys.stderr)
                    except Exception as e:
                        failed.append((path, label, str(e)))
                        print(f"[convert] SKIP (convert): {path} — {e}", file=sys.stderr)
                return new_rows, written, skipped, failed

            for (path, label) in _iter_inputs(input_roots):
                batch.append((path, label))
                if len(batch) < chunk_size:
                    continue
                # This 1% is scanned; now process it and update log
                new_rows, written, skipped, failed = process_batch(batch)
                if new_rows:
                    append_rows_to_conversion_log(conv_csv, new_rows)
                total_written += written
                total_skipped += skipped
                total_failed.extend(failed)
                processed += len(batch)
                pct = min(100, processed * 100 // total_inputs)
                print(f"[convert] Progress: {pct}% ({processed}/{total_inputs} inputs) — {total_written} PNGs written, log updated.")
                batch.clear()

            if batch:
                new_rows, written, skipped, failed = process_batch(batch)
                if new_rows:
                    append_rows_to_conversion_log(conv_csv, new_rows)
                total_written += written
                total_skipped += skipped
                total_failed.extend(failed)
                processed += len(batch)
                print(f"[convert] Progress: 100% ({processed}/{total_inputs} inputs) — {total_written} PNGs written, log updated.")

            if total_skipped:
                print(f"[convert] Skipped {total_skipped} already-converted input(s).")
            if total_failed:
                failed_log = conv_csv.parent / "conversion_failed.csv"
                try:
                    write_header = not failed_log.exists() or failed_log.stat().st_size == 0
                    with failed_log.open("a", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow(["path", "label", "error"])
                        for (p, lbl, err) in total_failed:
                            w.writerow([str(p), lbl, err])
                    print(f"[convert] {len(total_failed)} file(s) skipped due to errors; see {failed_log}", file=sys.stderr)
                except Exception as e:
                    print(f"[convert] {len(total_failed)} file(s) skipped due to errors (could not write {failed_log}: {e})", file=sys.stderr)
            print(f"[convert] Wrote {total_written} PNG(s) under {images_root}")
            print(f"[convert] Using target sizes: resize={resize_size}, truncate={truncate_size}")

    # Rebuild CSV only when not converting (rebuild_only/skip_convert), so incremental updates are preserved
    if rebuild_only or skip_convert:
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

