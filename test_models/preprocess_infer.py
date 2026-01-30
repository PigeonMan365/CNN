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


def _parse_interpolation_method(method: str) -> Image.Resampling:
    """Parse interpolation method string to PIL Resampling enum."""
    method_lower = method.lower().strip()
    mapping = {
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
        'area': Image.Resampling.BOX,  # AREA is called BOX in PIL
    }
    return mapping.get(method_lower, Image.Resampling.LANCZOS)

def bytes_to_img_resize(b: bytes, width: int = 256, target_size: tuple[int, int] = (64, 64),
                        resample: Image.Resampling | str | None = None,
                        entropy_hybrid: bool = False,
                        entropy_ratio: float = 0.6) -> Image.Image:
    """
    Resize method: Map all bytes to grayscale pixels, then resize to target_size.
    
    BYTE BUDGET CLARIFICATION:
    - A 64×64 image contains 4,096 *pixels* (not literal bytes)
    - Each pixel represents one byte value (0-255) from the original binary
    - Resizing is a LOSSY representation: the full binary is compressed into fewer pixels
    - The resize_target_size variable in config controls the fidelity vs. efficiency trade-off
    
    Args:
        b: Raw byte stream
        width: Fixed image width for initial mapping (default 256)
        target_size: Target image size as (width, height) tuple. This controls the lossy compression ratio.
        resample: Resampling method (PIL enum, string like 'lanczos', or None to read from config)
        entropy_hybrid: If True, use hybrid entropy + uniform sampling
        entropy_ratio: Fraction of rows selected by entropy (0.0-1.0). Default 0.6 means 60% entropy, 40% uniform.
    
    Returns:
        PIL Image in grayscale mode, size target_size
    """
    import collections
    
    target_width, target_height = int(target_size[0]), int(target_size[1])
    
    n = len(b)
    if n == 0:
        return Image.new("L", (target_width, target_height), 0)
    
    # Parse resampling method
    if isinstance(resample, str):
        resample = _parse_interpolation_method(resample)
    elif resample is None:
        try:
            cfg = load_config()
            inference = cfg.get("inference", {})
            method = inference.get("resize_interpolation", "lanczos")
            resample = _parse_interpolation_method(method)
        except Exception:
            resample = Image.Resampling.LANCZOS
    
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
    
    # Hybrid entropy + uniform sampling (if enabled)
    if entropy_hybrid and height > target_height:
        import numpy as np
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
        
        row_entropies.sort(key=lambda x: -x[0])
        
        # Hybrid selection: use configurable ratio (default 60% entropy, 40% uniform)
        entropy_ratio = max(0.0, min(1.0, entropy_ratio))
        uniform_ratio = 1.0 - entropy_ratio
        
        num_rows_needed = min(target_height, height)
        num_entropy_rows = int(num_rows_needed * entropy_ratio)
        num_uniform_rows = num_rows_needed - num_entropy_rows
        
        entropy_indices = [idx for _, idx in row_entropies[:num_entropy_rows]]
        if num_uniform_rows > 0:
            uniform_indices = [int(i * height / num_uniform_rows) for i in range(num_uniform_rows)]
            all_indices = sorted(set(entropy_indices + uniform_indices))
        else:
            all_indices = sorted(entropy_indices)
        
        if len(all_indices) < num_rows_needed:
            remaining = [idx for _, idx in row_entropies if idx not in all_indices]
            all_indices.extend(remaining[:num_rows_needed - len(all_indices)])
            all_indices = sorted(all_indices[:num_rows_needed])
        elif len(all_indices) > num_rows_needed:
            all_indices = sorted(all_indices[:num_rows_needed])
        
        selected_rows = img_array[all_indices, :]
        if len(selected_rows) < target_height:
            padding = np.zeros((target_height - len(selected_rows), width), dtype=np.uint8)
            selected_rows = np.vstack([selected_rows, padding])
        
        img = Image.fromarray(selected_rows, mode='L')
        width, height = img.size
    
    # Fractional progressive downsampling: always apply to reduce aliasing and preserve texture
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


def _compute_shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = collections.Counter(data)
    entropy = 0.0
    length = len(data)
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_frequency_score(data: bytes) -> float:
    if not data:
        return 0.0
    counts = collections.Counter(data)
    length = len(data)
    score = 0.0
    for count in counts.values():
        p = count / length
        score += p * (1 - p)
    return min(1.0, score)


def _detect_file_type(data: bytes) -> str | None:
    if len(data) >= 2 and data[:2] == b"MZ":
        return "pe"
    if len(data) >= 4 and data[:4] == b"\x7fELF":
        return "elf"
    return None


def _detect_structural_chunk_indices(data: bytes, chunk_size: int, num_chunks: int) -> set[int]:
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
            if ei_class == 1:
                e_phoff = int.from_bytes(data[28:32], "little", signed=False)
                e_shoff = int.from_bytes(data[32:36], "little", signed=False)
            elif ei_class == 2:
                e_phoff = int.from_bytes(data[32:40], "little", signed=False)
                e_shoff = int.from_bytes(data[40:48], "little", signed=False)
            else:
                e_phoff = e_shoff = None
            add_offset(e_phoff)
            add_offset(e_shoff)
    return {idx for idx in indices if 0 <= idx < num_chunks}


def _entropy_weighted_allocation(chunk_infos: list[dict], target_bytes: int) -> bytearray:
    if not chunk_infos:
        return bytearray()
    weights = [max(info["weight"], 1e-6) for info in chunk_infos]
    total_weight = sum(weights)
    allocations = [0] * len(chunk_infos)
    remaining = target_bytes
    for idx, weight in enumerate(weights):
        quota = int(round(weight / total_weight * target_bytes))
        allocations[idx] = quota
        remaining -= quota
    if remaining != 0:
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
    if len(selected) < target_bytes:
        ordered = sorted(range(len(chunk_infos)), key=lambda i: (-chunk_infos[i]["weight"], chunk_infos[i]["index"]))
        for idx in ordered:
            needed = target_bytes - len(selected)
            if needed <= 0:
                break
            chunk_remaining = chunk_infos[idx]["data"][consumed[idx]:]
            if not chunk_remaining:
                chunk_remaining = chunk_infos[idx]["data"]
            take = min(len(chunk_remaining), needed)
            selected.extend(chunk_remaining[:take])
            consumed[idx] += take
    return selected


def _stratified_chunk_selection(
    chunk_infos: list[dict],
    target_bytes: int,
    structural_indices: set[int],
) -> bytearray:
    selected = bytearray()
    bytes_needed = target_bytes
    added: set[int] = set()

    def add(info: dict) -> None:
        nonlocal bytes_needed
        if bytes_needed <= 0 or info["index"] in added:
            return
        take = min(len(info["data"]), bytes_needed)
        if take > 0:
            selected.extend(info["data"][:take])
            bytes_needed -= take
            added.add(info["index"])

    structural_infos = [info for info in chunk_infos if info["index"] in structural_indices]
    structural_infos.sort(key=lambda x: x["index"])
    for info in structural_infos:
        add(info)
    if bytes_needed <= 0:
        return selected

    remaining = [info for info in chunk_infos if info["index"] not in added]
    if not remaining:
        return selected

    remaining.sort(key=lambda x: (-x["entropy"], x["index"]))
    n = len(remaining)
    top_count = max(1, n // 3)
    mid_count = max(1, (n - top_count) // 2)
    low_count = max(1, n - top_count - mid_count)

    top = remaining[:top_count]
    mid = remaining[top_count:top_count + mid_count]
    low = remaining[top_count + mid_count:]
    if not mid:
        mid = top
    if not low:
        low = mid

    groups = [top, mid, low]
    cycle = 0
    while bytes_needed > 0 and any(groups):
        group = groups[cycle % len(groups)]
        for info in group:
            add(info)
            if bytes_needed <= 0:
                break
        cycle += 1
        if cycle > len(chunk_infos) * 2:
            break

    if bytes_needed > 0:
        for info in remaining:
            add(info)
            if bytes_needed <= 0:
                break
    return selected


def bytes_to_img_truncate(
    b: bytes,
    target_size: tuple[int, int] = (256, 256),
    chunk_size: int | None = None,
    entropy_stratify: bool = True,
    entropy_weighted: bool = False,
    use_frequency: bool = False,
) -> Image.Image:
    target_width, target_height = int(target_size[0]), int(target_size[1])
    target_bytes = target_width * target_height

    n = len(b)
    if n == 0:
        return Image.new("L", (target_width, target_height), 0)

    base_chunk = chunk_size if chunk_size is not None else 512
    base_chunk = max(64, int(base_chunk))

    file_type = _detect_file_type(b)
    adaptive_chunk = base_chunk
    if file_type in {"pe", "elf"}:
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
        start = i * chunk_size
        end = min(start + chunk_size, n)
        chunk = b[start:end]
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

    if entropy_weighted:
        selected_bytes = _entropy_weighted_allocation(
            sorted(chunk_infos, key=lambda x: (-x["weight"], x["index"])), target_bytes
        )
    else:
        if entropy_stratify:
            selected_bytes = _stratified_chunk_selection(
                sorted(chunk_infos, key=lambda x: (-x["entropy"], x["index"])),
                target_bytes,
                structural_indices,
            )
        else:
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


def preprocess_file(input_path: Path, output_root: Path, 
                   resize_target_size: tuple[int, int] = (64, 64),
                   truncate_target_size: tuple[int, int] = (256, 256),
                   config_path: str = "config.yaml") -> dict[str, Optional[Path]]:
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
        try:
            cfg = load_config(config_path)
            inference = cfg.get("inference", {})
        except Exception:
            inference = {}
        
        # Process resize mode
        resize_dir = output_root / "resize"
        resize_dir.mkdir(parents=True, exist_ok=True)
        resize_png = resize_dir / f"{sha}.png"
        if not resize_png.exists():
            # Read resize options from config (backward compatibility)
            try:
                resize_interpolation = inference.get("resize_interpolation", "lanczos")
                resize_entropy_hybrid = bool(inference.get("resize_entropy_hybrid", False))
                resize_entropy_ratio = float(inference.get("resize_entropy_ratio", 0.6))
            except Exception:
                resize_interpolation = "lanczos"
                resize_entropy_hybrid = False
                resize_entropy_ratio = 0.6
            
            # Progressive downsampling is always applied (no config option needed)
            img_resize = bytes_to_img_resize(data, width=256, target_size=resize_target_size,
                                             resample=resize_interpolation,
                                             entropy_hybrid=resize_entropy_hybrid,
                                             entropy_ratio=resize_entropy_ratio)
            img_resize.save(resize_png, format="PNG", optimize=True, compress_level=6)
        results["resize"] = resize_png
        
        # Process truncate mode
        truncate_dir = output_root / "truncate"
        truncate_dir.mkdir(parents=True, exist_ok=True)
        truncate_png = truncate_dir / f"{sha}.png"
        if not truncate_png.exists():
            try:
                truncate_chunk_size = int(inference.get("truncate_chunk_size", 512))
                truncate_entropy_stratify = bool(inference.get("truncate_entropy_stratify", True))
                truncate_entropy_weighted = bool(inference.get("truncate_entropy_weighted", False))
                truncate_use_frequency = bool(inference.get("truncate_use_frequency", False))
            except Exception:
                truncate_chunk_size = 512
                truncate_entropy_stratify = True
                truncate_entropy_weighted = False
                truncate_use_frequency = False

            img_truncate = bytes_to_img_truncate(
                data,
                chunk_size=truncate_chunk_size,
                target_size=truncate_target_size,
                entropy_stratify=truncate_entropy_stratify,
                entropy_weighted=truncate_entropy_weighted,
                use_frequency=truncate_use_frequency,
            )
            img_truncate.save(truncate_png, format="PNG", optimize=True, compress_level=6)
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

