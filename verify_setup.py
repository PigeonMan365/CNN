#!/usr/bin/env python3
"""
verify_setup.py — environment + data audit with training knob recommendations.

What it does:
- Prints Python & PyTorch info, CUDA/GPU, CPU cores, RAM.
- Reads config.yaml to find:
    - train_io.data_csv   (usually logs/conversion_log.csv)
    - train_io.images_root (dataset/output by default)
    - paths.cache_root
- Scans the CSV to summarize dataset size per mode (compress/truncate) and
  approximate unique groups (sha256).
- Recommends:
    training.batch_size
    training.num_workers
    training.prefetch_batches
    training.decode_cache_mem_mb
    paths.cache_max_bytes
    training.kfold
- DOES NOT change your files; prints suggestions for you to copy.
"""

import os
import sys
import csv
import json
import math
import platform
from pathlib import Path
from typing import Dict, Tuple

# Optional deps
try:
    import yaml
except Exception:
    yaml = None

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


# ----------------------------- helpers -----------------------------

def _hb(n: float) -> str:
    """Human bytes."""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f}{units[i]}"

def _read_cfg(cfg_path: Path) -> dict:
    if yaml is None:
        print("[WARN] PyYAML not installed; cannot read config.yaml. Install: pip install pyyaml")
        return {}
    if not cfg_path.exists():
        print(f"[WARN] {cfg_path} not found.")
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text())
    except Exception as e:
        print(f"[WARN] Could not parse {cfg_path}: {e}")
        return {}

def _dataset_summary(csv_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns:
      (by_mode_count, by_mode_unique_groups)
    """
    by_mode = {}
    by_mode_groups = {}
    if not csv_path.exists():
        print(f"[WARN] CSV not found: {csv_path}")
        return by_mode, by_mode_groups

    try:
        with csv_path.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                mode = row.get("mode", "").strip()
                by_mode[mode] = by_mode.get(mode, 0) + 1
                g = (row.get("sha256") or "").strip()
                if mode not in by_mode_groups:
                    by_mode_groups[mode] = set()
                if g:
                    by_mode_groups[mode].add(g)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return by_mode, {k: len(v) for k, v in by_mode_groups.items()}

    return by_mode, {k: len(v) for k, v in by_mode_groups.items()}

def _gpu_info():
    if torch is None:
        return None
    if not torch.cuda.is_available():
        return None
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "name": props.name,
        "total_mem": props.total_memory,
        "sm_count": getattr(props, "multi_processor_count", None)
    }

def _cpu_info():
    cores = os.cpu_count() or 1
    ram_total = None
    ram_avail = None
    if psutil:
        vm = psutil.virtual_memory()
        ram_total = vm.total
        ram_avail = vm.available
    return {"cores": cores, "ram_total": ram_total, "ram_avail": ram_avail}

def _disk_free(path: Path):
    if psutil is None:
        return None
    try:
        usage = psutil.disk_usage(str(path))
        return usage  # total, used, free, percent
    except Exception:
        return None


# ------------------------ recommendations --------------------------

def recommend_batch_size(gpu, cpu) -> int:
    """
    Heuristic for grayscale 256x256, small CNN:
    - Large VRAM -> larger batches
    - CPU-only -> conservative based on RAM/cores
    """
    if gpu:
        vram = gpu["total_mem"]
        if vram >= 24 * 1024**3: return 128
        if vram >= 16 * 1024**3: return 64
        if vram >= 12 * 1024**3: return 48
        if vram >=  8 * 1024**3: return 32
        if vram >=  6 * 1024**3: return 24
        if vram >=  4 * 1024**3: return 16
        return 8
    # CPU-only
    cores = cpu["cores"]
    avail = cpu["ram_avail"] or (8 * 1024**3)
    if avail >= 32 * 1024**3 and cores >= 16: return 32
    if avail >= 16 * 1024**3 and cores >= 8:  return 16
    if avail >=  8 * 1024**3 and cores >= 4:  return 8
    return 4

def recommend_workers_prefetch(cores: int, rows: int) -> Tuple[int, int]:
    """
    General-purpose default:
    - workers: clamp between 2 and 8 by cores
    - prefetch: 2 for many workers; 4 if workers small or dataset small
    """
    workers = max(2, min(8, (cores // 2) or 1))
    if rows < 200:
        # tiny dataset: keep it simple
        prefetch = 2
        workers = max(2, min(workers, 4))
    else:
        prefetch = 2 if workers >= 4 else 4
    return workers, prefetch

def recommend_decode_cache_mb(avail_ram: int) -> int:
    """
    Allocate ~10% of available RAM, clamp to [256MB, 4096MB].
    """
    if not avail_ram:
        return 512
    rec = int(0.10 * (avail_ram / (1024*1024)))
    rec = max(256, min(rec, 4096))
    return rec

def recommend_cache_max_bytes(free_disk_bytes: int) -> str:
    """
    Use ~40% of free space, clamp to [10GB, 200GB].
    """
    if not free_disk_bytes:
        return "40GB"
    target = int(0.40 * free_disk_bytes)
    target = max(10 * 1024**3, min(target, 200 * 1024**3))
    return _hb(target)

def recommend_kfold(unique_groups: int) -> int:
    """
    Rule of thumb (by unique sha256 groups per mode):
      >= 2000 -> 10
      1000-1999 -> 8
      500-999 -> 5
      200-499 -> 4
      100-199 -> 3
      <100 -> 2  (or consider holdout if extremely tiny)
    """
    g = unique_groups
    if g >= 2000: return 10
    if g >= 1000: return 8
    if g >=  500: return 5
    if g >=  200: return 4
    if g >=  100: return 3
    return 2


# ------------------------------ main --------------------------------

def main():
    print("=== VERIFY SETUP ===")
    print(f"Python : {platform.python_version()} ({sys.executable})")
    if torch is not None:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA   : {'available' if torch.cuda.is_available() else 'not available'}")
    else:
        print("PyTorch: NOT INSTALLED (pip install torch)")

    gpu = _gpu_info() if torch and torch.cuda.is_available() else None
    if gpu:
        print(f"GPU    : {gpu['name']}  VRAM={_hb(gpu['total_mem'])}")

    cpu = _cpu_info()
    print(f"CPU    : cores={cpu['cores']}")
    if cpu["ram_total"] is not None:
        print(f"RAM    : total={_hb(cpu['ram_total'])}  avail≈{_hb(cpu['ram_avail'])}")

    # Config paths
    cfg = _read_cfg(Path("config.yaml"))
    ti = cfg.get("train_io", {})
    paths = cfg.get("paths", {})

    data_csv = Path(ti.get("data_csv", "logs/conversion_log.csv"))
    images_root = Path(ti.get("images_root", "dataset/output"))
    cache_root = Path(paths.get("cache_root", "cache"))

    print("\n--- Paths (from config or defaults) ---")
    print(f"data_csv   : {data_csv}")
    print(f"images_root: {images_root.resolve()}")
    print(f"cache_root : {cache_root.resolve()}")

    # CSV summary
    by_mode, by_mode_groups = _dataset_summary(data_csv)
    total_rows = sum(by_mode.values()) if by_mode else 0
    print("\n--- Dataset summary (from conversion_log.csv) ---")
    if total_rows == 0:
        print("No rows found (did you run conversion yet?).")
    else:
        print(f"Total rows: {total_rows}")
        for m in ("compress", "truncate"):
            n = by_mode.get(m, 0)
            g = by_mode_groups.get(m, 0)
            print(f"  {m:9s}: rows={n:6d} | unique_groups(sha256)={g}")

    # Disk free for cache_root (if psutil available)
    free_bytes = None
    if psutil is not None:
        du = _disk_free(cache_root)
        if du:
            free_bytes = du.free
            print(f"\ncache_root free space: {_hb(du.free)} / total {_hb(du.total)}")

    # ---------------- Recommendations ----------------
    print("\n=== RECOMMENDATIONS (copy into config.yaml) ===")

    # batch_size
    bs = recommend_batch_size(gpu, cpu)
    print("training.batch_size:", bs)

    # workers & prefetch (use total rows to guess)
    workers, prefetch = recommend_workers_prefetch(cpu["cores"], total_rows)
    print("training.num_workers:", workers)
    print("training.prefetch_batches:", prefetch)
    print("training.use_disk_cache: true")  # good default for consistent staging

    # decode cache (RAM)
    dec_mb = recommend_decode_cache_mb(cpu["ram_avail"])
    print("training.decode_cache_mem_mb:", dec_mb)

    # cache size on disk
    cache_cap = recommend_cache_max_bytes(free_bytes)
    print("paths.cache_max_bytes:", cache_cap)

    # kfold per mode (choose the smaller of the two mode-based recs so both are feasible)
    if total_rows > 0:
        k1 = recommend_kfold(by_mode_groups.get("compress", 0))
        k2 = recommend_kfold(by_mode_groups.get("truncate", 0))
        k = min(max(2, k1), max(2, k2)) if (k1 and k2) else max(k1 or 0, k2 or 0, 2)
        print("training.kfold:", k)
    else:
        print("training.kfold: 5  # (default suggestion; adjust after you have data)")

    print("\nNotes:")
    print("- batch_size is constrained by GPU VRAM (or CPU RAM if no GPU).")
    print("- num_workers are background loader processes; persistent workers + prefetch keep batches ready.")
    print("- decode_cache_mem_mb limits in-RAM decoded PNG tensors; increase on big-memory desktop.")
    print("- cache_max_bytes caps SSD staging cache; keep it below ~50% of free SSD space.")
    print("- kfold uses unique sha256 counts; we picked a conservative value that fits both modes.")


if __name__ == "__main__":
    main()

