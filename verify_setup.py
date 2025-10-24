#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_setup.py — richer environment & data diagnostics + speed-oriented recommendations
Now also considers SSD/disk space for images_root and cache_root in its guidance.

It prints:
- Python, PyTorch, CUDA, GPU name/VRAM, CPU cores, System RAM
- Requirements check (imports) from requirements.txt
- Paths (from config.yaml or defaults)
- Disk usage (total/free) for images_root and cache_root + same-device check
- Dataset summary (from logs/conversion_log.csv)
- Speed recommendations (batch size, workers, prefetch, pin_memory, persistent_workers, AMP)
- Cache recommendations based on *free disk* at cache_root (paths.cache_max_bytes & use_disk_cache)
"""

from __future__ import annotations
import os
import sys
import csv
import json
import shutil
import importlib
from pathlib import Path

# ---------------- helpers ----------------

def load_yaml(path="config.yaml") -> dict:
    try:
        import yaml
    except Exception:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text())
    except Exception:
        return {}

def bytes_to_gb(n: int) -> float:
    return round(n / (1024**3), 2)

def get_ram_info():
    # psutil is preferred; fallbacks provided
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total": vm.total, "available": vm.available, "used": vm.used}
    except Exception:
        if os.name == "nt":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return {
                    "total": int(stat.ullTotalPhys),
                    "available": int(stat.ullAvailPhys),
                    "used": int(stat.ullTotalPhys - stat.ullAvailPhys),
                }
            except Exception:
                return {}
        else:
            try:
                info = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        k, v = line.strip().split(":", 1)
                        info[k] = v.strip()
                def kB(x): return int(x.split()[0]) * 1024
                total = kB(info.get("MemTotal", "0 kB"))
                free = kB(info.get("MemAvailable", "0 kB"))
                used = total - free
                return {"total": total, "available": free, "used": used}
            except Exception:
                return {}

def get_gpu_info():
    info = {"cuda_available": False, "name": None, "total_mem": None}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"] and torch.cuda.device_count() > 0:
            i = torch.cuda.current_device()
            info["name"] = torch.cuda.get_device_name(i)
            info["total_mem"] = torch.cuda.get_device_properties(i).total_memory
    except Exception:
        pass
    return info

def parse_requirements(req_path="requirements.txt"):
    reqs = []
    p = Path(req_path)
    if not p.exists():
        return reqs
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        pkg = s.split(";")[0].split("==")[0].split(">=")[0].split("<=")[0].strip()
        if pkg:
            reqs.append(pkg)
    return reqs

def import_name_for(pkg: str) -> str:
    m = {
        "pillow": "PIL",
        "pyyaml": "yaml",
        "scikit-learn": "sklearn",
        "opencv-python": "cv2",
        "pandas": "pandas",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
        "psutil": "psutil",
        "torch": "torch",
        "torchvision": "torchvision",
        "triton": "triton",
    }
    return m.get(pkg.lower(), pkg.replace("-", "_"))

def requirement_status(reqs):
    missing = []
    present = []
    for pkg in reqs:
        mod = import_name_for(pkg)
        try:
            importlib.import_module(mod)
            present.append(pkg)
        except Exception:
            missing.append(pkg)
    return present, missing

def summarize_csv(csv_path: Path):
    total = 0
    by_mode = {}
    by_label = {}
    if not csv_path.exists():
        return total, by_mode, by_label
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            by_mode[row.get("mode","")] = by_mode.get(row.get("mode",""), 0) + 1
            by_label[row.get("label","")] = by_label.get(row.get("label",""), 0) + 1
    return total, by_mode, by_label

def disk_usage_info(path: Path):
    """
    Return (exists, total, used, free) for the filesystem hosting 'path'.
    """
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        return True, usage.total, usage.used, usage.free
    except Exception:
        return False, None, None, None

def same_device(a: Path, b: Path) -> bool | None:
    """
    Best-effort check if two paths are on the same device/volume.
    On POSIX: compares st_dev; on Windows: compares drive letter root.
    Returns True/False or None if unknown.
    """
    try:
        if os.name == "nt":
            return a.drive.lower() == b.drive.lower()
        else:
            return os.stat(a).st_dev == os.stat(b).st_dev
    except Exception:
        try:
            if os.name == "nt":
                return a.drive.lower() == b.drive.lower()
        except Exception:
            pass
        return None

# -------------- main --------------

def main():
    print("=== VERIFY SETUP ===")
    print(f"Python : {sys.version.split()[0]} ({sys.executable})")
    # Torch/CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except Exception:
        print("PyTorch: not installed")
        torch = None

    gpu = get_gpu_info()
    if gpu["cuda_available"]:
        mem_gb = bytes_to_gb(gpu["total_mem"]) if gpu["total_mem"] else "?"
        print(f"CUDA   : available | GPU='{gpu['name']}' | VRAM={mem_gb} GB")
    else:
        print("CUDA   : not available")

    # CPU + RAM
    try:
        import multiprocessing as mp
        cores = mp.cpu_count()
    except Exception:
        cores = "?"
    ram = get_ram_info()
    if ram:
        print(f"CPU    : cores={cores}")
        print(f"RAM    : total={bytes_to_gb(ram['total'])} GB | available={bytes_to_gb(ram['available'])} GB")
    else:
        print(f"CPU    : cores={cores}")
        print("RAM    : unknown (install psutil for detailed RAM)")

    # Paths from config
    cfg = load_yaml("config.yaml") or {}
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})
    data_csv    = Path(ti.get("data_csv", paths.get("conversion_log", "logs/conversion_log.csv")))
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    cache_root  = Path(paths.get("cache_root", "cache")).resolve()

    print("\n--- Paths (from config or defaults) ---")
    print(f"data_csv    : {data_csv}")
    print(f"images_root : {images_root}")
    print(f"cache_root  : {cache_root}")

    # Disk usage for images_root and cache_root
    print("\n--- Disk usage (host volumes) ---")
    img_exists, img_total, img_used, img_free = disk_usage_info(images_root)
    cac_exists, cac_total, cac_used, cac_free = disk_usage_info(cache_root)
    if img_exists:
        print(f"images_root : total={bytes_to_gb(img_total)} GB | used={bytes_to_gb(img_used)} GB | free={bytes_to_gb(img_free)} GB")
    else:
        print("images_root : (volume not found; check path)")
    if cac_exists:
        print(f"cache_root  : total={bytes_to_gb(cac_total)} GB | used={bytes_to_gb(cac_used)} GB | free={bytes_to_gb(cac_free)} GB")
    else:
        print("cache_root  : (volume not found; check path)")

    sd = same_device(images_root, cache_root)
    if sd is True:
        print("NOTE       : images_root and cache_root appear to be on the SAME device/volume.")
        print("            Staging cache still helps (decode/prefetch), but copy speed gains may be limited.")
    elif sd is False:
        print("NOTE       : images_root and cache_root are on DIFFERENT devices (good for staging).")
    else:
        print("NOTE       : Could not determine if images_root and cache_root share the same device.")

    # Dataset summary
    print("\n--- Dataset summary (from conversion_log.csv) ---")
    total, by_mode, by_label = summarize_csv(data_csv)
    print(f"Total rows: {total}")
    print(f"  modes   : {by_mode}")
    print(f"  labels  : {by_label}")

    # Requirements check
    print("\n--- Requirements check ---")
    reqs = parse_requirements("requirements.txt")
    if not reqs:
        print("requirements.txt not found or empty (skipping import test).")
        present = []
        missing = []
    else:
        present, missing = requirement_status(reqs)
        if present:
            print("present :", ", ".join(sorted(present)))
        if missing:
            print("missing :", ", ".join(sorted(missing)))
            print("Hint    : pip install -r requirements.txt")

    # ---------------- SPEED & CACHE RECOMMENDATIONS ----------------
    print("\n=== RECOMMENDATIONS (copy into config.yaml) ===")

    # Batch size heuristic (single-channel 256x256):
    # Larger VRAM -> larger batch; CPU-only -> small/moderate.
    if gpu["cuda_available"] and gpu["total_mem"]:
        vram_gb = gpu["total_mem"] / (1024**3)
        if vram_gb < 4.5:
            bs = 32
        elif vram_gb < 9:
            bs = 64
        elif vram_gb < 13:
            bs = 96
        else:
            bs = 128
    else:
        bs = 12 if ram and ram["available"] > 8*(1024**3) else 8

    # Workers/prefetch tuned by CPU cores and RAM
    if isinstance(cores, int):
        nw = max(2, min(12, cores // 2))
    else:
        nw = 4

    if ram and ram["available"] >= 8*(1024**3):
        prefetch = 4
        decode_mb = 1024
    else:
        prefetch = 2
        decode_mb = 512

    pin_mem = True if (gpu["cuda_available"]) else False
    persistent = True if nw > 0 else False
    amp = True if gpu["cuda_available"] else False

    # Cache sizing from *free space* at cache_root
    cache_cap_str = "40GB"
    use_disk_cache = True
    if cac_exists and cac_free is not None:
        # If free space is tiny, recommend disabling disk cache to avoid churn
        if cac_free < 8*(1024**3):  # < 8 GB free
            use_disk_cache = False
            cache_cap_str = "0GB"
        else:
            # use ~30% of free space, minimum 10 GB, capped at total
            suggest_bytes = int(cac_free * 0.30)
            suggest_gb = max(10, suggest_bytes // (1024**3))
            # never exceed total
            if cac_total:
                suggest_gb = min(suggest_gb, max(1, int(cac_total // (1024**3)) - 4))  # keep 4GB headroom
            cache_cap_str = f"{suggest_gb}GB"
    else:
        # Unknown volume; conservative default
        use_disk_cache = True
        cache_cap_str = "40GB"

    print("training.batch_size:", bs)
    print("training.num_workers:", nw)
    print("training.prefetch_batches:", prefetch)
    print("training.pin_memory:", str(pin_mem).lower())
    print("training.persistent_workers:", str(persistent).lower())
    print("training.amp:", str(amp).lower())
    print(f"training.use_disk_cache: {str(use_disk_cache).lower()}")
    print(f"training.decode_cache_mem_mb: {decode_mb}")
    print(f"paths.cache_max_bytes: {cache_cap_str}")

    # Extra human notes
    print("\nNotes:")
    print("- Batch size increases GPU utilization until you hit OOM; then back off.")
    print("- num_workers/prefetch keep the pipeline fed; tune with CPU cores and RAM.")
    print("- pin_memory+persistent_workers reduce H2D latency on CUDA.")
    print("- decode_cache_mem_mb is an in-RAM budget for decoded PNG tensors; raise on big-RAM hosts.")
    print("- paths.cache_max_bytes is sized from FREE space on the cache volume (~30% by default).")
    if sd is True:
        print("- Cache and dataset on SAME device: staging still helps decode/prefetch, but copy benefits are smaller.")
    elif sd is False:
        print("- Cache and dataset on DIFFERENT devices: good for I/O parallelism; keep cache_root on an SSD if possible.")
    if missing:
        print("- Install missing requirements to avoid slow fallbacks/import errors.")

if __name__ == "__main__":
    main()
