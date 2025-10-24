#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-performance trainer for MalNet-FocusAug.

- Config-first: reads training values from config.yaml (no hidden hardcoded defaults).
- CLI overrides: --epochs, --kfold, --mode, --seed.
- Auto-increment seed per mode if --seed not provided (logs/seed_counters.json).
- Group-aware K-Fold by sha256 (sklearn GroupKFold if available).
- CUDA optimizations: AMP (new API), cudnn.benchmark, TF32, channels_last, non-blocking H2D.
- DataLoader optimizations: persistent_workers, pin_memory, prefetch_factor, num_workers.
- Optional torch.compile (only if training.torch_compile:true AND Triton present; disabled on Windows).
- Exports TorchScript to export_models/ with monotonically increasing iteration per mode.

Saved filename: export_models/cnn_<mode>_<seed>_<iter>.ts.pt
"""

from __future__ import annotations
import argparse
import contextlib
import csv
import json
import math
import os
import platform
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Tuple

# Robust imports for layout
try:
    from training.dataset import ByteImageDataset
except Exception:
    from dataset import ByteImageDataset
try:
    from training.model import MalNetFocusAug
except Exception:
    from model import MalNetFocusAug

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x  # minimal fallback


# ----------------- Config helpers -----------------

@dataclass
class TrainCfg:
    epochs: int = 5
    batch_size: int = 24
    amp: bool = True
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "onecycle"
    max_lr: str|float = "auto"
    mode: str = "compress"
    num_workers: int = 4
    prefetch_batches: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    kfold: int = 5
    holdout: int|float = 0
    use_disk_cache: bool = False
    decode_cache_mem_mb: int = 0
    torch_compile: bool = False

def cfg_from_yaml(training_sec: dict) -> TrainCfg:
    """
    Build TrainCfg using only keys present in config.yaml's 'training' section.
    Falls back to dataclass defaults when a key is missing.
    """
    tc = TrainCfg()
    if not isinstance(training_sec, dict):
        return tc
    name_to_field = {f.name: f for f in fields(TrainCfg)}
    for k, v in training_sec.items():
        if k in name_to_field:
            typ = name_to_field[k].type
            try:
                if typ is bool:
                    setattr(tc, k, bool(v))
                elif typ is int:
                    setattr(tc, k, int(v))
                elif typ is float:
                    setattr(tc, k, float(v))
                else:
                    setattr(tc, k, v)
            except Exception:
                setattr(tc, k, v)
    return tc

def load_yaml_cfg(path="config.yaml") -> dict:
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

def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def csv_mode_counts(csv_path: Path) -> Tuple[Dict[str, int], Dict[str, int], int]:
    mode_ct: Dict[str, int] = {}
    label_ct: Dict[str, int] = {}
    total = 0
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            mode_ct[row.get("mode","")] = mode_ct.get(row.get("mode",""), 0) + 1
            label_ct[row.get("label","")] = label_ct.get(row.get("label",""), 0) + 1
    return mode_ct, label_ct, total

def read_csv_rows(csv_path: Path) -> List[dict]:
    rows = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def write_csv_rows(csv_path: Path, rows: List[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def filter_by_mode(rows: List[dict], mode: str) -> List[dict]:
    return [r for r in rows if str(r.get("mode","")).lower() == mode.lower()]

def group_hashes(rows: List[dict]) -> List[str]:
    return [r.get("sha256","") for r in rows]

def auto_lr(max_lr_cfg: str|float, batch_size: int) -> float:
    if isinstance(max_lr_cfg, (int, float)):
        return float(max_lr_cfg)
    # mild scaling with batch size
    base = 1e-3
    return base * (batch_size / 32.0)**0.5

def try_torch_compile(model: nn.Module, enable_flag: bool) -> nn.Module:
    if not enable_flag:
        return model
    # Avoid on Windows by default (Triton wheels often unavailable)
    if platform.system().lower() == "windows":
        print("[compile] Skipping torch.compile on Windows.")
        return model
    # Require Triton
    try:
        import triton  # noqa: F401
    except Exception:
        print("[compile] Triton not found; running eager.")
        return model
    if hasattr(torch, "compile"):
        try:
            print("[compile] Enabling torch.compile (inductor).")
            return torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"[compile] Disabled (reason: {e.__class__.__name__}). Falling back to eager.")
            return model
    return model

def load_json(p: Path, default):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return default
    return default

def save_json(p: Path, obj):
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2))

def next_seed_for_mode(mode: str, override: int|None) -> int:
    """
    If override is provided (CLI --seed), use that.
    Else auto-increment per-mode seed counter in logs/seed_counters.json.
    """
    if override is not None:
        return int(override)
    meta_path = Path("logs/seed_counters.json")
    meta = load_json(meta_path, {})
    cur = int(meta.get(mode, -1))  # start from -1 so first becomes 0
    nxt = cur + 1
    meta[mode] = nxt
    save_json(meta_path, meta)
    return nxt

def next_export_iter_for_mode(mode: str) -> int:
    meta_path = Path("logs/export_iters.json")
    meta = load_json(meta_path, {})
    cur = int(meta.get(mode, 0))
    meta[mode] = cur + 1
    save_json(meta_path, meta)
    return cur  # return current for naming


# ----------------- Data classes -----------------

@dataclass
class TrainCfg:
    epochs: int = 5
    batch_size: int = 24
    amp: bool = True
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "onecycle"
    max_lr: str|float = "auto"
    mode: str = "compress"
    num_workers: int = 4
    prefetch_batches: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    kfold: int = 5
    holdout: int|float = 0  # unused here
    use_disk_cache: bool = False
    decode_cache_mem_mb: int = 0
    torch_compile: bool = False  # opt-in


def load_yaml_cfg(path="config.yaml") -> dict:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        print(f"[CONFIG] WARNING: {p.resolve()} not found. Using TrainCfg defaults.")
        return {}
    try:
        import yaml
    except Exception:
        print(f"[CONFIG] WARNING: PyYAML not installed; cannot parse {p.resolve()}. Using TrainCfg defaults.")
        return {}
    try:
        cfg = yaml.safe_load(p.read_text())
        if not isinstance(cfg, dict):
            print(f"[CONFIG] WARNING: {p.resolve()} did not parse to a dict. Using TrainCfg defaults.")
            return {}
        return cfg
    except Exception as e:
        print(f"[CONFIG] WARNING: Failed to parse {p.resolve()} ({type(e).__name__}: {e}). Using TrainCfg defaults.")
        return {}



# ----------------- Core training -----------------

def build_dataloaders(dataset, tr_idx, va_idx, cfg: TrainCfg):
    tr_ds = Subset(dataset, tr_idx)
    va_ds = Subset(dataset, va_idx)

    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=max(2, cfg.prefetch_batches) if cfg.num_workers > 0 else None,
    )
    tr_loader = DataLoader(tr_ds, shuffle=True, **{k:v for k,v in dl_kwargs.items() if v is not None})
    va_loader = DataLoader(va_ds, shuffle=False, **{k:v for k,v in dl_kwargs.items() if v is not None})
    return tr_loader, va_loader

def one_epoch(model, loader, device, criterion, optimizer, scaler, cfg: TrainCfg, train=True):
    model.train(train)
    running_loss = 0.0
    correct = 0
    total = 0

    iterator = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for xb, yb, _ in iterator:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.to(memory_format=torch.channels_last)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with (torch.amp.autocast(device_type="cuda") if (cfg.amp and device.type=="cuda") else contextlib.nullcontext()):
            logits = model(xb)
            loss = criterion(logits.view(-1), yb.float())

        if train:
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        running_loss += loss.detach().item() * xb.size(0)
        with torch.no_grad():
            preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.numel()

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def _as_val_frac(holdout) -> float:
    # 0–1 means fraction; 1–100 means percent
    try:
        x = float(holdout)
    except Exception:
        return 0.20
    if x <= 0:
        return 0.20
    if x <= 1.0:
        return max(0.05, min(0.95, x))
    return max(0.05, min(0.95, x / 100.0))

def compute_folds(N: int, groups: list[str], kfold: int, seed: int, holdout) -> list[tuple[list[int], list[int]]]:
    """
    Returns a list of (train_idx, val_idx) folds.
    - kfold>=2: GroupKFold if available, else hash-based K buckets.
    - kfold==1: group-aware shuffle split using 'holdout' as validation fraction.
    """
    folds: list[tuple[list[int], list[int]]] = []

    if kfold and kfold >= 2:
        # Preferred: sklearn GroupKFold
        try:
            from sklearn.model_selection import GroupKFold
            kf = GroupKFold(n_splits=kfold)
            for tr_idx, va_idx in kf.split(list(range(N)), groups=groups):
                folds.append((tr_idx.tolist(), va_idx.tolist()))
            return folds
        except Exception:
            # Fallback: hash bucket by group
            from collections import defaultdict
            g2idx = defaultdict(list)
            for i, g in enumerate(groups):
                g2idx[g].append(i)
            buckets = [[] for _ in range(kfold)]
            for g, idxs in g2idx.items():
                fid = (hash(g) + seed) % kfold
                buckets[fid].extend(idxs)
            for f in range(kfold):
                va = buckets[f]
                tr = [i for j in range(kfold) if j != f for i in buckets[j]]
                folds.append((tr, va))
            return folds

    # kfold == 1: group-aware shuffle split
    val_frac = _as_val_frac(holdout)  # e.g., 0.2
    rng = random.Random(seed)
    # Preserve order of first occurrence, then shuffle groups
    unique_groups = list(dict.fromkeys(groups))
    rng.shuffle(unique_groups)
    n_val_groups = max(1, int(round(len(unique_groups) * val_frac)))
    val_set = set(unique_groups[:n_val_groups])

    va_idx = [i for i, g in enumerate(groups) if g in val_set]
    tr_idx = [i for i, g in enumerate(groups) if g not in val_set]

    # Ensure both sides non-empty
    if len(tr_idx) == 0:
        # move one group from val to train
        g0 = unique_groups[0]
        val_set.discard(g0)
        va_idx = [i for i, g in enumerate(groups) if g in val_set]
        tr_idx = [i for i, g in enumerate(groups) if g not in val_set]

    folds.append((tr_idx, va_idx))
    return folds


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=str, required=True)
    ap.add_argument("--images-root", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["compress","truncate"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    # Optional overrides
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs from config")
    ap.add_argument("--kfold", type=int, default=None, help="Override training.kfold from config")
    args = ap.parse_args()

    # Config
    cfg_yaml = load_yaml_cfg("config.yaml")
    tsec = (cfg_yaml.get("training") or {})
    tcfg = cfg_from_yaml(tsec)  # <-- no hardcoded defaults here

    print("\n[CONFIG] training section loaded from config.yaml:")
    print(" ", tsec)

    # Apply CLI overrides
    if args.mode is not None:
        tcfg.mode = args.mode
    if args.epochs is not None:
        tcfg.epochs = int(args.epochs)
    if args.kfold is not None:
        tcfg.kfold = int(args.kfold)

    # Seed: auto-increment per mode unless provided on CLI
    seed = next_seed_for_mode(tcfg.mode, args.seed)
    set_all_seeds(seed)

    data_csv = Path(args.data_csv).resolve()
    images_root = Path(args.images_root).resolve()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device.type}")
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Preflight CSV
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {data_csv}")
    mode_ct, label_ct, total = csv_mode_counts(data_csv)
    print(f"[PREFLIGHT] CSV summary: {data_csv}")
    print(f"  total rows: {total}")
    print(f"  by mode   : {mode_ct}")
    print(f"  by label  : {label_ct}")

    rows = read_csv_rows(data_csv)
    rows_mode = filter_by_mode(rows, tcfg.mode)
    if len(rows_mode) == 0:
        raise RuntimeError(f"No rows for mode='{tcfg.mode}' found in {data_csv}. Did you convert that mode?")

    tmp_csv = Path("tmp") / "conversion_log.filtered.csv"
    ensure_dir(tmp_csv.parent)
    write_csv_rows(tmp_csv, rows_mode)
    print(f"[INFO] Filtered CSV by mode='{tcfg.mode}': kept {len(rows_mode)} rows -> {tmp_csv.as_posix()}")

    # Dataset
    dataset = ByteImageDataset(csv_path=str(tmp_csv), images_root=str(images_root))
    N = len(dataset)
    groups = group_hashes(rows_mode)
    assert len(groups) == N, "Group list misaligned with filtered rows."

    # Effective settings print
    print("\n[CONFIG] Effective training settings")
    print(f"  mode   : {tcfg.mode}")
    print(f"  seed   : {seed}")
    print(f"  epochs : {tcfg.epochs}")
    print(f"  kfold  : {tcfg.kfold}")
    print(f"  batch  : {tcfg.batch_size}")
    print(f"  workers: {tcfg.num_workers} | prefetch: {tcfg.prefetch_batches} | "
          f"pin_memory: {tcfg.pin_memory} | persistent_workers: {tcfg.persistent_workers}")
    print(f"  torch_compile: {tcfg.torch_compile}")

    # Group-aware K-Fold
    folds: List[Tuple[List[int], List[int]]] = []
    try:
        from sklearn.model_selection import GroupKFold
        kf = GroupKFold(n_splits=tcfg.kfold)
        for tr_idx, va_idx in kf.split(list(range(N)), groups=groups):
            folds.append((tr_idx.tolist(), va_idx.tolist()))
    except Exception:
        print("[WARN] sklearn not available; using hash-based group split.")
        from collections import defaultdict
        g2idx = defaultdict(list)
        for i, g in enumerate(groups):
            g2idx[g].append(i)
        fold_buckets = [[] for _ in range(tcfg.kfold)]
        for g, idxs in g2idx.items():
            fid = (hash(g) + seed) % tcfg.kfold
            fold_buckets[fid].extend(idxs)
        for f in range(tcfg.kfold):
            va = fold_buckets[f]
            tr = [i for j in range(tcfg.kfold) if j != f for i in fold_buckets[j]]
            folds.append((tr, va))

    # Model, loss, opt, sched
    model = MalNetFocusAug()  # attention is default True in your model.py
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    model = try_torch_compile(model, enable_flag=tcfg.torch_compile)

    criterion = nn.BCEWithLogitsLoss()
    max_lr = auto_lr(tcfg.max_lr, tcfg.batch_size)

    def make_opt_sched(m: nn.Module, steps_per_epoch: int):
        if tcfg.optimizer.lower() == "adamw":
            opt = optim.AdamW(m.parameters(), lr=max_lr, weight_decay=tcfg.weight_decay)
        else:
            opt = optim.Adam(m.parameters(), lr=max_lr, weight_decay=tcfg.weight_decay)
        if tcfg.scheduler.lower() == "onecycle":
            sch = optim.lr_scheduler.OneCycleLR(
                opt, max_lr=max_lr,
                epochs=tcfg.epochs,
                steps_per_epoch=max(1, steps_per_epoch),
                pct_start=0.1, anneal_strategy="cos"
            )
        else:
            sch = None
        return opt, sch

    use_amp = (tcfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    run_dir = Path("runs") / f"{tcfg.mode}_seed{seed}"
    ensure_dir(run_dir)

    # Train across folds
    # --- folds ---
    folds = compute_folds(N, groups, tcfg.kfold, seed, tcfg.holdout)
    for fidx, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fidx}/{tcfg.kfold} ===")
        print(f"[DEBUG] fold sizes -> train: {len(tr_idx)} | val: {len(va_idx)}")
        tr_loader, va_loader = build_dataloaders(dataset, tr_idx, va_idx, tcfg)
        steps_per_epoch = math.ceil(len(tr_idx) / max(1, tcfg.batch_size))
        optimizer, scheduler = make_opt_sched(model, steps_per_epoch)

        for epoch in range(tcfg.epochs):
            tr_loss, tr_acc = one_epoch(model, tr_loader, device, criterion, optimizer, scaler, tcfg, train=True)
            if scheduler is not None:
                scheduler.step()
            model.eval()
            with torch.inference_mode():
                va_loss, va_acc = one_epoch(model, va_loader, device, criterion, optimizer, scaler, tcfg, train=False)
            print(f"Epoch {epoch+1}/{tcfg.epochs}: "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}")

        # save light checkpoint per fold
        fold_file = run_dir / f"fold{fidx}_last.pt"
        torch.save({"model": model.state_dict(),
                    "cfg": tcfg.__dict__,
                    "seed": seed,
                    "fold": fidx}, fold_file)
        print(f"[CHECKPOINT] Saved fold {fidx} end state to: {run_dir}")

    # Export TorchScript with per-mode iteration
    export_dir = Path("export_models")
    ensure_dir(export_dir)
    iter_id = next_export_iter_for_mode(tcfg.mode)
    out_path = export_dir / f"cnn_{tcfg.mode}_{seed}_{iter_id}.ts.pt"
    model.eval()
    example = torch.randn(1, 1, 256, 256, device=device)
    with torch.inference_mode():
        scripted = torch.jit.trace(model, example)
    scripted.save(str(out_path))
    print(f"[EXPORT] Saved TorchScript model: {out_path.as_posix()}")
    print(f"[EXPORT] Load with: model = torch.jit.load('{out_path.as_posix()}')")


if __name__ == "__main__":
    main()
