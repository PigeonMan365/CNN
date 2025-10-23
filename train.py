#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# project-local
from training.dataset import ByteImageDataset
from training.model import MalNetFocusAug

try:
    import yaml
except ImportError:
    print("[ERROR] Missing PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# ----------------------------- utils ---------------------------------
def load_cfg(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def preflight_csv(csv_path: Path):
    total = 0
    by_mode = {}
    by_label = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            m = row.get("mode", "")
            l = str(row.get("label", ""))
            by_mode[m] = by_mode.get(m, 0) + 1
            by_label[l] = by_label.get(l, 0) + 1
    print(f"[PREFLIGHT] CSV summary: {csv_path}")
    print(f"  total rows: {total}")
    print(f"  by mode   : {by_mode}")
    print(f"  by label  : {by_label}")

def filter_csv_by_mode(csv_in: Path, mode: str) -> Path:
    tmp = Path("/tmp/conversion_log.filtered.csv")
    kept = 0
    with open(csv_in, "r", newline="") as fi, open(tmp, "w", newline="") as fo:
        r = csv.DictReader(fi)
        fieldnames = r.fieldnames or ["rel_path", "label", "mode", "sha256"]
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for row in r:
            if row.get("mode") == mode:
                w.writerow(row)
                kept += 1
    print(f"[INFO] Filtered CSV by mode='{mode}': kept {kept} rows -> {tmp}")
    if kept == 0:
        raise RuntimeError(
            f"No rows for mode='{mode}' found in {csv_in}. Did you convert that mode?"
        )
    return tmp

def load_groups_from_csv(csv_path: Path, key: str = "sha256") -> List[str]:
    groups = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            groups.append(row.get(key, "NA"))
    return groups

def make_group_kfold_splits(groups: List[str], k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """
    Deterministic group-aware K-fold by 'key' (sha256):
      - unique groups shuffled by seed
      - sliced into k folds
      - for fold i: val = indices whose group in fold i; train = others
    """
    seen = {}
    for g in groups:
        if g not in seen:
            seen[g] = len(seen)
    uniq = list(seen.keys())
    rnd = random.Random(seed)
    rnd.shuffle(uniq)

    folds = []
    n = len(uniq)
    for i in range(k):
        start = math.floor(i * n / k)
        end = math.floor((i + 1) * n / k)
        folds.append(set(uniq[start:end]))

    splits = []
    for i in range(k):
        val_groups = folds[i]
        tr_idx, va_idx = [], []
        for idx, g in enumerate(groups):
            (va_idx if g in val_groups else tr_idx).append(idx)
        splits.append((tr_idx, va_idx))
    return splits

def accuracy_from_logits(logits, targets):
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return (preds == targets.long()).float().mean().item()

# --------- per-mode monotonic export iteration counter ----------
def next_iteration_id(mode: str, index_path: Path) -> int:
    """
    Maintain a per-mode monotonically increasing iteration number (starting at 0).
    Stored in logs/export_iters.json as: {"compress": 3, "truncate": 7}
    Returns the next id to use and updates the file.
    """
    data = {}
    if index_path.exists():
        try:
            data = json.loads(index_path.read_text())
        except Exception:
            data = {}
    cur = int(data.get(mode, -1))
    nxt = cur + 1
    data[mode] = nxt
    ensure_dir(index_path.parent)
    index_path.write_text(json.dumps(data, indent=2))
    return nxt

# ----------------------------- epochs --------------------------------
def train_one_epoch(model, tr_indices, base_dataset, device, criterion, optimizer,
                    batch_size, num_workers, prefetch_batches, pin_memory,
                    sample_fraction_percent: int, scaler=None):
    """
    Build the train DataLoader fresh each epoch to support per-epoch subsampling via sample_fraction_percent.
    """
    if sample_fraction_percent is not None and 0 <= sample_fraction_percent < 100:
        k = max(1, int(len(tr_indices) * (sample_fraction_percent / 100.0)))
        tr_indices = tr_indices[:]  # copy so we can shuffle
        random.shuffle(tr_indices)
        tr_indices = tr_indices[:k]

    tr_loader = DataLoader(
        Subset(base_dataset, tr_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_batches if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=pin_memory,
        drop_last=False,
    )

    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    pbar = tqdm(tr_loader, desc="Train", leave=False)
    for xb, yb, _ in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(xb).flatten()
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb).flatten()
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits.detach(), yb) * bs
        total_n    += bs
        pbar.set_postfix(loss=f"{total_loss/total_n:.4f}")
    return total_loss/total_n, total_acc/total_n

@torch.no_grad()
def validate_one_epoch(model, va_indices, base_dataset, device, criterion,
                       batch_size, num_workers, prefetch_batches, pin_memory):
    val_loader = DataLoader(
        Subset(base_dataset, va_indices),
        batch_size=max(1, batch_size * 2),
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_batches if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        pin_memory=pin_memory,
        drop_last=False,
    )

    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    pbar = tqdm(val_loader, desc="Valid", leave=False)
    for xb, yb, _ in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()
        logits = model(xb).flatten()
        loss = criterion(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, yb) * bs
        total_n    += bs
        pbar.set_postfix(loss=f"{total_loss/total_n:.4f}")
    return total_loss/total_n, total_acc/total_n

# --------------------- fold-level checkpointing ----------------------
def save_fold_checkpoint(run_dir: Path, fold_idx: int, model, optimizer, scaler,
                         meta: dict):
    ensure_dir(run_dir)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "meta": meta,
    }
    torch.save(ckpt, run_dir / f"fold{fold_idx}_end.pt")
    # write resume pointer
    (run_dir / "resume.json").write_text(json.dumps(meta, indent=2))

def load_last_checkpoint_if_any(run_dir: Path):
    res_p = run_dir / "resume.json"
    if not res_p.exists():
        return None, None, None, None
    try:
        meta = json.loads(res_p.read_text())
    except Exception:
        return None, None, None, None
    # load last fold end if exists
    last_fold = meta.get("last_completed_fold", 0)
    ckpt_p = run_dir / f"fold{last_fold}_end.pt"
    if ckpt_p.exists():
        ckpt = torch.load(ckpt_p, map_location="cpu")
        return ckpt, meta, last_fold, ckpt_p
    return None, meta, last_fold, None

# ------------------------------ main ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--data-csv", type=str, default=None)
    ap.add_argument("--images-root", type=str, default=None)
    # Orchestration overrides:
    ap.add_argument("--mode", type=str, choices=["compress", "truncate"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    # New: fold-level resume
    ap.add_argument("--resume", action="store_true", help="Resume this run from the next unfinished fold if possible.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # IO paths
    ti = cfg.get("train_io", {})
    data_csv = Path(args.data_csv or ti.get("data_csv", "logs/conversion_log.csv"))
    images_root = Path(args.images_root or ti.get("images_root", "dataset/output")).resolve()
    export_dir = Path("export_models"); ensure_dir(export_dir)
    export_index = Path("logs/export_iters.json")

    # Training params
    tr = cfg.get("training", {})
    # Mode + seed (overridable by CLI)
    mode = (args.mode if args.mode is not None else str(tr.get("mode", "compress"))).strip()
    base_seed = int(tr.get("seed", 0))
    seed = (args.seed if args.seed is not None else base_seed)
    set_seed(seed)

    epochs = int(tr.get("epochs", 5))
    batch_size = int(tr.get("batch_size", 8))
    lr = float(tr.get("lr", 1e-3))
    weight_decay = float(tr.get("weight_decay", 1e-4))
    amp = bool(tr.get("amp", True))
    kfold = int(tr.get("kfold", 5))
    # Perf knobs
    num_workers = int(tr.get("num_workers", 2))
    prefetch_batches = int(tr.get("prefetch_batches", 2))
    sample_fraction_percent = int(tr.get("sample_fraction", 100))  # percent 0..100

    # Caching knobs to pass to dataset
    use_disk_cache = bool(tr.get("use_disk_cache", True))
    paths = cfg.get("paths", {})
    cache_root = paths.get("cache_root", "cache")
    cache_max_bytes = paths.get("cache_max_bytes", "40GB")
    decode_cache_mem_mb = int(tr.get("decode_cache_mem_mb", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    print(f"[INFO] Using device: {device}")

    # Preflight & mode filtering
    preflight_csv(data_csv)
    filtered_csv = filter_csv_by_mode(data_csv, mode)

    # Groups for K-fold split (by sha256)
    groups = load_groups_from_csv(filtered_csv, key="sha256")
    splits = make_group_kfold_splits(groups, max(2, kfold), seed)
    total_folds = len(splits)

    # Dataset
    dataset = ByteImageDataset(
        csv_path=str(filtered_csv),
        images_root=str(images_root),
        normalize="01",
        use_disk_cache=use_disk_cache,
        cache_root=cache_root,
        cache_max_bytes=cache_max_bytes,
        decode_cache_mem_mb=decode_cache_mem_mb,
    )

    # Model / Optimizer / Loss / AMP
    model = MalNetFocusAug().to(device)  # attention handled inside implementation
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None

    # Run directory (for fold checkpoints & resume pointer)
    run_dir = Path("runs") / f"{mode}_seed{seed}"
    ensure_dir(run_dir)

    # ----- RESUME HANDLING -----
    start_fold = 1
    # If resume requested and pointer exists, load the last completed fold state
    if args.resume:
        ckpt, meta, last_fold, ckpt_path = load_last_checkpoint_if_any(run_dir)
        if meta is not None:
            # Restore model/optimizer/scaler if we have a real checkpoint file
            if ckpt is not None and ckpt_path is not None:
                model.load_state_dict(ckpt["model"])
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception:
                    pass
                if scaler is not None and ckpt.get("scaler"):
                    try:
                        scaler.load_state_dict(ckpt["scaler"])
                    except Exception:
                        pass
                start_fold = int(meta.get("last_completed_fold", 0)) + 1
                print(f"[RESUME] Loaded {ckpt_path.name}. Resuming at fold {start_fold}/{total_folds}.")
            else:
                start_fold = int(meta.get("last_completed_fold", 0)) + 1
                print(f"[RESUME] No checkpoint file, but resume.json present. Starting at fold {start_fold}/{total_folds}.")
        else:
            print("[RESUME] No resume.json found; starting from fold 1.")

    best_val_loss = float("inf")
    best_state = None

    # Main K-fold loop
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        if fold_idx < start_fold:
            print(f"[SKIP] Fold {fold_idx} already completed (per resume.json).")
            continue

        print(f"\n=== Fold {fold_idx}/{total_folds} ===")
        for epoch in range(epochs):
            tr_loss, tr_acc = train_one_epoch(
                model, tr_idx, dataset, device, criterion, optimizer,
                batch_size, num_workers, prefetch_batches, pin,
                sample_fraction_percent, scaler
            )
            va_loss, va_acc = validate_one_epoch(
                model, va_idx, dataset, device, criterion,
                batch_size, num_workers, prefetch_batches, pin
            )
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}")

        # track best (by val_loss) across folds (using final epoch val)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {
                "model": model.state_dict(),
                "fold": fold_idx,
                "epoch": epoch + 1,
                "val_loss": va_loss,
                "seed": seed,
                "mode": mode,
                "kfold": total_folds,
            }

        # ---- save fold-level checkpoint & resume pointer ----
        meta = {
            "mode": mode,
            "seed": seed,
            "kfold": total_folds,
            "last_completed_fold": fold_idx,
            "best_val_loss": best_val_loss,
        }
        save_fold_checkpoint(run_dir, fold_idx, model, optimizer, scaler, meta)
        print(f"[CHECKPOINT] Saved fold {fold_idx} end state to: {run_dir}")

    # Save best checkpoint as reference
    if best_state is not None:
        ckpt_path = Path("runs") / f"best_{mode}_seed{seed}.pt"
        ensure_dir(ckpt_path.parent)
        torch.save(best_state, ckpt_path)

    # TorchScript export with per-mode iteration number (no overwrite)
    iter_id = next_iteration_id(mode, export_index)  # 0,1,2,...
    export_path = export_dir / f"cnn_{mode}_{seed}_{iter_id}.ts.pt"
    model.eval()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(export_path))
    print(f"[EXPORT] Saved TorchScript model: {export_path}")
    print(f"[EXPORT] Load with: model = torch.jit.load('{export_path.as_posix()}')")

if __name__ == "__main__":
    main()

