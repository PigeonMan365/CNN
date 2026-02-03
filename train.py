#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer for MalNet-FocusAug — Option A (best single fold export)
- Config-driven defaults with CLI override
- Grouped+stratified folds (by sha256, label)
- Mild positive oversampling per batch (≈5–10% positives ≈ 1:19–1:9)
- Metrics: PR-AUC (primary), ROC-AUC, max-F1, F1 at operating point
- Operating point chosen by FPR budget
- Per-fold best checkpointing by PR-AUC; export best fold as TorchScript
- Safe interrupt + resume: writes runstate.json and fold{N}_interrupt.pt; supports --resume
- Windows-safe dataloading (num_workers=0, no persistent workers)
"""

from __future__ import annotations
import argparse, csv, json, math, os, random, signal, sys, platform
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset, BatchSampler

# ---------- Robust imports for dataset/model (avoid folder collisions) ----------
import importlib.util as _ilu
from importlib import import_module as _import

_HERE = Path(__file__).resolve().parent

def _dyn_import(candidates, attr):
    for fp in candidates:
        if fp.exists():
            spec = _ilu.spec_from_file_location(fp.stem, fp)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, attr)
    # fallbacks to package imports if available
    for name in ["dataset", "training.dataset", "model", "training.model"]:
        try:
            mod = _import(name)
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except Exception:
            continue
    raise ImportError(f"Could not import {attr} from candidates: {candidates}")

ByteImageDataset = _dyn_import([_HERE / "dataset.py", _HERE / "training" / "dataset.py"], "ByteImageDataset")
MalNetFocusAug  = _dyn_import([_HERE / "model.py",   _HERE / "training" / "model.py"],   "MalNetFocusAug")

# ---------- Metrics utils ----------
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.utils import check_random_state


# ---------- Config loader ----------
def _load_config_dict() -> Dict:
    # Preferred: project helper if available
    try:
        from utils.paths import load_config as _load_config
        cfg = _load_config()
        if hasattr(cfg, 'get'):
            return cfg
        if hasattr(cfg, '__dict__'):
            return dict(cfg.__dict__)
    except Exception:
        pass
    # Fallback: read ./config.yaml using PyYAML if present
    cfg_path = _HERE / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def _get(d: Dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------- CLI ----------
@dataclass
class TrainArgs:
    data_csv: str
    images_root: str
    mode: str
    seed: int = 0
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 4
    prefetch_batches: int = 2
    pin_memory: bool = False
    persistent_workers: bool = True
    device: str = "auto"
    kfold: int = 5
    holdout: int = 0
    resume: bool = False
    fpr_budget: float = 0.001
    oversample_pos_min: float = 0.05
    oversample_pos_max: float = 0.10
    optimizer: str = "adamw"
    scheduler: str = "onecycle"
    max_lr: str = "auto"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = False
    runs_root: str = "runs"
    export_root: str = "export_models"
    decode_cache_mem_mb: int = 0


def parse_cli() -> TrainArgs:
    """
    Simplified CLI - only essential arguments.
    Everything else comes from config.yaml.
    """
    p = argparse.ArgumentParser(description="Train MalNet-FocusAug")
    p.add_argument("--data-csv", help="Override data CSV path (default: from config)")
    p.add_argument("--images-root", help="Override images root (default: from config)")
    p.add_argument("--mode", choices=["resize", "truncate"], 
                   help="Override training mode (default: from config.training.mode)")
    p.add_argument("--seed", type=int, help="Override seed (default: auto-increment or from config)")
    p.add_argument("--resume", action="store_true", 
                   help="Resume training from checkpoint")
    a = p.parse_args()

    # Create minimal args - will be filled from config
    args = TrainArgs(
        data_csv=a.data_csv or "",  # Will be set from config if not provided
        images_root=a.images_root or "",  # Will be set from config if not provided
        mode=a.mode or "",  # Will be set from config if not provided
    )
    return args, a


def apply_config_and_cli_defaults(args: TrainArgs, raw_cli) -> TrainArgs:
    cfg = _load_config_dict()
    
    # Helper to parse target size from string or config
    def _parse_target_size(value, default: tuple[int, int]) -> tuple[int, int]:
        if value is None:
            return default
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
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                return (int(value[0]), int(value[1]))
            elif len(value) == 1:
                val = int(value[0])
                return (val, val)
        return default

    # paths - read from config if not provided via CLI
    data_csv = raw_cli.data_csv if raw_cli.data_csv else _get(cfg, "train_io.data_csv", "logs/conversion_log.csv")
    images_root = raw_cli.images_root if raw_cli.images_root else _get(cfg, "train_io.images_root", "dataset/output")
    runs_root = _get(cfg, "train_io.runs_root", "runs")
    export_root = _get(cfg, "training.export_root", "export_models")
    
    # mode - read from config if not provided via CLI
    mode = raw_cli.mode if raw_cli.mode else _get(cfg, "training.mode", "resize")
    if mode not in ("resize", "truncate"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'resize' or 'truncate' (not 'both' - that's handled by main.py)")
    
    # Parse target sizes from config (no CLI override - use config only)
    training_cfg = cfg.get("training", {})
    resize_size = _parse_target_size(training_cfg.get("resize_target_size"), (64, 64))
    truncate_size = _parse_target_size(training_cfg.get("truncate_target_size"), (256, 256))

    # training defaults
    epochs = _get(cfg, "training.epochs", _get(cfg, "training.epoch", args.epochs))
    batch_size = _get(cfg, "training.batch_size", args.batch_size)
    num_workers = _get(cfg, "training.num_workers", args.num_workers)
    prefetch_batches = _get(cfg, "training.prefetch_batches", args.prefetch_batches)
    pin_memory = bool(_get(cfg, "training.pin_memory", args.pin_memory))
    persistent_workers = bool(_get(cfg, "training.persistent_workers", args.persistent_workers))
    kfold = _get(cfg, "training.kfold", args.kfold)
    holdout = _get(cfg, "training.holdout", args.holdout)
    optimizer = _get(cfg, "training.optimizer", args.optimizer)
    scheduler = _get(cfg, "training.scheduler", args.scheduler)
    max_lr = str(_get(cfg, "training.max_lr", args.max_lr))
    weight_decay = float(_get(cfg, "training.weight_decay", args.weight_decay))
    grad_clip = float(_get(cfg, "training.grad_clip", args.grad_clip))
    amp = bool(_get(cfg, "training.amp", args.amp))
    device = _get(cfg, "training.device", args.device) or args.device

    # operating point
    op_type = _get(cfg, "training.metrics.operating_point.type", "fpr_budget")
    op_value = _get(cfg, "training.metrics.operating_point.value", args.fpr_budget)
    fpr_budget = float(op_value) if op_type == "fpr_budget" else args.fpr_budget

    # oversampling (string like "0.05-0.10")
    osr = _get(cfg, "training.oversample_pos_range", f"{args.oversample_pos_min}-{args.oversample_pos_max}")
    try:
        mn, mx = str(osr).split("-")
        oversample_pos_min, oversample_pos_max = float(mn), float(mx)
    except Exception:
        oversample_pos_min, oversample_pos_max = args.oversample_pos_min, args.oversample_pos_max

    # helper choose - CLI only for seed and resume, everything else from config
    def choose(val_cfg, val_cli, val_def):
        return val_cli if (val_cli is not None and val_cli != "") else (val_cfg if val_cfg is not None else val_def)

    # Seed handling - CLI override or auto-increment (handled elsewhere)
    seed = raw_cli.seed if raw_cli.seed is not None else _get(cfg, "training.seed", args.seed)
    
    # Decode cache: read from config, with mode-specific defaults
    # Truncate mode benefits more from caching due to larger, expensive-to-decode PNGs
    decode_cache_mem_mb = _get(cfg, "training.decode_cache_mem_mb", None)
    if decode_cache_mem_mb is None:
        # Mode-specific defaults: truncate gets more cache due to larger files
        if mode == "truncate":
            decode_cache_mem_mb = 1024  # 1GB default for truncate
        else:
            decode_cache_mem_mb = 0  # Disabled for resize by default
    decode_cache_mem_mb = int(decode_cache_mem_mb)

    args = replace(
        args,
        data_csv=data_csv,
        images_root=images_root,
        mode=mode,
        seed=seed if seed is not None else args.seed,
        epochs=epochs if epochs is not None else args.epochs,
        batch_size=batch_size if batch_size is not None else args.batch_size,
        num_workers=num_workers if num_workers is not None else args.num_workers,
        prefetch_batches=prefetch_batches if prefetch_batches is not None else args.prefetch_batches,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        device=device,
        kfold=kfold if kfold is not None else args.kfold,
        holdout=holdout if holdout is not None else args.holdout,
        resume=raw_cli.resume if hasattr(raw_cli, 'resume') else args.resume,
        fpr_budget=fpr_budget if fpr_budget is not None else args.fpr_budget,
        optimizer=optimizer if optimizer is not None else args.optimizer,
        scheduler=scheduler if scheduler is not None else args.scheduler,
        max_lr=max_lr if max_lr is not None else args.max_lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        amp=amp,
        runs_root=runs_root,
        export_root=export_root,
        oversample_pos_min=oversample_pos_min,
        oversample_pos_max=oversample_pos_max,
        decode_cache_mem_mb=decode_cache_mem_mb,
    )

    tr_cfg = _get(cfg, "training", {})
    if isinstance(tr_cfg, dict) and len(tr_cfg) > 0:
        print("[CONFIG] training section loaded from config.yaml:")
        print("  " + json.dumps(tr_cfg, indent=2))
    return args


# ---------- helpers ----------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------- CSV ----------
def read_filtered_rows(csv_path: Path, mode: str) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        need = {"rel_path", "label", "mode", "sha256"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{csv_path} missing columns; need {sorted(need)}")
        for row in r:
            if row["mode"].strip().lower() != mode:
                continue
            rows.append({
                "rel_path": row["rel_path"].strip(),
                "label": 1 if str(row["label"]).strip() in ("1", "malware") else 0,
                "sha256": (row.get("sha256") or "").strip() or f"nog_{len(rows)}"
            })
    return rows


# ---------- Folds ----------
def build_grouped_stratified_folds(rows: List[Dict], k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = check_random_state(seed)
    groups: Dict[str, Dict] = {}
    for i, r in enumerate(rows):
        g = r["sha256"]; y = int(r["label"])
        groups.setdefault(g, {"idxs": [], "label": 0})
        groups[g]["idxs"].append(i)
        groups[g]["label"] = max(groups[g]["label"], y)

    pos = [(g, v) for g, v in groups.items() if v["label"] == 1]
    neg = [(g, v) for g, v in groups.items() if v["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    folds = [{"p": 0, "n": 0, "idxs": []} for _ in range(k)]
    for g, v in pos:
        j = min(range(k), key=lambda fi: folds[fi]["p"]); folds[j]["p"] += 1; folds[j]["idxs"].extend(v["idxs"])
    for g, v in neg:
        j = min(range(k), key=lambda fi: folds[fi]["n"]); folds[j]["n"] += 1; folds[j]["idxs"].extend(v["idxs"])

    N = len(rows)
    all_idx = np.arange(N, dtype=int)
    splits = []
    for fi in range(k):
        val_idx = np.array(sorted(folds[fi]["idxs"]), dtype=int)
        mask = np.ones(N, dtype=bool); mask[val_idx] = False
        train_idx = all_idx[mask]
        splits.append((train_idx, val_idx))
    return splits


def build_holdout_split(rows: List[Dict], holdout_pct: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a stratified holdout split that maintains class balance.
    Groups samples by SHA256 to prevent data leakage.
    """
    assert 0 <= holdout_pct <= 50
    rng = check_random_state(seed)
    groups: Dict[str, Dict] = {}
    for i, r in enumerate(rows):
        g = r["sha256"]; y = int(r["label"])
        groups.setdefault(g, {"idxs": [], "label": 0})
        groups[g]["idxs"].append(i)
        groups[g]["label"] = max(groups[g]["label"], y)
    
    # Separate positive and negative groups
    pos_groups = [(g, v) for g, v in groups.items() if v["label"] == 1]
    neg_groups = [(g, v) for g, v in groups.items() if v["label"] == 0]
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)
    
    # Calculate target sizes for each class to maintain balance
    N = len(rows)
    target_total = int(round(N * (holdout_pct / 100.0)))
    
    # Count samples per class
    n_pos_total = sum(len(v["idxs"]) for _, v in pos_groups)
    n_neg_total = sum(len(v["idxs"]) for _, v in neg_groups)
    
    # Allocate validation samples proportionally to class distribution
    if n_pos_total > 0 and n_neg_total > 0:
        pos_ratio = n_pos_total / N
        target_pos = max(1, int(round(target_total * pos_ratio)))  # At least 1 positive
        target_neg = target_total - target_pos
    elif n_pos_total > 0:
        # Only positive samples
        target_pos = min(target_total, n_pos_total)
        target_neg = 0
    elif n_neg_total > 0:
        # Only negative samples
        target_pos = 0
        target_neg = min(target_total, n_neg_total)
    else:
        # No samples (shouldn't happen)
        target_pos = 0
        target_neg = 0
    
    # Select groups for validation set, maintaining class balance
    val_idxs = []
    
    # Add positive groups
    pos_count = 0
    for g, v in pos_groups:
        if pos_count >= target_pos:
            break
        val_idxs.extend(v["idxs"])
        pos_count += len(v["idxs"])
    
    # Add negative groups
    neg_count = 0
    for g, v in neg_groups:
        if neg_count >= target_neg:
            break
        val_idxs.extend(v["idxs"])
        neg_count += len(v["idxs"])
    
    val_idx = np.array(sorted(val_idxs), dtype=int)
    mask = np.ones(N, dtype=bool)
    mask[val_idx] = False
    train_idx = np.arange(N, dtype=int)[mask]
    
    return [(train_idx, val_idx)]


# ---------- Mild positive oversampling ----------
class StratifiedRatioBatchSampler(BatchSampler):
    def __init__(self, labels: np.ndarray, batch_size: int, pos_min: float, pos_max: float, seed: int):
        assert 0.0 < pos_min <= pos_max < 0.5
        self.labels = labels.astype(int)
        self.batch_size = int(batch_size)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)
        self.rng = random.Random(seed)
        self.pos_pool = [i for i, y in enumerate(self.labels) if y == 1]
        self.neg_pool = [i for i, y in enumerate(self.labels) if y == 0]
        if not self.pos_pool or not self.neg_pool:
            raise ValueError("Both classes are required for stratified batch sampling.")
        self._shuffle()

    def _shuffle(self):
        self.rng.shuffle(self.pos_pool)
        self.rng.shuffle(self.neg_pool)
        self._pi = 0
        self._ni = 0

    def __iter__(self):
        self._shuffle()
        total = len(self.labels); emitted = 0
        while emitted < total:
            target = self.rng.uniform(self.pos_min, self.pos_max)
            k_pos = max(1, min(len(self.pos_pool), int(round(target * self.batch_size))))
            k_neg = self.batch_size - k_pos
            batch = []
            for _ in range(k_pos):
                if self._pi >= len(self.pos_pool):
                    self._pi = 0
                batch.append(self.pos_pool[self._pi]); self._pi += 1
            for _ in range(k_neg):
                if self._ni >= len(self.neg_pool):
                    self._ni = 0
                batch.append(self.neg_pool[self._ni]); self._ni += 1
            self.rng.shuffle(batch)
            emitted += len(batch)
            yield batch

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)


# ---------- Metrics ----------
def compute_metrics(scores: np.ndarray, labels: np.ndarray, fpr_budget: float) -> Dict[str, float]:
    y_true = labels.astype(int)
    y_score = scores.astype(float)
    
    # Count positive and negative samples
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    
    # Handle edge case: no positive samples
    if n_pos == 0:
        # Return NaN for metrics that require positive samples
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "f1_max": float("nan"),
            "thr_f1_max": float("nan"),
            "thr_op": float("nan"),
            "f1_at_op": float("nan"),
            "tpr_at_op": float("nan"),
            "fpr_at_op": float("nan"),
            "support_pos": 0,
            "support_neg": n_neg,
        }
    
    # Handle edge case: no negative samples
    if n_neg == 0:
        # Perfect classifier case
        return {
            "pr_auc": 1.0,
            "roc_auc": 1.0,
            "f1_max": 1.0,
            "thr_f1_max": 0.5,
            "thr_op": 0.5,
            "f1_at_op": 1.0,
            "tpr_at_op": 1.0,
            "fpr_at_op": 0.0,
            "support_pos": n_pos,
            "support_neg": 0,
        }
    
    # Normal case: compute metrics with warnings suppressed
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = average_precision_score(y_true, y_score)
        except Exception:
            pr_auc = float("nan")

        P, R, T = precision_recall_curve(y_true, y_score)
        f1s = (2 * P * R) / np.clip(P + R, 1e-12, None)
        f1_idx = int(np.nanargmax(f1s)) if len(f1s) else 0
        f1_max = float(f1s[f1_idx]) if len(f1s) else float("nan")
        thr_f1_max = float(T[max(0, f1_idx - 1)]) if len(T) > 0 else 0.5

    uniq = np.unique(y_score)
    thr_candidates = np.r_[uniq[::-1], -np.inf]
    best_thr, best_fpr, best_tpr = 1.0, 1.0, 0.0
    for thr in thr_candidates:
        y_pred = (y_score >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fpr = fp / max(1, (fp + tn))
        tpr = tp / max(1, (tp + fn))
        if fpr <= fpr_budget:
            best_thr, best_fpr, best_tpr = float(thr), float(fpr), float(tpr)
            break
    y_pred_op = (y_score >= best_thr).astype(int)
    tp = int(((y_pred_op == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_op == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_op == 0) & (y_true == 1)).sum())
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1_at_op = (2 * precision * recall) / max(1e-12, precision + recall)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1_max": float(f1_max),
        "thr_f1_max": float(thr_f1_max),
        "thr_op": float(best_thr),
        "f1_at_op": float(f1_at_op),
        "tpr_at_op": float(best_tpr),
        "fpr_at_op": float(best_fpr),
        "support_pos": n_pos,
        "support_neg": n_neg,
    }


# ---------- Custom collate function for variable-size images ----------
def collate_variable_size(batch):
    """
    Custom collate function that handles variable-size images by padding to the maximum size.
    This is useful if images have different sizes (though preprocessing should produce fixed sizes).
    
    Args:
        batch: List of (x, y, rel_path) tuples from dataset
    
    Returns:
        (x_padded, y, rel_paths) where x_padded is batched and padded to max size
    """
    # Unpack batch
    images, labels, rel_paths = zip(*batch)
    
    # Convert to tensors if needed
    images = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in images]
    labels = [y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long) for y in labels]
    
    # Find max height and width
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)
    
    # Pad all images to max size (pad right and bottom with zeros)
    padded_images = []
    for img in images:
        # img shape: (C, H, W) or (1, H, W)
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            # Pad: (left, right, top, bottom) for last 2 dims
            padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        else:
            padded = img
        padded_images.append(padded)
    
    # Stack into batch
    x_batch = torch.stack(padded_images, dim=0)  # (B, C, H, W)
    y_batch = torch.stack(labels, dim=0)  # (B,)
    
    return x_batch, y_batch, list(rel_paths)

# ---------- One epoch (robust batch unpack) ----------
def _extract_xy(batch):
    """Return (x, y) from a variety of batch shapes."""
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch has fewer than 2 elements; expected at least (x, y).")
        return batch[0], batch[1]
    if isinstance(batch, dict):
        # common dict keys
        for kx, ky in (("image", "label"), ("x", "y"), ("inputs", "targets")):
            if kx in batch and ky in batch:
                return batch[kx], batch[ky]
    raise ValueError(f"Unsupported batch type for unpacking (got {type(batch)}).")


def one_epoch(model, loader, device, optimizer=None, grad_clip=1.0, desc: str = ""):
    """
    Runs one epoch over 'loader' with an optional optimizer (train if not None, else eval).
    Shows a tqdm progress bar with batches completed.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []

    # nice, compact progress bar per epoch
    pbar = tqdm(loader, desc=desc or ("train" if is_train else "val"),
                unit="batch", dynamic_ncols=True, leave=False)

    for batch in pbar:
        # Custom collate returns (x, y, rel_paths), _extract_xy handles it
        x, y = _extract_xy(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y.float())

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
        all_logits.extend(logits.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

        # optional per-batch postfix (kept light to avoid flicker)
        if n > 0:
            pbar.set_postfix(loss=f"{total_loss / n:.4f}")

    avg = total_loss / max(1, n)
    return avg, np.array(all_logits, np.float32), np.array(all_labels, np.int32)


# ---------- Train run ----------
def train_run(cfg: TrainArgs):
    # Windows: avoid multi-proc DataLoader with dynamic imports
    if platform.system().lower().startswith("win"):
        if cfg.num_workers != 0 or cfg.persistent_workers:
            print("[INFO] Windows detected: forcing num_workers=0 and persistent_workers=False to avoid pickling errors.")
        cfg.num_workers = 0
        cfg.persistent_workers = False

    runs_root = Path(cfg.runs_root) / f"{cfg.mode}_seed{cfg.seed}"
    export_root = Path(cfg.export_root)
    ensure_dir(runs_root)
    ensure_dir(export_root)

    rows = read_filtered_rows(Path(cfg.data_csv), cfg.mode)
    print(f"[PREFLIGHT] CSV summary: {cfg.data_csv}")
    print(f"  total rows: {len(rows)}")
    by_label = {'0': sum(1 for r in rows if r['label'] == 0),
                '1': sum(1 for r in rows if r['label'] == 1)}
    print(f"  by label  : {by_label}")

    splits = build_grouped_stratified_folds(rows, cfg.kfold, cfg.seed) if cfg.kfold > 1 \
        else build_holdout_split(rows, cfg.holdout, cfg.seed)
    print(f"[INFO] Using {len(splits)} fold(s)")

    # Get target sizes from config (already parsed in apply_config_and_cli_defaults)
    # Note: These are used during conversion; dataset loads whatever PNGs exist
    cfg_dict = _load_config_dict()
    training_cfg = cfg_dict.get("training", {})
    from preprocessing.convert import _parse_target_size
    resize_size = _parse_target_size(training_cfg.get("resize_target_size"), (64, 64))
    truncate_size = _parse_target_size(training_cfg.get("truncate_target_size"), (256, 256))
    expected_size = resize_size if cfg.mode == "resize" else truncate_size
    print(f"[INFO] Mode: {cfg.mode}, expected image size: {expected_size[0]}x{expected_size[1]}")

    # Create dataset (loads all rows from CSV)
    # Use mode-specific decode cache (truncate gets 1GB default, resize gets 0)
    # Pass target_size so dataset can resize on-the-fly if PNGs don't match config
    dataset = ByteImageDataset(
        csv_path=str(cfg.data_csv),
        images_root=str(cfg.images_root),
        normalize="01",
        use_disk_cache=True,
        cache_root="cache",
        cache_max_bytes="40GB",
        decode_cache_mem_mb=cfg.decode_cache_mem_mb,
        target_size=tuple(expected_size),  # (width, height) - enables on-the-fly resizing
    )
    
    if cfg.decode_cache_mem_mb > 0:
        print(f"[INFO] Tensor cache enabled: {cfg.decode_cache_mem_mb} MB (decoded PNGs cached in RAM)")
    else:
        print(f"[INFO] Tensor cache disabled (decode_cache_mem_mb=0)")
    
    # Diagnostic: Check actual image dimensions vs expected
    if len(rows) > 0:
        from PIL import Image
        sample_size = min(20, len(rows))  # Sample first 20 files
        actual_sizes = []
        file_sizes = []
        images_root_path = Path(cfg.images_root)
        for i in range(sample_size):
            row = rows[i]
            img_path = images_root_path / row["rel_path"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        actual_sizes.append(img.size)  # (width, height)
                    file_sizes.append(img_path.stat().st_size)
                except Exception:
                    pass
        
        if actual_sizes:
            # Get most common size
            from collections import Counter
            size_counts = Counter(actual_sizes)
            most_common_size = size_counts.most_common(1)[0][0]
            most_common_count = size_counts.most_common(1)[0][1]
            
            print(f"[INFO] Actual PNG dimensions (sampled {len(actual_sizes)} files):")
            print(f"  Most common: {most_common_size[0]}x{most_common_size[1]} ({most_common_count}/{len(actual_sizes)} files)")
            print(f"  Expected from config: {expected_size[0]}x{expected_size[1]}")
            
            if most_common_size != tuple(expected_size):
                print(f"[WARN] Image dimensions don't match config!")
                print(f"[WARN] PNGs on disk are {most_common_size[0]}x{most_common_size[1]}, but config expects {expected_size[0]}x{expected_size[1]}")
                print(f"[WARN] Dataset will resize images on-the-fly to match config (this may slow down training)")
                print(f"[WARN] To fix: re-run 'python main.py convert' with the new target_size to regenerate PNGs")
            
            avg_file_size_kb = (sum(file_sizes) / len(file_sizes)) / 1024
            print(f"[INFO] Average PNG file size: {avg_file_size_kb:.1f} KB")
            if cfg.mode == "truncate" and avg_file_size_kb > 30:
                print(f"[INFO] Truncate files are large ({avg_file_size_kb:.1f} KB avg) due to high-entropy content.")
                print(f"[INFO] Consider: increasing decode_cache_mem_mb for faster repeated access, or using larger cache_max_bytes.")
    
    # Create mapping from filtered row indices to dataset indices
    # The dataset loads ALL rows, but rows is filtered by mode
    # We need to map filtered_row_idx -> dataset_idx
    print(f"[INFO] Building index mapping: filtered rows ({len(rows)}) -> dataset items ({len(dataset)})")
    filtered_to_dataset_idx = []
    dataset_to_filtered_idx = {}  # Reverse mapping for validation
    
    # Build mapping by matching rel_path and sha256
    for filtered_idx, row in enumerate(rows):
        row_rel = row["rel_path"]
        row_sha = row["sha256"]
        row_label = row["label"]
        
        # Find matching item in dataset
        for dataset_idx, (item_rel, item_label, item_sha) in enumerate(dataset.items):
            if item_rel == row_rel and item_sha == row_sha and item_label == row_label:
                filtered_to_dataset_idx.append(dataset_idx)
                dataset_to_filtered_idx[dataset_idx] = filtered_idx
                break
        else:
            # If not found, this is a problem - filtered row doesn't exist in dataset
            raise ValueError(
                f"Filtered row {filtered_idx} (rel_path={row_rel}, sha256={row_sha}) "
                f"not found in dataset. This indicates a mismatch between CSV filtering and dataset loading."
            )
    
    if len(filtered_to_dataset_idx) != len(rows):
        raise ValueError(
            f"Index mapping incomplete: {len(filtered_to_dataset_idx)}/{len(rows)} rows mapped. "
            f"Dataset may be missing rows or have duplicates."
        )
    
    print(f"[INFO] Index mapping complete: {len(filtered_to_dataset_idx)} rows mapped")

    device = torch.device("cuda" if (cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"[INFO] Using device: {device}")

    best_overall = {"pr_auc": float("-inf"), "fold": -1, "ckpt_path": ""}

    # Optional resume discovery for this (mode,seed) run
    resume_state = None
    runstate_path = runs_root / "runstate.json"
    if cfg.resume and runstate_path.exists():
        try:
            rs = json.loads(runstate_path.read_text(encoding="utf-8"))
            cur_fold = int(rs.get("current_fold", 1))
            next_epoch = int(rs.get("next_epoch", 1))
            intr_path = runs_root / f"fold{cur_fold}_interrupt.pt"
            if intr_path.exists():
                print(f"[RESUME] Found interrupt checkpoint: {intr_path} (fold={cur_fold}, next_epoch={next_epoch})")
                state = torch.load(intr_path, map_location="cpu")
                resume_state = {"fold": cur_fold, "next_epoch": next_epoch, "state": state}
            else:
                print("[RESUME] runstate.json present but interrupt checkpoint not found; starting fresh.")
        except Exception as e:
            print(f"[RESUME] Failed to parse runstate.json: {e}; starting fresh.")

    for fi, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fi}/{len(splits)} ===")
        print(f"[DEBUG] fold sizes -> train: {len(train_idx)} | val: {len(val_idx)}")
        
        # Map filtered row indices to dataset indices
        train_dataset_idx = np.array([filtered_to_dataset_idx[i] for i in train_idx], dtype=int)
        val_dataset_idx = np.array([filtered_to_dataset_idx[i] for i in val_idx], dtype=int)
        
        # Show class distribution in train/val splits (from filtered rows)
        train_labels_split = np.array([rows[i]['label'] for i in train_idx], dtype=int)
        val_labels_split = np.array([rows[i]['label'] for i in val_idx], dtype=int)
        train_pos = int(train_labels_split.sum())
        train_neg = len(train_labels_split) - train_pos
        val_pos = int(val_labels_split.sum())
        val_neg = len(val_labels_split) - val_pos
        print(f"[DEBUG] class distribution -> train: pos={train_pos} neg={train_neg} | val: pos={val_pos} neg={val_neg}")
        
        if val_pos == 0:
            print(f"[WARN] Validation set has no positive samples! This will cause metric computation issues.")
            print(f"[WARN] Consider using kfold > 1 for stratified cross-validation, or check your dataset balance.")

        # Skip completed folds when resuming
        if resume_state and fi < resume_state['fold']:
            print(f"[RESUME] Skipping fold {fi} (completed previously).")
            continue

        # Create subsets using dataset indices
        ds_train = Subset(dataset, train_dataset_idx)
        ds_val   = Subset(dataset, val_dataset_idx)

        # Verify labels match between filtered rows and dataset
        # Sample a few indices to verify
        sample_indices = np.concatenate([train_dataset_idx[:min(5, len(train_dataset_idx))], 
                                        val_dataset_idx[:min(5, len(val_dataset_idx))]])
        for dataset_idx in sample_indices:
            filtered_idx = dataset_to_filtered_idx.get(dataset_idx)
            if filtered_idx is not None:
                dataset_label = dataset.items[dataset_idx][1]  # label is at index 1
                row_label = rows[filtered_idx]['label']
                if dataset_label != row_label:
                    print(f"[WARN] Label mismatch at dataset_idx={dataset_idx}, filtered_idx={filtered_idx}: "
                          f"dataset={dataset_label}, row={row_label}")

        # Reuse train_labels_split computed above
        train_labels = train_labels_split
        batch_sampler = StratifiedRatioBatchSampler(
            train_labels, cfg.batch_size,
            cfg.oversample_pos_min, cfg.oversample_pos_max,
            cfg.seed + fi
        )

        train_loader = DataLoader(
            ds_train,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            collate_fn=collate_variable_size,  # Handle variable sizes gracefully
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=collate_variable_size,  # Handle variable sizes gracefully
        )

        model = MalNetFocusAug(input_size=expected_size[0], attention=True).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg.weight_decay) \
            if cfg.optimizer == "adamw" else optim.Adam(model.parameters(), lr=1e-3)

        scheduler = None
        if cfg.scheduler == "onecycle":
            total_steps = max(1, cfg.epochs * len(train_loader))
            max_lr = 1e-3 if str(cfg.max_lr) == "auto" else float(cfg.max_lr)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

        # Safe-interrupt
        interrupted = {"flag": False}

        def _handle(sig, frame):
            print("\n[INTERRUPT] Caught signal; saving safe checkpoint and exiting...")
            interrupted["flag"] = True

        old_handler = signal.signal(signal.SIGINT, _handle)

        ckpt_dir = runs_root
        ensure_dir(ckpt_dir)
        fold_best = {"pr_auc": float("-inf"), "epoch": -1, "path": ""}

        try:
            # Determine starting epoch if resuming this fold
            start_epoch = 1
            if resume_state and fi == resume_state['fold']:
                start_epoch = max(1, int(resume_state['next_epoch']))
                state = resume_state['state']
                # best-effort state restore
                try:
                    model.load_state_dict(state.get('model', {}), strict=False)
                except Exception:
                    pass
                if state.get('optimizer'):
                    try:
                        optimizer.load_state_dict(state['optimizer'])
                    except Exception:
                        pass
                if scheduler and state.get('scheduler'):
                    try:
                        scheduler.load_state_dict(state['scheduler'])
                    except Exception:
                        pass
                if start_epoch > cfg.epochs:
                    print(f"[RESUME] Fold {fi} already completed in prior run (next_epoch={start_epoch} > epochs={cfg.epochs}); skipping.")
                    continue
                print(f"[RESUME] Continuing fold {fi} from epoch {start_epoch}")

            for epoch in range(start_epoch, cfg.epochs + 1):
                tr_loss, _, _ = one_epoch(model, train_loader, device, optimizer=optimizer, grad_clip=cfg.grad_clip, desc=f"train f{fi} e{epoch}")
                if scheduler is not None:
                    scheduler.step()

                val_loss, val_logits, val_labels = one_epoch(model, val_loader, device,optimizer=None, desc=f"val   f{fi} e{epoch}")

                val_scores = sigmoid_np(val_logits)
                
                # Debug: Check actual labels from dataset on first epoch
                if epoch == 1:
                    actual_pos = int(val_labels.sum())
                    actual_neg = len(val_labels) - actual_pos
                    expected_pos = int(val_labels_split.sum())
                    expected_neg = len(val_labels_split) - expected_pos
                    if actual_pos != expected_pos or actual_neg != expected_neg:
                        print(f"[ERROR] Label mismatch detected!")
                        print(f"[ERROR] Expected from split: pos={expected_pos}, neg={expected_neg}")
                        print(f"[ERROR] Actual from dataset: pos={actual_pos}, neg={actual_neg}")
                        print(f"[ERROR] This indicates the dataset indices don't match the filtered row indices.")
                        print(f"[ERROR] Check the index mapping logic.")
                
                metrics = compute_metrics(val_scores, val_labels, cfg.fpr_budget)
                acc05 = float(((val_scores >= 0.5).astype(int) == val_labels).mean()) if len(val_labels) > 0 else float("nan")

                # Format metrics with NaN handling
                def fmt_metric(v):
                    if np.isnan(v):
                        return "nan"
                    return f"{v:.3f}"
                
                # Warn if no positive samples in validation
                if metrics["support_pos"] == 0:
                    print(f"[WARN] Validation set has no positive samples (pos={metrics['support_pos']}, neg={metrics['support_neg']})")
                    print(f"[WARN] Metrics requiring positive samples will be NaN. Consider adjusting your split or dataset.")

                print(
                    f"Epoch {epoch}/{cfg.epochs}: "
                    f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} acc@0.5={acc05:.3f} | "
                    f"val: pr_auc={fmt_metric(metrics['pr_auc'])} roc_auc={fmt_metric(metrics['roc_auc'])} f1_max={fmt_metric(metrics['f1_max'])} | "
                    f"op: thr={fmt_metric(metrics['thr_op'])} f1={fmt_metric(metrics['f1_at_op'])} tpr={fmt_metric(metrics['tpr_at_op'])} fpr={fmt_metric(metrics['fpr_at_op'])}"
                )

                # Update best checkpoint if pr_auc is valid and better
                pr_auc_val = metrics["pr_auc"]
                if not np.isnan(pr_auc_val) and pr_auc_val > fold_best["pr_auc"]:
                    fold_best.update({"pr_auc": pr_auc_val, "epoch": epoch})
                    ckpt_path = ckpt_dir / f"fold{fi}_best.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "epoch": epoch,
                        "metrics": metrics,
                        "fpr_budget": cfg.fpr_budget,
                        "seed": cfg.seed,
                        "mode": cfg.mode,
                        "fold": fi,
                    }, ckpt_path)
                    fold_best["path"] = str(ckpt_path)
                    print(f"[CHECKPOINT] Saved fold {fi} best @ epoch {epoch} pr_auc={pr_auc_val:.3f} -> {ckpt_path}")

                if interrupted["flag"]:
                    ckpt_path = ckpt_dir / f"fold{fi}_interrupt.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "epoch": epoch,
                        "seed": cfg.seed,
                        "mode": cfg.mode,
                        "fold": fi,
                    }, ckpt_path)
                    with (ckpt_dir / "runstate.json").open("w", encoding="utf-8") as f:
                        json.dump({"current_fold": fi, "next_epoch": epoch, "interrupted_at": utc_now_iso()}, f, indent=2)
                    print(f"[INTERRUPT] Checkpoint written: {ckpt_path}. Runstate saved.")
                    print("[INTERRUPT] To resume: run the same command, or simply `python main.py train` to auto-resume latest.")
                    return
        finally:
            signal.signal(signal.SIGINT, old_handler)

        pr_auc_fold = fold_best['pr_auc']
        if np.isnan(pr_auc_fold):
            print(f"[FOLD {fi}] best pr_auc=nan @ epoch {fold_best['epoch']} (no valid metrics)")
        else:
            print(f"[FOLD {fi}] best pr_auc={pr_auc_fold:.3f} @ epoch {fold_best['epoch']}")
        if fold_best["path"]:
            print(f"[CHECKPOINT] Saved fold {fi} best state to: {fold_best['path']}")

        # Update best overall only if pr_auc is valid and better
        if not np.isnan(pr_auc_fold) and pr_auc_fold > best_overall["pr_auc"]:
            best_overall.update({"pr_auc": pr_auc_fold, "fold": fi, "ckpt_path": fold_best["path"]})

    # ---------- Export best fold ----------
    if not best_overall["ckpt_path"]:
        print("[WARN] No best checkpoint found; skipping export.")
        return

    pr_auc_best = best_overall['pr_auc']
    if np.isnan(pr_auc_best) or pr_auc_best == float("-inf"):
        print(f"[SELECT] Best fold: {best_overall['fold']} pr_auc=nan (no valid metrics found)")
    else:
        print(f"[SELECT] Best fold: {best_overall['fold']} pr_auc={pr_auc_best:.3f}")
    state = torch.load(best_overall["ckpt_path"], map_location="cpu")
    model = MalNetFocusAug(input_size=expected_size[0], attention=True).eval()
    model.load_state_dict(state["model"], strict=False)

    export_dir = Path(cfg.export_root)
    ensure_dir(export_dir)
    existing = sorted(export_dir.glob(f"cnn_{cfg.mode}_{cfg.seed}_*.ts.pt"))
    iter_id = 1
    if existing:
        try:
            iter_id = max(int(p.stem.split("_")[-1]) for p in existing) + 1
        except Exception:
            iter_id = len(existing) + 1
    out_path = export_dir / f"cnn_{cfg.mode}_{cfg.seed}_{iter_id}.ts.pt"

    example = torch.randn(1, 1, 256, 256)
    with torch.inference_mode():
        scripted = torch.jit.trace(model, example)
    scripted.save(str(out_path))

    print(f"[EXPORT] Saved TorchScript model: {out_path.as_posix()}")
    print(f"[EXPORT] Load with: model = torch.jit.load('{out_path.as_posix()}')")

    meta = {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "fold": int(best_overall["fold"]),
        "metrics": state.get("metrics", {}),
        "fpr_budget": cfg.fpr_budget,
        "exported_at": utc_now_iso(),
        "export_path": out_path.as_posix(),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[EXPORT] Saved meta: {meta_path.as_posix()}")


# ---------- Main ----------
def main():
    args_min, raw_cli = parse_cli()
    args = apply_config_and_cli_defaults(args_min, raw_cli)
    print("[CONFIG] Effective training settings")
    print("  " + "\n  ".join(f"{k}: {v}" for k, v in asdict(args).items()))
    train_run(args)


if __name__ == "__main__":
    main()
