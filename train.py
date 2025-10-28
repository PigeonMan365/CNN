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


def parse_cli() -> TrainArgs:
    p = argparse.ArgumentParser(description="Train MalNet-FocusAug")
    p.add_argument("--data-csv", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--mode", required=True, choices=["compress", "truncate"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-batches", type=int, default=None)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--kfold", type=int, default=None)
    p.add_argument("--holdout", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fpr-budget", type=float, default=None)
    p.add_argument("--oversample-pos-range", type=str, default=None)
    p.add_argument("--optimizer", default=None, choices=["adam", "adamw"])
    p.add_argument("--scheduler", default=None, choices=["none", "onecycle"])
    p.add_argument("--max-lr", default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--runs-root", default=None)
    p.add_argument("--export-root", default=None)
    a = p.parse_args()

    args = TrainArgs(
        data_csv=a.data_csv,
        images_root=a.images_root,
        mode=a.mode,
    )
    return args, a


def apply_config_and_cli_defaults(args: TrainArgs, raw_cli) -> TrainArgs:
    cfg = _load_config_dict()

    # paths
    data_csv = _get(cfg, "train_io.data_csv", args.data_csv) or args.data_csv
    images_root = _get(cfg, "train_io.images_root", args.images_root) or args.images_root
    runs_root = _get(cfg, "train_io.runs_root", args.runs_root) or args.runs_root
    export_root = _get(cfg, "training.export_root", args.export_root) or args.export_root

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

    # helper choose
    def choose(val_cfg, val_cli, val_def):
        return val_cli if (val_cli is not None and val_cli != "") else (val_cfg if val_cfg is not None else val_def)

    args = replace(
        args,
        data_csv=choose(data_csv, raw_cli.data_csv, args.data_csv),
        images_root=choose(images_root, raw_cli.images_root, args.images_root),
        seed=choose(_get(cfg, "training.seed", args.seed), raw_cli.seed, args.seed),
        epochs=choose(epochs, raw_cli.epochs, args.epochs),
        batch_size=choose(batch_size, raw_cli.batch_size, args.batch_size),
        num_workers=choose(num_workers, raw_cli.num_workers, args.num_workers),
        prefetch_batches=choose(prefetch_batches, raw_cli.prefetch_batches, args.prefetch_batches),
        pin_memory=True if raw_cli.pin_memory else pin_memory,
        persistent_workers=choose(persistent_workers, raw_cli.persistent_workers, args.persistent_workers),
        device=choose(device, raw_cli.device, args.device),
        kfold=choose(kfold, raw_cli.kfold, args.kfold),
        holdout=choose(holdout, raw_cli.holdout, args.holdout),
        resume=True if raw_cli.resume else args.resume,
        fpr_budget=choose(fpr_budget, raw_cli.fpr_budget, args.fpr_budget),
        optimizer=choose(optimizer, raw_cli.optimizer, args.optimizer),
        scheduler=choose(scheduler, raw_cli.scheduler, args.scheduler),
        max_lr=choose(max_lr, raw_cli.max_lr, args.max_lr),
        weight_decay=choose(weight_decay, None, args.weight_decay),
        grad_clip=choose(grad_clip, raw_cli.grad_clip, args.grad_clip),
        amp=True if raw_cli.amp else amp,
        runs_root=choose(runs_root, raw_cli.runs_root, args.runs_root),
        export_root=choose(export_root, raw_cli.export_root, args.export_root),
        oversample_pos_min=oversample_pos_min,
        oversample_pos_max=oversample_pos_max,
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
    assert 0 <= holdout_pct <= 50
    rng = check_random_state(seed)
    groups: Dict[str, Dict] = {}
    for i, r in enumerate(rows):
        g = r["sha256"]; y = int(r["label"])
        groups.setdefault(g, {"idxs": [], "label": 0})
        groups[g]["idxs"].append(i)
        groups[g]["label"] = max(groups[g]["label"], y)
    glist = list(groups.items()); rng.shuffle(glist)
    N = len(rows); target = int(round(N * (holdout_pct / 100.0)))
    val_idxs = []
    for _, v in glist:
        if len(val_idxs) >= target:
            break
        val_idxs.extend(v["idxs"])
    val_idx = np.array(sorted(val_idxs), dtype=int)
    mask = np.ones(N, dtype=bool); mask[val_idx] = False
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
        "support_pos": int(y_true.sum()),
        "support_neg": int((1 - y_true).sum()),
    }


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

    dataset = ByteImageDataset(
        csv_path=str(cfg.data_csv),
        images_root=str(cfg.images_root),
        normalize="01",
        use_disk_cache=True,
        cache_root="cache",
        cache_max_bytes="40GB",
        decode_cache_mem_mb=0,
    )

    device = torch.device("cuda" if (cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"[INFO] Using device: {device}")

    best_overall = {"pr_auc": -1.0, "fold": -1, "ckpt_path": ""}

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

        # Skip completed folds when resuming
        if resume_state and fi < resume_state['fold']:
            print(f"[RESUME] Skipping fold {fi} (completed previously).")
            continue

        ds_train = Subset(dataset, train_idx)
        ds_val   = Subset(dataset, val_idx)

        train_labels = np.array([rows[i]['label'] for i in train_idx], dtype=int)
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
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        model = MalNetFocusAug(attention=True).to(device)
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
        fold_best = {"pr_auc": -1.0, "epoch": -1, "path": ""}

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
                metrics = compute_metrics(val_scores, val_labels, cfg.fpr_budget)
                acc05 = float(((val_scores >= 0.5).astype(int) == val_labels).mean()) if len(val_labels) > 0 else float("nan")

                print(
                    f"Epoch {epoch}/{cfg.epochs}: "
                    f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} acc@0.5={acc05:.3f} | "
                    f"val: pr_auc={metrics['pr_auc']:.3f} roc_auc={metrics['roc_auc']:.3f} f1_max={metrics['f1_max']:.3f} | "
                    f"op: thr={metrics['thr_op']:.3f} f1={metrics['f1_at_op']:.3f} tpr={metrics['tpr_at_op']:.3f} fpr={metrics['fpr_at_op']:.4f}"
                )

                if metrics["pr_auc"] > fold_best["pr_auc"]:
                    fold_best.update({"pr_auc": metrics["pr_auc"], "epoch": epoch})
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
                    print(f"[CHECKPOINT] Saved fold {fi} best @ epoch {epoch} pr_auc={metrics['pr_auc']:.3f} -> {ckpt_path}")

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

        print(f"[FOLD {fi}] best pr_auc={fold_best['pr_auc']:.3f} @ epoch {fold_best['epoch']}")
        if fold_best["path"]:
            print(f"[CHECKPOINT] Saved fold {fi} best state to: {fold_best['path']}")

        if fold_best["pr_auc"] > best_overall["pr_auc"]:
            best_overall.update({"pr_auc": fold_best["pr_auc"], "fold": fi, "ckpt_path": fold_best["path"]})

    # ---------- Export best fold ----------
    if not best_overall["ckpt_path"]:
        print("[WARN] No best checkpoint found; skipping export.")
        return

    print(f"[SELECT] Best fold: {best_overall['fold']} pr_auc={best_overall['pr_auc']:.3f}")
    state = torch.load(best_overall["ckpt_path"], map_location="cpu")
    model = MalNetFocusAug(attention=True).eval()
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
