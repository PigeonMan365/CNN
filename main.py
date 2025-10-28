#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — project entrypoint (cross-platform)

Subcommands:
  verify         -> environment & data audit
  convert        -> convert binaries to PNGs (compress/truncate) if a converter is present
  train          -> single-job training (resume latest if no args; config-driven; seed ledger)
  orchestrate    -> plan/status/resume multi-run experiments (seed rounds)
  reset          -> clean caches/logs/tmp/runs/export_models and (re)create folder skeleton
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
SEED_LEDGER = HERE / "runs" / "seed_state.json"


# --------------------------- config helpers ---------------------------

def load_config():
    try:
        from utils.paths import load_config as _load
        cfg = _load()
        if hasattr(cfg, 'get'):
            return cfg
        if hasattr(cfg, '__dict__'):
            return dict(cfg.__dict__)
    except Exception:
        pass
    cfg_path = HERE / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def cfg_get(cfg, path, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ----------------------------- seed ledger ----------------------------

def _read_last_seed(default_seed: int) -> int:
    SEED_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    try:
        if SEED_LEDGER.exists():
            data = json.loads(SEED_LEDGER.read_text(encoding="utf-8"))
            return int(data.get("last_seed", default_seed - 1))
    except Exception:
        pass
    return default_seed - 1


def _write_last_seed(last_seed: int) -> None:
    SEED_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    SEED_LEDGER.write_text(json.dumps({"last_seed": int(last_seed)}, indent=2), encoding="utf-8")


def _next_seed_from_ledger(cfg) -> int:
    base = int(cfg_get(cfg, "training.seed", 0) or 0)
    last = _read_last_seed(base)
    return max(base, last + 1)


# ------------------------------ verify -------------------------------

def cmd_verify(_args):
    py = sys.executable
    script = HERE / "verify_setup.py"
    if not script.exists():
        print("[verify] verify_setup.py not found.")
        return
    sys.exit(subprocess.run([py, str(script)], check=False).returncode)


# ------------------------------ convert ------------------------------

def cmd_convert(_args):
    """
    Try common converter entrypoints in order:
      - converter.py
      - dual_convert.py
      - python -m preprocess.convert
    """
    tried = []
    py = sys.executable

    p = HERE / "converter.py"
    if p.exists():
        tried.append(p.name)
        print(f"[convert] running {p.name} ...")
        rc = subprocess.run([py, str(p)], check=False).returncode
        if rc == 0:
            return
        print(f"[convert] {p.name} exited with {rc}.")

    p = HERE / "dual_convert.py"
    if p.exists():
        tried.append(p.name)
        print(f"[convert] running {p.name} ...")
        rc = subprocess.run([py, str(p)], check=False).returncode
        if rc == 0:
            return
        print(f"[convert] {p.name} exited with {rc}.")

    try:
        import preprocess  # noqa: F401
        tried.append("python -m preprocess.convert")
        print("[convert] running module preprocess.convert ...")
        rc = subprocess.run([py, "-m", "preprocess.convert"], check=False).returncode
        if rc == 0:
            return
        print("[convert] preprocess.convert exited with", rc)
    except Exception:
        pass

    if not tried:
        print("[convert] No converter entrypoint found (converter.py / dual_convert.py / preprocess.convert).")
    else:
        print("[convert] Tried:", ", ".join(tried))


# ------------------------------- train -------------------------------

def _build_train_cmd(cfg, mode, seed=None, extra=None):
    data_csv = cfg_get(cfg, "train_io.data_csv", "logs/conversion_log.csv")
    images_root = cfg_get(cfg, "train_io.images_root", "dataset/output")
    cmd = [sys.executable, str(HERE / "train.py"),
           "--data-csv", str(data_csv),
           "--images-root", str(images_root),
           "--mode", mode]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if extra:
        cmd += extra
    return cmd


def _find_last_interrupted(runs_root: Path) -> Optional[tuple]:
    if not runs_root.exists():
        return None
    candidates = []
    for run_dir in runs_root.glob("*_seed*"):
        rs = run_dir / "runstate.json"
        if rs.exists():
            candidates.append((rs.stat().st_mtime, run_dir))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_dir = candidates[0][1]
    m = re.match(r"^(compress|truncate)_seed(\d+)$", latest_dir.name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), latest_dir


def cmd_train(args):
    cfg = load_config()
    runs_root = Path(cfg_get(cfg, "train_io.runs_root", "runs"))

    # No explicit --mode: try to resume; else start new using defaults + ledger
    if args.mode is None:
        last = _find_last_interrupted(runs_root)
        if last:
            mode, seed, run_dir = last
            print(f"[train] Resuming most recent interrupted run: {run_dir.name}")
            cmd = _build_train_cmd(cfg, mode, seed, ["--resume"])
            print("> " + " ".join(cmd))
            sys.exit(subprocess.run(cmd, check=False).returncode)

        # Start new from config defaults and seed ledger
        default_mode = cfg_get(cfg, "training.default_mode", None) or "both"
        s = _next_seed_from_ledger(cfg)
        if default_mode == "both":
            plan = [("compress", s), ("truncate", s)]
        else:
            plan = [(default_mode, s)]
        for mode, seed in plan:
            cmd = _build_train_cmd(cfg, mode, seed, [])
            print("> " + " ".join(cmd))
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"[train] {mode} failed with exit code {rc}.")
                sys.exit(rc)
        _write_last_seed(s)  # bump only after successful completion of the round
        sys.exit(0)

    # Explicit mode provided: pass through (honor CLI seed if given)
    mode = args.mode
    seed = args.seed
    extra = []
    if args.resume:
        extra.append("--resume")
    if args.kfold is not None:
        extra += ["--kfold", str(args.kfold)]
    if args.holdout is not None:
        extra += ["--holdout", str(args.holdout)]
    if args.epochs is not None:
        extra += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra += ["--batch-size", str(args.batch_size)]
    if args.num_workers is not None:
        extra += ["--num-workers", str(args.num_workers)]
    if args.device is not None:
        extra += ["--device", args.device]

    # If seed not specified, consume from ledger
    if seed is None:
        s = _next_seed_from_ledger(cfg)
        if mode == "both":
            plan = [("compress", s), ("truncate", s)]
        else:
            plan = [(mode, s)]
        for m, ss in plan:
            cmd = _build_train_cmd(cfg, m, ss, extra)
            print("> " + " ".join(cmd))
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"[train] {m} failed with exit code {rc}.")
                sys.exit(rc)
        _write_last_seed(s)
        sys.exit(0)
    else:
        # Manual seed: do not touch ledger
        if mode == "both":
            plan = [("compress", seed), ("truncate", seed)]
        else:
            plan = [(mode, seed)]
        for m, ss in plan:
            cmd = _build_train_cmd(cfg, m, ss, extra)
            print("> " + " ".join(cmd))
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"[train] {m} failed with exit code {rc}.")
                sys.exit(rc)
        sys.exit(0)


# ---------------------------- orchestrate ----------------------------

def cmd_orchestrate(args):
    py = sys.executable
    script = HERE / "orchestrate.py"
    if not script.exists():
        print("[orchestrate] orchestrate.py not found.")
        return

    if args.subcmd is None:
        if args.runs is None:
            print("Usage:")
            print("  python main.py orchestrate --runs N")
            print("  python main.py orchestrate plan --runs N")
            print("  python main.py orchestrate status")
            print("  python main.py orchestrate resume")
            sys.exit(1)
        cmd = [py, str(script), "plan", "--runs", str(args.runs)]
    elif args.subcmd == "plan":
        cmd = [py, str(script), "plan", "--runs", str(args.runs)]
    elif args.subcmd == "status":
        cmd = [py, str(script), "status"]
    elif args.subcmd == "resume":
        cmd = [py, str(script), "resume"]
    else:
        print(f"[orchestrate] unknown subcmd: {args.subcmd}")
        sys.exit(2)

    print("> " + " ".join(cmd))
    sys.exit(subprocess.run(cmd, check=False).returncode)


# ------------------------------ reset ------------------------------

import io
import os
import hashlib
import shutil
from pathlib import Path

def _safe_rmtree(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _hash_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _rebuild_conversion_log_from_dataset(images_root: Path, log_csv: Path) -> int:
    """
    Walk dataset/output and rebuild logs/conversion_log.csv with:
      rel_path,label,mode,sha256
    where:
      mode   = {compress, truncate} (taken from 1st-level subdir)
      label  = 1 if 'malware' subdir, 0 if 'benign'
      rel_path = POSIX path relative to images_root (e.g., 'compress/malware/foo.png')
      sha256 = SHA-256 of the file contents (PNG/JPG/etc.)
    Returns number of rows written (excluding header).
    """
    _ensure_dir(log_csv.parent)
    rows = []
    if not images_root.exists():
        _ensure_dir(images_root)  # create skeleton if totally missing
    # Accept common image extensions; add more if you store something else.
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}

    for mode in ("compress", "truncate"):
        for cls, label in (("benign", 0), ("malware", 1)):
            base = images_root / mode / cls
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    rel_path = p.relative_to(images_root).as_posix()
                    sha256 = _hash_file_sha256(p)
                    rows.append(f"{rel_path},{label},{mode},{sha256}")

    # Always (re)write header + rows
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("rel_path,label,mode,sha256\n")
        for r in rows:
            f.write(r + "\n")
    print(f"[reset] rebuilt {log_csv} with {len(rows)} rows from {images_root}")
    return len(rows)

def _create_project_skeleton_preserve_dataset_and_rebuild_csv(cfg):
    """
    Reset the project WITHOUT touching dataset contents.
    - Clears: runs/, export_models/, cache/, logs/, tmp/
    - Rebuilds: logs/conversion_log.csv by scanning dataset/output
    - Preserves: dataset/ (and any subfolders/files)
    - Recreates minimal folder skeletons if missing.
    - Resets seed ledger.
    """
    images_root = Path(cfg_get(cfg, "train_io.images_root", "dataset/output"))
    cache_root  = Path(cfg_get(cfg, "paths.cache_root", "cache"))
    runs_root   = Path(cfg_get(cfg, "train_io.runs_root", "runs"))
    logs_root   = HERE / "logs"
    tmp_root    = HERE / "tmp"
    export_root = Path(cfg_get(cfg, "training.export_root", "export_models"))

    # wipe selected dirs (NOT dataset)
    for d in [runs_root, export_root, cache_root, logs_root, tmp_root]:
        _safe_rmtree(d)
        _ensure_dir(d)

    # ensure dataset skeleton exists, but do NOT delete anything inside
    for mode in ("compress", "truncate"):
        for cls in ("benign", "malware"):
            _ensure_dir(images_root / mode / cls)

    # reset seed ledger
    if SEED_LEDGER.exists():
        SEED_LEDGER.unlink(missing_ok=True)

    # rebuild conversion log from dataset
    log_csv = logs_root / "conversion_log.csv"
    _rebuild_conversion_log_from_dataset(images_root, log_csv)

    print("[reset] project reset complete (dataset preserved).")
    print(f"        images_root (preserved): {images_root}")
    print(f"        cache_root  (cleared)  : {cache_root}")
    print(f"        runs_root   (cleared)  : {runs_root}")
    print(f"        export_root (cleared)  : {export_root}")
    print(f"        logs_root   (rebuilt)  : {logs_root}")
    print(f"        tmp_root    (cleared)  : {tmp_root}")

def cmd_reset(_args):
    cfg = load_config()
    _create_project_skeleton_preserve_dataset_and_rebuild_csv(cfg)
    print("[reset] done.")

# -------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser(prog="main", description="MalNet-FocusAug entrypoint")
    sub = ap.add_subparsers(dest="cmd")

    ap_verify = sub.add_parser("verify", help="environment & data audit")
    ap_verify.set_defaults(func=cmd_verify)

    ap_convert = sub.add_parser("convert", help="convert binaries to PNGs (if a converter exists)")
    ap_convert.set_defaults(func=cmd_convert)

    ap_train = sub.add_parser("train", help="train a model (resume latest if no args)")
    ap_train.add_argument("--mode", choices=["compress", "truncate", "both"], required=False,
                          help="If omitted: resume last interrupted or use config default/ledger.")
    ap_train.add_argument("--seed", type=int, default=None)
    ap_train.add_argument("--resume", action="store_true", help="resume same (mode,seed) if interrupted")
    ap_train.add_argument("--kfold", type=int, default=None)
    ap_train.add_argument("--holdout", type=int, default=None)
    ap_train.add_argument("--epochs", type=int, default=None)
    ap_train.add_argument("--batch-size", type=int, default=None)
    ap_train.add_argument("--num-workers", type=int, default=None)
    ap_train.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    ap_train.set_defaults(func=cmd_train)

    ap_orch = sub.add_parser("orchestrate", help="multi-run planner/executor")
    ap_orch.add_argument("subcmd", nargs="?", choices=["plan", "status", "resume"])
    ap_orch.add_argument("--runs", type=int, help="number of rounds (each round = compress@seed, truncate@seed)")
    ap_orch.set_defaults(func=cmd_orchestrate)

    ap_reset = sub.add_parser("reset", help="wipe caches/logs/tmp/runs/export_models and re-create folders")
    ap_reset.set_defaults(func=cmd_reset)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
