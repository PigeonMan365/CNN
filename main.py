#!/usr/bin/env python3
"""
main.py — project entrypoint

Subcommands:
  verify            -> environment & data audit with knob recommendations
  convert           -> convert binaries to PNGs (both modes) using preprocessing tools
  train             -> run a single training job (optionally override --mode/--seed)
  orchestrate       -> plan/status/resume multi-run experiments (resumable)
  reset             -> clean artifacts; rebuild conversion_log.csv AND recreate project skeleton

Notes:
- Orchestration is implemented in orchestrate.py and called from here.
- Training is implemented in train.py and called from here.
- Conversion is implemented in preprocessing/* and called from here.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------- config helpers ---------------------------

try:
    import yaml
except Exception:
    yaml = None

def load_cfg(path="config.yaml"):
    if yaml is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text())
    except Exception:
        return {}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------- RESET UTILITIES --------------------------

def _parse_label_from_parts(parts):
    # Expect .../<mode>/<label>/<filename>.png
    # Labels map: benign -> 0, malware -> 1
    for token in parts:
        t = str(token).lower()
        if t == "benign": return 0
        if t == "malware": return 1
    return None

def rebuild_conversion_log(images_root: Path, out_csv: Path):
    """
    Scan images_root for PNGs and rebuild conversion_log.csv with
    columns: rel_path,label,mode,sha256
    """
    MODES = {"compress", "truncate"}
    rows = []
    if images_root.exists():
        for mode_dir in images_root.iterdir():
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name.lower()
            if mode not in MODES:
                continue
            # expect mode_dir / (benign|malware) / *.png
            for cls_dir in mode_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                label = _parse_label_from_parts([cls_dir.name])
                if label is None:
                    continue
                for png in cls_dir.rglob("*.png"):
                    sha = png.stem  # filename without extension
                    # write path relative to images_root
                    rel = png.relative_to(images_root).as_posix()
                    rows.append((rel, label, mode, sha))

    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "label", "mode", "sha256"])
        if rows:
            w.writerows(rows)
    print(f"[RESET] Rebuilt {out_csv} with {len(rows)} rows (relative to images_root={images_root})")

def ensure_index_csv(index_csv: Path):
    """Create logs/index.csv with a sane header if missing."""
    if not index_csv.exists():
        ensure_dir(index_csv.parent)
        with index_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id","mode","seed","kfold","epochs",
                "start_time","end_time","status","export_path"
            ])
        print(f"[RESET] Created stub run index: {index_csv}")

def ensure_orchestrate_log(plan_log: Path):
    """Touch the orchestrate plan log so status/resume can run without errors."""
    ensure_dir(plan_log.parent)
    if not plan_log.exists():
        plan_log.write_text("")
        print(f"[RESET] Created orchestrate plan log: {plan_log}")

def ensure_dataset_skeleton(images_root: Path, input_roots: list[str] | None = None):
    """
    Create dataset skeleton:
      dataset/input/{benign,malware}/
      images_root/{compress,truncate}/{benign,malware}/
    """
    # Input roots
    roots = input_roots or []
    if not roots:
        # default to dataset/input
        roots = ["dataset/input"]
    for r in roots:
        base = Path(r)
        ensure_dir(base / "benign")
        ensure_dir(base / "malware")

    # Output structure under images_root
    for mode in ("compress", "truncate"):
        for cls in ("benign", "malware"):
            ensure_dir(images_root / mode / cls)

def ensure_project_skeleton(cfg: dict):
    """
    After reset, ensure all required directories and stub files exist.
    """
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})

    # Resolve roots
    runs_dir   = Path(ti.get("runs_root", "runs")).resolve()
    cache_dir  = Path(paths.get("cache_root", "cache")).resolve()
    tmp_dir    = Path(paths.get("tmp_root", "tmp")).resolve()
    tb_dir     = Path("logs/tensorboard").resolve()
    index_csv  = Path(ti.get("run_index", "logs/index.csv")).resolve()
    conv_csv   = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    plan_log   = Path("logs/orchestrate_plan.log").resolve()
    export_dir = Path("export_models").resolve()

    # Ensure dirs
    ensure_dir(runs_dir)
    ensure_dir(cache_dir)
    ensure_dir(tmp_dir)
    ensure_dir(tb_dir)
    ensure_dir(export_dir)
    ensure_dir(conv_csv.parent)

    # Ensure dataset structure
    input_roots = paths.get("input_roots") or cfg.get("paths", {}).get("input_roots")
    ensure_dataset_skeleton(images_root, input_roots)

    # Ensure stub files
    ensure_index_csv(index_csv)
    ensure_orchestrate_log(plan_log)

    # Ensure conversion log exists (empty with header if not already)
    if not conv_csv.exists():
        rebuild_conversion_log(images_root, conv_csv)

def do_reset(yes: bool, cfg: dict):
    # Resolve paths
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})
    runs_dir   = Path(ti.get("runs_root", "runs")).resolve()
    cache_dir  = Path(paths.get("cache_root", "cache")).resolve()
    tmp_dir    = Path(paths.get("tmp_root", "tmp")).resolve()
    tb_dir     = Path("logs/tensorboard").resolve()
    index_csv  = Path(ti.get("run_index", "logs/index.csv")).resolve()
    conv_csv   = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()

    # Dry-run listing
    print("[RESET] Dry-run listing (use --yes to actually delete):")
    for p in [runs_dir, cache_dir, tmp_dir, tb_dir]:
        print(f"  dir : {p}")
    for f in [index_csv, conv_csv]:
        print(f"  file: {f}")
    print("  note: conversion_log.csv will be rebuilt from images_root if PNGs exist")

    if not yes:
        return

    # Delete dirs/files if they exist
    for p in [runs_dir, cache_dir, tmp_dir, tb_dir]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"[RESET] Deleted dir: {p}")

    # Delete index (we recreate a fresh stub)
    if index_csv.exists():
        try:
            index_csv.unlink()
            print(f"[RESET] Deleted file: {index_csv}")
        except Exception:
            pass

    # Always delete conversion_log then rebuild it
    if conv_csv.exists():
        try:
            conv_csv.unlink()
            print(f"[RESET] Deleted file: {conv_csv}")
        except Exception:
            pass

    # Recreate the full skeleton and rebuild CSV
    ensure_project_skeleton(cfg)
    rebuild_conversion_log(images_root, conv_csv)
    print("[RESET] Done.")

# --------------------------- SUBCOMMANDS ------------------------------

def cmd_verify(args):
    # Call verify_setup.py directly (same interpreter)
    cmd = [sys.executable, "verify_setup.py"]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

def cmd_convert(args):
    # Keep your existing conversion flow: dual_convert, plus explicit per-mode runs
    ensure_dir(Path("logs"))
    print("[CONVERT] Converting binaries -> images in both modes (compress, truncate)")
    cmds = [
        [sys.executable, "preprocessing/dual_convert.py", "--config", "config.yaml"],
        [sys.executable, "preprocessing/converter.py", "--config", "config.yaml", "--mode", "truncate"],
        [sys.executable, "preprocessing/converter.py", "--config", "config.yaml", "--mode", "compress"],
    ]
    for c in cmds:
        print("> " + " ".join(c))
        subprocess.run(c, check=False)

def cmd_train(args):
    """
    Pass through to train.py. Supports optional --mode and --seed overrides.
    Always passes --data-csv and --images-root (simplifies UX).
    """
    cfg = load_cfg(args.config)
    ti = cfg.get("train_io", {})
    data_csv = args.data_csv or ti.get("data_csv", "logs/conversion_log.csv")
    images_root = args.images_root or ti.get("images_root", "dataset/output")

    cmd = [sys.executable, "train.py",
           "--data-csv", data_csv,
           "--images-root", images_root]

    # Optional overrides agreed upon
    if args.mode:
        cmd += ["--mode", args.mode]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.resume:
        cmd += ["--resume"]

    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

def cmd_orchestrate(args):
    """
    Wire through to orchestrate.py:
      - plan (PLAN ONLY) via --runs N
      - status
      - resume
    """
    ensure_dir(Path("logs"))
    if args.action == "status":
        cmd = [sys.executable, "orchestrate.py", "status"]
    elif args.action == "resume":
        cmd = [sys.executable, "orchestrate.py", "resume"]
    else:
        if args.runs is None or args.runs <= 0:
            print("[ORCH] --runs N is required and must be > 0.")
            sys.exit(2)
        cmd = [sys.executable, "orchestrate.py", "plan", "--runs", str(args.runs)]

    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

def cmd_reset(args):
    cfg = load_cfg(args.config)
    do_reset(args.yes, cfg)

# ------------------------------- CLI ---------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="MalNet-FocusAug: convert, train, orchestrate, and manage runs."
    )
    sub = ap.add_subparsers(dest="cmd")

    # verify
    p_verify = sub.add_parser("verify", help="Check environment & data; print optimization recommendations.")
    p_verify.set_defaults(func=cmd_verify)

    # convert
    p_convert = sub.add_parser("convert", help="Convert input binaries to PNGs (both modes).")
    p_convert.set_defaults(func=cmd_convert)

    # train
    p_train = sub.add_parser("train", help="Train a single model (optionally override mode/seed).")
    p_train.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p_train.add_argument("--data-csv", type=str, default=None, help="Override path to conversion log CSV")
    p_train.add_argument("--images-root", type=str, default=None, help="Override path to images root (PNG dataset)")
    # agreed overrides:
    p_train.add_argument("--mode", type=str, choices=["compress", "truncate"], help="Override training mode (config default otherwise)")
    p_train.add_argument("--seed", type=int, help="Override RNG seed (config default otherwise)")
    p_train.add_argument("--resume", action="store_true", help="Resume this run at fold boundaries if checkpoints exist.")
    p_train.set_defaults(func=cmd_train)

    # orchestrate
    p_orch = sub.add_parser("orchestrate", help="Plan/status/resume multi-run training (resumable).")
    orch_sub = p_orch.add_subparsers(dest="action")

    # orchestrate (implicit plan) via --runs
    p_orch.add_argument("--runs", type=int, help="Number of repeats per mode (total runs = 2 * runs). If provided without action, creates a new plan.")

    # orchestrate plan (explicit)
    p_orch_plan = orch_sub.add_parser("plan", help="Create a new plan (PLAN ONLY; does not execute).")
    p_orch_plan.add_argument("--runs", type=int, required=True, help="Repeats per mode (total runs = 2 * runs).")
    p_orch_plan.set_defaults(func=cmd_orchestrate)

    # orchestrate status
    p_orch_status = orch_sub.add_parser("status", help="Show the latest plan and its run statuses.")
    p_orch_status.set_defaults(func=cmd_orchestrate)

    # orchestrate resume
    p_orch_resume = orch_sub.add_parser("resume", help="Execute all remaining PENDING runs in the latest plan.")
    p_orch_resume.set_defaults(func=cmd_orchestrate)

    # reset
    p_reset = sub.add_parser("reset", help="Clean artifacts, rebuild conversion_log.csv, and recreate project skeleton. Use --yes to confirm.")
    p_reset.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p_reset.add_argument("--yes", action="store_true", help="Actually delete. Without this flag it's a dry run.")
    p_reset.set_defaults(func=cmd_reset)

    return ap

def main():
    parser = build_parser()
    args, extra = parser.parse_known_args()

    # Special case: allow `python main.py orchestrate --runs N` (no explicit sub-action)
    if args.cmd == "orchestrate" and getattr(args, "action", None) is None:
        if getattr(args, "runs", None):
            args.action = "plan"
        else:
            print("Usage:")
            print("  python main.py orchestrate --runs N       # PLAN ONLY (shorthand)")
            print("  python main.py orchestrate plan --runs N  # PLAN ONLY (explicit)")
            print("  python main.py orchestrate status         # show latest plan")
            print("  python main.py orchestrate resume         # run all pending in latest plan")
            sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

