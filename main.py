#!/usr/bin/env python3
"""
main.py — project entrypoint

Subcommands:
  verify      -> environment & data audit with knob recommendations
  convert     -> convert binaries to PNGs (both modes) using preprocessing tools
  train       -> run a single training job (optionally override --mode/--seed/--resume)
  orchestrate -> plan/status/resume multi-run experiments (resumable)
  reset       -> clean artifacts; rebuild conversion_log.csv AND recreate project skeleton
"""

import argparse
import csv
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

def rebuild_conversion_log(images_root: Path, out_csv: Path):
    """
    Scan images_root for PNGs and rebuild conversion_log.csv with
    columns: rel_path,label,mode,sha256

    Only include files that match our expected structure:
      images_root/<mode>/<label>/<sha256>.png
      where <mode> in {compress, truncate}, <label> in {benign, malware},
      and <sha256> is 64 hex chars.
    """
    import re
    MODES  = {"compress", "truncate"}
    LABELS = {"benign": 0, "malware": 1}
    HEX64  = re.compile(r"^[0-9a-fA-F]{64}\.png$")

    rows = []
    ignored = 0

    if images_root.exists():
        for mode_dir in images_root.iterdir():
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name.lower()
            if mode not in MODES:
                continue

            for cls_dir in mode_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                cls = cls_dir.name.lower()
                if cls not in LABELS:
                    continue

                for png in cls_dir.glob("*.png"):
                    if not HEX64.match(png.name):
                        ignored += 1
                        continue
                    sha = png.stem.lower()
                    rel = png.relative_to(images_root).as_posix()  # POSIX in CSV
                    rows.append((rel, LABELS[cls], mode, sha))

    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "label", "mode", "sha256"])
        if rows:
            w.writerows(rows)
    print(f"[INFO] Rebuilt {out_csv} with {len(rows)} rows (images_root={images_root})")
    if ignored:
        print(f"[INFO] Ignored {ignored} non-conforming PNGs (expected 64-hex filenames).")

def ensure_index_csv(index_csv: Path):
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
    ensure_dir(plan_log.parent)
    if not plan_log.exists():
        plan_log.write_text("")
        print(f"[RESET] Created orchestrate plan log: {plan_log}")

def ensure_dataset_skeleton(images_root: Path, input_roots):
    roots = input_roots or ["dataset/input"]
    for r in roots:
        base = Path(r)
        ensure_dir(base / "benign")
        ensure_dir(base / "malware")
    for mode in ("compress", "truncate"):
        for cls in ("benign", "malware"):
            ensure_dir(images_root / mode / cls)

def ensure_project_skeleton(cfg: dict):
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})

    runs_dir   = Path(ti.get("runs_root", "runs")).resolve()
    cache_dir  = Path(paths.get("cache_root", "cache")).resolve()
    tmp_dir    = Path(paths.get("tmp_root", "tmp")).resolve()
    tb_dir     = Path("logs/tensorboard").resolve()
    index_csv  = Path(ti.get("run_index", "logs/index.csv")).resolve()
    conv_csv   = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    plan_log   = Path("logs/orchestrate_plan.log").resolve()
    export_dir = Path("export_models").resolve()

    for d in (runs_dir, cache_dir, tmp_dir, tb_dir, export_dir, conv_csv.parent):
        ensure_dir(d)

    input_roots = paths.get("input_roots") or ["dataset/input"]
    ensure_dataset_skeleton(images_root, input_roots)
    ensure_index_csv(index_csv)
    ensure_orchestrate_log(plan_log)
    # Do NOT rebuild conversion log here; we do it exactly once in reset/convert.

def do_reset(yes: bool, cfg: dict):
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})
    runs_dir   = Path(ti.get("runs_root", "runs")).resolve()
    cache_dir  = Path(paths.get("cache_root", "cache")).resolve()
    tmp_dir    = Path(paths.get("tmp_root", "tmp")).resolve()
    tb_dir     = Path("logs/tensorboard").resolve()
    index_csv  = Path(ti.get("run_index", "logs/index.csv")).resolve()
    conv_csv   = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()

    print("[RESET] Dry-run listing (use --yes to actually delete):")
    for p in [runs_dir, cache_dir, tmp_dir, tb_dir]:
        print(f"  dir : {p}")
    for f in [index_csv, conv_csv]:
        print(f"  file: {f}")
    print("  note: conversion_log.csv will be rebuilt from images_root if PNGs exist")

    if not yes:
        return

    for p in [runs_dir, cache_dir, tmp_dir, tb_dir]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"[RESET] Deleted dir: {p}")

    for f in [index_csv, conv_csv]:
        if f.exists():
            try:
                f.unlink()
                print(f"[RESET] Deleted file: {f}")
            except Exception:
                pass

    ensure_project_skeleton(cfg)
    # Single rebuild here (no duplicates)
    rebuild_conversion_log(images_root, conv_csv)
    print("[RESET] Done.")

# --------------------------- SUBCOMMANDS ------------------------------

def cmd_verify(args):
    cmd = [sys.executable, "verify_setup.py"]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

def cmd_convert(args):
    cfg   = load_cfg(args.config)
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})

    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    conv_csv    = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()

    ensure_dir(Path("logs"))
    ensure_dir(images_root)

    print("[CONVERT] Converting binaries -> images in both modes via preprocessing/dual_convert.py")
    cmd = [sys.executable, "preprocessing/dual_convert.py", "--config", "config.yaml"]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

    rebuild_conversion_log(images_root, conv_csv)

    counts = {}
    if images_root.exists():
        for png in images_root.rglob("*.png"):
            rel = png.relative_to(images_root).as_posix()
            parts = rel.split("/")
            if len(parts) >= 3:
                key = (parts[0], parts[1])  # (mode, label)
                counts[key] = counts.get(key, 0) + 1
    if counts:
        print("[CONVERT] PNG counts under images_root:")
        for (mode, label), n in sorted(counts.items()):
            print(f"  {mode:9s} / {label:7s} : {n}")
    else:
        print("[CONVERT] No PNGs found under images_root — check config.paths.input_roots and images_root.")


def cmd_train(args):
    cfg = load_cfg(args.config)
    ti = cfg.get("train_io", {})
    data_csv = args.data_csv or ti.get("data_csv", "logs/conversion_log.csv")
    images_root = args.images_root or ti.get("images_root", "dataset/output")

    cmd = [sys.executable, "train.py",
           "--data-csv", data_csv,
           "--images-root", images_root]

    if args.mode:
        cmd += ["--mode", args.mode]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.resume:
        cmd += ["--resume"]

    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

def cmd_orchestrate(args):
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

    p_verify = sub.add_parser("verify", help="Check environment & data; print optimization recommendations.")
    p_verify.set_defaults(func=cmd_verify)

    p_convert = sub.add_parser("convert", help="Convert input binaries to PNGs (both modes).")
    p_convert.add_argument("--config", type=str, default="config.yaml")
    p_convert.set_defaults(func=cmd_convert)

    p_train = sub.add_parser("train", help="Train a single model (optionally override mode/seed).")
    p_train.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p_train.add_argument("--data-csv", type=str, default=None, help="Override path to conversion log CSV")
    p_train.add_argument("--images-root", type=str, default=None, help="Override path to images root (PNG dataset)")
    p_train.add_argument("--mode", type=str, choices=["compress", "truncate"], help="Override training mode (config default otherwise)")
    p_train.add_argument("--seed", type=int, help="Override RNG seed (config default otherwise)")
    p_train.add_argument("--resume", action="store_true", help="Resume this run at fold boundaries if checkpoints exist.")
    p_train.set_defaults(func=cmd_train)

    p_orch = sub.add_parser("orchestrate", help="Plan/status/resume multi-run training (resumable).")
    orch_sub = p_orch.add_subparsers(dest="action")
    p_orch.add_argument("--runs", type=int, help="Number of repeats per mode (total runs = 2 * runs). If provided without action, creates a new plan.")
    p_orch_plan = orch_sub.add_parser("plan", help="Create a new plan (PLAN ONLY; does not execute).")
    p_orch_plan.add_argument("--runs", type=int, required=True, help="Repeats per mode (total runs = 2 * runs).")
    p_orch_plan.set_defaults(func=cmd_orchestrate)
    p_orch_status = orch_sub.add_parser("status", help="Show the latest plan and its run statuses.")
    p_orch_status.set_defaults(func=cmd_orchestrate)
    p_orch_resume = orch_sub.add_parser("resume", help="Execute all remaining PENDING runs in the latest plan.")
    p_orch_resume.set_defaults(func=cmd_orchestrate)

    p_reset = sub.add_parser("reset", help="Clean artifacts, rebuild conversion_log.csv, and recreate project skeleton. Use --yes to confirm.")
    p_reset.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p_reset.add_argument("--yes", action="store_true", help="Actually delete. Without this flag it's a dry run.")
    p_reset.set_defaults(func=cmd_reset)

    return ap

def main():
    parser = build_parser()
    args, extra = parser.parse_known_args()

    if args.cmd == "orchestrate" and getattr(args, "action", None) is None:
        if getattr(args, "runs", None):
            args.action = "plan"
        else:
            print("Usage:")
            print("  python main.py orchestrate --runs N")
            print("  python main.py orchestrate plan --runs N")
            print("  python main.py orchestrate status")
            print("  python main.py orchestrate resume")
            sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
