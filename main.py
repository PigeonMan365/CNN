#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — project CLI hub

subcommands:
  verify        : run environment/data checks (auto-creates folders, rebuilds CSV)
  convert       : convert binaries -> images (both modes) + rebuild conversion log
  train         : train a model (auto-resume if an interrupt exists; else config-driven new run)
  reset         : clear runs/cache/logs/tmp/exports, PRESERVE dataset, rebuild conversion_log.csv
  clear         : clear dataset/output/ and conversion_log.csv; preserve dataset/input/
  generate      : generate random test files (500 bits to 500KB) in input folders
  orchestrate   : plan/resume multi-run experiments via orchestrate.py

notes:
- seeds are tracked in runs/seed_state.json
- "train --mode both" trains resize and truncate with the SAME seed, then advances it
- use "clear" before re-converting with new target sizes in config
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

HERE = Path(__file__).resolve().parent
SEED_LEDGER = HERE / "runs" / "seed_state.json"

# ----------------- basic helpers -----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_cfg(path: Optional[str]) -> dict:
    """Load YAML config; tolerate absence."""
    if path is None:
        path = "config.yaml"
    try:
        import yaml
    except Exception:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}

def cfg_get(cfg: dict, key: str, default=None):
    """Dot-path getter: cfg_get(cfg, 'training.epochs', 10)"""
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# ----------------- seed ledger -----------------

def _seed_state_load() -> dict:
    ensure_dir(SEED_LEDGER.parent)
    if SEED_LEDGER.exists():
        try:
            return json.loads(SEED_LEDGER.read_text())
        except Exception:
            pass
    return {}

def _seed_state_save(state: dict) -> None:
    ensure_dir(SEED_LEDGER.parent)
    SEED_LEDGER.write_text(json.dumps(state, indent=2))

def _seed_get_current(cfg: dict) -> int:
    state = _seed_state_load()
    if "current_seed" in state:
        return int(state["current_seed"])
    # default from config if present, else 0
    return int(cfg_get(cfg, "training.base_seed", 0))

def _seed_set_current(seed: int) -> None:
    state = _seed_state_load()
    state["current_seed"] = int(seed)
    _seed_state_save(state)

def _seed_bump() -> int:
    """Increment and persist, returning the new value."""
    state = _seed_state_load()
    cur = int(state.get("current_seed", 0))
    nxt = cur + 1
    state["current_seed"] = nxt
    _seed_state_save(state)
    return nxt

# ----------------- resume discovery -----------------

@dataclass
class ResumeInfo:
    mode: str
    seed: int
    fold: int
    ckpt: Path

INTERRUPT_RE = re.compile(r"^(?P<mode>resize|truncate)_seed(?P<seed>\d+)$")

def _find_most_recent_interrupt(runs_root: Path = HERE / "runs") -> Optional[ResumeInfo]:
    """
    Find the most recent runs/<mode>_seedX/foldY_interrupt.pt by mtime.
    """
    if not runs_root.exists():
        return None
    latest: Optional[Tuple[float, ResumeInfo]] = None
    for run_dir in runs_root.glob("*_seed*"):
        m = INTERRUPT_RE.match(run_dir.name)
        if not m:
            continue
        mode = m.group("mode")
        seed = int(m.group("seed"))
        for f in run_dir.glob("fold*_interrupt.pt"):
            try:
                mt = f.stat().st_mtime
            except Exception:
                continue
            # parse fold number
            fold_num = 0
            m2 = re.match(r"^fold(\d+)_interrupt\.pt$", f.name)
            if m2:
                fold_num = int(m2.group(1))
            info = ResumeInfo(mode=mode, seed=seed, fold=fold_num, ckpt=f)
            if latest is None or mt > latest[0]:
                latest = (mt, info)
    return latest[1] if latest else None

# ----------------- conversion log rebuild (used by reset) -----------------

def _hash_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _rebuild_conversion_log_from_dataset(images_root: Path, log_csv: Path) -> int:
    """Scan images_root and write logs/conversion_log.csv (rel_path,label,mode,sha256)."""
    ensure_dir(log_csv.parent)
    ensure_dir(images_root)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
    rows: List[str] = []
    for cls, label in (("benign", 0), ("malware", 1)):
        for mode in ("resize", "truncate"):
            base = images_root / cls / mode
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    rel = p.relative_to(images_root).as_posix()
                    name = p.stem
                    if len(name) == 64 and all(c in "0123456789abcdef" for c in name.lower()):
                        sha = name.lower()
                    else:
                        sha = _hash_file_sha256(p)
                    rows.append(f"{rel},{label},{mode},{sha}")
    log_csv.write_text("rel_path,label,mode,sha256\n" + "\n".join(rows), encoding="utf-8")
    print(f"[reset] rebuilt {log_csv} with {len(rows)} rows from {images_root}")
    return len(rows)

# ----------------- subcommand: verify -----------------

def cmd_verify(args):
    py = sys.executable
    script = HERE / "verify_setup.py"
    if not script.exists():
        print("[verify] verify_setup.py not found next to main.py")
        sys.exit(2)
    cmd = [py, str(script)]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

# ----------------- subcommand: convert -----------------

def cmd_convert(args):
    """
    Convert binaries -> images (both modes) + rebuild conversion log via
    preprocessing/convert.py.
    """
    cfg = load_cfg(getattr(args, "config", None))
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})

    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    conv_csv    = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()
    ensure_dir(Path("logs"))
    ensure_dir(images_root)

    # Parse target sizes from CLI if provided
    from preprocessing.convert import _parse_target_size
    resize_size = None
    truncate_size = None
    if getattr(args, "resize_size", None):
        resize_size = _parse_target_size(args.resize_size, (64, 64))
    if getattr(args, "truncate_size", None):
        truncate_size = _parse_target_size(args.truncate_size, (256, 256))
    
    used_fallback = False
    try:
        from preprocessing.convert import run_all as _run_all
        print("[convert] using preprocessing.convert.run_all")
        _run_all(config_path=getattr(args, "config", "config.yaml"),
                 rebuild_only=False, skip_convert=False,
                 resize_target_size=resize_size,
                 truncate_target_size=truncate_size)
    except Exception as e:
        used_fallback = True
        print(f"[convert] import failed: {e.__class__.__name__}: {e}")
        print("[convert] falling back to subprocess: python -m preprocessing.convert")
        cmd = [sys.executable, "-m", "preprocessing.convert", "--config", getattr(args, "config", "config.yaml")]
        if resize_size:
            cmd.extend(["--resize-size", f"{resize_size[0]},{resize_size[1]}"])
        if truncate_size:
            cmd.extend(["--truncate-size", f"{truncate_size[0]},{truncate_size[1]}"])
        print("> " + " ".join(cmd))
        subprocess.run(cmd, check=False)

    # Summary
    counts = {}
    if images_root.exists():
        for png in images_root.rglob("*.png"):
            try:
                rel = png.relative_to(images_root).as_posix()
            except ValueError:
                continue
            parts = rel.split("/")
            if len(parts) >= 3:
                k = (parts[0], parts[1])  # (label, mode)
                counts[k] = counts.get(k, 0) + 1

    if counts:
        print("[convert] PNG counts under images_root:")
        for (label, mode), n in sorted(counts.items()):
            print(f"  {label:7s} / {mode:9s} : {n}")
    else:
        print("[convert] No PNGs found under images_root.")

    if conv_csv.exists():
        import csv as _csv
        total = 0
        with conv_csv.open("r", encoding="utf-8", newline="") as f:
            for _ in _csv.DictReader(f):
                total += 1
        print(f"[convert] conversion_log.csv rows: {total}")

    if used_fallback:
        print("[convert] NOTE: subprocess path worked; consider fixing imports so direct import succeeds.")

# ----------------- subcommand: train -----------------

def _build_train_cmd(mode: str, seed: int, cfg: dict, resume: bool) -> list:
    """
    Build command to run train.py.
    Most settings come from config.yaml, only pass essential overrides.
    """
    py = sys.executable
    train_py = HERE / "train.py"
    cmd = [
        py, str(train_py),
        "--mode", mode,
        "--seed", str(seed),
    ]
    # data-csv and images-root will be read from config if not provided
    # Only pass resume if requested
    if resume:
        cmd.append("--resume")
    return cmd

def cmd_train(args):
    """
    - If --resume given, resume that run (mode/seed) or latest interrupt if not specified.
    - If no --resume:
        * if an interrupt exists, resume it automatically
        * else start a new run based on config.yaml defaults
    - If --mode both, train resize then truncate with SAME seed; bump seed once after both succeed.
    """
    cfg = load_cfg(getattr(args, "config", None))
    default_mode = cfg_get(cfg, "training.mode", "resize")
    allow_resume = bool(cfg_get(cfg, "training.allow_resume", True))

    # Discover most recent interrupt if any
    recent = _find_most_recent_interrupt()

    # Determine requested mode
    mode = getattr(args, "mode", None)
    if mode is None:
        mode = default_mode
    mode = mode.lower()
    if mode not in ("resize", "truncate", "both"):
        print(f"[train] invalid mode '{mode}'. use resize|truncate|both")
        sys.exit(2)

    # Determine seed
    if getattr(args, "seed", None) is not None:
        seed = int(args.seed)
    else:
        seed = _seed_get_current(cfg)

    # Resume logic
    want_resume = bool(getattr(args, "resume", False))
    if not want_resume and allow_resume and recent:
        print(f"[train] Resuming most recent interrupted run: {recent.mode}_seed{recent.seed}")
        cmd = _build_train_cmd(recent.mode, recent.seed, cfg, resume=True)
        print("> " + " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            print("[train] resume finished ok.")
        else:
            print(f"[train] resume exited with code {rc}")
        return

    # Otherwise: new run (single mode or both)
    if mode == "both":
        seed_to_use = seed
        # resize
        cmd = _build_train_cmd("resize", seed_to_use, cfg, resume=False)
        print("> " + " ".join(cmd))
        rc1 = subprocess.run(cmd, check=False).returncode
        # truncate
        if rc1 == 0:
            cmd = _build_train_cmd("truncate", seed_to_use, cfg, resume=False)
            print("> " + " ".join(cmd))
            rc2 = subprocess.run(cmd, check=False).returncode
        else:
            rc2 = 1

        if rc1 == 0 and rc2 == 0:
            # Both succeeded: advance seed once
            _seed_set_current(seed_to_use + 1)
            print(f"[train] both modes completed. advanced seed to {seed_to_use + 1}.")
        else:
            print("[train] one or both modes failed; seed not advanced.")
        return

    # single mode path
    cmd = _build_train_cmd(mode, seed, cfg, resume=want_resume)
    print("> " + " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc == 0 and not want_resume:
        # advance seed after a successful single-mode new run
        _seed_set_current(seed + 1)
        print(f"[train] {mode} completed. advanced seed to {seed + 1}.")

# ----------------- subcommand: reset -----------------

def cmd_reset(_args):
    """
    Clear runs/, export_models/, cache/, logs/, tmp/; PRESERVE dataset/,
    then rebuild logs/conversion_log.csv from dataset/output.
    """
    cfg = load_cfg(None)
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})

    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    cache_root  = Path(paths.get("cache_root", "cache")).resolve()
    runs_root   = Path(cfg_get(cfg, "train_io.runs_root", "runs")).resolve()
    logs_root   = HERE / "logs"
    tmp_root    = HERE / "tmp"
    export_root = Path(cfg_get(cfg, "training.export_root", "export_models")).resolve()

    def _wipe(p: Path):
        if p.exists():
            subprocess.run([sys.executable, "-c", "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)", str(p)], check=False)
        ensure_dir(p)

    for d in (runs_root, export_root, cache_root, logs_root, tmp_root):
        _wipe(d)

    # Ensure dataset skeleton exists, but DO NOT delete contents
    for cls in ("benign", "malware"):
        for mode in ("resize", "truncate"):
            ensure_dir(images_root / cls / mode)

    # Reset seed ledger
    if SEED_LEDGER.exists():
        SEED_LEDGER.unlink(missing_ok=True)

    # Rebuild conversion log from dataset
    conv_csv = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()
    _rebuild_conversion_log_from_dataset(images_root, conv_csv)

    print("[reset] project reset complete (dataset preserved).")

# ----------------- subcommand: clear -----------------

def cmd_clear(_args):
    """
    Clear dataset/output/ (all converted PNGs) and logs/conversion_log.csv.
    Preserves dataset/input/ (original binary files).
    Useful when you need to re-convert with different target sizes.
    """
    cfg = load_cfg(None)
    paths = cfg.get("paths", {})
    ti    = cfg.get("train_io", {})
    
    images_root = Path(ti.get("images_root", paths.get("images_root", "dataset/output"))).resolve()
    conv_csv = Path(paths.get("conversion_log", ti.get("data_csv", "logs/conversion_log.csv"))).resolve()
    
    # Clear all converted images
    if images_root.exists():
        print(f"[clear] Removing all converted images from: {images_root}")
        subprocess.run([sys.executable, "-c", "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)", str(images_root)], check=False)
    
    # Recreate directory structure
    ensure_dir(images_root)
    for cls in ("benign", "malware"):
        for mode in ("resize", "truncate"):
            ensure_dir(images_root / cls / mode)
    
    # Clear conversion log
    ensure_dir(conv_csv.parent)
    if conv_csv.exists():
        conv_csv.unlink(missing_ok=True)
        print(f"[clear] Removed conversion log: {conv_csv}")
    
    # Create empty conversion log with header
    conv_csv.write_text("rel_path,label,mode,sha256\n", encoding="utf-8")
    print(f"[clear] Created empty conversion log: {conv_csv}")
    
    print("[clear] Output dataset cleared. Run 'python main.py convert' to regenerate images with current config settings.")

# ----------------- subcommand: orchestrate -----------------

def cmd_orch_plan(args):
    py = sys.executable
    orch = HERE / "orchestrate.py"
    if not orch.exists():
        print("[orchestrate] orchestrate.py not found.")
        sys.exit(2)
    runs = str(getattr(args, "runs", 1))
    cmd = [py, str(orch), "plan", "--runs", runs]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

def cmd_orch_resume(_args):
    py = sys.executable
    orch = HERE / "orchestrate.py"
    if not orch.exists():
        print("[orchestrate] orchestrate.py not found.")
        sys.exit(2)
    cmd = [py, str(orch), "resume"]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=False)

# ----------------- subcommand: generate -----------------

def cmd_generate(args):
    """
    Generate random test files in input folders.
    Creates 50 files per label (benign, malware) with random sizes
    between 500 bits (63 bytes) and 500KB (512000 bytes).
    """
    import random
    
    cfg = load_cfg(getattr(args, "config", None))
    paths = cfg.get("paths", {})
    input_roots = paths.get("input_roots", ["dataset/input"])
    
    num_files = getattr(args, "num_files", 50)
    min_size = 63  # 500 bits = 62.5 bytes, round to 63
    max_size = 512000  # 500KB = 500 * 1024 bytes
    
    print(f"[generate] Generating {num_files} random files per label")
    print(f"[generate] Size range: {min_size} bytes ({min_size*8} bits) to {max_size} bytes ({max_size/1024:.1f} KB)")
    
    total_generated = 0
    for input_root_str in input_roots:
        input_root = Path(input_root_str).resolve()
        print(f"[generate] Processing input root: {input_root}")
        
        for label in ("benign", "malware"):
            label_dir = input_root / label
            ensure_dir(label_dir)
            
            # Generate files
            for i in range(num_files):
                # Random file size between min_size and max_size
                file_size = random.randint(min_size, max_size)
                
                # Generate random bytes
                file_content = os.urandom(file_size)
                
                # Create filename with index and size info
                filename = f"random_{label}_{i:03d}_{file_size}b.bin"
                file_path = label_dir / filename
                
                # Write file
                file_path.write_bytes(file_content)
                total_generated += 1
            
            print(f"[generate] Generated {num_files} files in {label_dir}")
    
    print(f"[generate] Total files generated: {total_generated}")
    print(f"[generate] Files are ready in: {', '.join(input_roots)}")

# ----------------- subcommand: test -----------------

def cmd_test(args):
    """Handle test subcommands."""
    test_cmd = getattr(args, "test_cmd", None)
    if test_cmd == "convert":
        py = sys.executable
        test_script = HERE / "preprocessing" / "test_convert.py"
        if not test_script.exists():
            print("[test] preprocessing/test_convert.py not found.")
            sys.exit(2)
        cmd = [py, str(test_script)]
        if hasattr(args, "num_files") and args.num_files and args.num_files != 5:
            cmd.extend(["--num-files", str(args.num_files)])
        if hasattr(args, "no_cleanup") and args.no_cleanup:
            cmd.append("--no-cleanup")
        if hasattr(args, "config") and args.config:
            cmd.extend(["--config", str(args.config)])
        print("> " + " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        sys.exit(rc)
    else:
        print(f"[test] Unknown test command: {test_cmd}")
        print("[test] Available: convert")
        sys.exit(2)

# ----------------- CLI -----------------

def build_parser():
    ap = argparse.ArgumentParser(prog="main", description="Project CLI")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_verify = sp.add_parser("verify", help="Verify environment and dataset; auto-creates folders; rebuilds CSV.")
    sp_verify.set_defaults(func=cmd_verify)

    sp_convert = sp.add_parser("convert", help="Convert binaries -> images (both modes) and rebuild CSV.")
    sp_convert.add_argument("--resize-size", type=str, default=None,
                           help="Override resize target size (e.g., '64,64' or '64x64')")
    sp_convert.add_argument("--truncate-size", type=str, default=None,
                           help="Override truncate target size (e.g., '256,256' or '256x256')")
    sp_convert.set_defaults(func=cmd_convert)

    sp_train = sp.add_parser("train", help="Train a model; auto-resume if an interrupt exists; else start new.")
    sp_train.add_argument("--mode", choices=["resize", "truncate", "both"], 
                         help="Override training mode from config (default: from config.training.mode).")
    sp_train.add_argument("--seed", type=int, help="Override seed (default: from seed ledger or auto-increment).")
    sp_train.add_argument("--resume", action="store_true", help="Resume this run (or the most recent interrupted run if --mode/--seed omitted).")
    sp_train.set_defaults(func=cmd_train)

    sp_reset = sp.add_parser("reset", help="Clear runs/cache/logs/tmp/export_models; preserve dataset; rebuild CSV.")
    sp_reset.set_defaults(func=cmd_reset)

    sp_clear = sp.add_parser("clear", help="Clear dataset/output/ and conversion_log.csv; preserve dataset/input/. Use before re-converting with new target sizes.")
    sp_clear.set_defaults(func=cmd_clear)

    sp_generate = sp.add_parser("generate", help="Generate random test files in input folders.")
    sp_generate.add_argument("--num-files", type=int, default=50, help="Number of files to generate per label (default: 50)")
    sp_generate.set_defaults(func=cmd_generate)

    sp_orch = sp.add_parser("orchestrate", help="Plan/resume orchestrated experiments")
    orch_sub = sp_orch.add_subparsers(dest="orch_cmd", required=True)
    orch_plan = orch_sub.add_parser("plan", help="Create an experiment plan")
    orch_plan.add_argument("--runs", type=int, default=1, help="Number of rounds/seeds to schedule")
    orch_plan.set_defaults(func=cmd_orch_plan)
    orch_resume = orch_sub.add_parser("resume", help="Resume an experiment plan")
    orch_resume.set_defaults(func=cmd_orch_resume)

    sp_test = sp.add_parser("test", help="Run tests")
    test_sub = sp_test.add_subparsers(dest="test_cmd", required=True, title="test commands")
    test_convert = test_sub.add_parser("convert", help="Test conversion with random files")
    test_convert.add_argument("--num-files", type=int, default=5, help="Number of test files per label (default: 5)")
    test_convert.add_argument("--no-cleanup", action="store_true", help="Don't clean up test files after test")
    test_convert.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    test_convert.set_defaults(func=cmd_test)

    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func") and callable(args.func):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

