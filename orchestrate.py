#!/usr/bin/env python3
# Orchestration planner/executor (resumable)
# Spec (final):
# - `python main.py orchestrate --runs N` -> PLAN ONLY. Append a new plan with 2*N runs (compress then truncate).
# - `python main.py orchestrate status`   -> Show ONLY the latest plan; if no pending runs, say so.
# - `python main.py orchestrate resume`   -> Execute ALL remaining pending runs in the latest plan (sequential).
# - Seeds autoincrement per run within a plan, starting at 0 (not read from config).
# - Modes always: compress, truncate (grouped: all compress first, then all truncate).
# - kfold is read from config.yaml (for visibility) but not overridden by CLI here.
# - Success = a new export matching cnn_{mode}_{seed}_*.ts.pt is created in export_models/.
# - On success mark run DONE; on failure mark FAILED and stop. Ctrl+C leaves current run PENDING.
# - Log file: logs/orchestrate_plan.log (append plans; rewrite only the LAST plan block when updating statuses).

import argparse
import os
import re
import sys
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

try:
    import yaml  # only for displaying kfold info (no mutation)
except Exception:
    yaml = None

LOGS_DIR = Path("logs")
PLAN_LOG = LOGS_DIR / "orchestrate_plan.log"
EXPORT_DIR = Path("export_models")
MODES = ["compress", "truncate"]  # grouped execution order

# ----------------------------- Utilities -----------------------------

def iso_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def read_log_text() -> str:
    if PLAN_LOG.exists():
        return PLAN_LOG.read_text(encoding="utf-8", errors="ignore")
    return ""

def write_log_text(text: str):
    ensure_logs_dir()
    PLAN_LOG.write_text(text, encoding="utf-8")

def list_exports() -> set:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return set(str(p) for p in EXPORT_DIR.glob("*.ts.pt"))

def fmt_plan_block(plan_id: int, created: str, runs: List[Dict]) -> str:
    lines = []
    lines.append(f"PLAN {plan_id} CREATED {created}")
    for i, r in enumerate(runs, start=1):
        # RUN <idx> <mode> seed=<seed> status=<STATUS>
        lines.append(f"RUN {i} {r['mode']} seed={r['seed']} status={r['status']}")
    lines.append("ENDPLAN")
    return "\n".join(lines)

def parse_plans(text: str) -> List[Dict]:
    """
    Parses the append-only log into plan dicts:
    [{"plan_id": int, "created": str, "runs":[{"mode":..., "seed":..., "status":...}, ...]}]
    """
    plans = []
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        m = re.match(r"^PLAN\s+(\d+)\s+CREATED\s+(.+)$", lines[i].strip())
        if m:
            plan_id = int(m.group(1))
            created = m.group(2).strip()
            i += 1
            runs = []
            while i < len(lines) and lines[i].strip() != "ENDPLAN":
                ln = lines[i].strip()
                mrun = re.match(r"^RUN\s+(\d+)\s+(compress|truncate)\s+seed=(\d+)\s+status=(PENDING|RUNNING|DONE|FAILED)$", ln)
                if mrun:
                    runs.append({
                        "mode": mrun.group(2),
                        "seed": int(mrun.group(3)),
                        "status": mrun.group(4)
                    })
                i += 1
            # Expect ENDPLAN at current i or EOF
            plans.append({"plan_id": plan_id, "created": created, "runs": runs})
        else:
            i += 1
    return plans

def rewrite_last_plan(plans: List[Dict]):
    """
    Rewrites the entire file but only based on current `plans` state.
    We keep the same format; only the latest plan might change statuses.
    """
    blocks = []
    for pl in plans:
        blocks.append(fmt_plan_block(pl["plan_id"], pl["created"], pl["runs"]))
    text = "\n".join(blocks) + ("\n" if blocks else "")
    write_log_text(text)

# -------------------------- Plan Operations --------------------------

def next_plan_id(existing: List[Dict]) -> int:
    return (max((p["plan_id"] for p in existing), default=0) + 1)

def create_plan(runs_n: int) -> Dict:
    """
    Create a new plan dict with grouped modes and auto-increment seeds starting at 0:
    compress seeds [0..runs_n-1], then truncate seeds [runs_n..2*runs_n-1]
    """
    created = iso_now()
    runs = []
    # compress first
    for s in range(runs_n):
        runs.append({"mode": "compress", "seed": s, "status": "PENDING"})
    # then truncate continuing seeds
    for s in range(runs_n, 2 * runs_n):
        runs.append({"mode": "truncate", "seed": s, "status": "PENDING"})
    return {"created": created, "runs": runs}

def append_new_plan_to_log(runs_n: int) -> int:
    txt = read_log_text()
    plans = parse_plans(txt)
    pid = next_plan_id(plans)
    newp = create_plan(runs_n)
    newp["plan_id"] = pid
    plans.append(newp)
    rewrite_last_plan(plans)
    return pid

def get_latest_plan() -> Tuple[List[Dict], Dict]:
    txt = read_log_text()
    plans = parse_plans(txt)
    if not plans:
        return plans, None
    return plans, plans[-1]

def latest_plan_has_pending(plan: Dict) -> bool:
    return any(r["status"] == "PENDING" for r in plan["runs"])

def summarize_plan(plan: Dict) -> str:
    if not plan:
        return "No plans recorded."
    total = len(plan["runs"])
    n_pending = sum(1 for r in plan["runs"] if r["status"] == "PENDING")
    n_running = sum(1 for r in plan["runs"] if r["status"] == "RUNNING")
    n_done = sum(1 for r in plan["runs"] if r["status"] == "DONE")
    n_failed = sum(1 for r in plan["runs"] if r["status"] == "FAILED")
    lines = [
        f"Plan {plan['plan_id']} (created {plan['created']}): total={total} | PENDING={n_pending} RUNNING={n_running} DONE={n_done} FAILED={n_failed}"
    ]
    for i, r in enumerate(plan["runs"], start=1):
        lines.append(f"  #{i:02d} mode={r['mode']} seed={r['seed']} status={r['status']}")
    return "\n".join(lines)

# ---------------------------- Execution ------------------------------

def build_train_cmd(mode: str, seed: int) -> List[str]:
    # Always call via main.py so it uses the same environment/entrypoint
    return [sys.executable, "main.py", "train", "--mode", mode, "--seed", str(seed)]

def run_one(plans: List[Dict], plan: Dict, run_index: int) -> bool:
    """
    Execute a single pending run at position run_index in the plan.
    Returns True on SUCCESS, False on FAILURE. On KeyboardInterrupt, keeps status PENDING.
    """
    run = plan["runs"][run_index]
    mode = run["mode"]
    seed = run["seed"]

    # Mark RUNNING
    run["status"] = "RUNNING"
    rewrite_last_plan(plans)

    # Snapshot exports before
    before = list_exports()

    cmd = build_train_cmd(mode, seed)
    print(f"[ORCH] Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n[ORCH] Interrupted. Leaving current run as PENDING.")
        run["status"] = "PENDING"
        rewrite_last_plan(plans)
        raise

    # Detect success by new export matching pattern cnn_{mode}_{seed}_*.ts.pt
    after = list_exports()
    new_files = sorted(after - before)
    pattern = re.compile(rf"^.*?/cnn_{mode}_{seed}_(\d+)\.ts\.pt$")
    success = any(pattern.match(p) for p in new_files)

    if success:
        run["status"] = "DONE"
        rewrite_last_plan(plans)
        print(f"[ORCH] SUCCESS: mode={mode} seed={seed}")
        return True
    else:
        run["status"] = "FAILED"
        rewrite_last_plan(plans)
        print(f"[ORCH] FAILED (no new export detected) for mode={mode} seed={seed}")
        return False

def resume_all_pending():
    plans, plan = get_latest_plan()
    if not plan:
        print("[ORCH] No plan on record. Use: python main.py orchestrate --runs N")
        return

    if not latest_plan_has_pending(plan):
        print("[ORCH] No pending runs in latest plan. Nothing to resume.")
        print(summarize_plan(plan))
        return

    # Run all pending in sequence until failure or completion
    for idx, r in enumerate(plan["runs"]):
        if r["status"] == "PENDING":
            ok = run_one(plans, plan, idx)
            if not ok:
                print("[ORCH] Stopping after failure. Run 'python main.py orchestrate resume' to try remaining later.")
                print(summarize_plan(plan))
                return

    # If we made it here, no more pending
    print("[ORCH] Plan complete. Current plan has no remaining PENDING runs.")
    print(summarize_plan(plan))

# ------------------------------ CLI ---------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="orchestrate.py",
        description="Plan and resume multi-run training (compress + truncate, k-fold from config)."
    )
    subparsers = ap.add_subparsers(dest="subcmd")

    # plan (default when invoked via `python main.py orchestrate --runs N`)
    plan_p = subparsers.add_parser("plan", help="Create a new plan (PLAN ONLY, does not execute).")
    plan_p.add_argument("--runs", type=int, required=True, help="Number of repeats per mode (total runs = 2 * runs).")

    # status
    subparsers.add_parser("status", help="Show the latest plan and its run statuses.")

    # resume
    subparsers.add_parser("resume", help="Execute all remaining PENDING runs in the latest plan.")

    # If no subcmd given but --runs present, treat as plan (compat with 'main.py orchestrate --runs N')
    ap.add_argument("--runs", type=int, required=False, help=argparse.SUPPRESS)

    args = ap.parse_args()

    # Handle top-level fallback: if called as `python main.py orchestrate --runs N` with no 'plan'
    if args.subcmd is None and args.runs is not None:
        args.subcmd = "plan"
        args.__dict__["runs"] = args.runs  # ensure value present under plan parser view

    if args.subcmd == "plan":
        runs_n = args.runs
        if runs_n is None or runs_n <= 0:
            print("[ORCH] --runs N (N>0) is required to create a plan.")
            sys.exit(2)

        # Show kfold info from config (if available) just for visibility
        if yaml is not None and Path("config.yaml").exists():
            try:
                cfg = yaml.safe_load(Path("config.yaml").read_text())
                kfold = cfg.get("training", {}).get("kfold", None)
                if kfold:
                    print(f"[ORCH] Using kfold={kfold} (from config.yaml)")
                else:
                    print("[ORCH] kfold not set in config.yaml (training will use whatever train.py defaults to).")
            except Exception:
                pass

        pid = append_new_plan_to_log(runs_n)
        print(f"[ORCH] Created plan {pid} with total runs = {2 * runs_n} (grouped: {runs_n} compress, {runs_n} truncate).")
        print(f"[ORCH] Log: {PLAN_LOG}")
        return

    elif args.subcmd == "status":
        plans, plan = get_latest_plan()
        if not plan:
            print("[ORCH] No plans recorded.")
            return
        print(summarize_plan(plan))
        if not latest_plan_has_pending(plan):
            print("[ORCH] Latest plan has no PENDING runs.")
        return

    elif args.subcmd == "resume":
        resume_all_pending()
        return

    else:
        # If user typed just `python main.py orchestrate` with nothing else, explain usage briefly.
        print("Usage:")
        print("  python main.py orchestrate --runs N       # PLAN ONLY (also supported: python main.py orchestrate plan --runs N)")
        print("  python main.py orchestrate status         # show latest plan")
        print("  python main.py orchestrate resume         # run all pending in latest plan")
        sys.exit(1)


if __name__ == "__main__":
    main()

