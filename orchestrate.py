#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orchestrate.py — seed-round planner/executor

Plan format:
  Round s:  resize@s → truncate@s
  Round s+1: resize@s+1 → truncate@s+1
The plan preserves per-round ordering and only increments the seed after both
modes for a round complete.

It writes progress and allows resume. On finishing the full plan it updates
runs/seed_state.json: { "last_seed": s_last }
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PLANS_JSON = HERE / "logs" / "orchestrate_plans.json"
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


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------ plan model ---------------------------

@dataclass
class Job:
    mode: str
    seed: int
    status: str = "PENDING"  # PENDING, RUNNING, DONE, FAILED
    export_path: Optional[str] = None
    log: Optional[str] = None


@dataclass
class Plan:
    id: int
    created_at: str
    rounds: int
    jobs: List[Job]
    cursor: int = 0  # index of next job to run

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at,
            "rounds": self.rounds,
            "cursor": self.cursor,
            "jobs": [asdict(j) for j in self.jobs],
        }


# ------------------------------- storage -----------------------------

def _load_all_plans() -> List[Plan]:
    if not PLANS_JSON.exists():
        return []
    data = json.loads(PLANS_JSON.read_text(encoding="utf-8"))
    out = []
    for p in data.get("plans", []):
        jobs = [Job(**j) for j in p["jobs"]]
        out.append(Plan(id=p["id"], created_at=p["created_at"], rounds=p["rounds"], jobs=jobs, cursor=p.get("cursor", 0)))
    return out


def _save_all_plans(plans: List[Plan]) -> None:
    PLANS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PLANS_JSON.write_text(json.dumps({"plans": [pl.to_dict() for pl in plans]}, indent=2), encoding="utf-8")


def _next_plan_id(plans: List[Plan]) -> int:
    return (max([p.id for p in plans]) + 1) if plans else 1


# ------------------------------- commands ----------------------------

def cmd_plan(args):
    cfg = load_config()
    base_seed = int(cfg_get(cfg, "training.seed", 0) or 0)
    last_seed = _read_last_seed(base_seed)
    start_seed = max(base_seed, last_seed + 1)

    rounds = int(args.runs or 1)
    jobs: List[Job] = []
    for i in range(rounds):
        s = start_seed + i
        jobs.append(Job("resize", s))
        jobs.append(Job("truncate", s))

    plans = _load_all_plans()
    pid = _next_plan_id(plans)
    plan = Plan(id=pid, created_at=utc_now_iso(), rounds=rounds, jobs=jobs, cursor=0)
    plans.append(plan)
    _save_all_plans(plans)
    print(f"[ORCH] Created plan {pid} with total jobs = {len(jobs)} "
          f"(grouped into {rounds} round(s) of resize->truncate by seed).")
    print(f"[ORCH] Log: {PLANS_JSON}")


def _run_job(job: Job) -> Tuple[bool, Optional[str]]:
    py = sys.executable
    cmd = [py, str(HERE / "train.py"),
           "--data-csv", "logs/conversion_log.csv",
           "--images-root", "dataset/output",
           "--mode", job.mode,
           "--seed", str(job.seed)]
    print("[ORCH] Executing:", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        return False, None

    # try to discover last export for this (mode,seed)
    export_dir = HERE / "export_models"
    pattern = f"cnn_{job.mode}_{job.seed}_*.ts.pt"
    ex = sorted(export_dir.glob(pattern))
    return True, (ex[-1].as_posix() if ex else None)


def _update_ledger_if_complete(plan: Plan):
    # if all jobs in plan are DONE, set ledger last_seed = max seed in plan
    if all(j.status == "DONE" for j in plan.jobs):
        seeds = [j.seed for j in plan.jobs]
        if seeds:
            _write_last_seed(max(seeds))


def cmd_resume(_args):
    plans = _load_all_plans()
    if not plans:
        print("[ORCH] No plans found. Use: python main.py orchestrate plan --runs N")
        return
    plan = plans[-1]  # latest
    # find next pending
    while plan.cursor < len(plan.jobs):
        i = plan.cursor
        job = plan.jobs[i]
        if job.status in ("DONE", "FAILED"):
            plan.cursor += 1
            continue
        job.status = "RUNNING"
        _save_all_plans(plans)
        ok, export_path = _run_job(job)
        if ok:
            job.status = "DONE"
            job.export_path = export_path
            plan.cursor += 1
            _save_all_plans(plans)
        else:
            job.status = "FAILED"
            _save_all_plans(plans)
            print(f"[ORCH] FAILED for mode={job.mode} seed={job.seed} (stopping here).")
            _print_status(plan)
            return
    _update_ledger_if_complete(plan)
    _save_all_plans(plans)
    _print_status(plan)


def _print_status(plan: Plan):
    done = sum(1 for j in plan.jobs if j.status == "DONE")
    failed = sum(1 for j in plan.jobs if j.status == "FAILED")
    pending = sum(1 for j in plan.jobs if j.status == "PENDING")
    running = sum(1 for j in plan.jobs if j.status == "RUNNING")
    print(f"Plan {plan.id} (created {plan.created_at}): total={len(plan.jobs)} | "
          f"PENDING={pending} RUNNING={running} DONE={done} FAILED={failed}")
    for i, j in enumerate(plan.jobs, 1):
        ep = f" export={j.export_path}" if j.export_path else ""
        print(f"  #{i:02d} mode={j.mode} seed={j.seed} status={j.status}{ep}")


def cmd_status(_args):
    plans = _load_all_plans()
    if not plans:
        print("[ORCH] No plans found.")
        return
    _print_status(plans[-1])


# -------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser(prog="orchestrate", description="Seed-round orchestrator")
    sub = ap.add_subparsers(dest="cmd")

    ap_plan = sub.add_parser("plan", help="create a new plan")
    ap_plan.add_argument("--runs", type=int, default=1, help="number of rounds (seed values)")
    ap_plan.set_defaults(func=cmd_plan)

    ap_status = sub.add_parser("status", help="show status of the latest plan")
    ap_status.set_defaults(func=cmd_status)

    ap_resume = sub.add_parser("resume", help="execute/resume the latest plan")
    ap_resume.set_defaults(func=cmd_resume)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
