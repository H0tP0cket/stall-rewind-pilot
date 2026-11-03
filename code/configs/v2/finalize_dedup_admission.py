"""Finish the authorized dedup admission and launch only its first pair when capacity permits.

This is an operational queue, not an experiment policy. It cannot approve findings,
alter task/evaluator bytes, retry mines, substitute models, or waive a failed gate.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import natural

ROOT = Path(__file__).resolve().parents[2]
TASK = "distributed-dedup"
TASK_ID = "terminalbench-v4-pilot-training-" + TASK
BASE = ROOT / "artifacts/v2/lane-v4-pilot-training"
QUEUE = BASE / "dedup-queue"
PLAN = ROOT / "artifacts/v2/screen/pilot-dedup-neutral-20261028/plans/screen-stage1-1-r2.json"
LEGACY = ROOT / "artifacts/v2/screen/pilot-legacy-neutral-20261028/plans/screen-stage1-1-r2.json"
PRETRAIN = ROOT / "artifacts/v2/screen/pilot-pretrain-neutral-20261028/plans/screen-stage1-1-r2.json"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def record(phase, **details):
    QUEUE.mkdir(parents=True, exist_ok=True)
    row = {"phase": phase, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "pid": os.getpid(), "task": TASK, "plan": str(PLAN), **details}
    temp = QUEUE / "status.tmp"
    temp.write_text(json.dumps(row, indent=2) + "\n")
    temp.replace(QUEUE / "status.json")
    print(json.dumps(row), flush=True)


def checked_plan():
    plan, approved = natural.load_plan(PLAN), natural.load_plan(LEGACY)
    assert plan["tasks"] == [TASK_ID] and plan["repeats"] == 2 and plan["concurrency"] == 1
    for field in ("model", "settings", "context_limit", "submission_stop", "unresolved_retry",
                  "submission_feedback", "budget"):
        if plan[field] != approved[field]:
            raise ValueError("Dedup plan differs from authorized worker policy: " + field)
    assert plan["budget"]["cost_limit_usd"] == 3
    return plan


def pending_jobs(path):
    plan = natural.load_plan(path)
    directory = Path(plan["store"]) / "status"
    pending = []
    for job in natural.jobs(plan):
        row = read(directory / (job["key"] + ".json")) or {}
        if row.get("state") not in ("finished", "error"):
            pending.append({"key": job["key"], "state": row.get("state", "pending")})
    return pending


def capacity():
    # Pretrain's serial driver can immediately start its queued second job:
    # reserve four CPUs until the whole pair is finished. Legacy uses two each.
    legacy, pretrain = pending_jobs(LEGACY), pending_jobs(PRETRAIN)
    continuations = []
    for reservation in read(QUEUE / "worker-reservations.json") or []:
        result = read(Path(reservation["result_path"]))
        if not result or result.get("state") != "finished":
            continuations.append(reservation)
    reserved = 2 * len(legacy) + (4 if pretrain else 0) + sum(r["cpus"] for r in continuations)
    return {"available": not pretrain and reserved + 8 <= (os.cpu_count() or 0),
            "other_worker_cpu_reservations": reserved, "dedup_worker_cpus": 8,
            "host_cpus": os.cpu_count(), "legacy_unfinished": legacy, "pretrain_unfinished": pretrain,
            "technical_continuations": continuations}


def stopped(plan):
    return (QUEUE / "STOP").exists() or (Path(plan["store"]) / "STOP").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="read-only readiness check; never seals or launches")
    parser.add_argument("--pipeline-pid", type=int)
    parser.add_argument("--resume-admitted", action="store_true",
                        help="verify the existing seals and resume only the capacity queue")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = checked_plan()
    ledger = BASE / "tasks" / TASK / "ledger.json"
    if args.resume_admitted and not ledger.exists():
        raise ValueError("Cannot resume admission queue without the existing ledger")
    if args.check:
        print(json.dumps({"admission": read(ledger), "capacity": capacity(), "stopped": stopped(plan),
                          "provider_calls": 0}, indent=2))
        return
    if (not args.pipeline_pid and not args.resume_admitted) or os.environ.get("DOCKER_CONTEXT") != "colima-tr-admission":
        raise ValueError("Require the owned admission pipeline PID and colima-tr-admission context")
    QUEUE.mkdir(parents=True, exist_ok=True)
    lock = os.open(QUEUE / "claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock, str(os.getpid()).encode()); os.close(lock)
    bound = natural.binding_sha(plan)
    record("waiting_for_admission", plan_sha256=bound, pipeline_pid=args.pipeline_pid)
    while not ledger.exists():
        if stopped(plan):
            record("stopped_before_launch"); return
        process = subprocess.run(["ps", "-p", str(args.pipeline_pid), "-o", "command="], capture_output=True, text=True)
        if process.returncode or "pipeline distributed-dedup" not in process.stdout:
            raise RuntimeError("Admission process ended without a ledger; no mine launched")
        time.sleep(15)
    if natural.binding_sha(checked_plan()) != bound:
        raise ValueError("Queued plan changed; no mine launched")
    from trajectory_retention.tasks.terminalbench_v4_pilot_training import admission as A
    from trajectory_retention.tasks.terminalbench_v4_pilot_training import static as S
    from trajectory_retention.tasks.terminalbench_v4_pilot_training.__main__ import main as lane_main
    from trajectory_retention.tasks.terminalbench_v4_pilot_training.report import write
    # Reviews are exact, previously inspected finding IDs. New findings or failed
    # reference, repeatability, isolation or restore assertions cannot pass this.
    if not args.resume_admitted:
        lane_main(["rescan-leak", TASK])
    latest = read(ledger)
    if latest["status"] != "VALIDATED":
        record("admission_hold", reasons=latest.get("reasons"), provider_calls=0); return
    previous = A.ADMISSION / "tasks" / "terminalbench-v4-pilot-training-pretrain-shard-corruption" / "admission.json"
    previous_hash = hashlib.sha256(previous.read_bytes()).hexdigest()
    if not args.resume_admitted:
        static = S.screen_all()
        A.write_task_records(TASK, next(r for r in static["tasks"] if r["task"] == TASK))
        A.seal_all(static)
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == previous_hash
    if args.resume_admitted:
        # The real task loader verifies this exact task's sealed package; never
        # recreate admission evidence when resuming an operational queue.
        natural.load_task(plan, TASK_ID, Path(plan["store"]) / "task-scratch" / "queue-resume-verification")
    else:
        assert A.verify_loads()["passed"]
        write()
    record("admitted_waiting_for_capacity", plan_sha256=bound, capacity=capacity())
    while not capacity()["available"]:
        if stopped(plan):
            record("stopped_before_launch"); return
        time.sleep(15)
    if stopped(plan):
        record("stopped_before_launch"); return
    if natural.binding_sha(checked_plan()) != bound:
        raise ValueError("Queued plan changed; no mine launched")
    command = [sys.executable, str(ROOT / "configs/v2/natural.py"), "--plan", str(PLAN)]
    subprocess.run(command + ["--dry-run"], check=True)
    record("launching_first_pair", plan_sha256=bound, capacity=capacity(), repeats=2, concurrency=1,
           guard_usd_per_run=3)
    subprocess.run(command, check=True)
    record("first_pair_driver_finished", summary=natural.merged(plan, natural.layout(plan)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record("error", error=f"{type(exc).__name__}: {exc}")
        raise
