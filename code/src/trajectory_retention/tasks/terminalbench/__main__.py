"""terminalbench-v1 lane CLI (no model or provider calls).

  split                      write benchmark_tasks/admission/terminalbench-v1/family-split.json
  pipeline TASK [TASK ...]   build images + package, validate, runner check (calibration-split tasks only)
  hold TASK --reason TEXT    record a measured hold for a task without packaging
  seal                       seal validated tasks, write registry + cohort
  verify-loads --attempt N   load checks (dev/cal ok for calibration, held-out and main refused)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from . import source as S


def wait_for_load(limit: float = 20.0, poll: float = 180.0) -> None:
    while os.getloadavg()[0] > limit:
        print(f"load {os.getloadavg()[0]:.1f} > {limit}; waiting", flush=True)
        time.sleep(poll)


def pipeline(task: str) -> dict:
    from .package import EVIDENCE, build_package, write_json
    from .runner_check import run_check
    from .validate import validate
    from .admission import load_split
    split = load_split()["tasks"][task]["split"]
    ledger_path = EVIDENCE / "tasks" / task / "ledger.json"
    ledger = {"task": task, "family_id": S.family_id(task), "split": split, "started": time.time()}
    if split != "calibration":
        ledger.update(status="NOT_PACKAGED", reason="held-out family: only calibration-split tasks are packaged")
        write_json(ledger_path, ledger)
        return ledger
    wait_for_load()
    built = build_package(task, split=split)
    ledger["package"] = {k: v for k, v in built.items() if k not in ("records",)}
    if built["status"] != "PACKAGED":
        ledger.update(status="HOLD", reason=built.get("reason"))
        write_json(ledger_path, ledger)
        return ledger
    wait_for_load()
    record = validate(Path(built["package"]))
    ledger["validation"] = {"passed": record["passed"], "gates": record["gates"], "timing": record["timing"]}
    if not record["passed"]:
        ledger.update(status="HOLD", reason="validation gate failed: " + ", ".join(k for k, v in record["gates"].items() if not v))
        write_json(ledger_path, ledger)
        return ledger
    wait_for_load()
    summary = run_check(Path(built["package"]), EVIDENCE / "tasks" / task / "runner-check")
    ledger["runner_check"] = {"passed": summary.get("passed"), "error": summary.get("error"),
                              "failed": [k for k, v in summary.get("assertions", {}).items() if not v]}
    ledger["status"] = "VALIDATED" if summary.get("passed") else "HOLD"
    if not summary.get("passed"):
        ledger["reason"] = "runner check failed: " + json.dumps(ledger["runner_check"])
    ledger["finished"] = time.time()
    write_json(ledger_path, ledger)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m trajectory_retention.tasks.terminalbench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("split")
    p = sub.add_parser("pipeline"); p.add_argument("tasks", nargs="+")
    h = sub.add_parser("hold"); h.add_argument("task"); h.add_argument("--reason", required=True)
    h.add_argument("--evidence", default=None)
    sub.add_parser("seal")
    v = sub.add_parser("verify-loads"); v.add_argument("--attempt", required=True)
    args = parser.parse_args(argv)
    if args.command == "split":
        from .admission import ADMISSION
        record = S.write_split(ADMISSION / "family-split.json")
        print(json.dumps(record["counts"]))
    elif args.command == "pipeline":
        for task in args.tasks:
            result = pipeline(task)
            print(json.dumps({k: result.get(k) for k in ("task", "split", "status", "reason")}), flush=True)
    elif args.command == "hold":
        from .package import EVIDENCE, write_json
        from .admission import load_split
        write_json(EVIDENCE / "tasks" / args.task / "ledger.json",
                   {"task": args.task, "family_id": S.family_id(args.task),
                    "split": load_split()["tasks"][args.task]["split"], "status": "HOLD", "reason": args.reason,
                    "evidence": args.evidence, "recorded": time.time()})
    elif args.command == "seal":
        from .admission import seal_all
        print(json.dumps(seal_all(), indent=1))
    elif args.command == "verify-loads":
        from .admission import verify_loads
        summary = verify_loads(args.attempt)
        print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))


if __name__ == "__main__":
    main()
