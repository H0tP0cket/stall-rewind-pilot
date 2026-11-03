"""terminalbench-v4 lane CLI (Docker for builds and grades; no model or provider calls).

  static                         static screen of the ratified tasks -> <evidence>/static-screen.json
  pipeline TASK [TASK ...]       build + package, admission gate, restore check, local per-task ledger
  run                            the pipeline over the ratified tasks in the ratified order; stops building when the
                                 Docker disk would fall below the floor and holds the rest with reason "docker disk"
  rescan-leak TASK [TASK ...]    re-run only the leak scan on the gate's captured states (after recording reviews)
  hold TASK --reason TEXT        record a hold for a task without building it
  record                         committed evidence, per-task ledger.json, lane ledger.json and registry.json (no seal)
  seal                           the user's step: seal VALIDATED tasks, rewrite registry.json and the ledgers
  verify-loads                   load every registry entry for development, calibration and main
  report                         write the admission README table and <evidence>/gate-report.json
  restore-private [TASK_ID ...]  rebuild private package files outside Git from the pinned checkout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from . import environment as E
from . import source as S
from . import static as ST
from .package import EVIDENCE, PACKAGES, write_json

STATIC = EVIDENCE / "static-screen.json"


def load_static() -> dict:
    record = ST.screen_all()
    write_json(STATIC, record)
    return record


def ledger_path(task: str) -> Path:
    return EVIDENCE / "tasks" / task / "ledger.json"


def pipeline(task: str, static_row: dict) -> dict:
    from .admission import verdict
    from .gate import validate
    from .package import build_package
    from .restore_check import run_check
    ledger = {"task": task, "task_id": S.package_task_id(task), "family_id": S.family_id(task), "started": time.time()}
    if static_row["static_verdict"] != "CANDIDATE":
        ledger.update(status=static_row["static_verdict"], stage="static", reasons=static_row["static_reasons"],
                      not_built="static screen: " + "; ".join(static_row["static_reasons"]))
        ledger["finished"] = time.time()
        write_json(ledger_path(task), ledger)
        return ledger
    try:
        built = build_package(task, static_row)
        ledger["package"] = {k: v for k, v in built.items() if k not in ("records", "build", "collect")}
        if built["status"] != "PACKAGED":
            ledger.update(status="HOLD", stage="package", reasons=[built.get("reason")],
                          docker_disk=bool(built.get("docker_disk")))
            return ledger
        ledger["hidden_tests"] = built["hidden_tests"]
        gate = validate(Path(built["package"]), static_row=static_row)
        if gate.get("stopped"):  # the gate could not run (docker disk); the restore check would fail the same way
            if (EVIDENCE / "tasks" / task / "restore-check").exists():  # never keep an earlier attempt's summary
                S.remove_tree(EVIDENCE / "tasks" / task / "restore-check")
            status, reasons = verdict(gate, None)
            ledger.update(status=status, stage="gate", reasons=reasons, docker_disk_runtime=True,
                          gate_items={k: v["passed"] for k, v in gate["items"].items()})
            return ledger
        restore = run_check(Path(built["package"]), EVIDENCE / "tasks" / task / "restore-check")
        status, reasons = verdict(gate, restore)
        ledger.update(status=status, stage="gate", reasons=reasons, docker_disk_runtime=bool(gate.get("docker_disk")),
                      gate_items={**{k: v["passed"] for k, v in gate["items"].items()},
                                  "restore_check": bool(restore.get("passed"))})
    except Exception as exc:  # recorded as a hold with its reason
        ledger.update(status="HOLD", stage="pipeline", reasons=[f"{type(exc).__name__}: {str(exc)[:500]}"],
                      traceback=traceback.format_exc()[-4000:])
    finally:
        ledger["finished"] = time.time()
        write_json(ledger_path(task), ledger)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m trajectory_retention.tasks.terminalbench_v4")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    p = sub.add_parser("pipeline"); p.add_argument("tasks", nargs="+")
    sub.add_parser("run")
    lk = sub.add_parser("rescan-leak"); lk.add_argument("tasks", nargs="+")
    h = sub.add_parser("hold"); h.add_argument("task"); h.add_argument("--reason", required=True)
    sub.add_parser("record")
    sub.add_parser("seal")
    sub.add_parser("verify-loads")
    sub.add_parser("report")
    r = sub.add_parser("restore-private"); r.add_argument("task_ids", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "static":
        print(json.dumps(load_static()["counts"]))
    elif args.command in ("pipeline", "run"):
        rows = {r["task"]: r for r in load_static()["tasks"]}
        tasks = args.tasks if args.command == "pipeline" else list(S.RATIFIED_TASKS)
        for i, task in enumerate(tasks):
            result = pipeline(task, rows[task])
            print(json.dumps({k: result.get(k) for k in ("task", "status", "stage", "reasons")}), flush=True)
            if args.command == "run" and result.get("docker_disk"):
                for rest in tasks[i + 1:]:
                    if rows[rest]["static_verdict"] != "CANDIDATE":
                        pipeline(rest, rows[rest])
                        continue
                    write_json(ledger_path(rest), {"task": rest, "task_id": S.package_task_id(rest),
                                                   "family_id": S.family_id(rest), "status": "HOLD", "stage": "build",
                                                   "docker_disk": True, "not_built": "docker disk",
                                                   "reasons": ["docker disk: building stopped before this task ("
                                                               + "; ".join(result.get("reasons") or []) + ")"],
                                                   "disk": E.docker_disk(), "recorded": time.time()})
                    print(json.dumps({"task": rest, "status": "HOLD", "reasons": ["docker disk"]}), flush=True)
                break
    elif args.command == "rescan-leak":
        from .admission import verdict
        from .gate import rescan_leak
        rows = {r["task"]: r for r in load_static()["tasks"]}
        for task in args.tasks:
            gate = rescan_leak(PACKAGES / S.package_task_id(task), rows[task])
            path = ledger_path(task)
            ledger = json.loads(path.read_text())
            restore = EVIDENCE / "tasks" / task / "restore-check" / "summary.json"
            status, reasons = verdict(gate, json.loads(restore.read_text()) if restore.is_file() else None)
            ledger.update(status=status, reasons=reasons, gate_items={**{k: v["passed"] for k, v in gate["items"].items()},
                          "restore_check": ledger.get("gate_items", {}).get("restore_check")})
            write_json(path, ledger)
            print(json.dumps({"task": task, "status": status, "reasons": reasons,
                              "leak": gate["items"]["leak_scan"]["counts"]}), flush=True)
    elif args.command == "hold":
        write_json(ledger_path(args.task), {"task": args.task, "task_id": S.package_task_id(args.task),
                                            "family_id": S.family_id(args.task), "status": "HOLD", "stage": "manual",
                                            "reasons": [args.reason], "recorded": time.time()})
    elif args.command == "record":
        from .admission import record_all
        print(json.dumps(record_all(load_static()), indent=1))
    elif args.command == "seal":
        from .admission import seal_all
        print(json.dumps(seal_all(load_static()), indent=1))
    elif args.command == "verify-loads":
        from .admission import verify_loads
        summary = verify_loads()
        print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))
    elif args.command == "report":
        from .report import write
        print(json.dumps(write(), indent=1))
    elif args.command == "restore-private":
        from .package import restore_private
        ids = args.task_ids or sorted(p.name for p in PACKAGES.iterdir() if (p / "package.json").is_file())
        for task_id in ids:
            print(json.dumps(restore_private(task_id)), flush=True)


if __name__ == "__main__":
    main()
