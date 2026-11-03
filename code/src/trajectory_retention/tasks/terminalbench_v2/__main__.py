"""terminalbench-v2 lane CLI (Docker for builds and grades; no model or provider calls).

  static                         static screen of all 89 tasks -> artifacts/v2/lane/static-screen.json
  pipeline TASK [TASK ...]       build + package, admission gate, restore check, per-task ledger
  candidates [--limit N]         run the pipeline over static candidates in build-priority order, one at a time,
                                 recording a hold for every candidate left unbuilt (with the reason)
  rescan-leak TASK [TASK ...]    re-run only the leak scan on the gate's captured states (after recording reviews)
  hold TASK --reason TEXT        record a hold for a task without building it
  hold-unbuilt --cause TEXT      hold every candidate not yet carried through whose build needs apt-get, naming where
                                 (task Dockerfile, worker python3 layer, grader packages) and the cause
  seal                           seal validated tasks; write registry.json and the ledger of every task considered
  verify-loads                   load every registry entry for development, calibration and main
  report                         write the gate report JSON and README table
  restore-private [TASK_ID ...]  rebuild private package files outside Git from the pinned checkout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import traceback

from . import environment as E
from . import source as S
from . import static as ST
from .package import EVIDENCE, PACKAGES, write_json

STATIC = EVIDENCE / "static-screen.json"


def load_static() -> dict:
    if not STATIC.is_file():
        ST.write(STATIC)
    return json.loads(STATIC.read_text())


def cached_tasks() -> set[str]:
    """Tasks whose terminalbench-v1 images exist locally (their rebuild reuses Docker's layer cache)."""
    out = E.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], check=False).stdout.decode().splitlines()
    return {line.split("/", 1)[1].split(":", 1)[0] for line in out if line.startswith("tr-terminalbench/")
            and line.endswith(":grader-v1")}


def apt_dependency(task: str, row: dict) -> list[str]:
    """Where this task's three image builds would run apt-get (read from the recipe, nothing built)."""
    dockerfile = (S.CHECKOUT / task / "environment" / "Dockerfile").read_text()
    bases = list(dict.fromkeys(re.findall(r"^FROM\s+(\S+)", dockerfile, re.M)))
    out = []
    if "apt-get" in dockerfile:
        out.append("the task Dockerfile runs apt-get")
    elif not any(b.startswith("python:") for b in bases):
        out.append(f"the worker layer installs python3 with apt-get (base {', '.join(bases)} ships none)")
    if row.get("grade_time_apt_packages"):
        out.append("the grader layer installs " + ", ".join(row["grade_time_apt_packages"]) + " with apt-get")
    return out


def pipeline(task: str, static_row: dict) -> dict:
    from .admission import verdict
    from .gate import validate
    from .package import build_package
    from .restore_check import run_check
    path = EVIDENCE / "tasks" / task / "ledger.json"
    ledger = {"task": task, "task_id": S.package_task_id(task), "family_id": S.family_id(task), "started": time.time()}
    try:
        built = build_package(task)
        ledger["package"] = {k: v for k, v in built.items() if k not in ("records", "build", "collect")}
        if built["status"] != "PACKAGED":
            ledger.update(status="HOLD", stage="package", reasons=[built.get("reason")])
            return ledger
        ledger["hidden_tests"] = built["hidden_tests"]
        gate = validate(Path(built["package"]), static_row=static_row)
        restore = run_check(Path(built["package"]), EVIDENCE / "tasks" / task / "restore-check")
        status, reasons = verdict(gate, restore)
        ledger.update(status=status, stage="gate", reasons=reasons,
                      gate_items={**{k: v["passed"] for k, v in gate["items"].items()},
                                  "restore_check": bool(restore.get("passed"))})
    except Exception as exc:  # recorded as a hold with its reason
        ledger.update(status="HOLD", stage="pipeline", reasons=[f"{type(exc).__name__}: {str(exc)[:500]}"],
                      traceback=traceback.format_exc()[-4000:])
    finally:
        ledger["finished"] = time.time()
        write_json(path, ledger)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m trajectory_retention.tasks.terminalbench_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("static")
    p = sub.add_parser("pipeline"); p.add_argument("tasks", nargs="+")
    c = sub.add_parser("candidates"); c.add_argument("--limit", type=int, default=None)
    lk = sub.add_parser("rescan-leak"); lk.add_argument("tasks", nargs="+")
    h = sub.add_parser("hold"); h.add_argument("task"); h.add_argument("--reason", required=True)
    hu = sub.add_parser("hold-unbuilt"); hu.add_argument("--cause", required=True)
    sub.add_parser("seal")
    sub.add_parser("verify-loads")
    sub.add_parser("report")
    r = sub.add_parser("restore-private"); r.add_argument("task_ids", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "static":
        record = ST.write(STATIC)
        print(json.dumps(record["counts"]))
    elif args.command == "pipeline":
        rows = {r["task"]: r for r in load_static()["tasks"]}
        for task in args.tasks:
            result = pipeline(task, rows[task])
            print(json.dumps({k: result.get(k) for k in ("task", "status", "reasons")}), flush=True)
    elif args.command == "candidates":
        static = load_static()
        cached = cached_tasks()
        order = sorted((r for r in static["tasks"] if r["static_verdict"] == "CANDIDATE"),
                       key=lambda r: ST.build_priority(r, cached))
        write_json(EVIDENCE / "build-order.json", {"rule": ST.build_priority.__doc__, "v1_cached": sorted(cached),
                                                   "order": [r["task"] for r in order]})
        done = 0
        for row in order:
            path = EVIDENCE / "tasks" / row["task"] / "ledger.json"
            if path.is_file() and json.loads(path.read_text()).get("status") in ("VALIDATED", "EXCLUDED", "HOLD") \
                    and not json.loads(path.read_text()).get("not_built"):
                continue
            if args.limit is not None and done >= args.limit:
                break
            try:
                free = E.require_headroom("the next candidate")
            except RuntimeError as exc:
                for rest in order[order.index(row):]:
                    rp = EVIDENCE / "tasks" / rest["task"] / "ledger.json"
                    if not rp.is_file() or json.loads(rp.read_text()).get("not_built"):
                        write_json(rp, {"task": rest["task"], "task_id": rest["task_id"], "family_id": rest["family_id"],
                                        "status": "HOLD", "stage": "build", "not_built": str(exc),
                                        "reasons": ["not built: " + str(exc)], "recorded": time.time()})
                print(json.dumps({"stopped": str(exc)}), flush=True)
                break
            result = pipeline(row["task"], row)
            done += 1
            print(json.dumps({"task": row["task"], "status": result.get("status"), "reasons": result.get("reasons"),
                              "free_gb_before": round(free / 1e9, 1)}), flush=True)
    elif args.command == "rescan-leak":
        from .admission import verdict
        from .gate import rescan_leak
        for task in args.tasks:
            gate = rescan_leak(PACKAGES / S.package_task_id(task))
            path = EVIDENCE / "tasks" / task / "ledger.json"
            ledger = json.loads(path.read_text())
            restore = EVIDENCE / "tasks" / task / "restore-check" / "summary.json"
            status, reasons = verdict(gate, json.loads(restore.read_text()) if restore.is_file() else None)
            ledger.update(status=status, reasons=reasons, gate_items={**{k: v["passed"] for k, v in gate["items"].items()},
                          "restore_check": ledger.get("gate_items", {}).get("restore_check")})
            write_json(path, ledger)
            print(json.dumps({"task": task, "status": status, "reasons": reasons,
                              "leak": gate["items"]["leak_scan"]["counts"]}), flush=True)
    elif args.command == "hold":
        row = {r["task"]: r for r in load_static()["tasks"]}[args.task]
        write_json(EVIDENCE / "tasks" / args.task / "ledger.json",
                   {"task": args.task, "task_id": row["task_id"], "family_id": row["family_id"], "status": "HOLD",
                    "stage": "manual", "reasons": [args.reason], "recorded": time.time()})
    elif args.command == "hold-unbuilt":
        for row in load_static()["tasks"]:
            path = EVIDENCE / "tasks" / row["task"] / "ledger.json"
            if row["static_verdict"] != "CANDIDATE" or (path.is_file() and not json.loads(path.read_text()).get("not_built")):
                continue
            needs = apt_dependency(row["task"], row)
            if not needs:
                print(json.dumps({"task": row["task"], "skipped": "no apt-get in its builds; run the pipeline"}))
                continue
            why = "; ".join(needs) + "; " + args.cause
            write_json(path, {"task": row["task"], "task_id": row["task_id"], "family_id": row["family_id"],
                              "status": "HOLD", "stage": "build", "not_built": why, "apt_dependency": needs,
                              "reasons": ["not built: " + why], "recorded": time.time()})
            print(json.dumps({"task": row["task"], "status": "HOLD", "apt_dependency": needs}), flush=True)
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
