"""MettleBench admission CLI (no model/provider calls).

  python -m trajectory_retention.tasks.mettlebench_v1 dev --task UPSTREAM_ID --solve SOLVE.sh [--steps K] [--regrade]
  python -m trajectory_retention.tasks.mettlebench_v1 qualify --task UPSTREAM_ID
  python -m trajectory_retention.tasks.mettlebench_v1 admit --task UPSTREAM_ID
  python -m trajectory_retention.tasks.mettlebench_v1 verify-loads
  python -m trajectory_retention.tasks.mettlebench_v1 write-cohort
  python -m trajectory_retention.tasks.mettlebench_v1 verify-cohort
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

from . import source as S

REPO = Path(__file__).resolve().parents[4]
REFERENCES = REPO / "benchmark_tasks/mettlebench-references"
DEV_ROOT = S.DEFAULT_CACHE / "devpkg"


def dev_package(upstream_id: str, solve: Path) -> Path:
    from .harness import image_id
    from .package import build_package
    dest = DEV_ROOT / upstream_id
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ids = [r["task_id"] for r in S.verify_source()["manifest"]]
    split = S.family_split(ids)[S.family_for(upstream_id)]
    build_package(upstream_id, dest, image_id=image_id(upstream_id), reference=solve, split=split)
    return dest


def cmd_dev(args) -> int:
    from .adapter import MettleBenchTaskAdapter
    from .harness import run_and_grade, split_steps
    solve = Path(args.solve).resolve() if args.solve else None
    placeholder = None
    if solve is None:
        placeholder = Path(tempfile.mkdtemp()) / "solve.sh"
        placeholder.write_text("#!/usr/bin/env bash\n# placeholder: untouched start\n")
    package = dev_package(args.task, solve or placeholder)
    adapter = MettleBenchTaskAdapter(package, scratch_root=S.DEFAULT_CACHE / "devscratch")
    out = Path(tempfile.mkdtemp(prefix=f"dev-{args.task[:20]}-", dir=str(S.DEFAULT_CACHE / "devscratch")
                                if (S.DEFAULT_CACHE / "devscratch").exists() else None))
    script = solve.read_text() if solve else None
    report = run_and_grade(adapter, script, out, steps=args.steps, regrade=args.regrade, timeout=args.timeout)
    evaluation = report.pop("_evaluation")
    if script is not None:
        report["declared_steps"] = len(split_steps(script)[1])
    report["failures"] = [{"id": c.check_id, "status": c.status.value, "raw": c.raw_status,
                           "message": c.details.get("message"), "excerpt": (c.details.get("excerpt") or "")[-1500:],
                           "traceback": (c.details.get("traceback") or "")[-1500:]}
                          for c in evaluation.checks if c.status.value != "PASS"][:3]
    report["private_output"] = str(out)
    print(json.dumps(report, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mettlebench_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("dev", help="run a private witness script in a real worker container and grade it")
    dev.add_argument("--task", required=True)
    dev.add_argument("--solve")
    dev.add_argument("--steps", type=int)
    dev.add_argument("--regrade", action="store_true")
    dev.add_argument("--timeout", type=float, default=1800)
    for name in ("qualify", "admit"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True, action="append")
    sub.add_parser("verify-loads")
    sub.add_parser("write-cohort")
    sub.add_parser("verify-cohort")
    args = parser.parse_args(argv)
    if args.command == "dev":
        return cmd_dev(args)
    if args.command == "qualify":
        from .qualify import qualify
        ok = True
        for task in args.task:
            result = qualify(task)
            print(json.dumps({"task": task, "passed": result["passed"], "summary": result.get("summary")}, default=str))
            ok &= result["passed"]
        return 0 if ok else 1
    if args.command == "admit":
        from .admit import admit
        for task in args.task:
            print(json.dumps(admit(task), default=str))
        return 0
    if args.command == "verify-loads":
        from .admit import verify_loads
        print(json.dumps(verify_loads(), indent=2, default=str))
        return 0
    if args.command == "write-cohort":
        from .admit import write_cohort
        print(json.dumps(write_cohort(), indent=2, default=str))
        return 0
    if args.command == "verify-cohort":
        from .admit import verify_cohort
        result = verify_cohort()
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["member_checks_passed"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
