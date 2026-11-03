"""CLI: python -m trajectory_retention.tasks.swesmith_long <command>."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pipeline

SELECTION = pipeline.EVIDENCE / "selection.json"


def _rows(ids):
    import pandas as pd
    frame = pd.read_pickle(pipeline.CACHE / "combine.pkl")
    frame = frame[frame.instance_id.isin(ids)]
    return {r["instance_id"]: {k: (list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v) for k, v in r.items()}
            for r in frame.to_dict("records")}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="swesmith_long")
    sub = parser.add_subparsers(dest="command", required=True)
    fam = sub.add_parser("family")
    fam.add_argument("repo_key")
    fam.add_argument("--force", action="store_true")
    inst = sub.add_parser("instance")
    inst.add_argument("instance_id")
    rc = sub.add_parser("runner-check")
    rc.add_argument("task_id")
    fi = sub.add_parser("family-instances", help="package/validate a family's seeded candidates with the reserve rule")
    fi.add_argument("repo_key")
    fr = sub.add_parser("family-runner-checks")
    fr.add_argument("repo_key")
    se = sub.add_parser("seal", help="seal every validated task, write registry and cohort")
    tm = sub.add_parser("timing", help="serial start/reference grade timing for sealed tasks")
    tm.add_argument("--label", required=True)
    tm.add_argument("--split", choices=("calibration", "held_out"))
    v2f = sub.add_parser("v2-family", help="v2 (native start): image needle scan, build+validate, runner checks")
    v2f.add_argument("repo_key")
    v2r = sub.add_parser("v2-rescan", help="recompute image needles + withheld leak scans; optional runner re-checks")
    v2r.add_argument("repo_key")
    v2r.add_argument("--runner-recheck", action="store_true")
    sub.add_parser("v2-seal")
    v2v = sub.add_parser("v2-verify-loads")
    v2v.add_argument("--attempt", required=True)
    v3f = sub.add_parser("v3-family", help="v3 (Git start + native template): image, scans, build+validate, runner checks")
    v3f.add_argument("repo_key")
    v3f.add_argument("--reuse-image", action="store_true", help="reuse an already recorded v3 image for this family")
    v3f.add_argument("--only", nargs="*", help="restrict to these v2 task ids")
    sub.add_parser("v3-seal")
    v3v = sub.add_parser("v3-verify-loads")
    v3v.add_argument("--attempt", required=True)
    v4s = sub.add_parser("v4-select", help="v4: continue the frozen family order, apply I1/I2, draw candidates")
    v4s.add_argument("--last-position", type=int, required=True)
    v4f = sub.add_parser("v4-family", help="v4: build one drawn calibration family through the v1->v2->v3 pipeline")
    v4f.add_argument("repo_key")
    v4e = sub.add_parser("v4-seal", help="v4: seal validated new tasks; write the v4 registry and cohort")
    v4e.add_argument("families", nargs="+", help="repo keys of the admitted-prefix families, in selection order")
    v4v = sub.add_parser("v4-verify-loads")
    v4v.add_argument("--attempt", required=True)
    sub.add_parser("v5-select", help="v5: rank the published hard band, apply filters, I2 and comment-leak verdicts")
    v5b = sub.add_parser("v5-build", help="v5: build eligible instances of one family in rank order (family cap)")
    v5b.add_argument("repo_key")
    v5e = sub.add_parser("v5-seal", help="v5: stall rationales, seals, v5 registry and cohort")
    v5e.add_argument("families", nargs="+", help="repo keys of the built families")
    v5v = sub.add_parser("v5-verify-loads")
    v5v.add_argument("--attempt", required=True)
    vl = sub.add_parser("verify-loads")
    vl.add_argument("--attempt", required=True)
    args = parser.parse_args(argv)
    selection = json.loads(SELECTION.read_text())
    families = {f["repo_key"]: f for f in selection["families"]}
    if args.command == "family-instances":
        family_json = json.loads((pipeline.EVIDENCE / "families" / args.repo_key / "family.json").read_text())
        chosen = families[args.repo_key]
        ledger = []
        if family_json["status"] != "READY":
            for iid in chosen["candidates"]:
                ledger.append({"instance_id": iid, "role": "candidate", "status": "HOLD",
                               "reason": "family environment: " + str(family_json.get("reason"))})
        else:
            family_json["profile"] = chosen["profile"]
            rows = _rows(chosen["candidates"] + chosen["reserves"])
            reserves = list(chosen["reserves"])
            queue = [(iid, "candidate") for iid in chosen["candidates"]]
            while queue:
                iid, role = queue.pop(0)
                record = pipeline.package_instance(rows[iid], family_json)
                ledger.append({"instance_id": iid, "task_id": record["task_id"], "role": role, "status": record["status"],
                               "reason": record.get("reason"), "counts": record.get("counts"),
                               "grade_seconds": (record.get("validation") or {}).get("grade_seconds")})
                if record["status"] != "VALIDATED" and reserves:
                    queue.append((reserves.pop(0), "reserve"))
            for iid in reserves:
                ledger.append({"instance_id": iid, "role": "reserve", "status": "NOT_USED"})
        pipeline.write_json(pipeline.EVIDENCE / "candidates" / f"{args.repo_key}.json", ledger)
        print(json.dumps(ledger, indent=1, default=str))
        return
    if args.command == "family-runner-checks":
        from .runner_check import run_check
        ledger_path = pipeline.EVIDENCE / "candidates" / f"{args.repo_key}.json"
        ledger = json.loads(ledger_path.read_text())
        for row in ledger:
            if row["status"] == "VALIDATED":
                summary = run_check(pipeline.PACKAGES / row["task_id"],
                                    pipeline.EVIDENCE / "tasks" / row["task_id"] / "runner-check")
                row["runner_check"] = "passed" if summary.get("passed") else "failed"
                if not summary.get("passed"):
                    row["status"] = "HOLD"
                    failed = [k for k, v in summary.get("assertions", {}).items() if not v]
                    row["reason"] = "runner check failed: " + (summary.get("error") or ", ".join(failed))
        pipeline.write_json(ledger_path, ledger)
        print(json.dumps([(r.get("task_id"), r["status"], r.get("runner_check"), r.get("reason")) for r in ledger], indent=1))
        return
    if args.command == "seal":
        from . import admission
        import hashlib
        selection_sha = hashlib.sha256(SELECTION.read_bytes()).hexdigest()
        sealed = []
        for key in families:
            ledger_path = pipeline.EVIDENCE / "candidates" / f"{key}.json"
            if not ledger_path.is_file():
                continue
            for row in json.loads(ledger_path.read_text()):
                if row["status"] == "VALIDATED" and row.get("runner_check") == "passed":
                    sealed.append(admission.seal(row["task_id"], selection_sha256=selection_sha))
        import shutil
        shutil.copy2(SELECTION, pipeline.ADMISSION / "selection.json")
        outcomes = []
        for key, fam in families.items():
            fam_path = pipeline.EVIDENCE / "families" / key / "family.json"
            fam_record = json.loads(fam_path.read_text()) if fam_path.is_file() else {"status": "NOT_RUN"}
            ledger_path = pipeline.EVIDENCE / "candidates" / f"{key}.json"
            outcomes.append({"repo_key": key, "family_id": fam["family_id"], "split": fam["split"], "order": fam["order"],
                             "environment_status": fam_record["status"], "environment_reason": fam_record.get("reason"),
                             "candidates": json.loads(ledger_path.read_text()) if ledger_path.is_file() else None})
        pipeline.write_json(pipeline.ADMISSION / "candidates.json", {"selection_sha256": selection_sha, "families": outcomes})
        registry = admission.write_registry(sealed)
        cohort = admission.write_cohort(sealed, registry)
        print(json.dumps({"sealed": len(sealed), "registry": str(registry), "cohort": str(cohort)}, indent=1))
        return
    if args.command == "timing":
        import os, shutil, subprocess, time
        from .adapter import SWESmithLongTaskAdapter
        registry = json.loads((pipeline.ADMISSION / "registry.json").read_text())
        out = pipeline.EVIDENCE / "timing" / args.label
        out.mkdir(parents=True, exist_ok=False)
        rows = []
        for entry in registry["tasks"]:
            if args.split and entry["split"] != args.split:
                continue
            package = pipeline.WORKTREE / "benchmark_tasks" / entry["package"]
            adapter = SWESmithLongTaskAdapter(package, scratch_root=pipeline.CACHE / "grade-scratch")
            work = pipeline.CACHE / "timing" / args.label / entry["task_id"]
            if work.exists():
                shutil.rmtree(work)
            adapter.materialize(work / "start")
            reference = work / "reference"
            shutil.copytree(work / "start", reference, symlinks=True)
            patch = (package / "private/bug.patch").read_text()
            subprocess.run(["git", "apply", "-R", "-"], cwd=reference, input=patch.encode(), check=True,
                           capture_output=True, env={**os.environ, "GIT_CEILING_DIRECTORIES": str(reference.parent)})
            row = {"task_id": entry["task_id"], "load_average_before": os.getloadavg()}
            for label, tree in (("start", work / "start"), ("reference", reference)):
                t0 = time.monotonic()
                evaluation = adapter.grade(tree, stage_id="repair", private_output_dir=out / entry["task_id"] / label)
                row[label] = {"host_seconds": round(time.monotonic() - t0, 2),
                              "container_seconds": evaluation.details["grade_wall_seconds"],
                              "score": evaluation.score, "valid": evaluation.valid, "full_solve": evaluation.full_solve}
            rows.append(row)
            print(json.dumps(row), flush=True)
        pipeline.write_json(pipeline.ADMISSION / f"timing-{args.label}.json",
                            {"label": args.label, "mode": "serial, one grade at a time", "tasks": rows})
        return
    if args.command == "v2-family":
        from . import pipeline_v2
        from .adapter_v2 import SWESmithLongV2TaskAdapter
        from .runner_check import run_check
        registry = json.loads((pipeline.ADMISSION / "registry.json").read_text())
        v1_ids = sorted(t["task_id"] for t in registry["tasks"]
                        if t["task_id"].startswith("swesmith-" + pipeline.slug(args.repo_key) + "-"))
        pipeline_v2.family_image_needles(args.repo_key, v1_ids)
        ledger = []
        for v1_id in v1_ids:
            record = pipeline_v2.build(v1_id)
            row = {"task_id": record["task_id"], "predecessor": v1_id, "status": record["status"],
                   "reason": record.get("reason"), "counts": record.get("counts"),
                   "grade_seconds": (record.get("validation") or {}).get("grade_seconds")}
            if record["status"] == "VALIDATED":
                summary = run_check(pipeline_v2.PACKAGES / record["task_id"],
                                    pipeline_v2.EVIDENCE_V2 / "tasks" / record["task_id"] / "runner-check",
                                    adapter_cls=SWESmithLongV2TaskAdapter)
                row["runner_check"] = "passed" if summary.get("passed") else "failed"
                if not summary.get("passed"):
                    row["status"] = "HOLD"
                    row["reason"] = "runner check failed: " + (summary.get("error") or ", ".join(
                        k for k, v in summary.get("assertions", {}).items() if not v))
            ledger.append(row)
        pipeline.write_json(pipeline_v2.EVIDENCE_V2 / "candidates" / f"{args.repo_key}.json", ledger)
        print(json.dumps(ledger, indent=1, default=str))
        return
    if args.command == "v2-rescan":
        from . import pipeline_v2
        from .adapter_v2 import SWESmithLongV2TaskAdapter
        from .runner_check import run_check
        registry = json.loads((pipeline.ADMISSION / "registry.json").read_text())
        v1_ids = sorted(t["task_id"] for t in registry["tasks"]
                        if t["task_id"].startswith("swesmith-" + pipeline.slug(args.repo_key) + "-"))
        pipeline_v2.family_image_needles(args.repo_key, v1_ids)
        ledger_path = pipeline_v2.EVIDENCE_V2 / "candidates" / f"{args.repo_key}.json"
        ledger = json.loads(ledger_path.read_text())
        for row in ledger:
            record = pipeline_v2.rescan(row["task_id"])
            if "validation" not in record:
                continue
            if not record["validation"]["passed"]:
                row.update(status="HOLD", reason="; ".join(record["validation"]["failures"]))
                continue
            if args.runner_recheck or row.get("runner_check") != "passed":
                summary = run_check(pipeline_v2.PACKAGES / row["task_id"],
                                    pipeline_v2.EVIDENCE_V2 / "tasks" / row["task_id"] / "runner-check",
                                    adapter_cls=SWESmithLongV2TaskAdapter)
                row["runner_check"] = "passed" if summary.get("passed") else "failed"
            row.update(status="VALIDATED" if row.get("runner_check") == "passed" else "HOLD",
                       reason=None if row.get("runner_check") == "passed" else "runner check failed")
        pipeline.write_json(ledger_path, ledger)
        print(json.dumps([(r["task_id"], r["status"], r.get("runner_check"), r.get("reason")) for r in ledger], indent=1))
        return
    if args.command == "v2-seal":
        from . import admission_v2, pipeline_v2
        sealed, outcomes = [], []
        for key in families:
            path = pipeline_v2.EVIDENCE_V2 / "candidates" / f"{key}.json"
            if not path.is_file():
                continue
            ledger = json.loads(path.read_text())
            outcomes.append({"repo_key": key, "family_id": families[key]["family_id"], "split": families[key]["split"],
                             "tasks": ledger})
            for row in ledger:
                if row["status"] == "VALIDATED" and row.get("runner_check") == "passed":
                    sealed.append(admission_v2.seal(row["task_id"]))
        pipeline.write_json(pipeline_v2.ADMISSION / "candidates.json", {"families": outcomes})
        registry = admission_v2.write_registry(sealed)
        cohort = admission_v2.write_cohort(sealed, registry)
        print(json.dumps({"sealed": len(sealed), "registry": str(registry), "cohort": str(cohort)}, indent=1))
        return
    if args.command == "v2-verify-loads":
        from . import admission_v2
        summary = admission_v2.verify_loads(args.attempt)
        print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))
        return
    if args.command == "v3-family":
        from . import pipeline_v3
        from .adapter_v3 import SWESmithLongV3TaskAdapter
        from .environment import image_id
        from .runner_check import run_check
        image_path = pipeline_v3.EVIDENCE_V3 / "families" / args.repo_key / "image.json"
        family = json.loads(image_path.read_text()) if args.reuse_image and image_path.is_file() else None
        if family is None or image_id(family["tag"]) != family["image_id"]:
            family = pipeline_v3.build_family_image(args.repo_key)
            family["scans"] = pipeline_v3.family_scans(args.repo_key, family["image_id"])
            pipeline.write_json(image_path, family)
        ledger_path = pipeline_v3.EVIDENCE_V3 / "candidates" / f"{args.repo_key}.json"
        ledger = {r["predecessor"]: r for r in (json.loads(ledger_path.read_text()) if ledger_path.is_file() else [])}
        if not family["image_ok"]:
            raise SystemExit(f"v3 image check failed for {args.repo_key}: {image_path}")
        for v2_id in pipeline_v3.family_packages(args.repo_key):
            if args.only and v2_id not in args.only:
                continue
            record = pipeline_v3.build(v2_id, family)
            row = {"task_id": record["task_id"], "predecessor": v2_id, "status": record["status"],
                   "reason": record.get("reason"), "grade_seconds": (record.get("validation") or {}).get("grade_seconds"),
                   "grades_equal_v2": (record.get("v2_comparison") or {}).get("all_equal")}
            if record["status"] == "VALIDATED":
                summary = run_check(pipeline_v3.PACKAGES / record["task_id"],
                                    pipeline_v3.EVIDENCE_V3 / "tasks" / record["task_id"] / "runner-check",
                                    adapter_cls=SWESmithLongV3TaskAdapter)
                row["runner_check"] = "passed" if summary.get("passed") else "failed"
                if not summary.get("passed"):
                    row["status"] = "HOLD"
                    row["reason"] = "runner check failed: " + (summary.get("error") or ", ".join(
                        k for k, v in summary.get("assertions", {}).items() if not v))
            ledger[v2_id] = row
            pipeline.write_json(ledger_path, sorted(ledger.values(), key=lambda r: r["predecessor"]))
            print(json.dumps(row, default=str), flush=True)
        return
    if args.command == "v3-seal":
        from . import admission_v3, pipeline_v3
        sealed, outcomes = [], []
        for key in families:
            path = pipeline_v3.EVIDENCE_V3 / "candidates" / f"{key}.json"
            if not path.is_file():
                continue
            ledger = json.loads(path.read_text())
            outcomes.append({"repo_key": key, "family_id": families[key]["family_id"], "split": families[key]["split"],
                             "tasks": ledger})
            for row in ledger:
                if row["status"] == "VALIDATED" and row.get("runner_check") == "passed":
                    sealed.append(admission_v3.seal(row["task_id"]))
        pipeline.write_json(pipeline_v3.ADMISSION / "candidates.json", {"families": outcomes})
        registry = admission_v3.write_registry(sealed)
        cohort = admission_v3.write_cohort(sealed, registry)
        print(json.dumps({"sealed": len(sealed), "registry": str(registry), "cohort": str(cohort)}, indent=1))
        return
    if args.command == "v3-verify-loads":
        from . import admission_v3
        summary = admission_v3.verify_loads(args.attempt)
        print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))
        return
    if args.command == "v4-select":
        from . import pipeline_v4
        record = pipeline_v4.write_selection(args.last_position)
        print(json.dumps({"pending_review": record["pending_review"],
                          "families": [(f["order"], f["repo_key"], f["split"], f["v4_status"], f.get("candidates"),
                                        f.get("reserves")) for f in record["families"]]}, indent=1))
        return
    if args.command == "v4-family":
        from . import pipeline_v4
        ledger = pipeline_v4.process_family(args.repo_key)
        print(json.dumps([{k: r.get(k) for k in ("instance_id", "role", "status", "stage", "task_id")} for r in ledger],
                         indent=1))
        return
    if args.command == "v4-seal":
        from . import admission_v4
        admission_v4.write_ledgers(args.families)
        stop = json.loads(admission_v4.write_stop_rule(args.families).read_text())
        prefix = {r["repo_key"] for r in stop["families"] if r["in_admitted_prefix"]}
        sealed = []
        for key in args.families:
            if key not in prefix:
                continue
            for row in json.loads((admission_v4.ADMISSION / "candidates" / f"{key}.json").read_text()):
                if row["status"] == "VALIDATED" and row.get("runner_check") == "passed":
                    sealed.append(admission_v4.seal(row["task_id"]))
        registry = admission_v4.write_registry(sealed)
        cohort = admission_v4.write_cohort(registry)
        print(json.dumps({"sealed": len(sealed), "registry": str(registry), "cohort": str(cohort)}, indent=1))
        return
    if args.command == "v4-verify-loads":
        from . import admission_v4
        summary = admission_v4.verify_loads(args.attempt)
        print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))
        return
    if args.command == "v5-select":
        from . import pipeline_v5
        record = pipeline_v5.write_selection()
        print(json.dumps(record["counts"], indent=1))
        return
    if args.command == "v5-build":
        from . import pipeline_v5
        from .selection_v5 import PER_FAMILY_CAP
        selection = json.loads(pipeline_v5.SELECTION.read_text())
        queue = [s["instance_id"] for s in selection["ranked"] if s["repo_key"] == args.repo_key and s["status"] == "ELIGIBLE"]
        ledger_path = pipeline_v5.EVIDENCE_V5 / "candidates" / f"{args.repo_key}.json"
        while True:
            ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
            queue = [i for i in queue if i not in {r["instance_id"] for r in ledger}]
            if sum(r["status"] == "VALIDATED" for r in ledger) >= PER_FAMILY_CAP or not queue:
                break
            pipeline_v5.process_instances(args.repo_key, [queue[0]])
        return
    if args.command == "v5-seal":
        from . import admission_v5, pipeline_v5
        admission_v5.write_ledgers(args.families)
        sealed = []
        for key in args.families:
            for row in json.loads((admission_v5.ADMISSION / "candidates" / f"{key}.json").read_text()):
                if row["status"] == "VALIDATED" and row.get("runner_check") == "passed":
                    pipeline_v5.stall_rationale(row["task_id"])
        admission_v5.write_stop_rule(args.families)
        for key in args.families:
            for row in json.loads((admission_v5.ADMISSION / "candidates" / f"{key}.json").read_text()):
                if row["status"] == "VALIDATED" and row.get("runner_check") == "passed":
                    sealed.append(admission_v5.seal(row["task_id"]))
        registry = admission_v5.write_registry(sealed)
        admission_v5.egg_info_new_tasks([r["task_id"] for r in sealed])
        egg = json.loads((pipeline_v5.LEAKS / "egg-info.json").read_text()) if (pipeline_v5.LEAKS / "egg-info.json").is_file() else None
        cohort = admission_v5.write_cohort(registry, admission_v5.egg_info_summary(egg))
        print(json.dumps({"sealed": len(sealed), "registry": str(registry), "cohort": str(cohort)}, indent=1))
        return
    if args.command == "v5-verify-loads":
        from . import admission_v5
        summary = admission_v5.verify_loads(args.attempt)
        print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))
        return
    if args.command == "verify-loads":
        from . import admission
        summary = admission.verify_loads(args.attempt)
        print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=1))
        return
    if args.command == "runner-check":
        from .runner_check import run_check
        summary = run_check(pipeline.PACKAGES / args.task_id, pipeline.EVIDENCE / "tasks" / args.task_id / "runner-check")
        print(json.dumps({k: summary.get(k) for k in ("task_id", "passed", "error", "assertions", "source_grades", "runs")},
                         indent=1, default=str))
        return
    if args.command == "family":
        record = pipeline.prepare_family(families[args.repo_key], force=args.force)
        print(json.dumps({k: record.get(k) for k in ("repo_key", "status", "reason", "reference_stable_pass",
                                                       "reference_flaky", "reference_runs")}, indent=1, default=str))
    elif args.command == "instance":
        key = args.instance_id.rsplit(".", 1)[0]
        family = json.loads((pipeline.EVIDENCE / "families" / key / "family.json").read_text())
        family["profile"] = families[key]["profile"]
        row = _rows([args.instance_id])[args.instance_id]
        record = pipeline.package_instance(row, family)
        summary = {k: record.get(k) for k in ("task_id", "status", "reason", "counts")}
        if "validation" in record:
            summary["checks"] = record["validation"]["checks"]
            summary["grades"] = record["validation"]["grades"]
            summary["partial"] = record["validation"]["partial_controls"]
            summary["grade_seconds"] = record["validation"]["grade_seconds"]
        print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
