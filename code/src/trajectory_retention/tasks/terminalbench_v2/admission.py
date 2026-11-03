"""Verdicts, seals, the lane registry and the ledger of every task considered (no model calls).

Verdicts: ADMITTED only when every gate item and the restore check pass. A failed exclusion rule (hidden tests, network,
emulation, leak scan, starter solved, flaky tests) is EXCLUDED; a failure of the machinery or of the Mac (build,
collection, reference broken with and without network, restore check, grader isolation, disk headroom) is a HOLD with
its reason, never a scientific result. Committed evidence is compact: counts, hashes and pass flags, never hidden-test
names, grader diagnostics or reference content.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from trajectory_retention.contracts import to_jsonable
from trajectory_retention.registry import read_registry

from . import source as S
from .adapter import ADAPTER_ENTRY, SCHEMA, SEAL_SCHEMA, TerminalBenchV2TaskAdapter, evaluator_sources, sha256
from .package import ADMISSION, EVIDENCE, PACKAGES, WORKTREE, write_json

ROOT = WORKTREE / "benchmark_tasks"
EXCLUSION_ITEMS = ("hidden_tests", "network", "emulation", "leak_scan", "starter_fails", "flaky_tests")
# The user's rulings on tasks that pass the gate but are excluded anyway; they override the gate verdict.
USER_EXCLUSIONS = {
    "path-tracing": "user ruling 2026-10-07: the task ships /app/orig, the compiled original renderer; the tests "
                    "cannot call it, but reversing it is a route to the solution",
    "tune-mjcf": "user ruling 2026-10-07: the task ships /app/eval.py, which runs essentially the hidden tests' checks "
                 "(same physics within 1e-5, no NaN/Inf, at most 60% of the reference time), so the hidden tests are "
                 "not hidden and the failing count adds nothing",
}


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def task_ledgers() -> dict[str, dict]:
    out = {}
    for path in sorted((EVIDENCE / "tasks").glob("*/ledger.json")):
        row = json.loads(path.read_text())
        out[row["task"]] = row
    return out


def verdict(gate: dict | None, restore: dict | None) -> tuple[str, list[str]]:
    reasons = []
    if gate is None:
        return "HOLD", ["gate not run"]
    items = gate["items"]
    for name in EXCLUSION_ITEMS:
        if not items[name]["passed"]:
            detail = ""
            if name == "network":
                detail = items[name].get("classification") or ("instruction asks to fetch a remote resource"
                                                                if items[name]["instruction_remote_fetch"] else "")
                if "not a network finding" in detail:
                    continue
            if name == "hidden_tests":
                detail = f"{items[name]['count']} collected"
            if name == "leak_scan":
                detail = "failed on " + ", ".join(items[name]["failed_on"])
            reasons.append(f"{name}: {detail}".rstrip(": "))
    if reasons:
        return "EXCLUDED", reasons
    holds = [f"{k}: did not pass" for k, v in items.items() if not v["passed"] and k not in EXCLUSION_ITEMS]
    if not items["reference"]["passed"]:
        holds = [h for h in holds if not h.startswith("reference")] + [
            "reference: " + (items["network"].get("classification") or "fails offline")]
    if restore is None or not restore.get("passed"):
        failed = [k for k, v in (restore or {}).get("assertions", {}).items() if not v]
        holds.append("restore_check: " + (", ".join(failed) or (restore or {}).get("error", "not run")))
    return ("HOLD", holds) if holds else ("VALIDATED", [])


def compact_gate(gate: dict) -> dict:
    return {k: gate[k] for k in ("version", "task_id", "upstream_task", "package_sha256", "items", "grades", "timing",
                                 "reference_run", "reference_runtime", "provider_calls", "passed")}


def compact_restore(summary: dict) -> dict:
    keep = ("version", "task_id", "provider_calls", "detector_rule", "assertions", "runs", "restored_checkpoint",
            "restore_seconds", "probe_r_tool", "passed", "error")
    out = {k: summary.get(k) for k in keep}
    out["submission_grades"] = [{k: g.get(k) for k in ("ordinal", "status", "consecutive_same", "outcome", "share_passed",
                                                        "failing_count")} for g in summary.get("submission_grades", [])]
    return out


def write_evidence(task: str) -> Path:
    task_id = S.package_task_id(task)
    out = ADMISSION / "tasks" / task_id
    base = EVIDENCE / "tasks" / task
    gate = json.loads((base / "gate" / "gate.json").read_text())
    write_json(out / "gate.json", compact_gate(gate))
    restore = base / "restore-check" / "summary.json"
    if restore.is_file():
        write_json(out / "restore-check.json", compact_restore(json.loads(restore.read_text())))
    builds = {}
    for kind in ("native", "worker", "grader"):
        rec = json.loads((base / f"{kind}-build.json").read_text())
        builds[kind] = {k: rec.get(k) for k in ("result", "seconds", "tag", "bases", "dockerfile_sha256",
                                                "upstream_dockerfile_sha256", "extends_native_layers",
                                                "extends_worker_layers", "interpreter", "adopted", "adopted_from",
                                                "v1_adoption")}
        builds[kind]["image"] = {k: rec["image"].get(k) for k in ("id", "architecture", "os", "size", "workdir", "user")}
    write_json(out / "environment.json", builds)
    return out


def seal(task: str, static_row: dict) -> dict:
    task_id = S.package_task_id(task)
    package = PACKAGES / task_id
    out = write_evidence(task)
    gate = json.loads((out / "gate.json").read_text())
    restore = json.loads((out / "restore-check.json").read_text())
    if gate["package_sha256"] != sha256(package / "package.json") or not gate["passed"] or not restore["passed"]:
        raise ValueError(f"{task}: evidence does not bind the current package or a gate item failed")
    write_json(out / "static.json", static_row)
    gate_items = {**gate["items"], "restore_check": {"passed": restore["passed"], "version": restore["version"]}}
    evidence = [out / "gate.json", out / "restore-check.json", out / "environment.json", out / "static.json"]
    manifest = json.loads((package / "package.json").read_text())
    admission = {"schema_version": SEAL_SCHEMA, "admission": "ADMITTED", "task_id": task_id, "upstream_task": task,
                 "package_format": SCHEMA, "family_id": manifest["family_id"], "package": rel(package),
                 "package_sha256": sha256(package / "package.json"), "split_policy": S.SPLIT_POLICY,
                 "hidden_tests": gate["items"]["hidden_tests"]["count"], "runtime_class": "CAPTURED",
                 "gate": {k: {"passed": v["passed"]} for k, v in gate_items.items()},
                 "evaluator_sources": evaluator_sources(),
                 "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in evidence],
                 "verdict": "Admitted: every terminalbench-v2 gate item passed (no model calls)",
                 "not_established": ["DeepSeek V4.1 Flash solve rate and completion (the staged band screen)",
                                     "stall yield"]}
    write_json(out / "admission.json", admission)
    adapter = TerminalBenchV2TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "terminalbench-v2-seal")
    described = adapter.describe()
    if described.admission.value != "ADMITTED":
        raise ValueError("Seal did not verify")
    write_json(out / "task-record.json", to_jsonable(described))
    return {"task_id": task_id, "upstream_task": task, "family_id": manifest["family_id"], "package": rel(package),
            "task_record": rel(out / "task-record.json"), "attestation": rel(out / "admission.json"),
            "hidden_tests": admission["hidden_tests"]}


def seal_all(static: dict) -> dict:
    rows = {r["task"]: r for r in static["tasks"]}
    ledgers = task_ledgers()
    sealed = []
    for task, row in sorted(ledgers.items()):
        if task in USER_EXCLUSIONS:
            if (EVIDENCE / "tasks" / task / "gate" / "gate.json").is_file():
                out = write_evidence(task)
                for stale in ("admission.json", "task-record.json"):  # an exclusion revokes any earlier attestation
                    (out / stale).unlink(missing_ok=True)
                write_json(out / "user-exclusion.json", {"task": task, "verdict": "EXCLUDED",
                                                         "reason": USER_EXCLUSIONS[task]})
        elif row.get("status") == "VALIDATED":
            sealed.append(seal(task, rows[task]))
        elif (EVIDENCE / "tasks" / task / "gate" / "gate.json").is_file():
            write_evidence(task)
    entries = [{"task_id": r["task_id"], "family_id": r["family_id"], "adapter": ADAPTER_ENTRY, "admission": "ADMITTED",
                "package": r["package"], "task_record": r["task_record"], "attestation": r["attestation"],
                "upstream_task": r["upstream_task"], "package_format": SCHEMA, "hidden_tests": r["hidden_tests"],
                "runtime_class": "CAPTURED",
                "eligibility": {"development_and_calibration": True, "held_out_main_evaluation": True}}
               for r in sorted(sealed, key=lambda r: r["task_id"])]
    registry = ADMISSION / "registry.json"
    write_json(registry, {"registry_version": "task-admission.registry.v1", "lane": "terminalbench", "lane_version": "v2",
                          "source": {"url": S.SOURCE_URL, "revision": S.SOURCE_REVISION,
                                     "registry_dataset": S.REGISTRY_DATASET, "registry_revision": S.REGISTRY_REVISION},
                          "split_policy": S.SPLIT_POLICY,
                          "artifact_root": "benchmark_tasks (default read_registry root for this path)",
                          "tasks": entries})
    considered = []
    for task, row in sorted(rows.items()):
        led = ledgers.get(task, {})
        status = ("EXCLUDED" if task in USER_EXCLUSIONS else "ADMITTED" if led.get("status") == "VALIDATED"
                  else led.get("status") or ("EXCLUDED" if row["static_verdict"] == "EXCLUDED" else "HOLD"))
        reasons = ([USER_EXCLUSIONS[task]] if task in USER_EXCLUSIONS else led.get("reasons") or row["static_reasons"]
                   or (["not built: " + led["not_built"]] if led.get("not_built") else ["not built"]))
        considered.append({"task": task, "task_id": row["task_id"], "family_id": row["family_id"], "status": status,
                           "stage": led.get("stage", "static"), "reasons": reasons,
                           "hidden_tests_static": row["hidden_tests_static"],
                           "hidden_tests_collected": led.get("hidden_tests"),
                           "gate": led.get("gate_items")})
    counts = {s: sum(c["status"] == s for c in considered) for s in ("ADMITTED", "HOLD", "EXCLUDED")}
    write_json(ADMISSION / "ledger.json", {"schema_version": "terminalbench-v2.ledger.v1", "counts": counts,
                                           "tasks": considered})
    return {"sealed": [s["task_id"] for s in sealed], "registry": str(registry), "counts": counts}


def verify_loads() -> dict:
    registry = ADMISSION / "registry.json"
    entries = read_registry(registry)
    cases = []
    for task_id, entry in sorted(entries.items()):
        case = {"task_id": task_id, "loads": {}}
        for purpose in ("development", "calibration", "main"):
            try:
                entry.load(scratch_root=Path(tempfile.mkdtemp(prefix="tbv2-load-")), purpose=purpose)
                case["loads"][purpose] = True
            except ValueError as exc:
                case["loads"][purpose] = "refused: " + str(exc)[:200]
        adapter = TerminalBenchV2TaskAdapter(entry.path("package"), scratch_root=Path(tempfile.mkdtemp(prefix="tbv2-")))
        prompt = adapter.worker_prompt("task")
        upstream = S.CHECKOUT / adapter.upstream_task / "instruction.md"
        case["prompt_is_instruction_verbatim"] = prompt.text == upstream.read_text()
        case["protocol"] = adapter.native_submission_protocol["version"]
        case["repeated_failure_termination"] = adapter.native_submission_protocol["repeated_failure_termination"]
        cases.append(case)
    summary = {"registry_sha256": sha256(registry), "tasks": cases,
               "passed": all(all(v is True for v in c["loads"].values()) and c["prompt_is_instruction_verbatim"]
                             for c in cases)}
    write_json(EVIDENCE / "load-verification.json", summary)
    return summary
