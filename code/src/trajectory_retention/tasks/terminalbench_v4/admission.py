"""Verdicts, committed evidence, per-task ledgers, the lane registry and the seal (no model calls).

Adapted from the terminalbench-v2 lane (lanes never import each other), with recording and sealing split in two:

  record  (the build's step) writes, for every ratified task, the compact committed evidence and a per-task
          ``ledger.json`` under benchmark_tasks/admission/terminalbench-v4/tasks/<task_id>/, the lane ``ledger.json``
          and ``registry.json``. A task that passed every gate item is VALIDATED and listed as awaiting the seal; the
          registry admits nothing until the seal.
  seal    (the user's step) turns each VALIDATED task into ADMITTED: it checks that the committed evidence still binds
          the current package and evaluator sources, writes ``admission.json`` and ``task-record.json``, and rewrites
          the registry with the admitted entries and the ledgers with the new status.

Verdicts: VALIDATED only when every gate item and the restore check pass. A failed exclusion rule (hidden tests, network,
emulation, starter solved, flaky tests, a leak) is EXCLUDED; a failure of the machinery or of this Mac (build, disk,
collection, x86-only binaries, reference broken with and without network, restore check, grader isolation, start
fidelity, a leak finding still awaiting review) is a HOLD with its reason, never a scientific result. Committed evidence
is compact: counts, hashes and pass flags, never hidden-test names, grader diagnostics or reference content.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from trajectory_retention.contracts import to_jsonable
from trajectory_retention.registry import read_registry

from . import source as S
from .adapter import (ADAPTER_ENTRY, SCHEMA, SEAL_SCHEMA, TerminalBenchV4TaskAdapter, evaluator_sources, sha256,
                      strip_canary)
from .package import ADMISSION, EVIDENCE, PACKAGES, WORKTREE, write_json

ROOT = WORKTREE / "benchmark_tasks"
EXCLUSION_ITEMS = ("hidden_tests", "network", "emulation", "starter_fails", "flaky_tests")
LEDGER_SCHEMA = "terminalbench-v4.ledger.v1"


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def local_ledger(task: str) -> dict | None:
    path = EVIDENCE / "tasks" / task / "ledger.json"
    return json.loads(path.read_text()) if path.is_file() else None


def verdict(gate: dict | None, restore: dict | None) -> tuple[str, list[str]]:
    if gate is None:
        return "HOLD", ["gate not run"]
    if gate.get("docker_disk"):  # the Docker VM disk, not the task: never an exclusion
        disk = gate["docker_disk"].get("disk") or {}
        return "HOLD", ["docker disk: " + "; ".join(gate["docker_disk"]["seen_in"])
                        + (f" (Docker VM disk: {disk['free_bytes'] / 1e9:.1f} GB free counting root-reserved blocks, "
                           f"{disk['non_root_available_bytes'] / 1e9:.1f} GB available to non-root users)" if disk
                           else "")]
    items = gate["items"]
    reasons = []
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
            reasons.append(f"{name}: {detail}".rstrip(": "))
    leak = items["leak_scan"]
    if not leak["passed"] and (leak["counts"]["hashes"] or leak["counts"]["locations"]
                               or any(r["verdict"] not in (None, "not_a_leak") for r in leak["reviews"])):
        reasons.append("leak_scan: failed on " + ", ".join(leak["failed_on"]))
    if reasons:
        return "EXCLUDED", reasons
    holds = [f"{k}: did not pass" for k, v in items.items()
             if not v["passed"] and k not in EXCLUSION_ITEMS and k not in ("leak_scan", "reference")]
    if not leak["passed"]:
        holds.append(f"leak_scan: {leak['counts']['unresolved']} finding(s) await a recorded review "
                     f"({', '.join(leak['failed_on'])})")
    if not items["reference"]["passed"]:
        holds.append("reference: " + (items["network"].get("classification") or "fails offline"))
    if restore is None or not restore.get("passed"):
        failed = [k for k, v in (restore or {}).get("assertions", {}).items() if not v]
        holds.append("restore_check: " + (", ".join(failed) or (restore or {}).get("error", "not run")))
    return ("HOLD", holds) if holds else ("VALIDATED", [])


def compact_gate(gate: dict) -> dict:
    return {k: gate.get(k) for k in ("version", "task_id", "upstream_task", "package_sha256", "items", "grades", "timing",
                                     "reference_run", "reference_runtime", "provider_calls", "disk_at_start",
                                     "disk_at_end", "worker_write_probe", "docker_disk", "stopped", "passed")}


def compact_restore(summary: dict) -> dict:
    keep = ("version", "task_id", "provider_calls", "detector_rule", "assertions", "runs", "restored_checkpoint",
            "restore_seconds", "probe_r_tool", "passed", "error")
    out = {k: summary.get(k) for k in keep}
    out["submission_grades"] = [{k: g.get(k) for k in ("ordinal", "status", "consecutive_same", "outcome", "share_passed",
                                                        "failing_count")} for g in summary.get("submission_grades", [])]
    return out


def compact_build(rec: dict) -> dict:
    out = {k: rec.get(k) for k in ("result", "seconds", "tag", "bases", "official_image", "dockerfile_sha256",
                                   "upstream_dockerfile_sha256", "context_identical_to_upstream", "extends_native_layers",
                                   "python3", "disk_before", "disk_after", "returncode")}
    if rec.get("image"):
        out["image"] = {k: rec["image"].get(k) for k in ("id", "architecture", "os", "size", "workdir", "user")}
    if rec.get("result") != "built":
        out["log_tail"] = (rec.get("log_tail") or "")[-1500:]
    return out


def write_task_records(task: str, static_row: dict) -> dict:
    """Committed per-task evidence and ledger.json from the local evidence (never hidden-test names)."""
    task_id = S.package_task_id(task)
    out = ADMISSION / "tasks" / task_id
    base = EVIDENCE / "tasks" / task
    led = local_ledger(task) or {}
    write_json(out / "static.json", static_row)
    evidence = [out / "static.json"]
    gate_path, restore_path = base / "gate" / "gate.json", base / "restore-check" / "summary.json"
    if gate_path.is_file():
        write_json(out / "gate.json", compact_gate(json.loads(gate_path.read_text())))
        evidence.append(out / "gate.json")
    if restore_path.is_file():
        write_json(out / "restore-check.json", compact_restore(json.loads(restore_path.read_text())))
        evidence.append(out / "restore-check.json")
    builds = {kind: compact_build(json.loads((base / f"{kind}-build.json").read_text()))
              for kind in ("native", "worker", "verifier") if (base / f"{kind}-build.json").is_file()}
    if builds:
        write_json(out / "environment.json", builds)
        evidence.append(out / "environment.json")
    package = PACKAGES / task_id
    manifest = json.loads((package / "package.json").read_text()) if (package / "package.json").is_file() else None
    status = led.get("status") or static_row["static_verdict"].replace("CANDIDATE", "HOLD")
    reasons = led.get("reasons") or static_row["static_reasons"] or ["not built"]
    ledger = {"schema_version": LEDGER_SCHEMA, "task": task, "task_id": task_id, "family_id": S.family_id(task),
              "lane": S.LANE, "source": {"url": S.SOURCE_URL, "tag": S.SOURCE_TAG, "revision": S.SOURCE_REVISION},
              "ratification": S.RATIFICATION,
              "status": status, "admission": "awaiting the user's seal" if status == "VALIDATED" else None,
              "stage": led.get("stage", "static"), "reasons": reasons,
              "hidden_tests_static": static_row["hidden_tests_static"], "hidden_tests_collected": led.get("hidden_tests"),
              "gate": led.get("gate_items"),
              "package": rel(package) if manifest else None,
              "package_sha256": sha256(package / "package.json") if manifest else None,
              "deviations": manifest["deviations"] if manifest else None,
              "static_flags": {"solve_time_installs": static_row["solve_time_installs"],
                               "x86_only_binary_count": static_row["x86_only_binary_count"],
                               "x86_only_binaries": static_row["x86_only_binaries"][:5]},
              "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in evidence],
              "docker_disk": led.get("docker_disk") or led.get("docker_disk_runtime") or None, "provider_calls": 0}
    write_json(out / "ledger.json", ledger)
    return ledger


def write_lane_records(static: dict, *, sealed: list[dict] | None = None) -> dict:
    rows = {r["task"]: r for r in static["tasks"]}
    ledgers = [json.loads((ADMISSION / "tasks" / S.package_task_id(t) / "ledger.json").read_text())
               for t in S.RATIFIED_TASKS]
    sealed = sealed or []
    entries = [{"task_id": r["task_id"], "family_id": r["family_id"], "adapter": ADAPTER_ENTRY, "admission": "ADMITTED",
                "package": r["package"], "task_record": r["task_record"], "attestation": r["attestation"],
                "upstream_task": r["upstream_task"], "package_format": SCHEMA, "hidden_tests": r["hidden_tests"],
                "runtime_class": "CAPTURED",
                "eligibility": {"development_and_calibration": True, "held_out_main_evaluation": True}}
               for r in sorted(sealed, key=lambda r: r["task_id"])]
    awaiting = [led["task_id"] for led in ledgers if led["status"] == "VALIDATED"]
    write_json(ADMISSION / "registry.json", {
        "registry_version": "task-admission.registry.v1", "lane": "terminalbench", "lane_version": "v4",
        "source": {"url": S.SOURCE_URL, "tag": S.SOURCE_TAG, "revision": S.SOURCE_REVISION},
        "split_policy": S.SPLIT_POLICY, "ratification": S.RATIFICATION,
        "artifact_root": "benchmark_tasks (default read_registry root for this path)",
        "seal": "registry entries appear only after `python -m trajectory_retention.tasks.terminalbench_v4 seal` (the "
                "user's step)",
        "awaiting_seal": awaiting, "tasks": entries})
    counts = {s: sum(led["status"] == s for led in ledgers) for s in ("ADMITTED", "VALIDATED", "HOLD", "EXCLUDED")}
    write_json(ADMISSION / "ledger.json", {
        "schema_version": LEDGER_SCHEMA, "lane": S.LANE, "ratification": S.RATIFICATION, "counts": counts,
        "tasks": [{k: led[k] for k in ("task", "task_id", "family_id", "status", "stage", "reasons",
                                        "hidden_tests_static", "hidden_tests_collected", "gate", "package_sha256")}
                  for led in ledgers]})
    return {"counts": counts, "awaiting_seal": awaiting, "admitted": [e["task_id"] for e in entries]}


def record_all(static: dict) -> dict:
    """The build's step: committed evidence and ledgers for every ratified task; nothing is sealed."""
    for row in static["tasks"]:
        write_task_records(row["task"], row)
    return write_lane_records(static)


def seal(task: str) -> dict:
    task_id = S.package_task_id(task)
    package = PACKAGES / task_id
    out = ADMISSION / "tasks" / task_id
    ledger = json.loads((out / "ledger.json").read_text())
    gate = json.loads((out / "gate.json").read_text())
    restore = json.loads((out / "restore-check.json").read_text())
    if ledger["status"] != "VALIDATED" or ledger["package_sha256"] != sha256(package / "package.json"):
        raise ValueError(f"{task}: not VALIDATED, or the ledger binds a different package")
    if gate["package_sha256"] != sha256(package / "package.json") or not gate["passed"] or not restore["passed"]:
        raise ValueError(f"{task}: evidence does not bind the current package or a gate item failed")
    for binding in ledger["evidence"]:
        if sha256(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"{task}: committed evidence changed since it was recorded: {binding['path']}")
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
                 "deviations": manifest["deviations"],
                 "verdict": "Admitted: every terminalbench-v4 gate item passed (no model calls); sealed by the user",
                 "not_established": ["DeepSeek V4.1 Flash solve rate and completion (the staged band screen)",
                                     "stall yield"]}
    write_json(out / "admission.json", admission)
    adapter = TerminalBenchV4TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "terminalbench-v4-seal")
    described = adapter.describe()
    if described.admission.value != "ADMITTED":
        raise ValueError("Seal did not verify")
    write_json(out / "task-record.json", to_jsonable(described))
    ledger.update(status="ADMITTED", admission="sealed", attestation=rel(out / "admission.json"))
    write_json(out / "ledger.json", ledger)
    return {"task_id": task_id, "upstream_task": task, "family_id": manifest["family_id"], "package": rel(package),
            "task_record": rel(out / "task-record.json"), "attestation": rel(out / "admission.json"),
            "hidden_tests": admission["hidden_tests"]}


def seal_all(static: dict) -> dict:
    """The user's step: seal every VALIDATED task and rewrite the registry and lane ledger."""
    sealed = []
    for task in S.RATIFIED_TASKS:
        ledger = json.loads((ADMISSION / "tasks" / S.package_task_id(task) / "ledger.json").read_text())
        if ledger["status"] == "VALIDATED":
            sealed.append(seal(task))
        elif ledger["status"] == "ADMITTED":
            out = ADMISSION / "tasks" / S.package_task_id(task)
            sealed.append({"task_id": ledger["task_id"], "upstream_task": task, "family_id": ledger["family_id"],
                           "package": ledger["package"], "task_record": rel(out / "task-record.json"),
                           "attestation": rel(out / "admission.json"),
                           "hidden_tests": json.loads((out / "admission.json").read_text())["hidden_tests"]})
    summary = write_lane_records(static, sealed=sealed)
    return {"sealed": [s["task_id"] for s in sealed], **summary}


def verify_loads() -> dict:
    registry = ADMISSION / "registry.json"
    entries = read_registry(registry)
    cases = []
    for task_id, entry in sorted(entries.items()):
        case = {"task_id": task_id, "loads": {}}
        for purpose in ("development", "calibration", "main"):
            try:
                entry.load(scratch_root=Path(tempfile.mkdtemp(prefix="tbv4-load-")), purpose=purpose)
                case["loads"][purpose] = True
            except ValueError as exc:
                case["loads"][purpose] = "refused: " + str(exc)[:200]
        adapter = TerminalBenchV4TaskAdapter(entry.path("package"), scratch_root=Path(tempfile.mkdtemp(prefix="tbv4-")))
        prompt = adapter.worker_prompt("task")
        upstream = S.task_dir(adapter.upstream_task) / "instruction.md"
        case["prompt_is_harbor_delivered_instruction"] = prompt.text == strip_canary(upstream.read_text())
        case["protocol"] = adapter.native_submission_protocol["version"]
        case["repeated_failure_termination"] = adapter.native_submission_protocol["repeated_failure_termination"]
        cases.append(case)
    summary = {"registry_sha256": sha256(registry), "tasks": cases,
               "passed": all(all(v is True for v in c["loads"].values()) and c["prompt_is_harbor_delivered_instruction"]
                             for c in cases)}
    write_json(EVIDENCE / "load-verification.json", summary)
    return summary
