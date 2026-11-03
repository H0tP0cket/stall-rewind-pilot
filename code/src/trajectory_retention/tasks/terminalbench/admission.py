"""Seal validated terminalbench-v1 tasks, write the lane registry and calibration cohort, verify loads.

No model calls. Only calibration-split tasks are packaged, so every sealed task is
calibration-split; the registry still enforces the partition (held-out and main
loads are refused by the shared registry/purpose checks).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from trajectory_retention.contracts import to_jsonable
from trajectory_retention.registry import read_registry

from . import source as S
from .adapter import (ADAPTER_ENTRY, SCHEMA, SEAL_SCHEMA, TerminalBenchTaskAdapter, evaluator_sources, file_manifest,
                      sha256)
from .catalog import STALL_RATIONALE
from .package import ADMISSION, EVIDENCE, PACKAGES, WORKTREE, package_task_id, write_json

ROOT = WORKTREE / "benchmark_tasks"
COHORT = WORKTREE / "configs" / "calibration" / "cohort-terminalbench-v1.json"
MODELS = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def load_split() -> dict:
    return json.loads((ADMISSION / "family-split.json").read_text())


def ledgers() -> dict[str, dict]:
    out = {}
    for path in sorted((EVIDENCE / "tasks").glob("*/ledger.json")):
        row = json.loads(path.read_text())
        out[row["task"]] = row
    return out


def seal(task: str) -> dict:
    ledger = json.loads((EVIDENCE / "tasks" / task / "ledger.json").read_text())
    if ledger.get("status") != "VALIDATED":
        raise ValueError(f"{task} is not validated")
    task_id = package_task_id(task)
    package = PACKAGES / task_id
    manifest = json.loads((package / "package.json").read_text())
    validation = json.loads((EVIDENCE / "tasks" / task / "validation" / "validation.json").read_text())
    runner = json.loads((EVIDENCE / "tasks" / task / "runner-check" / "summary.json").read_text())
    if validation["package_sha256"] != sha256(package / "package.json") or not validation["passed"] or not runner["passed"]:
        raise ValueError(f"{task}: evidence does not bind the current package")
    out = ADMISSION / "tasks" / task_id
    compact = {k: validation[k] for k in ("task_id", "upstream_task", "package_sha256", "gates", "grades",
                                          "partial_control", "timing", "resources", "provider_calls")}
    compact["reference_run"] = {k: validation["reference_run"][k] for k in ("returncode", "seconds")}
    iso = validation["isolation"]
    compact["isolation"] = {"private_files_hashed": iso["private_files_hashed"], "start": iso["start"],
                            "native_exposures": iso["native_exposures"], "lane_introduced": iso["lane_introduced"],
                            "worker_image_files_scanned": iso["worker_image"]["files_scanned"],
                            "native_image_files_scanned": iso["native_image"]["files_scanned"],
                            "history_references": iso["history_references"]}
    if "binary_only_note" in validation:
        compact["binary_only_note"] = validation["binary_only_note"]
    write_json(out / "validation.json", compact)
    write_json(out / "runner-check.json", {k: runner.get(k) for k in (
        "task_id", "passed", "detector_rule", "assertions", "source_grades", "runs", "visible_start_excerpt",
        "partial_control_used_for_B", "provider_calls", "natural_model_outcomes", "fixture")})
    builds = {}
    for kind in ("native", "worker", "grader"):
        rec = json.loads((EVIDENCE / "tasks" / task / f"{kind}-build.json").read_text())
        builds[kind] = {k: rec.get(k) for k in ("result", "seconds", "tag", "bases", "dockerfile_sha256",
                                                "upstream_dockerfile_sha256", "extends_native_layers",
                                                "extends_worker_layers", "interpreter", "freeze")}
        builds[kind]["image"] = {k: rec["image"].get(k) for k in ("id", "architecture", "os", "size", "workdir", "user")}
    write_json(out / "environment.json", builds)
    split = load_split()["tasks"][task]["split"]
    if split != manifest["planned_split"]:
        raise ValueError("Split changed after packaging")
    evidence = [out / "validation.json", out / "runner-check.json", out / "environment.json",
                ADMISSION / "family-split.json"]
    admission = {"schema_version": SEAL_SCHEMA, "admission": "ADMITTED", "task_id": task_id, "upstream_task": task,
                 "package_format": SCHEMA, "family_id": manifest["family_id"], "package": rel(package),
                 "package_sha256": sha256(package / "package.json"), "split": split,
                 "eligibility": {"development_and_calibration": split == "calibration", "held_out_main_evaluation": False},
                 "main_study_frozen": False, "evaluator_sources": evaluator_sources(),
                 "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in evidence],
                 "stall_rationale": STALL_RATIONALE.get(task),
                 "verdict": ("Technically admitted (native TB2 single-shot task, arm64 rebuild, private W+R grader); "
                             "calibration/development only" if split == "calibration" else
                             "Technically admitted; held-out family, refused until a main-study freeze"),
                 "not_established": ["model difficulty/solve rate under this harness (no model calls)", "stall yield",
                                     "main-study eligibility (settings unfrozen)"]}
    write_json(out / "admission.json", admission)
    adapter = TerminalBenchTaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "terminalbench-seal")
    described = adapter.describe()
    if described.admission.value != "ADMITTED":
        raise ValueError("Seal did not verify")
    write_json(out / "task-record.json", to_jsonable(described))
    return {"task_id": task_id, "upstream_task": task, "family_id": manifest["family_id"], "split": split,
            "package": rel(package), "task_record": rel(out / "task-record.json"), "attestation": rel(out / "admission.json")}


def write_registry(sealed: list[dict]) -> Path:
    tasks = [{"task_id": r["task_id"], "family_id": r["family_id"], "adapter": ADAPTER_ENTRY, "admission": "ADMITTED",
              "package": r["package"], "task_record": r["task_record"], "attestation": r["attestation"],
              "split": r["split"], "upstream_task": r["upstream_task"], "package_format": SCHEMA,
              "eligibility": {"development_and_calibration": r["split"] == "calibration", "held_out_main_evaluation": False}}
             for r in sorted(sealed, key=lambda r: r["task_id"])]
    path = ADMISSION / "registry.json"
    write_json(path, {"registry_version": "task-admission.registry.v1", "lane": "terminalbench", "lane_version": "v1",
                      "source": {"url": S.SOURCE_URL, "revision": S.SOURCE_REVISION,
                                 "registry_dataset": S.REGISTRY_DATASET, "registry_revision": S.REGISTRY_REVISION},
                      "split_record": {"path": rel(ADMISSION / "family-split.json"),
                                       "sha256": sha256(ADMISSION / "family-split.json")},
                      "artifact_root": "benchmark_tasks (default read_registry root for this path)", "tasks": tasks})
    return path


def _member(entry: dict) -> dict:
    package = ROOT / entry["package"]
    adapter = TerminalBenchTaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "terminalbench-cohort")
    record = adapter.describe()
    m = adapter.manifest
    return {"task_id": entry["task_id"], "upstream_task": entry["upstream_task"], "family_id": entry["family_id"],
            "adapter": entry["adapter"],
            "attestation": {"path": "benchmark_tasks/" + entry["attestation"], "sha256": sha256(ROOT / entry["attestation"])},
            "task_record": {"path": "benchmark_tasks/" + entry["task_record"], "sha256": sha256(ROOT / entry["task_record"])},
            "package_manifest": {"path": "benchmark_tasks/" + entry["package"] + "/package.json",
                                 "sha256": sha256(package / "package.json")},
            "environment_version": record.environment_version, "environment_tag": m["environment"]["worker"]["tag"],
            "grader_image_id": m["environment"]["grader"]["image_id"],
            "evaluator_versions": [record.stages[0].evaluator_version], "stage_count": 1,
            "checks": len(adapter.scorer.expected), "native_limits": m["native_limits"],
            "worker_prompt": {k: m["worker_prompt"][k] for k in ("version", "rendered_sha256")},
            "stall_rationale": STALL_RATIONALE.get(entry["upstream_task"])}


def write_cohort(registry: Path) -> Path:
    entries = json.loads(registry.read_text())["tasks"]
    members = [_member(e) for e in entries if e["split"] == "calibration"]
    cohort = {"version": "calibration-cohort.lane.v1", "cohort_id": "terminalbench-calibration-v1", "lane": "terminalbench",
              "purpose": "calibration_preparation", "main_study_eligible": False, "model_ids": MODELS,
              "same_primary_members_for_each_model": True,
              "registry": {"path": "benchmark_tasks/" + rel(registry), "sha256": sha256(registry),
                           "artifact_root": "benchmark_tasks"},
              "membership_policy": {
                  "basis": "Every technically admitted calibration-split task of the terminalbench-v1 registry "
                           "(family split frozen in family-split.json before any build).",
                  "independence": "Cluster by family_id (path-tracing and path-tracing-reverse share one family).",
                  "failures": "Retain every outcome; no outcome-based substitution."},
              "native_lifecycle": "single shot; no feedback; no resubmission; hidden tests after the agent stops",
              "numeric_settings": {"status": "unfrozen", "required_before_each_launch": [
                  "per-task worker allowance", "model/provider settings", "detector settings", "call/tool limits"],
                  "flag": "No detector score_min_gain is ratified for this lane (0.06 is ratified only for ALE-Bench)."},
              "primary_tasks": members, "primary_family_count": len({m["family_id"] for m in members})}
    write_json(COHORT, cohort)
    return COHORT


def seal_all() -> dict:
    rows = ledgers()
    sealed = [seal(task) for task, row in sorted(rows.items()) if row.get("status") == "VALIDATED"]
    registry = write_registry(sealed)
    cohort = write_cohort(registry)
    write_json(ADMISSION / "ledger.json", {"tasks": [{k: row.get(k) for k in ("task", "family_id", "split", "status",
                                                                                 "reason")} for row in rows.values()]})
    return {"sealed": [s["task_id"] for s in sealed], "registry": str(registry), "cohort": str(cohort)}


def verify_loads(attempt: str) -> dict:
    registry_path = ADMISSION / "registry.json"
    out = EVIDENCE / "load-verification" / attempt
    out.mkdir(parents=True, exist_ok=False)
    entries = read_registry(registry_path)
    cohort = json.loads(COHORT.read_text())
    if sha256(WORKTREE / cohort["registry"]["path"]) != cohort["registry"]["sha256"]:
        raise ValueError("Cohort registry binding changed")
    cases = []
    for task_id, entry in sorted(entries.items()):
        scratch = Path(tempfile.mkdtemp(prefix="load-", dir=out))
        case = {"task_id": task_id, "split": entry.entry["split"], "loads": {}, "refusals": {}}
        for purpose in ("development", "calibration", "main"):
            try:
                entry.load(scratch_root=scratch, purpose=purpose)
                case["loads"][purpose] = True
            except ValueError as exc:
                case["loads"][purpose] = False
                case["refusals"][purpose] = str(exc)
        expected = {"development": entry.entry["split"] == "calibration",
                    "calibration": entry.entry["split"] == "calibration", "main": False}
        case["partition_enforced"] = case["loads"] == expected
        adapter = TerminalBenchTaskAdapter(entry.path("package"), scratch_root=scratch)
        case["describe_admitted"] = adapter.describe().admission.value == "ADMITTED"
        workspace = scratch / "W"
        adapter.materialize(workspace)
        case["natural_start_files"] = len(file_manifest(workspace))
        upstream = S.CHECKOUT / adapter.upstream_task / "instruction.md"
        case["instruction_byte_identical_to_upstream"] = (adapter.package / "public/instruction.md").read_bytes() == \
            upstream.read_bytes()
        prompt = adapter.worker_prompt("task")
        case["prompt_is_instruction_verbatim"] = prompt.text == upstream.read_text()
        case["prompt_sha256"] = hashlib.sha256(prompt.text.encode()).hexdigest()
        case["private_absent_from_start"] = not any((workspace / n).exists() for n in ("tests", "solution", "solve.sh"))
        cases.append(case)
    # Synthetic held-out refusal: a copy of one entry relabelled held_out must be refused for dev/cal/main.
    held_probe = {}
    if entries:
        from trajectory_retention.registry import RegisteredTask
        first = next(iter(entries.values()))
        probe = RegisteredTask({**first.entry, "split": "held_out",
                                "eligibility": {"development_and_calibration": False, "held_out_main_evaluation": False}},
                               first.root)
        for purpose in ("development", "calibration", "main"):
            try:
                probe.load(scratch_root=Path(tempfile.mkdtemp(prefix="held-", dir=out)), purpose=purpose)
                held_probe[purpose] = "loaded"
            except ValueError as exc:
                held_probe[purpose] = "refused: " + str(exc)[:200]
    for member in cohort["primary_tasks"]:
        for key in ("attestation", "task_record", "package_manifest"):
            if sha256(WORKTREE / member[key]["path"]) != member[key]["sha256"]:
                raise ValueError(f"Cohort binding changed: {member['task_id']} {key}")
    cli = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "tasks", "--registry", str(registry_path)],
                         capture_output=True, text=True, timeout=600)
    listed = json.loads(cli.stdout) if cli.returncode == 0 else []
    template = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "mining-plan", "template",
                               "--cohort", str(COHORT), "--output", str(out / "mining-plan-template.json")],
                              capture_output=True, text=True, timeout=300)
    plan = json.loads((out / "mining-plan-template.json").read_text()) if (out / "mining-plan-template.json").is_file() else {}
    summary = {"schema_version": "terminalbench.load-verification.v1", "registry_sha256": sha256(registry_path),
               "cohort_sha256": sha256(COHORT), "tasks": cases, "cli_tasks_returncode": cli.returncode,
               "cli_tasks_listed": len(listed), "provider_calls": 0, "cohort_members": len(cohort["primary_tasks"]),
               "cohort_model_ids": cohort["model_ids"], "mining_plan_template_returncode": template.returncode,
               "mining_plan_template_stderr": template.stderr[-1500:],
               "mining_plan_template_tasks": sorted(plan.get("admission", {})),
               "held_out_relabel_probe": held_probe,
               "held_out_relabel_refused_all": bool(held_probe) and all(v.startswith("refused") for v in held_probe.values())}
    summary["passed"] = (cli.returncode == 0 and len(listed) == len(entries)
                         and summary["held_out_relabel_refused_all"]
                         and summary["mining_plan_template_tasks"] == sorted(m["task_id"] for m in cohort["primary_tasks"])
                         and all(c["partition_enforced"] and c["describe_admitted"]
                                 and c["instruction_byte_identical_to_upstream"] and c["prompt_is_instruction_verbatim"]
                                 and c["private_absent_from_start"] for c in cases))
    write_json(out / "summary.json", summary)
    write_json(ADMISSION / "load-verification.json", {k: v for k, v in summary.items() if k != "tasks"} | {
        "tasks": [{k: c[k] for k in ("task_id", "split", "loads", "partition_enforced", "describe_admitted",
                                     "prompt_sha256")} for c in cases]})
    return summary
