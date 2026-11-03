"""Seal validated tasks, write the lane registry/cohort and verify real loads.

Seals bind the package manifest, committed compact evidence and evaluator
sources; the adapter re-verifies them on every load. No model/provider calls.
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
from .adapter import ADAPTER_ENTRY, LANE, SWESmithLongTaskAdapter, evaluator_sources, file_manifest, sha256
from .pipeline import ADMISSION, EVIDENCE, PACKAGES, WORKTREE, write_json

ROOT = WORKTREE / "benchmark_tasks"
COHORT = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v1.json"


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def compact_family(repo_key: str) -> Path:
    family = json.loads((EVIDENCE / "families" / repo_key / "family.json").read_text())
    keep = {k: family.get(k) for k in ("repo_key", "family_id", "split", "status", "official_image", "official", "mirror",
                                        "environment", "generated_files", "native_test_command", "pytest_args",
                                        "reference_runs", "reference_stable_pass", "reference_flaky", "project_top_levels")}
    scan = family.get("image_leak_scan", {})
    keep["image_leak_scan"] = {k: scan.get(k) for k in ("files_scanned", "identical_to_clean_source", "project_package_copies",
                                                          "project_archives", "pip_cache_files", "leak", "rule")}
    keep["environment_attempts"] = family.get("environment_attempts")
    path = ADMISSION / "families" / f"{repo_key}.json"
    write_json(path, keep)
    return path


def seal(task_id: str, *, selection_sha256: str) -> dict:
    record = json.loads((EVIDENCE / "tasks" / task_id / "package-record.json").read_text())
    runner = json.loads((EVIDENCE / "tasks" / task_id / "runner-check" / "summary.json").read_text())
    if record["status"] != "VALIDATED" or not runner.get("passed"):
        raise ValueError(f"{task_id} is not validated")
    package = PACKAGES / task_id
    out = ADMISSION / "tasks" / task_id
    validation = out / "validation.json"
    write_json(validation, {k: record[k] for k in ("task_id", "instance_id", "repo_key", "family_id", "split",
                                                   "package_sha256", "counts", "bug_shape", "provenance", "validation")})
    runner_path = out / "runner-check.json"
    write_json(runner_path, {k: runner.get(k) for k in ("task_id", "passed", "detector_rule", "assertions", "source_grades",
                                                        "runs", "visible_feedback_excerpt", "provider_calls",
                                                        "natural_model_outcomes", "fixture")})
    family_path = compact_family(record["repo_key"])
    split = record["split"]
    admission = {"schema_version": "swesmith-long.admission.v1", "admission": "ADMITTED", "task_id": task_id,
                 "family_id": record["family_id"], "package": rel(package), "package_sha256": sha256(package / "package.json"),
                 "split": split, "eligibility": {"development_and_calibration": split == "calibration",
                                                 "held_out_main_evaluation": False},
                 "main_study_frozen": False, "selection_sha256": selection_sha256,
                 "evaluator_sources": evaluator_sources(),
                 "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in (validation, runner_path, family_path)],
                 "verdict": ("Technically admitted for lane calibration/development" if split == "calibration" else
                             "Technically admitted; held-out family, refused for development/calibration until a main-study freeze"),
                 "not_established": ["model difficulty/solve rate (no model calls)", "stall yield",
                                     "main-study eligibility (settings unfrozen)"]}
    write_json(out / "admission.json", admission)
    adapter = SWESmithLongTaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-seal")
    described = adapter.describe()
    if described.admission.value != "ADMITTED":
        raise ValueError("Seal did not verify")
    write_json(out / "task-record.json", to_jsonable(described))
    return {"task_id": task_id, "family_id": record["family_id"], "split": split,
            "package": rel(package), "task_record": rel(out / "task-record.json"), "attestation": rel(out / "admission.json")}


def write_registry(sealed: list[dict]) -> Path:
    tasks = []
    for row in sorted(sealed, key=lambda r: r["task_id"]):
        tasks.append({"task_id": row["task_id"], "family_id": row["family_id"], "adapter": ADAPTER_ENTRY,
                      "admission": "ADMITTED", "package": row["package"], "task_record": row["task_record"],
                      "attestation": row["attestation"], "split": row["split"],
                      "eligibility": {"development_and_calibration": row["split"] == "calibration",
                                      "held_out_main_evaluation": False}})
    path = ADMISSION / "registry.json"
    write_json(path, {"registry_version": "task-admission.registry.v1", "lane": "swesmith-long",
                      "artifact_root": "benchmark_tasks (default read_registry root for this path)", "tasks": tasks})
    return path


def write_cohort(sealed: list[dict], registry: Path) -> Path:
    members = []
    for row in sorted((r for r in sealed if r["split"] == "calibration"), key=lambda r: r["task_id"]):
        package = ROOT / row["package"]
        adapter = SWESmithLongTaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-cohort")
        record = adapter.describe()
        members.append({"task_id": row["task_id"], "family_id": row["family_id"], "adapter": ADAPTER_ENTRY,
            "attestation": {"path": "benchmark_tasks/" + row["attestation"], "sha256": sha256(ROOT / row["attestation"])},
            "task_record": {"path": "benchmark_tasks/" + row["task_record"], "sha256": sha256(ROOT / row["task_record"])},
            "package_manifest": {"path": "benchmark_tasks/" + row["package"] + "/package.json",
                                 "sha256": sha256(package / "package.json")},
            "environment_version": record.environment_version, "environment_tag": adapter.manifest["environment_tag"],
            "evaluator_versions": [record.stages[0].evaluator_version], "stage_count": 1,
            "upstream_instance_id": adapter.manifest["upstream_instance_id"],
            "counts": adapter.manifest["counts"]})
    cohort = {"version": "calibration-cohort.lane.v1", "cohort_id": "swesmith-long-calibration-v1", "lane": "swesmith-long",
              "purpose": "calibration_preparation", "main_study_eligible": False,
              "model_ids": ["openai/gpt-5.6-luna", "deepseek/deepseek-v4.1-flash"],
              "same_primary_members_for_each_model": True,
              "registry": {"path": "benchmark_tasks/" + rel(registry), "sha256": sha256(registry),
                           "artifact_root": "benchmark_tasks"},
              "membership_policy": {
                  "basis": "Every technically admitted task of the calibration-split families of the seeded swesmith-long "
                           "selection. Held-out families are excluded and refused for development/calibration loading.",
                  "independence": "Cluster by family_id; several tasks share each repository family.",
                  "failures": "Retain every outcome; no outcome-based substitution."},
              "numeric_settings": {"status": "unfrozen", "required_before_each_launch": [
                  "per-task worker allowance", "model/provider settings", "detector settings", "call/tool limits"]},
              "primary_tasks": members,
              "primary_family_count": len({m["family_id"] for m in members})}
    write_json(COHORT, cohort)
    return COHORT


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
        scratch = Path(tempfile.mkdtemp(prefix="swesmith-long-load-", dir=out))
        case = {"task_id": task_id, "split": entry.entry["split"], "loads": {}, "refusals": {}}
        for purpose in ("development", "calibration", "main"):
            try:
                adapter = entry.load(scratch_root=scratch, purpose=purpose)
                case["loads"][purpose] = True
            except ValueError as exc:
                case["loads"][purpose] = False
                case["refusals"][purpose] = str(exc)
        expected = {"development": entry.entry["split"] == "calibration", "calibration": entry.entry["split"] == "calibration",
                    "main": False}
        case["partition_enforced"] = case["loads"] == expected
        adapter = SWESmithLongTaskAdapter(entry.path("package"), scratch_root=scratch)
        record = adapter.describe()
        case["describe_admitted"] = record.admission.value == "ADMITTED"
        workspace = scratch / "W"
        adapter.materialize(workspace)
        case["natural_start_equals_public_workspace"] = file_manifest(workspace) == file_manifest(adapter.package / "public/workspace")
        case["prompt_rendered"] = adapter.worker_prompt("repair").text.endswith("edgecases and make sure your fix handles them as well\n")
        cases.append(case)
    for member in cohort["primary_tasks"]:
        for key in ("attestation", "task_record", "package_manifest"):
            if sha256(WORKTREE / member[key]["path"]) != member[key]["sha256"]:
                raise ValueError(f"Cohort binding changed: {member['task_id']} {key}")
    cli = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "tasks", "--registry", str(registry_path)],
                         capture_output=True, text=True, timeout=300)
    listed = json.loads(cli.stdout) if cli.returncode == 0 else []
    summary = {"schema_version": "swesmith-long.load-verification.v1", "registry_sha256": sha256(registry_path),
               "cohort_sha256": sha256(COHORT), "tasks": cases, "cli_tasks_returncode": cli.returncode,
               "cli_tasks_listed": len(listed), "provider_calls": 0,
               "passed": cli.returncode == 0 and len(listed) == len(entries) and all(
                   c["partition_enforced"] and c["describe_admitted"] and c["natural_start_equals_public_workspace"]
                   and c["prompt_rendered"] for c in cases),
               "cohort_members": len(cohort["primary_tasks"]),
               "held_out_refused_all_purposes": all(not any(c["loads"].values()) for c in cases if c["split"] == "held_out")}
    template = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "mining-plan", "template",
                               "--cohort", str(COHORT), "--output", str(out / "mining-plan-template.json")],
                              capture_output=True, text=True, timeout=300)
    plan = json.loads((out / "mining-plan-template.json").read_text()) if (out / "mining-plan-template.json").is_file() else {}
    summary["mining_plan_template"] = {"written": bool(plan), "tasks": sorted(plan.get("admission", {})),
                                       "launchable": False, "note": "template only; numeric settings unresolved"}
    summary["passed"] = summary["passed"] and sorted(plan.get("admission", {})) == sorted(
        m["task_id"] for m in cohort["primary_tasks"])
    write_json(out / "summary.json", summary)
    return summary
