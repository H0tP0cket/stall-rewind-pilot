"""Seal v2 tasks, write the v2 lane registry/cohort and verify real loads (no model calls)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from trajectory_retention.contracts import to_jsonable
from trajectory_retention.registry import read_registry
from .adapter import file_manifest, sha256
from .adapter_v2 import ADAPTER_ENTRY, SWESmithLongV2TaskAdapter, evaluator_sources
from .pipeline import EVIDENCE, WORKTREE, write_json
from .pipeline_v2 import ADMISSION, EVIDENCE_V2, PACKAGES

ROOT = WORKTREE / "benchmark_tasks"
V1_ADMISSION = ROOT / "admission" / "swesmith-long-v1"
COHORT = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v2.json"
MODELS = ["openai/gpt-5.6-luna", "deepseek/deepseek-v4.1-flash"]


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def seal(task_id: str) -> dict:
    record = json.loads((EVIDENCE_V2 / "tasks" / task_id / "package-record.json").read_text())
    runner = json.loads((EVIDENCE_V2 / "tasks" / task_id / "runner-check" / "summary.json").read_text())
    if record["status"] != "VALIDATED" or not runner.get("passed"):
        raise ValueError(f"{task_id} is not validated")
    package = PACKAGES / task_id
    out = ADMISSION / "tasks" / task_id
    validation = out / "validation.json"
    write_json(validation, {k: record[k] for k in ("task_id", "predecessor", "instance_id", "repo_key", "family_id",
                                                   "split", "package_sha256", "counts", "bug_shape", "native_removal",
                                                   "start_equals_mirror_branch_head", "validation", "withheld_leak_scan")})
    runner_path = out / "runner-check.json"
    write_json(runner_path, {k: runner.get(k) for k in ("task_id", "passed", "detector_rule", "assertions", "source_grades",
                                                        "runs", "visible_feedback_excerpt", "provider_calls",
                                                        "natural_model_outcomes", "fixture")})
    key = record["repo_key"]
    needles = json.loads((EVIDENCE_V2 / "families" / key / "image-needles.json").read_text())
    needle_path = ADMISSION / "families" / f"{key}-image-needles.json"
    write_json(needle_path, {"image_id": needles["image_id"], "files_scanned": needles["files_scanned"],
                             "needles": needles["needles"], "hits": needles["hits"]})
    family_path = V1_ADMISSION / "families" / f"{key}.json"
    split = record["split"]
    admission = {"schema_version": "swesmith-long.admission.v2", "admission": "ADMITTED", "task_id": task_id,
                 "predecessor": record["predecessor"], "family_id": record["family_id"], "package": rel(package),
                 "package_sha256": sha256(package / "package.json"), "split": split,
                 "eligibility": {"development_and_calibration": split == "calibration", "held_out_main_evaluation": False},
                 "main_study_frozen": False, "evaluator_sources": evaluator_sources(),
                 "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in (validation, runner_path, needle_path, family_path)],
                 "verdict": ("Technically admitted (native SWE-smith start) for lane calibration/development" if split == "calibration"
                             else "Technically admitted (native SWE-smith start); held-out family, refused until a main-study freeze"),
                 "not_established": ["model difficulty/solve rate (no model calls)", "stall yield",
                                     "main-study eligibility (settings unfrozen)"]}
    write_json(out / "admission.json", admission)
    adapter = SWESmithLongV2TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v2-seal")
    described = adapter.describe()
    if described.admission.value != "ADMITTED":
        raise ValueError("Seal did not verify")
    write_json(out / "task-record.json", to_jsonable(described))
    return {"task_id": task_id, "family_id": record["family_id"], "split": split, "predecessor": record["predecessor"],
            "package": rel(package), "task_record": rel(out / "task-record.json"), "attestation": rel(out / "admission.json")}


def write_registry(sealed: list[dict]) -> Path:
    tasks = [{"task_id": r["task_id"], "family_id": r["family_id"], "adapter": ADAPTER_ENTRY, "admission": "ADMITTED",
              "package": r["package"], "task_record": r["task_record"], "attestation": r["attestation"],
              "split": r["split"], "predecessor": r["predecessor"],
              "eligibility": {"development_and_calibration": r["split"] == "calibration", "held_out_main_evaluation": False}}
             for r in sorted(sealed, key=lambda r: r["task_id"])]
    path = ADMISSION / "registry.json"
    write_json(path, {"registry_version": "task-admission.registry.v1", "lane": "swesmith-long", "lane_version": "v2",
                      "supersedes": "benchmark_tasks/admission/swesmith-long-v1/registry.json (unchanged)",
                      "artifact_root": "benchmark_tasks (default read_registry root for this path)", "tasks": tasks})
    return path


def write_cohort(sealed: list[dict], registry: Path) -> Path:
    members = []
    for row in sorted((r for r in sealed if r["split"] == "calibration"), key=lambda r: r["task_id"]):
        package = ROOT / row["package"]
        adapter = SWESmithLongV2TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v2-cohort")
        record = adapter.describe()
        members.append({"task_id": row["task_id"], "predecessor": row["predecessor"], "family_id": row["family_id"],
            "adapter": ADAPTER_ENTRY,
            "attestation": {"path": "benchmark_tasks/" + row["attestation"], "sha256": sha256(ROOT / row["attestation"])},
            "task_record": {"path": "benchmark_tasks/" + row["task_record"], "sha256": sha256(ROOT / row["task_record"])},
            "package_manifest": {"path": "benchmark_tasks/" + row["package"] + "/package.json",
                                 "sha256": sha256(package / "package.json")},
            "environment_version": record.environment_version, "environment_tag": adapter.manifest["environment_tag"],
            "evaluator_versions": [record.stages[0].evaluator_version], "stage_count": 1,
            "upstream_instance_id": adapter.manifest["upstream_instance_id"], "counts": adapter.manifest["counts"],
            "withheld_test_files": adapter.manifest["native_start"]["withheld_files"]})
    cohort = {"version": "calibration-cohort.lane.v1", "cohort_id": "swesmith-long-calibration-v2", "lane": "swesmith-long",
              "supersedes": "configs/calibration/cohort-swesmith-long-v1.json (unchanged)",
              "purpose": "calibration_preparation", "main_study_eligible": False, "model_ids": MODELS,
              "same_primary_members_for_each_model": True,
              "registry": {"path": "benchmark_tasks/" + rel(registry), "sha256": sha256(registry),
                           "artifact_root": "benchmark_tasks"},
              "membership_policy": {
                  "basis": "Every technically admitted v2 (native SWE-smith start) task of the calibration-split families "
                           "of the seeded swesmith-long selection; same family split as v1.",
                  "independence": "Cluster by family_id; several tasks share each repository family.",
                  "failures": "Retain every outcome; no outcome-based substitution."},
              "numeric_settings": {"status": "unfrozen", "required_before_each_launch": [
                  "per-task worker allowance", "model/provider settings", "detector settings", "call/tool limits"]},
              "primary_tasks": members, "primary_family_count": len({m["family_id"] for m in members})}
    write_json(COHORT, cohort)
    return COHORT


def verify_loads(attempt: str) -> dict:
    registry_path = ADMISSION / "registry.json"
    out = EVIDENCE_V2 / "load-verification" / attempt
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
        expected = {"development": entry.entry["split"] == "calibration", "calibration": entry.entry["split"] == "calibration",
                    "main": False}
        case["partition_enforced"] = case["loads"] == expected
        adapter = SWESmithLongV2TaskAdapter(entry.path("package"), scratch_root=scratch)
        case["describe_admitted"] = adapter.describe().admission.value == "ADMITTED"
        workspace = scratch / "W"
        adapter.materialize(workspace)
        case["natural_start_equals_public_workspace"] = file_manifest(workspace) == file_manifest(adapter.package / "public/workspace")
        case["withheld_absent_in_natural_start"] = not any((workspace / r["path"]).exists() for r in adapter.withheld["files"])
        v1 = ROOT / "packages" / "swesmith-long-v1" / entry.entry["predecessor"]
        case["instructions_byte_identical_to_v1"] = ((v1 / "public/instructions.md").read_bytes()
                                                     == (adapter.package / "public/instructions.md").read_bytes())
        case["prompt_rendered"] = adapter.worker_prompt("repair").text.endswith("edgecases and make sure your fix handles them as well\n")
        cases.append(case)
    for member in cohort["primary_tasks"]:
        for key in ("attestation", "task_record", "package_manifest"):
            if sha256(WORKTREE / member[key]["path"]) != member[key]["sha256"]:
                raise ValueError(f"Cohort binding changed: {member['task_id']} {key}")
    cli = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "tasks", "--registry", str(registry_path)],
                         capture_output=True, text=True, timeout=300)
    listed = json.loads(cli.stdout) if cli.returncode == 0 else []
    template = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "mining-plan", "template",
                               "--cohort", str(COHORT), "--output", str(out / "mining-plan-template.json")],
                              capture_output=True, text=True, timeout=300)
    plan = json.loads((out / "mining-plan-template.json").read_text()) if (out / "mining-plan-template.json").is_file() else {}
    summary = {"schema_version": "swesmith-long.load-verification.v2", "registry_sha256": sha256(registry_path),
               "cohort_sha256": sha256(COHORT), "tasks": cases, "cli_tasks_returncode": cli.returncode,
               "cli_tasks_listed": len(listed), "provider_calls": 0, "cohort_members": len(cohort["primary_tasks"]),
               "held_out_refused_all_purposes": all(not any(c["loads"].values()) for c in cases if c["split"] == "held_out"),
               "mining_plan_template_tasks": sorted(plan.get("admission", {}))}
    summary["passed"] = (cli.returncode == 0 and len(listed) == len(entries)
                         and summary["mining_plan_template_tasks"] == sorted(m["task_id"] for m in cohort["primary_tasks"])
                         and all(c["partition_enforced"] and c["describe_admitted"] and c["natural_start_equals_public_workspace"]
                                 and c["withheld_absent_in_natural_start"] and c["instructions_byte_identical_to_v1"]
                                 and c["prompt_rendered"] for c in cases))
    write_json(out / "summary.json", summary)
    return summary
