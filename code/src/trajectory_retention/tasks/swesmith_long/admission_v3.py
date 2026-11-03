"""Seal v3 tasks, write the v3 lane registry/cohort and verify real loads (no model calls)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from trajectory_retention.contracts import to_jsonable
from trajectory_retention.registry import read_registry
from .adapter import file_manifest, sha256
from .adapter_v3 import ADAPTER_ENTRY, GIT_TAR, SWESmithLongV3TaskAdapter, evaluator_sources
from .pipeline import WORKTREE, write_json
from .pipeline_v3 import ADMISSION, EVIDENCE_V3, PACKAGES

ROOT = WORKTREE / "benchmark_tasks"
V1_ADMISSION = ROOT / "admission" / "swesmith-long-v1"
COHORT = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v3.json"
COHORT_V2 = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v2.json"
MODELS = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def seal(task_id: str) -> dict:
    record = json.loads((EVIDENCE_V3 / "tasks" / task_id / "package-record.json").read_text())
    runner = json.loads((EVIDENCE_V3 / "tasks" / task_id / "runner-check" / "summary.json").read_text())
    if record["status"] != "VALIDATED" or not runner.get("passed"):
        raise ValueError(f"{task_id} is not validated")
    package = PACKAGES / task_id
    out = ADMISSION / "tasks" / task_id
    validation = out / "validation.json"
    write_json(validation, {k: record[k] for k in ("task_id", "predecessor", "instance_id", "repo_key", "family_id",
                                                   "split", "package_sha256", "counts", "bug_shape", "native_start",
                                                   "git_start", "validation", "withheld_leak_scan", "v2_comparison")})
    runner_path = out / "runner-check.json"
    write_json(runner_path, {k: runner.get(k) for k in (
        "task_id", "passed", "detector_rule", "assertions", "source_grades", "runs", "visible_feedback_excerpt",
        "git_probe_excerpt", "restore_B_git_status", "provider_calls", "natural_model_outcomes", "fixture")})
    key = record["repo_key"]
    family_dir = EVIDENCE_V3 / "families" / key
    image = json.loads((family_dir / "image.json").read_text())
    needles = json.loads((family_dir / "image-needles.json").read_text())
    clean_scan = json.loads((family_dir / "image-leak-scan.json").read_text())
    family_path = ADMISSION / "families" / f"{key}.json"
    write_json(family_path, {"image": image, "image_needles": {k: needles[k] for k in ("image_id", "files_scanned", "needles", "hits")},
                             "clean_source_scan": clean_scan})
    v1_family = V1_ADMISSION / "families" / f"{key}.json"
    split = record["split"]
    admission = {"schema_version": "swesmith-long.admission.v3", "admission": "ADMITTED", "task_id": task_id,
                 "predecessor": record["predecessor"], "family_id": record["family_id"], "package": rel(package),
                 "package_sha256": sha256(package / "package.json"), "split": split,
                 "eligibility": {"development_and_calibration": split == "calibration", "held_out_main_evaluation": False},
                 "main_study_frozen": False, "evaluator_sources": evaluator_sources(),
                 "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in (validation, runner_path, family_path, v1_family)],
                 "verdict": ("Technically admitted (native SWE-smith start in a Git repository, native instance template) "
                             "for lane calibration/development" if split == "calibration" else
                             "Technically admitted (native SWE-smith start in a Git repository, native instance template); "
                             "held-out family, refused until a main-study freeze"),
                 "not_established": ["model difficulty/solve rate (no model calls)", "stall yield",
                                     "main-study eligibility (settings unfrozen)"]}
    write_json(out / "admission.json", admission)
    adapter = SWESmithLongV3TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v3-seal")
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
    write_json(path, {"registry_version": "task-admission.registry.v1", "lane": "swesmith-long", "lane_version": "v3",
                      "supersedes": "benchmark_tasks/admission/swesmith-long-v2/registry.json (unchanged)",
                      "artifact_root": "benchmark_tasks (default read_registry root for this path)", "tasks": tasks})
    return path


def write_cohort(sealed: list[dict], registry: Path) -> Path:
    members = []
    for row in sorted((r for r in sealed if r["split"] == "calibration"), key=lambda r: r["task_id"]):
        package = ROOT / row["package"]
        adapter = SWESmithLongV3TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v3-cohort")
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
            "withheld_test_files": adapter.manifest["native_start"]["withheld_files"],
            "git_start_commit": adapter.git_start["commit"],
            "worker_prompt": {k: adapter.manifest["worker_prompt"][k] for k in ("version", "rendered_sha256")}})
    cohort = {"version": "calibration-cohort.lane.v1", "cohort_id": "swesmith-long-calibration-v3", "lane": "swesmith-long",
              "supersedes": "configs/calibration/cohort-swesmith-long-v2.json (unchanged)",
              "purpose": "calibration_preparation", "main_study_eligible": False, "model_ids": MODELS,
              "model_ids_note": "Study worker model switched to openai/gpt-6-luna on 2026-09-24 (v2 cohort: openai/gpt-5.6-luna).",
              "same_primary_members_for_each_model": True,
              "registry": {"path": "benchmark_tasks/" + rel(registry), "sha256": sha256(registry),
                           "artifact_root": "benchmark_tasks"},
              "membership_policy": {
                  "basis": "Every technically admitted v3 task (v2 native SWE-smith start in a fresh Git repository, native "
                           "instance template) of the calibration-split families; same members and split as cohort v2.",
                  "independence": "Cluster by family_id; several tasks share each repository family.",
                  "failures": "Retain every outcome; no outcome-based substitution."},
              "numeric_settings": {"status": "unfrozen", "required_before_each_launch": [
                  "per-task worker allowance", "model/provider settings", "detector settings", "call/tool limits"]},
              "primary_tasks": members, "primary_family_count": len({m["family_id"] for m in members})}
    write_json(COHORT, cohort)
    return COHORT


def verify_loads(attempt: str) -> dict:
    from ..prompts import swesmith_prompt_v3  # noqa: F401  (import check)
    registry_path = ADMISSION / "registry.json"
    out = EVIDENCE_V3 / "load-verification" / attempt
    out.mkdir(parents=True, exist_ok=False)
    entries = read_registry(registry_path)
    cohort = json.loads(COHORT.read_text())
    if sha256(WORKTREE / cohort["registry"]["path"]) != cohort["registry"]["sha256"]:
        raise ValueError("Cohort registry binding changed")
    v2_members = sorted(m["task_id"][:-3] for m in json.loads(COHORT_V2.read_text())["primary_tasks"])
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
        adapter = SWESmithLongV3TaskAdapter(entry.path("package"), scratch_root=scratch)
        case["describe_admitted"] = adapter.describe().admission.value == "ADMITTED"
        workspace = scratch / "W"
        adapter.materialize(workspace)
        manifest = file_manifest(workspace)
        case["natural_start_equals_public_workspace_plus_git"] = (
            {k: v for k, v in manifest.items() if not k.startswith(".git/")} == file_manifest(adapter.package / "public/workspace")
            and sorted(k for k in manifest if k.startswith(".git/")) == sorted(
                m.name for m in tarfile.open(adapter.package / GIT_TAR).getmembers() if m.isfile()))
        case["withheld_absent_in_natural_start"] = not any((workspace / r["path"]).exists() for r in adapter.withheld["files"])
        v2 = ROOT / "packages" / "swesmith-long-v2" / entry.entry["predecessor"]
        case["instructions_byte_identical_to_v2"] = ((v2 / "public/instructions.md").read_bytes()
                                                     == (adapter.package / "public/instructions.md").read_bytes())
        case["workspace_files_identical_to_v2"] = (file_manifest(v2 / "public/workspace")
                                                   == file_manifest(adapter.package / "public/workspace"))
        prompt = adapter.worker_prompt("repair")
        case["prompt_native_v3"] = (prompt.version == "benchmark-task-prompt.v3"
                                    and prompt.text.startswith("<uploaded_files>\n/workspace\n</uploaded_files>\n")
                                    and prompt.text.endswith("Your thinking should be thorough and so it's fine if it's very long.")
                                    and "in the directory /workspace." in prompt.text
                                    and prompt.text.count(adapter.stage("repair").instructions) == 1)
        case["prompt_sha256"] = hashlib.sha256(prompt.text.encode()).hexdigest()
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
    summary = {"schema_version": "swesmith-long.load-verification.v3", "registry_sha256": sha256(registry_path),
               "cohort_sha256": sha256(COHORT), "tasks": cases, "cli_tasks_returncode": cli.returncode,
               "cli_tasks_listed": len(listed), "provider_calls": 0, "cohort_members": len(cohort["primary_tasks"]),
               "cohort_members_same_as_v2": sorted(m["task_id"][:-3] for m in cohort["primary_tasks"]) == v2_members,
               "cohort_model_ids": cohort["model_ids"], "mining_plan_template_returncode": template.returncode,
               "held_out_refused_all_purposes": all(not any(c["loads"].values()) for c in cases if c["split"] == "held_out"),
               "mining_plan_template_tasks": sorted(plan.get("admission", {}))}
    summary["passed"] = (cli.returncode == 0 and len(listed) == len(entries) and summary["cohort_members_same_as_v2"]
                         and summary["mining_plan_template_tasks"] == sorted(m["task_id"] for m in cohort["primary_tasks"])
                         and all(c["partition_enforced"] and c["describe_admitted"]
                                 and c["natural_start_equals_public_workspace_plus_git"]
                                 and c["withheld_absent_in_natural_start"] and c["instructions_byte_identical_to_v2"]
                                 and c["workspace_files_identical_to_v2"] and c["prompt_native_v3"] for c in cases))
    write_json(out / "summary.json", summary)
    return summary
