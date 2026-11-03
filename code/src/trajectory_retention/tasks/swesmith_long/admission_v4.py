"""Seal the v4-release tasks, write the v4 registry/cohort (v3 tasks + new tasks) and verify loads.

No model calls. The 33 v3 entries are copied from the v3 registry unchanged
(adapter_v3, v3 seals); the new tasks load through adapter_v4 with seals under
admission/swesmith-long-v4. The v3 registry, cohort, seals and packages are not
modified.
"""
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
from .adapter_v3 import GIT_TAR, SWESmithLongV3TaskAdapter, evaluator_sources
from .adapter_v4 import ADAPTER_ENTRY, PACKAGE_FORMAT, SEAL_SCHEMA, SWESmithLongV4TaskAdapter, loader_sources
from .pipeline import EVIDENCE, WORKTREE, write_json
from .pipeline_v4 import ADMISSION, EVIDENCE_V4, PACKAGES, PACKAGES_V2_STAGE, SELECTION

ROOT = WORKTREE / "benchmark_tasks"
REGISTRY_V3 = ROOT / "admission" / "swesmith-long-v3" / "registry.json"
COHORT_V3 = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v3.json"
COHORT = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v4.json"
MODELS = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def compact_family(repo_key: str) -> Path:
    """Base environment (v1 step) + git image (v3 step) + scans for one new family."""
    base = json.loads((EVIDENCE / "families" / repo_key / "family.json").read_text())
    keep = {k: base.get(k) for k in ("repo_key", "family_id", "split", "status", "official_image", "official", "mirror",
                                      "environment", "generated_files", "native_test_command", "pytest_args",
                                      "reference_runs", "reference_stable_pass", "reference_flaky", "project_top_levels",
                                      "environment_attempts")}
    scan = base.get("image_leak_scan", {})
    keep["image_leak_scan"] = {k: scan.get(k) for k in ("files_scanned", "identical_to_clean_source", "project_package_copies",
                                                          "project_archives", "pip_cache_files", "leak", "rule")}
    v3_dir = EVIDENCE / "v3" / "families" / repo_key
    needles = json.loads((v3_dir / "image-needles.json").read_text())
    path = ADMISSION / "families" / f"{repo_key}.json"
    write_json(path, {"base_environment": keep, "git_image": json.loads((v3_dir / "image.json").read_text()),
                      "image_needles": {k: needles[k] for k in ("image_id", "files_scanned", "needles", "hits")},
                      "clean_source_scan": json.loads((v3_dir / "image-leak-scan.json").read_text())})
    return path


def seal(task_id: str) -> dict:
    record = json.loads((EVIDENCE_V4 / "tasks" / task_id / "package-record.json").read_text())
    runner = json.loads((EVIDENCE_V4 / "tasks" / task_id / "runner-check" / "summary.json").read_text())
    if record["status"] != "VALIDATED" or not runner.get("passed"):
        raise ValueError(f"{task_id} is not validated")
    package = PACKAGES / task_id
    out = ADMISSION / "tasks" / task_id
    validation = out / "validation.json"
    write_json(validation, {k: record[k] for k in ("task_id", "predecessor_stage", "instance_id", "repo_key", "family_id",
                                                   "split", "package_sha256", "counts", "bug_shape", "native_start",
                                                   "git_start", "validation", "withheld_leak_scan", "v2_comparison")})
    runner_path = out / "runner-check.json"
    write_json(runner_path, {k: runner.get(k) for k in (
        "task_id", "passed", "detector_rule", "assertions", "source_grades", "runs", "visible_feedback_excerpt",
        "git_probe_excerpt", "restore_B_git_status", "provider_calls", "natural_model_outcomes", "fixture")})
    family_path = compact_family(record["repo_key"])
    ledger_path = ADMISSION / "candidates" / f"{record['repo_key']}.json"
    split = record["split"]
    evidence = [validation, runner_path, family_path, ADMISSION / "selection.json", ADMISSION / "issue-review.json",
                ledger_path, ADMISSION / "stop-rule.json"]
    admission = {"schema_version": SEAL_SCHEMA, "admission": "ADMITTED", "task_id": task_id,
                 "package_format": PACKAGE_FORMAT, "family_id": record["family_id"], "package": rel(package),
                 "package_sha256": sha256(package / "package.json"), "split": split,
                 "eligibility": {"development_and_calibration": split == "calibration", "held_out_main_evaluation": False},
                 "main_study_frozen": False, "evaluator_sources": evaluator_sources(), "loader_sources": loader_sources(),
                 "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in evidence],
                 "verdict": ("Technically admitted (v3 package format: native SWE-smith start with withheld FAIL_TO_PASS "
                             "files in a one-commit Git repository, native instance template); new in lane release v4; "
                             "calibration/development only" if split == "calibration" else
                             "Technically admitted; held-out family, refused until a main-study freeze"),
                 "not_established": ["model difficulty/solve rate (no model calls)", "stall yield",
                                     "main-study eligibility (settings unfrozen)"]}
    write_json(out / "admission.json", admission)
    adapter = SWESmithLongV4TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v4-seal")
    described = adapter.describe()
    if described.admission.value != "ADMITTED":
        raise ValueError("Seal did not verify")
    write_json(out / "task-record.json", to_jsonable(described))
    return {"task_id": task_id, "family_id": record["family_id"], "split": split,
            "package": rel(package), "task_record": rel(out / "task-record.json"), "attestation": rel(out / "admission.json")}


def write_ledgers(families: list[str]) -> list[dict]:
    """Committed copies of the per-family candidate ledgers (every drawn instance and its outcome)."""
    rows = []
    for key in families:
        ledger = json.loads((EVIDENCE_V4 / "candidates" / f"{key}.json").read_text())
        write_json(ADMISSION / "candidates" / f"{key}.json", ledger)
        rows.append({"repo_key": key, "ledger": ledger})
    return rows


def write_stop_rule(families: list[str]) -> Path:
    """Apply the declared stop rule in family order to the processed calibration families."""
    from .selection_v4 import CALIBRATION_TARGET
    selection = json.loads(SELECTION.read_text())
    rows, total, stop_at = [], 0, None
    for family in selection["families"]:
        if family["split"] != "calibration":
            continue
        processed = family["repo_key"] in families
        ledger = json.loads((EVIDENCE_V4 / "candidates" / f"{family['repo_key']}.json").read_text()) if processed else []
        admitted = sum(r["status"] == "VALIDATED" and r.get("runner_check") == "passed" for r in ledger)
        in_prefix = stop_at is None
        if in_prefix:
            total += admitted
        rows.append({"order": family["order"], "repo_key": family["repo_key"], "v4_status": family["v4_status"],
                     "processed": processed, "admitted": admitted, "in_admitted_prefix": in_prefix,
                     "cumulative_admitted": total if in_prefix else None})
        if in_prefix and total >= CALIBRATION_TARGET:
            stop_at = family["order"]
    unprocessed_drawn = [r["repo_key"] for r in rows if r["v4_status"] == "DRAWN" and not r["processed"]
                         and r["in_admitted_prefix"]]
    record = {"rule": selection["criteria"]["family_extension"]["stop_rule"], "target": CALIBRATION_TARGET,
              "families": rows, "stopped_after_order": stop_at, "admitted_in_prefix": total,
              "target_reached": total >= CALIBRATION_TARGET,
              "drawn_calibration_families_not_processed": unprocessed_drawn,
              "note": ("target reached" if total >= CALIBRATION_TARGET else
                       "target not reached: every drawn calibration family through the last processed position was "
                       "processed; processing stopped there on the coordinator's instruction (tenacity and "
                       "exceptiongroup were the last planned families)")}
    path = ADMISSION / "stop-rule.json"
    write_json(path, record)
    return path


def write_registry(sealed: list[dict]) -> Path:
    v3 = json.loads(REGISTRY_V3.read_text())
    new = [{"task_id": r["task_id"], "family_id": r["family_id"], "adapter": ADAPTER_ENTRY, "admission": "ADMITTED",
            "package": r["package"], "task_record": r["task_record"], "attestation": r["attestation"],
            "split": r["split"], "predecessor": None, "new_in": "v4", "package_format": PACKAGE_FORMAT,
            "eligibility": {"development_and_calibration": r["split"] == "calibration", "held_out_main_evaluation": False}}
           for r in sorted(sealed, key=lambda r: r["task_id"])]
    tasks = sorted([dict(t) for t in v3["tasks"]] + new, key=lambda t: t["task_id"])
    path = ADMISSION / "registry.json"
    write_json(path, {"registry_version": "task-admission.registry.v1", "lane": "swesmith-long", "lane_version": "v4",
                      "supersedes": "benchmark_tasks/admission/swesmith-long-v3/registry.json (unchanged)",
                      "v3_registry_sha256": sha256(REGISTRY_V3),
                      "composition": {"from_v3_unchanged": len(v3["tasks"]), "new_in_v4": len(new)},
                      "artifact_root": "benchmark_tasks (default read_registry root for this path)", "tasks": tasks})
    return path


def _member(entry: dict) -> dict:
    package = ROOT / entry["package"]
    cls = SWESmithLongV4TaskAdapter if entry["adapter"] == ADAPTER_ENTRY else SWESmithLongV3TaskAdapter
    adapter = cls(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v4-cohort")
    record = adapter.describe()
    return {"task_id": entry["task_id"], "predecessor": entry.get("predecessor"), "family_id": entry["family_id"],
            "adapter": entry["adapter"], "new_in_v4": entry.get("new_in") == "v4",
            "attestation": {"path": "benchmark_tasks/" + entry["attestation"], "sha256": sha256(ROOT / entry["attestation"])},
            "task_record": {"path": "benchmark_tasks/" + entry["task_record"], "sha256": sha256(ROOT / entry["task_record"])},
            "package_manifest": {"path": "benchmark_tasks/" + entry["package"] + "/package.json",
                                 "sha256": sha256(package / "package.json")},
            "environment_version": record.environment_version, "environment_tag": adapter.manifest["environment_tag"],
            "evaluator_versions": [record.stages[0].evaluator_version], "stage_count": 1,
            "upstream_instance_id": adapter.manifest["upstream_instance_id"], "counts": adapter.manifest["counts"],
            "withheld_test_files": adapter.manifest["native_start"]["withheld_files"],
            "git_start_commit": adapter.git_start["commit"],
            "worker_prompt": {k: adapter.manifest["worker_prompt"][k] for k in ("version", "rendered_sha256")}}


def write_cohort(registry: Path) -> Path:
    entries = json.loads(registry.read_text())["tasks"]
    v3_cohort = {m["task_id"]: m for m in json.loads(COHORT_V3.read_text())["primary_tasks"]}
    members = []
    for entry in entries:
        if entry["split"] != "calibration":
            continue
        member = _member(entry)
        if not member["new_in_v4"]:
            # v3 members are carried verbatim; their bindings must still hold.
            if {k: v for k, v in member.items() if k != "new_in_v4"} != v3_cohort.get(entry["task_id"]):
                raise ValueError("v3 cohort member no longer matches its registry entry: " + entry["task_id"])
            member = {**v3_cohort[entry["task_id"]], "new_in_v4": False}
        members.append(member)
    v3_members = sorted(v3_cohort)
    new = sorted(m["task_id"] for m in members if m["new_in_v4"])
    if sorted(m["task_id"] for m in members if not m["new_in_v4"]) != v3_members:
        raise ValueError("v4 cohort must keep exactly the v3 calibration members")
    cohort = {"version": "calibration-cohort.lane.v1", "cohort_id": "swesmith-long-calibration-v4", "lane": "swesmith-long",
              "supersedes": "configs/calibration/cohort-swesmith-long-v3.json (unchanged)",
              "purpose": "calibration_preparation", "main_study_eligible": False, "model_ids": MODELS,
              "same_primary_members_for_each_model": True,
              "registry": {"path": "benchmark_tasks/" + rel(registry), "sha256": sha256(registry),
                           "artifact_root": "benchmark_tasks"},
              "membership_policy": {
                  "basis": "Every technically admitted calibration-split task of the v4 registry: the 15 v3 calibration "
                           "tasks (unchanged) plus the new v4 tasks (v3 package format) from calibration-position families "
                           "of the continued seeded family order.",
                  "new_members": "listed in new_task_ids and flagged new_in_v4",
                  "independence": "Cluster by family_id; several tasks share each repository family.",
                  "failures": "Retain every outcome; no outcome-based substitution."},
              "numeric_settings": {"status": "unfrozen", "required_before_each_launch": [
                  "per-task worker allowance", "model/provider settings", "detector settings", "call/tool limits"]},
              "new_task_ids": new, "v3_task_ids": v3_members,
              "primary_tasks": members, "primary_family_count": len({m["family_id"] for m in members}),
              "new_family_ids": sorted({m["family_id"] for m in members if m["new_in_v4"]})}
    write_json(COHORT, cohort)
    return COHORT


def verify_loads(attempt: str) -> dict:
    registry_path = ADMISSION / "registry.json"
    out = EVIDENCE_V4 / "load-verification" / attempt
    out.mkdir(parents=True, exist_ok=False)
    entries = read_registry(registry_path)
    cohort = json.loads(COHORT.read_text())
    if sha256(WORKTREE / cohort["registry"]["path"]) != cohort["registry"]["sha256"]:
        raise ValueError("Cohort registry binding changed")
    v3_registry = {t["task_id"]: t for t in json.loads(REGISTRY_V3.read_text())["tasks"]}
    cases = []
    for task_id, entry in sorted(entries.items()):
        scratch = Path(tempfile.mkdtemp(prefix="load-", dir=out))
        new = entry.entry.get("new_in") == "v4"
        case = {"task_id": task_id, "split": entry.entry["split"], "new_in_v4": new, "loads": {}, "refusals": {}}
        if not new:
            case["v3_entry_unchanged"] = v3_registry.get(task_id) == dict(entry.entry)
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
        cls = SWESmithLongV4TaskAdapter if new else SWESmithLongV3TaskAdapter
        adapter = cls(entry.path("package"), scratch_root=scratch)
        case["describe_admitted"] = adapter.describe().admission.value == "ADMITTED"
        workspace = scratch / "W"
        adapter.materialize(workspace)
        manifest = file_manifest(workspace)
        case["natural_start_equals_public_workspace_plus_git"] = (
            {k: v for k, v in manifest.items() if not k.startswith(".git/")} == file_manifest(adapter.package / "public/workspace")
            and sorted(k for k in manifest if k.startswith(".git/")) == sorted(
                m.name for m in tarfile.open(adapter.package / GIT_TAR).getmembers() if m.isfile()))
        case["withheld_absent_in_natural_start"] = not any((workspace / r["path"]).exists() for r in adapter.withheld["files"])
        row = json.loads((adapter.package / "private/source-row.json").read_text())
        case["instructions_byte_identical_to_dataset"] = (adapter.package / "public/instructions.md").read_bytes() == \
            row["problem_statement"].encode()
        if new:
            staged = PACKAGES_V2_STAGE / adapter.manifest["predecessor"]["task_id"]
            case["staged_v2_available"] = staged.is_dir()
            if staged.is_dir():
                case["workspace_files_identical_to_staged_v2"] = (file_manifest(staged / "public/workspace")
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
                         capture_output=True, text=True, timeout=600)
    listed = json.loads(cli.stdout) if cli.returncode == 0 else []
    template = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "mining-plan", "template",
                               "--cohort", str(COHORT), "--output", str(out / "mining-plan-template.json")],
                              capture_output=True, text=True, timeout=300)
    plan = json.loads((out / "mining-plan-template.json").read_text()) if (out / "mining-plan-template.json").is_file() else {}
    v3_cohort = json.loads(COHORT_V3.read_text())
    summary = {"schema_version": "swesmith-long.load-verification.v4", "registry_sha256": sha256(registry_path),
               "cohort_sha256": sha256(COHORT), "tasks": cases, "cli_tasks_returncode": cli.returncode,
               "cli_tasks_listed": len(listed), "provider_calls": 0, "cohort_members": len(cohort["primary_tasks"]),
               "cohort_new_members": len(cohort["new_task_ids"]),
               "cohort_keeps_v3_members": sorted(m["task_id"] for m in v3_cohort["primary_tasks"]) == cohort["v3_task_ids"],
               "cohort_model_ids": cohort["model_ids"], "mining_plan_template_returncode": template.returncode,
               "held_out_refused_all_purposes": all(not any(c["loads"].values()) for c in cases if c["split"] == "held_out"),
               "mining_plan_template_tasks": sorted(plan.get("admission", {})),
               "selection_sha256": sha256(ADMISSION / "selection.json"), "raw_selection_sha256": sha256(SELECTION)}
    summary["passed"] = (cli.returncode == 0 and len(listed) == len(entries) and summary["cohort_keeps_v3_members"]
                         and summary["mining_plan_template_tasks"] == sorted(m["task_id"] for m in cohort["primary_tasks"])
                         and all(c["partition_enforced"] and c["describe_admitted"]
                                 and c["natural_start_equals_public_workspace_plus_git"]
                                 and c["withheld_absent_in_natural_start"] and c["instructions_byte_identical_to_dataset"]
                                 and c["prompt_native_v3"] and c.get("v3_entry_unchanged", True)
                                 and c.get("workspace_files_identical_to_staged_v2", True) for c in cases))
    write_json(out / "summary.json", summary)
    write_json(ADMISSION / "load-verification.json", {k: v for k, v in summary.items() if k != "tasks"} | {
        "tasks": [{k: c[k] for k in ("task_id", "split", "new_in_v4", "loads", "partition_enforced", "describe_admitted",
                                     "prompt_sha256")} for c in cases]})
    return summary
