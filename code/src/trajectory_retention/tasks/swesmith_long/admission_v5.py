"""Seal the v5-release tasks, write the v5 registry/cohort and verify loads (no model calls).

Registry v5 = the 50 v4 entries copied unchanged (adapter_v3/adapter_v4, their seals)
plus the new v5 tasks (adapter_v5, seals under admission/swesmith-long-v5).
Cohort v5 = only the tasks intended for the next calibration batch: the new
v5 tasks plus the kept earlier calibration tasks (every v4 cohort member except
the ratified comment-leak exclusions). v1-v4 registries, cohorts, seals and
packages are not modified.
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
from .adapter_v4 import ADAPTER_ENTRY as V4_ENTRY, SWESmithLongV4TaskAdapter
from .adapter_v5 import ADAPTER_ENTRY, PACKAGE_FORMAT, SEAL_SCHEMA, SWESmithLongV5TaskAdapter, loader_sources
from .pipeline import EVIDENCE, WORKTREE, write_json
from .pipeline_v5 import (ADMISSION, EVIDENCE_V5, LANE, LEAKS, PACKAGES, PACKAGES_V2_STAGE, SELECTION,
                          STAGE_EVIDENCE_V3)

ROOT = WORKTREE / "benchmark_tasks"
REGISTRY_V4 = ROOT / "admission" / "swesmith-long-v4" / "registry.json"
COHORT_V4 = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v4.json"
COHORT = WORKTREE / "configs" / "calibration" / "cohort-swesmith-long-v5.json"
MODELS = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]
COMMENT_LEAK = LEAKS / "comment-leak.json"


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def adapter_for(entry: str):
    return {ADAPTER_ENTRY: SWESmithLongV5TaskAdapter, V4_ENTRY: SWESmithLongV4TaskAdapter}.get(
        entry, SWESmithLongV3TaskAdapter)


def compact_family(repo_key: str) -> Path:
    base = json.loads((EVIDENCE / "families" / repo_key / "family.json").read_text())
    keep = {k: base.get(k) for k in ("repo_key", "family_id", "split", "status", "official_image", "official", "mirror",
                                      "environment", "generated_files", "native_test_command", "pytest_args",
                                      "reference_runs", "reference_stable_pass", "reference_flaky", "project_top_levels",
                                      "environment_attempts")}
    scan = base.get("image_leak_scan", {})
    keep["image_leak_scan"] = {k: scan.get(k) for k in ("files_scanned", "identical_to_clean_source", "project_package_copies",
                                                          "project_archives", "pip_cache_files", "leak", "rule")}
    v3_dir = STAGE_EVIDENCE_V3 / "families" / repo_key
    needles = json.loads((v3_dir / "image-needles.json").read_text())
    path = ADMISSION / "families" / f"{repo_key}.json"
    write_json(path, {"base_environment": keep, "git_image": json.loads((v3_dir / "image.json").read_text()),
                      "image_needles": {k: needles[k] for k in ("image_id", "files_scanned", "needles", "hits")},
                      "clean_source_scan": json.loads((v3_dir / "image-leak-scan.json").read_text()),
                      "v5_environment": json.loads((EVIDENCE_V5 / "families" / repo_key / "environment.json").read_text())})
    return path


def seal(task_id: str) -> dict:
    record = json.loads((EVIDENCE_V5 / "tasks" / task_id / "package-record.json").read_text())
    runner = json.loads((EVIDENCE_V5 / "tasks" / task_id / "runner-check" / "summary.json").read_text())
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
    rationale = out / "stall-rationale.json"
    if not rationale.is_file():
        raise ValueError(f"{task_id}: stall rationale missing")
    family_path = compact_family(record["repo_key"])
    ledger_path = ADMISSION / "candidates" / f"{record['repo_key']}.json"
    split = record["split"]
    evidence = [validation, runner_path, rationale, family_path, ADMISSION / "selection.json",
                ADMISSION / "issue-review.json", ADMISSION / "external-evidence.json",
                ADMISSION / "comment-leak-review.json", ledger_path, ADMISSION / "stop-rule.json"]
    admission = {"schema_version": SEAL_SCHEMA, "admission": "ADMITTED", "task_id": task_id,
                 "package_format": PACKAGE_FORMAT, "family_id": record["family_id"], "package": rel(package),
                 "package_sha256": sha256(package / "package.json"), "split": split,
                 "eligibility": {"development_and_calibration": split == "calibration", "held_out_main_evaluation": False},
                 "main_study_frozen": False, "evaluator_sources": evaluator_sources(), "loader_sources": loader_sources(),
                 "evidence": [{"path": rel(p), "sha256": sha256(p)} for p in evidence],
                 "verdict": ("Technically admitted (v3 package format: native SWE-smith start with withheld FAIL_TO_PASS "
                             "files in a one-commit Git repository, native instance template); new in lane release v5 "
                             "(published-hard-band selection); calibration/development only"),
                 "not_established": ["model difficulty/solve rate on our workers (no model calls)", "stall yield",
                                     "main-study eligibility (settings unfrozen)"]}
    write_json(out / "admission.json", admission)
    adapter = SWESmithLongV5TaskAdapter(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v5-seal")
    described = adapter.describe()
    if described.admission.value != "ADMITTED":
        raise ValueError("Seal did not verify")
    write_json(out / "task-record.json", to_jsonable(described))
    return {"task_id": task_id, "family_id": record["family_id"], "split": split, "package": rel(package),
            "task_record": rel(out / "task-record.json"), "attestation": rel(out / "admission.json")}


def write_ledgers(families: list[str]) -> None:
    for key in families:
        write_json(ADMISSION / "candidates" / f"{key}.json",
                   json.loads((EVIDENCE_V5 / "candidates" / f"{key}.json").read_text()))


def write_stop_rule(families: list[str]) -> Path:
    """Record the walk outcome in rank order: admitted, held, not needed (family cap), stop point."""
    from .selection_v5 import CALIBRATION_TARGET, PER_FAMILY_CAP
    selection = json.loads(SELECTION.read_text())
    outcomes = {}
    for key in families:
        for row in json.loads((EVIDENCE_V5 / "candidates" / f"{key}.json").read_text()):
            outcomes[row["instance_id"]] = row
    rows, admitted, per_family, stop_rank = [], 0, {}, None
    for step in selection["ranked"]:
        row = {"rank": step["rank"], "tier": step["tier"], "instance_id": step["instance_id"], "repo_key": step["repo_key"],
               "selection_status": step["status"]}
        if step["status"] == "ELIGIBLE":
            built = outcomes.get(step["instance_id"])
            if stop_rank is not None:
                row["outcome"] = "NOT_NEEDED_TARGET_MET"
            elif built is None:
                row["outcome"] = ("NOT_NEEDED_FAMILY_CAP" if per_family.get(step["repo_key"], 0) >= PER_FAMILY_CAP
                                  else "NOT_BUILT")
            else:
                row["outcome"] = "ADMITTED" if built["status"] == "VALIDATED" else "HOLD"
                row["task_id"] = built.get("task_id")
                if row["outcome"] == "HOLD":
                    row["hold"] = {"stage": built.get("stage"), "reason": built.get("reason")}
                else:
                    admitted += 1
                    per_family[step["repo_key"]] = per_family.get(step["repo_key"], 0) + 1
                    if admitted >= CALIBRATION_TARGET:
                        stop_rank = step["rank"]
            row["cumulative_admitted"] = admitted
        rows.append(row)
    record = {"rule": selection["criteria"]["stop_rule"], "target": CALIBRATION_TARGET, "family_cap": PER_FAMILY_CAP,
              "walk": rows, "admitted": admitted, "target_reached": admitted >= CALIBRATION_TARGET,
              "stopped_at_rank": stop_rank if stop_rank is not None else rows[-1]["rank"] if rows else None,
              "note": ("target reached" if admitted >= CALIBRATION_TARGET else
                       "target not reached: every eligible instance of the ranked list was built (within the family "
                       "cap) or not needed because its family was full; the eligible list is exhausted"),
              "not_built": [r["instance_id"] for r in rows if r.get("outcome") == "NOT_BUILT"],
              "pending_i2_not_needed": {r["instance_id"]: "not reviewed: on the stall-supply comment-leak exclusion list "
                                        "(excluded regardless of I2)" for r in rows
                                        if r["selection_status"] == "PENDING_I2"},
              "line_profiler_note": ("every built line_profiler instance failed the unchanged runner check "
                                     "(checkpoint_B_grade_differs_from_A): the fixture's edit B appends a raising line to "
                                     "the first bug file (a line_profiler/autoprofile module); only the withheld "
                                     "FAIL_TO_PASS file tests/test_autoprofile.py exercises autoprofile, and it already "
                                     "fails at A, while no PASS_TO_PASS test imports the module, so A and B grade "
                                     "identically (deterministic, not flaky). The restore round trip itself passed. "
                                     "Held; the gate was not loosened.")}
    path = ADMISSION / "stop-rule.json"
    write_json(path, record)
    return path


def write_registry(sealed: list[dict]) -> Path:
    v4 = json.loads(REGISTRY_V4.read_text())
    new = [{"task_id": r["task_id"], "family_id": r["family_id"], "adapter": ADAPTER_ENTRY, "admission": "ADMITTED",
            "package": r["package"], "task_record": r["task_record"], "attestation": r["attestation"],
            "split": r["split"], "predecessor": None, "new_in": "v5", "package_format": PACKAGE_FORMAT,
            "eligibility": {"development_and_calibration": r["split"] == "calibration", "held_out_main_evaluation": False}}
           for r in sorted(sealed, key=lambda r: r["task_id"])]
    tasks = sorted([dict(t) for t in v4["tasks"]] + new, key=lambda t: t["task_id"])
    path = ADMISSION / "registry.json"
    write_json(path, {"registry_version": "task-admission.registry.v1", "lane": "swesmith-long", "lane_version": "v5",
                      "supersedes": "benchmark_tasks/admission/swesmith-long-v4/registry.json (unchanged)",
                      "v4_registry_sha256": sha256(REGISTRY_V4),
                      "composition": {"from_v4_unchanged": len(v4["tasks"]), "new_in_v5": len(new)},
                      "artifact_root": "benchmark_tasks (default read_registry root for this path)", "tasks": tasks})
    return path


def comment_leak_exclusions() -> list[str]:
    data = json.loads(COMMENT_LEAK.read_text())
    return sorted(data["confirmed_exclusions_calibration"])


def _member(entry: dict) -> dict:
    package = ROOT / entry["package"]
    adapter = adapter_for(entry["adapter"])(package, scratch_root=Path(tempfile.gettempdir()) / "swesmith-long-v5-cohort")
    record = adapter.describe()
    return {"task_id": entry["task_id"], "predecessor": entry.get("predecessor"), "family_id": entry["family_id"],
            "adapter": entry["adapter"], "new_in_v5": entry.get("new_in") == "v5",
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


def write_cohort(registry: Path, egg_info: dict) -> Path:
    entries = {e["task_id"]: e for e in json.loads(registry.read_text())["tasks"]}
    v4_cohort = json.loads(COHORT_V4.read_text())
    v4_members = {m["task_id"]: m for m in v4_cohort["primary_tasks"]}
    excluded = comment_leak_exclusions()
    if not set(excluded) <= set(v4_members):
        raise ValueError("comment-leak exclusions are not all v4 cohort members")
    kept = sorted(set(v4_members) - set(excluded))
    new = sorted(t for t, e in entries.items() if e.get("new_in") == "v5" and e["split"] == "calibration")
    members = []
    for task_id in kept:
        member = _member(entries[task_id])
        member.pop("new_in_v5")
        if member != {k: v for k, v in v4_members[task_id].items() if k != "new_in_v4"}:
            raise ValueError("kept v4 cohort member no longer matches its registry entry: " + task_id)
        members.append({**v4_members[task_id], "new_in_v5": False})
    members += [_member(entries[t]) for t in new]
    cohort = {"version": "calibration-cohort.lane.v1", "cohort_id": "swesmith-long-calibration-v5", "lane": "swesmith-long",
              "supersedes": "configs/calibration/cohort-swesmith-long-v4.json (unchanged)",
              "purpose": "calibration_preparation", "main_study_eligible": False, "model_ids": MODELS,
              "same_primary_members_for_each_model": True,
              "registry": {"path": "benchmark_tasks/" + rel(registry), "sha256": sha256(registry),
                           "artifact_root": "benchmark_tasks"},
              "membership_policy": {
                  "basis": "Only the tasks intended for the next calibration batch: the new v5 tasks (published-hard-band "
                           "selection, v3 package format) plus the kept earlier calibration tasks = every v4 cohort member "
                           "except the ratified comment-leak exclusions (bug patch adds comments describing the bugs).",
                  "comment_leak_exclusions": {"source": "benchmark_tasks/admission/stall-supply-v1/swesmith-leaks/"
                                                        "comment-leak.json (branch claude/stall-supply)",
                                              "sha256": sha256(COMMENT_LEAK), "excluded_task_ids": excluded,
                                              "decision": json.loads(COMMENT_LEAK.read_text())["ratified_decision"]},
                  "egg_info_listing": egg_info,
                  "new_members": "listed in new_task_ids and flagged new_in_v5",
                  "independence": "Cluster by family_id; several tasks share each repository family.",
                  "failures": "Retain every outcome; no outcome-based substitution."},
              "numeric_settings": {"status": "unfrozen", "required_before_each_launch": [
                  "per-task worker allowance", "model/provider settings", "detector settings", "call/tool limits"]},
              "new_task_ids": new, "kept_task_ids": kept,
              "primary_tasks": members, "primary_family_count": len({m["family_id"] for m in members}),
              "new_family_ids": sorted({m["family_id"] for m in members if m["new_in_v5"]})}
    write_json(COHORT, cohort)
    return COHORT


def verify_loads(attempt: str) -> dict:
    registry_path = ADMISSION / "registry.json"
    out = EVIDENCE_V5 / "load-verification" / attempt
    out.mkdir(parents=True, exist_ok=False)
    entries = read_registry(registry_path)
    cohort = json.loads(COHORT.read_text())
    if sha256(WORKTREE / cohort["registry"]["path"]) != cohort["registry"]["sha256"]:
        raise ValueError("Cohort registry binding changed")
    v4_registry = {t["task_id"]: t for t in json.loads(REGISTRY_V4.read_text())["tasks"]}
    cases = []
    for task_id, entry in sorted(entries.items()):
        scratch = Path(tempfile.mkdtemp(prefix="load-", dir=out))
        new = entry.entry.get("new_in") == "v5"
        case = {"task_id": task_id, "split": entry.entry["split"], "new_in_v5": new, "loads": {}, "refusals": {}}
        if not new:
            case["v4_entry_unchanged"] = v4_registry.get(task_id) == dict(entry.entry)
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
        adapter = adapter_for(entry.entry["adapter"])(entry.path("package"), scratch_root=scratch)
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
    by_id = {c["task_id"]: c for c in cases}
    summary = {"schema_version": "swesmith-long.load-verification.v5", "registry_sha256": sha256(registry_path),
               "cohort_sha256": sha256(COHORT), "cli_tasks_returncode": cli.returncode,
               "cli_tasks_listed": len(listed), "provider_calls": 0, "cohort_members": len(cohort["primary_tasks"]),
               "cohort_new_members": len(cohort["new_task_ids"]), "cohort_kept_members": len(cohort["kept_task_ids"]),
               "cohort_model_ids": cohort["model_ids"], "mining_plan_template_returncode": template.returncode,
               "cohort_members_all_calibration": all(by_id[m["task_id"]]["split"] == "calibration"
                                                     for m in cohort["primary_tasks"]),
               "cohort_excludes_comment_leak_tasks": not set(comment_leak_exclusions()) & {
                   m["task_id"] for m in cohort["primary_tasks"]},
               "held_out_refused_all_purposes": all(not any(c["loads"].values()) for c in cases if c["split"] == "held_out"),
               "mining_plan_template_tasks": sorted(plan.get("admission", {})),
               "selection_sha256": sha256(ADMISSION / "selection.json")}
    summary["passed"] = (cli.returncode == 0 and len(listed) == len(entries) and summary["cohort_members_all_calibration"]
                         and summary["cohort_excludes_comment_leak_tasks"]
                         and summary["mining_plan_template_tasks"] == sorted(m["task_id"] for m in cohort["primary_tasks"])
                         and all(c["partition_enforced"] and c["describe_admitted"]
                                 and c["natural_start_equals_public_workspace_plus_git"]
                                 and c["withheld_absent_in_natural_start"] and c["instructions_byte_identical_to_dataset"]
                                 and c["prompt_native_v3"] and c.get("v4_entry_unchanged", True)
                                 and c.get("workspace_files_identical_to_staged_v2", True) for c in cases))
    write_json(out / "summary.json", {**summary, "tasks": cases})
    write_json(ADMISSION / "load-verification.json", summary | {
        "tasks": [{k: c[k] for k in ("task_id", "split", "new_in_v5", "loads", "partition_enforced", "describe_admitted",
                                     "prompt_sha256")} for c in cases]})
    return summary


# --------------------------------------------------------------------------- egg-info SOURCES.txt listing
NATIVE_EGGINFO = Path.home() / ".cache/trajectory-retention/swesmith-v5/native-egginfo"


def egg_info_new_tasks(task_ids: list[str]) -> Path:
    """For each new v5 task: withheld paths named by our start tree's metadata vs by the native official image.

    Native side: the official image's /testbed/**/*.egg-info/SOURCES.txt, streamed from Docker Hub (HTTP only; the
    stall-supply script native_egginfo.py with a v5 output directory). Ours: public/workspace/**/SOURCES.txt.
    """
    from .pipeline import official_image
    rows = []
    for task_id in task_ids:
        package = PACKAGES / task_id
        manifest = json.loads((package / "package.json").read_text())
        withheld = [r["path"] for r in json.loads((package / "private/withheld.json").read_text())["files"]]
        ours = set()
        for path in (package / "public/workspace").rglob("SOURCES.txt"):
            lines = set(path.read_text(errors="ignore").splitlines())
            ours |= {w for w in withheld if w in lines}
        native_dir = NATIVE_EGGINFO / official_image(manifest["repo_key"]).split("swesmith.x86_64.", 1)[-1]
        native, measured = set(), (native_dir / "native.json").is_file()
        if measured:
            for path in (native_dir / "files").rglob("SOURCES.txt"):
                lines = set(path.read_text(errors="ignore").splitlines())
                native |= {w for w in withheld if w in lines}
        rows.append({"task_id": task_id, "withheld_files": withheld, "ours_names": sorted(ours),
                     "native_measured": measured, "native_names": sorted(native) if measured else None,
                     "ours_not_native": sorted(ours - native) if measured else None})
    path = ADMISSION / "egg-info-new-tasks.json"
    write_json(path, {"schema_version": "swesmith-long.egg-info-listing.v5",
                      "rule": "names only: which withheld FAIL_TO_PASS file paths appear as lines of a SOURCES.txt",
                      "tasks": rows,
                      "all_measured": all(r["native_measured"] for r in rows),
                      "ours_never_names_more_than_native": all(r["ours_not_native"] == [] for r in rows)})
    return path


def egg_info_summary(egg: dict | None) -> dict:
    if egg is None:
        raise ValueError("egg-info verdict missing")
    new = ADMISSION / "egg-info-new-tasks.json"
    return {"verdict_source": "benchmark_tasks/admission/stall-supply-v1/swesmith-leaks/egg-info.json (claude/stall-supply)",
            "verdict_sha256": sha256(LEAKS / "egg-info.json"), "decision": egg["verdict"]["decision"],
            "reissue": "none: the listing is native (same withheld names natively and in ours for all 32 v4 calibration "
                       "tasks), so kept v3/v4 tasks stay unchanged and v5 keeps the v3 package format",
            "new_tasks_check": {"path": "benchmark_tasks/" + rel(new), "sha256": sha256(new)} if new.is_file() else None}
