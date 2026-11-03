"""Issue technical admission seals, the lane registry and the calibration cohort file.

Admission here is technical (task.admission.attestation.v1 via tasks/evidence.py):
it requires a passed qualification bound to the exact package. Held-out family
tasks may be sealed ADMITTED but the registry/TaskRecord refuse development,
calibration and main loading (main stays closed until a separate study freeze).
"""
from __future__ import annotations

from dataclasses import replace
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from trajectory_retention.contracts import AdmissionStatus, to_jsonable
from trajectory_retention.registry import read_registry
from trajectory_retention.tasks.evidence import portable
from . import source as S
from .adapter import MettleBenchTaskAdapter
from .package import ADAPTER_ENTRY, BOUND_FILES, sha, verify_package, write_json
from .qualify import ADMISSION, ARTIFACTS, PACKAGES, REPO, natural_start

# v1 registry/cohort/load verification are sealed history and are never rewritten. New admissions go
# to v2, which carries every v1 entry unchanged and supersedes v1.
REGISTRY_V1 = ADMISSION / "registry-mettlebench-v1.json"
COHORT_V1 = REPO / "configs/calibration/cohort-mettlebench-v1.json"
REGISTRY = ADMISSION / "registry-mettlebench-v2.json"
COHORT = REPO / "configs/calibration/cohort-mettlebench-v2.json"
LOADS = REPO / "docs/research/mettlebench-admission/load-verification-v2.json"
CHECK_COHORT = REPO / "configs/calibration/check_cohort.py"


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO).as_posix()


def _binding(path: Path) -> dict:
    return {"path": _rel(path), "sha256": sha(path)}


def seed_registry() -> dict:
    """A new v2 registry holding every v1 entry byte-for-byte (v1 itself is left untouched)."""
    v1 = json.loads(REGISTRY_V1.read_text())
    return {"registry_version": "task-admission.registry.v1", "lane_registry_version": 2,
            "supersedes": _binding(REGISTRY_V1), "tasks": list(v1["tasks"])}


def eligibility(split: str) -> dict:
    return {"development_and_calibration": split == "calibration", "held_out_main_evaluation": False,
            "calibration_settings_frozen": False, "family_split": split}


def admit(upstream_id: str) -> dict:
    task_id = S.package_task_id(upstream_id)
    package = PACKAGES / task_id
    qualification = ADMISSION / task_id / "qualification.json"
    q = json.loads(qualification.read_text())
    manifest = verify_package(package)
    if not q.get("passed") or q.get("package_sha256") != sha(package / "package.json"):
        raise ValueError("Qualification failed or is bound to a different package: " + task_id)
    attestation = ADMISSION / f"{task_id}.json"
    if attestation.exists():
        raise ValueError("Attestation already issued; preserve it and issue a new version instead")
    adapter = MettleBenchTaskAdapter(package)
    provisional = adapter.describe()
    if provisional.admission != AdmissionStatus.HOLD:
        raise ValueError("Unexpected pre-seal admission state")
    split = manifest["family_split"]
    record = replace(provisional, admission=AdmissionStatus.ADMITTED, unresolved=(),
                     evidence={**provisional.evidence, "admission_attestation": _rel(attestation),
                               "qualification": _binding(qualification), "eligibility": eligibility(split),
                               "technical_admission_only": True, "main_study_frozen": False})
    record_path = ADMISSION / task_id / "task-record.json"
    write_json(record_path, portable(to_jsonable(record), REPO))
    sources = [MODULE / name for name in BOUND_FILES]
    data = {"record_version": "task.admission.attestation.v1", "admission": "ADMITTED", "task_id": task_id,
            "adapter": ADAPTER_ENTRY, "package": _rel(package), "package_sha256": sha(package / "package.json"),
            "evidence": [_binding(qualification)],
            "source_files": [_binding(p) for p in sources],
            "task_record": portable(to_jsonable(record), REPO),
            "review": {"lane": "mettlebench", "family_split": split, "split_rule": S.SPLIT_RULE,
                       "eligibility": eligibility(split), "gates": q["gates"], "summary": q["summary"],
                       "scope": ("Technical admission for development/calibration use. No provider calls; witness is an "
                                 "independent researcher script, not the gated upstream reference."),
                       "issued_at": dt.datetime.now(dt.timezone.utc).isoformat()}}
    write_json(attestation, data)
    entry = {"task_id": task_id, "family_id": manifest["family_id"], "adapter": ADAPTER_ENTRY, "admission": "ADMITTED",
             "package": _rel(package), "task_record": _rel(record_path), "attestation": _rel(attestation),
             "attestation_sha256": sha(attestation), "split": split, "eligibility": eligibility(split),
             "upstream_task_id": upstream_id, "source_benchmark": manifest["source_benchmark"]}
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else seed_registry()
    registry["tasks"] = [t for t in registry["tasks"] if t["task_id"] != task_id] + [entry]
    registry["tasks"].sort(key=lambda t: t["task_id"])
    write_json(REGISTRY, registry)
    loaded = read_registry(REGISTRY, root=REPO)[task_id]
    with tempfile.TemporaryDirectory() as tmp:
        if split == "calibration":
            loaded.load(scratch_root=Path(tmp), purpose="calibration")
        else:
            try:
                loaded.load(scratch_root=Path(tmp), purpose="development")
            except ValueError:
                pass
            else:
                raise ValueError("Held-out task loaded for development")
    return {"task_id": task_id, "split": split, "attestation": _rel(attestation)}


MODULE = Path(__file__).resolve().parent


def verify_loads() -> dict:
    entries = read_registry(REGISTRY, root=REPO)
    cli = subprocess.run([sys.executable, "-m", "trajectory_retention.cli", "tasks", "--registry", str(REGISTRY),
                          "--artifact-root", str(REPO)], capture_output=True, text=True, timeout=300)
    if cli.returncode:
        raise ValueError("Shared CLI cannot inspect the lane registry: " + cli.stderr[-2000:])
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        for task_id, entry in sorted(entries.items()):
            split = entry.entry["split"]
            outcome = {"task_id": task_id, "split": split}
            refusals = {}
            for purpose in ("development", "calibration", "main"):
                try:
                    adapter = entry.load(scratch_root=Path(tmp) / task_id / purpose, purpose=purpose)
                except ValueError as exc:
                    refusals[purpose] = str(exc)
                else:
                    outcome.setdefault("loaded", []).append(purpose)
                    if purpose == "calibration":
                        record = adapter.describe()
                        ns = natural_start(adapter)
                        outcome.update(admission=record.admission.value, natural_start_equal=ns["equal_to_archive"],
                                       environment_version=record.environment_version,
                                       evaluator_version=record.stages[0].evaluator_version)
            outcome["refused"] = refusals
            expected = {"development", "calibration"} if split == "calibration" else set()
            outcome["ok"] = set(outcome.get("loaded", [])) == expected and "main" in refusals and (
                split != "calibration" or outcome.get("natural_start_equal") is True)
            cases.append(outcome)
    summary = {"schema_version": "mettlebench.load-verification.v1", "registry": _rel(REGISTRY),
               "registry_sha256": sha(REGISTRY), "shared_cli_tasks": len(json.loads(cli.stdout)),
               "tasks": cases, "passed": all(c["ok"] for c in cases), "provider_calls": 0,
               "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(LOADS, summary)
    return {"passed": summary["passed"], "tasks": len(cases), "path": _rel(LOADS)}


def write_cohort(model_ids=("openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"), seed: int = 20260924) -> dict:
    """Cohort v2: every admitted calibration-split task is primary (no reserves); v1 is left unchanged."""
    registry = json.loads(REGISTRY.read_text())
    v1_members = {e["task_id"] for e in json.loads(REGISTRY_V1.read_text())["tasks"]}
    rows = []
    for entry in registry["tasks"]:
        if entry["split"] != "calibration" or entry["admission"] != "ADMITTED":
            continue
        manifest = json.loads((REPO / entry["package"] / "package.json").read_text())
        adapter = MettleBenchTaskAdapter(REPO / entry["package"])
        record = adapter.describe()
        rows.append({"task_id": entry["task_id"], "adapter": entry["adapter"], "family_id": entry["family_id"],
                     "attestation": _binding(REPO / entry["attestation"]),
                     "task_record": _binding(REPO / entry["task_record"]),
                     "package_manifest": _binding(REPO / entry["package"] / "package.json"),
                     "environment_version": record.environment_version,
                     "evaluator_versions": [record.stages[0].evaluator_version], "stage_count": 1,
                     "upstream_task_id": manifest["upstream_task_id"], "criteria": manifest["criteria"],
                     "source_benchmark": manifest["source_benchmark"]})
    # Declared order only (launch plans choose their own schedule): seeded shuffle over sorted task ids.
    primary = sorted(rows, key=lambda r: r["task_id"])
    random.Random(seed).shuffle(primary)
    new = sorted(r["task_id"] for r in primary if r["task_id"] not in v1_members)
    instrument = sorted(set(str(p.relative_to(REPO)) for p in (REPO / "src/trajectory_retention").rglob("*.py")))
    cohort = {
        "version": "calibration-cohort.v1", "cohort_id": "calibration-mettlebench-2026-09-26-v2",
        "declared_at": dt.datetime.now(dt.timezone.utc).isoformat(), "purpose": "calibration_preparation",
        "main_study_eligible": False, "model_ids": list(model_ids), "same_primary_members_for_each_model": True,
        "registry_at_declaration": _binding(REGISTRY),
        "supersedes": {**_binding(COHORT_V1), "note": "cohort v1 is unchanged; its ten members are all primary here"},
        "primary_tasks": primary, "reserve_tasks": [], "deferred_reserve_tasks": [],
        "new_since_v1": new,
        "carried_from_v1": sorted(r["task_id"] for r in primary if r["task_id"] in v1_members),
        "membership_policy": {
            "basis": ("All technically admitted MettleBench calibration-split tasks are primary, as requested for the "
                      "next fresh calibration batch (lane request 2026-09-26); no reserves. List order is a seeded "
                      f"shuffle (random.Random({seed}) over sorted task ids) and is not a schedule."),
            "independence": "Cluster by recorded family (source project / Terminal-Bench source task).",
            "failures": "Retain easy solves, no-warning runs, technical failures and censored runs; no outcome-based substitution.",
            "stage_scope": "Single native assignment; native Continue submission feedback; no stage resets.",
            "primary_count_note": ("configs/calibration/check_cohort.py expects exactly ten primaries; this cohort "
                                   "deliberately has more. Member checks: mettlebench_v1 verify-cohort.")},
        "numeric_settings": {"status": "unfrozen", "required_before_each_launch": [
            "explicit per-task worker allowance", "model/provider settings", "detector thresholds",
            "call/idle/tool limits", "repeat schedule and resource limits"]},
        "qualified_instrument_sources": [_binding(REPO / p) for p in ["pyproject.toml", *instrument]],
        "remaining_scope": {"inventory_is_not_batch_plan": True, "calibration_already_completed": False,
                            "main_freeze_issued": False,
                            "note": "Binds membership and instrument versions only; explicit Campaign/BatchPlan files are required to spend compute."}}
    write_json(COHORT, cohort)
    return {"primary": len(primary), "new_since_v1": len(new), "families": len({r["family_id"] for r in primary}),
            "path": _rel(COHORT)}


def verify_cohort(path: Path = COHORT) -> dict:
    """Run the shared configs/calibration/check_cohort.py on the cohort.

    That checker also requires exactly ten primaries. The per-member checks (bindings, registry identity,
    development/calibration load, main refusal) are the same code for primary and reserve rows, so they are
    run over a temporary view in which the first ten members are labelled primary and the rest reserve.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("mettlebench_check_cohort", CHECK_COHORT)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    data = json.loads(Path(path).read_bytes())
    direct: dict = {"checker": _binding(CHECK_COHORT)}
    try:
        checker.check(Path(path), REPO)
    except ValueError as exc:
        direct.update(passed=False, error=str(exc))
    else:
        direct["passed"] = True
    members = data["primary_tasks"] + data["reserve_tasks"]
    view = {**data, "primary_tasks": members[:10], "reserve_tasks": members[10:]}
    S.DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cohort-view-", dir=S.DEFAULT_CACHE) as tmp:
        view_path = Path(tmp) / "cohort-view.json"
        view_path.write_text(json.dumps(view))
        members_result = checker.check(view_path, REPO)
    rows = [{k: v for k, v in r.items() if k != "role"} for r in members_result["tasks"]]
    summary = {"schema_version": "mettlebench.cohort-check.v1", "cohort": _binding(Path(path)),
               "shared_checker_direct": direct,
               "member_checks": {"passed": members_result["passed"], "members": len(rows),
                                 "families": len({r["family_id"] for r in rows}), "tasks": rows},
               "provider_calls": 0, "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    out = LOADS.with_name("cohort-check-v2.json")
    write_json(out, summary)
    return {"direct_passed": direct["passed"], "direct_error": direct.get("error"),
            "member_checks_passed": members_result["passed"], "members": len(rows), "path": _rel(out)}
