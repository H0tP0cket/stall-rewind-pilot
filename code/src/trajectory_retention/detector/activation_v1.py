"""Explicit activation of certified frontier through the ordinary campaign CLI.

The launch config selects the instrument; the admitted task declares its native
certificate provider and per-stage policies. Neither side can substitute for the
other. Unselected campaigns keep their existing engine, source and behavior.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path

from ..contracts import from_jsonable, to_jsonable
from .threshold_policy_v1 import DetectorThresholdProfile, validate_detector_assignments
from ..scoring.certified_frontier_v1 import CertifiedFrontierPolicy

KEY = "certified_detector_v1"
VERSION = "certified-detector-activation/v1"


def selected(config) -> bool:
    return KEY in config.mining.metadata


def configuration(config):
    """Validate opt-in config without loading a task, provider or environment."""
    value = config.mining.metadata.get(KEY)
    if not isinstance(value, dict) or set(value) != {"version", "threshold_profile", "assignments"}:
        raise ValueError("Certified detector requires its exact versioned activation object")
    if value["version"] != VERSION:
        raise ValueError("Unsupported certified detector activation version")
    if (config.mode != "mine" or config.source_natural_run_id is not None
            or config.mining.purpose == "main"):
        raise ValueError("Certified detector v1 is fresh development/calibration mining only")
    profile = from_jsonable(DetectorThresholdProfile, value["threshold_profile"])
    report = validate_detector_assignments(profile, assignments=value["assignments"])
    return profile, report


def validate_batch(plan):
    """Check all selected repeats and cross-worker settings before any job starts.

An opted-in batch is one instrument/profile. Every emitted repeat is checked
before deduplicating task/model cells. Ordinary historical batches are unchanged.
"""
    flags = [selected(job.config) for job in plan.jobs]
    if not any(flags):
        return None
    if not all(flags):
        raise ValueError("A certified detector batch cannot mix selected and legacy instruments")
    baseline, rows, jobs = None, {}, []
    for job in plan.jobs:
        _, report = configuration(job.config)
        if baseline is None:
            baseline = report
        elif report != baseline:
            raise ValueError("Certified batch threshold profile/assignments differ between jobs")
        matches = [row for row in report["assignments"]
                   if (row["task_id"], row["model"]) == (job.task_id, job.config.mining.model)]
        if len(matches) != 1 or matches[0]["detector_settings"] != dict(job.config.mining.detector_settings):
            raise ValueError("Every emitted task/model/repeat must match its threshold assignment")
        row = matches[0]
        key = (row["task_id"], row["model"])
        if key in rows and rows[key] != row:
            raise ValueError("Repeated detector cell changed configuration")
        rows[key] = row
        jobs.append({"campaign_id": job.config.campaign_id, "run_id": job.config.mining.run_id,
                     "repeat_id": job.config.mining.repeat_id, **row})
    return {"version": VERSION, "threshold_binding": baseline, "executed_jobs": jobs}


def build_runner(store, task, config, *, scratch: Path, api_key_env="OPENROUTER_API_KEY"):
    """Construct the real candidate runner only from task-declared native proof.

This happens before public route metadata lookup or worker initialization. Missing
task certificates are an eligibility error, never inferred from status/old flags.
The ordinary Campaign still owns admission, source closure, live T and exports.
"""
    profile, report = configuration(config)
    record = task.describe()
    from ..registry import validate_task_purpose
    validate_task_purpose(record, config.mining.purpose)
    declaration = record.evidence.get(KEY)
    if not isinstance(declaration, dict) or set(declaration) != {"version", "policies", "certificate_provider"}:
        raise ValueError("Task has no frozen certified-frontier activation declaration")
    if declaration["version"] != VERSION:
        raise ValueError("Task certified-frontier activation version differs")
    policies = {stage: from_jsonable(CertifiedFrontierPolicy, value)
                for stage, value in declaration["policies"].items()}
    provider = declaration["certificate_provider"]
    if not isinstance(provider, dict) or set(provider) != {"entrypoint", "source_sha256"}:
        raise ValueError("Task must declare its exact native certificate provider source")
    entrypoint = provider["entrypoint"]
    if (not isinstance(entrypoint, str) or entrypoint.count(":") != 1
            or not entrypoint.startswith("trajectory_retention.tasks.")):
        raise ValueError("Native certificate provider must be an explicit task module function")
    module_name, name = entrypoint.split(":")
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError("Native certificate provider must be a public module-level function")
    function = getattr(importlib.import_module(module_name), name, None)
    source = inspect.getsourcefile(function) if inspect.isfunction(function) else None
    if (source is None or function.__module__ != module_name or function.__name__ != name
            or function.__closure__):
        raise ValueError("Native certificate provider does not match its source-backed entrypoint")
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != provider["source_sha256"]:
        raise ValueError("Native certificate provider source digest differs from task binding")
    for ref in profile.evidence_refs:
        store.get_bytes(ref)
    from ..calibration_candidate_v1 import CandidateMiningRunnerV1
    runner = CandidateMiningRunnerV1(task, config.mining, threshold_profile=profile,
        assignments=report["assignments"], frontier_policies=policies,
        certificate_provider=function, scratch=scratch, api_key_env=api_key_env)
    # The run binding also archives native validator bytes. This campaign record
    # links the explicit opt-in to ordinary factory/CLI provenance before calls.
    store.put_record("certified_detector_activation_v1", config.campaign_id,
        {"version": VERSION, "task_id": record.task_id, "task_version": record.task_version,
         "task_declaration": declaration, "threshold_binding": report,
         "activation_source_ref": to_jsonable(store.put_bytes(Path(__file__).read_bytes()))})
    return runner
