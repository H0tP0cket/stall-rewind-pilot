"""Replaceable private status views over immutable execution receipts."""
from __future__ import annotations

import fcntl
import hashlib
from html import escape
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from .accounting import CostLedger, summarize
from .contracts import ArtifactRef, from_jsonable
from .store import ArtifactStore


def _records(store: ArtifactStore, kind: str) -> dict[str, Any]:
    return {r["record_id"]: store.get_record(kind, r["record_id"]) for r in store.list_records(kind)}


def _phase_state(store: ArtifactStore, identity: str, started: Any, finished: Any) -> str:
    if finished is not None:
        return "completed" if finished.get("status") == "ok" else "failed"
    if started is None:
        return "pending"
    lockfile = store.root / "locks" / hashlib.sha256(identity.encode()).hexdigest()
    try:
        with lockfile.open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "active"
            fcntl.flock(lock, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass
    return "uncertain"


def status_report(store: ArtifactStore, *, launch_plans: Sequence[Mapping[str, Any]] = (),
                  admissions: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Read receipts without creating accounting snapshots or changing phases.

    Active means an owned phase lock is currently held; an abandoned started
    phase is uncertain. This observation does not authorize replay.
    """
    results = _records(store, "run_result")
    with sqlite3.connect(store.database.as_uri() + "?mode=ro", uri=True) as db:
        run_ids = {row[0] for row in db.execute(
            "SELECT DISTINCT run_id FROM events UNION SELECT DISTINCT run_id FROM records WHERE run_id IS NOT NULL")}
    run_ids.update(results)
    runs, incidents = [], []
    for run_id in sorted(run_ids):
        run = results.get(run_id, {})
        events = list(store.iter_events(run_id))
        costs = CostLedger(store).records(run_id)
        accounting = summarize(costs)
        for event in events:
            if event.kind == "incident":
                incidents.append({"run_id": run_id, "event_id": event.event_id,
                    "sequence": event.sequence, "details": dict(event.payload)})
        evaluation = {}
        if run.get("final_evaluation_ref"):
            try:
                evaluation = store.get_json(from_jsonable(ArtifactRef, run["final_evaluation_ref"]))
            except (KeyError, ValueError, OSError) as exc:
                evaluation = {"read_error": type(exc).__name__}
        runs.append({"run_id": run_id, "status": run.get("status", "no_final_receipt"),
            "stop_reason": run.get("termination_reason"), "checkpoint_count": len(store.list_checkpoints(run_id)),
            "last_recorded_evaluation": evaluation, "accounting": accounting,
            "accounting_interrupted": (run.get("termination_reason") == "accounting_interruption" or
                any(e.payload.get("what") == "worker_accounting_interruption" for e in events)),
            "uncertain_operations": [c.operation_id for c in costs if c.status == "uncertain"],
            "worker_usage_missing_operations": accounting["worker_generated_tokens"]["missing_operations"],
            "source": "all recorded operations, including partial work and closure; no inferred zero usage"})
    starts, finishes, specs = (_records(store, kind) for kind in ("phase_started", "phase_finished", "phase_spec"))
    phases = [{"phase_id": identity, "status": _phase_state(store, identity, starts.get(identity), finishes.get(identity)),
               "started": starts.get(identity), "finished": finishes.get(identity)}
              for identity in sorted(set(starts) | set(finishes) | set(specs))]
    campaigns = []
    from .observers.live import live_observer_status
    live_coverage = []
    for identity, campaign in _records(store, "campaign").items():
        pending = [key for key in ("recovery_generated_tokens", "horizons", "repeats", "arms") if not campaign.get(key)]
        live_config = campaign.get("live_observer")
        coverage = None
        if live_config is not None:
            run_id = campaign.get("source_natural_run_id") or campaign["mining"]["run_id"]
            coverage = live_observer_status(store, run_id, namespace=identity)
            live_coverage.append({"campaign_id": identity, **coverage})
        pending_observer = []
        if live_config is not None or campaign.get("mining", {}).get("purpose") in {"calibration", "main"}:
            for key in ("live_observer", "observer_settings", "observer_context_policy"):
                if not campaign.get(key):
                    pending_observer.append(key)
            if campaign.get("observer_source_policy") != "native-session-primary/v1":
                pending_observer.append("observer_source_policy")
            if type(campaign.get("observer_include_returned_reasoning")) is not bool:
                pending_observer.append("observer_include_returned_reasoning")
        campaigns.append({"campaign_id": identity, "mode": campaign.get("mode", "full"),
            "recovery_execution": "enabled" if campaign.get("mode", "full") == "full" else "deferred",
            "pending_recovery_settings": pending,
            "observer_execution": "live" if live_config is not None else (
                "deferred" if campaign.get("mode") == "mine" else "configured"),
            "live_observer_policy": live_config,
            "live_observer_complete": coverage["complete"] if coverage is not None else None,
            "pending_live_observer_settings": pending_observer,
            "fixture": campaign.get("fixture"), "calibration_status": campaign.get("calibration_status")})
    admission_rows = [dict(row) for row in admissions]
    for identity, task in _records(store, "campaign_task").items():
        admission_rows.append({"campaign_id": identity, "task_id": task.get("task_id"),
                               "admission": task.get("admission", "not_recorded")})
    from .launch_plan import validate_mining_launch_plan
    plans = [*list(_records(store, "launch_plan").values()), *launch_plans]
    return {"version": "factory-status.v2", "runs": runs, "phases": phases, "incidents": incidents,
            "campaigns": campaigns, "admissions": admission_rows,
            "launch_plans": [validate_mining_launch_plan(plan) for plan in plans],
            "batch_cancellations": list(_records(store, "batch_cancellation").values()),
            "batch_job_outcomes": list(_records(store, "batch_job_outcome").values()),
            "live_observer_coverage": live_coverage,
            "live_observer_circuits": list(_records(store, "live_observer_circuit_v1").values()),
            "live_observer_resumes": list(_records(store, "campaign_live_observer_resume_v1").values()),
            "notes": ["HTML/JSON are replaceable views; immutable receipts remain authoritative.",
                      "Failed and uncertain phases are not replayed. New execution needs an explicit new identity.",
                      "Batch cancellation drains active campaigns; their requests and costs may continue.",
                      "Live observer circuit pauses new mines; active workers remain unchanged. Coverage is read from durable queue state."]}


def write_status(store: ArtifactStore, output: Path, *, launch_plans: Sequence[Mapping[str, Any]] = (),
                 admissions: Sequence[Mapping[str, Any]] = ()) -> Path:
    report = status_report(store, launch_plans=launch_plans, admissions=admissions)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if output.suffix == ".json":
        output.write_text(encoded + "\n")
        return output
    output.with_suffix(".json").write_text(encoded + "\n")
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Trajectory factory status</title><style>
body{font:15px system-ui,sans-serif;margin:32px;color:#17252f;background:#f5f7fa}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}p{max-width:1000px}
</style><h1>Trajectory factory status</h1>
<p>Execution receipts and unresolved settings. Scripted and engineering runs do not establish research effects.
Known costs and missing usage remain separate; active campaigns drain after batch cancellation.</p>'''
    for key, title in (("runs", "Runs and accounting"), ("phases", "Active, failed and uncertain phases"),
                       ("incidents", "Incidents"), ("campaigns", "Campaign settings"),
                       ("admissions", "Admission and HOLDs"), ("launch_plans", "Launch settings"),
                       ("live_observer_coverage", "Live observer checkpoint coverage"),
                       ("live_observer_circuits", "Observer circuit receipts"),
                       ("live_observer_resumes", "Observer-only recovery"),
                       ("batch_cancellations", "Cancellation receipts"), ("batch_job_outcomes", "Every job outcome")):
        html += f"<h2>{title}</h2><pre>{escape(json.dumps(report[key], indent=2, sort_keys=True))}</pre>"
    output.write_text(html + "</html>\n")
    return output
