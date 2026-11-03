"""Post-run reconciliation: artifacts resolve, costs agree, checkpoints are honest.

Checks (all evidence-based, none repairs anything):

* every checkpoint's W/R/H/harness/observation/evaluation/accounting refs resolve
  and W archive bytes re-hash to the recorded fidelity manifest;
* every finished worker model attempt has exactly one ledger record; ledger worker
  generated-token totals (known and unknown counts) equal the sums over finished attempts;
* each checkpoint's accounting snapshot covers exactly the operations at or
  before its event cut; unknown usage stays unknown;
* branchable checkpoints have no pending actions and a supported runtime class;
* the detector replay reproduces the live outputs byte for byte.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..accounting import CostLedger
from ..contracts import ArtifactRef, ArtifactStoreProtocol, RuntimeClass
from ..detector.core import DetectorConfig
from ..execution.environment import comparable, fidelity_manifest_from_tar
from .replay import replay_run


def audit_run(store: ArtifactStoreProtocol, run_id: str, detector_config: DetectorConfig) -> dict[str, Any]:
    problems: list[str] = []
    events = list(store.iter_events(run_id))
    checkpoints = store.list_checkpoints(run_id)  # type: ignore[attr-defined]
    ledger = CostLedger(store)  # type: ignore[arg-type]
    records = ledger.records(run_id)
    by_op = {r.operation_id: r for r in records}

    # ---- artifact resolution and W fidelity
    for c in checkpoints:
        refs = [c.state.workspace, c.state.history, c.state.harness, c.state.runtime, c.evaluation_ref, c.accounting_ref,
                *c.state.observations]
        for ref in refs:
            if ref is not None:
                try:
                    store.get_bytes(ref)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{c.checkpoint_id}: unresolved ref {ref.sha256[:12]}: {e}")
        try:
            w = store.get_bytes(c.state.workspace)
            fidelity = store.get_json(_ref(c.validation["workspace_manifest_ref"]))
            if comparable(fidelity_manifest_from_tar(w)) != comparable(fidelity):
                problems.append(f"{c.checkpoint_id}: workspace archive does not match its fidelity manifest")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{c.checkpoint_id}: workspace fidelity check failed: {e}")
        if c.branchable and (c.state.execution_cursor.get("pending_actions") or c.state.runtime_class == RuntimeClass.UNSUPPORTED):
            problems.append(f"{c.checkpoint_id}: branchable despite pending actions or unsupported runtime")
        if c.event_sequence > max((e.sequence for e in events), default=0):
            problems.append(f"{c.checkpoint_id}: event cut beyond recorded events")

    # ---- attempts vs ledger
    finished = [e for e in events if e.kind == "model_attempt" and e.payload.get("phase") == "finished"]
    for a in finished:
        op = f"{a.call_id}:attempt{a.attempt}"
        if op not in by_op:
            problems.append(f"attempt {op} has no ledger record")
        elif by_op[op].status != a.payload["status"]:
            problems.append(f"attempt {op}: ledger status {by_op[op].status} != event {a.payload['status']}")
    # Worker reconciliation only: extraction/summary/audit calls recorded later have other roles.
    model_ops = [r for r in records if r.kind == "model" and r.role == "worker"]
    if len(model_ops) != len(finished):
        problems.append(f"ledger worker model operations {len(model_ops)} != finished worker attempts {len(finished)}")

    # ---- generated tokens: every finished attempt (failed/rejected attempts may carry usage) vs ledger
    attempt_known = [a.usage.generated_tokens for a in finished if a.usage and a.usage.generated_tokens is not None]
    attempt_unknown = sum(1 for a in finished if not a.usage or a.usage.generated_tokens is None)
    ledger_known = sum(r.usage.generated_tokens for r in model_ops if r.usage.generated_tokens is not None)
    ledger_unknown = sum(1 for r in model_ops if r.usage.generated_tokens is None)
    if sum(attempt_known) != ledger_known or attempt_unknown != ledger_unknown:
        problems.append(f"generated tokens: attempts {sum(attempt_known)}/{attempt_unknown} unknown != ledger {ledger_known}/{ledger_unknown} unknown")
    responses = [e for e in events if e.kind == "model_response"]
    known = [e.usage.generated_tokens for e in responses if e.usage and e.usage.generated_tokens is not None]
    unknown = sum(1 for e in responses if not e.usage or e.usage.generated_tokens is None)

    # ---- checkpoint accounting snapshots
    for c in checkpoints:
        if c.accounting_ref is None:
            problems.append(f"{c.checkpoint_id}: no accounting snapshot")
            continue
        snap = store.get_json(c.accounting_ref)
        expected = sorted(r.operation_id for r in records if r.source_event_sequence is not None and r.source_event_sequence <= c.event_sequence)
        if sorted(snap["operation_ids"]) != expected:
            problems.append(f"{c.checkpoint_id}: accounting snapshot operations differ from ledger through cut")

    # ---- detector replay
    rep = replay_run(store, run_id, detector_config)
    if not rep["agrees"]:
        problems.append("detector replay disagrees with live outputs")

    terminal = [e for e in events if e.kind == "terminal"]
    return {"run_id": run_id, "events": len(events), "checkpoints": len(checkpoints),
            "branchable": sum(c.branchable for c in checkpoints), "finished_attempts": len(finished),
            "responses": len(responses), "responses_with_unknown_usage": unknown, "known_generated_tokens": sum(known),
            "attempts_with_unknown_usage": attempt_unknown, "attempt_known_generated_tokens": sum(attempt_known),
            "ledger_records": len(records), "detector_replay": rep, "terminal": terminal[-1].payload if terminal else None,
            "problems": problems, "ok": not problems}


def _ref(value: Any) -> ArtifactRef:
    return ArtifactRef(**value) if isinstance(value, dict) else value


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
