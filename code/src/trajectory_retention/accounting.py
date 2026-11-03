"""Operation-based accounting with explicit missingness and cost roles.

Provider input/output/reasoning fields can overlap. Only the explicitly
normalized generated_tokens field is summed as the worker budget unit. Input,
cached, output and reasoning fields remain separate descriptive measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Any, Iterable

from .contracts import ArtifactRef, Usage, canonical_json, from_jsonable, to_jsonable
from .store import ArtifactStore
from .persistence import persist

COST_ROLES = frozenset({"worker", "extraction", "summary", "audit", "advisor",
                        "manager", "render", "grading", "restore", "infrastructure"})


@dataclass(frozen=True, kw_only=True)
class CostRecord:
    run_id: str
    operation_id: str
    role: str
    kind: str
    usage: Usage
    status: str
    source_event_id: str | None = None
    source_event_sequence: int | None = None

    def __post_init__(self) -> None:
        if self.role not in COST_ROLES:
            raise ValueError(f"Unknown accounting role: {self.role}")
        if self.kind not in ("model", "tool", "grading", "restore", "local", "infrastructure"):
            raise ValueError(f"Unknown accounting kind: {self.kind}")
        if self.source_event_sequence is not None and self.source_event_sequence < 0:
            raise ValueError("Negative source sequence")


def _metric(records: Iterable[CostRecord], field: str) -> dict[str, Any]:
    relevant = list(records)
    values = [getattr(r.usage, field) for r in relevant]
    known = [v for v in values if v is not None]
    missing = len(values) - len(known)
    if field in ("cost_usd", "elapsed_seconds"):
        total: int | float = float(sum((Decimal(str(v)) for v in known), Decimal(0)))
    else:
        total = sum(known)
    return {"value": total if not missing else None, "known_sum": total,
            "missing_operations": missing, "operations": len(relevant), "complete": missing == 0}


def summarize(records: Iterable[CostRecord]) -> dict[str, Any]:
    records = list(records)
    models = [r for r in records if r.kind == "model"]
    result: dict[str, Any] = {
        "operations": len(records),
        "model_calls": len(models),
        "tool_calls": sum(r.kind == "tool" for r in records),
        "non_ok_operations": sum(r.status != "ok" for r in records),
        "cost_usd": _metric(records, "cost_usd"),
        # Summed operation duration is not concurrent campaign wall-clock time.
        "operation_seconds": _metric(records, "elapsed_seconds"),
    }
    for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens",
                 "generated_tokens", "total_tokens"):
        result[name] = _metric(models, name)
    result["worker_generated_tokens"] = _metric(
        (r for r in models if r.role == "worker"), "generated_tokens")
    return result


class CostLedger:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def record(self, *, run_id: str, operation_id: str, role: str, usage: Usage,
               kind: str = "model", status: str = "ok", source_event_id: str | None = None,
               source_event_sequence: int | None = None) -> ArtifactRef:
        entry = CostRecord(run_id=run_id, operation_id=operation_id, role=role, kind=kind,
                           usage=usage, status=status, source_event_id=source_event_id,
                           source_event_sequence=source_event_sequence)
        # Unique per operation, not per role, preventing duplicate allocation
        # of the same provider attempt to both worker and observer totals.
        return persist(lambda: self.store.put_record("cost", f"{run_id}:{operation_id}", entry, run_id=run_id),
                       evidence=entry)

    def records(self, run_id: str, *, through_sequence: int | None = None) -> list[CostRecord]:
        result = []
        for item in self.store.list_records("cost", run_id=run_id):
            rec = from_jsonable(CostRecord, self.store.get_record("cost", item["record_id"]))
            if through_sequence is not None:
                if rec.source_event_sequence is None or rec.source_event_sequence > through_sequence:
                    continue
            result.append(rec)
        # A killed process cannot execute a finally block. Derive an explicit
        # uncertain operation from each durable attempt start lacking a ledger
        # receipt at this cut. This is a view over evidence, not a fabricated
        # provider bill or mutation of earlier immutable cost records.
        present = {r.operation_id for r in result}
        events = list(self.store.iter_events(run_id, through_sequence=through_sequence))
        finished = {(e.call_id, e.attempt): e for e in events
                    if e.kind == "model_attempt" and e.payload.get("phase") == "finished"}
        for event in events:
            if event.kind != "model_attempt" or event.payload.get("phase") != "started":
                continue
            operation_id = f"{event.call_id}:attempt{event.attempt}"
            if operation_id not in present:
                receipt = finished.get((event.call_id, event.attempt))
                result.append(CostRecord(run_id=run_id, operation_id=operation_id, role="worker", kind="model",
                    status=receipt.payload.get("status", "uncertain") if receipt else "uncertain",
                    usage=(receipt.usage if receipt else None) or Usage(measurement_basis="started_attempt_without_final_usage"),
                    source_event_id=receipt.event_id if receipt else event.event_id,
                    source_event_sequence=receipt.sequence if receipt else event.sequence))
                present.add(operation_id)
        waits = {}
        for event in events:
            if event.kind == "operation" and event.payload.get("name") == "worker_retry_wait":
                waits[event.payload["operation_id"]] = event
        for operation_id, event in waits.items():
            if operation_id not in present:
                finished = event.payload.get("phase") == "finished"
                result.append(CostRecord(run_id=run_id, operation_id=operation_id, role="infrastructure",
                    kind="infrastructure", status=event.payload.get("status", "uncertain") if finished else "uncertain",
                    usage=event.usage if finished and event.usage is not None else
                        Usage(measurement_basis="started_retry_wait_without_duration"),
                    source_event_id=event.event_id, source_event_sequence=event.sequence))
                present.add(operation_id)
        # Tool starts are durable before execution. A hard exit can leave only
        # that start, or a private result without its ledger write. Match by the
        # exact parent/run/scope, never by reusable provider tool-call IDs alone.
        tool_results = {}
        for event in events:
            if event.kind == "tool_result":
                tool_results.setdefault(event.parent_event_id, []).append(event)
        indices = {}
        for event in events:
            if event.kind != "tool_call" or event.actor != "worker":
                continue
            model_call = event.payload.get("model_call_id")
            key = (event.scope, model_call)
            index = indices.get(key, 0)
            indices[key] = index + 1
            # Older durable starts lack an explicit operation ID. The agent
            # executed its action batch in order and used this same index key.
            operation_id = event.payload.get("operation_id") or (
                f"{model_call}:tool{index}" if model_call is not None else f"tool_attempt:{event.event_id}")
            if operation_id in present:
                continue
            receipt = next((r for r in tool_results.get(event.event_id, ())
                if r.run_id == event.run_id and r.scope == event.scope and r.call_id == event.call_id
                and r.sequence > event.sequence and r.actor in {"tool", "harness"}
                and r.payload.get("operation_id", operation_id) == operation_id), None)
            status, usage = "uncertain", Usage(measurement_basis="started_tool_without_final_usage")
            if receipt is not None:
                duration = receipt.payload.get("duration_seconds")
                if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                    duration = None
                usage = receipt.usage or Usage(elapsed_seconds=duration,
                    measurement_basis="durable_tool_result" if duration is not None else "tool_result_without_duration")
                status = receipt.payload.get("status") or (
                    "control_transfer" if receipt.payload.get("control_transfer") == "RewindRequested" else
                    "error" if receipt.payload.get("control_transfer") else
                    "ok" if type(receipt.payload.get("returncode")) is int and receipt.payload["returncode"] == 0 else
                    "uncertain")  # A legacy nonzero CLI code cannot identify command versus transport failure.
            source = receipt or event
            result.append(CostRecord(run_id=run_id, operation_id=operation_id, role="worker", kind="tool",
                status=status, usage=usage, source_event_id=source.event_id, source_event_sequence=source.sequence))
            present.add(operation_id)
        if through_sequence is None:
            for row in self.store.list_records("provider_usage_lookup_started", run_id=run_id):
                operation_id = row["record_id"]
                if operation_id not in present:
                    try:
                        lookup = self.store.get_record("provider_usage_lookup_finished", operation_id)
                    except KeyError:
                        lookup = None
                    result.append(CostRecord(run_id=run_id, operation_id=operation_id, role="infrastructure",
                        kind="infrastructure", status=lookup["status"] if lookup else "uncertain",
                        usage=Usage(elapsed_seconds=lookup["elapsed_seconds"], measurement_basis="measured_usage_recovery")
                            if lookup else Usage(measurement_basis="started_usage_lookup_without_duration")))
                    present.add(operation_id)
            # Observers run outside the worker event clock. Their durable
            # starts also survive cancellation and must not disappear as free
            # successful computation when no final provider receipt exists.
            for row in self.store.list_records("provider_call_started", run_id=run_id):
                started = self.store.get_record("provider_call_started", row["record_id"])
                operation_id = started["call_id"]
                if operation_id not in present:
                    try:
                        receipt = self.store.get_record("provider_call_finished", operation_id)["receipt"]
                    except KeyError:
                        receipt = None
                    result.append(CostRecord(run_id=run_id, operation_id=operation_id,
                        role=started["cost_role"], kind="model", status=receipt["status"] if receipt else "uncertain",
                        usage=from_jsonable(Usage, receipt["usage"]) if receipt else
                              Usage(measurement_basis="started_observer_without_final_usage")))
                    present.add(operation_id)
        return result

    def snapshot(self, run_id: str, *, through_sequence: int | None = None) -> ArtifactRef:
        records = self.records(run_id, through_sequence=through_sequence)
        snapshot = {
            "schema_version": "cost-summary.v1", "run_id": run_id,
            "through_sequence": through_sequence,
            "scope": "all_recorded_operations" if through_sequence is None else "operations_with_source_sequence_at_or_before_cut",
            "totals": summarize(records),
            "roles": {role: summarize(r for r in records if r.role == role) for role in sorted({r.role for r in records})},
            "operation_ids": [r.operation_id for r in records],
        }
        return persist(lambda: self.store.put_json(snapshot), evidence=snapshot)


def openrouter_usage(raw: dict[str, Any], *, elapsed_seconds: float,
                     raw_usage_ref: ArtifactRef | None = None) -> Usage:
    """Normalize documented OpenRouter chat usage without fabricating counters.

    completion_tokens is the total generated count on this route; reasoning
    detail is its subset. Other providers must declare their own normalization.
    A missing completion count leaves the budget unit unknown.
    """
    if not isinstance(raw, dict):
        return Usage(elapsed_seconds=elapsed_seconds, measurement_basis="malformed_provider_usage",
                     raw_usage=raw_usage_ref)
    def integer(value: Any) -> int | None:
        return value if type(value) is int and value >= 0 else None
    def number(value: Any) -> float | None:
        if type(value) not in (int, float):
            return None
        try:
            result = float(value)
            return result if Decimal(str(result)).is_finite() and result >= 0 else None
        except (ValueError, OverflowError):
            return None
    completion = integer(raw.get("completion_tokens"))
    completion_details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    return Usage(input_tokens=integer(raw.get("prompt_tokens")), output_tokens=completion,
                 reasoning_tokens=integer(completion_details.get("reasoning_tokens")),
                 cached_input_tokens=integer(prompt_details.get("cached_tokens")),
                 total_tokens=integer(raw.get("total_tokens")), generated_tokens=completion,
                 reasoning_accounting="included_in_output", cost_usd=number(raw.get("cost")),
                 elapsed_seconds=elapsed_seconds, measurement_basis="provider_reported_openrouter",
                 raw_usage=raw_usage_ref)
