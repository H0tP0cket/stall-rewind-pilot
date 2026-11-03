"""Read-only total-policy attribution, separate from actual shared spend.

Operation identity is (run_id, operation_id). Standalone policies attribute
their required frozen preparation once each; those counterfactual totals must
never be summed into actual campaign spend. No local price is invented.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Sequence

from ..accounting import CostLedger, CostRecord, summarize
from ..contracts import TreatmentManifest, to_jsonable
from .continuations import BudgetClock, ContinuationRecord, grade_value_problems, quality_barriers

OperationKey = tuple[str, str]


def grade_availability(store: Any, checkpoint: Any, events: Sequence[Any]) -> dict[str, Any]:
    """Bind a grade's availability to its actual receipt, after state capture."""
    grade = store.get_json(checkpoint.evaluation_ref) if checkpoint.evaluation_ref else {}
    receipts = [e for e in events if e.kind == "grade" and
                e.payload.get("checkpoint_id") == checkpoint.checkpoint_id and
                e.payload.get("evaluation_id") == grade.get("evaluation_id")]
    if len(receipts) != 1 or checkpoint.validation.get("evaluation_reused"):
        return {"available_at_sequence": None, "evaluation_id": grade.get("evaluation_id"),
                "status": "reused_grade" if checkpoint.validation.get("evaluation_reused") else "grade_receipt_missing_or_ambiguous"}
    event = receipts[0]
    if event.scope is not None and event.scope != checkpoint.scope:
        return {"available_at_sequence": None, "evaluation_id": grade.get("evaluation_id"),
                "status": "grade_receipt_scope_mismatch"}
    if event.sequence < checkpoint.event_sequence:
        return {"available_at_sequence": None, "evaluation_id": grade.get("evaluation_id"),
                "status": "grade_receipt_precedes_checkpoint"}
    if (event.payload.get("evaluation_ref") is not None
            and event.payload["evaluation_ref"] != to_jsonable(checkpoint.evaluation_ref)):
        return {"available_at_sequence": None, "evaluation_id": grade.get("evaluation_id"),
                "status": "grade_receipt_reference_mismatch"}
    return {"available_at_sequence": event.sequence, "evaluation_id": grade.get("evaluation_id"),
            "grade_event_id": event.event_id, "status": "receipt_bound"}


def _key(r: CostRecord) -> OperationKey:
    return r.run_id, r.operation_id


def _source_placed(record: CostRecord, events: Mapping[int, Any]) -> bool:
    event = events.get(record.source_event_sequence)
    return event is not None and (record.source_event_id is None or record.source_event_id == event.event_id)


def _summary(records: Sequence[CostRecord]) -> dict[str, Any]:
    unique = {_key(r): r for r in records}
    rows = [unique[k] for k in sorted(unique)]
    return {"totals": summarize(rows),
            "roles": {role: summarize([r for r in rows if r.role == role]) for role in sorted({r.role for r in rows})},
            "kinds": {kind: summarize([r for r in rows if r.kind == kind]) for kind in sorted({r.kind for r in rows})},
            "operation_keys": [list(k) for k in sorted(unique)]}


def _checkpoint_resources(records: Sequence[CostRecord]) -> dict[str, Any]:
    """Disjoint operational categories, with the existing exact missingness."""
    groups: dict[str, list[CostRecord]] = {k: [] for k in ("worker_model", "worker_tool", "grading", "overhead")}
    for record in records:
        category = ("worker_model" if record.role == "worker" and record.kind == "model" else
                    "worker_tool" if record.role == "worker" and record.kind == "tool" else
                    "grading" if record.role == "grading" or record.kind == "grading" else "overhead")
        groups[category].append(record)
    return {**_summary(records), "categories": {k: summarize(v) for k, v in groups.items()}}


def _checkpoint_resource_delta(before: Sequence[CostRecord], after: Sequence[CostRecord]) -> dict[str, Any]:
    """Boundary differences, including explicit revisions of pending operations.

    A known-sum change is not a lower bound on interval spend: an earlier
    unknown operation can resolve at the later cut. Never subtract null values
    as zero or count that resolution as another call.
    """
    def difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for field in right:
            a, b = left[field], right[field]
            if isinstance(b, dict):
                def subtract(x: Any, y: Any) -> Any:
                    return float(Decimal(str(x)) - Decimal(str(y))) if field in ("cost_usd", "operation_seconds") else x - y
                result[field] = {"value": subtract(b["value"], a["value"]) if a["complete"] and b["complete"] else None,
                    "known_sum_change": subtract(b["known_sum"], a["known_sum"]),
                    "missing_operations_before": a["missing_operations"], "missing_operations_after": b["missing_operations"],
                    "complete": a["complete"] and b["complete"]}
            else:
                result[field] = b - a
        return result

    old, new = {_key(r): r for r in before}, {_key(r): r for r in after}
    a, b = _checkpoint_resources(before), _checkpoint_resources(after)
    return {"basis": "cumulative_boundary_difference; known_sum_change is not an interval-spend lower bound",
        "totals": difference(a["totals"], b["totals"]),
        "categories": {k: difference(a["categories"][k], b["categories"][k]) for k in b["categories"]},
        "new_operation_keys": [list(k) for k in sorted(new.keys() - old.keys())],
        "removed_operation_keys": [list(k) for k in sorted(old.keys() - new.keys())],
        "revised_operations": [{"operation_key": list(k), "before": to_jsonable(old[k]), "after": to_jsonable(new[k])}
                               for k in sorted(old.keys() & new.keys()) if old[k] != new[k]]}


def checkpoint_resource_report(store: Any, *, run_ids: Sequence[str]) -> dict[str, Any]:
    """Read-only checkpoint resource cuts over current durable ledger evidence.

    Capture and grade availability are different event cuts. Historical cuts
    retain a started operation as uncertain until its recorded completion;
    later ledger writes may resolve historical evidence, so the immutable
    accounting snapshot is retained separately rather than overwritten.
    Unplaced operations have no defensible checkpoint allocation and appear
    once in each run's all-ledger catalog, never in a guessed interval.
    """
    ledger = CostLedger(store)
    rows, runs = [], []
    for run_id in dict.fromkeys(run_ids):
        events = list(store.iter_events(run_id))
        event_index = {e.sequence: e for e in events}
        stored: dict[OperationKey, list[dict[str, Any]]] = {}
        stored_bodies = {}
        for item in store.list_records("cost", run_id=run_id):
            body = store.get_record("cost", item["record_id"])
            key = body["run_id"], body["operation_id"]
            stored.setdefault(key, []).append({"record_id": item["record_id"], "ref": item["ref"]})
            stored_bodies[key] = body
        provider_records: dict[str, list[dict[str, Any]]] = {}
        for kind in ("provider_call_started", "provider_call_finished"):
            for item in store.list_records(kind, run_id=run_id):
                provider_records.setdefault(item["record_id"], []).append({"kind": kind, **item})

        def unique(records: Sequence[CostRecord]) -> list[CostRecord]:
            if any(r.run_id != run_id for r in records):
                raise ValueError(f"Cost record run identity mismatch: {run_id}")
            keys = [_key(r) for r in records]
            if len(set(keys)) != len(keys):
                raise ValueError(f"Duplicate operation identity in checkpoint resources: {run_id}")
            return sorted(records, key=_key)

        def placed(record: CostRecord) -> bool:
            return type(record.source_event_sequence) is int and _source_placed(record, event_index)

        all_records = unique(ledger.records(run_id))
        unplaced = [r for r in all_records if not placed(r)]

        def operation(record: CostRecord) -> dict[str, Any]:
            event = event_index.get(record.source_event_sequence)
            has_cut = placed(record)
            body = to_jsonable(record)
            return {"operation_key": list(_key(record)), "record": body,
                "cost_records": stored.get(_key(record), []),
                "provider_call_records": provider_records.get(record.operation_id, []),
                "evidence_basis": "stored_cost_record" if stored_bodies.get(_key(record)) == body else "derived_from_durable_receipts",
                "placement": ("exact_event_id_and_sequence" if record.source_event_id is not None else "legacy_sequence_only")
                             if has_cut else "unplaced",
                "event_at_source_sequence": {"event_id": event.event_id, "sequence": event.sequence,
                    "kind": event.kind, "scope": to_jsonable(event.scope)} if event else None}

        def boundary(cut: int) -> tuple[dict[str, Any], list[CostRecord]]:
            candidates = unique(ledger.records(run_id, through_sequence=cut))
            included = [r for r in candidates if placed(r)]
            excluded = [r for r in candidates if not placed(r)]
            return {"through_sequence": cut, "cumulative": _checkpoint_resources(included),
                "attribution_complete": not unplaced and not excluded,
                "unplaced_at_cut": [operation(r) for r in excluded],
                "derived_operations": [operation(r) for r in included if stored_bodies.get(_key(r)) != to_jsonable(r)]}, included

        source_bindings = []
        for event in events:
            transition = event.payload.get("transition")
            source_id = (event.payload.get("resume_from") if transition == "run_started" else
                         event.payload.get("source_checkpoint_id") if transition == "branch_started" else None)
            if event.kind == "lifecycle" and source_id is not None:
                source_bindings.append({"source_checkpoint_id": source_id, "source_run_id": event.payload.get("source_run_id"),
                    "source_event_id": event.event_id, "source_event_sequence": event.sequence, "basis": transition})
        source_ids = {b["source_checkpoint_id"] for b in source_bindings}
        runs.append({"run_id": run_id, "all_ledger_scope": "all current operations, including work after terminal and observer work",
            "all_ledger": _checkpoint_resources(all_records), "unplaced": _checkpoint_resources(unplaced),
            "operation_catalog": [operation(r) for r in all_records], "source_checkpoint_bindings": source_bindings})
        previous, previous_records = None, []
        for cp in sorted(store.list_checkpoints(run_id), key=lambda c: c.ordinal):
            if previous is not None and cp.event_sequence < previous.event_sequence:
                raise ValueError(f"Checkpoint event cuts regress within {run_id}")
            capture, capture_records = boundary(cp.event_sequence)
            capture["delta_from_previous_capture"] = _checkpoint_resource_delta(previous_records, capture_records)
            capture_event = event_index.get(cp.event_sequence)
            capture["source_event_id"] = capture_event.event_id if capture_event else None
            capture["source_event_kind"] = capture_event.kind if capture_event else None
            capture["source_event_checkpoint_id"] = capture_event.payload.get("checkpoint_id") if capture_event else None
            snapshot = store.get_json(cp.accounting_ref) if cp.accounting_ref else None
            snapshot_problems = []
            if snapshot is not None:
                if snapshot.get("run_id") != run_id:
                    snapshot_problems.append("snapshot_run_mismatch")
                if type(snapshot.get("through_sequence")) is not int or snapshot["through_sequence"] != cp.event_sequence:
                    snapshot_problems.append("snapshot_capture_cut_mismatch")
            availability = grade_availability(store, cp, events) if cp.evaluation_ref else {
                "available_at_sequence": None, "evaluation_id": None, "status": "ungraded"}
            grade_boundary = None
            if availability["available_at_sequence"] is not None:
                grade_boundary, grade_records = boundary(availability["available_at_sequence"])
                grade_boundary["source_event_id"] = availability["grade_event_id"]
                grade_boundary["delta_since_capture"] = _checkpoint_resource_delta(capture_records, grade_records)
            rows.append({"run_id": run_id, "checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal,
                "scope": to_jsonable(cp.scope), "parent_checkpoint_id": cp.parent_checkpoint_id,
                "previous_same_run_checkpoint_id": previous.checkpoint_id if previous else None,
                "previous_capture_sequence": previous.event_sequence if previous else None,
                "source_checkpoint_id": next(iter(source_ids)) if len(source_ids) == 1 else None,
                "source_checkpoint_bindings": source_bindings, "capture": capture,
                "evaluation_ref": to_jsonable(cp.evaluation_ref), "grade_availability": availability,
                "grade_available": grade_boundary, "accounting_ref": to_jsonable(cp.accounting_ref),
                "recorded_accounting_snapshot": snapshot,
                "snapshot_binding_problems": snapshot_problems,
                "snapshot_matches_derived_capture": (snapshot.get("totals") == capture["cumulative"]["totals"]
                    and set(snapshot.get("operation_ids", [])) == {r.operation_id for r in capture_records}) if snapshot is not None else None})
            previous, previous_records = cp, capture_records
    return {"version": "checkpoint-resources.v1", "basis": "current ledger evidence at recorded event cuts; not wall-clock time",
        "category_basis": {"worker_model": "worker role, model kind", "worker_tool": "worker role, tool kind",
            "grading": "remaining operations with grading role or kind", "overhead": "all remaining operations, including observers"},
        "delta_basis": "difference from previous same-run capture, or empty run prefix for first checkpoint; grades use delta since own capture",
        "unplaced_policy": "retained in run all-ledger totals; excluded from every cut; attribution incomplete independently of metric missingness",
        "checkpoints": rows, "runs": runs}


def preparation_operations(store: Any, manifest: TreatmentManifest) -> dict[str, Any]:
    """Resolve the actual frozen artifact's transitive call lineage, not latest T.

    Summary calls are independent of T. Typed/prose/filter policies share the
    same primary incremental chain through the source. Full-prefix audit calls
    stay out unless explicitly attributed by the caller.
    """
    packet = manifest.metadata.get("packet", {})
    if packet.get("kind", "none") == "none":
        return {"operation_keys": [], "artifact_ids": [], "missing": []}
    keys, artifacts, missing = set(), [], []
    source = store.get_checkpoint(manifest.source_checkpoint_id)
    if packet.get("kind") == "summary":
        try:
            s = store.get_record("summary", packet["packet_id"])
            if (s.get("run_id"), s.get("target_checkpoint_id")) != (source.run_id, source.checkpoint_id):
                raise ValueError("summary preparation belongs to another run/source checkpoint")
            cut = s.get("source_visibility", {}).get("interval", {}).get("through_seq")
            if type(cut) is not int or cut != source.event_sequence:
                raise ValueError("summary preparation source cutoff differs from selected checkpoint")
            artifacts.append(["summary", packet["packet_id"]])
            keys.update((s["run_id"], cid) for cid in s.get("call_record_ids", []))
            if not s.get("call_record_ids"):
                missing.append("summary_call_lineage_missing")
        except KeyError:
            missing.append("summary_artifact_missing")
    elif packet.get("kind") == "research_packet":
        # Registered external packet: its preparation calls live outside this store (recorded in the record's
        # source_ids/construction); they are not allocated here rather than invented.
        artifacts.append(["research_packet", packet["packet_id"]])
        missing.append("research_packet_preparation_recorded_externally")
    else:
        vid = packet.get("source_version_id")
        if not vid:
            missing.append("T_source_version_missing")
        seen = set()
        expected_parent_cut = None
        while vid:
            if vid in seen:
                raise ValueError("cycle in immutable T lineage")
            seen.add(vid)
            try:
                version = store.get_record("t_version", vid)
            except KeyError:
                missing.append(f"T_version_missing:{vid}")
                break
            if version.get("role") != "t_retention":
                raise ValueError("primary treatment depends on non-retention T")
            cut = version.get("interval", {}).get("through_seq")
            if type(cut) is not int:
                raise ValueError("T preparation lacks a recorded interval cutoff")
            if version.get("run_id") != source.run_id or cut > source.event_sequence:
                raise ValueError("preparation lineage imports another run or the future")
            if len(seen) == 1 and (version.get("target_checkpoint_id") != source.checkpoint_id
                                   or cut != source.event_sequence):
                raise ValueError("selected T preparation is not from the selected source checkpoint")
            if expected_parent_cut is not None and cut != expected_parent_cut:
                raise ValueError("T preparation parent interval does not meet child interval")
            expected_parent_cut = version.get("interval", {}).get("after_seq")
            artifacts.append(["t_version", vid])
            keys.update((version["run_id"], cid) for cid in version.get("call_record_ids", []))
            vid = version.get("parent_version_id")
    # Optional recorded local rendering/measurement operations are allocated
    # only when their exact operation keys are bound in the packet metadata.
    keys.update(tuple(k) for k in packet.get("preparation_operation_keys", []))
    return {"operation_keys": [list(k) for k in sorted(keys)], "artifact_ids": artifacts, "missing": missing}


def policy_cost_report(store: Any, *, manifests: Sequence[TreatmentManifest],
                       records: Sequence[ContinuationRecord],
                       actual_operation_keys: Sequence[OperationKey] | None = None,
                       actual_run_ids: Sequence[str] = (),
                       include_source_prefix: bool = False,
                       audit_operation_keys_by_treatment: Mapping[str, Sequence[OperationKey]] | None = None,
                       dollar_horizons: Sequence[float] = ()) -> dict[str, Any]:
    """Return shared actual cost, standalone attribution and raw cost/quality points.

    Pass frozen snapshot operation keys for a campaign-specific accounting cut.
    ``actual_run_ids`` explicitly means every current operation on those runs;
    it may include repairs from several campaigns. Across reports, union keys
    rather than summing snapshots. Dollar horizon values are unavailable when
    any attributed price or operation placement is missing.
    """
    audit_operation_keys_by_treatment = audit_operation_keys_by_treatment or {}
    by_run: dict[str, list[CostRecord]] = {}

    def run_costs(run_id: str) -> list[CostRecord]:
        if run_id not in by_run:
            by_run[run_id] = CostLedger(store).records(run_id)
        return by_run[run_id]

    def resolve(keys: Sequence[OperationKey]) -> tuple[list[CostRecord], list[list[str]]]:
        found, missing = [], []
        for run, op in sorted(set(tuple(k) for k in keys)):
            candidates = {r.operation_id: r for r in run_costs(run)}
            if op in candidates:
                found.append(candidates[op])
            else:
                missing.append([run, op])
        return found, missing

    actual_keys = set(tuple(k) for k in (actual_operation_keys or ()))
    for run_id in actual_run_ids:
        actual_keys.update(_key(r) for r in run_costs(run_id))
    actual, missing_actual = resolve(sorted(actual_keys))
    actual_report = {**_summary(actual), "missing_operation_keys": missing_actual,
                     "scope": "explicit operation union; never sum standalone attributions"}
    by_treatment = {r.treatment_id: r for r in records}
    if len(by_treatment) != len(records):
        raise ValueError("duplicate treatment continuations")
    policies, curve, horizons = [], [], []
    for m in sorted(manifests, key=lambda m: m.treatment_id):
        r = by_treatment.get(m.treatment_id)
        prep = preparation_operations(store, m)
        prep_keys = [tuple(k) for k in prep["operation_keys"]]
        prep_keys.extend(audit_operation_keys_by_treatment.get(m.treatment_id, ()))
        source = store.get_checkpoint(m.source_checkpoint_id)
        source_events = list(store.iter_events(source.run_id))
        source_event_index = {e.sequence: e for e in source_events}
        source_grade_cut = grade_availability(store, source, source_events)
        selection_cut = max(source.event_sequence, source_grade_cut["available_at_sequence"] or source.event_sequence)
        if m.selection_kind == "primary_s1":
            annotations = [store.get_record("annotation", a["record_id"])
                           for a in store.list_records("annotation", run_id=source.run_id)]
            warnings = [a for a in annotations if a.get("kind") == "S" and
                        a.get("checkpoint_id") == source.checkpoint_id and a.get("origin") == "online"]
            if warnings:
                selection_cut = max(selection_cut, min(a["evidence_cutoff"] for a in warnings))
        common_prefix = [c for c in run_costs(source.run_id) if _source_placed(c, source_event_index)
                         and c.source_event_sequence <= selection_cut]
        preparation, missing_prep = resolve(prep_keys)
        branch = run_costs(r.branch_run_id) if r else []
        branch_events = list(store.iter_events(r.branch_run_id)) if r else []
        branch_event_index = {e.sequence: e for e in branch_events}
        unplaced = [_key(c) for c in branch if not _source_placed(c, branch_event_index)]
        recovery_cost = _summary([*preparation, *branch])
        end_to_end_cost = _summary([*common_prefix, *preparation, *branch])
        source_unplaced = [_key(c) for c in run_costs(source.run_id) if not _source_placed(c, source_event_index)
                           and c.role in {"worker", "grading", "restore", "infrastructure"}]
        source_cost_eligibility = not source_unplaced and source_grade_cut["available_at_sequence"] is not None
        end_to_end_cost["source_prefix_attribution_complete"] = source_cost_eligibility
        standalone = end_to_end_cost if include_source_prefix else recovery_cost
        policy = {"treatment_id": m.treatment_id, "arm_id": m.metadata.get("arm_id"),
            "source_checkpoint_id": m.source_checkpoint_id, "branch_run_id": r.branch_run_id if r else None,
            "worker_model": m.worker_model, "repeat_id": m.repeat_id, "nominal_worker_allowance": m.generated_token_limit,
            "worker_observed_generated_tokens": r.observed_horizon if r else None,
            "worker_accounting_complete": r.horizon_complete if r else None,
            "worker_overshoot": max(0, r.observed_horizon-m.generated_token_limit) if r and r.horizon_complete else None,
            "status": "recorded" if r else "rollout_missing", "preparation_lineage": prep,
            "preparation": _summary(preparation), "branch_execution": _summary(branch),
            "common_source_prefix": _summary(common_prefix), "standalone": standalone,
            "standalone_recovery": recovery_cost, "standalone_end_to_end": end_to_end_cost,
            "unplaced_source_operations": [list(k) for k in sorted(source_unplaced)],
            "source_selection_cost_cut": selection_cut, "source_grade_availability": source_grade_cut,
            "missing_preparation_operations": missing_prep,
            "unplaced_branch_operations": [list(k) for k in sorted(unplaced)],
            "attribution_complete": (not missing_prep and not prep["missing"] and r is not None
                                     and (not include_source_prefix or source_cost_eligibility)),
            "source_prefix_included": include_source_prefix,
            "audit_cost_policy": "explicit keys only; other audits remain actual research spend"}
        policies.append(policy)
        if r is None:
            continue
        best = r.baseline_score
        policy_points = []
        for cp in sorted(store.list_checkpoints(r.branch_run_id), key=lambda c: c.ordinal):
            ev = store.get_json(cp.evaluation_ref) if cp.evaluation_ref else {}
            availability = grade_availability(store, cp, branch_events)
            cost_cut = availability["available_at_sequence"]
            boundary_cut = cost_cut if cost_cut is not None else cp.event_sequence
            target = m.metadata.get("recovery_target", {})
            compatible = (ev.get("target_id") == m.target_id and ev.get("stage_id") == m.stage_id
                          and all(ev.get(k) == getattr(cp.scope, k) for k in ("task_id", "stage_id", "target_id", "evaluator_version"))
                          and all(getattr(cp.scope, k) == getattr(source.scope, k)
                                  for k in ("task_id", "stage_id", "target_id", "requirement_version", "evaluator_version"))
                          and (not target.get("evaluator_version") or ev.get("evaluator_version") == target["evaluator_version"]))
            exclusions = grade_value_problems(ev)
            if not compatible:
                exclusions.append("grade_scope_incompatible")
            if cost_cut is None:
                exclusions.append(availability["status"])
            valid = not exclusions
            score = ev.get("score") if valid else None
            if score is not None:
                best = score if best is None else max(best, score)
            placed = [c for c in branch if _source_placed(c, branch_event_index)
                      and c.source_event_sequence <= boundary_cut]
            cumulative = _summary([*(common_prefix if include_source_prefix else []), *preparation, *placed])
            model_spend = cumulative["kinds"].get("model", summarize([]))["cost_usd"]
            own_grade_key = (r.branch_run_id, f"grade:{ev.get('evaluation_id')}")
            grade_cost_recorded = own_grade_key in {_key(c) for c in placed}
            price_complete = (cumulative["totals"]["cost_usd"]["complete"] and not unplaced and grade_cost_recorded
                              and policy["attribution_complete"])
            row = {"treatment_id": m.treatment_id, "branch_run_id": r.branch_run_id,
                   "checkpoint_id": cp.checkpoint_id, "event_sequence": cp.event_sequence,
                   "measurement_kind": "checkpoint_grade", "source_event_id": availability.get("grade_event_id"),
                   "cost_cut_sequence": cost_cut, "grade_availability": availability,
                   "measurement_boundary_sequence": boundary_cut,
                   "measurement_boundary_cost_usd": cumulative["totals"]["cost_usd"]["value"]
                       if policy["attribution_complete"] and not unplaced else None,
                   "own_grade_cost_recorded": grade_cost_recorded,
                   "arm_id": m.metadata.get("arm_id"), "worker_model": m.worker_model,
                   "valid_fixed_target_grade": valid, "score": score, "best_observed_score": best,
                   "grade_exclusions": exclusions,
                   "raw_grade_values": {key: ev.get(key) for key in ("valid", "score", "full_solve")},
                   "full_solve": ev.get("full_solve") if valid else None,
                   "standalone_cost_usd": cumulative["totals"]["cost_usd"]["value"] if price_complete else None,
                   "known_cost_usd_lower_bound": cumulative["totals"]["cost_usd"]["known_sum"],
                   "reported_model_cost_usd": model_spend["value"] if cost_cut is not None
                                              and policy["attribution_complete"] and not unplaced else None,
                   "model_price_missing_operations": model_spend["missing_operations"],
                   "price_complete": price_complete,
                   "unpriced_operations": cumulative["totals"]["cost_usd"]["missing_operations"],
                   "operation_seconds": cumulative["totals"]["operation_seconds"],
                   "worker_generated_tokens": cumulative["totals"]["worker_generated_tokens"],
                   "cost_operation_keys": cumulative["operation_keys"]}
            curve.append(row)
            policy_points.append(row)
        for barrier in quality_barriers(branch_events, BudgetClock.from_ledger(store, r.branch_run_id)):
            placed = [c for c in branch if _source_placed(c, branch_event_index)
                      and c.source_event_sequence <= barrier["event_sequence"]]
            cumulative = _summary([*(common_prefix if include_source_prefix else []), *preparation, *placed])
            model_spend = cumulative["kinds"].get("model", summarize([]))["cost_usd"]
            row = {"treatment_id": m.treatment_id, "branch_run_id": r.branch_run_id,
                   "checkpoint_id": None, "event_sequence": barrier["event_sequence"],
                   "measurement_kind": "event_quality_barrier", "source_event_id": barrier["source_event_id"],
                   "cost_cut_sequence": None, "grade_availability": {"status": barrier["reason"], "available_at_sequence": None},
                   "measurement_boundary_sequence": barrier["event_sequence"],
                   "measurement_boundary_cost_usd": cumulative["totals"]["cost_usd"]["value"]
                       if policy["attribution_complete"] and not unplaced else None,
                   "own_grade_cost_recorded": False, "arm_id": m.metadata.get("arm_id"), "worker_model": m.worker_model,
                   "valid_fixed_target_grade": False, "score": None, "best_observed_score": None,
                   "grade_exclusions": [barrier["reason"]], "raw_grade_values": {}, "full_solve": None,
                   "standalone_cost_usd": None, "known_cost_usd_lower_bound": cumulative["totals"]["cost_usd"]["known_sum"],
                   "reported_model_cost_usd": model_spend["value"] if policy["attribution_complete"] and not unplaced else None,
                   "model_price_missing_operations": model_spend["missing_operations"], "price_complete": False,
                   "unpriced_operations": cumulative["totals"]["cost_usd"]["missing_operations"],
                   "operation_seconds": cumulative["totals"]["operation_seconds"],
                   "worker_generated_tokens": cumulative["totals"]["worker_generated_tokens"],
                   "cost_operation_keys": cumulative["operation_keys"]}
            curve.append(row)
            policy_points.append(row)
        policy_points.sort(key=lambda p: p["measurement_boundary_sequence"])
        final_cost = standalone["totals"]["cost_usd"]["value"]
        fixed = [*(common_prefix if include_source_prefix else []), *preparation]
        fixed_price = _summary(fixed)["totals"]["cost_usd"]
        missing_sequences = [c.source_event_sequence for c in branch
                             if c.source_event_sequence is not None and c.usage.cost_usd is None]
        first_missing = min(missing_sequences) if missing_sequences else None
        reliable_price_cut = (_summary([*fixed, *(c for c in branch if c.source_event_sequence is not None
            and c.source_event_sequence < first_missing)])["totals"]["cost_usd"]["value"]
            if first_missing is not None and fixed_price["complete"] and not unplaced
            and policy["attribution_complete"] else None)
        for h in sorted(set(dollar_horizons)):
            if h < 0:
                raise ValueError("negative dollar horizon")
            candidates = [p for p in policy_points if p["measurement_boundary_cost_usd"] is not None
                          and p["measurement_boundary_cost_usd"] <= h]
            complete = (standalone["totals"]["cost_usd"]["complete"] and not unplaced
                        and policy["attribution_complete"])
            coverage = (complete and final_cost >= h) or (reliable_price_cut is not None and h < reliable_price_cut)
            # Before preparation finishes, no policy grade is available. Never
            # project a later improvement backwards onto a smaller cost budget.
            current = candidates[-1] if candidates else None
            usable = current is not None and current["valid_fixed_target_grade"] and current["price_complete"]
            scores = [p["score"] for p in candidates if p["price_complete"] and p["score"] is not None]
            horizons.append({"treatment_id": m.treatment_id, "dollar_horizon": h,
                "curve": "actual_compute", "score_at": current["score"] if coverage and usable else None,
                "best_observed_score_at": max(scores) if coverage and scores else None,
                "current_checkpoint_id": current["checkpoint_id"] if current else None,
                "status": "observed_current_grade" if coverage and usable else
                          "grade_measurement_missing" if coverage and current is not None else
                          "price_or_attribution_missing" if not complete else "unobserved_cost_horizon"})
    curve.sort(key=lambda p: (p["treatment_id"], p["measurement_boundary_sequence"]))
    return {"version": "policy-costs.v2", "actual": actual_report, "standalone_policies": policies,
            "cost_quality_points": curve, "dollar_horizons": horizons,
            "units": {"operation_seconds": "summed durations; not elapsed wall time",
                      "reported_model_cost_usd": "provider model charges only; excludes local compute/tool/grading/restore prices",
                      "dollar_horizons": "actual priced compute only; generated-token allowance absorption does not invent dollar spend",
                      "standalone": "source prefix + preparation + branch" if include_source_prefix else "preparation + branch; source prefix reported separately"},
            "warning": "Standalone policy attributions overlap; only actual.operation_keys define shared spend. Missing prices are not zero."}
