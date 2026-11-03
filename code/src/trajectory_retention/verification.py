"""Read-only artifact and lineage verification, separate from restore testing.

Hash verification proves byte integrity, not runtime equivalence. Restoration
and isolation evidence must come from the execution adapter's actual probes.
"""
from __future__ import annotations

import json
from typing import Any

from .contracts import ArtifactRef, Visibility, from_jsonable
from .store import ArtifactStore


def verify_store(store: ArtifactStore) -> dict[str, Any]:
    errors, visited = [], set()
    records = events = checkpoints = 0

    def check_ref(ref: ArtifactRef, where: str) -> None:
        key = (ref.sha256, ref.size_bytes, ref.media_type)
        if key in visited:
            return
        visited.add(key)
        try:
            data = store.get_bytes(ref)
        except Exception as exc:
            errors.append({"where": where, "sha256": ref.sha256, "error": str(exc)})
            return
        if ref.media_type == "application/json":
            try:
                walk(json.loads(data), where + "/json")
            except (ValueError, UnicodeDecodeError):
                errors.append({"where": where, "sha256": ref.sha256, "error": "Invalid declared JSON bytes"})

    def walk(value: Any, where: str) -> None:
        if isinstance(value, dict):
            if {"sha256", "size_bytes"} <= set(value):
                # Semantic packet records sometimes carry only hash/size. These
                # still must resolve, but do not promise JSON content.
                try:
                    ref = ArtifactRef(sha256=value["sha256"], size_bytes=value["size_bytes"],
                                      media_type=value.get("media_type", "application/octet-stream"))
                    check_ref(ref, where)
                except (TypeError, ValueError) as exc:
                    errors.append({"where": where, "error": str(exc)})
            else:
                for k, v in value.items():
                    walk(v, f"{where}/{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{where}/{i}")

    with store._connect() as db:
        for table in ("records", "events", "checkpoints"):
            rows = db.execute(f"SELECT ref_json FROM {table}").fetchall()
            for i, row in enumerate(rows):
                check_ref(from_jsonable(ArtifactRef, json.loads(row[0])), f"{table}/{i}")
            if table == "records":
                records = len(rows)
            elif table == "events":
                events = len(rows)
            else:
                checkpoints = len(rows)
        run_ids = [r[0] for r in db.execute("SELECT DISTINCT run_id FROM events")]
    for run_id in run_ids:
        try:
            stream = list(store.iter_events(run_id))
            if [e.sequence for e in stream] != list(range(1, len(stream) + 1)):
                errors.append({"where": run_id, "error": "Event sequence has gaps"})
            event_ids = {e.event_id for e in stream}
            for e in stream:
                if e.parent_event_id and e.parent_event_id not in event_ids:
                    errors.append({"where": e.event_id, "error": "Parent event absent from run"})
            for cp in store.list_checkpoints(run_id):
                if cp.event_sequence > len(stream):
                    errors.append({"where": cp.checkpoint_id, "error": "Checkpoint refers to future events"})
                if cp.parent_checkpoint_id:
                    parent = store.get_checkpoint(cp.parent_checkpoint_id)
                    if parent.run_id == run_id and parent.ordinal >= cp.ordinal:
                        errors.append({"where": cp.checkpoint_id, "error": "Parent checkpoint is not earlier"})
        except Exception as exc:
            errors.append({"where": run_id, "error": str(exc)})
    return {"valid": not errors, "unique_references_checked": len(visited), "records": records,
            "events": events, "checkpoints": checkpoints, "errors": errors,
            "claim": "Stored byte integrity and basic lineage only; runtime restoration requires execution evidence."}


def _verify_agentrewind_history(store: ArtifactStore, treatment: dict, source, first, checkpoints, events) -> dict[str, bool]:
    """Audit exact baseline additions and each worker-authorized memory lineage.

    This does not normalize away arbitrary system messages. The only initial
    additions are the pinned policy and fixed source target. Rewind notebooks
    are reconstructed from accepted commit responses, then checked against
    both the recorded delivery and the next coherent checkpoint/API request.
    """
    from .baselines.agentrewind import POLICY, SYSTEM_PROMPT

    def messages(cp):
        return store.get_json(cp.state.history)["messages"]

    def artifact(ref):
        return store.get_json(from_jsonable(ArtifactRef, ref))

    def visible(history):
        return [{k: v for k, v in m.items() if k != "extra"} for m in history]

    source_bundle = store.get_json(source.state.history)
    additions = [{"role": "system", "content": SYSTEM_PROMPT,
                  "extra": {"agentrewind_policy": POLICY["policy_id"]}},
                 {"role": "user", "content": source_bundle["task"],
                  "extra": {"agentrewind_fixed_target": True}}]
    initialized = [e for e in events if e.kind == "baseline" and e.payload.get("transition") == "initialized"]
    installed = [e for e in events if e.kind == "lifecycle" and e.payload.get("transition") == "baseline_tools_installed"]
    checks = {
        "baseline_declared_policy_matches": treatment["metadata"].get("baseline_policy") == POLICY,
        "baseline_aligned_source_without_packets": treatment["history_policy"] == "native"
            and all(treatment["component_donors"].get(k) == source.checkpoint_id for k in ("workspace", "history", "runtime"))
            and not treatment["packet_refs"],
        "baseline_initialization_recorded": len(initialized) == 1 and initialized[0].sequence < first.event_sequence
            and initialized[0].payload.get("source_checkpoint_id") == source.checkpoint_id
            and initialized[0].payload.get("policy") == POLICY,
        "baseline_policy_disclosure_recorded": len(installed) == 1 and installed[0].sequence < first.event_sequence
            and installed[0].payload.get("policy") == POLICY
            and artifact(installed[0].payload["disclosed_messages_ref"]) == additions,
        "baseline_history_matches_source_plus_policy": messages(first) == source_bundle["messages"] + additions,
    }
    requests = [e for e in events if e.kind == "model_request"]
    checks["baseline_initial_api_history_matches"] = not requests or (
        requests[0].sequence > first.event_sequence
        and store.get_json(requests[0].input_ref)["messages"] == visible(messages(first)))
    event_by_sequence = {e.sequence: e for e in events}
    prepared, memory, generation, offered = {}, [], 0, []
    active = [cp.checkpoint_id for cp in store.list_checkpoints(source.run_id)
        if cp.ordinal <= source.ordinal and cp.event_sequence <= source.event_sequence and cp.branchable
        and all(getattr(cp.scope, key) == getattr(source.scope, key) for key in
                ("task_id", "stage_id", "target_id", "requirement_version", "evaluator_version"))]
    ancestry_valid = bool(active) and active[-1] == source.checkpoint_id
    registered = set()
    for event in events:
        if event.kind != "baseline":
            continue
        p = event.payload
        transition = p.get("transition")
        if transition == "checkpoint_registered":
            cp = store.get_checkpoint(p["checkpoint_id"])
            ancestry_valid &= (cp.run_id == first.run_id and cp.checkpoint_id not in registered
                and cp.event_sequence < event.sequence and p.get("generation") == generation
                and bool(active) and p.get("active_parent") == active[-1])
            registered.add(cp.checkpoint_id)
            active.append(cp.checkpoint_id)
        elif transition == "candidates_offered":
            offered = p["candidate_ids"]
            ancestry_valid &= offered == active[:-1][-POLICY["candidate_limit"]:] and p.get("generation") == generation
        elif transition == "rewind_prepared":
            if p["plan_id"] in prepared:
                ancestry_valid = False
            prepared[p["plan_id"]] = event
        elif transition == "rewind_applied":
            generation += 1
            key = f"baseline_rewind_{generation}"
            plan_event = prepared.pop(p["plan_id"], None)
            if plan_event is None:
                checks[key + "_authorized"] = False
                continue
            plan = plan_event.payload
            donor = store.get_checkpoint(p["donor_checkpoint_id"])
            worker = event_by_sequence.get(plan["response_event_sequence"])
            output = store.get_json(worker.output_ref) if worker and worker.output_ref else {}
            calls = output.get("tool_calls") or []
            args = calls[0].get("arguments") if len(calls) == 1 else None
            args = json.loads(args) if isinstance(args, str) else args
            expected_args = {"record_uid": donor.checkpoint_id, "memory_summary": plan["memory_summary"]}
            if (isinstance(args, dict) and "reason" in args) or plan.get("reason"):
                expected_args["reason"] = plan.get("reason", "")
            checks[key + "_authorized"] = (worker is not None and worker.kind == "model_response"
                and worker.actor == "worker" and worker.visibility == Visibility.WORKER_VISIBLE
                and worker.payload.get("accepted") is True and len(calls) == 1
                and calls[0].get("name") == "backtrack_commit" and args == expected_args
                and isinstance(plan["memory_summary"], str) and bool(plan["memory_summary"].strip())
                and worker.sequence < plan_event.sequence < event.sequence
                and plan["donor"]["checkpoint_id"] == donor.checkpoint_id
                and plan["generation"] == generation - 1 and plan["memory_ordinal"] == generation
                and donor.checkpoint_id in offered and donor.checkpoint_id in active[:-1])
            memory.append({"ordinal": generation, "record_uid": donor.checkpoint_id,
                "response_event_sequence": plan["response_event_sequence"], "memory_summary": plan["memory_summary"],
                "reason": plan["reason"], "source_run_id": first.run_id, "plan_id": p["plan_id"]})
            checks[key + "_cumulative_worker_memory"] = p.get("memory") == memory and p.get("generation") == generation
            expected = [m for m in messages(donor) if not (m.get("role") == "system" and
                any(m.get("extra", {}).get(k) for k in ("agentrewind_policy", "agentrewind_memory")))]
            expected.append(additions[0])
            if additions[1] not in expected:
                expected.append(additions[1])
            expected.append({"role": "system", "content": "Backtrack memory notebook from abandoned future branches:\n"
                + json.dumps(memory, ensure_ascii=False), "extra": {"agentrewind_memory": POLICY["policy_id"]}})
            following_cp = next((cp for cp in checkpoints if cp.event_sequence > event.sequence), None)
            checks[key + "_exact_restored_history"] = (artifact(p["delivered_history_ref"]) == expected
                and following_cp is not None and messages(following_cp) == expected)
            following_request = next((r for r in requests if r.sequence > event.sequence), None)
            checks[key + "_api_history_matches"] = following_request is None or (
                store.get_json(following_request.input_ref)["messages"] == visible(expected))
            if donor.checkpoint_id in active:
                active = active[:active.index(donor.checkpoint_id) + 1]
            else:
                ancestry_valid = False
            offered = []
    checks["baseline_active_ancestry_matches"] = ancestry_valid
    return checks


def verify_restoration_bindings(store: ArtifactStore) -> dict[str, Any]:
    """Compare actual branch baseline captures with their declared donor bytes.

    This checks executed restoration receipts and component provenance. It is
    additional to, not a replacement for, the adapter's private-access probes.
    """
    from .execution.environment import comparable
    rows, errors, container_ids = [], [], set()
    fresh_contexts = {}
    with store._connect() as db:
        run_bindings = dict(db.execute("SELECT record_id, run_id FROM records WHERE kind='treatment'"))
    for row in store.list_records("treatment"):
        treatment = store.get_record("treatment", row["record_id"])
        run_id = run_bindings[row["record_id"]]
        checks = {}
        try:
            cps = sorted(store.list_checkpoints(run_id), key=lambda c: c.ordinal)
            if not cps:
                raise ValueError("Branch has no captured baseline")
            first = cps[0]
            source = store.get_checkpoint(treatment["source_checkpoint_id"])
            donor = store.get_checkpoint(treatment["component_donors"]["workspace"])
            expected_w = store.get_json(from_jsonable(ArtifactRef, donor.validation["workspace_manifest_ref"]))
            actual_w = store.get_json(from_jsonable(ArtifactRef, first.validation["workspace_manifest_ref"]))
            checks["workspace_matches_donor"] = comparable(expected_w) == comparable(actual_w)
            checks["target_stays_fixed"] = all(c.scope.target_id == treatment["target_id"] and
                c.scope.evaluator_version == source.scope.evaluator_version for c in cps)
            if treatment["runtime_policy"] == "restore_captured":
                r_donor = store.get_checkpoint(treatment["component_donors"]["runtime"])
                expected_r, actual_r = store.get_json(r_donor.state.runtime), store.get_json(first.state.runtime)
                checks["runtime_matches_donor"] = (comparable(expected_r["entries"]) == comparable(actual_r["entries"])
                    and sorted(expected_r["deleted"]) == sorted(actual_r["deleted"]))
            history = store.get_json(first.state.history)["messages"]
            h_id = treatment["component_donors"].get("history", source.checkpoint_id)
            h_donor = store.get_checkpoint(h_id) if h_id != "fresh" else source
            source_history = store.get_json(h_donor.state.history)["messages"]
            if "manual_experiment" in treatment.get("metadata", {}):
                from .experiments.manual import validate_manual_manifest
                from .contracts import TreatmentManifest
                checks["manual_manifest_valid"] = not validate_manual_manifest(from_jsonable(TreatmentManifest, treatment), store)
            policy = treatment["history_policy"]
            manual_packets = bool("manual_experiment" in treatment.get("metadata", {})
                                  and policy != "fresh" and treatment["packet_refs"])
            restored_history = history
            if manual_packets:
                packets = [store.get_bytes(from_jsonable(ArtifactRef, ref)).decode("utf-8")
                           for ref in treatment["packet_refs"]]
                restored_history = history[:-len(packets)]
                checks["manual_packet_delivery_exact"] = len(history) == len(source_history) + len(packets) and all(
                    message.get("role") == "user" and message.get("content") == packet
                    for message, packet in zip(history[-len(packets):], packets))
            if treatment.get("metadata", {}).get("arm_id") == "agentrewind":
                checks.update(_verify_agentrewind_history(store, treatment, source, first, cps,
                                                        list(store.iter_events(run_id))))
            elif policy == "native":
                checks["history_matches_source"] = restored_history == source_history
            elif policy == "o_masked":
                from .experiments.observation_mask import REVISION, apply_observation_mask, validate_observation_mask
                mask = treatment["metadata"].get("mask") or {}
                if mask.get("revision") == REVISION:
                    checks["mask_window_binding_valid"] = not validate_observation_mask(store, h_donor, mask,
                        selection_kind=treatment["selection_kind"])
                    expected, mask_report = apply_observation_mask(source_history, mask)
                    checks["mask_preserves_surrounding_history"] = restored_history == expected
                    restored = [e for e in store.iter_events(run_id) if e.kind == "lifecycle"
                        and e.payload.get("transition") == "history_restored"]
                    checks["mask_application_receipt_exact"] = (len(restored) == 1
                        and restored[0].payload.get("mask") == mask_report
                        and store.get_json(from_jsonable(ArtifactRef, restored[0].payload["delivered_history_ref"])) == expected)
                else:
                    # Read historical all-history masks by their original
                    # semantics; new manifests require the bound policy above.
                    placeholder = mask.get("placeholder", "[tool output withheld]")
                    checks["mask_preserves_surrounding_history"] = len(history) == len(source_history) and all(
                        actual == ({**original, "content": placeholder,
                                    "extra": {**original.get("extra", {}), "masked": True}}
                                   if original.get("role") == "tool" else original)
                        for actual, original in zip(history, source_history))
                requests = [e for e in store.iter_events(run_id) if e.kind == "model_request"]
                if requests:
                    request = store.get_json(requests[0].input_ref)
                    # Extra is recorder metadata, including protected raw output.
                    # It must never leak through the actual API request.
                    checks["mask_delivered_without_recorder_metadata"] = request["messages"] == [
                        {k: v for k, v in m.items() if k != "extra"} for m in history]
                else:
                    checks["mask_delivered_without_recorder_metadata"] = False
            elif policy == "fresh":
                packets = [store.get_bytes(from_jsonable(ArtifactRef, ref)).decode() for ref in treatment["packet_refs"]]
                checks["packet_bytes_match_manifest"] = all(
                    m.get("role") == "user" and m.get("content") == packet
                    for m, packet in zip(history[len(history)-len(packets):], packets)) if packets else True
                checks["packet_count_supported"] = len(history) >= len(packets) + 2
                # Exclude operational metadata from the model-visible prefix.
                prefix = [{k: v for k, v in m.items() if k != "extra"}
                          for m in (history[:-len(packets)] if packets else history)]
                key = (source.checkpoint_id, treatment["worker_model"])
                expected_prefix = fresh_contexts.setdefault(key, prefix)
                checks["fresh_context_identical_across_treatments"] = prefix == expected_prefix
            starts = [e.payload for e in store.iter_events(run_id) if e.kind == "lifecycle"
                      and e.payload.get("transition") == "branch_started"]
            checks["one_branch_start"] = len(starts) == 1
            if starts:
                start = starts[0]
                container = start["environment"]["container_id"]
                checks["distinct_branch_container"] = container not in container_ids
                container_ids.add(container)
                checks["workspace_restore_receipt_validated"] = start["workspace_restore"].get("validated") is True
                checks["runtime_restore_receipt_validated"] = start["runtime_restore"].get("validated") is True
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                errors.append({"run_id": run_id, "failed_checks": failed})
            rows.append({"run_id": run_id, "treatment_id": treatment["treatment_id"], "checks": checks})
        except Exception as exc:
            errors.append({"run_id": run_id, "error": str(exc)})
    return {"valid": not errors and bool(rows), "branches_checked": len(rows), "branches": rows,
            "errors": errors, "claim": "Executed branch baseline fidelity and declared treatment provenance; private-access isolation requires adapter controls."}
