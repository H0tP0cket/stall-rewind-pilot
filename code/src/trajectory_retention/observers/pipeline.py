"""End-to-end observer orchestration for one recorded run.

Entry points for the campaign integration:

  observe_run(store, run_id, client, config, ...)
      Walk the run's checkpoints in order, extend the primary incremental
      T chain (idempotent: existing t_chain pointers are reused), persist
      versions/advisor/calls.

  audit_checkpoint(store, run_id, checkpoint_id, client, config, repeats)
      Independent full-prefix extraction(s) and comparison against the
      persisted primary version. Never modifies the primary chain.

  freeze_packets(store, run_id, checkpoint_id, ...)
      From the persisted primary T at the checkpoint, render the full typed
      packet, the same-T prose packet, the deterministic filtered views, and
      (with a client) the matched independent summary. Returns packet IDs and
      byte refs for treatment manifests.

All functions read only through PrefixView; none touches worker state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import ArtifactRef, ArtifactStoreProtocol, Checkpoint, from_jsonable, new_id, to_jsonable
from ..persistence import persist
from . import artifacts
from ._canonical import content_id, sha256_hex
from .audit import audit_recorded_ancestry, compare_versions, repeat_agreement
from .client import ObserverClient, CallReceipt
from .extractor import CheckpointInfo, ExtractorConfig, assert_lineage_step, extract_full_prefix, update_t
from .prefix import SourceScope, SESSION_LINEAGE_REVISION, view_for_checkpoint
from .prompts import render_events
from .render import Packet, render_prose, render_typed
from .summary import generate_summary, summary_prompt_binding, words_for_tokens
from .tokens import TokenCounter, TokenMatchConfig, byte_proxy_counter
from .context import RequestTokenCounter
from .tstore import TVersion
from .update import validate_version
from .views import FILTERS, filtered_view


def _info(cp: Checkpoint) -> CheckpointInfo:
    return CheckpointInfo(cp.checkpoint_id, cp.event_sequence, cp.scope.session_id, cp.scope.requirement_version)


def _checkpoints(store: ArtifactStoreProtocol, run_id: str) -> list[Checkpoint]:
    lister = getattr(store, "list_checkpoints", None)
    if lister is None:
        raise TypeError("store must support list_checkpoints")
    cps = sorted(lister(run_id), key=lambda c: c.ordinal)
    for a, b in zip(cps, cps[1:]):
        if b.event_sequence < a.event_sequence:
            raise ValueError(f"checkpoint {b.checkpoint_id} cut precedes {a.checkpoint_id}")
    return cps


@dataclass(frozen=True)
class ObserveResult:
    run_id: str
    version_ids: tuple[str, ...]
    statuses: tuple[str, ...]
    reused: tuple[bool, ...]


def observe_run(
    store: ArtifactStoreProtocol,
    run_id: str,
    client: ObserverClient,
    config: ExtractorConfig,
    *,
    scope: SourceScope | None = None,
    through_checkpoint_id: str | None = None,
    chain_namespace: str | None = None,
    reuse_chain_namespace: str | None = None,
    context_counter: RequestTokenCounter | None = None,
) -> ObserveResult:
    cps = _checkpoints(store, run_id)
    if through_checkpoint_id is not None and through_checkpoint_id not in {c.checkpoint_id for c in cps}:
        raise KeyError(f"checkpoint {through_checkpoint_id} is not in run {run_id}")
    cuts = {c.checkpoint_id: c.event_sequence for c in cps}
    parent: TVersion | None = None
    prior: CheckpointInfo | None = None
    ids, statuses, reused = [], [], []
    effective_scope = scope or SourceScope()
    if effective_scope.lineage_revision == SESSION_LINEAGE_REVISION:
        # A repeated ID after another native session would make prefix filtering
        # include a prior lifetime. Detect this before issuing any observer call.
        closed_sessions = set()
        previous_session = None
        for cp in cps:
            session = cp.scope.session_id
            if type(session) is not str or not session:
                raise ValueError("Versioned observer lineage requires an identified native session")
            if session != previous_session:
                if session in closed_sessions:
                    raise ValueError("Native session identity reappears after its closed lifetime")
                if previous_session is not None:
                    closed_sessions.add(previous_session)
                previous_session = session
            if through_checkpoint_id == cp.checkpoint_id:
                break
    for cp in cps:
        info = _info(cp)
        if (effective_scope.lineage_revision == SESSION_LINEAGE_REVISION
                and effective_scope.session_scope == "current_session"
                and prior is not None and prior.session_id != info.session_id):
            # The archived parent remains immutable. It is not an input to this
            # session's primary extraction, even if its entries were filtered.
            parent, prior = None, None
        existing = artifacts.load_primary(store, run_id, cp.checkpoint_id, chain_namespace=chain_namespace)
        if existing is not None:
            assert_lineage_step(existing, run_id=run_id, parent=parent, checkpoint=info,
                                prior_checkpoint=prior, config=config, scope=scope)
            view = view_for_checkpoint(store, run_id, through_seq=cp.event_sequence,
                                       target_session_id=cp.scope.session_id, scope=scope)
            violations = validate_version(existing, view)
            if violations:
                raise ValueError(f"frozen primary T failed source validation: {violations}")
            out_version, was_reused = existing, True
        else:
            claim_id = content_id({"revision": "observer-update-claim.v2", "run_id": run_id,
                                   "checkpoint_id": cp.checkpoint_id, "chain_namespace": chain_namespace})
            source_view = view_for_checkpoint(store, run_id, through_seq=cp.event_sequence,
                after_seq=prior.event_cut_seq if prior else -1, target_session_id=cp.scope.session_id, scope=scope)
            claim_binding = {"checkpoint": to_jsonable(info), "chain_namespace": chain_namespace,
                             "parent_version_id": parent.version_id() if parent else None,
                             "config_id": config.config_id(), "source_text_sha256": sha256_hex(render_events(source_view.events())),
                             "source_visibility": source_view.visibility_manifest()}
            try:
                completed = store.get_record("observer_update_finished_v2", claim_id)
            except KeyError:
                completed = None
            if completed is not None:
                if completed["binding"] != claim_binding:
                    raise ValueError("completed T-update binding changed; start an explicit new observer chain")
                out_version = artifacts.load_version(store, completed["version_id"])
                assert_lineage_step(out_version, run_id=run_id, parent=parent, checkpoint=info,
                                    prior_checkpoint=prior, config=config, scope=scope)
                violations = validate_version(out_version, source_view)
                if violations:
                    raise ValueError(f"completed T update failed source validation: {violations}")
                was_reused = True
            else:
                try:
                    received = store.get_record("observer_update_received_v3", claim_id)
                except KeyError:
                    received = None
                try:
                    started = store.get_record("observer_update_started_v2", claim_id)
                except KeyError:
                    started = {"binding": claim_binding, "claim_id": new_id("observer_claim")}
                    persist(lambda: store.put_record("observer_update_started_v2", claim_id, started, run_id=run_id),
                            evidence=started)
                else:
                    if started["binding"] != claim_binding:
                        raise ValueError("started T-update binding changed; start an explicit new observer chain")
                    if received is None:
                        raise ValueError("T update has no completion receipt; inspect the claim before an explicit new chain")
                was_reused = received is not None
                if received is None:
                    reusable = None
                    if reuse_chain_namespace is not None:
                        source = artifacts.load_primary(store, run_id, cp.checkpoint_id, chain_namespace=reuse_chain_namespace)
                        if source is not None and source.extractor_config_id == config.config_id():
                            reusable = artifacts.reusable_call(store, source)
                    out = update_t(store=store, run_id=run_id, parent=parent, checkpoint=info, prior_checkpoint=prior,
                                   checkpoint_cuts=cuts, client=client, config=config, scope=scope,
                                   reusable_call=reusable, context_counter=context_counter)
                    # Retain actual call/cost identities and a reconstructable
                    # result before publishing either the version or pointer.
                    artifacts.record_calls(store, run_id, out.receipts, out.raw_response)
                    received = {"binding": claim_binding, "version": out.version.as_record(),
                                "advisor_ops": [dict(op) for op in out.advisor_ops],
                                "receipts": [receipt.as_record() for receipt in out.receipts], "raw_response": out.raw_response}
                    persist(lambda: store.put_record("observer_update_received_v3", claim_id, received, run_id=run_id),
                            evidence=received)
                if received["binding"] != claim_binding:
                    raise ValueError("received T-update binding changed; refusing replay")
                out_version = TVersion.from_record(received["version"])
                assert_lineage_step(out_version, run_id=run_id, parent=parent, checkpoint=info,
                                    prior_checkpoint=prior, config=config, scope=scope)
                violations = validate_version(out_version, source_view)
                if violations:
                    raise ValueError(f"received T update failed source validation: {violations}")
                artifacts.save_version(store, out_version, advisor_ops=received["advisor_ops"],
                    receipts=tuple(from_jsonable(CallReceipt, receipt) for receipt in received["receipts"]),
                    raw_response=received["raw_response"])
                completion = {"binding": claim_binding, "version_id": out_version.version_id()}
                persist(lambda: store.put_record("observer_update_finished_v2", claim_id, completion, run_id=run_id),
                        evidence=received)
            artifacts.save_version(store, out_version, primary=True, chain_namespace=chain_namespace)
        ids.append(out_version.version_id())
        statuses.append(out_version.status)
        reused.append(was_reused)
        parent, prior = out_version, info
        if through_checkpoint_id is not None and cp.checkpoint_id == through_checkpoint_id:
            break
    return ObserveResult(run_id, tuple(ids), tuple(statuses), tuple(reused))


@dataclass(frozen=True)
class ScopedObserveResult:
    run_id: str
    primary: ObserveResult
    archive: ObserveResult
    primary_namespace: str
    archive_namespace: str
    pair_ids: tuple[str, ...]
    source_policy_revision: str = SESSION_LINEAGE_REVISION


def observe_scoped_run(store: ArtifactStoreProtocol, run_id: str, client: ObserverClient,
                       config: ExtractorConfig, *, namespace: str,
                       include_returned_reasoning: bool = False,
                       through_checkpoint_id: str | None = None,
                       context_counter: RequestTokenCounter | None = None) -> ScopedObserveResult:
    """Q12: global archival T and independently clean native-session primary T.

    The namespace is explicit: this never replaces a legacy primary pointer.
    Identical first-session requests reuse the received call exactly once. Global
    archive links are research references and never primary extractor inputs.
    """
    if type(namespace) is not str or not namespace.strip():
        raise ValueError("Session-scoped extraction requires an explicit new namespace")
    primary_namespace, archive_namespace = f"{namespace}:primary-session-v1", f"{namespace}:archive-global-v1"
    common = {"include_returned_reasoning": include_returned_reasoning, "lineage_revision": SESSION_LINEAGE_REVISION}
    archive = observe_run(store, run_id, client, config, scope=SourceScope(session_scope="full_trajectory", **common),
        chain_namespace=archive_namespace, through_checkpoint_id=through_checkpoint_id, context_counter=context_counter)
    primary = observe_run(store, run_id, client, config, scope=SourceScope(session_scope="current_session", **common),
        chain_namespace=primary_namespace, through_checkpoint_id=through_checkpoint_id,
        reuse_chain_namespace=archive_namespace, context_counter=context_counter)
    pairs = []
    for primary_id, archive_id in zip(primary.version_ids, archive.version_ids, strict=True):
        current, global_version = artifacts.load_version(store, primary_id), artifacts.load_version(store, archive_id)
        if current.target_checkpoint_id != global_version.target_checkpoint_id:
            raise ValueError("Observer source scopes do not bind the same checkpoint")
        pair_id = content_id({"revision": "observer-scope-pair/v1", "namespace": namespace,
                             "run_id": run_id, "checkpoint_id": current.target_checkpoint_id})
        binding = {"revision": "observer-scope-pair/v1", "namespace": namespace, "run_id": run_id,
                   "checkpoint_id": current.target_checkpoint_id, "primary_version_id": primary_id,
                   "archive_version_id": archive_id, "primary_namespace": primary_namespace,
                   "archive_namespace": archive_namespace, "native_session_id": current.source_visibility["target_session_id"],
                   "primary_scope": current.source_visibility["scope"], "archive_scope": global_version.source_visibility["scope"],
                   "archive_access": "retained_secondary_reference_not_primary_extractor_input",
                   "primary_entry_access": [{"entry_id": entry.entry_id, "source_session_id": entry.session_id,
                       "source_seqs": [span.seq for span in entry.spans],
                       "access": "observed_in_current_native_session", "applicability": "unknown_unless_separately_supported"}
                       for entry in current.entries],
                   "shared_call_record_ids": sorted(set(current.call_record_ids) & set(global_version.call_record_ids))}
        persist(lambda: store.put_record("observer_scope_pair_v1", pair_id, binding, run_id=run_id), evidence=binding)
        pairs.append(pair_id)
    return ScopedObserveResult(run_id, primary, archive, primary_namespace, archive_namespace, tuple(pairs))


def audit_checkpoint(
    store: ArtifactStoreProtocol,
    run_id: str,
    checkpoint_id: str,
    client: ObserverClient,
    config: ExtractorConfig,
    *,
    repeats: int = 1,
    scope: SourceScope | None = None,
    chain_namespace: str | None = None,
    context_counter: RequestTokenCounter | None = None,
) -> dict[str, Any]:
    primary = artifacts.load_primary(store, run_id, checkpoint_id, chain_namespace=chain_namespace)
    if primary is None:
        raise KeyError(f"no primary T at {checkpoint_id}")
    cps = _checkpoints(store, run_id)
    cuts = {c.checkpoint_id: c.event_sequence for c in cps}
    cp = next(c for c in cps if c.checkpoint_id == checkpoint_id)
    view = view_for_checkpoint(store, run_id, through_seq=cp.event_sequence, target_session_id=cp.scope.session_id,
                               scope=scope)
    refs: list[TVersion] = []
    audit_ids = []
    for i in range(repeats):
        out = extract_full_prefix(store=store, run_id=run_id, checkpoint=_info(cp), checkpoint_cuts=cuts,
                                  client=client, config=config, scope=scope, repeat_index=i, context_counter=context_counter)
        artifacts.save_version(store, out.version, advisor_ops=out.advisor_ops, receipts=out.receipts,
                               raw_response=out.raw_response)
        refs.append(out.version)
        if out.version.status != "failed":
            audit_ids.append(artifacts.save_audit(store, compare_versions(primary, out.version, view)))
    return {"primary_version_id": primary.version_id(), "reference_version_ids": [v.version_id() for v in refs],
            "audit_ids": audit_ids, "repeat_agreement": repeat_agreement([v for v in refs if v.status != "failed"])}


@dataclass(frozen=True)
class FrozenPackets:
    run_id: str
    checkpoint_id: str
    version_id: str
    packets: Mapping[str, tuple[str, ArtifactRef]]  # key -> (packet_id, bytes_ref)
    token_counts: Mapping[str, int]
    summary_id: str | None
    summary_ref: ArtifactRef | None
    freeze_id: str | None = None
    token_measurements: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    matching_report: Mapping[str, Any] = field(default_factory=dict)
    binding: Mapping[str, Any] = field(default_factory=dict)


def freeze_packets(
    store: ArtifactStoreProtocol,
    run_id: str,
    checkpoint_id: str,
    *,
    filters: Sequence[str] = ("facts_only", "facts_lessons"),
    client: ObserverClient | None = None,
    config: ExtractorConfig | None = None,
    scope: SourceScope | None = None,
    chain_namespace: str | None = None,
    token_counter: TokenCounter | None = None,
    context_counter: RequestTokenCounter | None = None,
    matching: TokenMatchConfig | None = None,
    freeze_namespace: str | None = None,
) -> FrozenPackets:
    """Freeze one immutable packet set, including independent summary attempts.

    An existing namespace is reused only with identical source/configuration,
    actual packet bytes and token measurements. Changing a renderer, filter,
    prompt, source scope or matching policy requires an explicit new namespace.
    Old campaign FrozenPackets snapshots remain readable through default fields.
    """
    if (client is None) != (config is None):
        raise ValueError("independent summary requires both client and config")
    version = artifacts.load_primary(store, run_id, checkpoint_id, chain_namespace=chain_namespace)
    if version is None:
        raise KeyError(f"no primary T at {checkpoint_id}")
    if version.status in {"failed", "missing"}:
        raise ValueError(f"primary T at {checkpoint_id} is {version.status}; T treatments must wait")
    ancestry_audit = audit_recorded_ancestry(store, version.version_id())
    if not ancestry_audit["prefix_and_ancestry_valid"]:
        raise ValueError("primary T ancestry contains invalid prefix exposure; use an explicit repaired observer chain")
    scope, matching = scope or SourceScope(), matching or TokenMatchConfig()
    if version.source_visibility.get("scope") != scope.as_record():
        raise ValueError("frozen T and independent summary/packet source scopes differ")
    cp = next(c for c in _checkpoints(store, run_id) if c.checkpoint_id == checkpoint_id)
    if version.through_seq != cp.event_sequence:
        raise ValueError("frozen T event cut differs from checkpoint")
    if config is not None and config.config_id() != version.extractor_config_id:
        raise ValueError("summary must use the frozen T extractor model/settings configuration")
    recorded_model = config.model if config else None
    ancestor = version
    while recorded_model is None:
        for call_id in ancestor.call_record_ids:
            recorded_model = store.get_record("observer_call", call_id)["requested_model"]
            if recorded_model:
                break
        if recorded_model or ancestor.parent_version_id is None:
            break
        ancestor = artifacts.load_version(store, ancestor.parent_version_id)
    counter = token_counter or byte_proxy_counter(recorded_model or "unspecified-worker-model")
    if recorded_model is not None and counter.model != recorded_model:
        raise ValueError("token measurement model differs from recorded extractor model")
    if config is not None and counter.model != config.model:
        raise ValueError("token measurement model differs from summary/worker model")
    if len(set(filters)) != len(filters):
        raise ValueError("duplicate filter names")
    packets: dict[str, Packet] = {"t_typed": render_typed(version), "t_prose": render_prose(version)}
    for f in filters:
        if f not in FILTERS:
            raise KeyError(f"unknown filter {f}")
        packets[f"t_view:{f}"] = filtered_view(version, f)
    measurements = {k: counter.measure(p.text, context={"run_id": run_id, "checkpoint_id": checkpoint_id,
                                  "source_version_id": version.version_id(), "packet_id": p.packet_id(), "arm": k})
                    for k, p in packets.items()}
    tokens = {k: m["token_count"] for k, m in measurements.items()}
    view = view_for_checkpoint(store, run_id, through_seq=cp.event_sequence,
                               target_session_id=cp.scope.session_id, scope=scope)
    source_hash = sha256_hex(render_events(view.events()))
    binding = {"revision": "frozen-packets.v2", "run_id": run_id, "checkpoint_id": checkpoint_id,
               "chain_namespace": chain_namespace, "freeze_namespace": freeze_namespace,
               "source_version_id": version.version_id(), "source_visibility": view.visibility_manifest(),
               "ancestry_validation": {"revision": ancestry_audit["revision"],
                   "prefix_and_ancestry_valid": ancestry_audit["prefix_and_ancestry_valid"],
                   "failed_or_missing_version_ids": ancestry_audit["failed_or_missing_version_ids"],
                   "quality_gate": "pending_calibration"},
               "source_text_sha256": source_hash, "summary_requested": config is not None,
               "summary_config_id": None if config is None else config.config_id(),
               "summary_prompt": summary_prompt_binding(matching.max_words or words_for_tokens(tokens["t_typed"])),
               "matching": matching.as_record(), "counter": counter.as_record(),
               "packets": {k: {"packet_id": p.packet_id(), "renderer_revision": p.renderer_revision,
                                "filter_id": p.filter_id, "measurement": measurements[k]} for k, p in packets.items()}}
    freeze_id = content_id({"revision": "frozen-packets.v2", "run_id": run_id,
                            "checkpoint_id": checkpoint_id, "chain_namespace": chain_namespace,
                            "freeze_namespace": freeze_namespace})
    try:
        frozen_record = store.get_record("frozen_packets_v2", freeze_id)
    except KeyError:
        frozen_record = None
    if frozen_record is not None:
        if frozen_record["binding"] != binding:
            raise ValueError("frozen packet source/configuration changed; use an explicit new freeze_namespace")
        frozen = from_jsonable(FrozenPackets, frozen_record["result"])
        for key, (_, ref) in frozen.packets.items():
            if store.get_bytes(ref).decode("utf-8") != packets[key].text:
                raise ValueError("frozen packet bytes disagree with bound renderer output")
        if frozen.summary_ref is not None:
            text = store.get_bytes(frozen.summary_ref).decode("utf-8")
            measurement = frozen.token_measurements["summary"]
            if counter.measure(text, context=measurement["context"]) != measurement:
                raise ValueError("frozen summary token measurement changed")
        return frozen
    try:
        started = store.get_record("packet_freeze_started_v2", freeze_id)
    except KeyError:
        store.put_record("packet_freeze_started_v2", freeze_id, binding, run_id=run_id)
    else:
        if started != binding:
            raise ValueError("started packet freeze configuration changed; use an explicit new freeze_namespace")
    saved = {k: artifacts.save_packet(store, run_id, p) for k, p in packets.items()}
    summary_id = summary_ref = None
    report: Mapping[str, Any] = {"status": "summary_not_requested", "provider_exact": False}
    if client is not None and config is not None:
        summary = generate_summary(store=store, run_id=run_id, checkpoint=_info(cp),
                                   matched_token_target=tokens["t_typed"], client=client, config=config, scope=scope,
                                   token_counter=counter, matching=matching, target_measurement=measurements["t_typed"],
                                   context_counter=context_counter,
                                   generation_id=f"{freeze_id}:summary")
        summary_id = artifacts.save_summary(store, summary)
        report = summary.matching_report
        if summary.token_measurement is not None:
            measurements["summary"] = summary.token_measurement
            tokens["summary"] = summary.token_measurement["token_count"]
        if summary.status == "valid" and summary.text is not None:
            summary_ref = store.put_bytes(summary.text.encode("utf-8"), media_type="text/plain")
    result = FrozenPackets(run_id, checkpoint_id, version.version_id(), saved, tokens, summary_id, summary_ref,
                           freeze_id, measurements, report, binding)
    store.put_record("frozen_packets_v2", freeze_id, {"binding": binding, "result": to_jsonable(result)}, run_id=run_id)
    return result


def measure_frozen_packets(
    store: ArtifactStoreProtocol,
    frozen: FrozenPackets,
    token_counter: TokenCounter,
    *,
    matching: TokenMatchConfig | None = None,
) -> dict[str, Any]:
    """Remeasure existing exact injected bytes without mutation or regeneration.

    This also accepts historical FrozenPackets snapshots. A new report does not
    change their treatment manifest, original length claim or summary selection.
    Both old and new measurements should be retained by the campaign exporter.
    """
    from .tokens import compare_measurements

    matching = matching or TokenMatchConfig()
    primary = artifacts.load_version(store, frozen.version_id)
    measurements: dict[str, Mapping[str, Any]] = {}
    for key, (packet_id, ref) in frozen.packets.items():
        packet = store.get_record("packet", packet_id)
        text = store.get_bytes(ref).decode("utf-8")
        if packet["source_version_id"] != frozen.version_id or packet["text_sha256"] != sha256_hex(text):
            raise ValueError("packet bytes/source version disagree with frozen treatment")
        measurements[key] = token_counter.measure(text, context={"run_id": frozen.run_id,
            "checkpoint_id": frozen.checkpoint_id, "source_version_id": frozen.version_id, "packet_id": packet_id})
    summary = None
    if frozen.summary_id is not None:
        summary = store.get_record("summary", frozen.summary_id)
        if (summary["role"] != "summary_independent" or summary["run_id"] != frozen.run_id
                or summary["target_checkpoint_id"] != frozen.checkpoint_id):
            raise ValueError("independent summary does not bind the frozen checkpoint")
        for key in ("scope", "target_session_id"):
            if summary["source_visibility"].get(key) != primary.source_visibility.get(key):
                raise ValueError("summary and typed T allowed source universes differ")
        if summary["source_visibility"]["interval"]["through_seq"] != primary.through_seq:
            raise ValueError("summary and typed T event cuts differ")
    if frozen.summary_ref is not None:
        text = store.get_bytes(frozen.summary_ref).decode("utf-8")
        if summary is None or summary["text_sha256"] != sha256_hex(text):
            raise ValueError("frozen summary bytes disagree with summary artifact")
        measurements["summary"] = token_counter.measure(text, context={"run_id": frozen.run_id,
            "checkpoint_id": frozen.checkpoint_id, "summary_id": frozen.summary_id})
    report = compare_measurements(measurements["t_typed"], measurements["summary"], matching) if "summary" in measurements else {
        "status": "summary_not_injected", "within_tolerance": None, "provider_exact": False}
    return {"revision": "frozen-packet-remeasurement.v1", "run_id": frozen.run_id,
            "checkpoint_id": frozen.checkpoint_id, "source_version_id": frozen.version_id,
            "original_freeze_id": frozen.freeze_id, "original_token_counts": dict(frozen.token_counts),
            "original_measurements": dict(frozen.token_measurements), "measurements": measurements,
            "summary_matching": report, "counter": token_counter.as_record(), "regenerated": False,
            "original_treatment_modified": False, "scientific_matching_calibrated": False}
