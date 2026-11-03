"""Persist observer artifacts through the shared store and cost ledger.

Record kinds written with store.put_record (all research-private):

  t_version       id = TVersion.version_id()          content: TVersion.as_record()
  t_advisor       id = <version_id>:advisor           content: diverted extractor-inference ops
  t_chain         id = <run_id>:<checkpoint_id>       content: pointer to the primary version at a checkpoint
  t_audit         id = AuditReport primary:reference  content: AuditReport.as_record()
  summary         id = SummaryArtifact.artifact_id()  content: SummaryArtifact.as_record()
  packet          id = Packet.packet_id()             content: Packet.as_record()
  observer_call   id = CallReceipt.call_record_id     content: CallReceipt.as_record() + raw response ref

Every model call is also recorded in the CostLedger under its cost role, so
extraction, summary, audit and advisor costs stay separate from worker
compute. Cost records carry no source_event_sequence: observer calls are not
worker events and must not be misdated as worker compute at the checkpoint.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..accounting import CostLedger
from ..contracts import ArtifactRef, ArtifactStoreProtocol, Usage, Visibility, from_jsonable
from ..persistence import persist, preserve_failure
from .audit import AuditReport
from .client import CallReceipt
from .render import Packet
from .summary import SummaryArtifact
from .tstore import TVersion


def record_calls(store: ArtifactStoreProtocol, run_id: str, receipts: Sequence[CallReceipt],
                 raw_response: str | None) -> list[ArtifactRef]:
    refs = []
    ledger = CostLedger(store)  # type: ignore[arg-type]
    for r in receipts:
        evidence = {"receipt": r.as_record(), "raw_response": raw_response}
        try:
            raw_ref = None
            if raw_response is not None:
                raw_ref = persist(lambda: store.put_bytes(raw_response.encode("utf-8"), media_type="text/plain",
                                          visibility=Visibility.RESEARCH_PRIVATE), evidence=evidence)
            body = {**r.as_record(), "raw_response_ref": None if raw_ref is None else
                    {"sha256": raw_ref.sha256, "size_bytes": raw_ref.size_bytes}}
            refs.append(persist(lambda: store.put_record("observer_call", r.call_record_id, body, run_id=run_id),
                                evidence=evidence))
            ledger.record(run_id=run_id, operation_id=r.call_record_id, role=r.cost_role,
                          usage=r.usage or Usage(measurement_basis="missing"), kind="model", status=r.status)
        except BaseException as exc:
            preserve_failure(exc, evidence)
            raise
    return refs


def save_version(store: ArtifactStoreProtocol, version: TVersion, *, advisor_ops: Sequence[Mapping[str, Any]] = (),
                 receipts: Sequence[CallReceipt] = (), raw_response: str | None = None,
                 primary: bool = False, chain_namespace: str | None = None) -> str:
    vid = version.version_id()
    evidence = {"version": version.as_record(), "advisor_ops": [dict(op) for op in advisor_ops],
                "receipts": [receipt.as_record() for receipt in receipts], "raw_response": raw_response}
    record_calls(store, version.run_id, receipts, raw_response)
    persist(lambda: store.put_record("t_version", vid, version.as_record(), run_id=version.run_id), evidence=evidence)
    if advisor_ops:
        advisor = {"role": "advisor_t", "source_version_id": vid, "ops": [dict(o) for o in advisor_ops],
                   "note": "extractor_inference content; never part of retention T"}
        persist(lambda: store.put_record("t_advisor", f"{vid}:advisor", advisor, run_id=version.run_id), evidence=evidence)
    if primary:
        if version.role != "t_retention":
            raise ValueError("only t_retention versions can be the primary chain pointer")
        chain_id = f"{version.run_id}:{version.target_checkpoint_id}"
        if chain_namespace:
            chain_id = f"{chain_namespace}:{chain_id}"
        pointer = {"run_id": version.run_id, "checkpoint_id": version.target_checkpoint_id,
                   "chain_namespace": chain_namespace,
                   "version_id": vid, "status": version.status, "update_index": version.update_index,
                   "parent_version_id": version.parent_version_id}
        persist(lambda: store.put_record("t_chain", chain_id, pointer, run_id=version.run_id), evidence=evidence)
    return vid


def load_version(store: ArtifactStoreProtocol, version_id: str) -> TVersion:
    version = TVersion.from_record(store.get_record("t_version", version_id))
    if version.version_id() != version_id:
        raise ValueError("Stored T version does not match its immutable identity")
    return version


def reusable_call(store: ArtifactStoreProtocol, version: TVersion) -> tuple[str | None, CallReceipt, str] | None:
    """Load one exact received extraction, never issue or retry an external call."""
    if len(version.call_record_ids) != 1:
        return None
    body = store.get_record("observer_call", version.call_record_ids[0])
    receipt = from_jsonable(CallReceipt, {key: value for key, value in body.items() if key != "raw_response_ref"})
    raw_ref = body.get("raw_response_ref")
    if receipt.status == "ok" and raw_ref is None:
        return None
    text = store.get_bytes(ArtifactRef(**raw_ref)).decode("utf-8") if raw_ref is not None else None
    return text, receipt, version.version_id()


def load_primary(store: ArtifactStoreProtocol, run_id: str, checkpoint_id: str, *, chain_namespace: str | None = None) -> TVersion | None:
    try:
        chain_id = f"{run_id}:{checkpoint_id}"
        if chain_namespace:
            chain_id = f"{chain_namespace}:{chain_id}"
        pointer = store.get_record("t_chain", chain_id)
    except KeyError:
        return None
    return load_version(store, pointer["version_id"])


def save_audit(store: ArtifactStoreProtocol, report: AuditReport) -> str:
    rid = f"{report.primary_version_id}:{report.reference_version_id}"
    store.put_record("t_audit", rid, report.as_record(), run_id=report.run_id)
    return rid


def save_summary(store: ArtifactStoreProtocol, summary: SummaryArtifact, raw_response: str | None = None) -> str:
    if summary.raw_responses:
        if len(summary.raw_responses) != len(summary.receipts):
            raise ValueError("summary raw responses must correspond one-to-one with receipts")
        for receipt, raw in zip(summary.receipts, summary.raw_responses):
            record_calls(store, summary.run_id, (receipt,), raw)
    else:
        # Legacy callers may supply a single raw response; never attach one
        # attempt's output to several different receipt identities.
        if raw_response is not None and len(summary.receipts) > 1:
            raise ValueError("multiple summary receipts require per-attempt raw responses")
        record_calls(store, summary.run_id, summary.receipts, raw_response)
    sid = summary.artifact_id()
    store.put_record("summary", sid, summary.as_record(), run_id=summary.run_id)
    return sid


def save_packet(store: ArtifactStoreProtocol, run_id: str, packet: Packet) -> tuple[str, ArtifactRef]:
    """Store the exact packet bytes as a blob and its record; returns (packet_id, bytes_ref)."""
    bytes_ref = store.put_bytes(packet.text.encode("utf-8"), media_type="text/plain",
                                visibility=Visibility.RESEARCH_PRIVATE)
    pid = packet.packet_id()
    store.put_record("packet", pid, {**packet.as_record(), "bytes_ref": {"sha256": bytes_ref.sha256,
                                                                          "size_bytes": bytes_ref.size_bytes}},
                     run_id=run_id)
    return pid, bytes_ref
