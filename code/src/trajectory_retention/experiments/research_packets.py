"""Immutable external research packets for manual fresh-history treatments (research-packet.v1).

A research packet is worker-visible text constructed outside this repository's live T pipeline (for example a
T2.1 view or an independently generated summary) and registered once, before any branch uses it. The record
binds the exact bytes (stored as a WORKER_VISIBLE artifact), a kind label, the natural run and source
checkpoint, the declared source-visibility scope, source/version identities, construction metadata and a
measured token count. Its ID is the content ID of the record body, so a record can never be silently replaced.

Registration checks only what the store can prove: the source checkpoint exists in the named natural run, the
declared event cut is that checkpoint's cut, the visibility interval does not pass it, and a current-session
scope names that checkpoint's native session. Legality for a particular treatment (same run, at or before the
decision checkpoint, fresh history) is decided by the manual builder and the common validator, never here.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from ..contracts import ArtifactRef, ArtifactStoreProtocol, Visibility, canonical_json, from_jsonable, to_jsonable
from ..observers._canonical import content_id

RESEARCH_PACKET_VERSION = "research-packet.v1"
RECORD_KIND = "research_packet"
PACKET_KIND = "research_packet"
BODY_FIELDS = ("label", "run_id", "source_checkpoint_id", "source_event_cut", "token_count", "token_measurement",
               "source_visibility", "source_ids", "construction")
SESSION_SCOPES = ("current_session", "full_trajectory")


class ResearchPacketError(ValueError):
    pass


def _check_body(body: Mapping[str, Any]) -> None:
    if not isinstance(body, Mapping) or set(body) != set(BODY_FIELDS):
        missing = sorted(set(BODY_FIELDS) - set(body or {}))
        extra = sorted(set(body or {}) - set(BODY_FIELDS))
        raise ResearchPacketError(f"research packet body needs exactly {BODY_FIELDS}; missing {missing}, unknown {extra}")
    for name in ("label", "run_id", "source_checkpoint_id"):
        if type(body[name]) is not str or not body[name]:
            raise ResearchPacketError(f"{name} must be a non-empty string")
    if type(body["source_event_cut"]) is not int or body["source_event_cut"] < 0:
        raise ResearchPacketError("source_event_cut must be a nonnegative integer")
    if type(body["token_count"]) is not int or body["token_count"] <= 0:
        raise ResearchPacketError("token_count must be a positive integer measured on the exact bytes")
    for name in ("token_measurement", "source_ids", "construction"):
        if not isinstance(body[name], Mapping) or not body[name]:
            raise ResearchPacketError(f"{name} must be a non-empty object")
    if type(body["token_measurement"].get("tokenizer")) is not str:
        raise ResearchPacketError("token_measurement.tokenizer must name the counter used")
    vis = body["source_visibility"]
    scope = vis.get("scope") if isinstance(vis, Mapping) else None
    if (not isinstance(scope, Mapping) or scope.get("session_scope") not in SESSION_SCOPES
            or type(scope.get("include_returned_reasoning")) is not bool):
        raise ResearchPacketError("source_visibility.scope needs session_scope and explicit include_returned_reasoning")
    if type(vis.get("through_seq")) is not int:
        raise ResearchPacketError("source_visibility.through_seq must be the last source event the packet may draw on")


def _record_body(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != "packet_id"}


def register_research_packet(store: ArtifactStoreProtocol, body: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Store exact worker-visible bytes and one immutable record; idempotent for identical inputs."""
    _check_body(body)
    if type(text) is not str or not text.strip():
        raise ResearchPacketError("research packet text must be non-empty")
    try:
        checkpoint = store.get_checkpoint(body["source_checkpoint_id"])
    except KeyError as exc:
        raise ResearchPacketError(f"source checkpoint {body['source_checkpoint_id']} missing") from exc
    if checkpoint.run_id != body["run_id"]:
        raise ResearchPacketError("source checkpoint belongs to a different run")
    if body["source_event_cut"] != checkpoint.event_sequence:
        raise ResearchPacketError("source_event_cut differs from the source checkpoint's event cut")
    vis = body["source_visibility"]
    if vis["through_seq"] > checkpoint.event_sequence:
        raise ResearchPacketError("declared source visibility extends past the source checkpoint (future leak)")
    if vis["scope"]["session_scope"] == "current_session" and vis.get("session_id") != checkpoint.scope.session_id:
        raise ResearchPacketError("current-session packet must name the source checkpoint's native session")
    data = text.encode("utf-8")
    ref = store.put_bytes(data, media_type="text/plain; charset=utf-8", visibility=Visibility.WORKER_VISIBLE)
    record = {"version": RESEARCH_PACKET_VERSION, **to_jsonable(dict(body)), "text_sha256": ref.sha256,
              "bytes_ref": to_jsonable(ref)}
    packet_id = f"rp:{content_id(record)}"
    record = {"packet_id": packet_id, **record}
    store.put_record(RECORD_KIND, packet_id, record, run_id=body["run_id"])
    return record


def load_research_packet(store: ArtifactStoreProtocol, packet_id: str) -> tuple[dict[str, Any], ArtifactRef, str]:
    """Return (record, exact bytes ref, text) after identity and byte checks."""
    try:
        record = store.get_record(RECORD_KIND, packet_id)
    except KeyError as exc:
        raise ResearchPacketError(f"research packet {packet_id} missing") from exc
    if record.get("version") != RESEARCH_PACKET_VERSION or record.get("packet_id") != packet_id:
        raise ResearchPacketError("research packet record identity or version differs from its ID")
    if f"rp:{content_id(_record_body(record))}" != packet_id:
        raise ResearchPacketError("research packet record content does not match its content ID")
    _check_body({k: record[k] for k in BODY_FIELDS if k in record})
    ref = from_jsonable(ArtifactRef, record["bytes_ref"])
    if ref.visibility != Visibility.WORKER_VISIBLE or ref.sha256 != record["text_sha256"]:
        raise ResearchPacketError("research packet bytes are not the recorded worker-visible artifact")
    try:
        data = store.get_bytes(ref)
    except Exception as exc:
        raise ResearchPacketError(f"research packet bytes unavailable: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != record["text_sha256"]:
        raise ResearchPacketError("research packet bytes differ from the recorded hash")
    return record, ref, data.decode("utf-8")


def research_packet_problems(store: ArtifactStoreProtocol, packet_meta: Mapping[str, Any],
                             packet_refs: tuple[ArtifactRef, ...], source_scope: Mapping[str, Any],
                             run_id: str, packet_source_id: str, decision_event_cut: int) -> list[str]:
    """Common-validator checks for an injected research packet (no suppression of any other check)."""
    problems: list[str] = []
    try:
        record, ref, _ = load_research_packet(store, packet_meta.get("packet_id"))
    except (ResearchPacketError, TypeError) as exc:
        return [f"research packet provenance invalid: {exc}"]
    if len(packet_refs) != 1 or packet_refs[0].sha256 != record["text_sha256"] or canonical_json(
            to_jsonable(packet_refs[0])) != canonical_json(to_jsonable(ref)):
        problems.append("injected bytes differ from the registered research packet")
    if record["run_id"] != run_id:
        problems.append("research packet is bound to a different run")
    if record["source_checkpoint_id"] != packet_source_id:
        problems.append("research packet is from a different source checkpoint")
    if record["source_event_cut"] > decision_event_cut or record["source_visibility"]["through_seq"] > decision_event_cut:
        problems.append("research packet draws on events after the decision checkpoint")
    if packet_meta.get("label") != record["label"]:
        problems.append("research packet label differs from treatment metadata")
    scope = record["source_visibility"]["scope"]
    if (scope.get("session_scope") != source_scope.get("session_scope")
            or scope.get("include_returned_reasoning") != source_scope.get("include_returned_reasoning")):
        problems.append("declared treatment source scope differs from the research packet's frozen information")
    return problems
