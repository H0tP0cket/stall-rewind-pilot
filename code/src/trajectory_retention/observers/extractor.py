"""Incremental T extraction and independent full-prefix extraction.

  T_i = update(T_(i-1), allowed_events(C_(i-1), C_i])

`update_t` performs one primary incremental step. `extract_full_prefix`
performs an independent reference extraction with no prior T. Both produce
immutable TVersion records with call receipts and a source-visibility
manifest; a failed model call yields a status="failed" version that carries
the parent's entries unchanged and is not usable for treatments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..contracts import ArtifactStoreProtocol
from ._canonical import content_id
from .client import CallReceipt, ModelRequest, ObserverClient
from .context import ContextPolicy, RequestTokenCounter, context_preflight
from .prefix import PrefixView, SourceScope, view_for_checkpoint
from .prompts import PROMPT_REVISION, prompt_hash, t_full_messages, t_update_messages
from .tstore import TVersion
from .update import OPS_SCHEMA_VERSION, VALIDATION_REVISION, VALIDATION_SCOPE, apply_ops, validate_version


@dataclass(frozen=True)
class ExtractorConfig:
    model: str
    settings: Mapping[str, Any]
    prompt_revision: str = PROMPT_REVISION
    ops_schema: str = OPS_SCHEMA_VERSION
    validation_revision: str | None = VALIDATION_REVISION
    context_policy: ContextPolicy | None = None

    def config_id(self) -> str:
        record = {
            "model": self.model,
            "settings": dict(self.settings),
            "prompt_revision": self.prompt_revision,
            "ops_schema": self.ops_schema,
        }
        if self.validation_revision is not None:
            record["validation_revision"] = self.validation_revision
        if self.context_policy is not None:
            record["context_policy"] = self.context_policy.as_record()
        return content_id(record)


@dataclass(frozen=True)
class CheckpointInfo:
    checkpoint_id: str
    event_cut_seq: int
    session_id: str | None
    requirement_version: str | None


@dataclass(frozen=True)
class ExtractionOutput:
    version: TVersion
    advisor_ops: tuple[Mapping[str, Any], ...]
    receipts: tuple[CallReceipt, ...]
    raw_response: str | None
    prompt_hash: str


def validate_lineage_step(
    version: TVersion, *, run_id: str, parent: TVersion | None,
    checkpoint: CheckpointInfo, prior_checkpoint: CheckpointInfo | None,
    config: ExtractorConfig, scope: SourceScope | None = None,
) -> list[str]:
    """Fail-closed cache binding, including historically allowed source scope.

    This checks immutable lineage metadata. Call validate_version with the
    corresponding prefix to additionally verify all recorded evidence bytes.
    """
    problems = []
    effective_scope = (scope or SourceScope()).as_record()
    expected = {
        "run_id": run_id, "role": "t_retention",
        "target_checkpoint_id": checkpoint.checkpoint_id,
        "parent_version_id": parent.version_id() if parent else None,
        "after_seq": prior_checkpoint.event_cut_seq if prior_checkpoint else -1,
        "through_seq": checkpoint.event_cut_seq,
        "update_index": parent.update_index + 1 if parent else 0,
        "extractor_config_id": config.config_id(),
    }
    for field_name, value in expected.items():
        if getattr(version, field_name) != value:
            problems.append(f"cached T {field_name} differs from requested lineage")
    visibility = version.source_visibility
    for field_name, value in {
        "run_id": run_id, "scope": effective_scope,
        "target_session_id": checkpoint.session_id,
        "interval": {"after_seq": expected["after_seq"], "through_seq": expected["through_seq"]},
    }.items():
        if visibility.get(field_name) != value:
            problems.append(f"cached T source visibility {field_name} differs")
    if (parent is None) != (prior_checkpoint is None):
        problems.append("parent and prior checkpoint must both be supplied or both absent")
    if parent is not None and prior_checkpoint is not None:
        if parent.run_id != run_id or parent.role != "t_retention":
            problems.append("parent is not retention T from this run")
        if parent.target_checkpoint_id != prior_checkpoint.checkpoint_id or parent.through_seq != prior_checkpoint.event_cut_seq:
            problems.append("parent does not bind the prior checkpoint")
        if parent.extractor_config_id != config.config_id():
            problems.append("parent extractor config differs; frozen chains cannot change validation/prompt/model")
        if parent.source_visibility.get("scope") != effective_scope:
            problems.append("parent source scope differs")
        if effective_scope["session_scope"] == "current_session" and prior_checkpoint.session_id != checkpoint.session_id:
            problems.append("current_session incremental chain crosses a session reset; scoped restart policy required")
    return problems


def assert_lineage_step(version: TVersion, **kwargs: Any) -> None:
    problems = validate_lineage_step(version, **kwargs)
    if problems:
        raise ValueError("; ".join(problems))


def parse_ops(text: str | None) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    if text is None:
        return None, "no response"
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    try:
        def reject_nonfinite(value: str) -> None:
            raise ValueError(f"nonfinite JSON constant {value}")

        parsed = json.loads(stripped, parse_constant=reject_nonfinite)
    except ValueError as exc:
        return None, f"unparseable ops JSON: {exc}"
    if isinstance(parsed, Mapping) and isinstance(parsed.get("ops"), list):
        parsed = parsed["ops"]
    if not isinstance(parsed, list):
        return None, "ops JSON is not an array"
    return parsed, None


def _run(
    *,
    store: ArtifactStoreProtocol,
    role: str,
    parent: TVersion | None,
    view: PrefixView,
    messages: tuple[Mapping[str, str], ...],
    client: ObserverClient,
    config: ExtractorConfig,
    cost_role: str,
    purpose: str,
    run_id: str,
    checkpoint: CheckpointInfo,
    update_index: int,
    checkpoint_cuts: Mapping[str, int],
    reusable_call: tuple[str | None, CallReceipt, str] | None = None,
    context_counter: RequestTokenCounter | None = None,
) -> ExtractionOutput:
    if (config.validation_revision != VALIDATION_REVISION or config.prompt_revision != PROMPT_REVISION
            or config.ops_schema != OPS_SCHEMA_VERSION):
        raise ValueError("new extraction requires the executable prompt/schema/validation revision")
    events = view.events()  # populates the access log
    request = ModelRequest(
        model=config.model, messages=messages, settings=config.settings,
        cost_role=cost_role, purpose=purpose, run_id=run_id,
        target_checkpoint_id=checkpoint.checkpoint_id,
    )
    base = dict(
        role=role, run_id=run_id, target_checkpoint_id=checkpoint.checkpoint_id,
        update_index=update_index,
        parent_version_id=parent.version_id() if parent else None,
        after_seq=view.after_seq, through_seq=view.through_seq,
        extractor_config_id=config.config_id(), call_record_ids=(),
        source_visibility={**view.visibility_manifest(), "prompt_hash": prompt_hash(messages),
                           "prompt_revision": config.prompt_revision,
                           "validation": dict(VALIDATION_SCOPE),
                           "eligible_event_count": len(events)},
    )
    if config.context_policy is not None:
        guard = context_preflight(store, request, config.context_policy, token_counter=context_counter)
        base["source_visibility"]["context_preflight"] = guard
        if not guard["client_call_allowed"]:
            version = TVersion(entries=parent.entries if parent else (), ops_applied=(), rejected_ops=(),
                               status="failed", failure="context preflight: " + guard["reason"], **base)
            return ExtractionOutput(version, (), (), None, prompt_hash(messages))
    reused_from = None
    if (reusable_call is not None and reusable_call[1].request_hash == request.request_hash()
            and reusable_call[1].requested_model == request.model and reusable_call[1].cost_role == request.cost_role):
        text, receipt, reused_from = reusable_call
    else:
        text, receipt = client.complete(request)
    ops, failure = parse_ops(text) if receipt.status == "ok" else (None, receipt.error or "call failed")
    if (receipt.request_hash != request.request_hash() or receipt.requested_model != request.model
            or receipt.cost_role != request.cost_role):
        ops, failure = None, "observer receipt does not bind the requested model, role and prompt"
    base["call_record_ids"] = (receipt.call_record_id,)
    if reused_from is not None:
        base["source_visibility"]["shared_extraction"] = {
            "revision": "identical-observer-request-reuse/v1", "source_version_id": reused_from,
            "request_hash": request.request_hash(), "call_record_id": receipt.call_record_id,
            "basis": "Exact model/settings/messages equality; original call and cost identity retained."}
    if ops is None:
        version = TVersion(
            entries=parent.entries if parent else (), ops_applied=(), rejected_ops=(),
            status="failed", failure=failure, **base,
        )
        return ExtractionOutput(version, (), (receipt,), text, prompt_hash(messages))
    result = apply_ops(
        parent, view, ops, run_id=run_id, update_index=update_index,
        checkpoint_cuts=checkpoint_cuts, requirement_version=checkpoint.requirement_version,
    )
    version = TVersion(
        entries=tuple(result.entries), ops_applied=tuple(result.applied),
        rejected_ops=tuple(result.rejected),
        status="valid" if not result.rejected else "valid_with_rejections", **base,
    )
    return ExtractionOutput(version, tuple(result.advisor), (receipt,), text, prompt_hash(messages))


def update_t(
    *,
    store: ArtifactStoreProtocol,
    run_id: str,
    parent: TVersion | None,
    checkpoint: CheckpointInfo,
    prior_checkpoint: CheckpointInfo | None,
    checkpoint_cuts: Mapping[str, int],
    client: ObserverClient,
    config: ExtractorConfig,
    scope: SourceScope | None = None,
    reusable_call: tuple[str | None, CallReceipt, str] | None = None,
    context_counter: RequestTokenCounter | None = None,
) -> ExtractionOutput:
    """One primary incremental step. Reads only (prior cut, this cut]."""
    after = prior_checkpoint.event_cut_seq if prior_checkpoint else -1
    if config.validation_revision != VALIDATION_REVISION:
        raise ValueError("historical validation config supports read-only reuse; start a named chain for new extraction")
    candidate = TVersion(
        role="t_retention", run_id=run_id, target_checkpoint_id=checkpoint.checkpoint_id,
        parent_version_id=parent.version_id() if parent else None,
        update_index=parent.update_index + 1 if parent else 0,
        after_seq=after, through_seq=checkpoint.event_cut_seq, entries=(), ops_applied=(),
        rejected_ops=(), status="valid", extractor_config_id=config.config_id(), call_record_ids=(),
        source_visibility={"run_id": run_id, "scope": (scope or SourceScope()).as_record(),
                           "target_session_id": checkpoint.session_id,
                           "interval": {"after_seq": after, "through_seq": checkpoint.event_cut_seq}},
    )
    assert_lineage_step(candidate, run_id=run_id, parent=parent, checkpoint=checkpoint,
                        prior_checkpoint=prior_checkpoint, config=config, scope=scope)
    # A failed/missing parent still chains: it carries the last valid content and
    # the gap stays visible in the lineage. Only its cut must line up.
    if parent is not None and parent.through_seq != after:
        raise ValueError("parent T cut does not match prior checkpoint cut")
    if parent is not None and prior_checkpoint is not None:
        parent_view = view_for_checkpoint(store, run_id, through_seq=parent.through_seq,
                                          target_session_id=prior_checkpoint.session_id, scope=scope)
        problems = validate_version(parent, parent_view)
        if problems:
            raise ValueError("invalid parent T: " + "; ".join(problems))
    view = view_for_checkpoint(
        store, run_id, through_seq=checkpoint.event_cut_seq, after_seq=after,
        target_session_id=checkpoint.session_id, scope=scope,
    )
    prior_record = None if parent is None else {"entries": [e.as_record() for e in parent.entries]}
    messages = t_update_messages(prior_record, view.events(), (after, checkpoint.event_cut_seq))
    return _run(
        store=store, role="t_retention", parent=parent, view=view, messages=messages, client=client,
        config=config, cost_role="extraction", purpose="t_update", run_id=run_id,
        checkpoint=checkpoint, update_index=(parent.update_index + 1) if parent else 0,
        checkpoint_cuts=checkpoint_cuts,
        reusable_call=reusable_call,
        context_counter=context_counter,
    )


def extract_full_prefix(
    *,
    store: ArtifactStoreProtocol,
    run_id: str,
    checkpoint: CheckpointInfo,
    checkpoint_cuts: Mapping[str, int],
    client: ObserverClient,
    config: ExtractorConfig,
    scope: SourceScope | None = None,
    repeat_index: int = 0,
    context_counter: RequestTokenCounter | None = None,
) -> ExtractionOutput:
    """Independent reference extraction from the full allowed prefix, no prior T."""
    view = view_for_checkpoint(
        store, run_id, through_seq=checkpoint.event_cut_seq, after_seq=-1,
        target_session_id=checkpoint.session_id, scope=scope,
    )
    messages = t_full_messages(view.events(), checkpoint.event_cut_seq)
    out = _run(
        store=store, role="t_audit_fullprefix", parent=None, view=view, messages=messages, client=client,
        config=config, cost_role="audit", purpose="t_full_prefix", run_id=run_id,
        checkpoint=checkpoint, update_index=0, checkpoint_cuts=checkpoint_cuts,
        context_counter=context_counter,
    )
    version = replace(out.version, notes=out.version.notes + (f"repeat_index={repeat_index}",))
    return ExtractionOutput(version, out.advisor_ops, out.receipts, out.raw_response, out.prompt_hash)


def chain_lineage(versions: Sequence[TVersion]) -> list[dict[str, Any]]:
    """Lineage summary for reporting standalone policy cost and gaps."""
    return [
        {
            "update_index": v.update_index,
            "checkpoint_id": v.target_checkpoint_id,
            "version_id": v.version_id(),
            "parent_version_id": v.parent_version_id,
            "content_hash": v.content_hash(),
            "status": v.status,
            "entries": len(v.entries),
            "rejected_ops": len(v.rejected_ops),
            "calls": list(v.call_record_ids),
        }
        for v in versions
    ]
