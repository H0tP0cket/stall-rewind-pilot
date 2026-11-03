"""Auditable pre-submission recovery of an interrupted natural-v2 execution.

Separate physical execution, unchanged worker protocol, last restorable checkpoint,
remaining original guard. Not an independent mine or an experimental arm outcome.
The distinct policy prevents existing natural-source collectors from silently admitting it.
"""
from __future__ import annotations

from ..accounting import CostLedger
from ..contracts import RunConfig, from_jsonable, to_jsonable

POLICY = "technical-resume-v2/v1"


def prepare(store, source_run, run_id, *, prior_attempts=(), capture_gap_policy=None, conditions=None):
    """conditions: optional current run rules (submission_feedback, submission_stop, unresolved_retry) bound in
    place of the source's missing ones; only for sources that never submitted (user 2026-11-02)."""
    events = list(store.iter_events(source_run))
    start = next(e.payload for e in events if e.kind == "lifecycle" and
                 e.payload.get("transition") == "run_started")
    old = start["config"]
    # A technical continuation that was itself interrupted (user 2026-11-02: host disk full) is resumed from its own
    # latest point; its config already holds the remaining original guard, so the same arithmetic applies.
    chained = old["metadata"].get("v2_run_policy") == POLICY
    if start.get("treatment") or not (chained or (not start.get("resume_from") and
                                                  old["metadata"].get("v2_run_policy") == "natural-v2/v1")):
        raise ValueError("technical resume supports original natural-v2 sources and technical continuations only")
    if chained and prior_attempts:
        raise ValueError("a chained resume charges only the continuation it resumes")
    terminal = next((e for e in reversed(events) if e.kind == "terminal"), None)
    if terminal is None or terminal.payload.get("status") not in ("technical_failure", "interrupted", "accounting_interrupted"):
        raise ValueError("source must have a recorded technical interruption, not a completed or live run")
    if any(e.kind == "native_submission_feedback" or (e.kind == "lifecycle" and
           e.payload.get("transition") == "native_submission_verdict") for e in events):
        raise ValueError("post-submission resumes need counter-preserving support; this path refuses them")
    points = [c for c in store.list_checkpoints(source_run) if c.branchable]
    if not points:
        raise ValueError("source has no restorable checkpoint")
    cp = max(points, key=lambda c: c.ordinal)
    # not-sent-zero-cost-retry.v1 (user 2026-11-02), applied retroactively when current conditions are bound: a
    # recorded attempt whose connection never opened is a known $0 attempt and its worst-case charge is released.
    from ..provider_policy import NOT_SENT_RULE, recorded_never_sent
    not_sent = {e.event_id for e in events if conditions is not None and e.kind == "model_attempt"
                and e.payload.get("phase") == "finished" and e.payload.get("usage_known") is False
                and recorded_never_sent(e.payload)}
    records = [r for r in CostLedger(store).records(source_run) if r.kind == "model" and r.role == "worker"
               and r.source_event_id not in not_sent]
    known = sum(r.usage.cost_usd for r in records if r.usage.cost_usd is not None)
    unknown = sum(r.usage.cost_usd is None for r in records)
    charges = [e.payload["worst_case_charge"]["usd"] for e in events if e.kind == "model_attempt"
               and e.payload.get("phase") == "finished" and e.payload.get("worst_case_charge")
               and e.event_id not in not_sent]
    released = [{"event_id": e.event_id, "error": e.payload.get("error", "")[:200],
                 "released_usd": (e.payload.get("worst_case_charge") or {}).get("usd", 0.0)}
                for e in events if e.event_id in not_sent]
    retroactive = []
    if unknown != len(charges) and conditions is not None and conditions.get("unresolved_retry"):
        # Sources from before the charged-retry rule stopped on their unresolved attempt without a guard charge.
        # Charge each such attempt now with the same bound the live runner uses, from its recorded request.
        import json
        from ..provider_policy import worst_case_charge
        for e in events:
            if (e.kind == "model_attempt" and e.payload.get("phase") == "finished" and e.event_id not in not_sent and
                    e.payload.get("usage_known") is False and not e.payload.get("worst_case_charge")):
                body = store.get_bytes(e.input_ref)
                charge = worst_case_charge(conditions["unresolved_retry"], json.loads(body), body)
                retroactive.append({"event_id": e.event_id, **charge})
        charges += [c["usd"] for c in retroactive]
    if unknown != len(charges):
        raise ValueError("unresolved source costs do not have matching guard charges")
    remaining = old["cost_limit_usd"] - known - sum(charges)
    if remaining <= 0:
        raise ValueError("original cost guard has no remaining allowance")
    calls = sum(e.kind == "model_response" for e in events)
    counters = terminal.payload.get("counters", {})
    tokens = counters.get("known_generated_tokens_this_run", 0) if chained else counters.get("known_generated_tokens", 0)
    retry_receipts = []
    seen = {source_run}
    for attempt in prior_attempts:
        from .branch_store import ReadOnlyArtifactStore
        prior = ReadOnlyArtifactStore(attempt["store"])
        prior_id = attempt["run_id"]
        if prior_id in seen:
            raise ValueError("duplicate technical-attempt charge")
        seen.add(prior_id)
        prior_events = list(prior.iter_events(prior_id))
        prior_start = next(e.payload for e in prior_events if e.kind == "lifecycle" and
                           e.payload.get("transition") == "run_started")
        prior_config = prior_start["config"]
        prior_spec = prior_config["metadata"].get("technical_resume", {})
        if (prior_config["metadata"].get("v2_run_policy") != POLICY or
                prior_spec.get("source_run_id") != source_run or prior_start.get("treatment") or
                prior_start["task"] != start["task"] or prior_config["model"] != old["model"] or
                prior_config["model_settings"] != old["model_settings"]):
            raise ValueError("prior attempt is not a technical continuation of this exact source")
        prior_terminal = next((e for e in reversed(prior_events) if e.kind == "terminal"), None)
        if prior_terminal is None or prior_terminal.payload.get("status") not in (
                "technical_failure", "interrupted", "accounting_interrupted"):
            raise ValueError("prior attempt must be a recorded technical interruption")
        if any(e.kind == "native_submission_feedback" or (e.kind == "lifecycle" and
               e.payload.get("transition") == "native_submission_verdict") for e in prior_events):
            raise ValueError("post-submission retries require counter-preserving support")
        earlier = {a["run_id"] for a in prior_spec.get("prior_attempts", [])}
        if not earlier <= {a["run_id"] for a in prior_attempts}:
            raise ValueError("retry omits an earlier charged continuation")
        records2 = [r for r in CostLedger(prior).records(prior_id) if r.kind == "model" and r.role == "worker"]
        known2 = sum(r.usage.cost_usd for r in records2 if r.usage.cost_usd is not None)
        unknown2 = sum(r.usage.cost_usd is None for r in records2)
        charges2 = [e.payload["worst_case_charge"]["usd"] for e in prior_events if e.kind == "model_attempt"
                    and e.payload.get("phase") == "finished" and e.payload.get("worst_case_charge")]
        if unknown2 != len(charges2):
            raise ValueError("unresolved prior-attempt costs lack matching guard charges")
        remaining -= known2 + sum(charges2)
        calls2 = sum(e.kind == "model_response" for e in prior_events)
        tokens2 = prior_terminal.payload.get("counters", {}).get("known_generated_tokens_this_run", 0)
        calls += calls2
        tokens += tokens2
        retry_receipts.append({"store": str(prior.root), "run_id": prior_id,
            "terminal_event_id": prior_terminal.event_id, "known_cost_usd": known2,
            "unknown_cost_attempts": unknown2, "worst_case_charged_usd": sum(charges2),
            "model_calls": calls2, "known_generated_tokens": tokens2})
    if remaining <= 0:
        raise ValueError("original cost guard exhausted after prior attempts")
    if capture_gap_policy is not None:
        from .capture_gaps import POLICY as GAP_POLICY
        if capture_gap_policy != GAP_POLICY:
            raise ValueError("unrecognized capture-gap policy")
    if conditions is not None:
        if set(conditions) != {"submission_feedback", "submission_stop", "unresolved_retry"}:
            raise ValueError("current conditions must set submission_feedback, submission_stop and unresolved_retry")
        clash = [k for k in conditions if k in old["metadata"] and old["metadata"][k] != conditions[k]]
        if clash:
            raise ValueError(f"source already binds different {clash}; refusing to replace its rules")
    spec = {"version": POLICY, "source_run_id": source_run, "checkpoint_id": cp.checkpoint_id,
            "source_terminal_event_id": terminal.event_id, "source_known_cost_usd": known,
            "source_unknown_cost_attempts": unknown, "source_worst_case_charged_usd": sum(charges),
            "original_cost_guard_usd": old["cost_limit_usd"], "remaining_cost_guard_usd": remaining,
            "source_total_calls": calls, "restored_step": cp.validation.get("step_index"),
            "protocol": ("current conditions bound (user 2026-11-02): " + ", ".join(sorted(conditions)) +
                         "; source never submitted, so no earlier feedback is in its context; no review, hand-back "
                         "or new user message") if conditions else
                        "unchanged source protocol; no review, hand-back or new user message",
            "recording": "every new step; original source store read-only; detector record-only clock restarts",
            "analysis_status": "technical continuation; not an independent mine or arm outcome; not automatically admitted"}
    if chained:
        spec["chained_from"] = {"run_id": source_run, "technical_resume": old["metadata"]["technical_resume"],
                                "rule": "technical-resume-chain/v1: resume an interrupted technical continuation from "
                                        "its own latest restorable point under its remaining guard"}
    if retry_receipts:
        spec["retry_accounting_version"] = "technical-retry-accounting/v1"
        spec["prior_attempts"] = retry_receipts
    if capture_gap_policy is not None:
        spec["capture_gap_policy"] = capture_gap_policy
    if conditions is not None:
        spec["conditions"] = conditions
    if retroactive:
        spec["retroactive_worst_case_charges"] = retroactive
    if released:
        spec["not_sent_attempts"] = {"rule": NOT_SENT_RULE, "attempts": released}
    values = {**old, "run_id": run_id, "campaign_id": "technical-resume-v2",
              "cost_limit_usd": remaining, "max_model_calls": old["max_model_calls"] - calls,
              "generated_token_limit": old["generated_token_limit"] - tokens,
              "metadata": {**old["metadata"], "v2_run_policy": POLICY,
                           "execution_role": "technical_continuation", "technical_resume": spec,
                           **({"capture_gap_policy": capture_gap_policy} if capture_gap_policy else {}),
                           **(conditions or {})}}
    if values["max_model_calls"] <= 0 or values["generated_token_limit"] <= 0:
        raise ValueError("original call/token guard exhausted")
    return cp, from_jsonable(RunConfig, values), spec, start


def validate(store, config, resume, treatment):
    if resume is None or treatment is not None:
        raise ValueError("technical resume requires one aligned checkpoint and no treatment")
    spec = config.metadata.get("technical_resume", {})
    cp, expected, _, _ = prepare(store, resume.run_id, config.run_id,
        prior_attempts=spec.get("prior_attempts", []), capture_gap_policy=config.metadata.get("capture_gap_policy"),
        conditions=spec.get("conditions"))
    if resume != cp or to_jsonable(config) != to_jsonable(expected):
        raise ValueError("technical resume must use the latest restorable point and exact derived configuration")


def chain_source_ok(start, lookup, run_metadata) -> bool:
    """technical-resume-chain/v1: a technical continuation may resume another technical continuation whose chain of
    sources ends in a natural run; every other recovery keeps the natural-source rule."""
    from ..resample_source import recovery_source_ok
    if (run_metadata or {}).get("v2_run_policy") != POLICY:
        return False
    seen = set()
    while start and start["config"]["metadata"].get("v2_run_policy") == POLICY:
        source_id = start["config"]["metadata"]["technical_resume"]["source_run_id"]
        if source_id in seen:
            return False
        seen.add(source_id)
        start = lookup(source_id)
    return bool(start) and recovery_source_ok(start, lookup)

