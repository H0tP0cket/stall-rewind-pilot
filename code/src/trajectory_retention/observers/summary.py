"""Independent prefix-only summary with immutable attempts and measured length.

Only the permitted transcript and numeric length target enter generation.
Typed-T content, private scores and recovery outcomes never enter its prompts.
Prompt/encoding/tolerance calibration remains separate from this mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..contracts import ArtifactRef, ArtifactStoreProtocol, from_jsonable, new_id, to_jsonable
from ._canonical import content_id, sha256_hex
from .client import CallReceipt, ModelRequest, ObserverClient
from .context import RequestTokenCounter, context_preflight
from .extractor import CheckpointInfo, ExtractorConfig
from .prefix import SourceScope, view_for_checkpoint
from .prompts import prompt_hash, render_events, summary_messages
from .render import approx_tokens
from .tokens import TokenCounter, TokenMatchConfig, byte_proxy_counter, compare_measurements

SUMMARY_REVISION = "summary/v2-2026-09-15-token-measurement"
SUMMARY_PROMPT_REVISION = "summary-independent/dev-2026-09-15-v2-length-instruction"


@dataclass(frozen=True)
class SummaryArtifact:
    run_id: str
    target_checkpoint_id: str
    text: str | None
    status: str
    matched_token_target: int
    max_words: int
    source_visibility: Mapping[str, Any]
    receipts: tuple[CallReceipt, ...]
    config_id: str
    failure: str | None = None
    raw_responses: tuple[str | None, ...] = ()
    attempts: tuple[Mapping[str, Any], ...] = ()
    token_measurement: Mapping[str, Any] | None = None
    matching_report: Mapping[str, Any] = field(default_factory=dict)
    generation_id: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "role": "summary_independent", "revision": SUMMARY_REVISION, "run_id": self.run_id,
            "target_checkpoint_id": self.target_checkpoint_id, "text": self.text,
            "text_sha256": None if self.text is None else sha256_hex(self.text),
            "approx_token_count": None if self.text is None else approx_tokens(self.text),
            "status": self.status, "matched_token_target": self.matched_token_target,
            "token_matching_status": self.matching_report.get("status", "not_configured"),
            "max_words": self.max_words, "source_visibility": dict(self.source_visibility),
            "call_record_ids": [r.call_record_id for r in self.receipts], "config_id": self.config_id,
            "failure": self.failure, "generation_id": self.generation_id,
            "attempts": [dict(a) for a in self.attempts], "token_measurement": self.token_measurement,
            "matching_report": dict(self.matching_report),
        }

    def artifact_id(self) -> str:
        return content_id(self.as_record())


def words_for_tokens(token_target: int) -> int:
    # Legacy English prior retained as a development prompt setting, not proof
    # of token matching. Callers can explicitly override max_words.
    return max(50, int(token_target * 0.75))


def summary_prompt_binding(max_words: int) -> dict[str, Any]:
    return {"revision": SUMMARY_PROMPT_REVISION,
            "base_template_hash": prompt_hash(summary_messages([], 0, max_words)),
            "length_instruction_revision": "numeric-target-and-tolerance.v1", "max_words": max_words}


def generate_summary(
    *, store: ArtifactStoreProtocol, run_id: str, checkpoint: CheckpointInfo,
    matched_token_target: int, client: ObserverClient, config: ExtractorConfig,
    scope: SourceScope | None = None, token_counter: TokenCounter | None = None,
    matching: TokenMatchConfig | None = None, target_measurement: Mapping[str, Any] | None = None,
    generation_id: str | None = None,
    context_counter: RequestTokenCounter | None = None,
) -> SummaryArtifact:
    from .artifacts import record_calls  # Avoid module initialization cycle.

    if isinstance(matched_token_target, bool) or not isinstance(matched_token_target, int) or matched_token_target < 0:
        raise ValueError("matched_token_target must be a nonnegative integer")
    counter, matching = token_counter or byte_proxy_counter(config.model), matching or TokenMatchConfig()
    if counter.model != config.model:
        raise ValueError("summary token counter must bind the summary/worker model")
    if matching.configured and target_measurement is None:
        raise ValueError("matching requires a measurement of the exact frozen typed packet")
    if target_measurement is not None:
        if target_measurement["token_count"] != matched_token_target:
            raise ValueError("target count disagrees with the frozen typed measurement")
        compare_measurements(target_measurement, counter.measure(""), matching)
    view = view_for_checkpoint(store, run_id, through_seq=checkpoint.event_cut_seq, after_seq=-1,
                               target_session_id=checkpoint.session_id, scope=scope)
    events = view.events()
    max_words = matching.max_words or words_for_tokens(matched_token_target)
    base_messages = summary_messages(events, checkpoint.event_cut_seq, max_words)
    visibility = {**view.visibility_manifest(), "source_text_sha256": sha256_hex(render_events(events)),
                  "prompt_revision": SUMMARY_PROMPT_REVISION}
    binding = {"revision": SUMMARY_REVISION, "run_id": run_id, "checkpoint": to_jsonable(checkpoint),
               "config_id": config.config_id(), "source_visibility": visibility,
               "base_prompt_hash": prompt_hash(base_messages), "prompt": summary_prompt_binding(max_words),
               "counter": counter.as_record(), "matching": matching.as_record(),
               "target_measurement": target_measurement, "matched_token_target": matched_token_target}
    if config.context_policy is not None:
        binding["context_counter"] = None if context_counter is None else context_counter.as_record()
    generation_id = generation_id or content_id(binding)
    try:
        previous = store.get_record("summary_generation_v2", generation_id)
    except KeyError:
        store.put_record("summary_generation_v2", generation_id, binding, run_id=run_id)
    else:
        if previous != binding:
            raise ValueError("frozen summary generation configuration changed; use an explicit new generation ID")

    receipts, raw_responses, attempts = [], [], []
    chosen_text = chosen_measurement = None
    report: dict[str, Any] = {"status": "not_configured", "within_tolerance": None,
                              "matching_config": matching.as_record(), "provider_exact": False}
    failure = "no valid summary response"
    context_preflights = []
    context_blocked = False
    for index in range(matching.max_attempts):
        messages = base_messages
        if matching.configured:
            allowance = matching.allowance(matched_token_target)
            instruction = (f"\nLength-control attempt {index + 1} of {matching.max_attempts}: aim for "
                           f"{matched_token_target} tokens measured with {counter.tokenizer_id}. "
                           f"The allowed absolute difference is {allowance} tokens. This is a length "
                           "constraint only; preserve the transcript's uncertainty and do not add facts.")
            messages = ({**base_messages[0], "content": base_messages[0]["content"] + instruction},
                        base_messages[1])
        request = ModelRequest(model=config.model, messages=messages, settings=config.settings,
                               cost_role="summary", purpose="summary_independent", run_id=run_id,
                               target_checkpoint_id=checkpoint.checkpoint_id)
        if config.context_policy is not None:
            guard = context_preflight(store, request, config.context_policy, token_counter=context_counter)
            context_preflights.append(guard)
            if not guard["client_call_allowed"]:
                context_blocked = True
                failure = "context preflight: " + guard["reason"]
                break
        attempt_id = f"{generation_id}:attempt:{index + 1}"
        request_body = {"generation_id": generation_id, "attempt_index": index + 1,
                        "request": to_jsonable(request), "request_hash": request.request_hash()}
        try:
            completed = store.get_record("summary_attempt_v2", attempt_id)
        except KeyError:
            try:
                store.get_record("summary_attempt_started_v2", attempt_id)
            except KeyError:
                # Distinct claim bytes make concurrent missing->write racers
                # conflict atomically in the immutable store before either can
                # issue a duplicate request. Sequential uncertain retries also
                # fail closed above; this does not assume exactly-once APIs.
                store.put_record("summary_attempt_started_v2", attempt_id,
                                 {**request_body, "claim_id": new_id("summary_claim")}, run_id=run_id)
            else:
                raise ValueError("summary attempt has no completion receipt; refusing an unrecorded retry")
            raw_text, receipt = client.complete(request)
            # Persist each returned attempt before another request. The live
            # provider boundary separately retains wire responses and charges.
            record_calls(store, run_id, (receipt,), raw_text)
            raw_ref = None if raw_text is None else store.put_bytes(raw_text.encode("utf-8"), media_type="text/plain")
            measured = (counter.measure(raw_text, context={"generation_id": generation_id, "attempt_index": index + 1})
                        if receipt.status == "ok" and raw_text is not None and raw_text.strip() else None)
            completed = {**request_body, "receipt": receipt.as_record(), "raw_response_ref": to_jsonable(raw_ref),
                         "token_measurement": measured}
            store.put_record("summary_attempt_v2", attempt_id, completed, run_id=run_id)
        else:
            if any(completed[k] != v for k, v in request_body.items()):
                raise ValueError("cached summary attempt request binding changed")
            receipt = from_jsonable(CallReceipt, completed["receipt"])
            raw_ref = completed["raw_response_ref"]
            raw_text = None if raw_ref is None else store.get_bytes(from_jsonable(ArtifactRef, raw_ref)).decode("utf-8")
        receipts.append(receipt)
        raw_responses.append(raw_text)
        if (receipt.request_hash != request.request_hash() or receipt.requested_model != request.model
                or receipt.cost_role != request.cost_role):
            raise ValueError("summary receipt does not bind the requested model, role and prompt")
        ok = receipt.status == "ok" and raw_text is not None and bool(raw_text.strip())
        measurement = counter.measure(raw_text, context={"generation_id": generation_id,
                                      "attempt_index": index + 1}) if ok else None
        if measurement != completed["token_measurement"]:
            raise ValueError("cached summary tokenizer behavior changed despite unchanged revision")
        attempt_report = (compare_measurements(target_measurement, measurement, matching)
                          if target_measurement is not None and measurement is not None else None)
        attempts.append({"attempt_id": attempt_id, "call_record_id": receipt.call_record_id,
                         "request_hash": request.request_hash(), "prompt_hash": prompt_hash(messages),
                         "raw_response_ref": completed["raw_response_ref"], "status": receipt.status,
                         "usable_text": ok, "token_measurement": measurement, "matching_report": attempt_report})
        if ok:
            chosen_text, chosen_measurement = raw_text, measurement
            if attempt_report is not None:
                report = attempt_report
            failure = None
            if not matching.configured or report["within_tolerance"]:
                break
        else:
            failure = receipt.error or "empty or absent summary text"
    status = "failed" if chosen_text is None else "valid"
    if context_blocked:
        status = "failed"
        report = {**report, "status": "context_preflight_blocked", "within_tolerance": False,
                  "blocked_preflight_id": context_preflights[-1]["preflight_id"]}
    elif matching.configured and report.get("within_tolerance") is not True:
        status = "unmatched" if chosen_text is not None else "failed"
        failure = "bounded attempts exhausted without a token match" if chosen_text is not None else failure
        if chosen_text is None:
            report = {**report, "status": "no_valid_candidate"}
    report = {**report, "attempts_executed": len(attempts), "selected_attempt_id": next(
        (a["attempt_id"] for a in reversed(attempts) if a["token_measurement"] == chosen_measurement), None)
        if chosen_measurement is not None else None,
        "delivery_eligible": status == "valid", "scientific_matching_calibrated": False}
    return SummaryArtifact(run_id=run_id, target_checkpoint_id=checkpoint.checkpoint_id,
                           text=chosen_text, status=status, matched_token_target=matched_token_target,
                           max_words=max_words, source_visibility=({**visibility, "context_preflights": context_preflights}
                           if config.context_policy is not None else visibility), receipts=tuple(receipts),
                           config_id=config.config_id(), failure=failure, raw_responses=tuple(raw_responses),
                           attempts=tuple(attempts), token_measurement=chosen_measurement,
                           matching_report=report, generation_id=generation_id)
