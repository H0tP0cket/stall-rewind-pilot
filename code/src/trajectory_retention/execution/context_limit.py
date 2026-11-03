"""The context-limit rule (user, 2026-10-16): when the next request won't fit, hide the oldest tool outputs, the same way
in every arm, and log it; review views are built from the full record.

Before every worker request (natural runs, every branch arm, and the one-shot review and note turns) the agent measures
the request it is about to send. If it does not fit the worker's context window, the oldest tool outputs are replaced
by the placeholder one at a time, oldest first, until it fits. System and task messages, the agent's own messages and
reasoning, submission feedback, format-error messages and the hand-back are never touched; only tool outputs are. An
output already showing the placeholder (an arm's own drop) is not counted again. If the request still does not fit with
every tool output hidden, the run ends with the end reason ``context_window_exhausted`` and nothing is sent.

Size of a request (SIZE_RULE): the prompt as the provider would count it plus the request's response cap (its
``max_tokens``, the remaining allowance capped at the ordinary response cap, as ``_BaseModel.prepare_request`` sets it).
The provider's count is not known before sending, so the prompt is measured locally (MEASURE_RULE: tiktoken o200k_base
over the JSON of every message of the actual serialized request plus the JSON of the rest of the request) and scaled by
the provider-to-local ratio of the latest request of the same context whose provider prompt count is known (the
anchor): provider prompt tokens of that request over its local measure. Without an anchor the ratio is 1. The local
measure is additive over messages by definition, so the saving of hiding one output is exact under it.

The hidden state is part of the context channel: a hidden output is the placeholder in the context itself, marked in
the message's harness-only ``extra`` with a reference to the original delivered text, so every later checkpoint carries
the masked view and a branch restored to such a point resumes with the same masked view. Accounting is untouched.
Views built for review (the Ledger, the arm 3 context view, the recipe) read the original text back through
``unmasked_messages`` (the full record).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

RULE_VERSION = "context-limit/v1"
MASK_KEY = "context_limit_mask"  # in a hidden tool message's extra
COUNTER_ID = "tiktoken:o200k_base:per-message-json/v1"
MEASURE_RULE = ("local measure: tiktoken o200k_base tokens of json.dumps(message, sort_keys=True) for every message of "
                "the actual serialized request, plus those of json.dumps(the request without its messages, "
                "sort_keys=True); additive over messages and independent of key order")
SIZE_RULE = ("request size = ceil(local measure x anchor ratio) + the request's response cap (max_tokens or "
             "max_completion_tokens); anchor ratio = provider prompt tokens / local measure of the latest request of "
             "this context with a provider prompt count (1 without one); fits when size <= window_tokens")
RULE = ("context-limit/v1: before a worker request whose size (SIZE_RULE) exceeds the window, replace the oldest tool "
        "outputs, one at a time from the oldest, with OUTPUT_PLACEHOLDER until it fits; system and task messages, the "
        "agent's own messages and reasoning and every non-tool message are never touched; outputs already showing the "
        "placeholder are skipped; if it does not fit with every tool output hidden, nothing is sent and the run ends "
        "with end reason context_window_exhausted; the same rule in natural runs, every branch arm and the review and "
        "note turns; hidden outputs stay hidden in the context (checkpoints carry them); accounting is untouched; "
        "review views read the original outputs from the record")
END_REASON = "context_window_exhausted"
EXIT_STATUS = "ContextWindowExhausted"


_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return _ENC


@dataclass
class ContextLimit:
    """One run's binding of the rule: the window and the placeholder, plus a per-message token cache."""
    window_tokens: int
    placeholder: str
    source: str = ""
    _cache: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.window_tokens) is not int or self.window_tokens <= 0:
            raise ValueError("context window must be a positive integer number of tokens")
        if not isinstance(self.placeholder, str) or not self.placeholder:
            raise ValueError("context-limit placeholder must be a non-empty string")

    @classmethod
    def from_metadata(cls, spec: Any, placeholder: str) -> "ContextLimit | None":
        """RunConfig.metadata["context_limit"] -> the binding (None: the rule is not bound for this run)."""
        if spec is None:
            return None
        if not isinstance(spec, Mapping) or spec.get("rule") != RULE_VERSION or spec.get("window_tokens") is None:
            raise ValueError(f"context_limit must be {{rule: {RULE_VERSION}, window_tokens: int, ...}}")
        return cls(window_tokens=spec["window_tokens"], placeholder=placeholder, source=str(spec.get("source") or ""))

    def binding(self) -> dict[str, Any]:
        return {"rule": RULE_VERSION, "window_tokens": self.window_tokens, "placeholder": self.placeholder,
                "counter": COUNTER_ID, "measure_rule": MEASURE_RULE, "size_rule": SIZE_RULE, "text": RULE,
                "source": self.source}

    # ---------------------------------------------------------------- measure
    def tokens(self, text: str) -> int:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        got = self._cache.get(key)
        if got is None:
            got = self._cache[key] = len(_encoder().encode_ordinary(text))
        return got

    def message_tokens(self, message: Mapping[str, Any]) -> int:
        # sorted keys: a restored context (stored as canonical JSON) measures exactly as the live one
        return self.tokens(json.dumps(message, sort_keys=True))

    def measure(self, request: Mapping[str, Any]) -> dict[str, int]:
        """Local measure of one serialized request (MEASURE_RULE) and its response cap."""
        rest = {k: v for k, v in request.items() if k != "messages"}
        per = [self.message_tokens(m) for m in request["messages"]]
        cap = request.get("max_tokens", request.get("max_completion_tokens")) or 0
        return {"local_tokens": sum(per) + self.tokens(json.dumps(rest, sort_keys=True)), "response_cap": int(cap),
                "messages": len(per)}

    def size(self, local_tokens: int, response_cap: int, ratio: float) -> int:
        return math.ceil(local_tokens * ratio) + response_cap


def is_masked(message: Mapping[str, Any]) -> bool:
    return MASK_KEY in (message.get("extra") or {})


def candidates(messages: list[dict], placeholder: str) -> list[int]:
    """Tool outputs that can still be hidden, oldest first."""
    return [i for i, m in enumerate(messages) if m.get("role") == "tool" and isinstance(m.get("content"), str)
            and m["content"] != placeholder]


def fit(messages: list[dict], limit: ContextLimit, serialize: Callable[[list[dict]], bytes], *, ratio: float,
        put_original: Callable[[str], Any], step: int, context_steps: Callable[[list[dict]], list[list[int]]]
        ) -> dict[str, Any]:
    """Apply the rule to the context the agent is about to send.

    Returns {"messages": the context to send (the input list when nothing is hidden), "fits", "masked": [...],
    "before": measure, "after": measure, "body": the serialized request of the returned context}. Hidden messages are
    new dicts (the originals are not mutated); ``put_original`` stores a hidden output's delivered text and returns
    its reference."""
    body = serialize(messages)
    request = json.loads(body)
    before = limit.measure(request)
    out = {"ratio": ratio, "window_tokens": limit.window_tokens, "before": before,
           "size_before": limit.size(before["local_tokens"], before["response_cap"], ratio)}
    if out["size_before"] <= limit.window_tokens:
        return {**out, "messages": messages, "fits": True, "masked": [], "after": before,
                "size_after": out["size_before"], "body": body}
    work = list(messages)
    local = before["local_tokens"]
    placeholder_tokens = None
    chosen: list[int] = []
    fits = False
    for i in candidates(messages, limit.placeholder):
        sent = request["messages"][i]  # the message as serialized (no extra)
        masked_sent = dict(sent, content=limit.placeholder)
        saving = limit.message_tokens(sent) - limit.message_tokens(masked_sent)
        placeholder_tokens = limit.message_tokens(masked_sent)
        local -= saving
        chosen.append(i)
        if limit.size(local, before["response_cap"], ratio) <= limit.window_tokens:
            fits = True
            break
    steps = context_steps(messages)
    step_of = {i: k for k, idx in enumerate(steps) for i in idx}
    masked = []
    for i in chosen:
        original = messages[i]["content"]
        ref = put_original(original)
        record = {"rule": RULE_VERSION, "hidden_at_step": step, "context_step": step_of.get(i),
                  "tool_call_id": messages[i].get("tool_call_id"),
                  "original_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
                  "original_chars": len(original), "original_ref": ref}
        work[i] = dict(messages[i], content=limit.placeholder,
                       extra={**(messages[i].get("extra") or {}), MASK_KEY: record})
        masked.append({"message_index": i, **{k: v for k, v in record.items() if k != "rule"},
                       "original_local_tokens": limit.message_tokens(request["messages"][i]),
                       "placeholder_local_tokens": placeholder_tokens})
    body_after = serialize(work)
    after = limit.measure(json.loads(body_after))
    size_after = limit.size(after["local_tokens"], after["response_cap"], ratio)
    if after["local_tokens"] != local:  # the additive measure must agree with the re-serialized request
        raise RuntimeError("context-limit measure is not additive over the serialized request")
    if fits != (size_after <= limit.window_tokens):
        raise RuntimeError("context-limit fit decision disagrees with the re-serialized request")
    return {**out, "messages": work, "fits": fits, "masked": masked, "after": after, "size_after": size_after,
            "body": body_after, "outputs_hidden_total": sum(1 for m in work if is_masked(m))}


def unmasked_messages(messages: list[dict], store: Any) -> tuple[list[dict], list[int]]:
    """The full record of a saved context: every output hidden by this rule gets its original delivered text back.

    Returns (messages, indices restored). The original is checked against its recorded sha256."""
    from ..contracts import ArtifactRef, from_jsonable
    out, restored = [], []
    for i, m in enumerate(messages):
        record = (m.get("extra") or {}).get(MASK_KEY)
        if record is None:
            out.append(m)
            continue
        ref = record["original_ref"]
        text = store.get_bytes(from_jsonable(ArtifactRef, ref) if isinstance(ref, dict) else ref).decode("utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["original_sha256"]:
            raise ValueError(f"hidden output at message {i} does not match its recorded original")
        extra = {k: v for k, v in (m.get("extra") or {}).items() if k != MASK_KEY}
        out.append(dict(m, content=text, extra=extra))
        restored.append(i)
    return out, restored
