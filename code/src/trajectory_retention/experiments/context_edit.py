"""Edits to the context (H) for the `edited` history policy: drop steps, drop outputs, drop an exact quote.

A step is one model response with everything delivered after it until the next model response: the assistant message
(or the format-error message that replaced a rejected response), its tool outputs and any submission feedback or other
harness message. Step 0 is the system and task messages; it is never edited. Step numbers are the source run's.

Edits change the context only, never files, and never the accounting: counters keep the source values, while the
cursor, message count and O (tool-output references) are rebuilt from the edited messages by the restore path.

- drop_steps {first, last}: every message of steps first..last is removed.
- drop_outputs {steps}: every tool output of those steps becomes OUTPUT_PLACEHOLDER; the commands stay.
- drop_quote {step, quote}: every occurrence of the exact text is removed from every text field of that step where it
  appears: content, reasoning, reasoning_content, reasoning_details text/summary (the duplicated copies of the
  returned reasoning), the command of a tool call, the tool outputs and harness messages. Commands and tool outputs are
  JSON in the context, so the quote is found in the decoded text and the matching raw characters are cut, leaving
  valid JSON that decodes to the text without the quote (checked). The scope is the interface's: one step; the same
  text in another step is not touched.

Outputs channel rule (OUTPUTS_RULE): with the context at point c and the outputs channel at point o < c, the outputs of
steps o+1..c are replaced by the placeholder; o >= c changes nothing (the context already holds no later outputs).
After edits the actual serialized next request is checked (validate_request), not only the message structure.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterator

from ..execution.model import history_replay_issue
from .v2_text import OUTPUT_PLACEHOLDER

EDITS_VERSION = "context-edits.v1"
OUTPUTS_RULE = ("outputs-rewind/v1: the outputs channel at point o below the context point c replaces the tool outputs "
                "of steps o+1..c with OUTPUT_PLACEHOLDER (commands kept); o >= c changes nothing")
OPS = {"drop_steps": {"op", "first", "last"}, "drop_outputs": {"op", "steps"}, "drop_quote": {"op", "step", "quote"}}
REASONING_FIELDS = ("reasoning", "reasoning_content")
OUTPUT_KEYS = ("output", "output_head", "output_tail", "exception_info")  # string parts of a rendered tool output


class EditError(ValueError):
    pass


def starts_step(message: dict) -> bool:
    return message.get("role") == "assistant" or (
        message.get("role") == "user" and (message.get("extra") or {}).get("interrupt_type") == "FormatError")


def steps_of(messages: list[dict]) -> list[list[int]]:
    """Message indices per step; index 0 is step 0 (system, task and anything before the first model response)."""
    steps: list[list[int]] = [[]]
    for i, message in enumerate(messages):
        if starts_step(message):
            steps.append([])
        steps[-1].append(i)
    return steps


def _text_slots(message: dict) -> Iterator[tuple[str, Any, Any]]:
    """(field path, container, key) for every plain editable text in a message (tool outputs are JSON; see below)."""
    content = message.get("content")
    if isinstance(content, str) and message.get("role") != "tool":
        yield "content", message, "content"
    elif isinstance(content, list):
        for j, part in enumerate(content):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield f"content[{j}].text", part, "text"
    for name in REASONING_FIELDS:
        if isinstance(message.get(name), str):
            yield name, message, name
    for j, detail in enumerate(message.get("reasoning_details") or []):
        if isinstance(detail, dict):
            for key in ("text", "summary"):
                if isinstance(detail.get(key), str):
                    yield f"reasoning_details[{j}].{key}", detail, key


def _decoded_output(message: dict) -> list[tuple[str, str, list[int] | None]]:
    """A tool output is a JSON object rendered into the context; (key, decoded text, decoded->raw offsets) per part."""
    raw = message.get("content")
    if message.get("role") != "tool" or not isinstance(raw, str):
        return []
    from ..ledger.text import json_string_offsets
    parts = [(key, *got) for key in OUTPUT_KEYS if (got := json_string_offsets(raw, key)) is not None]
    return parts or [("content", raw, None)]


def _decoded_commands(message: dict) -> list[tuple[int, str, list[int] | None]]:
    """(tool-call index, decoded command, decoded->raw offsets into the arguments string) per tool call."""
    from ..ledger.text import json_string_offsets
    out = []
    for j, call in enumerate(message.get("tool_calls") or []):
        raw = (call.get("function") or {}).get("arguments")
        if isinstance(raw, str):
            got = json_string_offsets(raw, "command")
            out.append((j, *got) if got is not None else (j, raw, None))
    return out


def _find_all(text: str, quote: str) -> list[int]:
    out, i = [], text.find(quote)
    while i >= 0:
        out.append(i)
        i = text.find(quote, i + len(quote))
    return out


def _arguments(message: dict) -> list[str]:
    return [text for _, text, _ in _decoded_commands(message)]


def visible_texts(message: dict) -> list[str]:
    """Every text the worker reads in a message, decoded (tool outputs and commands are JSON in the context)."""
    return ([c[k] for _, c, k in _text_slots(message)] + [t for _, t, _ in _decoded_output(message)]
            + _arguments(message))


def _cut(raw: str, spans: list[tuple[int, int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        raw = raw[:start] + raw[end:]
    return raw


def _spans(text: str, offsets: list[int] | None, quote: str) -> list[tuple[int, int]]:
    """Raw spans of every occurrence of quote in a decoded text (whole escape sequences, so the JSON stays valid)."""
    return [(offsets[i], offsets[i + len(quote)]) if offsets is not None else (i, i + len(quote))
            for i in _find_all(text, quote)]


def _remove_quote(message: dict, quote: str) -> list[dict]:
    """Remove every occurrence from every editable field of one message; returns what was removed where."""
    fields = []
    for path, container, key in _text_slots(message):
        count = container[key].count(quote)
        if count:
            container[key] = container[key].replace(quote, "")
            fields.append({"field": path, "occurrences": count})
    parts = _decoded_output(message)
    spans, expected = [], {}
    for key, text, offsets in parts:
        hits = _spans(text, offsets, quote)
        if hits:
            fields.append({"field": f"content.{key}" if offsets is not None else "content", "occurrences": len(hits)})
            spans += hits
        expected[key] = text.replace(quote, "")
    if spans:
        message["content"] = _cut(message["content"], spans)
        if {key: text for key, text, _ in _decoded_output(message)} != expected:
            raise EditError("quote removal from a tool output did not decode back to the expected text")
    for j, text, offsets in _decoded_commands(message):
        hits = _spans(text, offsets, quote)
        if not hits:
            continue
        function = message["tool_calls"][j]["function"]
        function["arguments"] = _cut(function["arguments"], hits)
        fields.append({"field": f"tool_calls[{j}].command", "occurrences": len(hits)})
        if dict((i, t) for i, t, _ in _decoded_commands(message)).get(j) != text.replace(quote, ""):
            raise EditError("quote removal from a command did not decode back to the expected text")
    return fields


def outputs_edit(outputs_point: int | None, context_point: int) -> dict | None:
    """The compiled edit for an independently chosen outputs channel (OUTPUTS_RULE)."""
    if outputs_point is None or outputs_point >= context_point:
        return None
    return {"op": "drop_outputs", "steps": list(range(outputs_point + 1, context_point + 1))}


def validate_edits(messages: list[dict], edits: list[dict]) -> list[str]:
    """Problems with an edit list against the context it applies to (empty list = valid)."""
    steps = steps_of(messages)
    last = len(steps) - 1
    problems = []
    if not isinstance(edits, list):
        return ["edits must be a list"]
    for n, e in enumerate(edits):
        op = e.get("op") if isinstance(e, dict) else None
        if op not in OPS or set(e) != OPS[op]:
            problems.append(f"edit {n}: unknown op or fields {e!r}")
            continue
        if op == "drop_steps":
            if type(e["first"]) is not int or type(e["last"]) is not int or not 1 <= e["first"] <= e["last"] <= last:
                problems.append(f"edit {n}: drop_steps needs 1 <= first <= last <= {last} (step 0 is kept)")
        elif op == "drop_outputs":
            if (not isinstance(e["steps"], list) or not e["steps"]
                    or any(type(k) is not int or not 1 <= k <= last for k in e["steps"])):
                problems.append(f"edit {n}: drop_outputs needs steps within 1..{last}")
        else:
            k, quote = e["step"], e["quote"]
            if type(k) is not int or not 1 <= k <= last:
                problems.append(f"edit {n}: drop_quote step must be within 1..{last}")
                continue
            if not isinstance(quote, str) or not quote.strip():
                problems.append(f"edit {n}: drop_quote needs a non-empty quote")
                continue
            if not any(quote in text for i in steps[k] for text in visible_texts(messages[i])):
                problems.append(f"edit {n}: the quote does not occur in step {k}")
    return problems


def apply_edits(messages: list[dict], edits: list[dict]) -> tuple[list[dict], dict]:
    """The edited context and a report (per edit: what changed; per kept message: its source step)."""
    problems = validate_edits(messages, edits)
    if problems:
        raise EditError("; ".join(problems))
    steps = steps_of(messages)
    out = deepcopy(messages)
    report: list[dict] = []
    for e in edits:
        if e["op"] == "drop_quote":
            fields = [dict(f, message_index=i) for i in steps[e["step"]] for f in _remove_quote(out[i], e["quote"])]
            report.append({**e, "occurrences_removed": sum(f["occurrences"] for f in fields), "fields": fields})
    for e in edits:
        if e["op"] == "drop_outputs":
            replaced = []
            for k in sorted(set(e["steps"])):
                for i in steps[k]:
                    if out[i].get("role") == "tool":
                        out[i]["content"] = OUTPUT_PLACEHOLDER
                        replaced.append(i)
            report.append({**e, "outputs_replaced": len(replaced), "message_indices": replaced})
    dropped: set[int] = set()
    for e in edits:
        if e["op"] == "drop_steps":
            dropped |= set(range(e["first"], e["last"] + 1))
            report.append({**e, "messages_removed": sum(len(steps[k]) for k in range(e["first"], e["last"] + 1))})
    kept, source_step = [], []
    for k, indices in enumerate(steps):
        if k in dropped:
            continue
        for i in indices:
            kept.append(out[i])
            source_step.append(k)
    issue = history_replay_issue(kept)
    if issue is not None:
        raise EditError(f"edited context is not a complete conversation: {issue}")
    return kept, {"version": EDITS_VERSION, "edits": report, "dropped_steps": sorted(dropped),
                  "source_step_of_message": source_step, "messages_before": len(messages), "messages_after": len(kept)}


def _count(messages: list[dict], quote: str) -> int:
    return sum(s.count(quote) for m in messages for s in visible_texts(m))


def validate_request(body: bytes, original: list[dict], edited: list[dict], report: dict, *,
                     appended: list[dict] = (), forbidden: tuple[str, ...] = (), tools: list | None = None,
                     tool_name: str | None = None, hidden: list[int] = (), kept_output: dict | None = None) -> dict:
    """Checks on the actual serialized next request after edits (and any appended hand-back or spliced messages).

    ``forbidden``: texts that must appear nowhere in the request (the review or note turn: its prompt, thinking and
    answer outside the hand-back); ``tools``: the worker's own tool list the request must carry; ``tool_name``: a tool
    that no message of the request may call (the review decision tool); ``hidden``: message indices whose tool output
    the context-limit rule (execution/context_limit.py) replaced with the placeholder just before this request;
    ``kept_output``: the review turn's output kept in the context (arm set v3), which must follow the kept context
    exactly, with its reasoning, and carry no tool call."""
    request = json.loads(body)
    sent = request["messages"]
    strip = lambda ms: [{k: v for k, v in m.items() if k != "extra"} for m in ms]
    source_step = list(report["source_step_of_message"]) + [None] * len(appended)
    texts = [s for m in sent for s in visible_texts(m)]
    expected = strip(list(edited) + list(appended))
    for i in hidden:
        expected[i] = dict(expected[i], content=OUTPUT_PLACEHOLDER)
    checks: dict[str, Any] = {
        "request_is_edited_context": sent == expected,
        "context_limit_hid_only_tool_outputs": all(0 <= i < len(sent) and sent[i].get("role") == "tool"
                                                   and sent[i].get("content") == OUTPUT_PLACEHOLDER for i in hidden),
        "system_and_task_kept": sent[:2] == strip(original[:2]),
        "complete_conversation": history_replay_issue(list(edited) + list(appended)) is None,
        "intervention_turn_absent": not any(f and f in s for f in forbidden for s in texts),
    }
    if kept_output is not None:
        n = len(edited)
        want = {k: v for k, v in kept_output.items() if k != "extra"}  # reasoning fields included, as returned
        checks["review_output_kept_after_context"] = (n < len(sent) and sent[n] == want
                                                      and sent[n].get("role") == "assistant"
                                                      and not sent[n].get("tool_calls"))
    if tools is not None:
        checks["worker_tools_only"] = request.get("tools") == tools
    if tool_name is not None:
        checks["no_intervention_tool_calls"] = not any(
            (c.get("function") or {}).get("name") == tool_name for m in sent for c in m.get("tool_calls") or [])
    steps = steps_of(original)
    quotes = []
    n_context = len(report["source_step_of_message"])
    for e in (x for x in report["edits"] if x["op"] == "drop_quote"):
        # The edit is step-scoped (brief-interfaces): the quote must be gone from every field of its step in the
        # request. Copies elsewhere (other steps, other edits' placeholders, the hand-back) are recorded, not judged.
        in_step = [m for m, k in zip(sent, source_step) if k == e["step"]]
        others_now = [m for m, k in zip(sent[:n_context], source_step) if k != e["step"]]
        quotes.append({"step": e["step"], "quote_chars": len(e["quote"]), "left_in_step": _count(in_step, e["quote"]),
                       "elsewhere_in_context": _count(others_now, e["quote"]),
                       "in_appended": _count(sent[n_context:], e["quote"])})
    checks["quotes_removed"] = all(q["left_in_step"] == 0 for q in quotes)
    # tool-call ids of dropped steps must be absent (ids also used by a kept step are not evidence either way)
    ids_of = lambda ks: {c.get("id") for k in ks for i in steps[k] for c in original[i].get("tool_calls") or []}
    dropped_calls = ids_of(report["dropped_steps"]) - ids_of(k for k in range(len(steps))
                                                             if k not in report["dropped_steps"])
    sent_calls = {c.get("id") for m in sent for c in m.get("tool_calls") or []} | {m.get("tool_call_id") for m in sent}
    checks["dropped_steps_absent"] = not (dropped_calls & sent_calls)
    placeholder_ok = True
    for e in (x for x in report["edits"] if x["op"] == "drop_outputs"):
        for m, k in zip(sent, source_step):
            if k in e["steps"] and m.get("role") == "tool" and m.get("content") != OUTPUT_PLACEHOLDER:
                placeholder_ok = False
    checks["outputs_replaced"] = placeholder_ok
    return {"checks": checks, "valid": all(v for v in checks.values() if isinstance(v, bool)), "quotes": quotes,
            "request_messages": len(sent), "forbidden_texts": len([f for f in forbidden if f])}
