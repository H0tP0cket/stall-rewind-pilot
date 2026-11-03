"""Steps of a run's context at a cut, read from the context itself and bound to the event log.

Definition (brief-interfaces): step k (1-based) is the k-th model response of the run with its full delivered tool
outputs and any submission feedback delivered right after it; point k is the state after step k (point 0 = start).
A rejected response (format error) is a model response too: its step holds only the delivered format-error message,
because the rejected text never entered the context.

Source. The steps are read from the context saved with the checkpoint at the cut (``state.history``): that is exactly
what the worker sees at that point, so nothing later and nothing private can enter. A tool output the context-limit
rule hid from that context is read back from the record (the original delivered text it references), so views are
built from the full record. Every piece is then bound to its
event (response, observation delivery, format error, submission feedback) with ``through_sequence = cut``; anything
that cannot be bound is listed in ``checks.issues`` rather than dropped or invented. Workspace-change evidence comes
from the run's mutation events (mechanical, never shown to the labeller).

Fields. Each step is a list of fields in context order: reasoning, message, command (one per tool call), output (one
per string part of a tool result: ``output``, or ``output_head`` / ``output_tail`` when delivery was shortened),
feedback, format_error, user. ``text`` is the decoded text shown in views; ``h`` locates it in the context
(message index and path; ``also`` lists duplicate copies such as ``reasoning_details[0].text``). Commands and outputs
are JSON in the context, so a quote found in the decoded text is mapped back to its raw characters through
``text.json_string_offsets`` (``raw_span``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .text import json_string_offsets, locate_quote

STEPS_VERSION = "ledger-steps/v1"
AGENT_KINDS = ("reasoning", "message", "command")
FIELD_ORDER = ("reasoning", "message", "command", "output", "format_error", "feedback", "user")


@dataclass
class Field:
    id: str
    kind: str
    text: str
    h: dict[str, Any]
    source: dict[str, Any]
    tool_call_id: str | None = None
    index: int | None = None          # tool-call index for commands and outputs
    part: str | None = None           # JSON key of an output part
    meta: dict[str, Any] = field(default_factory=dict)
    raw: str = ""                     # the raw context string (not serialized)
    offsets: list[int] | None = None  # decoded -> raw index map (None: raw == text)

    def raw_span(self, start: int, end: int) -> tuple[int, int] | None:
        if self.offsets is None:
            return (start, end) if self.raw[start:end] == self.text[start:end] else None
        return self.offsets[start], self.offsets[end]

    def as_record(self) -> dict[str, Any]:
        out = {"id": self.id, "kind": self.kind, "text": self.text, "h": self.h, "source": self.source}
        for k in ("tool_call_id", "index", "part"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        if self.meta:
            out["meta"] = self.meta
        return out


@dataclass
class Step:
    k: int
    response_seq: int | None
    accepted: bool
    fields: list[Field] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)
    submission: bool = False
    changed_paths: list[str] = field(default_factory=list)
    workspace_change: bool = False
    flags: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def agent_fields(self) -> list[Field]:
        return [f for f in self.fields if f.kind in AGENT_KINDS]

    def outputs(self) -> list[Field]:
        return [f for f in self.fields if f.kind == "output"]

    def as_record(self) -> dict[str, Any]:
        return {"k": self.k, "point_before": self.k - 1, "point_after": self.k, "flags": self.flags,
                "response_seq": self.response_seq, "accepted": self.accepted, "submission": self.submission,
                "workspace_change": self.workspace_change, "changed_paths": self.changed_paths,
                "commands": self.commands, "detail": self.detail, "fields": [f.as_record() for f in self.fields]}


@dataclass
class Context:
    run_id: str
    checkpoint_id: str
    cut: int
    stall_point: int
    head: list[Field]
    steps: list[Step]
    checks: dict[str, Any]

    def step(self, k: int) -> Step:
        return self.steps[k - 1]

    def fingerprint(self) -> str:
        """sha256 over every field's text, in order: the identity of the labelled content."""
        h = hashlib.sha256()
        for f in self.head + [f for s in self.steps for f in s.fields]:
            h.update(f"{f.id}\x00{f.kind}\x00{f.text}\x01".encode())
        return h.hexdigest()

    def as_record(self) -> dict[str, Any]:
        return {"version": STEPS_VERSION, "run_id": self.run_id, "checkpoint_id": self.checkpoint_id,
                "cut_event_sequence": self.cut, "stall_point": self.stall_point, "fingerprint": self.fingerprint(),
                "head": [f.as_record() for f in self.head], "checks": self.checks}


def _output_parts(raw: str) -> list[tuple[str, str, list[int] | None]] | None:
    """String parts of a JSON tool result, in raw key order: (key, decoded text, offsets)."""
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    keys = [k for k in value if isinstance(value[k], str) and k in ("output", "output_head", "output_tail")]
    parts = []
    for key in keys:
        mapped = json_string_offsets(raw, key)
        if mapped is None:
            return None
        parts.append((key, mapped[0], mapped[1]))
    return parts


def full_messages(store, history: dict) -> tuple[list[dict], list[int]]:
    """The saved context with every tool output the context-limit rule hid read back from the record (the full record
    review views are built from; execution/context_limit.py). Returns (messages, message indices read back)."""
    from ..execution.context_limit import unmasked_messages
    return unmasked_messages(history["messages"], store)


def load_context(store, run_id: str, checkpoint_id: str) -> Context:
    """Steps 1..k of the context saved at the checkpoint for point k (see module docstring)."""
    cps = [c for c in store.list_checkpoints(run_id) if c.checkpoint_id == checkpoint_id]
    if len(cps) != 1:
        raise ValueError(f"checkpoint {checkpoint_id} not found once in run {run_id}")
    cp = cps[0]
    if cp.state.history is None:
        raise ValueError(f"checkpoint {checkpoint_id} has no saved context")
    cut = cp.event_sequence
    history = store.get_json(cp.state.history)
    messages, restored = full_messages(store, history)
    events = list(store.iter_events(run_id, through_sequence=cut))
    if any(e.sequence > cut for e in events):
        raise ValueError("store returned events beyond the cut")
    responses = {e.sequence: e for e in events if e.kind == "model_response"}
    deliveries: dict[tuple[str | None, str], list] = {}
    texts_by_kind: dict[str, dict[str, list]] = {}

    def delivered(e) -> str:
        return store.get_bytes(e.delivered_ref).decode("utf-8")

    for e in events:
        if e.delivered_ref is None:
            continue
        if e.kind == "observation_delivery":
            deliveries.setdefault((e.call_id, delivered(e)), []).append(e)
        elif e.kind in ("task_disclosure", "format_error", "native_submission_feedback"):
            texts_by_kind.setdefault(e.kind, {}).setdefault(delivered(e), []).append(e)
    mutations: dict[int, list[str]] = {}
    for e in events:
        if e.kind == "mutation" and (e.payload or {}).get("relevant"):
            p = e.payload
            files = list(p.get("workspace_fidelity_paths_changed") or p.get("files") or []) + \
                ["/" + str(x).lstrip("/") for x in p.get("runtime_paths_changed") or ()]
            mutations.setdefault(int(p.get("response_event_sequence") or -1), []).extend(files or ["<unlisted>"])
    used: set[str] = set()
    issues: list[str] = []

    def bind(kind: str, text: str, after: int) -> int | None:
        for e in texts_by_kind.get(kind, {}).get(text, []):
            if e.event_id not in used and e.sequence > after:
                used.add(e.event_id)
                return e.sequence
        return None

    head: list[Field] = []
    steps: list[Step] = []
    last_seq = 0
    for i, m in enumerate(messages):
        role, extra = m.get("role"), m.get("extra") or {}
        content = m.get("content") if isinstance(m.get("content"), str) else None
        if role == "assistant":
            rseq = extra.get("response_event_sequence")
            ev = responses.get(rseq)
            step = Step(len(steps) + 1, rseq, True)
            steps.append(step)
            if ev is None:
                issues.append(f"step {step.k}: assistant message {i} has no model_response event at {rseq}")
                body = {}
            else:
                last_seq = max(last_seq, ev.sequence)
                body = store.get_json(ev.output_ref) if ev.output_ref is not None else {}
            n = 0

            def add(kind, text, h, source, **kw):
                nonlocal n
                n += 1
                step.fields.append(Field(f"s{step.k}.f{n}", kind, text, h, source, **kw))
            reasoning = m.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                also = [f"reasoning_details[{j}].text" for j, d in enumerate(m.get("reasoning_details") or ())
                        if isinstance(d, dict) and d.get("text") == reasoning]
                if body and body.get("reasoning_content") != reasoning:
                    issues.append(f"step {step.k}: context reasoning differs from the recorded response")
                add("reasoning", reasoning, {"message": i, "path": "reasoning", "also": also},
                    {"seq": rseq, "field": "returned_reasoning_text", "decoder": "raw"}, raw=reasoning)
            if content:
                add("message", content, {"message": i, "path": "content"},
                    {"seq": rseq, "field": "assistant_text", "decoder": "raw"}, raw=content)
            for j, call in enumerate(m.get("tool_calls") or ()):
                args = ((call or {}).get("function") or {}).get("arguments")
                if not isinstance(args, str):
                    issues.append(f"step {step.k}: tool call {j} has non-string arguments")
                    continue
                mapped = json_string_offsets(args, "command")
                text, offsets = mapped if mapped is not None else (args, None)
                step.commands.append({"index": j, "tool_call_id": call.get("id"), "command": text})
                add("command", text, {"message": i, "path": f"tool_calls[{j}].function.arguments"},
                    {"seq": rseq, "field": f"tool_call[{j}].arguments", "decoder": "json:command" if mapped else "raw"},
                    tool_call_id=call.get("id"), index=j, raw=args, offsets=offsets)
            continue
        if role == "tool":
            if not steps:
                issues.append(f"tool message {i} before any response")
                continue
            step = steps[-1]
            call_id = m.get("tool_call_id")
            evs = deliveries.get((call_id, content or ""), [])
            ev = next((e for e in evs if e.event_id not in used), None)
            if ev is None:
                issues.append(f"step {step.k}: tool message {i} ({call_id}) has no matching observation_delivery")
            else:
                used.add(ev.event_id)
                last_seq = max(last_seq, ev.sequence)
            index = next((c["index"] for c in step.commands if c["tool_call_id"] == call_id), None)
            parts = _output_parts(content or "")
            meta = {}
            try:
                value = json.loads(content or "")
                if isinstance(value, dict):
                    meta = {k: v for k, v in value.items() if k not in ("output", "output_head", "output_tail")}
            except ValueError:
                pass
            if ev is not None and (ev.payload or {}).get("native_submission_reopen"):
                meta["submission_reopen"] = True
            if parts is None:
                parts = [(None, content or "", None)]
            for key, text, offsets in parts:
                step.fields.append(Field(f"s{step.k}.f{len(step.fields) + 1}", "output", text,
                                         {"message": i, "path": "content"},
                                         {"seq": ev.sequence if ev else None, "field": "delivered_content",
                                          "decoder": f"json:{key}" if key else "raw"},
                                         tool_call_id=call_id, index=index, part=key, meta=dict(meta),
                                         raw=content or "", offsets=offsets))
            continue
        # system / user messages
        text = content or ""
        if extra.get("interrupt_type") == "FormatError":
            rseq = extra.get("response_event_sequence")
            step = Step(len(steps) + 1, rseq, False)
            steps.append(step)
            if rseq not in responses:
                issues.append(f"step {step.k}: format error {i} has no model_response event at {rseq}")
            seq = bind("format_error", text, -1)
            if seq is None:
                issues.append(f"step {step.k}: format error {i} has no format_error event")
            step.fields.append(Field(f"s{step.k}.f1", "format_error", text, {"message": i, "path": "content"},
                                     {"seq": seq, "field": "delivered_text", "decoder": "raw"}, raw=text))
            continue
        if not steps:
            seq = bind("task_disclosure", text, -1)
            if seq is None:
                issues.append(f"head message {i} has no task_disclosure event")
            head.append(Field(f"h{len(head) + 1}", role or "user", text, {"message": i, "path": "content"},
                              {"seq": seq, "field": "delivered_text", "decoder": "raw"}, raw=text))
            continue
        step = steps[-1]
        seq = bind("native_submission_feedback", text, -1)
        kind = "feedback"
        if seq is None:
            seq, kind = bind("task_disclosure", text, -1), "user"
        if seq is None:
            issues.append(f"step {step.k}: user message {i} has no delivery event")
        step.fields.append(Field(f"s{step.k}.f{len(step.fields) + 1}", kind, text, {"message": i, "path": "content"},
                                 {"seq": seq, "field": "delivered_text", "decoder": "raw"}, raw=text))
        if kind == "feedback":
            step.submission = True
    for step in steps:
        paths = mutations.get(step.response_seq or -2, [])
        step.workspace_change = bool(paths)
        step.changed_paths = sorted(set(paths))
        step.fields.sort(key=lambda f: (FIELD_ORDER.index(f.kind) if f.kind != "output" else 3,
                                        f.index if f.index is not None else 0))
        for n, f in enumerate(step.fields, 1):
            f.id = f"s{step.k}.f{n}"
    n_responses = len(responses)
    counters = history.get("counters") or {}
    checks = {"messages": len(messages), "steps": len(steps), "model_responses_through_cut": n_responses,
              "counters_n_calls": counters.get("n_calls"),
              "steps_match_responses": len(steps) == n_responses,
              "steps_match_counters": counters.get("n_calls") in (None, len(steps)),
              "unbound_events": sorted(e.sequence for e in events if e.kind == "observation_delivery"
                                       and e.event_id not in used),
              "outputs_read_back_from_record": restored,
              "issues": issues}
    return Context(run_id, checkpoint_id, cut, len(steps), head, steps, checks)


def resolve(step: Step, quote: str, *, kinds: tuple[str, ...] | None = None) -> tuple[Field, int, int] | None:
    """First exact occurrence of ``quote`` in the step's fields (optionally of the given kinds), in field order."""
    for f in step.fields:
        if kinds is not None and f.kind not in kinds:
            continue
        hit = locate_quote(f.text, quote)
        if hit is not None:
            return f, hit[0], hit[1]
    return None


def occurrences(step: Step, quote: str) -> int:
    return sum(f.text.count(quote) for f in step.fields)
