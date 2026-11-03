"""The units Ledger: fixed units cut by code, mechanical items, and step-by-step classification (user, 2026-10-22).

Built beside the item passes (``passes.build_items``, protocols v1–v2.1), which stay as they are and selectable.

1. Segmentation (no model, ``SEGMENT_VERSION``). Every step of the worker-visible record (``steps.load_context``) is
   cut into numbered units with stable ids, from that step's own text only:
     Tn      unit n of the task text (the head's user messages; the system message is not shown);
     Sk.rn   reasoning unit n of step k;  Sk.mn  message unit n;
     Sk.cn   command n (one tool call, whole);  Sk.on  the output of command n (all its delivered parts);
     Sk.fn   submission feedback;  Sk.en  format error;  Sk.un  a later user message;
     Sk.hn   a workspace hunk of step k (mechanical; never shown to the labeller).
   Prose (task, reasoning, message) is cut into sentences: every line is its own piece, a line is split at sentence
   ends (".", "!" or "?" then whitespace then a capital, digit, quote or bracket; not after e.g./i.e./etc.), a fenced
   code block is one unit, and a piece shorter than 12 characters or 3 words ("Hmm.", "Wait —") is joined to the next
   piece of the same paragraph (else to the previous one). Units are spans of the field text; every non-space
   character of a field lies in exactly one unit.
2. Mechanical items (no model): every command is an action (a decision carried out); every hunk carries out the
   command that produced it (``hunks``), or only its step when no single command can be named; every submission with
   the failing count read from the worker-visible feedback text.
3. Labels, one model call per step, prefix-only: the call for step k sees the task, steps 1..k and the frozen labels of
   steps < k, and answers with unit ids only: (a) tags of step k's reasoning and message units (claim / plan / idea /
   observation, several allowed, none = left out); (b) for each command of step k, the units it rests on (claims and
   observations, earlier outputs, feedback and user messages, task units); (c) for each claim so far (and step k's
   own), whether step k's outputs or feedback confirm or contradict it (left out = say nothing), with the unit that
   shows it. Labels of a step are never revised. An unknown id (including any id of a later step) is a failed try; two
   tries per call; a known id in a place it is not allowed is dropped and logged.
4. ``render_units_view``: the flat Ledger view from units and labels (not wired into any arm).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..observers.t2.glm import BudgetExceeded
from . import templates as T
from .flags import FLAGS_VERSION, compute_flags
from .hunks import HUNKS_VERSION, step_hunks
from .passes import parse_json_object_only
from .prompts import (LABEL_SYSTEM, UNITS_AFTER_LOG, UNITS_ANSWER, UNITS_LOG, UNITS_NO_COMMANDS, UNITS_NO_OUTPUTS,
                      UNITS_NO_TAGS, UNITS_PROMPT_VERSION, UNITS_RESTS_ON, UNITS_STATUS, UNITS_TAGS,
                      units_prompt_sha256, units_protocol_record)
from .render import _output_meta
from .steps import STEPS_VERSION, load_context
from .text import _ABBREV, _FENCE, _SENTENCE_END

SEGMENT_VERSION = "ledger-units-seg/v1"
UNITS_VERSION = "ledger-units/v1"
VIEW_VERSION = "ledger-units-view/v1-draft"
TAGS = ("claim", "plan", "idea", "observation")
VERDICTS = ("confirms", "contradicts")
TAGGABLE = ("reasoning", "message")
EVIDENCE = ("output", "feedback")
MIN_CHARS, MIN_WORDS = 12, 3
PURPOSE = "ledger:units-step"
_FIELD_LETTER = {"reasoning": "r", "message": "m", "command": "c", "output": "o", "feedback": "f",
                 "format_error": "e", "user": "u"}
_FAILING = re.compile(r"Hidden tests still failing: (\d+)\.")  # experiments/v2_text.SUBMISSION_FEEDBACK
ID_RX = re.compile(r"^(?:T\d+|S\d+\.[rmcofeuh]\d+)$")


# ---------------------------------------------------------------- 1. segmentation

def _words(s: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", s))


def segment_prose(text: str) -> list[dict[str, Any]]:
    """Sentence units of one prose field: [{"start", "end", "para", "code"}] in order (see module doc)."""
    pieces: list[list[Any]] = []  # [start, end, para, code]
    pos, para, in_fence, fence_start = 0, 0, False, 0
    for line in text.splitlines(keepends=True):
        ls, le = pos, pos + len(line)
        pos = le
        if _FENCE.match(line):
            if in_fence:
                pieces.append([fence_start, le, para, True])
                para += 1  # nothing is joined across a code block
            else:
                fence_start = ls
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            para += 1
            continue
        cursor = ls
        for m in _SENTENCE_END.finditer(text, ls, le):
            if _ABBREV.search(text[max(ls, m.start() - 8):m.start()]):
                continue
            pieces.append([cursor, m.start(), para, False])
            cursor = m.end()
        pieces.append([cursor, le, para, False])
    if in_fence:
        pieces.append([fence_start, len(text), para, True])
    trimmed = []
    for s, e, p, code in pieces:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append([s, e, p, code])

    def short(x):
        return not x[3] and (x[1] - x[0] < MIN_CHARS or _words(text[x[0]:x[1]]) < MIN_WORDS)

    out: list[list[Any]] = []
    pending = None
    for x in trimmed:
        if pending is not None:
            if not x[3] and x[2] == pending[2]:
                x = [pending[0], x[1], x[2], False]
            elif out and not out[-1][3] and out[-1][2] == pending[2]:
                out[-1][1] = pending[1]
            else:
                out.append(pending)
            pending = None
        if short(x):
            pending = x
            continue
        out.append(x)
    if pending is not None:
        if out and not out[-1][3] and out[-1][2] == pending[2]:
            out[-1][1] = pending[1]
        else:
            out.append(pending)
    return [{"start": s, "end": e, "para": p, "code": code} for s, e, p, code in out]


def _unit(uid: str, step: int, kind: str, parts: list[dict[str, Any]], text: str, **kw) -> dict[str, Any]:
    return {"id": uid, "step": step, "kind": kind, "parts": parts, "text": text, **kw}


def segment_context(ctx) -> dict[str, Any]:
    """{"task": [units], "steps": {k: [units]}} for a Context; ids depend only on each field's own text."""
    task, n = [], 0
    for f in ctx.head:
        if f.kind == "system":
            continue
        for seg in segment_prose(f.text):
            n += 1
            task.append(_unit(f"T{n}", 0, "task", [{"field": f.id, "start": seg["start"], "end": seg["end"]}],
                              f.text[seg["start"]:seg["end"]], para=f"{f.id}:{seg['para']}", code=seg["code"]))
    steps: dict[int, list[dict[str, Any]]] = {}
    for st in ctx.steps:
        units: list[dict[str, Any]] = []
        count: dict[str, int] = {}
        outputs: dict[Any, dict[str, Any]] = {}
        for f in st.fields:
            letter = _FIELD_LETTER[f.kind]
            if f.kind in TAGGABLE:
                for seg in segment_prose(f.text):
                    count[letter] = count.get(letter, 0) + 1
                    units.append(_unit(f"S{st.k}.{letter}{count[letter]}", st.k, f.kind,
                                       [{"field": f.id, "start": seg["start"], "end": seg["end"]}],
                                       f.text[seg["start"]:seg["end"]], para=f"{f.id}:{seg['para']}", code=seg["code"]))
            elif f.kind == "command":
                units.append(_unit(f"S{st.k}.c{f.index + 1}", st.k, "command",
                                   [{"field": f.id, "start": 0, "end": len(f.text)}], f.text, index=f.index))
            elif f.kind == "output":
                if f.index is not None and f.index in outputs:  # a second delivered part of the same output
                    u = outputs[f.index]
                    u["parts"].append({"field": f.id, "start": 0, "end": len(f.text), "part": f.part,
                                       "meta": f.meta})
                    u["text"] += "\n" + f.text
                    continue
                num = f.index + 1 if f.index is not None else 1000 + len(outputs)
                u = _unit(f"S{st.k}.o{num}", st.k, "output",
                          [{"field": f.id, "start": 0, "end": len(f.text), "part": f.part, "meta": f.meta}], f.text,
                          index=f.index)
                outputs[f.index if f.index is not None else ("x", len(outputs))] = u
                units.append(u)
            else:
                count[letter] = count.get(letter, 0) + 1
                units.append(_unit(f"S{st.k}.{letter}{count[letter]}", st.k, f.kind,
                                   [{"field": f.id, "start": 0, "end": len(f.text)}], f.text))
        # log order: reasoning, message, then each command followed by its output, then the rest (as render.py)
        cmd_out = {u["index"]: u for u in units if u["kind"] == "output" and u["index"] is not None}
        ordered = [u for u in units if u["kind"] in TAGGABLE]
        for u in units:
            if u["kind"] == "command":
                ordered.append(u)
                if u["index"] in cmd_out:
                    ordered.append(cmd_out.pop(u["index"]))
        ordered += [u for u in units if u["kind"] == "output" and u not in ordered]
        ordered += [u for u in units if u["kind"] not in TAGGABLE + ("command", "output")]
        assert len(ordered) == len(units)
        steps[st.k] = ordered
    return {"version": SEGMENT_VERSION, "task": task, "steps": steps}


def segmentation_coverage(ctx, seg: Mapping[str, Any]) -> dict[str, Any]:
    """Every non-space character of every shown field lies in exactly one unit; unit text equals its span."""
    fields = {f.id: f for f in ctx.head if f.kind != "system"}
    fields.update({f.id: f for s in ctx.steps for f in s.fields})
    cover: dict[str, list[int]] = {fid: [0] * len(f.text) for fid, f in fields.items()}
    bad = []
    units = list(seg["task"]) + [u for k in sorted(seg["steps"]) for u in seg["steps"][k]]
    for u in units:
        for p in u["parts"]:
            for i in range(p["start"], p["end"]):
                cover[p["field"]][i] += 1
        if u["kind"] != "output" and u["text"] != fields[u["parts"][0]["field"]].text[u["parts"][0]["start"]:
                                                                                   u["parts"][0]["end"]]:
            bad.append(f"{u['id']}: text differs from its span")
    for fid, c in cover.items():
        text = fields[fid].text
        miss = [i for i, n in enumerate(c) if n == 0 and not text[i].isspace()]
        dup = [i for i, n in enumerate(c) if n > 1]
        if miss or dup:
            bad.append(f"{fid}: {len(miss)} characters in no unit, {len(dup)} in two")
    ids = [u["id"] for u in units]
    if len(ids) != len(set(ids)):
        bad.append("duplicate unit ids")
    return {"pass": not bad, "units": len(units), "problems": bad[:10]}


# ---------------------------------------------------------------- 2. mechanical items

def mechanical(ctx, seg: Mapping[str, Any], hunks: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    actions, changes, submissions, status = [], [], [], {}
    for st in ctx.steps:
        k = st.k
        cmds = {u["index"]: u["id"] for u in seg["steps"][k] if u["kind"] == "command"}
        for u in seg["steps"][k]:
            if u["kind"] == "command":
                actions.append({"unit": u["id"], "step": k, "type": "action", "carried_out": True})
            if u["kind"] == "feedback":
                m = _FAILING.search(u["text"])
                submissions.append({"unit": u["id"], "step": k, "failing_count": int(m.group(1)) if m else None,
                                    "command": next((cmds[c["index"]] for c in st.commands
                                                     if "COMPLETE_TASK_AND_SUBMIT" in c["command"]), None)})
        h = hunks.get(k) or {"status": "no_record", "source": None, "hunks": [], "omitted": []}
        status[k] = {"status": h["status"], "source": h["source"], "omitted_paths": len(h["omitted"])}
        for n, x in enumerate(h["hunks"], 1):
            changes.append({"id": f"S{k}.h{n}", "step": k, "path": x["path"], "paths": x["paths"], "kind": x["kind"],
                            "binary": x["binary"], "header": x["header"], "added": x["added"],
                            "removed": x["removed"], "lines": x["lines"],
                            "carries_out": cmds.get(x["command_index"]) if x.get("command_index") is not None else None,
                            "carries_out_step": k, "attribution": x["attribution"],
                            "candidates": [cmds.get(i) for i in x.get("candidates", []) if i in cmds]})
    return {"actions": actions, "hunks": changes, "submissions": submissions, "hunk_status": status,
            "hunks_version": HUNKS_VERSION}


# ---------------------------------------------------------------- labeller input

def _unit_lines(units: Sequence[Mapping[str, Any]], mark: Callable[[Mapping[str, Any]], str]) -> list[str]:
    """Prose units one per line (blank line between paragraphs); other units with their header and body."""
    lines: list[str] = []
    last_para, last_kind = None, None
    for u in units:
        kind = u["kind"]
        if kind in TAGGABLE or kind == "task":
            if kind != last_kind and kind in TAGGABLE:
                if lines:
                    lines.append("")
                lines.append(f"{kind}:")
            elif last_para is not None and u["para"] != last_para:
                lines.append("")
            lines.append(f"{mark(u)} {u['text']}")
            last_para = u["para"]
        elif kind == "command":
            lines += ["", f"{mark(u)} command {u['index'] + 1}:", u["text"]]
        elif kind == "output":
            for n, p in enumerate(u["parts"]):
                num = u["index"] + 1 if u["index"] is not None else "?"
                head = T.OUTPUT_HEADER.format(oid="", n=num, meta=_output_meta(p, n == 0))
                lines.append(f"{mark(u)} {head}" if n == 0 else head)
                lines.append(u["text"] if len(u["parts"]) == 1 else _part_text(u, p))
        else:
            lines += ["", f"{mark(u)} {T.FIELD_HEADERS[kind].format(n=1)}", u["text"]]
        last_kind = kind
    return lines


def _part_text(u: Mapping[str, Any], p: Mapping[str, Any]) -> str:
    # the parts of a multi-part output are joined by "\n" in u["text"], in order
    texts, pos = [], 0
    for q in u["parts"]:
        n = q["end"] - q["start"]
        texts.append(u["text"][pos:pos + n])
        pos += n + 1
    return texts[u["parts"].index(p)]


def _bracket(u: Mapping[str, Any]) -> str:
    return f"[{u['id']}]"


def task_text(seg: Mapping[str, Any]) -> str:
    return "\n".join(["TASK"] + _unit_lines(seg["task"], _bracket))


def step_text(seg: Mapping[str, Any], k: int) -> str:
    return "\n".join([f"<step {k}>"] + _unit_lines(seg["steps"][k], _bracket) + [f"</step {k}>"])


def _ids(xs: Sequence[str]) -> str:
    return ", ".join(xs) if xs else "none"


def labels_text(k: int, lab: Mapping[str, Any] | None) -> str:
    out = [f"<labels of step {k}>"]
    if lab is None or lab["status"] == "no_call":
        out.append("nothing to label in this step")
    elif lab["status"] != "ok":
        out.append("labels missing: the labelling call for this step failed")
    else:
        for t in TAGS:
            out.append(f"{t}: {_ids([u for u, ts in lab['tags'].items() if t in ts])}")
        for c, targets in lab["rests_on"].items():
            out.append(f"{c} rests on: {_ids(targets)}")
        st = [f"{c} {v['verdict']} (shown by {v['evidence']})" for c, v in lab["claim_status"].items()]
        out.append(f"claim status from the outputs of step {k}: {'; '.join(st) if st else 'nothing confirmed or contradicted'}")
    out.append(f"</labels of step {k}>")
    return "\n".join(out)


def _flat(text: str) -> str:
    return " ".join(text.split())


def step_messages(seg: Mapping[str, Any], labels: Mapping[int, Mapping[str, Any]], k: int,
                  claims: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """The request for step k: the task, steps 1..k-1 each followed by its frozen labels, step k, then the
    instructions. Nothing of steps after k is read."""
    units = seg["steps"][k]
    taggable = [u["id"] for u in units if u["kind"] in TAGGABLE]
    commands = [u["id"] for u in units if u["kind"] == "command"]
    evidence = [u["id"] for u in units if u["kind"] in EVIDENCE]
    log = [task_text(seg), ""]
    for j in range(1, k):
        log += [step_text(seg, j), labels_text(j, labels.get(j)), ""]
    log.append(step_text(seg, k))
    parts = [UNITS_AFTER_LOG.format(k=k),
             UNITS_TAGS.format(k=k, units=", ".join(taggable)) if taggable else UNITS_NO_TAGS.format(k=k),
             UNITS_RESTS_ON.format(k=k, commands=", ".join(commands)) if commands else UNITS_NO_COMMANDS.format(k=k)]
    if evidence:
        lines = [f"[{c['id']}] (step {c['step']}) {_flat(c['text'])}" for c in claims] or ["none"]
        parts.append(UNITS_STATUS.format(k=k, outputs=", ".join(evidence)) + "\n" + "\n".join(lines))
    else:
        parts.append(UNITS_NO_OUTPUTS.format(k=k))
    fmt = ('{"tags": {"claim": [<unit ids>], "plan": [<unit ids>], "idea": [<unit ids>], "observation": [<unit ids>]}, '
           '"rests_on": {' + ", ".join(f'"{c}": [<unit ids>]' for c in commands) + '}, "claim_status": '
           + ('{"<claim unit id>": {"verdict": "<confirms or contradicts>", "evidence": "<unit id>"}}}'
              if evidence else "{}}"))
    parts.append(UNITS_ANSWER.format(format=fmt))
    user = UNITS_LOG + "\n".join(log) + "\n" + "\n\n".join(parts)
    return [{"role": "system", "content": LABEL_SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- 3. answer check and labelling

class AnswerRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def check_answer(answer: Any, *, k: int, seg: Mapping[str, Any], labels: Mapping[int, Mapping[str, Any]],
                 order: Mapping[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The step-k answer as labels, or AnswerRejected (a failed try). Known ids in a place they are not allowed are
    dropped and returned in the drop log."""
    if not isinstance(answer, dict) or set(answer) != {"tags", "rests_on", "claim_status"}:
        raise AnswerRejected("wrong_keys")
    units = seg["steps"][k]
    by_id = {u["id"]: u for u in seg["task"]}
    for j in range(1, k + 1):
        by_id.update({u["id"]: u for u in seg["steps"][j]})
    dropped: list[dict[str, Any]] = []

    def known(x: Any, where: str) -> str:
        if not isinstance(x, str) or x not in by_id:
            raise AnswerRejected(f"unknown_id in {where}: {str(x)[:60]!r}")
        return x

    tags_in = answer["tags"]
    if not isinstance(tags_in, dict) or not set(tags_in) <= set(TAGS) or \
            not all(isinstance(v, list) for v in tags_in.values()):
        raise AnswerRejected("wrong_shape tags")
    taggable = {u["id"] for u in units if u["kind"] in TAGGABLE}
    tags: dict[str, list[str]] = {}
    for t in TAGS:
        for x in tags_in.get(t, []):
            x = known(x, f"tags.{t}")
            if x not in taggable:
                dropped.append({"step": k, "where": f"tags.{t}", "id": x, "reason": "not_a_reasoning_or_message_unit_of_this_step"})
                continue
            if t not in tags.setdefault(x, []):
                tags[x].append(t)
    tags = {u: [t for t in TAGS if t in ts] for u, ts in sorted(tags.items(), key=lambda kv: order[kv[0]])}
    rests_in = answer["rests_on"]
    commands = [u["id"] for u in units if u["kind"] == "command"]
    if not isinstance(rests_in, dict) or set(rests_in) != set(commands) or \
            not all(isinstance(v, list) for v in rests_in.values()):
        raise AnswerRejected("wrong_shape rests_on (keys must be exactly the step's commands)")
    frozen = {u: ts for j in range(1, k) for u, ts in ((labels.get(j) or {}).get("tags") or {}).items()}
    allowed = {u["id"] for u in seg["task"]}
    allowed |= {u["id"] for j in range(1, k) for u in seg["steps"][j] if u["kind"] in ("output", "feedback", "user")}
    allowed |= {u for u, ts in frozen.items() if {"claim", "observation"} & set(ts)}
    allowed |= {u for u, ts in tags.items() if {"claim", "observation"} & set(ts)}
    rests: dict[str, list[str]] = {}
    for c in commands:
        keep = []
        for x in rests_in[c]:
            x = known(x, f"rests_on.{c}")
            if x not in allowed:
                dropped.append({"step": k, "where": f"rests_on.{c}", "id": x, "reason": "not_an_allowed_target"})
            elif x not in keep:
                keep.append(x)
        rests[c] = sorted(keep, key=lambda u: order[u])
    status_in = answer["claim_status"]
    if not isinstance(status_in, dict):
        raise AnswerRejected("wrong_shape claim_status")
    claims = {u for u, ts in frozen.items() if "claim" in ts} | {u for u, ts in tags.items() if "claim" in ts}
    evidence = {u["id"] for u in units if u["kind"] in EVIDENCE}
    status: dict[str, dict[str, str]] = {}
    for c, v in status_in.items():
        known(c, "claim_status")
        if not isinstance(v, dict) or v.get("verdict") not in VERDICTS + ("nothing", "says nothing", None):
            raise AnswerRejected(f"wrong_shape claim_status.{c}")
        if v.get("verdict") in ("nothing", "says nothing", None):
            continue  # an explicit "says nothing" is the same as leaving the claim out
        ev = known(v.get("evidence"), f"claim_status.{c}.evidence")
        if c not in claims:
            dropped.append({"step": k, "where": "claim_status", "id": c, "reason": "not_a_claim"})
            continue
        if ev not in evidence:
            dropped.append({"step": k, "where": f"claim_status.{c}", "id": ev,
                            "reason": "evidence_not_an_output_or_feedback_of_this_step"})
            continue
        status[c] = {"verdict": v["verdict"], "evidence": ev}
    status = dict(sorted(status.items(), key=lambda kv: order[kv[0]]))
    return {"tags": tags, "rests_on": rests, "claim_status": status}, dropped


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def label_step(client, seg, labels, k, order, claims) -> dict[str, Any]:
    """One step's labels (two tries). BudgetExceeded propagates (the build stops); other call errors are recorded."""
    units = seg["steps"][k]
    if not any(u["kind"] in TAGGABLE + ("command",) for u in units):
        return {"step": k, "status": "no_call", "tags": {}, "rests_on": {}, "claim_status": {}, "dropped": [],
                "tries": [], "step_sha256": _sha(step_text(seg, k))}
    messages = step_messages(seg, labels, k, claims)
    request_sha = _sha(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    tries = []
    for repeat in (0, 1):
        rec = {"repeat": repeat, "prompt_version": UNITS_PROMPT_VERSION, "prompt_sha256": units_prompt_sha256(),
               "request_chars": sum(len(m["content"]) for m in messages)}
        try:
            ans = client.complete(messages, purpose=PURPOSE, repeat=repeat)
        except BudgetExceeded:
            raise
        except Exception as exc:  # recorded, never filled in
            tries.append({**rec, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            break
        rec.update({"record_id": ans.call_record_id, "cached": ans.cached, "cost_usd": ans.cost_usd,
                    "provider": ans.provider, "model": ans.model, "finish_reason": ans.finish_reason,
                    "answer_chars": len(ans.text or "")})
        parsed = parse_json_object_only(ans.text)
        if parsed is None:
            tries.append({**rec, "status": "not_one_json_object"})
            continue
        try:
            lab, dropped = check_answer(parsed, k=k, seg=seg, labels=labels, order=order)
        except AnswerRejected as rej:
            tries.append({**rec, "status": "rejected", "reason": rej.reason})
            continue
        tries.append({**rec, "status": "ok"})
        return {"step": k, "status": "ok", **lab, "dropped": dropped, "tries": tries, "request_sha256": request_sha,
                "step_sha256": _sha(step_text(seg, k))}
    return {"step": k, "status": "failed", "tags": {}, "rests_on": {}, "claim_status": {}, "dropped": [],
            "tries": tries, "request_sha256": request_sha, "step_sha256": _sha(step_text(seg, k))}


def claims_before(seg, labels, k) -> list[dict[str, Any]]:
    """Claims tagged in steps < k, in log order, with their text."""
    out = []
    for j in range(1, k):
        lab = labels.get(j) or {}
        for u in seg["steps"][j]:
            if "claim" in (lab.get("tags") or {}).get(u["id"], []):
                out.append({"id": u["id"], "step": j, "text": u["text"]})
    return out


def claim_statuses(seg, labels) -> dict[str, dict[str, Any]]:
    """Per claim: every verdict given at later steps, and its status at the cut (contradicted later if any verdict
    contradicts it, else confirmed later if any confirms it, else unchecked)."""
    out = {}
    for j in sorted(labels):
        for u, ts in (labels[j].get("tags") or {}).items():
            if "claim" in ts:
                out[u] = {"step": j, "verdicts": []}
    for j in sorted(labels):
        for c, v in (labels[j].get("claim_status") or {}).items():
            if c in out:
                out[c]["verdicts"].append({"step": j, **v})
    for c, x in out.items():
        vs = [v["verdict"] for v in x["verdicts"]]
        x["status"] = "contradicted_later" if "contradicts" in vs else "confirmed_later" if "confirms" in vs \
            else "unchecked"
        first = next((v for v in x["verdicts"] if v["verdict"] == {"contradicted_later": "contradicts",
                                                                     "confirmed_later": "confirms"}.get(x["status"])),
                     None)
        x["status_step"] = first["step"] if first else None
    return out


def build_units(store, run_id: str, checkpoint_id: str, client, *, out_dir: str | Path | None = None,
                frozen: Mapping[str, Any] | None = None, upto: int | None = None,
                on_step: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Segment, add the mechanical items and label step by step (see module doc). ``frozen``: an earlier units doc of
    the same run at an earlier (or the same) cut; its labels are kept verbatim for every step whose content is
    unchanged, and only later steps are labelled. ``upto``: label only steps 1..upto (a prefix; status says so).
    Writes ``ledger-units.json`` (``ledger-units.partial.json`` when not every step was labelled) into ``out_dir``."""
    ctx = load_context(store, run_id, checkpoint_id)
    compute_flags(ctx)
    bind = getattr(client, "bind_context", None)
    if bind is not None:
        bind(ctx)
    seg = segment_context(ctx)
    order = {u["id"]: i for i, u in enumerate(list(seg["task"]) + [u for k in sorted(seg["steps"])
                                                                    for u in seg["steps"][k]])}
    n = len(ctx.steps)
    if n == 0:
        raise ValueError("no steps before the cut: nothing to label")
    labels: dict[int, dict[str, Any]] = {}
    reused = []
    if frozen is not None:
        if frozen.get("run_id") != run_id or frozen.get("task_sha256") != _sha(task_text(seg)):
            raise ValueError("the frozen labels are of another run or task text")
        for key, lab in sorted(((int(a), b) for a, b in frozen["labels"].items())):
            if key > n or lab.get("step_sha256") != _sha(step_text(seg, key)):
                break
            labels[key] = lab
            reused.append(key)
    stop = None
    last = min(n, upto) if upto else n
    for k in range(1, last + 1):
        if k in labels:
            continue
        try:
            labels[k] = label_step(client, seg, labels, k, order, claims_before(seg, labels, k))
        except BudgetExceeded as exc:
            stop = {"step": k, "reason": f"BudgetExceeded: {exc}"}
            break
        if on_step is not None:
            on_step(labels[k])
    hunks = step_hunks(store, run_id, ctx)
    mech = mechanical(ctx, seg, hunks)
    done = sorted(labels)
    complete = done == list(range(1, n + 1))
    status = ("ok" if complete and all(labels[k]["status"] in ("ok", "no_call") for k in done)
              else "incomplete" if complete else "partial")
    desc = client.describe() if hasattr(client, "describe") else {}
    tries = [t for k in done for t in labels[k]["tries"]]
    doc = {
        "version": UNITS_VERSION, "kind": "ledger-units", "status": status, "run_id": run_id,
        "checkpoint_id": checkpoint_id, "stall_point": n, "cut_event_sequence": ctx.cut,
        "context_fingerprint": ctx.fingerprint(), "task_sha256": _sha(task_text(seg)),
        "labelled_steps": done, "stopped": stop, "reused_frozen_steps": reused,
        "head": [f.as_record() for f in ctx.head], "steps": [s.as_record() for s in ctx.steps],
        "units": {"task": seg["task"], "steps": {str(k): v for k, v in seg["steps"].items()}},
        "labels": {str(k): labels[k] for k in done},
        "claims": claim_statuses(seg, labels),
        "mechanical": mech,
        "labeller": {"model": desc.get("model"), "provider": desc.get("provider"), "settings": desc.get("settings"),
                     "fake_labels": bool(desc.get("fake_labels")), "protocol": units_protocol_record(),
                     "calls": sum(1 for k in done if labels[k]["tries"]), "tries": len(tries),
                     "failed_tries": sum(1 for t in tries if t["status"] != "ok"),
                     "cost_usd": sum(t.get("cost_usd") or 0.0 for t in tries)},
        "fake_labels": bool(desc.get("fake_labels")), "checks": ctx.checks,
        "versions": {"steps": STEPS_VERSION, "flags": FLAGS_VERSION, "segment": SEGMENT_VERSION,
                     "hunks": HUNKS_VERSION, "units": UNITS_VERSION},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / ("ledger-units.json" if status != "partial" else "ledger-units.partial.json")).write_text(
            json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


# ---------------------------------------------------------------- 4. the flat Ledger view

LEGEND = ("Steps are numbered from 1. Point k is the state after step k; point 0 is the start, before step 1. Every "
          "unit is marked with its id: ⟦Tn⟧ the task text, ⟦Sk⟧ step k, ⟦Sk.rn⟧ its reasoning, ⟦Sk.mn⟧ its message, "
          "⟦Sk.cn⟧ command n, ⟦Sk.on⟧ the output of command n, ⟦Sk.fn⟧ submission feedback, ⟦Sk.hn⟧ a change to the "
          "files. Labels follow the id: claim (with its status: contradicted or confirmed later, at a step, or "
          "unchecked), plan, idea, observation; what each command rests on; which command each file change carries "
          "out; the failing count of each submission.")
STATUS_LABEL = {"contradicted_later": "contradicted later, step {s}", "confirmed_later": "confirmed later, step {s}",
                "unchecked": "unchecked"}


def render_units_view(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The flat Ledger view (arm 2 could use it later; not wired into any arm): every unit with its id and labels,
    every step header with its mechanical flags, hunks after the outputs of their step."""
    labels = {int(k): v for k, v in doc["labels"].items()}
    tags = {u: ts for lab in labels.values() for u, ts in (lab.get("tags") or {}).items()}
    rests = {c: r for lab in labels.values() for c, r in (lab.get("rests_on") or {}).items()}
    claims = doc["claims"]
    subs = {s["unit"]: s for s in doc["mechanical"]["submissions"]}
    hunks: dict[int, list] = {}
    for h in doc["mechanical"]["hunks"]:
        hunks.setdefault(h["step"], []).append(h)
    steps = {s["k"]: s for s in doc["steps"]}

    def mark(u):
        bits = []
        if u["id"] in tags:
            for t in tags[u["id"]]:
                if t == "claim" and u["id"] in claims:
                    c = claims[u["id"]]
                    bits.append("claim · " + STATUS_LABEL[c["status"]].format(s=c["status_step"]))
                else:
                    bits.append(t)
        if u["kind"] == "command":
            bits.append("action" + (f" · rests on {', '.join(rests[u['id']])}" if rests.get(u["id"]) else ""))
        if u["id"] in subs:
            n = subs[u["id"]]["failing_count"]
            bits.append("submission" + (f" · {n} hidden tests failing" if n is not None else ""))
        return T.OPEN + " ".join([u["id"]] + ([" · ".join(bits)] if bits else [])) + T.CLOSE

    lines = [LEGEND, "", "TASK"] + _unit_lines(doc["units"]["task"], mark) + ["", "CONTEXT"]
    units_map: dict[str, dict[str, Any]] = {u["id"]: {"kind": "task", "step": 0} for u in doc["units"]["task"]}
    for k_str, units in doc["units"]["steps"].items():
        k = int(k_str)
        st = steps[k]
        flags = st.get("flags") or {}
        head = f"{T.mark(f'S{k}')} step {k}"
        if flags.get("workspace_change"):
            head += T.FLAG_TEXT["workspace_change"]
        if flags.get("outcome") in ("tests_fail", "tests_pass"):
            t = (st.get("detail") or {}).get("outcome", {}).get("tests") or [{}]
            head += T.OUTCOME_TEXT[flags["outcome"]].format(tests=", ".join(f"{v} {a}" for a, v in t[-1].items()))
        if flags.get("output_current") in ("stale", "mixed"):
            head += T.FLAG_TEXT[flags["output_current"]]
        if flags.get("noise"):
            head += T.FLAG_TEXT["noise"]
        if k not in labels:
            head += " · not labelled"
        elif labels[k]["status"] == "failed":
            head += " · labels missing (the labelling call failed)"
        lines += ["", head] + _unit_lines(units, mark)
        units_map[f"S{k}"] = {"kind": "step", "step": k}
        for u in units:
            units_map[u["id"]] = {"kind": u["kind"], "step": k}
        for h in hunks.get(k, []):
            who = f"carries out {h['carries_out']}" if h["carries_out"] else \
                f"carries out step {k} (command not known" + (f"; one of {', '.join(h['candidates'])})"
                                                                if h["candidates"] else ")")
            lines.append(f"{T.OPEN}{h['id']} file change · {who}{T.CLOSE} {h['kind']} {h['path']} {h['header']} "
                         f"(+{h['added']} −{h['removed']})")
            units_map[h["id"]] = {"kind": "hunk", "step": k}
    return {"version": VIEW_VERSION, "kind": "units-ledger", "run_id": doc["run_id"],
            "checkpoint_id": doc["checkpoint_id"], "stall_point": doc["stall_point"],
            "fake_labels": doc["fake_labels"], "text": "\n".join(lines).rstrip() + "\n", "units": units_map}


# ---------------------------------------------------------------- scripted (fake) labels for end-to-end checks

_CUE = {"claim": re.compile(r"\b(works|passes|fixed|is correct|succeeded|matches|because|must be|the (?:bug|issue|"
                            r"problem) is)\b", re.I),
        "plan": re.compile(r"\b(I'll|I will|Let's|The fix|Now I'll|Let me (?:fix|rewrite|change|replace|add|"
                           r"implement|update|write|patch|create))", re.I),
        "idea": re.compile(r"\b(maybe|perhaps|could|might|alternatively)\b", re.I),
        "observation": re.compile(r"\b(shows|error|failed|exists|found|returns|output|contains|passed)\b", re.I)}
_FAIL = re.compile(r"(Error|FAIL|failed|Traceback|No such file|not found|still failing)")
_PASS = re.compile(r"\b(\d+ passed|OK)\b")


def _scripted_units_answer(prompt: str, repeat: int) -> dict[str, Any]:
    """Fake labels read only from the prompt (so prefix-only by construction). Malformed on purpose: on the first try
    of every 5th step an unknown id (a failed try), and on every 7th step a known id in a place it is not allowed
    (dropped and logged)."""
    k = int(re.search(r"Label step (\d+), the\s+last step", prompt).group(1))
    log = prompt[prompt.index("<recorded_log>\n") + 15:prompt.rindex("</recorded_log>")]
    block = log[log.rindex(f"<step {k}>"):]
    earlier = log[:log.rindex(f"<step {k}>")]
    units = re.findall(rf"^\[(S{k}\.[rm]\d+)\] (.*)$", block, re.M)
    tags: dict[str, list[str]] = {t: [] for t in TAGS}
    for uid, text in units:
        for t, rx in _CUE.items():
            if rx.search(text):
                tags[t].append(uid)
    frozen_cl = [x for line in re.findall(r"^(?:claim|observation): (.*)$", earlier, re.M)
                 for x in line.split(", ") if x != "none"]
    outs_before = re.findall(r"^\[(S\d+\.[of]\d+)\] ", earlier, re.M)
    commands = re.findall(rf"^\[(S{k}\.c\d+)\] command", block, re.M)
    rests = {}
    for c in commands:
        r = [x for x in (frozen_cl[-1:] + outs_before[-1:]) if x]
        if k == 1:
            r.append("T1")
        rests[c] = r
    status = {}
    ev = re.findall(rf"^\[(S{k}\.[of]\d+)\] ", block, re.M)
    if ev:
        out_text = block[block.index(f"[{ev[0]}]"):]
        claims = re.findall(r"^\[(S\d+\.[rm]\d+)\] \(step \d+\)", prompt[prompt.rindex("Claims so far:"):], re.M) \
            if "Claims so far:" in prompt else []
        claims += tags["claim"]
        if claims and _FAIL.search(out_text):
            status[claims[-1]] = {"verdict": "contradicts", "evidence": ev[0]}
        elif claims and _PASS.search(out_text):
            status[claims[-1]] = {"verdict": "confirms", "evidence": ev[0]}
    if k % 5 == 0 and repeat == 0:
        tags["claim"].append(f"S{k}.r999")
    if k % 7 == 0 and commands and tags["plan"]:
        rests[commands[0]] = rests[commands[0]] + [tags["plan"][0]]
    return {"tags": tags, "rests_on": rests, "claim_status": status}


def scripted_units_labeller(store):
    """The ScriptedLabeller (fake labels, own receipt kind) with an answer for the units step prompt."""
    from .labeller import ScriptedLabeller

    class ScriptedUnitsLabeller(ScriptedLabeller):
        model = "scripted/ledger-units-fake-labels-v1"

        def _respond(self, kind, prompt, repeat):
            if kind == "units-step":
                return _scripted_units_answer(prompt, repeat)
            return super()._respond(kind, prompt, repeat)

    return ScriptedUnitsLabeller(store)
