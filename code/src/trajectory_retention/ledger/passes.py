"""The two Ledger passes over a stall's prefix, with a labeller client (see ``labeller``).

``build_boundaries`` (before screening): steps and mechanical flags; decision tags (exact quotes) with their explicit
back-references; implemented-decision windows; one LLM grouping pass of consecutive windows into approaches; a
separate direct-boundary pass recorded for comparison only. Writes ``boundary.json`` (frozen: an existing different
file is never overwritten) and ``ledger-boundary.json``; a boundary whose labelling or grouping failed is written to
``boundary.failed.json`` instead and never frozen.

``build_full`` (after admission): facts (+ survives rewind to the frozen rewind point), claims (+ contradicted later /
confirmed later / unchecked, from later worker-visible steps only), ideas (+ tried later / never tried), and per
approach a status and a failure-evidence quote. Writes ``ledger.json``.

``build_items`` (flat Ledger, no approach layer): only the decisions and tags passes, with the stall point as the rewind
point (survives-rewind is never asked). Default labeller protocol v2 (``prompts.PROTOCOLS``): a system message, the log
fenced as data, instructions after the log, and an answer check that accepts only a whole reply that is one JSON object.
``build_boundaries`` and ``build_full`` keep the frozen protocol v1.

Rules for every labeller answer:
* A quote must occur verbatim in the step it cites (and in a field its type allows: decisions, claims and ideas only in
  the agent's reasoning, message or command; facts and evidence anywhere). An unresolved quote is dropped and logged
  with its reason, never repaired. Quotes shorter than MIN_QUOTE_CHARS are dropped.
* A status needs evidence from a LATER step of the same prefix; evidence that is not later, or not verbatim, downgrades
  a claim to ``unchecked`` and removes an idea's ``tried_later`` (logged). References must point backwards.
* Items whose spans overlap an earlier-placed item in the same field are dropped (logged), so tags can sit in place.
* A failed or unparseable call leaves its labels missing and the pass status not ``ok``; nothing is filled in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .flags import FLAGS_VERSION, compute_flags
from .prompts import PROMPTS, PROTOCOLS, prompt_record, prompt_sha256, protocol_record  # noqa: F401 (PROMPTS: old tools)
from .render import LABELLER_RENDERERS, labeller_text
from .steps import AGENT_KINDS, STEPS_VERSION, Context, load_context, occurrences, resolve
from .windows import WINDOWS_VERSION, implemented_decision_windows

PASSES_VERSION = "ledger-passes/v1"
LEDGER_VERSION = "ledger/v2.0"
DEFAULT_MAX_CHARS = 300_000  # soft per-call budget: target steps fill at most half; see _chunks and _fit
MIN_QUOTE_CHARS = 8
CLAIM_STATUSES = ("contradicted_later", "confirmed_later", "unchecked")
IDEA_STATUSES = ("tried_later", "never_tried")
APPROACH_STATUSES = ("failed", "abandoned", "worked", "open")
FIELD_RANK = {"reasoning": 0, "message": 1, "command": 2, "output": 3, "format_error": 4, "feedback": 5, "user": 6}


class FrozenBoundaryExists(RuntimeError):
    """A different boundary.json is already frozen for this stall."""


def as_int(x: Any) -> Any:
    """A step number as an int when the answer wrote an integer or an integer-like string; otherwise unchanged."""
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return x


def parse_json(text: str) -> Any:
    """The frozen v1 answer check: the first "{" to the last "}" anywhere in the reply, parsed as JSON. It accepts
    prose (or the agent's own commands) around a JSON fragment."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


_JSON_FENCE = re.compile(r"\A```json[ \t]*\n(.*?)\n?[ \t]*```\Z", re.S)


def parse_json_object_only(text: str) -> dict[str, Any] | None:
    """The v2 answer check (user, 2026-10-22): the whole reply, after stripping surrounding whitespace and at most one
    surrounding ```json fence, must parse as a single JSON object. Anything else (prose around JSON, tool-call markup,
    an array, two objects, a second fence) is a failed try."""
    body = (text or "").strip()
    m = _JSON_FENCE.match(body)
    if m:
        body = m.group(1).strip()
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


ANSWER_CHECKS = {"json_anywhere": parse_json, "whole_json_object": parse_json_object_only}


class _Calls:
    """Runs labeller calls and keeps one record per call (prompt version and hash, receipt id, status, cost).

    ``protocol`` picks the prompts, system message and answer check (``prompts.PROTOCOLS``); "v1" is the frozen
    procedure and its call records are unchanged. ``answer_check`` overrides the protocol's check. Two tries per call
    in every protocol."""

    def __init__(self, client, *, protocol: str = "v1", answer_check: str | None = None):
        self.client = client
        self.protocol = protocol
        self.answer_check = answer_check or PROTOCOLS[protocol]["answer_check"]
        # how the log is rendered for this protocol (chunk boundaries are sized on the v1 render in every protocol)
        self.render = LABELLER_RENDERERS[PROTOCOLS[protocol]["render"]]
        self.records: list[dict[str, Any]] = []

    def messages(self, name: str, text: str, fmt: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
        p = PROTOCOLS[self.protocol]
        template = p["prompts"][name][1]
        if p["system"] is None:  # v1: one user message, prompt then log
            return [{"role": "user", "content": template.format(**(fmt or {})) + "\n\n" + text}]
        return [{"role": "system", "content": p["system"][1]},
                {"role": "user", "content": template.format(**{**(fmt or {}), "log": text})}]

    def run(self, name: str, text: str, *, fmt: Mapping[str, Any] | None = None, chunk=None, valid=lambda p: True,
            note: Mapping[str, Any] | None = None) -> Any:
        version = PROTOCOLS[self.protocol]["prompts"][name][0]
        messages = self.messages(name, text, fmt)
        check = ANSWER_CHECKS[self.answer_check]
        v1 = self.protocol == "v1" and self.answer_check == "json_anywhere"
        extra = {} if v1 else {"protocol": PROTOCOLS[self.protocol]["id"], "answer_check": self.answer_check,
                               "system_chars": sum(len(m["content"]) for m in messages if m["role"] == "system")}
        for repeat in (0, 1):
            rec = {"prompt": name, "prompt_version": version, "prompt_sha256": prompt_sha256(name, self.protocol),
                   "purpose": f"ledger:{name}", "repeat": repeat, "chunk": list(chunk) if chunk else None,
                   "request_chars": sum(len(m["content"]) for m in messages), **extra, **(note or {})}
            try:
                answer = self.client.complete(messages, purpose=f"ledger:{name}", repeat=repeat)
            except Exception as exc:  # recorded, never silently dropped
                self.records.append({**rec, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                return None
            parsed = check(answer.text)
            ok = parsed is not None and valid(parsed)
            status = "ok" if ok else "unparseable" if v1 else "not_one_json_object" if parsed is None else "wrong_shape"
            self.records.append({**rec, "status": status, "record_id": answer.call_record_id,
                                 "cached": answer.cached, "cost_usd": answer.cost_usd, "provider": answer.provider,
                                 "model": answer.model, "finish_reason": answer.finish_reason,
                                 **({} if v1 else {"answer_chars": len(answer.text or "")})})
            if ok:
                return parsed
        return None

    def all_ok(self, name: str | None = None) -> bool:
        latest: dict[Any, str] = {}
        for r in self.records:
            if name is None or r["prompt"] == name:
                latest[(r["prompt"], json.dumps(r["chunk"]))] = r["status"]
        return all(s == "ok" for s in latest.values())


def _chunks(steps: Sequence[Mapping[str, Any]], head, max_chars: int) -> list[tuple[int, int]]:
    """Consecutive target ranges whose full rendering stays under half the call budget (at least one step each).
    The budget is soft: the task, the earlier steps' agent text and (elided) later steps come on top, and every call
    records its request size."""
    out, lo, size = [], 1, 0
    for s in steps:
        n = len(labeller_text([s], [], target=(s["k"], s["k"]), earlier=False))
        if size and size + n > max_chars // 2:
            out.append((lo, s["k"] - 1))
            lo, size = s["k"], 0
        size += n
    if steps:
        out.append((lo, steps[-1]["k"]))
    return out


def _fit(steps, head, target, max_chars, *, later: bool, render=labeller_text) -> tuple[str, dict[str, Any]]:
    text = render(steps, head, target=target, earlier=True, later=later)
    if later and len(text) > max_chars:
        text = render(steps, head, target=target, earlier=True, later=True, elide_later_outputs=True)
        return text, {"later_outputs_elided": True}
    return text, {}


def _whole(steps, head, max_chars) -> tuple[str, dict[str, Any]]:
    """The whole prefix for one call; outputs are elided (and the call record says so) when it is over budget."""
    text = labeller_text(steps, head, target=(1, len(steps)), earlier=False)
    if len(text) <= max_chars:
        return text, {}
    return labeller_text(steps, head, target=(1, len(steps)), earlier=False, elide_outputs=True), {"outputs_elided": True}


class _Placer:
    """Resolves quotes against the context and keeps non-overlapping spans per field."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.spans: dict[str, list[tuple[int, int, str]]] = {}
        self.dropped: list[dict[str, Any]] = []

    def drop(self, pass_: str, raw: Mapping[str, Any], reason: str) -> None:
        self.dropped.append({"pass": pass_, "step": raw.get("step"), "type": raw.get("type"),
                             "quote": raw.get("quote"), "reason": reason})

    def locate(self, k: Any, quote: Any, kinds: tuple[str, ...] | None) -> tuple[Any, str | None]:
        n = len(self.ctx.steps)
        if type(k) is not int or not 1 <= k <= n:
            return None, "step_out_of_range"
        if not isinstance(quote, str) or len(quote.strip()) < MIN_QUOTE_CHARS:
            return None, "quote_too_short"
        hit = resolve(self.ctx.step(k), quote, kinds=kinds)
        if hit is None:
            if kinds and resolve(self.ctx.step(k), quote) is not None:
                return None, "field_not_allowed_for_type"
            if any(resolve(s, quote) for s in self.ctx.steps if s.k != k):
                return None, "not_in_cited_step"
            return None, "not_verbatim"
        return hit, None

    def span_record(self, k: int, hit) -> dict[str, Any]:
        f, s, e = hit
        raw = f.raw_span(s, e)
        return {"step": k, "field": f.id, "field_kind": f.kind, "quote": f.text[s:e],
                "span": {"field_id": f.id, "start": s, "end": e},
                "source": {**f.source, "start": s, "end": e},
                "h": {**f.h, "raw_start": raw[0], "raw_end": raw[1]} if raw else {**f.h, "raw_start": None,
                                                                                   "raw_end": None},
                "occurrences_in_step": occurrences(self.ctx.step(k), f.text[s:e])}

    def place(self, rec: dict[str, Any], key: str) -> str | None:
        fid, s, e = rec["span"]["field_id"], rec["span"]["start"], rec["span"]["end"]
        for a, b, other in self.spans.get(fid, ()):
            if s < b and a < e:
                return other
        self.spans.setdefault(fid, []).append((s, e, key))
        return None


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def _context(store, run_id, checkpoint_id, client) -> Context:
    ctx = load_context(store, run_id, checkpoint_id)
    compute_flags(ctx)
    bind = getattr(client, "bind_context", None)
    if bind is not None:
        bind(ctx)
    return ctx


def _labeller_record(client, calls: _Calls) -> dict[str, Any]:
    desc = client.describe() if hasattr(client, "describe") else {"model": getattr(client, "model", None)}
    record = prompt_record(calls.protocol)
    out = {"model": desc.get("model"), "provider": desc.get("provider"), "settings": desc.get("settings"),
           "fake_labels": bool(desc.get("fake_labels")), "replay_only": desc.get("replay_only"),
           "prompt_sha256": {n: r["sha256"] for n, r in record.items()}, "prompts": record,
           "passes": calls.records, "cost_usd": sum(r.get("cost_usd") or 0.0 for r in calls.records),
           "cost_missing_calls": sum(1 for r in calls.records if r.get("status") == "ok" and r.get("cost_usd") is None)}
    if calls.protocol != "v1" or calls.answer_check != "json_anywhere":
        out["protocol"] = {**protocol_record(calls.protocol), "answer_check": calls.answer_check}
    return out


# ---------------------------------------------------------------- item passes (shared by every build)

def _decision_items(steps, head, calls: _Calls, placer: _Placer, max_chars: int, *,
                    pass_name: str = "boundary") -> list[dict[str, Any]]:
    """Decision tags with their explicit back-references, numbered D1.. in context order (moved unchanged from
    ``build_boundaries``)."""
    decisions: list[dict[str, Any]] = []
    for lo, hi in _chunks(steps, head, max_chars):
        text, note = _fit(steps, head, (lo, hi), max_chars, later=False, render=calls.render)
        parsed = calls.run("decisions", text, chunk=(lo, hi), note=note,
                           valid=lambda p: isinstance(p.get("decisions"), list))
        for raw in (parsed or {}).get("decisions", []):
            if not isinstance(raw, dict):
                continue
            raw = {**raw, "type": "decision"}
            k = raw["step"] = as_int(raw.get("step"))
            if type(k) is int and not lo <= k <= hi:
                placer.drop("decisions", raw, "step_outside_target")
                continue
            hit, why = placer.locate(k, raw.get("quote"), AGENT_KINDS)
            if hit is None:
                placer.drop("decisions", raw, why)
                continue
            rec = {**placer.span_record(k, hit), "type": "decision", "pass": pass_name}
            refs = []
            for t in [as_int(x) for x in raw.get("refers_to") or []]:
                if type(t) is int and 1 <= t < k:
                    refs.append(t)
                else:
                    placer.drop("decisions", {**raw, "quote": f"refers_to {t!r}"}, "reference_not_earlier")
            rec["refers_to"] = sorted(set(refs))
            other = placer.place(rec, f"decision@{k}:{rec['span']['start']}")
            if other:
                placer.drop("decisions", raw, f"overlaps {other}")
                continue
            decisions.append(rec)
    decisions.sort(key=lambda d: (d["step"], FIELD_RANK[d["field_kind"]], d["span"]["start"]))
    for m, d in enumerate(decisions, 1):
        d["bid"] = f"D{m}"
    return decisions


def _tag_items(steps, head, calls: _Calls, placer: _Placer, max_chars: int, rewind: int, *,
               pass_name: str = "full") -> list[dict[str, Any]]:
    """Facts (+ survives rewind for facts after ``rewind``), claims (+ later status), ideas (+ tried later); moved
    unchanged from ``build_full``."""
    n = len(steps)
    found: list[dict[str, Any]] = []
    for lo, hi in _chunks(steps, head, max_chars):
        text, note = _fit(steps, head, (lo, hi), max_chars, later=True, render=calls.render)
        parsed = calls.run("tags", text, chunk=(lo, hi), note=note,
                           fmt={"rewind_point": rewind, "first_undone": rewind + 1},
                           valid=lambda p: isinstance(p.get("items"), list))
        for raw in (parsed or {}).get("items", []):
            if not isinstance(raw, dict):
                continue
            typ, k = raw.get("type"), as_int(raw.get("step"))
            raw["step"] = k
            if typ not in ("fact", "claim", "idea"):
                placer.drop("tags", raw, "unknown_type")
                continue
            if type(k) is int and not lo <= k <= hi:
                placer.drop("tags", raw, "step_outside_target")
                continue
            hit, why = placer.locate(k, raw.get("quote"), None if typ == "fact" else AGENT_KINDS)
            if hit is None:
                placer.drop("tags", raw, why)
                continue
            rec = {**placer.span_record(k, hit), "type": typ, "pass": pass_name}
            status = raw.get("status")
            if typ == "fact":
                surv = raw.get("survives_rewind")
                rec["survives_rewind"] = surv if (k > rewind and isinstance(surv, bool)) else None
                if k > rewind and not isinstance(surv, bool):
                    placer.drop("tags", raw, "survives_rewind_missing (kept without it)")
            elif typ == "claim":
                if status not in CLAIM_STATUSES:
                    placer.drop("tags", raw, f"claim_status_invalid {status!r} (kept as unchecked)")
                    status = "unchecked"
                if status != "unchecked":
                    ev, why = _evidence(placer, raw, k, n)
                    if ev is None:
                        placer.drop("tags", raw, f"status_evidence_{why} (kept as unchecked)")
                        status = "unchecked"
                    else:
                        rec["status_evidence"] = ev
                rec["claim_status"] = status
            else:
                if status not in IDEA_STATUSES:
                    placer.drop("tags", raw, f"idea_status_invalid {status!r} (kept without status)")
                    status = None
                if status == "tried_later":
                    e = as_int(raw.get("evidence_step"))
                    if type(e) is int and k < e <= n:
                        rec["status_evidence"] = {"step": e}
                    else:
                        placer.drop("tags", raw, "status_evidence_not_later (kept without status)")
                        status = None
                rec["idea_status"] = status
            other = placer.place(rec, f"{typ}@{k}:{rec['span']['start']}")
            if other:
                placer.drop("tags", raw, f"overlaps {other}")
                continue
            found.append(rec)
    return found


# ---------------------------------------------------------------- boundary pass

def build_boundaries(store, run_id: str, checkpoint_id: str, client, *, out_dir: str | Path | None = None,
                     max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    ctx = _context(store, run_id, checkpoint_id, client)
    steps = [s.as_record() for s in ctx.steps]
    head = [f.as_record() for f in ctx.head]
    n = len(steps)
    if n == 0:
        raise ValueError("no steps before the cut: nothing to label")
    calls, placer = _Calls(client), _Placer(ctx)
    decisions = _decision_items(steps, head, calls, placer, max_chars)
    refers: dict[int, set[int]] = {}
    for d in decisions:
        refers.setdefault(d["step"], set()).update(d["refers_to"])
    commitments = [s["k"] for s in steps if s["workspace_change"] or s["submission"]]
    windows, wmeta = implemented_decision_windows(n, [d["step"] for d in decisions], commitments, refers)
    for w in windows:
        w["anchor_decision"] = next((d["bid"] for d in decisions if d["step"] == w["anchor_step"]), None)
    # grouping pass
    ids = [w["id"] for w in windows]
    grouping: dict[str, Any] = {"status": "not_needed" if len(windows) <= 1 else None}
    groups = [ids]
    if len(windows) > 1:
        dq = {d["bid"]: d for d in decisions}
        listing = "\n".join(f"{w['id']} steps {w['first_step']}–{w['last_step']}"
                            + (f" · starts with the decision \"{dq[w['anchor_decision']]['quote']}\" (step "
                               f"{w['anchor_step']})" if w["anchor_decision"] and w["id"] != "W1" else "")
                            for w in windows)
        full, note = _whole(steps, head, max_chars)
        parsed = calls.run("grouping", "WINDOWS\n" + listing + "\n\n" + full, note=note,
                           valid=lambda p: [w for g in p.get("approaches") or [] for w in (g if isinstance(g, list) else [None])] == ids
                           and all(isinstance(g, list) and g for g in p.get("approaches") or []))
        if parsed is None:
            grouping = {"status": "failed"}
            groups = [[i] for i in ids]
        else:
            grouping = {"status": "ok"}
            groups = parsed["approaches"]
    by_id = {w["id"]: w for w in windows}
    approaches = []
    for m, g in enumerate(groups, 1):
        ws = [by_id[i] for i in g]
        approaches.append({"id": f"A{m}", "windows": g, "first_step": ws[0]["first_step"], "last_step": ws[-1]["last_step"],
                           "start_decision": ws[0]["anchor_decision"]})
    current = approaches[-1]
    rewind_point = current["first_step"] - 1
    # direct-boundary pass (comparison only)
    full, note = _whole(steps, head, max_chars)
    parsed = calls.run("direct_boundary", full, note=note, valid=lambda p: type(as_int(p.get("first_step"))) is int)
    direct: dict[str, Any] = {"status": "failed" if parsed is None else "ok"}
    if parsed is not None:
        fs = as_int(parsed["first_step"])
        direct.update({"first_step": fs if 1 <= fs <= n else None, "raw_first_step": fs})
        hit, why = placer.locate(fs, parsed.get("quote"), AGENT_KINDS)
        direct["quote"] = placer.span_record(fs, hit) if hit else None
        direct["quote_drop_reason"] = why
        if direct["first_step"] is not None:
            direct["rewind_point"] = direct["first_step"] - 1
            direct["agrees_with_primary"] = direct["first_step"] == current["first_step"]
            direct["difference_steps"] = direct["first_step"] - current["first_step"]
    status = "ok" if calls.all_ok("decisions") and grouping["status"] in ("ok", "not_needed") else \
        "labelling_failed" if not calls.all_ok("decisions") else "grouping_failed"
    labeller = _labeller_record(client, calls)
    labeller["evidence_cutoff_event_sequence"] = ctx.cut
    boundary = {
        "version": LEDGER_VERSION, "kind": "boundary", "run_id": run_id, "checkpoint_id": checkpoint_id,
        "stall_point": n, "status": status,
        "current_approach_id": current["id"], "current_approach_first_step": current["first_step"],
        "rewind_point": rewind_point, "first_approach_stall": rewind_point == 0,
        "approaches": [{k: a[k] for k in ("id", "first_step", "last_step", "windows")} for a in approaches],
        "windows": [{k: w[k] for k in ("id", "first_step", "last_step", "anchor_step", "backfill")} for w in windows],
        "grouping": grouping, "direct_boundary": {k: v for k, v in direct.items() if k != "quote"} | (
            {"quote": direct["quote"]["quote"], "quote_step": direct["quote"]["step"]} if direct.get("quote") else {}),
        "context_fingerprint": ctx.fingerprint(), "cut_event_sequence": ctx.cut,
        "labeller": {k: labeller[k] for k in ("model", "provider", "fake_labels", "prompt_sha256")},
        "fake_labels": labeller["fake_labels"],
        "versions": {"steps": STEPS_VERSION, "flags": FLAGS_VERSION, "windows": WINDOWS_VERSION,
                     "passes": PASSES_VERSION},
    }
    partial = {"version": LEDGER_VERSION, "kind": "ledger-boundary", "run_id": run_id, "checkpoint_id": checkpoint_id,
               "stall_point": n, "context": ctx.as_record(), "head": head, "steps": steps, "decisions": decisions,
               "windows": windows, "windows_meta": wmeta, "approaches": approaches, "direct_boundary": direct,
               "boundary": boundary, "dropped": placer.dropped, "labeller": labeller}
    if out_dir is not None:
        out = Path(out_dir)
        path = out / "boundary.json"
        if status != "ok":  # recorded, never frozen: screening needs an ok boundary.json
            _write(out / "boundary.failed.json", boundary)
        elif path.exists():
            old = json.loads(path.read_text())
            if {k: v for k, v in old.items() if k != "labeller"} != {k: v for k, v in boundary.items() if k != "labeller"}:
                raise FrozenBoundaryExists(f"{path} is frozen and differs from the rebuilt boundary")
        else:
            _write(path, boundary)
        _write(out / ("ledger-boundary.json" if status == "ok" else "ledger-boundary.failed.json"), partial)
    return partial


# ---------------------------------------------------------------- full pass

def build_full(store, run_id: str, checkpoint_id: str, client, *, boundary_dir: str | Path,
               out_dir: str | Path | None = None, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    bdir = Path(boundary_dir)
    boundary = json.loads((bdir / "boundary.json").read_text())
    partial = json.loads((bdir / "ledger-boundary.json").read_text())
    ctx = _context(store, run_id, checkpoint_id, client)
    if ctx.fingerprint() != boundary["context_fingerprint"]:
        raise ValueError("the context at the cut differs from the one the boundary was built on")
    steps = [s.as_record() for s in ctx.steps]
    head = [f.as_record() for f in ctx.head]
    n = len(steps)
    rewind = boundary["rewind_point"]
    calls, placer = _Calls(client), _Placer(ctx)
    for d in partial["decisions"]:
        placer.place(d, d["bid"])
    found = _tag_items(steps, head, calls, placer, max_chars, rewind)
    # per-approach status and failure evidence
    approaches = [dict(a) for a in partial["approaches"]]
    listing = "\n".join(f"{a['id']}: steps {a['first_step']}–{a['last_step']}" for a in approaches)
    full, note = _whole(steps, head, max_chars)
    parsed = calls.run("approach_status", full, fmt={"approach_list": listing}, note=note,
                       valid=lambda p: isinstance(p.get("approaches"), list))
    answers = {a.get("id"): a for a in (parsed or {}).get("approaches", []) if isinstance(a, dict)}
    for a in approaches:
        ans = answers.get(a["id"])
        a["status"] = ans.get("status") if ans and ans.get("status") in APPROACH_STATUSES else None
        a["failure_quote"], a["failure_evidence"] = None, None
        if ans is None:
            placer.drop("approach_status", {"step": a["first_step"], "quote": a["id"]}, "approach_missing")
            continue
        if a["status"] is None:
            placer.drop("approach_status", {"step": a["first_step"], "quote": a["id"]},
                        f"status_invalid {ans.get('status')!r}")
        fq, fs = ans.get("failure_quote"), as_int(ans.get("failure_step"))
        if fq:
            if type(fs) is int and fs < a["first_step"]:
                placer.drop("approach_status", {"step": fs, "quote": fq}, "failure_evidence_before_approach")
                continue
            hit, why = placer.locate(fs, fq, None)
            if hit is None:
                placer.drop("approach_status", {"step": fs, "quote": fq}, f"failure_quote_{why}")
                continue
            a["failure_evidence"] = placer.span_record(fs, hit)
            a["failure_quote"] = a["failure_evidence"]["quote"]
    # assemble
    items = sorted([{**d, "type": "decision"} for d in partial["decisions"]] + found,
                   key=lambda i: (i["step"], FIELD_RANK[i["field_kind"]], int(i["field"].split(".f")[1]),
                                  i["span"]["start"]))
    for m, i in enumerate(items, 1):
        i["id"] = f"I{m}"
    by_bid = {i["bid"]: i["id"] for i in items if i.get("bid")}
    for a in approaches:
        a["start_item"] = by_bid.get(a.get("start_decision"))
    owner = {k: a["id"] for a in approaches for k in range(a["first_step"], a["last_step"] + 1)}
    for s in steps:
        s["flags"]["approach"] = owner.get(s["k"])
    labeller = _labeller_record(client, calls)
    labeller["evidence_cutoff_event_sequence"] = ctx.cut
    labeller["boundary_passes"] = partial["labeller"]["passes"]
    status = "ok" if calls.all_ok() and boundary["status"] == "ok" else "incomplete"
    ledger = {
        "version": LEDGER_VERSION, "kind": "ledger", "status": status, "run_id": run_id,
        "checkpoint_id": checkpoint_id, "stall_point": n, "cut_event_sequence": ctx.cut,
        "context_fingerprint": ctx.fingerprint(),
        "boundary": {k: boundary[k] for k in ("current_approach_id", "current_approach_first_step", "rewind_point",
                                              "first_approach_stall", "status")},
        "direct_boundary": boundary["direct_boundary"], "grouping": boundary["grouping"],
        "head": head, "steps": steps, "approaches": approaches, "windows": partial["windows"], "items": items,
        "labeller": labeller, "fake_labels": labeller["fake_labels"],
        "dropped": partial["dropped"] + placer.dropped, "checks": ctx.checks,
        "versions": {"steps": STEPS_VERSION, "flags": FLAGS_VERSION, "windows": WINDOWS_VERSION,
                     "passes": PASSES_VERSION},
    }
    if out_dir is not None:
        out = Path(out_dir)
        _write(out / "ledger.json", ledger)
        _write(out / "labeller_calls.json", {"boundary": partial["labeller"]["passes"], "full": calls.records,
                                             "fake_labels": labeller["fake_labels"]})
    return ledger


def _evidence(placer: _Placer, raw: Mapping[str, Any], k: int, n: int) -> tuple[dict[str, Any] | None, str | None]:
    e = as_int(raw.get("evidence_step"))
    if type(e) is not int or not k < e <= n:
        return None, "not_later"
    hit, why = placer.locate(e, raw.get("evidence_quote"), None)
    if hit is None:
        return None, why
    return placer.span_record(e, hit), None


# ---------------------------------------------------------------- flat item pass (protocol v2)

FLAT_VERSION = "ledger-flat-items/v1"


def build_items(store, run_id: str, checkpoint_id: str, client, *, protocol: str = "v2.1",
                answer_check: str | None = None, out_dir: str | Path | None = None,
                max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    """The flat Ledger's items (no approach layer): the decisions pass and the tags pass only, with the given labeller
    protocol (default v2.1). The rewind point is the stall point, so survives-rewind is never asked and stays null.
    Items are numbered as in ``build_full``. Writes ``ledger-items.json`` and ``labeller_calls.json`` when ``out_dir``
    is given (``ledger-items.failed.json`` when a call failed: its labels are missing, nothing is filled in)."""
    ctx = _context(store, run_id, checkpoint_id, client)
    steps = [s.as_record() for s in ctx.steps]
    head = [f.as_record() for f in ctx.head]
    n = len(steps)
    if n == 0:
        raise ValueError("no steps before the cut: nothing to label")
    calls, placer = _Calls(client, protocol=protocol, answer_check=answer_check), _Placer(ctx)
    decisions = _decision_items(steps, head, calls, placer, max_chars, pass_name="decisions")
    found = _tag_items(steps, head, calls, placer, max_chars, n, pass_name="tags")
    items = sorted(decisions + found, key=lambda i: (i["step"], FIELD_RANK[i["field_kind"]],
                                                     int(i["field"].split(".f")[1]), i["span"]["start"]))
    for m, i in enumerate(items, 1):
        i["id"] = f"I{m}"
    labeller = _labeller_record(client, calls)
    labeller["evidence_cutoff_event_sequence"] = ctx.cut
    status = "ok" if calls.all_ok() else "incomplete"
    doc = {"version": FLAT_VERSION, "kind": "ledger-items", "status": status, "run_id": run_id,
           "checkpoint_id": checkpoint_id, "stall_point": n, "cut_event_sequence": ctx.cut,
           "context_fingerprint": ctx.fingerprint(), "rewind_point": n,
           "passes_run": ["decisions", "tags"], "head": head, "steps": steps, "approaches": [], "items": items,
           "labeller": labeller, "fake_labels": labeller["fake_labels"], "dropped": placer.dropped,
           "checks": ctx.checks, "versions": {"steps": STEPS_VERSION, "flags": FLAGS_VERSION,
                                              "passes": PASSES_VERSION, "flat": FLAT_VERSION}}
    if out_dir is not None:
        out = Path(out_dir)
        _write(out / ("ledger-items.json" if status == "ok" else "ledger-items.failed.json"), doc)
        _write(out / "labeller_calls.json", {"items": calls.records, "fake_labels": labeller["fake_labels"]})
    return doc
