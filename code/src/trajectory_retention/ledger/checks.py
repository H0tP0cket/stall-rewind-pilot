"""End-to-end proofs over a built Ledger and its renderings (used by ``ledger.e2e``; read-only on the source store).

Failure modes these checks look for, written before the code:
  1. a quote that is not an exact span of the recorded context (internal offsets wrong, the context re-read from the
     store differs, or the raw context field at the recorded message/path does not decode to the quote, including the
     duplicated reasoning copies);
  2. content from after the cut (later responses, outputs or feedback) reaching a view, the recipe or a kept label;
  3. private evaluator data (grade records, submission verdicts, detector output, annotations) reaching any of them;
  4. the arm 2 and arm 3 views showing different steps or different step content;
  5. a unit ID shown in a view text that ``units`` does not map, or a mapped unit that the text never shows;
  6. a recipe over its cap, a cut that was not logged, a recipe line that is not an item's exact quote;
  7. a kept-item label that still carries a unit ID.
Checks 2 and 3 are also run on a copy of a view with a planted leak, which they must catch (negative controls).
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from . import templates as T
from .render import label_text
from .steps import full_messages, load_context

LEAK_MIN_FUTURE = 30
LEAK_MIN_PRIVATE = 16


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _decode_raw(raw: str, decoder: str) -> str:
    return raw if decoder == "raw" else json.loads('"' + raw + '"')


def _get_path(message: Mapping[str, Any], path: str) -> Any:
    cur: Any = message
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        cur = cur[int(part[1:-1])] if part.startswith("[") else cur[part]
    return cur


def check_quotes(store, ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Every quote (items, status evidence, failure evidence) is an exact span of the context re-read from the store."""
    ctx = load_context(store, ledger["run_id"], ledger["checkpoint_id"])
    cp = next(c for c in store.list_checkpoints(ledger["run_id"]) if c.checkpoint_id == ledger["checkpoint_id"])
    messages, _ = full_messages(store, store.get_json(cp.state.history))  # the full record the Ledger was built on
    fields = {f.id: f for s in ctx.steps for f in s.fields}
    quotes = []
    for i in ledger["items"]:
        quotes.append((i["id"], i))
        if isinstance(i.get("status_evidence"), Mapping) and "span" in i["status_evidence"]:
            quotes.append((i["id"] + ":evidence", i["status_evidence"]))
    for a in ledger["approaches"]:
        if a.get("failure_evidence"):
            quotes.append((a["id"] + ":failure", a["failure_evidence"]))
    bad = []
    raw_checked = raw_unmapped = 0
    for qid, q in quotes:
        sp = q["span"]
        f = fields.get(sp["field_id"])
        step = ledger["steps"][q["step"] - 1]
        mine = next((x for x in step["fields"] if x["id"] == sp["field_id"]), None)
        if f is None or mine is None or mine["text"] != f.text:
            bad.append((qid, "field differs on re-read"))
            continue
        if f.text[sp["start"]:sp["end"]] != q["quote"]:
            bad.append((qid, "offsets do not give the quote"))
            continue
        h = q["h"]
        if h.get("raw_start") is None:
            raw_unmapped += 1
            continue
        for path in [h["path"]] + list(h.get("also") or []):
            raw = _get_path(messages[h["message"]], path)
            if not isinstance(raw, str) or _decode_raw(raw[h["raw_start"]:h["raw_end"]],
                                                       q["source"]["decoder"]) != q["quote"]:
                bad.append((qid, f"raw context at message {h['message']} {path} does not give the quote"))
        raw_checked += 1
    return {"pass": not bad and ctx.fingerprint() == ledger["context_fingerprint"], "quotes": len(quotes),
            "raw_spans_checked": raw_checked, "raw_unmapped": raw_unmapped, "failures": bad[:20],
            "context_fingerprint_matches": ctx.fingerprint() == ledger["context_fingerprint"]}


def leak_sources(store, ledger: Mapping[str, Any]) -> dict[str, set[str]]:
    """Distinctive strings that must never reach agent-facing text: worker-visible content after the cut, and private
    evaluator data from the whole run. Strings already present in the prefix's context are excluded."""
    run_id, cut = ledger["run_id"], ledger["cut_event_sequence"]
    prefix = "\n".join([f["text"] for f in ledger["head"]] + [f["text"] for s in ledger["steps"] for f in s["fields"]])
    future, private = set(), set()
    for e in store.iter_events(run_id):
        texts: list[str] = []
        if e.sequence > cut and e.kind == "model_response" and e.output_ref is not None:
            body = store.get_json(e.output_ref)
            texts += [body.get("content") or "", body.get("reasoning_content") or ""]
            texts += [str(c.get("arguments")) for c in body.get("tool_calls") or ()]
        elif e.sequence > cut and e.delivered_ref is not None and e.kind in (
                "observation_delivery", "native_submission_feedback", "task_disclosure", "format_error"):
            texts.append(store.get_bytes(e.delivered_ref).decode("utf-8", "replace"))
        for t in texts:
            for line in t.splitlines():
                line = line.strip().strip("\"',").strip()  # quoting in a later JSON argument is not new content
                if len(line) >= LEAK_MIN_FUTURE and line not in prefix:
                    future.add(line)
        if str(e.visibility) != "worker_visible" and e.kind in ("grade", "lifecycle", "detector", "annotation",
                                                                 "terminal", "evaluation", "private_evaluator"):
            payload = dict(e.payload or {})
            if e.kind == "grade" and isinstance(payload.get("evaluation_ref"), Mapping):
                from ..contracts import ArtifactRef, from_jsonable
                payload["evaluation"] = store.get_json(from_jsonable(ArtifactRef, payload["evaluation_ref"]))
            for s in _strings(payload):
                s = s.strip()
                if len(s) >= LEAK_MIN_PRIVATE and s not in prefix and not re.fullmatch(r"[0-9a-f]{16,}|[\w.:/-]*\d{6,}[\w.:/-]*", s):
                    private.add(s)
    for _, rec in store.records("annotation", run_id=run_id):
        for s in _strings(rec):
            if len(s) >= LEAK_MIN_PRIVATE and s not in prefix and not re.fullmatch(r"[0-9a-f]{16,}", s):
                private.add(s)
    return {"future": future, "private": private}


def check_leaks(texts: Mapping[str, str], sources: Mapping[str, set[str]]) -> dict[str, Any]:
    hits = {name: {kind: sorted(s for s in pool if s in text)[:5] for kind, pool in sources.items()}
            for name, text in texts.items()}
    found = {n: h for n, h in hits.items() if any(h.values())}
    return {"pass": not found, "texts": sorted(texts), "future_strings": len(sources["future"]),
            "private_strings": len(sources["private"]), "hits": found}


_STEP_SPLIT = re.compile(r"^⟦S(\d+)⟧ step \d+.*$", re.M)
_ITEM_TAG = re.compile(r"⟦/?I\d+(?: [^⟧]*)?⟧")


def _step_bodies(text: str) -> dict[int, str]:
    body = text.split("\n" + T.CONTEXT_HEADER + "\n", 1)[1]
    parts = _STEP_SPLIT.split(body)
    return {int(parts[i]): _ITEM_TAG.sub("", parts[i + 1]).strip("\n") for i in range(1, len(parts), 2)}


def check_views(ledger: Mapping[str, Any], v2: Mapping[str, Any], v3: Mapping[str, Any]) -> dict[str, Any]:
    n = ledger["stall_point"]
    b2, b3 = _step_bodies(v2["text"]), _step_bodies(v3["text"])
    same_steps = sorted(b2) == sorted(b3) == list(range(1, n + 1))
    differ = [k for k in b2 if b2.get(k) != b3.get(k)]
    coverage = {}
    for name, v in (("ledger", v2), ("context", v3)):
        shown = set(re.findall(T.UNIT_RX, v["text"]))
        coverage[name] = {"ids_in_text": len(shown), "units": len(v["units"]),
                          "unmapped_ids": sorted(shown - set(v["units"]))[:10],
                          "units_not_shown": sorted(set(v["units"]) - shown)[:10]}
    ok_cov = all(not c["unmapped_ids"] and not c["units_not_shown"] for c in coverage.values())
    labels_in_context = bool(_ITEM_TAG.search(v3["text"])) or any(u["kind"] in ("item", "approach")
                                                                    for u in v3["units"].values())
    return {"pass": same_steps and not differ and ok_cov and not labels_in_context,
            "same_steps": same_steps, "steps": n, "steps_with_different_content": differ[:10],
            "units_cover_text": coverage, "context_view_has_labels": labels_in_context}


def check_recipe(ledger: Mapping[str, Any], recipe: Mapping[str, Any]) -> dict[str, Any]:
    quotes = {i["quote"] for i in ledger["items"]} | {a["failure_quote"] for a in ledger["approaches"]
                                                      if a.get("failure_quote")}
    lines = [l for l in recipe["text"].splitlines() if '"' in l]
    inexact = [l for l in lines if not any(f'"{q}"' in l for q in quotes)]
    cut_expected = len(recipe["entries_before_cap"]) > len(lines)
    return {"pass": recipe["tokens"] <= recipe["cap_tokens"] and (not cut_expected or bool(recipe["cut_log"]))
            and (not inexact or any(c["reason"] == "truncated_to_cap" for c in recipe["cut_log"])),
            "tokens": recipe["tokens"], "cap": recipe["cap_tokens"], "entries_before_cap": len(recipe["entries_before_cap"]),
            "lines_kept": len(lines), "cuts_logged": len(recipe["cut_log"]), "inexact_lines": inexact[:5]}


def check_labels(ledger: Mapping[str, Any], units: Iterable[str]) -> dict[str, Any]:
    bad = []
    n = 0
    for u in units:
        text = label_text(ledger, u)
        n += 1
        if T.OPEN in text and re.search(T.UNIT_RX, text):
            bad.append(u)
    return {"pass": not bad, "labels": n, "with_unit_ids": bad[:10]}


def no_stuck_wording(texts: Mapping[str, str]) -> dict[str, Any]:
    """Templates and prompts never say 'stuck' (the context itself may contain the word)."""
    from . import prompts
    own = [getattr(T, k) for k in dir(T) if k.isupper() and isinstance(getattr(T, k), str)]
    own += [str(v) for k in dir(T) if k.isupper() and isinstance(getattr(T, k), dict) for v in getattr(T, k).values()]
    own += [t for _, t in prompts.PROMPTS.values()]
    hits = [s[:60] for s in own if re.search(r"\bstuck\b", s, re.I)]
    return {"pass": not hits, "hits": hits}
