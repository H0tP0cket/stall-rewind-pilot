"""T2.1 classifier passes: descriptive item labels (A) and explicit pairwise relations (B).

Pass A labels each item from its own source event (item + nearby text of the
same event; tool output also gets its producing command). Pass B asks, for a
bounded set of (earlier, later) same-session pairs, whether the later wording
explicitly connects to the earlier item. Candidate retrieval is a logged,
deterministic cost bound (reference-cue gate, shared tags/words, recency); it
is not evidence of a relation. Every question allows a negative or ambiguous
answer. No question asks whether material helps recovery.

Prefix safety: pass-A labels use only the item's own event; pass-B candidates
are items available no later than the later item's cut in the same session.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

from .jev import BudgetExceeded, JevClient, RequestBlocked
from .questions21 import QUESTIONS
from .schema import UNKNOWN, Label, Link

LABEL21_VERSION = "t2.1-label/v1"
BATCH = 15
CONTEXT_CHARS = 4000
MAX_PAIR_CANDIDATES = 8
RECENT_ALWAYS = 2
PROSE = {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}
APPROACH_ROLES = {"explanation_or_hypothesis", "strategy_or_design_choice", "intended_action", "unresolved_question", "mixed"}
LATER_ROLES = APPROACH_ROLES | {"observation_report", "factual_assertion", "retrospective_takeaway"}
# Generic English wording that can make an explicit back-reference (retrieval gate, audited for misses).
REFERENCE_CUE = re.compile(
    r"\b(this|that|these|those|it|the (?:idea|hypothesis|theory|approach|plan|fix|change|issue|bug|problem)|"
    r"instead|still|again|already|earlier|previous|before|now|test|check|verify|confirm\w*|wrong|works?|worked|"
    r"didn't|doesn't|does not|did not|fails?|failed|revert\w*|drop|abandon\w*|switch\w*|actually|so|because)\b",
    re.IGNORECASE)
_STOP = set("that this with from have will would should could there their then than when what which while where "
            "into also only just like need needs make makes each some more most such very about after before because "
            "does done else here them they were been being file files code test tests function value".split())


def _choice_label(answer: dict[str, Any], threshold: float, method: str) -> Label:
    probs = answer.get("probabilities") or {}
    if not probs:
        return Label(UNKNOWN, None, method, answer.get("confidence"))
    top, p = max(probs.items(), key=lambda kv: kv[1])
    return Label(top if p >= threshold else UNKNOWN, probs, method, answer.get("confidence"))


def _context(decoded: dict[tuple[int, str], str], batch: Sequence) -> str:
    text = decoded.get((batch[0].seq, batch[0].span.field))
    if text is None:
        return ""
    start = max(0, min(i.span.start for i in batch) - 500)
    end = min(len(text), max(i.span.end for i in batch) + 200)
    return text[start:end][:CONTEXT_CHARS]


def _ask_with_fallback(client: JevClient, build: Callable[[bool, list], tuple[dict, dict]], batch: list, purpose: str,
                       repeat: int, stats: dict, lock: threading.Lock):
    """Ask for a batch; if the firewall blocks it, retry each item with minimal context (separately labelled)."""
    try:
        state, questions = build(False, batch)
        return [(batch, client.ask(state, questions, purpose=purpose, repeat=repeat), False)]
    except RequestBlocked:
        out = []
        for item in batch:
            try:
                state, questions = build(True, [item])
                out.append(([item], client.ask(state, questions, purpose=purpose + ":minimal-context", repeat=repeat), True))
            except RequestBlocked:
                with lock:
                    stats["blocked_items"] += 1
        return out


def pass_a(items: Sequence, decoded: dict[tuple[int, str], str], commands: dict[str, str], client: JevClient, *,
           workers: int = 12, repeat: int = 0, on_error: Callable[[str], None] = print,
           label_prefix: str = "") -> dict[str, Any]:
    base_method = f"jev:{client.model}:{LABEL21_VERSION}"
    groups: dict[tuple[int, str, str], list] = {}
    for item in items:
        if item.source_kind == "worker_action":
            item.labels[label_prefix + "role"] = Label("action", None, "deterministic:source_kind")
            continue
        if item.source_kind == "format_error":
            item.labels[label_prefix + "role"] = Label("harness_message", None, "deterministic:source_kind")
            continue
        if item.source_kind == "observation":
            item.labels[label_prefix + "role"] = Label("tool_output", None, "deterministic:source_kind")
        kind = "obs" if item.source_kind == "observation" else "prose"
        groups.setdefault((item.seq, item.span.field, kind), []).append(item)
    jobs = [(kind, g[k:k + BATCH]) for (seq, f, kind), g in groups.items() for k in range(0, len(g), BATCH)]
    stats = {"requests": 0, "cache_hits": 0, "failed": 0, "blocked_items": 0, "minimal_context_batches": 0}
    lock = threading.Lock()

    def build(kind):
        def _build(minimal: bool, batch: list):
            state: dict[str, Any] = {"source_text_excerpt": "" if minimal else _context(decoded, batch),
                                     "items": {f"I{n}": it.text for n, it in enumerate(batch)}}
            if kind == "obs" and not minimal:
                state = {"command": commands.get(batch[0].tool_call_id or "", "unknown"), **state}
            names = ("output_kind", "specificity") if kind == "obs" else ("role", "certainty", "specificity", "restatement")
            questions = {}
            for n in range(len(batch)):
                for name in names:
                    q = QUESTIONS[name]
                    questions[f"I{n}.{name}"] = {"type": "choice", "instructions": q["instruction"].format(ref=f"items.I{n}"),
                                                 "criteria": q["options"]}
            return state, questions
        return _build

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_ask_with_fallback, client, build(kind), batch, f"t2.1-passA:{kind}", repeat, stats, lock)
                   for kind, batch in jobs]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except BudgetExceeded:
                raise
            except Exception as exc:
                stats["failed"] += 1
                on_error(f"passA failure: {exc}")
    for batch, answer, minimal in results:
        stats["requests"] += 1
        stats["cache_hits"] += int(answer.cached)
        stats["minimal_context_batches"] += int(minimal)
        method = base_method + (":minimal-context" if minimal else "")
        for n, it in enumerate(batch):
            for name in ("role", "certainty", "specificity", "restatement", "output_kind"):
                a = answer.answers.get(f"I{n}.{name}")
                if a is not None:
                    it.labels[label_prefix + name] = _choice_label(a, QUESTIONS[name]["threshold"], method)
    return stats


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_][a-z0-9_]{3,}", text.lower()) if w not in _STOP}


def pair_candidates(items: Sequence, *, label_prefix: str = "") -> tuple[list[tuple[Any, list]], dict[str, Any]]:
    """Deterministic, prefix-safe retrieval of (later, [earlier...]) pairs to ask about."""
    lab = lambda i, n: i.label(label_prefix + n)
    prose = sorted((i for i in items if i.source_kind in PROSE and lab(i, "specificity") != "generic"
                    and not i.task_copy), key=lambda i: (i.available_seq, i.seq, i.order))
    earlier_pool = [i for i in prose if lab(i, "role") in APPROACH_ROLES]
    words = {i.item_id: _content_words(i.text) for i in prose}
    out, gated_out = [], 0
    for later in prose:
        if lab(later, "role") not in LATER_ROLES:
            continue
        if not REFERENCE_CUE.search(later.text):
            gated_out += 1
            continue
        pool = [e for e in earlier_pool if e.session_id == later.session_id and e.seq < later.seq
                and e.available_seq <= later.available_seq]
        if not pool:
            continue
        lw, lt = words[later.item_id], set(later.tags)
        scored = sorted(pool, key=lambda e: (-(len(lt & set(e.tags)) * 3 + len(lw & words[e.item_id])),
                                             -e.available_seq, -e.seq))
        chosen = [e for e in scored if len(lt & set(e.tags)) or len(lw & words[e.item_id]) >= 2][:MAX_PAIR_CANDIDATES - RECENT_ALWAYS]
        for e in sorted(pool, key=lambda e: (-e.available_seq, -e.seq))[:RECENT_ALWAYS]:
            if e not in chosen:
                chosen.append(e)
        out.append((later, chosen[:MAX_PAIR_CANDIDATES]))
    return out, {"later_items_considered": len(out), "later_items_gated_out_by_cue": gated_out,
                 "retrieval_rule": f"cue-gated later items; up to {MAX_PAIR_CANDIDATES - RECENT_ALWAYS} earlier items "
                                   f"sharing a path/test tag or >=2 content words, plus the {RECENT_ALWAYS} most recent "
                                   f"approach-role items; same session; earlier available at later cut"}


def pass_b(items: Sequence, decoded: dict[tuple[int, str], str], client: JevClient, *, workers: int = 12,
           repeat: int = 0, on_error: Callable[[str], None] = print, label_prefix: str = "") -> tuple[list[Link], dict]:
    base_method = f"jev:{client.model}:{LABEL21_VERSION}"
    pairs, retrieval = pair_candidates(items, label_prefix=label_prefix)
    q = QUESTIONS["relation"]
    stats = {"requests": 0, "cache_hits": 0, "failed": 0, "blocked_items": 0, "pairs_asked": 0, **retrieval}
    lock = threading.Lock()

    def job(later, earlier):
        def _build(minimal: bool, batch: list):
            ctx = "" if minimal else _context(decoded, [later])
            state = {"later": later.text, "later_context": ctx, "earlier": {f"E{n}": e.text for n, e in enumerate(batch)}}
            questions = {f"E{n}.rel": {"type": "choice", "instructions": q["instruction"].format(ref=f"E{n}"),
                                       "criteria": q["options"]} for n in range(len(batch))}
            return state, questions
        return _ask_with_fallback(client, _build, earlier, "t2.1-passB:relation", repeat, stats, lock)

    links: list[Link] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, later, earlier): later for later, earlier in pairs}
        for future in as_completed(futures):
            later = futures[future]
            try:
                results = future.result()
            except BudgetExceeded:
                raise
            except Exception as exc:
                stats["failed"] += 1
                on_error(f"passB failure: {exc}")
                continue
            for batch, answer, minimal in results:
                stats["requests"] += 1
                stats["cache_hits"] += int(answer.cached)
                method = base_method + (":minimal-context" if minimal else "")
                for n, e in enumerate(batch):
                    a = answer.answers.get(f"E{n}.rel")
                    if a is None:
                        continue
                    stats["pairs_asked"] += 1
                    lab = _choice_label(a, q["threshold"], method)
                    probs = lab.probabilities or {}
                    if lab.value not in {UNKNOWN, "no_explicit_relation", "ambiguous"}:
                        links.append(Link(later.item_id, e.item_id, lab.value, "expressed_classified",
                                          probs.get(lab.value), later.available_seq, method))
                    else:
                        top = max(probs.items(), key=lambda kv: kv[1]) if probs else (None, 0.0)
                        if top[0] not in {None, "no_explicit_relation", "ambiguous"} and top[1] >= 0.4:
                            links.append(Link(later.item_id, e.item_id, top[0], "observer_inferred",
                                              top[1], later.available_seq, method + ":below-threshold"))
    return links, stats
