"""T2.2 recovery probe: can the removed content be read back from what a removal view keeps?

A removal is real only if the removed content cannot be recovered from the
remaining packet. The probe asks GLM (``glm.py``), from the notes alone, to
state the agent's content of one kind; Jev then judges, for each removed item,
whether any probe statement says the same thing. Recovery is the
token-weighted share of removed items matched by at least one statement.

Rules:

* The probe sees packet text only: never grades, detector output, references
  or item ids.
* Every arm of one family is scored against the same removed items (those of
  the targeted view): the view itself (the recovery-probe rate), the full
  packet (the ceiling: the items are present) and each matched-random view
  (what the probe recovers when a size-matched random set is removed instead).
* ``same`` at p >= 0.7 is a match; ``partially`` is reported separately and
  never counted as a match. Items Jev refuses to score (firewall 403) are
  reported as unscored, never as unmatched.
* An empty probe answer is an error, not "cannot tell"; only an explicit
  "cannot tell" yields no statements.

Forced-choice reconstruction probe (``probe_family_choice``), added because the
open probe above did not discriminate (every arm: 0.0 same, 1.0 partially).
For each removed item GLM sees an arm's notes and two statements, A and B: the
removed item verbatim and a matched distractor, in a fixed pseudo-random order.
It must pick the one the agent wrote during the work the notes cover.
Scoring is deterministic accuracy; no Jev.

* Distractors come from the same run: worker prose written after the
  checkpoint cut, so the prefix cannot contain them. They exclude task copies,
  restated requirements, texts identical to a prefix item, items with a
  ``restates`` link to a prefix item, and items whose content tokens overlap
  heavily with any single prefix item: at least ``OVERLAP_MIN`` shared tokens
  covering at least ``OVERLAP_SHARE`` of the candidate's tokens (content
  words of 4+ characters outside a stop list, plus path/test-name tags).
* The distractor has the removed item's main role when possible (then the
  closest word count, unused first, earliest); otherwise any eligible later
  prose item (``fallback_any_role``, recorded). With none the pair is dropped
  from every denominator (``no_distractor``).
* Arms: view (target removed), full_packet (ceiling, about 1.0), each
  matched-random view, and empty (no notes: the floor, about 0.5; well above
  0.5 means the distractors give the answer away). Recovery index =
  (view - empty) / (full_packet - empty), None when the denominator is <= 0,
  computed on paired accuracy: only items answered in every arm.
* Unparsed answers, and every question of a call that returned no text
  (e.g. reasoning used the whole token allowance), are reported and excluded,
  never counted as wrong.
"""

from __future__ import annotations

import json
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Mapping, Sequence

from .jev import RequestBlocked
from .label21 import _content_words
from .segment import tags_for

PROBE22_VERSION = "t2.2-probe/v1"
PROBE_INSTRUCTION = ("From these notes only, state the agent's {target} as specifically as you can; "
                     "say 'cannot tell' if you cannot.")
FORMAT_INSTRUCTION = ("Write each statement on its own line starting with \"- \". "
                      "If you cannot tell, reply only: cannot tell")
# Probe target per filter family (names match the T2.2 filter families; the arms themselves are not ratified).
PROBE_TARGETS = {
    "forget_theories": "hypotheses or explanations about what was wrong",
    "forget_plans": "plans and chosen approach",
    "forget_done_claims": "claims that its fix worked or the task was done",
    "forget_current_approach": "current approach: what it was trying, why, and what it intended to do next",
    "forget_dead_ends": "approaches it tried and then dropped or found not to work",
    "facts_only": "hypotheses, plans, claims and conclusions about its own work",
    "denoise": "general process remarks and restatements of the task requirements",
}
SAME_OPTIONS = {
    "same": "The statement says the same thing as the item (a paraphrase or a restatement of its content).",
    "partially": "The statement covers part of what the item says, or says something close but less specific.",
    "different": "The statement says something else.",
}
_RANK = {"same": 0, "partially": 1, "different": 2}
SAME_INSTRUCTION = "Does statement `statements.{ref}` say the same thing as `removed_item`?"
SAME_THRESHOLD = 0.7
STATEMENT_BATCH = 12
_CANNOT = re.compile(r"^\W*cannot tell\W*$", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


class ProbeIncomplete(RuntimeError):
    """The probe returned no text (e.g. reasoning used the whole token allowance)."""


def parse_statements(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines or all(_CANNOT.match(ln) for ln in lines):
        return []
    bulleted = [ln for ln in lines if _BULLET.match(ln)]
    chosen = bulleted if bulleted else lines
    out = [_BULLET.sub("", ln).strip() for ln in chosen]
    return [s for s in out if s and not _CANNOT.match(s)]


def probe_messages(packet_text: str, target: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"Notes:\n<<<\n{packet_text.rstrip()}\n>>>\n\n"
                                        f"{PROBE_INSTRUCTION.format(target=target)}\n{FORMAT_INSTRUCTION}"}]


def recovery_probe(glm, packet_text: str, target: str, *, repeat: int = 0) -> list[str]:
    answer = glm.complete(probe_messages(packet_text, target), purpose=f"{PROBE22_VERSION}:{target[:40]}", repeat=repeat)
    if not answer.text.strip():
        raise ProbeIncomplete(f"empty probe answer (finish_reason={answer.finish_reason}, "
                              f"receipt {answer.call_record_id})")
    return parse_statements(answer.text)


def _label(answer: Mapping[str, Any] | None) -> tuple[str, str | None, float]:
    """(value at threshold or 'unknown', top option, its probability)."""
    probs = dict((answer or {}).get("probabilities") or {})
    if not probs:
        return "unknown", None, 0.0
    top, p = max(probs.items(), key=lambda kv: kv[1])
    return (top if p >= SAME_THRESHOLD else "unknown"), top, p


def score_recovery(jev, statements: Sequence[str], removed_items: Sequence, count: Callable[[str], int], *,
                   repeat: int = 0, workers: int = 8) -> dict[str, Any]:
    """Token-weighted share of removed items that at least one probe statement restates (Jev, p >= 0.7)."""
    base = {"version": PROBE22_VERSION, "n_statements": len(statements), "n_removed": len(removed_items),
            "threshold": SAME_THRESHOLD}
    if not removed_items:
        return {**base, "status": "no_removed_items", "recovery": None, "per_item": []}
    weights = {i.item_id: count(i.text) for i in removed_items}
    total = sum(weights.values())
    rows = {i.item_id: {"item_id": i.item_id, "tokens": weights[i.item_id], "best_value": None, "best_p": 0.0,
                        "best_statement": None, "matched": False, "partial": False, "blocked_chunks": 0}
            for i in removed_items}
    stats = {"requests": 0, "cache_hits": 0, "blocked_chunks": 0}
    lock = threading.Lock()

    def job(item, start: int):
        chunk = statements[start:start + STATEMENT_BATCH]
        state = {"removed_item": item.text, "statements": {f"S{start + n}": s for n, s in enumerate(chunk)}}
        questions = {f"S{start + n}.same": {"type": "choice", "instructions": SAME_INSTRUCTION.format(ref=f"S{start + n}"),
                                            "criteria": SAME_OPTIONS} for n in range(len(chunk))}
        try:
            answer = jev.ask(state, questions, purpose=f"{PROBE22_VERSION}:score", repeat=repeat)
        except RequestBlocked:
            with lock:
                stats["blocked_chunks"] += 1
                rows[item.item_id]["blocked_chunks"] += 1
            return
        with lock:
            stats["requests"] += 1
            stats["cache_hits"] += int(answer.cached)
            row = rows[item.item_id]
            for n in range(len(chunk)):
                value, top, p = _label(answer.answers.get(f"S{start + n}.same"))
                row["matched"] |= value == "same"
                row["partial"] |= value in {"same", "partially"}
                if top is not None and (row["best_value"] is None or _RANK.get(top, 3) < _RANK.get(row["best_value"], 3)
                                        or (top == row["best_value"] and p > row["best_p"])):
                    row["best_value"], row["best_p"], row["best_statement"] = top, p, start + n

    if statements:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, item, k) for item in removed_items for k in range(0, len(statements), STATEMENT_BATCH)]
            for f in futures:
                f.result()  # re-raises BudgetExceeded and any other failure
    unscored = [r for r in rows.values() if r["blocked_chunks"] and not r["matched"]]
    scored_tokens = total - sum(r["tokens"] for r in unscored)
    matched = sum(r["tokens"] for r in rows.values() if r["matched"])
    partial = sum(r["tokens"] for r in rows.values() if r["partial"])
    return {**base, "status": "scored", "removed_tokens": total, "scored_tokens": scored_tokens,
            "unscored_items": len(unscored), "matched_tokens": matched,
            "recovery": matched / scored_tokens if scored_tokens else None,
            "recovery_including_partial": partial / scored_tokens if scored_tokens else None,
            "per_item": list(rows.values()), **stats}


def probe_family(glm, jev, *, family: str, full_text: str, view_text: str, removed_items: Sequence,
                 count: Callable[[str], int], random_views: Mapping[str, tuple[str, Collection[str]]] | None = None,
                 repeat: int = 0) -> dict[str, Any]:
    """Recovery-probe rate for one removal view, with the full-packet ceiling and matched-random arms."""
    target = PROBE_TARGETS[family]
    removed_ids = {i.item_id for i in removed_items}
    tokens = {i.item_id: count(i.text) for i in removed_items}
    arms: dict[str, tuple[str, float | None]] = {"view": (view_text, 1.0), "full_packet": (full_text, 0.0)}
    for name, (text, rnd_removed) in (random_views or {}).items():
        overlap = sum(t for iid, t in tokens.items() if iid in set(rnd_removed))
        arms[f"random:{name}"] = (text, overlap / sum(tokens.values()) if tokens else None)
    out: dict[str, Any] = {"version": PROBE22_VERSION, "family": family, "target": target,
                           "removed_item_ids": sorted(removed_ids), "removed_tokens": sum(tokens.values()), "arms": {}}
    for arm, (text, overlap) in arms.items():
        statements = recovery_probe(glm, text, target, repeat=repeat)
        score = score_recovery(jev, statements, removed_items, count, repeat=repeat)
        out["arms"][arm] = {"statements": statements, "target_tokens_removed_in_arm": overlap,
                            **{k: v for k, v in score.items() if k != "version"}}
    out["recovery_probe_rate"] = out["arms"]["view"]["recovery"]
    out["ceiling"] = out["arms"]["full_packet"]["recovery"]
    out["random"] = {a: r["recovery"] for a, r in out["arms"].items() if a.startswith("random:")}
    return out


# ---------------------------------------------------------------- forced-choice reconstruction probe

CHOICE22_VERSION = "t2.2-choice-probe/v1"
CHOICE_BATCH = 12
CHOICE_DISPLAY_CHARS = 700
OVERLAP_MIN, OVERLAP_SHARE = 3, 0.6
PROSE = {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}
DISTRACTOR_RULE = (f"{CHOICE22_VERSION}: same-run worker prose with seq > cut; excludes task copies, "
                   f"restatement=restates_requirement, text identical to a prefix item, restates links to a prefix "
                   f"item, and >= {OVERLAP_MIN} shared content tokens covering >= {OVERLAP_SHARE:.0%} of the "
                   f"candidate's tokens with any one prefix item; same main role first, then closest word count")
CHOICE_INSTRUCTION = (
    "Each question below gives two statements, A and B. Exactly one of them was written by the agent during the "
    "work these notes cover; the other was not written during that work. For each question choose the statement "
    "the agent wrote. If you cannot tell, still choose one. Give your confidence from 0.5 (a guess) to 1.0 (certain). "
    'Reply with JSON only, in this form: {"answers": [{"q": 1, "choice": "A", "confidence": 0.5}]}')


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _content_tokens(text: str) -> set[str]:
    return _content_words(text) | {t.lower() for t in tags_for(text)}


@dataclass
class DistractorPool:
    cut: int
    candidates: list
    excluded: dict[str, int]
    role_label: str = "role"
    rule: str = DISTRACTOR_RULE
    _texts: set[str] = field(default_factory=set, repr=False)

    def pick(self, item, used: Collection[str] = ()) -> tuple[Any | None, str]:
        """Matched distractor for ``item`` and how it was chosen (same_role / fallback_any_role / none)."""
        pool = [c for c in self.candidates if _norm(c.text) != _norm(item.text)]
        same = [c for c in pool if c.label(self.role_label) == item.label(self.role_label)]
        chosen, mode = (same, "same_role") if same else (pool, "fallback_any_role")
        if not chosen:
            return None, "none"
        words = len(item.text.split())
        return min(chosen, key=lambda c: (c.item_id in used, abs(len(c.text.split()) - words), c.seq, c.order)), mode

    def as_record(self) -> dict[str, Any]:
        return {"cut": self.cut, "rule": self.rule, "role_label": self.role_label, "candidates": len(self.candidates),
                "candidate_roles": dict(Counter(c.label(self.role_label) for c in self.candidates)),
                "excluded": dict(self.excluded)}


def distractor_pool(ledger_items: Sequence, checkpoint_seq: int, links: Sequence, *,
                    role_label: str = "role") -> DistractorPool:
    """Eligible distractors for removed items of a checkpoint: same-run prose written after the cut."""
    prefix = [i for i in ledger_items if i.available_seq <= checkpoint_seq]
    prefix_ids = {i.item_id for i in prefix}
    restating = set()
    for l in links:
        if l.relation == "restates":
            if l.target_id in prefix_ids:
                restating.add(l.source_id)
            if l.source_id in prefix_ids:
                restating.add(l.target_id)
    prefix_texts = {_norm(i.text) for i in prefix}
    postings: dict[str, list[int]] = defaultdict(list)  # content token -> prefix items containing it
    for n, i in enumerate(prefix):
        for token in _content_tokens(i.text):
            postings[token].append(n)

    def overlaps(tokens: set[str]) -> bool:
        shared = Counter(n for token in tokens for n in postings.get(token, ()))
        return any(k >= OVERLAP_MIN and k / len(tokens) >= OVERLAP_SHARE for k in shared.values())

    excluded: Counter = Counter()
    candidates = []
    for c in sorted((i for i in ledger_items if i.seq > checkpoint_seq and i.source_kind in PROSE),
                    key=lambda i: (i.seq, i.order)):
        tokens = _content_tokens(c.text)
        if c.task_copy:
            excluded["task_copy"] += 1
        elif c.label("restatement") == "restates_requirement":
            excluded["restates_requirement"] += 1
        elif _norm(c.text) in prefix_texts:
            excluded["same_text_in_prefix"] += 1
        elif c.item_id in restating:
            excluded["restates_link_to_prefix"] += 1
        elif tokens and overlaps(tokens):
            excluded["overlaps_prefix_item"] += 1
        else:
            candidates.append(c)
    return DistractorPool(checkpoint_seq, candidates, dict(excluded), role_label)


def _one_line(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= CHOICE_DISPLAY_CHARS else line[:CHOICE_DISPLAY_CHARS - 1] + "\u2026"


def parse_choices(text: str, n: int) -> list[dict[str, Any]]:
    """Choices aligned to questions 1..n; a missing or malformed answer stays None."""
    out: list[dict[str, Any]] = [{"choice": None, "confidence": None} for _ in range(n)]
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except ValueError:
        data = None
    answers = data.get("answers") if isinstance(data, dict) else None
    for a in answers if isinstance(answers, list) else []:
        if not isinstance(a, dict):
            continue
        try:
            q = int(str(a.get("q")).strip().lstrip("Qq")) - 1
        except ValueError:
            continue
        choice = str(a.get("choice", "")).strip().upper()
        conf = a.get("confidence")
        if 0 <= q < n and choice in ("A", "B"):
            out[q] = {"choice": choice,
                      "confidence": float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else None}
    return out


def choice_probe(glm, packet_text: str, pairs: Sequence[tuple[str, str]], *, repeat: int = 0,
                 workers: int = 4) -> list[dict[str, Any]]:
    """Ask which of (A, B) the agent wrote, from ``packet_text`` only; batches of ``CHOICE_BATCH`` questions,
    sent concurrently (answers keep question order).

    A call that returns no text leaves its questions unanswered and flagged ``incomplete``."""
    notes = packet_text.rstrip() or "(no notes)"

    def batch(k: int) -> list[dict[str, Any]]:
        chunk = pairs[k:k + CHOICE_BATCH]
        questions = "\n\n".join(f"Q{n + 1}\nA: {_one_line(a)}\nB: {_one_line(b)}" for n, (a, b) in enumerate(chunk))
        answer = glm.complete([{"role": "user", "content": f"Notes:\n<<<\n{notes}\n>>>\n\n{CHOICE_INSTRUCTION}\n\n{questions}"}],
                              purpose=CHOICE22_VERSION, repeat=repeat)
        incomplete = not answer.text.strip()
        return [{**a, "incomplete": incomplete, "call_record_id": answer.call_record_id}
                for a in parse_choices(answer.text, len(chunk))]

    starts = list(range(0, len(pairs), CHOICE_BATCH))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(starts)))) as pool:
        return [a for rows in pool.map(batch, starts) for a in rows]


def probe_family_choice(glm, *, family: str, full_text: str, view_text: str, removed_items: Sequence,
                        distractor_pool: DistractorPool, count: Callable[[str], int],
                        random_views: Mapping[str, tuple[str, Collection[str]]] | None = None,
                        repeat: int = 0) -> dict[str, Any]:
    """Forced-choice recovery for one removal view: view, full-packet ceiling, matched-random and empty arms."""
    pool = distractor_pool
    pairs, used = [], set()
    for item in removed_items:
        d, mode = pool.pick(item, used)
        row: dict[str, Any] = {"item_id": item.item_id, "role": item.label(pool.role_label), "mode": mode,
                               "distractor_id": None if d is None else d.item_id, "tokens": count(item.text)}
        if d is not None:
            used.add(d.item_id)
            true_first = random.Random(f"{CHOICE22_VERSION}|{item.item_id}|{d.item_id}").random() < 0.5
            row.update(distractor_role=d.label(pool.role_label), true_label="A" if true_first else "B",
                       a=item.text if true_first else d.text, b=d.text if true_first else item.text)
        pairs.append(row)
    scored = [p for p in pairs if p["distractor_id"] is not None]
    tokens = {p["item_id"]: p["tokens"] for p in pairs}
    arms: dict[str, tuple[str, float | None]] = {"view": (view_text, 1.0), "full_packet": (full_text, 0.0)}
    for name, (text, rnd_removed) in (random_views or {}).items():
        overlap = sum(t for iid, t in tokens.items() if iid in set(rnd_removed))
        arms[f"random:{name}"] = (text, overlap / sum(tokens.values()) if tokens else None)
    arms["empty"] = ("", None)
    out: dict[str, Any] = {"version": CHOICE22_VERSION, "probe": "forced_choice", "family": family,
                           "distractors": pool.as_record(), "removed_item_ids": [i.item_id for i in removed_items],
                           "removed_tokens": sum(tokens.values()), "pairs": pairs, "n_pairs": len(scored),
                           "no_distractor": len(pairs) - len(scored),
                           "fallback_any_role": sum(p["mode"] == "fallback_any_role" for p in pairs), "arms": {}}
    qs = [(p["a"], p["b"]) for p in scored]
    with ThreadPoolExecutor(max_workers=max(1, len(arms))) as pool:
        answers_by_arm = dict(zip(arms, pool.map(lambda t: choice_probe(glm, t, qs, repeat=repeat) if qs else [],
                                                   [text for text, _ in arms.values()])))
    for arm, (text, overlap) in arms.items():
        answers = answers_by_arm[arm]
        per_item = [{"item_id": p["item_id"], "choice": a["choice"], "confidence": a["confidence"],
                     "correct": None if a["choice"] is None else a["choice"] == p["true_label"],
                     "incomplete": a["incomplete"], "call_record_id": a["call_record_id"]}
                    for p, a in zip(scored, answers)]
        done = [r for r in per_item if r["correct"] is not None]
        weight = sum(tokens[r["item_id"]] for r in done)
        confs = [r["confidence"] for r in done if r["confidence"] is not None]
        out["arms"][arm] = {"accuracy": sum(r["correct"] for r in done) / len(done) if done else None,
                            "token_weighted_accuracy": (sum(tokens[r["item_id"]] for r in done if r["correct"]) / weight
                                                        if weight else None),
                            "n_scored": len(done), "n_unparsed": len(per_item) - len(done),
                            "n_incomplete": sum(r["incomplete"] for r in per_item),
                            "mean_confidence": sum(confs) / len(confs) if confs else None,
                            "target_tokens_removed_in_arm": overlap, "per_item": per_item}
    answered = [{r["item_id"] for r in a["per_item"] if r["correct"] is not None} for a in out["arms"].values()]
    common = set.intersection(*answered) if answered else set()
    for r in out["arms"].values():
        hits = [x["correct"] for x in r["per_item"] if x["item_id"] in common]
        r["accuracy_paired"] = sum(hits) / len(hits) if hits else None
    out["n_paired"] = len(common)
    acc = {arm: r["accuracy_paired"] for arm, r in out["arms"].items()}
    out["recovery_probe_rate"], out["ceiling"], out["floor"] = acc["view"], acc["full_packet"], acc["empty"]
    out["random"] = {a: v for a, v in acc.items() if a.startswith("random:")}
    usable = None not in (acc["view"], acc["full_packet"], acc["empty"]) and acc["full_packet"] - acc["empty"] > 0
    out["recovery_index"] = (acc["view"] - acc["empty"]) / (acc["full_packet"] - acc["empty"]) if usable else None
    return out
