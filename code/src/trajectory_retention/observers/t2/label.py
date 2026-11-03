"""Classifier labeling of verbatim T2 items (two passes, both prefix-safe).

Pass 1 labels each item using only its own source event (the item plus nearby
text from the same event). Pass 2 asks, for selected later items, which earlier
item they build on and whether they explicitly drop, replace or confirm it. The
earlier candidates shown are items already available at the later item's cut in
the same native session, with their pass-1 labels. No future event is shown.

All questions are generic; none mention a task, benchmark or expected answer.
Labels below the confidence gates are recorded as ``unknown`` with their full
probability vector.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence

from .jev import BudgetExceeded, JevClient, RequestBlocked
from .schema import (CHOICE_THRESHOLD, KEEP_NO, KEEP_YES, LINK_THRESHOLD, OBSERVATION_SOURCES, STATUS_THRESHOLD,
                     UNKNOWN, WORKER_SOURCES, Item, Label, Link)

LABEL_VERSION = "t2-label/v4"
BATCH_ITEMS = 20
CONTEXT_CHARS = 5000
MAX_EARLIER = 60

ROLE_CRITERIA = {
    "evidence": "States a specific fact about the code, environment, requirements or a result the agent saw "
                "(for example which test fails, what a function returns, which module is missing).",
    "approach": "Proposes an explanation, hypothesis, design decision or strategy: why something happens or how "
                "the agent intends to solve the problem.",
    "intended_action": "Names a specific next action the agent will take on a named target (inspect, edit, run, "
                       "test, revert a particular file, function, command or check).",
    "question": "Raises an unresolved question or unknown that the agent still needs to answer.",
    "takeaway": "States a conclusion or lesson drawn from something that already happened, meant to guide later work.",
    "working_note": "Intermediate working: a calculation, worked example, code fragment, restated input or "
                    "step-by-step trace.",
    "narration": "Generic process talk with no specific content (for example 'let me look at the files', "
                 "'now I will implement it', 'done').",
}
STANCE_CRITERIA = {
    "asserted": "Presented as true or settled.",
    "uncertain": "Explicitly hedged or tentative (maybe, might, probably, I think, possibly, not sure).",
    "question": "Phrased as a question.",
    "not_a_claim": "Not a claim about anything (an action, instruction, heading or fragment).",
}
KEEP_PROSE = ("Would `{ref}` help a capable engineer who takes over this coding task in a fresh session understand "
              "something the agent learned, believed, tried, decided or planned? Answer no for generic process talk, "
              "restated instructions and bare code without explanation.")
KEEP_OBSERVATION = ("`{ref}` is output of the command in `command`. Does it show a specific result that the agent "
                    "would need to remember, such as an error message, a failing or passing check, a missing file or "
                    "module, or a measured or printed value that answers what the command was checking? Answer no "
                    "for routine listings, file contents shown for reading, progress noise and boilerplate.")
SPEC_STATEMENT = ("Is `{ref}` only a statement of what the task, specification or instructions say or require "
                  "(for example 'the spec says X', 'the output must be Y'), rather than something the agent found out, "
                  "decided, tried or concluded itself?")
REFERS = ("Which item in `earlier` does `{ref}` directly build on, test, respond to or continue? Choose none unless "
          "the connection is clear from the wording.")
STATUS_CRITERIA = {
    "none": "It does not say the agent is dropping, replacing or confirming an earlier idea or plan.",
    "abandons": "It says an earlier idea, explanation or plan is wrong, does not work, or is being dropped.",
    "replaces": "It says the agent is switching from an earlier idea or plan to a different one.",
    "confirms": "It says an earlier idea or explanation turned out to be correct.",
}
# Explicit change language in the worker's own words (generic English markers).
CHANGE_MARKER = re.compile(
    r"\b(instead|abandon\w*|give up|gave up|doesn't work|does not work|didn't work|did not work|won't work|"
    r"not the (?:issue|problem|cause)|wrong|incorrect|rather than|revert\w*|scrap|drop that|no longer|"
    r"that's not|that is not|turns out|confirmed|indeed|actually|wait\b|switch\w* to|different approach|"
    r"other approach|was right|was correct)", re.IGNORECASE)


def _choice(answer: dict[str, Any], threshold: float, method: str) -> Label:
    probs = answer.get("probabilities") or {}
    top = max(probs.items(), key=lambda kv: kv[1]) if probs else (answer.get("choice"), None)
    value = top[0] if top[1] is not None and top[1] >= threshold else UNKNOWN
    return Label(value, probs, method, answer.get("confidence"))


def _noul(answer: dict[str, Any], method: str) -> Label:
    p = answer.get("noul")
    if not isinstance(p, (int, float)):
        return Label(UNKNOWN, None, method)
    value = "yes" if p >= KEEP_YES else "no" if p <= KEEP_NO else UNKNOWN
    return Label(value, {"yes": float(p), "no": 1 - float(p)}, method)


def _context(texts: dict[tuple[int, str], str], batch: Sequence[Item]) -> str:
    first = batch[0]
    field_text = texts.get((first.seq, first.span.field))
    if field_text is None:
        return ""
    start = max(0, min(i.span.start for i in batch) - 600)
    end = min(len(field_text), max(i.span.end for i in batch) + 300)
    window = field_text[start:end]
    return window if len(window) <= CONTEXT_CHARS else window[:CONTEXT_CHARS]


def pass1(items: Sequence[Item], decoded_fields: dict[tuple[int, str], str], client: JevClient, *,
          workers: int = 12, repeat: int = 0, on_error: Callable[[str], None] = print,
          commands: dict[str, str] | None = None) -> dict[str, Any]:
    """Deterministic labels for structural sources; classifier labels for prose and observations."""
    method = f"jev:{client.model}"
    groups: dict[tuple[int, str, str], list[Item]] = {}
    for item in items:
        if item.source_kind == "worker_action":
            item.labels["role"] = Label("action", None, "deterministic:source_kind")
            continue
        if item.source_kind == "format_error":
            item.labels["role"] = Label("harness_message", None, "deterministic:source_kind")
            item.labels["keep"] = Label("no", None, "deterministic:harness_message")
            continue
        if item.task_copy:
            item.labels["keep"] = Label("no", None, "deterministic:task_copy")
        if item.source_kind in OBSERVATION_SOURCES:
            item.labels["role"] = Label("evidence", None, "deterministic:observation")
            item.labels["stance"] = Label("observed", None, "deterministic:observation")
        groups.setdefault((item.seq, item.span.field, "obs" if item.source_kind in OBSERVATION_SOURCES else "prose"), []).append(item)

    jobs = []
    for (seq, field, kind), group in groups.items():
        for k in range(0, len(group), BATCH_ITEMS):
            jobs.append((kind, group[k:k + BATCH_ITEMS]))

    def run(kind: str, batch: list[Item], minimal: bool = False):
        state = {"source_text_excerpt": "" if minimal else _context(decoded_fields, batch),
                 "items": {f"I{n}": it.text for n, it in enumerate(batch)}}
        if kind == "obs" and not minimal:
            # The producing command is an earlier event in the same prefix (structural tool-call binding).
            state = {"command": (commands or {}).get(batch[0].tool_call_id or "", "unknown"), **state}
        questions: dict[str, Any] = {}
        for n, it in enumerate(batch):
            ref = f"items.I{n}"
            if kind == "obs":
                if "keep" not in it.labels:
                    questions[f"I{n}.keep"] = {"type": "noul", "instructions": KEEP_OBSERVATION.format(ref=ref)}
                continue
            questions[f"I{n}.role"] = {"type": "choice", "instructions": f"What role does `{ref}` play in the coding agent's work?",
                                       "criteria": ROLE_CRITERIA}
            questions[f"I{n}.stance"] = {"type": "choice", "instructions": f"How is `{ref}` expressed?",
                                         "criteria": STANCE_CRITERIA}
            if "keep" not in it.labels:
                questions[f"I{n}.keep"] = {"type": "noul", "instructions": KEEP_PROSE.format(ref=ref)}
            questions[f"I{n}.spec"] = {"type": "noul", "instructions": SPEC_STATEMENT.format(ref=ref)}
        if not questions:
            return batch, None
        return batch, client.ask(state, questions, purpose=f"pass1:{kind}" + (":minimal" if minimal else ""),
                                 repeat=repeat)

    def run_with_fallback(kind: str, batch: list[Item]):
        try:
            return [run(kind, batch)]
        except RequestBlocked:
            # Firewall-blocked payloads are retried one item at a time without surrounding context.
            out = []
            for it in batch:
                try:
                    out.append(run(kind, [it], minimal=True))
                except RequestBlocked:
                    stats["blocked_items"] += 1
            return out

    stats = {"requests": 0, "failed": 0, "cache_hits": 0, "blocked_items": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_with_fallback, kind, batch) for kind, batch in jobs]
        results = []
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except BudgetExceeded:
                raise
            except Exception as exc:  # recorded as missing labels, never guessed
                stats["failed"] += 1
                on_error(f"pass1 failure: {exc}")
        for batch, answer in results:
            if answer is None:
                continue
            stats["requests"] += 1
            stats["cache_hits"] += int(answer.cached)
            for n, it in enumerate(batch):
                a = answer.answers
                if f"I{n}.role" in a:
                    it.labels["role"] = _choice(a[f"I{n}.role"], CHOICE_THRESHOLD, method)
                if f"I{n}.stance" in a:
                    it.labels["stance"] = _choice(a[f"I{n}.stance"], CHOICE_THRESHOLD, method)
                if f"I{n}.keep" in a:
                    it.labels["keep"] = _noul(a[f"I{n}.keep"], method)
                if f"I{n}.spec" in a:
                    it.labels["spec_statement"] = _noul(a[f"I{n}.spec"], method)
    return stats


LINKABLE_ROLES = {"approach", "intended_action", "takeaway", "question", "evidence"}
TARGET_ROLES = {"approach", "intended_action", "question", "takeaway"}


def pass2(items: Sequence[Item], client: JevClient, *, workers: int = 12, repeat: int = 0,
          on_error: Callable[[str], None] = print) -> tuple[list[Link], dict[str, Any]]:
    """Links from later prose items to earlier same-session items; prefix-safe by construction."""
    method = f"jev:{client.model}"
    prose = sorted((i for i in items if i.source_kind in WORKER_SOURCES and i.label("keep") == "yes"),
                   key=lambda i: (i.available_seq, i.seq, i.order))
    by_event: dict[tuple[str | None, int], list[Item]] = {}
    for item in prose:
        if item.label("role") in LINKABLE_ROLES:
            by_event.setdefault((item.session_id, item.seq), []).append(item)
    jobs = []
    for (session, seq), current in by_event.items():
        cut = min(i.available_seq for i in current)
        earlier = [i for i in prose if i.session_id == session and i.seq < seq and i.available_seq <= cut
                   and i.label("role") in TARGET_ROLES][-MAX_EARLIER:]
        if not earlier:
            continue
        for k in range(0, len(current), BATCH_ITEMS):
            jobs.append((current[k:k + BATCH_ITEMS], earlier))

    def run_with_fallback(current: list[Item], earlier: list[Item]):
        try:
            return [run(current, earlier)]
        except RequestBlocked:
            out = []
            for c in current:
                try:
                    out.append(run([c], earlier))
                except RequestBlocked:
                    stats["blocked_items"] += 1
            return out

    def run(current: list[Item], earlier: list[Item]):
        state = {"earlier": {f"E{n}": e.text for n, e in enumerate(earlier)},
                 "current": {f"C{n}": c.text for n, c in enumerate(current)}}
        options = {f"E{n}": None for n in range(len(earlier))} | {"none": "No clear connection to any earlier item."}
        questions: dict[str, Any] = {}
        for n, c in enumerate(current):
            questions[f"C{n}.refers"] = {"type": "choice", "instructions": REFERS.format(ref=f"current.C{n}"),
                                         "criteria": options}
            if CHANGE_MARKER.search(c.text):
                questions[f"C{n}.status"] = {"type": "choice",
                                             "instructions": f"Does `current.C{n}` say the agent is dropping, replacing "
                                                             "or confirming an idea or plan from `earlier`?",
                                             "criteria": STATUS_CRITERIA}
        return current, earlier, client.ask(state, questions, purpose="pass2:links", repeat=repeat)

    links: list[Link] = []
    stats = {"requests": 0, "failed": 0, "cache_hits": 0, "blocked_items": 0}
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_with_fallback, cur, ear) for cur, ear in jobs]
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except BudgetExceeded:
                raise
            except Exception as exc:
                stats["failed"] += 1
                on_error(f"pass2 failure: {exc}")
    for current, earlier, answer in results:
        stats["requests"] += 1
        stats["cache_hits"] += int(answer.cached)
        for n, c in enumerate(current):
            ref = answer.answers.get(f"C{n}.refers")
            if not ref:
                continue
            probs = ref.get("probabilities") or {}
            target_key, p = max(probs.items(), key=lambda kv: kv[1]) if probs else ("none", 0.0)
            c.labels["refers"] = Label(target_key if p >= LINK_THRESHOLD else UNKNOWN, probs, method,
                                       ref.get("confidence"))
            if target_key == "none" or p < LINK_THRESHOLD:
                continue
            target = earlier[int(target_key[1:])]
            links.append(Link(c.item_id, target.item_id, "refers_to", "observer_inferred", p, c.available_seq, method))
            status = answer.answers.get(f"C{n}.status")
            if status:
                lab = _choice(status, STATUS_THRESHOLD, method)
                c.labels["status_effect"] = lab
                if lab.value in {"abandons", "replaces", "confirms"}:
                    # The change is stated in the worker's own words (explicit marker); the target
                    # identification is the classifier's. Both facts are recorded.
                    links.append(Link(c.item_id, target.item_id, lab.value, "worker_stated",
                                      lab.probabilities.get(lab.value) if lab.probabilities else None,
                                      c.available_seq, method + "+marker"))
    return links, stats
