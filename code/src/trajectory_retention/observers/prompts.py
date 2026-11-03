"""Observer prompt templates. Development revision; nothing here is frozen.

Each template carries a revision string. Artifact records bind the revision
and the prompt hash so a later prompt change never silently alters a frozen
version. Prompts must never be tuned against recovery outcomes.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ._canonical import content_id
from .prefix import SourceEvent

PROMPT_REVISION = "observer-prompts/dev-2026-09-15-v2-exact-wire-shapes"


def render_events(events: Sequence[SourceEvent]) -> str:
    """Render eligible source events with explicit seq/field labels for citation."""
    parts: list[str] = []
    for ev in events:
        for name, text in ev.texts.items():
            parts.append(f"<source seq={ev.seq} kind={ev.kind} field=\"{name}\">\n{text}\n</source>")
    return "\n".join(parts)


T_UPDATE_SYSTEM = """You maintain an external typed record T of a coding agent's trajectory.
You will receive the prior T (JSON) and only the newly allowed source events. Propose update
operations as a JSON array. Rules:
Use precisely these flat operation shapes (replace example values with real source content):
{"op":"add","type":"plan_step","text":"EXACT WORKER QUOTE","source_category":"worker_expressed","scope":"worker_strategy","quotes":[{"seq":123,"field":"assistant_text","quote":"EXACT WORKER QUOTE"}]}
{"op":"set_status","entry_id":"EXISTING ENTRY ID","status":"worker_confirmed","quotes":[{"seq":124,"field":"assistant_text","quote":"EXACT CONFIRMING QUOTE"}]}
{"op":"supersede","entry_id":"OLD ENTRY ID","by":"NEW ENTRY ID OR tmp_id","quotes":[{"seq":125,"field":"assistant_text","quote":"EXACT CHANGE QUOTE"}]}
The add fields are at the top level of each operation. Do not wrap them in an
"entry" object. The citation list is named "quotes", not "evidence", "source",
or "citations". Use the exact source field name printed in the source tag.
Optional add fields: "tmp_id" (string), "depends_on" (array of existing IDs).
Do not include status on add; it is assigned by the validator. Return [] if
there is no supported update. These examples are schemas, not source evidence.
- Every "add" must cite exact verbatim quotes from the provided sources: {"seq", "field", "quote"}.
  The quote must be copied character-for-character.
- source_category "worker_expressed": the entry text MUST be exactly one of the cited quotes
  (a sentence or clause the agent itself wrote in its messages or tool-call arguments).
  Preserve hedges, negation and conditions. Do not improve the statement.
- source_category "observation_direct": a normalized fact directly read from delivered tool
  output or task text. Every number, path, identifier or name in the text must appear in the
  cited quote(s). Only types verified_fact or open_question.
- source_category "extractor_inference": any diagnosis, explanation, plan or lesson YOU infer
  that the agent did not write. These are stored separately and never enter the primary record.
  Use it whenever unsure whether the agent expressed the content.
- Types: verified_fact, hypothesis, plan_step, scratch, lesson, open_question.
- Scope: task_requirement, implementation_behavior, runtime_behavior, worker_strategy.
- "set_status" (worker_confirmed / worker_rejected / worker_abandoned / observed / stated)
  requires quotes from the new events; confirmation/rejection/abandonment needs the agent's
  own words. Silence is not abandonment. A later edit does not disprove an earlier observation.
- "supersede" links an old entry to a newer one with quotes showing the change.
- Do not restate entries already in the prior T. Omit types with nothing new.
Output only the JSON array."""


def t_update_messages(prior_t: Mapping[str, Any] | None, events: Sequence[SourceEvent],
                      interval: tuple[int, int]) -> tuple[Mapping[str, str], ...]:
    prior = json.dumps(prior_t or {"entries": []}, ensure_ascii=False, sort_keys=True)
    user = (
        f"Allowed interval: events with seq in ({interval[0]}, {interval[1]}].\n\n"
        f"PRIOR T:\n{prior}\n\nNEW SOURCE EVENTS:\n{render_events(events)}\n\n"
        "Return the JSON array of operations."
    )
    return ({"role": "system", "content": T_UPDATE_SYSTEM}, {"role": "user", "content": user})


T_FULL_SYSTEM = T_UPDATE_SYSTEM.replace(
    "You will receive the prior T (JSON) and only the newly allowed source events.",
    "You will receive the complete allowed source prefix and no prior T.",
)


def t_full_messages(events: Sequence[SourceEvent], through_seq: int) -> tuple[Mapping[str, str], ...]:
    user = (
        f"Allowed prefix: all events with seq <= {through_seq}.\n\n"
        f"SOURCE EVENTS:\n{render_events(events)}\n\nReturn the JSON array of operations."
    )
    return ({"role": "system", "content": T_FULL_SYSTEM}, {"role": "user", "content": user})


SUMMARY_SYSTEM = """Summarize the following coding-agent session transcript for an engineer who will
continue the work in a fresh session with the same workspace. Write plain prose. Describe what was
attempted, what was observed, what remains open. Do not invent results that were not observed.
Do not add advice beyond what the transcript supports. Stay within {max_words} words."""


def summary_messages(events: Sequence[SourceEvent], through_seq: int, max_words: int) -> tuple[Mapping[str, str], ...]:
    return (
        {"role": "system", "content": SUMMARY_SYSTEM.format(max_words=max_words)},
        {"role": "user", "content": f"TRANSCRIPT (events seq <= {through_seq}):\n{render_events(events)}"},
    )


def prompt_hash(messages: Sequence[Mapping[str, str]]) -> str:
    return content_id([dict(m) for m in messages])
