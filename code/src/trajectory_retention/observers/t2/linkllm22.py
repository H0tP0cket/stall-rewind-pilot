"""T2.2 / Ledger linker v2: an LLM reads a numbered stretch of the trajectory and lists explicit back-references.

Why: on 8 blind-labelled windows, two independent annotators agreed on explicit links at F1 0.79, while Jev's
pairwise question reached only F1 0.43 even when asked every pair (below the "previous sentence" baseline, 0.45).
The annotators read the whole stretch; a pairwise classifier sees two sentences at a time. This linker does what
the annotators did: one call per stretch, the model sees every numbered item and returns the links.

Rules kept from the ratified design: links come only from the later sentence's own wording (never topic
similarity or a shared file name); prefix-only (a stretch contains nothing after its last item); the relation
vocabulary is vocab22's; links are stored at tier ``explicit_wording`` with method ``llm-stretch:<model>``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .schema import Link
from .vocab22 import LINK_TIER_EXPLICIT, OUTPUT_RELATIONS, RELATION_OPTIONS

LINKLLM22_VERSION = "t2.2-linkllm/v1"
PROSE = {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}
OUTPUT = "observation"
NULL = {"no_explicit_relation", "ambiguous"}
SOURCE_NAMES = {"worker_text": "agent message", "worker_reasoning": "agent reasoning",
                "provider_summary": "agent reasoning summary", "worker_code_comment": "comment in agent command",
                "observation": "tool output"}
ITEM_CHARS = 700

INSTRUCTION = """Below are numbered items from a coding agent's work, in the order they happened: the agent's own
sentences (reasoning, messages, comments in its commands) and tool outputs.

For each agent sentence, list every EARLIER item that the sentence's own wording explicitly connects to. Mark a link
only when the later sentence itself:
- refers back to it ("that", "this", "so", "hence", "then", "as I suspected", "that didn't work"), or
- is incomplete without it (e.g. the second half of a split sentence, an elliptical continuation), or
- answers an earlier question, or
- drops, replaces or defers an earlier plan ("better only one file", "let's defer that"), or
- clearly repeats a specific earlier plan or claim (restates), or tests or checks an earlier idea, or
- reads a specific earlier tool output as showing success or failure.
Talking about the same topic, or naming the same file, function or test, is NOT a link. Most pairs have no link.
Only items marked "agent" can be the later item. The reads_output_* relations apply only when the earlier item is
a tool output; the other relations only when the earlier item is an agent sentence.

Relation options:
{options}

Return only JSON: {{"links": [{{"later": "<id>", "earlier": "<id>", "relation": "<option>"}}, ...]}}
Use the item ids exactly as shown. Return {{"links": []}} if there are none."""


def _render(items: Sequence, labels: dict[str, str], commands: dict[str, str]) -> str:
    lines = []
    for item in items:
        who = "agent" if item.source_kind in PROSE else "tool"
        text = item.text if len(item.text) <= ITEM_CHARS else item.text[:ITEM_CHARS] + " …"
        head = f"[{labels[item.item_id]}] ({who}: {SOURCE_NAMES.get(item.source_kind, item.source_kind)})"
        if item.source_kind == OUTPUT and commands.get(item.tool_call_id or ""):
            head += f" output of `{commands[item.tool_call_id][:160]}`"
        lines.append(f"{head}\n{text}")
    return "\n\n".join(lines)


def messages_for(items: Sequence, commands: dict[str, str]) -> tuple[list[dict[str, str]], dict[str, str]]:
    labels = {item.item_id: f"i{n:02d}" for n, item in enumerate(items)}
    options = "\n".join(f"- {k}: {v}" for k, v in RELATION_OPTIONS.items() if k not in NULL)
    body = INSTRUCTION.format(options=options) + "\n\nItems:\n\n" + _render(items, labels, commands)
    return [{"role": "user", "content": body}], labels


def parse_links(text: str) -> list[dict[str, str]]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [x for x in data.get("links", []) if isinstance(x, dict)]


def link_stretch(glm, items: Sequence, commands: dict[str, str], *, later_ids: set[str] | None = None,
                 purpose: str = LINKLLM22_VERSION, repeat: int = 0) -> dict[str, Any]:
    """One call over a prefix-safe stretch (items in source order). ``later_ids`` limits which sentences may be
    the later end (e.g. only the new response's sentences when earlier items are context)."""
    msgs, labels = messages_for(items, commands)
    inv = {v: k for k, v in labels.items()}
    by_id = {i.item_id: i for i in items}
    order = {i.item_id: n for n, i in enumerate(items)}
    answer = glm.complete(msgs, purpose=purpose, repeat=repeat)
    links, rejected = [], []
    method = f"llm-stretch:{getattr(glm, 'model', 'llm')}:{LINKLLM22_VERSION}"
    for raw in parse_links(answer.text):
        a, b, rel = inv.get(str(raw.get("later"))), inv.get(str(raw.get("earlier"))), raw.get("relation")
        reason = None
        if a is None or b is None or rel not in RELATION_OPTIONS or rel in NULL:
            reason = "unparseable"
        elif order[b] >= order[a] or by_id[a].source_kind not in PROSE:
            reason = "not_later_agent_sentence"
        elif later_ids is not None and a not in later_ids:
            reason = "later_outside_scope"
        elif (rel in OUTPUT_RELATIONS) != (by_id[b].source_kind == OUTPUT):
            reason = "relation_incompatible_with_earlier_kind"
        if reason:
            rejected.append({**raw, "reason": reason})
            continue
        links.append(Link(a, b, rel, LINK_TIER_EXPLICIT, None, by_id[a].available_seq, method))
    return {"links": links, "rejected": rejected, "call_record_id": answer.call_record_id,
            "empty_answer": not (answer.text or "").strip()}
