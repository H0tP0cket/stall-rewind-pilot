"""T2.1 packet construction, rendering and treatment views (guide §7.2–§7.3, §11, §15).

Declared base inclusion (descriptive, not a utility prediction):
  * worker prose and tool-output blocks only (actions appear as headers of their results);
  * exclude exact task copies (deterministic);
  * exclude role=narration;
  * exclude classifier judgements only at the stricter exclusion threshold:
    specificity=generic, restatement=restates_requirement, output_kind in
    {displayed_file_or_code, listing_or_progress}; below it the item stays.
Deduplication: exact text (whitespace-normalized) among worker prose with the
same source kind, session and certainty label; the latest occurrence is
displayed and all occurrences are recorded. Tool outputs are never merged.
Packing: freeze the selected unit first (approach members, reference
dependencies, associated evidence), then pack remaining units by recency: each
worker statement is a unit and all included results of one command form one
unit. The rendered packet is then trimmed, oldest non-reserved unit first,
until it fits the budget. Views remove declared items from that frozen packet
and never refill.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .questions21 import QUESTIONS
from .units21 import APPROACH_ROLES, EVIDENCE_ROLES, PROSE, Pursuit, View21

PACKET21_VERSION = "t2.1-packet/v2"
BUDGETS = (1500, 3000, 6000)
DEFAULT_BUDGET = 3000
DISPLAY_CHARS = 700
ACTION_CHARS = 160
RANDOM_TOLERANCE = 0.10
HEADER = ("Notes recorded from earlier work on this task. They are verbatim excerpts of what was observed, "
          "said or tried. They may describe code or files that have since changed.")
SECTIONS = (
    ("Stated observations and facts", {"observation_report", "factual_assertion"}),
    ("Explanations and hypotheses", {"explanation_or_hypothesis"}),
    ("Approaches and design choices", {"strategy_or_design_choice"}),
    ("Intended next steps", {"intended_action"}),
    ("Open questions", {"unresolved_question"}),
    ("Conclusions", {"retrospective_takeaway"}),
    ("Working notes", {"working_material"}),
    ("Other statements", {"mixed", "unknown"}),
)


def _p(item, name: str, value: str) -> float:
    lab = item.labels.get(name)
    return float((lab.probabilities or {}).get(value, 0.0)) if lab is not None else 0.0


def exclusion_reason(item) -> str | None:
    if item.source_kind not in PROSE | {"observation"}:
        return "not_content_kind"
    if item.task_copy:
        return "exact_task_copy"
    if item.source_kind in PROSE and item.label("role") == "narration":
        return "narration"
    if _p(item, "specificity", "generic") >= QUESTIONS["specificity"]["exclusion_threshold"]:
        return "generic"
    if item.source_kind in PROSE and _p(item, "restatement", "restates_requirement") >= QUESTIONS["restatement"]["exclusion_threshold"]:
        return "restates_requirement"
    if item.source_kind == "observation":
        for kind in ("displayed_file_or_code", "listing_or_progress"):
            if _p(item, "output_kind", kind) >= QUESTIONS["output_kind"]["exclusion_threshold"]:
                return kind
    return None


def dedup(items: Sequence) -> tuple[list, dict[str, list[str]]]:
    """Exact duplicates among worker prose under compatible source/session/certainty; keep latest, record all."""
    groups: dict[tuple, list] = {}
    for item in items:
        if item.source_kind not in PROSE:
            continue
        key = (" ".join(item.text.split()), item.source_kind, item.session_id, item.label("certainty"))
        groups.setdefault(key, []).append(item)
    drop, occurrences = set(), {}
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda i: (i.available_seq, i.seq, i.order))
        keep = group[-1]
        occurrences[keep.item_id] = [i.item_id for i in group]
        drop.update(i.item_id for i in group[:-1])
    return [i for i in items if i.item_id not in drop], occurrences


@dataclass
class Packet21:
    budget: int | None
    selected_unit: str | None
    item_ids: list[str]
    unit_reserved: bool
    unit_exceeds_budget: bool
    exclusions: dict[str, int]
    occurrences: dict[str, list[str]]
    packed_turns: int
    skipped_turns: int
    tokens_estimate: int
    protected_evidence: list[str] = field(default_factory=list)
    withheld_candidates: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _line_cost(item, count: Callable[[str], int]) -> int:
    return count(item.text[:DISPLAY_CHARS]) + 4


def build_packet(view: View21, unit: Pursuit | None, budget: int | None, count: Callable[[str], int]) -> Packet21:
    by_id = view.by_id
    exclusions: dict[str, int] = {}
    included = []
    for item in view.items:
        reason = exclusion_reason(item)
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
        else:
            included.append(item)
    included, occurrences = dedup(included)
    inc_ids = {i.item_id for i in included}
    result_of = {l.source_id: l.target_id for l in view.links if l.relation == "result_of"}

    unit_ids: list[str] = []
    protected: list[str] = []
    withheld: list[str] = []
    if unit is not None:
        withheld = [m for m in unit.approach_members + unit.reference_dependencies if m in inc_ids]
        protected = [e for e in unit.associated_evidence if e in inc_ids]
        unit_ids = withheld + protected
    chosen: list[str] = []
    used = 0
    unit_reserved = unit_exceeds = False
    if unit_ids:
        cost = sum(_line_cost(by_id[i], count) for i in unit_ids)
        if budget is None or cost <= budget:
            chosen.extend(unit_ids)
            used += cost
            unit_reserved = True
        else:
            unit_exceeds = True
    # Units: each worker statement alone; all included results of one command together.
    units: dict[str, list] = {}
    unit_seq: dict[str, tuple] = {}
    for item in included:
        if item.item_id in unit_ids:
            continue
        key = f"act:{result_of.get(item.item_id, item.item_id)}" if item.source_kind == "observation" else item.item_id
        units.setdefault(key, []).append(item)
        unit_seq[key] = max(unit_seq.get(key, (0, 0, 0)), (item.available_seq, item.seq, item.order))
    packed_units: list[str] = []
    packed = skipped = 0
    for key in sorted(units, key=lambda k: unit_seq[k], reverse=True):
        members = units[key]
        cost = sum(_line_cost(i, count) for i in members) + (count(" ".join(by_id[result_of[members[0].item_id]].text.split())[:ACTION_CHARS]) + 6
                                                             if members[0].source_kind == "observation" and members[0].item_id in result_of else 0)
        if budget is not None and used + cost > budget:
            skipped += 1
            continue
        chosen.extend(i.item_id for i in members)
        packed_units.append(key)
        used += cost
        packed += 1
    # Enforce the budget on the rendered text: drop the oldest non-reserved unit until it fits.
    if budget is not None:
        while packed_units and count(render21(view, chosen)) > budget:
            oldest = packed_units.pop()
            drop = {i.item_id for i in units[oldest]}
            chosen = [c for c in chosen if c not in drop]
            packed -= 1
            skipped += 1
    order = {i.item_id: (i.available_seq, i.seq, i.order) for i in view.items}
    chosen.sort(key=lambda i: order[i])
    return Packet21(budget, unit.unit_id if unit else None, chosen, unit_reserved, unit_exceeds, exclusions,
                    occurrences, packed, skipped, used, protected if unit_reserved else [],
                    withheld if unit_reserved else [])


def _one_line(text: str, limit: int) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def render21(view: View21, item_ids: Sequence[str], *, style: str = "typed") -> str:
    """Readable packet: tool results grouped under their command, then worker statements by role (typed) or as
    prose paragraphs with the same content and grouping (prose). No IDs, probabilities or graph data."""
    by_id = view.by_id
    ids = [i for i in item_ids if i in by_id]
    result_of = {l.source_id: l.target_id for l in view.links if l.relation == "result_of"}
    lines = [HEADER, ""]
    groups: dict[str, list] = {}
    order: list[str] = []
    for iid in ids:
        item = by_id[iid]
        if item.source_kind == "observation":
            key = result_of.get(iid, f"none:{item.seq}")
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(item)
    if order:
        lines.append("Tool results:" if style == "typed" else "Results of commands that were run:")
        for key in order:
            action = by_id.get(key)
            head = f"After `{_one_line(action.text, ACTION_CHARS)}`:" if action is not None else "Observed:"
            lines.append(("- " if style == "typed" else "") + head)
            for o in groups[key]:
                lines.extend("    " + ln for ln in o.text[:DISPLAY_CHARS].splitlines() if ln.strip())
        lines.append("")
    for title, roles in SECTIONS:
        section = [by_id[i] for i in ids if by_id[i].source_kind in PROSE and by_id[i].label("role") in roles]
        if not section:
            continue
        texts = [_one_line(i.text, DISPLAY_CHARS) for i in section]
        if style == "typed":
            lines += [f"{title}:"] + [f"- {t}" for t in texts] + [""]
        else:
            lines += [f"{title}: " + " ".join(t if t.endswith((".", "?", "!")) else t + "." for t in texts), ""]
    return "\n".join(lines).rstrip() + "\n"


# ---- treatment views over one frozen packet ----

def minus_selected(packet: Packet21) -> list[str]:
    drop = set(packet.withheld_candidates)
    return [i for i in packet.item_ids if i not in drop]


def random_unit_deletion(view: View21, packet: Packet21, seed: int, count: Callable[[str], int]) -> dict[str, Any]:
    """Delete complete non-evidence statements (approach roles first) matching the targeted deletion's token mass
    within ±10%, excluding the target unit and protected evidence. Returns the kept ids and the realized match."""
    by_id = view.by_id
    target_ids = set(packet.withheld_candidates)
    target_mass = sum(_line_cost(by_id[i], count) for i in target_ids)
    if not target_ids:
        return {"status": "no_target", "item_ids": list(packet.item_ids)}
    protected = set(packet.protected_evidence) | target_ids
    pool = [by_id[i] for i in packet.item_ids if i not in protected and by_id[i].source_kind in PROSE
            and by_id[i].label("role") not in EVIDENCE_ROLES]
    approach_pool = [i for i in pool if i.label("role") in APPROACH_ROLES]
    rng = random.Random(seed)
    for candidates, composition in ((approach_pool, "approach_roles"), (pool, "any_non_evidence_statement")):
        for _ in range(300):
            order = candidates[:]
            rng.shuffle(order)
            chosen, mass = [], 0
            for item in order:
                c = _line_cost(item, count)
                if mass + c <= target_mass * (1 + RANDOM_TOLERANCE):
                    chosen.append(item.item_id)
                    mass += c
                if mass >= target_mass * (1 - RANDOM_TOLERANCE):
                    drop = set(chosen)
                    return {"status": "matched", "seed": seed, "composition": composition,
                            "target_tokens": target_mass, "deleted_tokens": mass,
                            "mismatch": (mass - target_mass) / target_mass, "deleted_ids": sorted(drop),
                            "item_ids": [i for i in packet.item_ids if i not in drop]}
    return {"status": "no_matched_random_control", "seed": seed, "target_tokens": target_mass,
            "item_ids": list(packet.item_ids)}
