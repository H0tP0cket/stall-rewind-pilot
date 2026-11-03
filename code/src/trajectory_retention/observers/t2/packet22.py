"""T2.2 packet: whole approaches, cause-and-effect layout, one frozen packet.

Base inclusion and exact-duplicate merging are unchanged from T2.1
(``packet21.exclusion_reason`` and ``packet21.dedup``). An approach's packet
items are its included members, reference dependencies, outputs and linked
evidence; the kept occurrence of an exact duplicate of a member (same dedup
key) counts as that member. Packing:

1. the current approach is reserved whole; when it does not fit it is recorded
   as ``unit_exceeds_budget`` and left out, never split;
2. other approaches are added whole, most recently active first; an approach
   that does not fit whole is skipped and the skip is recorded;
3. loose items (owned by no approach) are added by recency: each statement
   alone, all included outputs of one command together.

Items of a skipped approach are never packed loose. The rendered packet is then
checked against the budget and the last-packed unit other than the current
approach is dropped until it fits. Removal views are taken from this frozen
packet and never refilled.

Rendering is verbatim only: the T2.1 header; one block per approach in order
of first activity, items in source order with each command shown as the header
of its outputs (After `cmd`:), blocks separated by a blank line; then "Other
notes:" in time order. No IDs, probabilities, role titles or status words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .packet21 import ACTION_CHARS, DISPLAY_CHARS, HEADER, _one_line, dedup, exclusion_reason
from .units21 import PROSE, View21
from .units22 import Approach, build_approaches

PACKET22_VERSION = "t2.2-packet/v1"
OTHER_NOTES = "Other notes:"
Count = Callable[[str], int]


def _pos(view: View21) -> dict[str, tuple[int, int, int]]:
    return {i.item_id: (i.available_seq, i.seq, i.order) for i in view.items}


def _dup_key(item) -> tuple:
    # Same key as packet21.dedup: exact text (whitespace-normalized), source kind, session, certainty.
    return (" ".join(item.text.split()), item.source_kind, item.session_id, item.label("certainty"))


def _result_of(view: View21) -> dict[str, str]:
    return {l.source_id: l.target_id for l in view.links if l.relation == "result_of"}


def approach_items(view: View21, approaches: Sequence[Approach], ids: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    """Per approach, which of ``ids`` are its prose and its evidence. Prose: its members and reference
    dependencies, and any statement with the same dedup key as one of its members (the displayed occurrence of an
    exact duplicate is the member). Evidence: its outputs and linked evidence. An item may belong to several
    approaches (a shared output, a duplicate sentence); rendering shows it once."""
    by_id, pos = view.by_id, _pos(view)
    owners: dict[Any, set[str]] = {}
    for a in approaches:
        for m in a.members + a.reference_dependencies:
            owners.setdefault(m, set()).add(a.unit_id)
        for m in a.members:
            if m in by_id:
                owners.setdefault(_dup_key(by_id[m]), set()).add(a.unit_id)
    idset = {i for i in ids if i in by_id}
    out = {a.unit_id: {"prose": [], "evidence": []} for a in approaches}
    for iid in sorted(idset, key=pos.get):
        if by_id[iid].source_kind in PROSE:
            for unit in owners.get(iid, set()) | owners.get(_dup_key(by_id[iid]), set()):
                out[unit]["prose"].append(iid)
    for a in approaches:
        out[a.unit_id]["evidence"] = sorted((e for e in set(a.evidence) if e in idset), key=pos.get)
    return out


def _item_lines(item) -> list[str]:
    if item.source_kind == "observation":
        return ["    " + ln for ln in item.text[:DISPLAY_CHARS].splitlines() if ln.strip()]
    return ["- " + _one_line(item.text, DISPLAY_CHARS)]


def item_tokens(item, count: Count) -> int:
    """Tokens of an item as displayed (the unit used for removal sizes and matched random controls)."""
    return count("\n".join(_item_lines(item)))


def _block_lines(view: View21, ids: Sequence[str], result_of: dict[str, str]) -> list[str]:
    by_id = view.by_id
    lines: list[str] = []
    header = None
    for iid in ids:
        item = by_id[iid]
        if item.source_kind == "observation":
            key = result_of.get(iid, f"none:{item.seq}")
            if key != header:
                action = by_id.get(result_of.get(iid, ""))
                lines.append(f"After `{_one_line(action.text, ACTION_CHARS)}`:" if action is not None else "Observed:")
                header = key
        else:
            header = None
        lines.extend(_item_lines(item))
    return lines


def render22(view: View21, packet_ids: Sequence[str], *, approaches: Sequence[Approach] | None = None) -> str:
    """Approach blocks in order of first activity, then "Other notes:" in time order. ``approaches`` defaults to
    ``build_approaches(view, ())``; pass the approaches the packet was built from."""
    if approaches is None:
        approaches = build_approaches(view, ())
    by_id, pos = view.by_id, _pos(view)
    ids = sorted({i for i in packet_ids if i in by_id}, key=pos.get)
    own = approach_items(view, approaches, ids)
    result_of = _result_of(view)
    lines, placed = [HEADER, ""], set()
    for a in sorted(approaches, key=lambda a: (a.first_seq, a.unit_id)):
        o = own[a.unit_id]
        block = [i for i in sorted(set(o["prose"]) | set(o["evidence"]), key=pos.get) if i not in placed]
        if block:
            placed.update(block)
            lines += _block_lines(view, block, result_of) + [""]
    loose = [i for i in ids if i not in placed]
    if loose:
        lines += [OTHER_NOTES] + _block_lines(view, loose, result_of) + [""]
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class Packet22:
    budget: int | None
    current: str | None
    item_ids: list[str]
    current_reserved: bool
    unit_exceeds_budget: bool
    approach_items: dict[str, dict[str, list[str]]]  # packed approaches: their prose and evidence in the packet
    approach_tokens: dict[str, int]                  # every approach with included items: whole-block tokens
    packed_approaches: list[str]
    skipped_approaches: list[dict]
    loose_packed: int
    loose_skipped: int
    withheld_with_skipped_approaches: int
    exclusions: dict[str, int]
    occurrences: dict[str, list[str]]
    tokens: int
    version: str = PACKET22_VERSION
    trimmed_units: list[str] = field(default_factory=list)

    @property
    def protected_evidence(self) -> list[str]:
        return (self.approach_items.get(self.current) or {}).get("evidence", []) if self.current_reserved else []

    def as_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_packet22(view: View21, approaches: Sequence[Approach], current: Approach | str | None, budget: int | None,
                   count: Count, current_share: float | None = None) -> Packet22:
    """current_share None: whole approach blocks, a block that does not fit is skipped (decision-bounded approaches).
    current_share set (implemented-decision windows, which are often larger than the budget): the current window gets
    at least this share of the budget (more when the earlier windows are shorter), and a block that does not fit
    keeps its newest items that do; packing stops after the first truncated earlier window, so the packet is the
    newest part of the current window plus the newest part of the history before it."""
    by_id, pos = view.by_id, _pos(view)
    current_id = current.unit_id if isinstance(current, Approach) else current
    exclusions: dict[str, int] = {}
    included = []
    for item in view.items:
        reason = exclusion_reason(item)
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
        else:
            included.append(item)
    included, occurrences = dedup(included)
    inc_ids = [i.item_id for i in included]
    own = approach_items(view, approaches, inc_ids)
    result_of = _result_of(view)
    whole = {u: sorted(set(o["prose"]) | set(o["evidence"]), key=pos.get) for u, o in own.items()}
    block_cost = lambda ids: count("\n".join(_block_lines(view, ids, result_of))) + 1
    approach_tokens = {u: block_cost(ids) for u, ids in whole.items() if ids}

    chosen: set[str] = set()
    packed: list[tuple[str, list[str]]] = []  # (unit key, ids) in packing order
    used = count(HEADER) + 1
    skipped: list[dict] = []

    def take(key: str, ids: list[str], cost: int) -> bool:
        nonlocal used
        if budget is not None and used + cost > budget:
            return False
        chosen.update(ids)
        packed.append((key, ids))
        used += cost
        return True

    def newest_fit(ids: list[str], room: int) -> list[str]:
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if block_cost(ids[-mid:]) <= room:
                lo = mid
            else:
                hi = mid - 1
        return ids[len(ids) - lo:]

    reserved = exceeds = False
    truncating = current_share is not None and budget is not None
    by_unit = {a.unit_id: a for a in approaches}
    if truncating and current_id in by_unit and whole.get(current_id):
        earlier = sum(t for u, t in approach_tokens.items() if u != current_id)
        cap = max(int(current_share * (budget - used)), budget - used - earlier)
        ids = whole[current_id]
        if approach_tokens[current_id] > cap:
            ids, exceeds = newest_fit(ids, cap), True
            skipped.append({"unit_id": current_id, "tokens": approach_tokens[current_id], "reason": "truncated_to_newest",
                            "kept_items": len(ids)})
        if ids and take(current_id, ids, block_cost(ids)):
            reserved = True
    elif current_id is not None and current_id in by_unit:
        if not whole.get(current_id):
            skipped.append({"unit_id": current_id, "tokens": 0, "reason": "no_included_items"})
        elif take(current_id, whole[current_id], approach_tokens[current_id]):
            reserved = True
        else:
            exceeds = True
            skipped.append({"unit_id": current_id, "tokens": approach_tokens[current_id], "reason": "exceeds_budget"})
    others = sorted((a for a in approaches if a.unit_id != current_id),
                    key=lambda a: (-a.last_activity_seq, a.first_seq, a.unit_id))
    stop = False
    for a in others:
        ids = [i for i in whole.get(a.unit_id, []) if i not in chosen]
        if not whole.get(a.unit_id):
            skipped.append({"unit_id": a.unit_id, "tokens": 0, "reason": "no_included_items"})
        elif ids and stop:
            skipped.append({"unit_id": a.unit_id, "tokens": approach_tokens[a.unit_id], "reason": "does_not_fit"})
        elif ids and not take(a.unit_id, ids, block_cost(ids)):
            part = newest_fit(ids, budget - used) if truncating else []
            if part and take(a.unit_id, part, block_cost(part)):
                skipped.append({"unit_id": a.unit_id, "tokens": approach_tokens[a.unit_id], "reason": "truncated_to_newest",
                                "kept_items": len(part)})
            else:
                skipped.append({"unit_id": a.unit_id, "tokens": approach_tokens[a.unit_id], "reason": "does_not_fit"})
            stop = truncating
        elif not ids:
            packed.append((a.unit_id, []))  # every item already packed with another approach

    owned = set().union(*whole.values()) if whole else set()
    loose: dict[str, list[str]] = {}
    for iid in inc_ids:
        if iid in owned:
            continue
        key = f"act:{result_of.get(iid, iid)}" if by_id[iid].source_kind == "observation" else iid
        loose.setdefault(key, []).append(iid)
    loose_packed = loose_skipped = 0
    if loose:
        used += count(OTHER_NOTES)
    for key in sorted(loose, key=lambda k: max(pos[i] for i in loose[k]), reverse=True):
        ids = sorted(loose[key], key=pos.get)
        if take(key, ids, block_cost(ids) - 1):
            loose_packed += 1
        else:
            loose_skipped += 1

    # The rendered packet must fit: drop the last-packed unit (never the reserved current approach) until it does.
    trimmed: list[str] = []
    render = lambda: render22(view, sorted(chosen), approaches=approaches)
    if budget is not None:
        while count(render()) > budget:
            droppable = [n for n, (key, _) in enumerate(packed) if not (reserved and key == current_id)]
            if not droppable:
                break
            key, ids = packed.pop(droppable[-1])
            others_ids = {i for _, other in packed for i in other}
            chosen.difference_update(set(ids) - others_ids)
            trimmed.append(key)
            if key in loose:
                loose_packed, loose_skipped = loose_packed - 1, loose_skipped + 1
            elif key in by_unit:
                skipped.append({"unit_id": key, "tokens": approach_tokens.get(key, 0), "reason": "trimmed_to_fit"})
    packed_units = [key for key, _ in packed if key in by_unit]
    item_ids = sorted(chosen, key=pos.get)
    final = approach_items(view, approaches, item_ids)
    skipped_ids = {s["unit_id"] for s in skipped}
    withheld = {i for u in skipped_ids for i in whole.get(u, [])} - chosen
    return Packet22(budget=budget, current=current_id, item_ids=item_ids, current_reserved=reserved,
                    unit_exceeds_budget=exceeds, approach_items={u: final[u] for u in packed_units},
                    approach_tokens=approach_tokens, packed_approaches=packed_units, skipped_approaches=skipped,
                    loose_packed=loose_packed, loose_skipped=loose_skipped,
                    withheld_with_skipped_approaches=len(withheld), exclusions=exclusions, occurrences=occurrences,
                    tokens=count(render()), trimmed_units=trimmed)
