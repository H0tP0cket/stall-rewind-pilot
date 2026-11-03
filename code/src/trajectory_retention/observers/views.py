"""Deterministic selected-type views of one frozen T (Q14).

A filter keeps entries by type. Surviving entries keep identity, order and
provenance; nothing is regenerated or refilled. References to omitted entries
are rendered by ID only (render.OMITTED_REFERENCE_POLICY). The initial filter
set below is a development proposal, to be frozen before the main study.
"""

from __future__ import annotations

from dataclasses import dataclass

from .render import Packet, render_prose, render_typed
from .tstore import TEntry, TVersion

FILTER_REVISION = "filters/dev-2026-09-15"


@dataclass(frozen=True)
class TypeFilter:
    filter_id: str
    keep_types: frozenset[str]
    drop_superseded: bool = False

    def apply(self, version: TVersion) -> tuple[list[TEntry], list[str]]:
        kept, omitted = [], []
        for e in version.entries:
            if e.type in self.keep_types and not (self.drop_superseded and e.superseded_by):
                kept.append(e)
            else:
                omitted.append(e.entry_id)
        return kept, omitted


FILTERS: dict[str, TypeFilter] = {
    "full": TypeFilter("full", frozenset({"verified_fact", "hypothesis", "plan_step", "scratch", "lesson", "open_question"})),
    "facts_only": TypeFilter("facts_only", frozenset({"verified_fact"})),
    "facts_lessons": TypeFilter("facts_lessons", frozenset({"verified_fact", "lesson"})),
    "facts_hypotheses": TypeFilter("facts_hypotheses", frozenset({"verified_fact", "hypothesis"})),
    "plans_only": TypeFilter("plans_only", frozenset({"plan_step"})),
    "no_facts": TypeFilter("no_facts", frozenset({"hypothesis", "plan_step", "scratch", "lesson", "open_question"})),
}


def filtered_view(version: TVersion, filter_id: str, *, prose: bool = False) -> Packet:
    flt = FILTERS[filter_id]
    kept, omitted = flt.apply(version)
    render = render_prose if prose else render_typed
    return render(version, kept, filter_id=f"{FILTER_REVISION}:{filter_id}", omitted=omitted)
