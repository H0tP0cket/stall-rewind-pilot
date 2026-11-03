"""Frozen-T renderings: full typed T, same-content plain prose, and packet bytes.

Both renderers are deterministic functions of one frozen TVersion. The prose
renderer preserves every entry's substantive text and type meaning in prose,
uncertainty (text is quoted verbatim), historical scope (where and when it was
observed), worker status and supersession. It adds no diagnosis, drops no
entry and changes no certainty. Renderer revision is recorded in every packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._canonical import content_id, sha256_hex
from .tstore import TEntry, TVersion

RENDERER_REVISION = "render/dev-2026-09-15-v3-same-content-prose"

_TYPE_LABEL = {
    "verified_fact": "Observed fact",
    "hypothesis": "Hypothesis",
    "plan_step": "Plan step",
    "scratch": "Note",
    "lesson": "Lesson",
    "open_question": "Open question",
}
_STATUS_LABEL = {
    "observed": "observed",
    "stated": "stated by the agent",
    "worker_confirmed": "later confirmed by the agent",
    "worker_rejected": "later rejected by the agent",
    "worker_abandoned": "later abandoned by the agent",
}
_CATEGORY_LABEL = {
    "worker_expressed": "agent's own words",
    "observation_direct": "read from delivered output",
}
_SCOPE_LABEL = {
    "task_requirement": "task requirement",
    "implementation_behavior": "behaviour of the implementation at that time",
    "runtime_behavior": "runtime behaviour at that time",
    "worker_strategy": "the agent's strategy",
}

OMITTED_REFERENCE_POLICY = "reference_by_id_only"  # never leak omitted text


@dataclass(frozen=True)
class Packet:
    role: str  # render_typed / render_prose / view_filtered
    text: str
    source_version_id: str
    source_content_hash: str
    renderer_revision: str
    filter_id: str | None
    entry_ids: tuple[str, ...]
    omitted_entry_ids: tuple[str, ...]
    lineage: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_hex(self.text)

    def as_record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "text_sha256": self.sha256,
            "byte_length": len(self.text.encode("utf-8")),
            "approx_token_count": approx_tokens(self.text),
            "source_version_id": self.source_version_id,
            "source_content_hash": self.source_content_hash,
            "renderer_revision": self.renderer_revision,
            "filter_id": self.filter_id,
            "entry_ids": list(self.entry_ids),
            "omitted_entry_ids": list(self.omitted_entry_ids),
            "omitted_reference_policy": OMITTED_REFERENCE_POLICY,
            "lineage": dict(self.lineage),
        }

    def packet_id(self) -> str:
        return content_id(self.as_record())


def approx_tokens(text: str) -> int:
    """Cheap, deterministic length proxy; provider token counts are recorded at delivery."""
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def _provenance(entry: TEntry) -> str:
    where = f"stage {entry.stage_id}" if entry.stage_id else "unknown stage"
    after = f", after checkpoint {entry.observed_after_checkpoint_id}" if entry.observed_after_checkpoint_id else ""
    session = f", session {entry.session_id}" if entry.session_id else ", unknown session"
    requirement = f", requirements {entry.requirement_version}" if entry.requirement_version else ""
    return f"{where}{session}{requirement}{after}, event {entry.first_source_seq}"


def _relationships(entry: TEntry, present: set[str]) -> list[str]:
    """Identical substantive metadata for both presentations; no copied successor text."""
    def reference(entry_id: str) -> str:
        return entry_id if entry_id in present else f"{entry_id} (not included here)"

    lines = []
    if entry.depends_on:
        lines.append("This depended on: " + ", ".join(reference(d) for d in entry.depends_on) + ".")
    if entry.superseded_by:
        lines.append("This was later replaced by: " + reference(entry.superseded_by) + ".")
    if entry.applicability:
        for assessment in entry.applicability:
            lines.append(f"Applicability at {assessment.target_checkpoint_id}: {assessment.value} "
                         f"(assessed by {assessment.method}).")
    else:
        lines.append("Current applicability is unknown.")
    return lines


def _entry_lines(entry: TEntry, present: set[str]) -> list[str]:
    lines = [f"- [{entry.entry_id}] {_TYPE_LABEL[entry.type]} ({_CATEGORY_LABEL[entry.source_category]}; "
             f"{_SCOPE_LABEL[entry.scope]}; {_STATUS_LABEL[entry.worker_status]}; {_provenance(entry)})"]
    lines.append(f"  \"{entry.text}\"")
    lines.extend("  " + line for line in _relationships(entry, present))
    return lines


def render_typed(version: TVersion, entries: Sequence[TEntry] | None = None, *,
                 filter_id: str | None = None, omitted: Sequence[str] = ()) -> Packet:
    entries = list(version.entries if entries is None else entries)
    present = {e.entry_id for e in entries}
    lines = [
        "Recorded trajectory notes (typed). Each entry preserves a statement or normalized observation;",
        "provenance says where. Entries describe the state when they were recorded, not necessarily now.",
        "",
    ]
    for e in entries:
        lines.extend(_entry_lines(e, present))
        lines.append("")
    return _packet("render_typed" if filter_id is None else "view_filtered", "\n".join(lines).rstrip() + "\n",
                   version, entries, filter_id, omitted)


def render_prose(version: TVersion, entries: Sequence[TEntry] | None = None, *,
                 filter_id: str | None = None, omitted: Sequence[str] = ()) -> Packet:
    """Same content and frozen-entry order as typed T, expressed as paragraphs."""
    entries = list(version.entries if entries is None else entries)
    present = {e.entry_id for e in entries}
    paragraphs = [
        "Notes from the earlier trajectory, in frozen entry order. Each note preserves a statement "
        "or normalized observation and describes the situation at the time it was recorded."
    ]
    for e in entries:
        sentence = (f"Note {e.entry_id} records the following {_TYPE_LABEL[e.type].lower()}. "
                    f"At {_provenance(e)}, {_CATEGORY_LABEL[e.source_category]}, "
                    f"{_STATUS_LABEL[e.worker_status]}, concerning {_SCOPE_LABEL[e.scope]}: \"{e.text}\"")
        sentence += " " + " ".join(_relationships(e, present))
        paragraphs.append(sentence)
    return _packet("render_prose" if filter_id is None else "view_filtered", "\n\n".join(paragraphs) + "\n",
                   version, entries, filter_id, omitted)


def _packet(role: str, text: str, version: TVersion, entries: Sequence[TEntry],
            filter_id: str | None, omitted: Sequence[str]) -> Packet:
    entry_map = {e.entry_id: e for e in version.entries}
    selected = [e.entry_id for e in entries]
    if len(set(selected)) != len(selected) or any(entry_map.get(e.entry_id) != e for e in entries):
        raise ValueError("rendered entries must be unique unchanged entries from the frozen T")
    if selected != [e.entry_id for e in version.entries if e.entry_id in set(selected)]:
        raise ValueError("rendered entries must preserve frozen T order")
    expected_omitted = [e.entry_id for e in version.entries if e.entry_id not in set(selected)]
    if list(omitted) != expected_omitted:
        raise ValueError("omission manifest must list exactly the omitted frozen entries in order")
    return Packet(
        role=role, text=text, source_version_id=version.version_id(),
        source_content_hash=version.content_hash(), renderer_revision=RENDERER_REVISION,
        filter_id=filter_id, entry_ids=tuple(e.entry_id for e in entries),
        omitted_entry_ids=tuple(omitted),
        lineage={"target_checkpoint_id": version.target_checkpoint_id, "role_of_source": version.role,
                 "update_index": version.update_index, "source_status": version.status,
                 "order_policy": "frozen_entry_order",
                 "rendered_content_hash": content_id([e.as_record() for e in entries])},
    )
