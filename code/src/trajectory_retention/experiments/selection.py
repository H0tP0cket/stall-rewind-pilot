"""Prospective source selection, independent of technical branch eligibility.

The first online detector warning is selected once for the complete natural
trajectory. Capture, grade, runtime or source-binding failures remain outcomes
of that selection. They never authorize replacement by a later warning.
Offline labels cannot select a prospective primary state.
"""
from typing import Any, Mapping, Sequence

FIRST_STALL_SELECTION_RULE = "first-online-S-by-evidence-cutoff-and-annotation-id; no technical-failure replacement.v1"


def first_stall_annotation(annotations: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    warnings = [a for a in annotations if a.get("kind") == "S" and a.get("origin") == "online"]
    return min(warnings, key=lambda a: (a["evidence_cutoff"], a["annotation_id"])) if warnings else None
