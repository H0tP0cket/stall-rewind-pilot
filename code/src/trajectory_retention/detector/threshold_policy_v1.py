"""Explicit, opt-in category thresholds; no inferred or fallback token limit.

Profiles bind the caller's numerical decision to retained calibration evidence.
They neither select a threshold from S frequency nor certify scientific adequacy
of that evidence. Existing launch plans and historical DetectorConfig are not
rewritten. Configurations are identical across workers within each category.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from trajectory_retention.contracts import ArtifactRef, to_jsonable
from trajectory_retention.detector.core import DetectorConfig, RULE_VERSION, LAP_RULE_VERSIONS, STALL_CLOCK_RULE_VERSIONS

VERSION = "detector-threshold-profile/v1"


@dataclass(frozen=True, kw_only=True)
class DetectorThresholdProfile:
    profile_id: str
    category_settings: Mapping[str, Mapping[str, Any]]
    evidence_refs: tuple[ArtifactRef, ...]
    rationale: str
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.version != VERSION or not self.profile_id.strip() or not self.rationale.strip():
            raise ValueError("Declare a supported, named threshold profile and rationale")
        if not self.evidence_refs or not all(isinstance(ref, ArtifactRef) for ref in self.evidence_refs):
            raise ValueError("Threshold profile requires retained calibration evidence")
        if not self.category_settings:
            raise ValueError("Declare at least one threshold category")
        for category, settings in self.category_settings.items():
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Threshold category must be a nonempty name")
            _config(settings)

    def config_for(self, category: str) -> DetectorConfig:
        if category not in self.category_settings:
            raise ValueError(f"No calibrated/provisional settings declared for category {category!r}")
        # Validate again: dataclass(frozen=True) does not freeze nested mappings.
        return _config(self.category_settings[category])


def _config(settings: Mapping[str, Any]) -> DetectorConfig:
    names = {field.name for field in fields(DetectorConfig)}
    # score_min_gain (stall-clock/v7) postdates older profiles; absent means 0.
    if not (names - {"score_min_gain"}) <= set(settings) <= names:
        raise ValueError("Declare every detector setting explicitly; no defaults or unknown fields")
    for key in ("stagnation_token_threshold", "persistent_failure_grades", "weak_region_grades"):
        if type(settings[key]) is not int:
            raise ValueError(f"Detector setting {key} must be an integer")
    if settings["rule_version"] not in (*STALL_CLOCK_RULE_VERSIONS, RULE_VERSION, *LAP_RULE_VERSIONS):
        raise ValueError("New threshold profiles require the certified-positive, observation-gap rule")
    return DetectorConfig(**dict(settings))


def validate_detector_assignments(profile: DetectorThresholdProfile, *,
                                  assignments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check an explicit task/model/category/settings matrix without changing it.

Each assignment needs task_id, model, category and detector_settings. A task may
not move categories between workers. No task/model cell may be duplicated. A
profile has no model-specific override and missing settings fail before launch.
The returned record is suitable for a launch manifest, not a readiness stamp.
"""
    if not assignments:
        raise ValueError("No detector assignments supplied")
    seen, categories, bindings = set(), {}, []
    for row in assignments:
        task, model, category = (row.get(key) for key in ("task_id", "model", "category"))
        if not all(isinstance(value, str) and value.strip() for value in (task, model, category)):
            raise ValueError("Each assignment requires task, model and category")
        key = (task, model)
        if key in seen:
            raise ValueError("Duplicate task/model detector assignment")
        seen.add(key)
        if categories.setdefault(task, category) != category:
            raise ValueError("One task cannot change detector category across workers")
        expected = profile.config_for(category)
        supplied = _config(row.get("detector_settings", {}))
        if supplied != expected:
            raise ValueError("Executed detector settings differ from the declared category profile")
        bindings.append({"task_id": task, "model": model, "category": category,
                         "detector_settings": supplied.to_record()})
    return {"version": VERSION, "profile": to_jsonable(profile), "assignments": bindings,
            "scope": "Configuration binding only; no threshold selected and no scientific freeze inferred."}
