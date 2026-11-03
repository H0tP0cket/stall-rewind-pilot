"""Pure, fixed-denominator scoring using the canonical shared records.

This module consumes trusted check observations. It does not run submissions,
track previous frontier state, or decide that a task is admitted.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Sequence

from trajectory_retention.contracts import (
    CapabilityResult, CheckResult, CheckStatus, Evaluation, StageSpec,
)


def _weight(record: dict) -> Fraction:
    value = Fraction(record["numerator"], record["denominator"])
    if value <= 0:
        raise ValueError("Weights must be positive")
    return value


class FileBackupScoringAdapter:
    def __init__(self, rubric: Path | None = None, *, task_id: str = "scb-file-backup-v1"):
        self.path = rubric or Path(__file__).with_name("rubric_draft.json")
        data = self.path.read_bytes()
        self.rubric = json.loads(data)
        self.task_id = task_id
        self.rubric_sha256 = hashlib.sha256(data).hexdigest()
        sources = [Path(__file__), Path(__file__).with_name("supplemental_checks.py")]
        task_dir = Path(__file__).parents[2] / "tasks" / "file_backup"
        sources += [task_dir / name for name in (
            "native_capture.py", "grading.py", "submission.py", "runtime_grading.py", "adapter.py")]
        bound = data + b"".join(p.name.encode() + p.read_bytes() for p in sources if p.exists())
        self.version = "file-backup.scoring.v1:" + hashlib.sha256(bound).hexdigest()
        self.check_specs = {c["check_id"]: c for c in (
            self.rubric["checks"] + self.rubric["proposed_supplemental_checks"])}
        self.cap_specs = {c["capability_id"]: c for c in self.rubric["capabilities"]}
        self.group_specs = {g["group_id"]: g for g in self.rubric["groups"]}
        self.stage_specs = {s["stage_id"]: s for s in self.rubric["stages"]}
        self._validate()

    def _validate(self) -> None:
        r = self.rubric
        if len(self.check_specs) != len(r["checks"]) + len(r["proposed_supplemental_checks"]):
            raise ValueError("Duplicate check identity")
        if len(self.cap_specs) != len(r["capabilities"]) or len(self.group_specs) != len(r["groups"]):
            raise ValueError("Duplicate capability/group identity")
        membership = [c for g in r["groups"] for c in g["capability_ids"]]
        if set(membership) != set(self.cap_specs) or len(membership) != len(self.cap_specs):
            raise ValueError("Each capability must belong to exactly one group")
        for check in self.check_specs.values():
            if check["primary_capability_id"] not in self.cap_specs:
                raise ValueError("Unknown primary capability")
        for cap in self.cap_specs.values():
            if cap["completion_kind"] != "binary":
                raise ValueError("No ordinal capability is approved for this task")
            if cap["capability_id"] not in self.group_specs[cap["group_id"]]["capability_ids"]:
                raise ValueError("Conflicting capability group")
            rule = cap["completion_rule"]
            required = rule["required_native_check_ids"] + rule["required_supplemental_check_ids"]
            if not required or not rule["positive_support_ids"]:
                raise ValueError("Capability requires nonempty checks and positive support")
            if not set(rule["positive_support_ids"]) <= set(required):
                raise ValueError("Positive support must be a required check")
            for check_id in required:
                if self.check_specs[check_id]["primary_capability_id"] != cap["capability_id"]:
                    raise ValueError("Check contributes completion to another primary capability")
        for stage in self.stage_specs.values():
            if set(stage["group_weights"]) != set(stage["active_group_ids"]):
                raise ValueError("Group weight identities differ")
            if sum(map(_weight, stage["group_weights"].values())) != 1:
                raise ValueError("Group weights must sum to one")
            if set(stage["effective_capability_weights"]) != set(stage["active_capability_ids"]):
                raise ValueError("Capability weight identities differ")
            for group_id in stage["active_group_ids"]:
                ids = [c for c in stage["active_capability_ids"] if self.cap_specs[c]["group_id"] == group_id]
                if sum(_weight(stage["within_group_capability_weights"][c]) for c in ids) != 1:
                    raise ValueError("Within-group weights must sum to one")
                for cap_id in ids:
                    if (_weight(stage["group_weights"][group_id]) *
                        _weight(stage["within_group_capability_weights"][cap_id]) !=
                            _weight(stage["effective_capability_weights"][cap_id])):
                        raise ValueError("Inconsistent effective weight")

    def expected_checks(self, stage_id: str) -> tuple[str, ...]:
        stage = self.stage_specs[stage_id]
        ordinal = int(stage_id.rsplit("_", 1)[1])
        native = tuple(stage["native_check_ids"])
        supplements = tuple(c["check_id"] for c in self.rubric["proposed_supplemental_checks"]
                            if ordinal in c["applicable_stages"])
        return native + supplements

    def score(self, checks: Sequence[CheckResult], *, stage: StageSpec) -> Evaluation:
        if stage.evaluator_version != self.version:
            raise ValueError("Stage evaluator binding does not match this scorer")
        spec = self.stage_specs[stage.stage_id]
        expected = self.expected_checks(stage.stage_id)
        received: dict[str, CheckResult] = {}
        for check in checks:
            if check.check_id in received:
                raise ValueError(f"Duplicate observation: {check.check_id}")
            if check.check_id not in expected:
                raise ValueError(f"Unexpected check for stage: {check.check_id}")
            primary = self.check_specs[check.check_id]["primary_capability_id"]
            if check.capability_id not in (None, primary):
                raise ValueError("Observation primary capability conflicts with rubric")
            received[check.check_id] = replace(check, capability_id=primary)
        completed_checks = tuple(received.get(check_id, CheckResult(
            check_id=check_id, status=CheckStatus.NOT_RUN,
            capability_id=self.check_specs[check_id]["primary_capability_id"],
            details={"missing_result": True, "reachable": None},
        )) for check_id in expected)
        by_id = {c.check_id: c for c in completed_checks}
        valid = not any(c.status == CheckStatus.INFRA_ERROR for c in completed_checks)
        cap_results: list[CapabilityResult] = []
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for cap_id in spec["active_capability_ids"]:
            cap = self.cap_specs[cap_id]
            rule = cap["completion_rule"]
            required = tuple(c for c in rule["required_native_check_ids"] +
                             rule["required_supplemental_check_ids"] if c in by_id)
            relevant = [by_id[c] for c in required]
            positives = [by_id[c] for c in rule["positive_support_ids"] if c in by_id]
            def positive_pass(c: CheckResult) -> bool:
                return c.status == CheckStatus.PASS and not c.details.get("generic_task_crash", False)
            complete = bool(relevant) and all(positive_pass(c) for c in relevant) and any(
                positive_pass(c) for c in positives)
            # PASS of a meaningful positive behavioral check is itself observable
            # success. Failed checks need explicit external-boundary evidence.
            reach_values = [c.details.get("reachable") for c in relevant]
            if not valid:
                reachable = None
            elif any(positive_pass(c) for c in positives) or True in reach_values:
                reachable = True
            elif relevant and all(v is False for v in reach_values):
                reachable = False
            else:
                reachable = None
            milestone = f"file_backup::{cap_id}::observable.v1"
            cap_results.append(CapabilityResult(
                capability_id=cap_id, completion=float(complete) if valid else None,
                reachable=reachable, requirement_group=cap["group_id"],
                milestone_ids=(milestone,) if reachable is True else (), check_ids=required,
            ))
            grouped[cap["group_id"]].append((cap_id, float(complete)))
        group_scores = {group_id: (float(sum(
            _weight(spec["within_group_capability_weights"][cap_id]) * int(value)
            for cap_id, value in values)) if valid else None)
            for group_id, values in grouped.items()}
        score = (float(sum(_weight(spec["effective_capability_weights"][c.capability_id]) *
                           int(c.completion) for c in cap_results)) if valid else None)
        native = [c for c in completed_checks if c.check_id in spec["native_check_ids"]]
        raw_pass = sum(c.raw_status in ("pass", "passed") for c in native)
        return Evaluation(
            task_id=self.task_id, stage_id=stage.stage_id, target_id=stage.target_id,
            evaluator_version=self.version, score=score,
            full_solve=all(c.completion == 1 for c in cap_results) if valid else None,
            valid=valid, checks=completed_checks, capabilities=tuple(cap_results),
            group_scores=group_scores,
            reachable_milestones=tuple(m for c in cap_results for m in c.milestone_ids),
            raw_score=float(raw_pass), raw_max_score=float(len(native)),
            details={"rubric_sha256": self.rubric_sha256,
                     "mapping_frozen": self.rubric["mapping_frozen"],
                     "native_raw_measure": "native passing-check count; not benchmark reward",
                     "completion_policy": "all required checks plus positive behavioral support",
                     "frontier_scope": "current observation only; detector owns trajectory-wide ever reached",
                     "missing_check_ids": [c.check_id for c in completed_checks if c.details.get("missing_result")],
                     "unknown_reachability": [c.capability_id for c in cap_results if c.reachable is None]},
        )
