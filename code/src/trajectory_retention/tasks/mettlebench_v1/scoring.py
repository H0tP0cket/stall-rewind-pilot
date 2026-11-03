"""Frozen MettleBench criterion scoring (mettlebench.scoring.v1).

Raw checks: one per ordered acceptance criterion (``criterion::<id>``).
Atomic capability: the criterion itself (1:1, binary completion).
Requirement group: the checklist ``kind`` of the criterion.
Score: equal-weight fraction of criteria observed PASS (independent of order).
Native reward: the upstream satisfied-prefix score, kept separately
(``raw_score`` / ``raw_max_score`` and ``details.native``).

Ordered-prerequisite translation: MettleBench declares its checklist as ordered
and interdependent. A non-PASS criterion that has an earlier non-PASS criterion
is ``BLOCKED`` (``blocked_by`` = the first earlier non-PASS criterion); its
observed outcome stays in ``raw_status``/details. The first non-PASS criterion
keeps its observed FAIL/ERROR_TASK. PASS is always PASS. Grader/harness faults
are INFRA_ERROR and invalidate the grade; worker-caused crashes and timeouts
are task outcomes.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from trajectory_retention.contracts import CapabilityResult, CheckResult, CheckStatus, Evaluation, StageSpec

VERSION = "mettlebench.scoring.v1"


def build_rubric(checklist: dict, *, upstream_id: str) -> dict:
    points = checklist["points"]
    ids = [p["id"] for p in points]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate checklist criterion identities")
    guards = [p["id"] for p in points if str(p.get("group") or "").lower() == "guard"]
    if guards:
        raise ValueError("Guard criteria require an explicit review before scoring")
    n = len(points)
    return {"version": VERSION, "upstream_task_id": upstream_id, "mapping_frozen": True,
            "criteria": [{"check_id": "criterion::" + p["id"], "criterion_id": p["id"], "index": i,
                          "capability_id": p["id"], "requirement_group": p.get("kind") or "unspecified",
                          "title": p.get("title", ""), "weight": {"numerator": 1, "denominator": n}}
                         for i, p in enumerate(points, start=1)],
            "score_rule": "equal-weight fraction of criteria observed PASS; FAIL/ERROR_TASK/BLOCKED/NOT_RUN earn 0; any INFRA_ERROR invalidates",
            "blocking_rule": "non-PASS criterion after an earlier non-PASS criterion -> BLOCKED(blocked_by=first earlier non-PASS)",
            "native_reward": "upstream checklist prefix progress (pass_rate), retained separately",
            "reachability_rule": "PASS -> reachable; first non-PASS -> reachable (executed); BLOCKED -> not reachable; otherwise unknown"}


class MettleBenchScorer:
    def __init__(self, rubric_path: Path, *, task_id: str, binding: dict[str, str]):
        self.rubric = json.loads(Path(rubric_path).read_text())
        if self.rubric.get("version") != VERSION or not self.rubric.get("mapping_frozen"):
            raise ValueError("Unsupported or unfrozen MettleBench rubric")
        self.task_id = task_id
        digest = hashlib.sha256(Path(rubric_path).read_bytes() + json.dumps(binding, sort_keys=True).encode()).hexdigest()
        self.version = VERSION + ":" + digest
        self.criteria = self.rubric["criteria"]
        self.expected = [c["check_id"] for c in self.criteria]

    def checks_from_observation(self, criteria: dict | None, *, infra_error: str | None) -> tuple[CheckResult, ...]:
        if infra_error is not None or criteria is None:
            return tuple(CheckResult(check_id=c["check_id"], status=CheckStatus.INFRA_ERROR, capability_id=c["capability_id"],
                                     details={"reachable": None, "reason": infra_error or "no criteria observation"})
                         for c in self.criteria)
        rows = {r["id"]: r for r in criteria["criteria"]}
        if criteria.get("main_error") and not any(r.get("status") != "NOT_RUN" for r in criteria["criteria"]):
            reason = "verifier main failed before any criterion: " + criteria["main_error"]["type"]
            return self.checks_from_observation(None, infra_error=reason)
        out: list[CheckResult] = []
        first_blocker: str | None = None
        killed = bool(criteria.get("killed"))
        for c in self.criteria:
            row = rows.get(c["criterion_id"])
            if row is None and killed:
                row, killed = {"status": "ERROR_TASK", "raw": "grading_deadline_killed"}, False
            row = row or {"status": "NOT_RUN", "raw": "missing"}
            observed = row.get("status", "NOT_RUN")
            if observed not in {s.value for s in CheckStatus} or observed == "INFRA_ERROR":
                raise ValueError("Unexpected criterion status " + str(observed))
            details: dict[str, Any] = {k: row[k] for k in ("raw", "exception_type", "message", "excerpt", "seconds",
                                                           "traceback") if row.get(k) is not None}
            details["observed_status"] = observed
            details["index"] = c["index"]
            if observed == "PASS":
                status, details["reachable"] = CheckStatus.PASS, True
            elif first_blocker is not None:
                status = CheckStatus.BLOCKED
                details.update(reachable=False, blocked_by="criterion::" + first_blocker,
                               reason="ordered checklist: earlier criterion unmet")
            else:
                status = CheckStatus(observed)
                details["reachable"] = True if observed in {"FAIL", "ERROR_TASK"} else None
                first_blocker = c["criterion_id"]
            out.append(CheckResult(check_id=c["check_id"], status=status, raw_status=observed,
                                   capability_id=c["capability_id"], details=details))
        return tuple(out)

    def score(self, checks, *, stage: StageSpec, native: dict | None = None, extra: dict | None = None) -> Evaluation:
        if stage.evaluator_version != self.version:
            raise ValueError("Evaluator mismatch")
        by_id = {c.check_id: c for c in checks}
        if set(by_id) != set(self.expected) or len(by_id) != len(checks):
            raise ValueError("Unexpected/duplicate/missing raw checks")
        ordered = [by_id[k] for k in self.expected]
        valid = not any(c.status == CheckStatus.INFRA_ERROR for c in ordered)
        n = len(ordered)
        capabilities = []
        for c, check in zip(self.criteria, ordered):
            completion = (1.0 if check.status == CheckStatus.PASS else 0.0) if valid else None
            reachable = check.details.get("reachable") if valid else None
            capabilities.append(CapabilityResult(capability_id=c["capability_id"], completion=completion,
                                                 reachable=reachable, requirement_group=c["requirement_group"],
                                                 check_ids=(c["check_id"],),
                                                 milestone_ids=(f"mettlebench::{c['capability_id']}::executed.v1",) if reachable else ()))
        groups: dict[str, float | None] = {}
        for g in dict.fromkeys(c["requirement_group"] for c in self.criteria):
            members = [cap for cap in capabilities if cap.requirement_group == g]
            groups[g] = (sum(cap.completion for cap in members) / len(members)) if valid else None
        passed = sum(1 for c in ordered if c.status == CheckStatus.PASS)
        native_summary = None
        if native is not None:
            first = native.get("first_failure") or {}
            metrics = native.get("metrics") or {}
            native_summary = {"status": native.get("status"), "pass_rate": native.get("pass_rate"),
                              "active_passed": metrics.get("active_passed"), "active_total": metrics.get("active_total"),
                              "first_failure_id": first.get("id"), "first_failure_index": first.get("index"),
                              "feedback": first.get("feedback") if first else None, "summary": native.get("summary")}
        details = {"scoring": VERSION, "mapping_frozen": True, "passed_criteria": passed, "criteria": n,
                   "native": native_summary, "frontier_scope": "current; ever-reached belongs to the detector",
                   "native_feedback_channel": "only native.feedback is worker-deliverable, on submission"}
        details.update(extra or {})
        return Evaluation(task_id=self.task_id, stage_id=stage.stage_id, target_id=stage.target_id,
                          evaluator_version=self.version, valid=valid,
                          score=(passed / n) if valid else None,
                          full_solve=(passed == n) if valid else None,
                          checks=tuple(ordered), capabilities=tuple(capabilities), group_scores=groups,
                          reachable_milestones=tuple(m for cap in capabilities for m in cap.milestone_ids),
                          raw_score=float(native_summary["active_passed"]) if valid and native_summary and native_summary["active_passed"] is not None else None,
                          raw_max_score=float(native_summary["active_total"]) if valid and native_summary and native_summary["active_total"] is not None else None,
                          details=details)


def native_consistency(checks, native: dict | None) -> dict:
    """Does the native prefix agree with the all-criteria observation?"""
    if native is None:
        return {"available": False}
    statuses = [(c.details.get("observed_status"), c.check_id.removeprefix("criterion::"), c.details.get("raw")) for c in checks]
    prefix = 0
    for observed, _, _ in statuses:
        if observed != "PASS":
            break
        prefix += 1
    first = native.get("first_failure") or {}
    expected_status = "pass" if prefix == len(statuses) else ("error" if statuses[prefix][2] == "native_error" else "fail")
    agree = native.get("status") == expected_status
    if expected_status == "fail":
        agree = agree and first.get("id") == statuses[prefix][1] and (native.get("metrics") or {}).get("active_passed") == prefix
    return {"available": True, "consistent": bool(agree), "criteria_prefix": prefix, "native_status": native.get("status"),
            "native_first_failure": first.get("id")}
