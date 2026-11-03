"""Task-scoring frontier retention without weakening completion validity.

The task translator supplies trusted per-check observations. A local infra error
invalidates the completion score, but cannot erase an independently observed
positive behavioral result. This helper never treats ordinary FAIL as reachable.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
from pathlib import Path

from trajectory_retention.contracts import CheckStatus, Evaluation


def frontier_version(base: str, source: str) -> str:
    return "task-frontier.v4:" + hashlib.sha256(
        base.encode() + Path(source).read_bytes() + Path(__file__).read_bytes()).hexdigest()


def retain_frontier(evaluation: Evaluation, *, positive_checks: Mapping[str, Sequence[str]],
                    milestone_ids: Mapping[str, str]) -> Evaluation:
    """Keep current positive observations; expose completeness independently.

    Only mapped positive behavioral PASS, or a translator-validated explicit
    boundary on PASS/FAIL/ERROR_TASK, supplies positive reachability. INFRA_ERROR,
    NOT_RUN, generic crashes and explicitly invalid evidence never do. Negative
    reachability requires every supporting check to give trustworthy False.
    Completion, weights, groups, score validity and raw checks are unchanged.
    """
    checks = {c.check_id: c for c in evaluation.checks}
    envelope_trusted = evaluation.details.get("evidence_valid") is not False
    caps = []
    for cap in evaluation.capabilities:
        relevant = [checks[cid] for cid in cap.check_ids]
        positive = set(positive_checks[cap.capability_id])
        if not positive <= set(cap.check_ids):
            raise ValueError("Positive support lies outside the capability's frozen checks")
        def trusted(c):
            return (envelope_trusted and c.status not in (CheckStatus.INFRA_ERROR, CheckStatus.NOT_RUN)
                    and c.details.get("evidence_valid") is not False)
        # A crash never establishes a positive behavioral boundary. It may,
        # however, carry the translator's explicit proof that execution failed
        # before that boundary. Do not erase that known negative observation.
        observed = any(trusted(c) and not c.details.get("generic_task_crash", False) and (
            (c.status == CheckStatus.PASS and c.check_id in positive)
            or (c.status in (CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.ERROR_TASK)
                and c.details.get("reachable") is True)) for c in relevant)
        unreachable = bool(relevant) and all(trusted(c) and c.details.get("reachable") is False
                                            for c in relevant)
        reach = True if observed else False if unreachable else None
        caps.append(replace(cap, reachable=reach,
                            milestone_ids=(milestone_ids[cap.capability_id],) if reach is True else ()))
    unknown = [c.capability_id for c in caps if c.reachable is None]
    return replace(evaluation, capabilities=tuple(caps),
                   reachable_milestones=tuple(m for c in caps for m in c.milestone_ids),
                   details={**evaluation.details, "frontier_policy_version": "task-frontier.v4",
                            "frontier_positive_evidence_valid": any(c.reachable is True for c in caps),
                            "frontier_complete": not unknown, "unknown_reachability": unknown,
                            "frontier_scope": "Current trusted per-check evidence, independent of completion validity; detector owns ever-reached."})
