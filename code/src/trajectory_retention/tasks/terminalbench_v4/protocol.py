"""Submission protocol of the terminalbench-v4 lane: submit, hear only the failing-test count, keep working.

The same v2 protocol as the terminalbench-v2 lane (copied; lanes never import each other). The runner's native-submission
loop (``mining/runner.py``) grades the submitted W and R and asks the adapter for a verdict. Here:

- the criterion of a failing submission is the sha256 of its sorted failing hidden-test ids (JSON list encoding), so
  two submissions repeat exactly when the same tests fail;
- the feedback is ``SUBMISSION_FEEDBACK`` with the failing count only (never test names, messages or the share);
- a submission passes when no hidden test fails (and Harbor's reward is 1); the run then ends;
- the run ends after 3 consecutive submissions with the same failing set;
- an invalid grade (grader infrastructure error) gives no verdict; the runner treats that as a technical failure, never
  as a worker failure.

Worker-visible text comes only from ``experiments/v2_text.py``; there is no fallback wording.
"""
from __future__ import annotations

import hashlib
import json

from trajectory_retention.experiments.v2_text import SUBMISSION_FEEDBACK as FEEDBACK_TEMPLATE

VERSION = "terminalbench-v4.submit-continue.v1"
REPEATED_FAILURE_LIMIT = 3
CRITERION_RULE = "sha256(json.dumps(sorted(failing hidden-test ids))) over frozen check ids whose status is not PASS"
FEEDBACK_SOURCE = "trajectory_retention.experiments.v2_text.SUBMISSION_FEEDBACK"
if FEEDBACK_TEMPLATE.count("{n}") != 1:
    raise ValueError("Submission feedback must contain exactly one {n}")

PROTOCOL = {
    "version": VERSION,
    "submission_feedback": "the count of hidden tests still failing, in the fixed template; nothing else",
    "feedback_template": FEEDBACK_TEMPLATE,
    "feedback_template_sha256": hashlib.sha256(FEEDBACK_TEMPLATE.encode("utf-8")).hexdigest(),
    "feedback_source": FEEDBACK_SOURCE,
    "success": "no hidden test fails (every frozen check PASS and Harbor's reward 1); the run ends",
    "criterion_rule": CRITERION_RULE,
    "repeated_failure_termination": REPEATED_FAILURE_LIMIT,
    "repeated_failure_rule": "the same failing set (criterion) on 3 consecutive submissions",
    "invalid_grade": "no verdict (grader infrastructure error); never shown to the worker",
    "graded_state": "captured W and captured filesystem R of the submitted state; baseline R is refused",
    "source_evidence": ["docs/research/experiment.md (Natural runs: submission loop)",
                        "src/trajectory_retention/tasks/terminalbench_v2/protocol.py (the same rules, terminalbench-v2)"],
}


def criterion(failing: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(failing)).encode("utf-8")).hexdigest()


def verdict(evaluation) -> dict:
    """Worker-deliverable verdict from a grade of the submitted state; private fields never reach the worker."""
    if evaluation is None or not evaluation.valid:
        return {"status": "unavailable", "reason": "invalid or missing grade"}
    details = evaluation.details or {}
    failing = details.get("failing_tests")
    if not isinstance(failing, list):
        return {"status": "unavailable", "reason": "grade lacks the failing-test list"}
    private = {"failing_count": len(failing), "failing_tests": sorted(failing),
               "tests_total": details.get("checks_total"), "share_passed": evaluation.score,
               "evaluation_id": getattr(evaluation, "evaluation_id", None)}
    if not failing:
        if evaluation.full_solve is not True:
            return {"status": "unavailable", "reason": "every check passed but Harbor's reward is not 1", **private}
        return {"status": "pass", "criterion_id": None, "feedback": None, **private}
    return {"status": "fail", "criterion_id": criterion(failing),
            "feedback": FEEDBACK_TEMPLATE.replace("{n}", str(len(failing))), **private}
