"""Development fixture: the terminalbench-v2 submission protocol shape around an already loaded adapter.

Only for engine end-to-end checks while the terminalbench-v2 lane (track L) is built; never for study data. It wraps a
loaded adapter (e.g. terminalbench-v1) without touching its sealed sources: criterion id = sha256 of the sorted failing
hidden-test IDs, feedback = SUBMISSION_FEEDBACK with the count only, pass when nothing fails, and 3 identical failing
sets in a row end the run. The wrapped task record is the inner adapter's, unchanged.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .v2_text import SUBMISSION_FEEDBACK

FIXTURE_VERSION = "count-feedback-fixture/v1"
PROTOCOL = {
    "version": FIXTURE_VERSION, "fixture": True,
    "submission_feedback": "SUBMISSION_FEEDBACK with the count of failing hidden tests only",
    "success": "no hidden test fails",
    "repeated_failure_termination": 3,
    "repeated_failure_rule": "the same failing set on three consecutive submissions",
    "criterion": "sha256 of the JSON list of sorted failing hidden-test IDs",
}


def failing_set(evaluation: Any) -> list[str]:
    return sorted(c.check_id for c in evaluation.checks if c.status.value != "PASS")


def criterion_id(failing: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(failing)).encode("utf-8")).hexdigest()


class CountFeedbackFixture:
    native_submission_protocol = PROTOCOL

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def native_submission_verdict(self, evaluation: Any) -> dict:
        if evaluation is None or not evaluation.valid or not evaluation.checks:
            return {"status": "unavailable"}
        failing = failing_set(evaluation)
        total = len(evaluation.checks)
        share = (total - len(failing)) / total
        if not failing:
            return {"status": "pass", "criterion_id": None, "feedback": None, "failing_count": 0, "share_passed": share,
                    "failing_ids": []}
        return {"status": "fail", "criterion_id": criterion_id(failing),
                "feedback": SUBMISSION_FEEDBACK.format(n=len(failing)), "failing_count": len(failing),
                "share_passed": share, "failing_ids": failing}


FIXTURES = {FIXTURE_VERSION: CountFeedbackFixture}
