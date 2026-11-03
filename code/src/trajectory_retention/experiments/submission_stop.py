"""Versioned run-level stop; separate from sealed task grading and submission protocols."""
from __future__ import annotations

RULE = "submission-no-improvement/v1"
SETTING = {"rule": RULE, "limit": 3, "baseline": "first_submission"}


def validate(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value != SETTING or type(value.get("limit")) is not int:
        raise ValueError(f"submission_stop must be {SETTING!r}")
    return dict(SETTING)


def progress(setting: dict | None, previous: dict | None, verdict: dict) -> dict:
    """First failure/new best counts as 1; ties and regressions extend the stretch.

    Each branch supplies only its own submissions, so a rewind starts fresh. Missing or invalid counts are
    technical failures, never invented progress. Call only after validating the task's submission verdict.
    """
    rule = validate(setting)
    if rule is None:
        return {}
    n = verdict.get("failing_count")
    status = verdict.get("status")
    if type(n) is not int or n < 0 or status not in {"pass", "fail"} \
            or (status == "pass") != (n == 0):
        raise ValueError("submission_stop requires a valid pass/fail verdict with a consistent failing_count")
    best = previous["best_failing_count"] if previous else None
    new_best = best is None or n < best
    return {"submission_stop": rule, "best_failing_count": n if new_best else best,
            "new_best_failing_count": new_best,
            "no_improvement_count": 0 if status == "pass" else
                (1 if new_best else previous["no_improvement_count"] + 1)}
