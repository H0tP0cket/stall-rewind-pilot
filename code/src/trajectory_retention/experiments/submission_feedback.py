"""Versioned worker feedback; grading and stopping still use the private verdict.

Absent policy preserves historical adapter feedback. Opting in affects future deliveries only;
it never rewrites feedback already present in a source trace.

Rules (the delivered text is fixed per rule; nothing else is ever added):
- v1 (user, 2026-10-27): "Continue working."
- v2 (user, 2026-11-02): "The task is not complete yet. Continue working." States that the task is unfinished
  without test, grade or submission words, so a failed submission is not read as a harness glitch.
New runs and branches bind v2 (SETTING). v1 stays valid so runs that started under it keep it on resume.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

RULE_V1 = "neutral-submission-feedback/v1"
RULE_V2 = "neutral-submission-feedback/v2"
TEXTS = {RULE_V1: "Continue working.", RULE_V2: "The task is not complete yet. Continue working."}
RULE = RULE_V2
SETTING = {"rule": RULE}
# The v1 text. The sealed pilot lanes (terminalbench_v4_pilot_legacy/training) import it as their native verdict
# text and check its bytes; the delivered text always comes from the bound rule (delivery below).
TEXT = TEXTS[RULE_V1]
SUBMISSION_RECORD = "Submission {i} (step {k})."


def validate(spec):
    if spec is None:
        return None
    if not isinstance(spec, Mapping) or set(spec) != {"rule"} or spec["rule"] not in TEXTS:
        raise ValueError(f"submission_feedback must be one of {[{'rule': r} for r in TEXTS]!r}")
    return dict(spec)


def text(spec) -> str:
    return TEXTS[validate(spec)["rule"]]


def binding(spec):
    if validate(spec) is None:
        return None
    t = text(spec)
    return {**dict(spec), "text": t, "text_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "submission_record": SUBMISSION_RECORD}


def delivery(config, native_feedback, wrapper=None):
    policy = binding(config.metadata.get("submission_feedback"))
    if policy is not None:
        if wrapper is not None:
            raise ValueError("Neutral submission feedback cannot be combined with a feedback wrapper")
        return policy["text"], policy
    return (native_feedback if wrapper is None else wrapper["template"].replace("{feedback}", native_feedback)), None
