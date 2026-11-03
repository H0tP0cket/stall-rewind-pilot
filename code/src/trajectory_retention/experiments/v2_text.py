"""Every worker-visible string of the v2 study, one named constant each (provisional wording, brief-interfaces.md).

The submission wording follows the user's ruling of 2026-10-07: nothing about the loop up front; the count comes back
like the next prompt in an ordinary coding session. Change it here only. No text anywhere may say or imply that
the agent seems stuck. Templates are versioned so a run records exactly which text its worker saw.
"""
from __future__ import annotations

import hashlib

from ..execution.agent import INSTANCE_TEMPLATE

SUBMIT_RULE = "To submit, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` alone."
SUBMISSION_FEEDBACK = "Hidden tests still failing: {n}."
REVIEW_PROMPT = "You can review your work and rewind or drop any part of it, or change nothing."
GENERIC_LINE = "Part of your earlier work was undone or removed from view."
HANDBACK_HEADER = "Notes carried over:"
GRADER_RECORD_LINE = "Submission {i} (step {k}): {n} hidden tests failing."
OUTPUT_PLACEHOLDER = "[output removed]"

# The shared instance template's submission line, replaced whole by SUBMIT_RULE (it restates how to submit).
_V1_SUBMIT_LINE = ("- To submit the current task or stage, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` alone. "
                   "Submission ends work on the current task or stage.")
if INSTANCE_TEMPLATE.count(_V1_SUBMIT_LINE) != 1:
    raise RuntimeError("Shared instance template no longer carries the v1 submission line")
INSTANCE_TEMPLATE_V2 = INSTANCE_TEMPLATE.replace(_V1_SUBMIT_LINE, "- " + SUBMIT_RULE)
INSTANCE_TEMPLATE_V2_VERSION = "v2-submit-loop/v2"
# Selected per run by RunConfig.metadata["instance_template"]; absent keeps the shared template (old runs unchanged).
INSTANCE_TEMPLATES = {INSTANCE_TEMPLATE_V2_VERSION: INSTANCE_TEMPLATE_V2}

# The review turn's message and its decision tool schema come from track S's prompt files (configs/v2/prompt_texts.py);
# the engine's own worker-visible strings for the review and note turns are these.
REVIEW_TOOL_NAME = "review_decision"
REVIEW_FORMAT_ERROR = ("Your response did not contain one valid review_decision call: {error} Respond with exactly one "
                       "review_decision tool call.")
NOTE_FORMAT_ERROR = "No tools are available in this turn. Answer in plain text only."
# Hand-back parts after the header, generic line and grader record (provisional formatting).
HANDBACK_ITEM = "{text}"
# The read-only files and process snapshot shown in the review turn (track S's prompt prints it as {SNAPSHOT}).
SNAPSHOT_WORKSPACE = "Files in the workspace ({n}):"
SNAPSHOT_RUNTIME = "Files added or changed outside the workspace ({n}):"
SNAPSHOT_DELETED = "Removed outside the workspace ({n}):"
SNAPSHOT_PROCESSES = "Processes ({n}):"
SNAPSHOT_NONE = "(none)"

# Arm set v3 (user, 2026-10-16/17): read-only review turns in arms 5 and 6, and the review turn's own output kept in
# the continuing context in arms 2, 3, 5 and 6. DRAFT wording (not ratified): the read-only opening line, and how a
# kept review_decision call reads once it is plain text in the context (the tool is not offered after the turn).
READ_ONLY_PROMPT = "You can review your work. Nothing can be changed in this turn."
REVIEW_DECISION_KEPT = "review_decision({arguments})"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binding() -> dict:
    """The exact worker-visible strings in force, for run records."""
    names = ("SUBMIT_RULE", "SUBMISSION_FEEDBACK", "REVIEW_PROMPT", "GENERIC_LINE", "HANDBACK_HEADER",
             "GRADER_RECORD_LINE", "OUTPUT_PLACEHOLDER", "REVIEW_FORMAT_ERROR", "NOTE_FORMAT_ERROR",
             "HANDBACK_ITEM", "SNAPSHOT_WORKSPACE", "SNAPSHOT_RUNTIME", "SNAPSHOT_DELETED", "SNAPSHOT_PROCESSES",
             "SNAPSHOT_NONE", "READ_ONLY_PROMPT", "REVIEW_DECISION_KEPT")
    return {"version": "v2-text/3", **{n: globals()[n] for n in names},
            "instance_template_sha256": {k: sha256(v) for k, v in INSTANCE_TEMPLATES.items()}}
