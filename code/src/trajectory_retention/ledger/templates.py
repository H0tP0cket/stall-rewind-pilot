"""Every string the Ledger renderers put in front of an agent: arm 2 and arm 3 review views, kept-item labels and the
arm 6 recipe. DRAFT: the tuning rounds may change these; any change bumps TEMPLATES_VERSION, which every rendered
view and recipe records. No string here may say or imply "you seem stuck".

Unit IDs appear only inside ``⟦ ⟧`` so that they cannot collide with text in the context (``checks`` scans for them).
"""

from __future__ import annotations

TEMPLATES_VERSION = "ledger-templates/v1-draft"

# ---- tokens (recipe cap; recorded with every recipe) ----
TOKENIZER = "tiktoken:o200k_base"
_ENC = None


def count_tokens(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return len(_ENC.encode(text, disallowed_special=()))


def truncate_tokens(text: str, cap: int) -> str:
    count_tokens("")
    return _ENC.decode(_ENC.encode(text, disallowed_special=())[:max(0, cap)])


# ---- unit markers ----
OPEN, CLOSE = "⟦", "⟧"
STEP_ID, OUTPUT_ID, APPROACH_ID, ITEM_ID = "S{k}", "O{k}", "A{n}", "I{n}"
UNIT_RX = r"⟦/?([SOAI]\d+)(?: [^⟧]*)?⟧"


def mark(unit_id: str) -> str:
    return f"{OPEN}{unit_id}{CLOSE}"


def tag_open(item_id: str, label: str) -> str:
    return f"{OPEN}{item_id} {label}{CLOSE}"


def tag_close(item_id: str) -> str:
    return f"{OPEN}/{item_id}{CLOSE}"


# ---- views ----
LEGEND_BASE = ("Steps are numbered from 1. Point k is the state after step k; point 0 is the start, before step 1. "
               "⟦Sk⟧ marks step k and ⟦Ok⟧ the outputs of step k.")
LEGEND_LEDGER = (LEGEND_BASE + " ⟦An⟧ marks an approach: consecutive steps pursuing one line of attack. "
                 "⟦In type⟧…⟦/In⟧ marks an exact quote and its label. A fact marked 'survives rewind' would still "
                 "hold if files and runtime returned to point {rewind_point}, where the current approach began.")
# The flat Ledger (arm set v3, user 2026-10-16/17): no approach layer, so no approach marks, table, statuses or rewind
# point in the legend; facts carry no survives-rewind status (it is measured against the approach boundary).
FLAT_TEMPLATES_VERSION = "ledger-templates-flat/v1-draft"
LEGEND_LEDGER_FLAT = LEGEND_BASE + " ⟦In type⟧…⟦/In⟧ marks an exact quote and its label."
STEP_HEADER_LEDGER_FLAT = "{sid} step {k}{flags}"
APPROACHES_HEADER = "APPROACHES"
APPROACH_ROW = "{aid} steps {first}–{last} · {status}{current} · started with {start} · failure evidence: {failure}"
APPROACH_START = '{iid} "{quote}"'
APPROACH_FAILURE = '"{quote}" (step {k})'
NONE = "none"
CURRENT = " (current)"
TASK_HEADER = "TASK"
CONTEXT_HEADER = "CONTEXT"
STEP_HEADER = "{sid} step {k}"
STEP_HEADER_LEDGER = "{sid} step {k} · approach {aid}{flags}"
FIELD_HEADERS = {"reasoning": "reasoning:", "message": "message:", "command": "command {n}:",
                 "feedback": "submission feedback:", "format_error": "format error:", "user": "user message:"}
OUTPUT_HEADER = "{oid}output of command {n}{meta}:"
OUTPUT_PART = "({part})"
FLAG_TEXT = {"workspace_change": " · files changed", "noise": " · display only",
             "stale": " · output stale", "mixed": " · some output stale"}
OUTCOME_TEXT = {"tests_fail": " · tests: {tests}", "tests_pass": " · tests: {tests}", "error": " · exit code {codes}"}
ITEM_LABELS = {"decision": "decision", "fact": "fact", "claim": "claim", "idea": "idea"}
STATUS_TEXT = {"survives": "survives rewind", "lost": "lost on rewind",
               "contradicted_later": "contradicted later, step {e}", "confirmed_later": "confirmed later, step {e}",
               "unchecked": "unchecked", "tried_later": "tried later, step {e}", "never_tried": "never tried"}
STATUS_TEXT_NO_STEP = {"contradicted_later": "contradicted later", "confirmed_later": "confirmed later",
                       "tried_later": "tried later"}

# ---- labeller input (not shown to agents) ----
LABELLER_STEP_HEADER = "=== Step {k} ==="
LABELLER_SECTIONS = {"earlier": "EARLIER STEPS (context only: the agent's own words and commands; outputs not shown)",
                     "target": "TARGET STEPS (label these)", "all": "STEPS",
                     "later": "LATER STEPS (context only; do not label)"}
LABELLER_ELIDED = "[output of {n} characters not shown here]"
# labeller input, render v2 (protocol v2.1): each step is wrapped in its own open and close lines, and every part of
# it (reasoning, message, command, output, feedback) carries its step number, so a quote's step can be read off the
# line above it; tool outputs often print "=== name ===" banners that look like the v1 step header.
LABELLER_V2_RENDER_VERSION = "ledger-labeller-render/v2"
LABELLER_V2_STEP_OPEN = "<step {k}>"
LABELLER_V2_STEP_CLOSE = "</step {k}>"
LABELLER_V2_PART_PREFIX = "[step {k}] "

# ---- kept items (hand-back; labels kept, IDs dropped) ----
KEPT_ITEM = '{label} (step {k}{status}): "{quote}"'
KEPT_STEP = "step {k}:\n{text}"
KEPT_OUTPUT = "output of step {k}:\n{text}"
KEPT_APPROACH = 'approach of steps {first}–{last} ({status}): started with "{start}"; failure evidence: {failure}'

# ---- arm 6 recipe ----
RECIPE_FAILURE = 'Failure evidence from the undone steps: "{quote}" (step {k})'
RECIPE_FACTS_HEADER = "Facts from the undone steps that still hold:"
RECIPE_FACT = '- "{quote}" (step {k})'
RECIPE_IDEAS_HEADER = "Ideas noted earlier that were never tried:"
RECIPE_IDEA = '- "{quote}" (step {k})'
