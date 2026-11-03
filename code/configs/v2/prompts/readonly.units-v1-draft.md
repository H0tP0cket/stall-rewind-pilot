<!--
DRAFT read-only review prompt, version v1-draft (arm set v3, arms 5 and 6, 2026-10-21). Not ratified: the user's
rulings fix what arms 5 and 6 do (one turn showing the flat Ledger, or the plain context, with no rewind, drop or
keep tool; then the agent continues; its output from the turn stays in the context) but not this wording.
Written to match the arm 2 and 3 prompt (review.v1-draft.md) as closely as possible, so that arm 5 differs from
arm 2 (and arm 6 from arm 3) only in what the agent can do. {READ_ONLY_PROMPT} is the v2_text constant. Arm 5 also
gets `ledger_reading`: the label explanations of review.v1-draft without anything about rewinding, dropping or
keeping. Lines starting `=== section:` / `=== field:` and these comments are never shown to the agent. No section
may say or imply that the agent is stuck.
-->
=== section: base ===
{READ_ONLY_PROMPT}

This is a one-turn review. Below is {VIEW_DESCRIPTION}, then a read-only snapshot of the files and processes in your environment. No commands run in this turn and no tool is available. Reply in plain text. Afterwards you continue working on the task. What you write in this turn, your reasoning included, stays in your context; the review material below does not.
{LEDGER_READING}
===== {VIEW_TITLE} =====
{VIEW_TEXT}
===== FILES AND PROCESSES (read-only snapshot) =====
{SNAPSHOT}
===== END OF REVIEW MATERIAL =====

Reply in plain text.
=== section: view:ledger_flat ===
=== field: VIEW_DESCRIPTION ===
the Ledger of your work so far (your full context, split into numbered steps, with labels on parts of each step)
=== field: VIEW_TITLE ===
LEDGER
=== section: view:context ===
=== field: VIEW_DESCRIPTION ===
your full context so far, split into numbered steps
=== field: VIEW_TITLE ===
YOUR CONTEXT
=== section: ledger_reading ===
How to read the Ledger (automatic labels can be wrong):
- Reasoning and messages are split into fixed units labelled claim, plan, idea, observation, or multiple labels.
- Claim status is contradicted later, confirmed later, or unchecked, using only observations in this record.
- Commands show what earlier units they rest on; file changes show which commands they carry out.
