<!--
DRAFT stalled-told variant of review.units-v1-draft (user, 2026-11-03): identical except the review turn opens with
"This run looks stalled." Reason: in the pilot every stall review chose change_nothing, with no failed submission in
view; this variant tells the agent the run is stalled before it reviews. Not ratified.
-->
<!--
DRAFT fixed-unit review prompt, version units-v1-draft (arm set v3, 2026-10-21). Not ratified. Built from v0-draft for the user's
rulings of 2026-10-16/17:
- Q1: the review turn's own output (reasoning and the decision) stays in the continuing context, so the line saying
  "this review turn is removed from your context" is replaced.
- Flat Ledger: no approach layer, so the Ledger view description, the drop sentence and the worked examples no longer
  mention approaches, an approach table or survives-rewind statuses.
- Q24(a): the review request carries only the system message, the task message and this message; the view below is
  the agent's whole record, so nothing else is needed.
Arms 2 and 3 share every section except `view:*` and `ledger_examples` (arm 2 only). {REVIEW_PROMPT} and
{OUTPUT_PLACEHOLDER} are the v2_text constants. Lines starting `=== section:` / `=== field:` and these comments are
never shown to the agent. No section may say or imply that the agent is stuck.
2026-11-02 (user): the submission sentence no longer promises hidden-test counts; neutral-feedback hand-backs list
submissions without counts and the prompt names no tests or grades.
-->
=== section: base ===
This run looks stalled.

{REVIEW_PROMPT}

This is a one-turn review. Below is {VIEW_DESCRIPTION}, then a read-only snapshot of the files and processes in your environment. No commands run in this turn: the only action available is one call to the `review_decision` tool. Afterwards you continue working on the task. What you write in this turn, your reasoning included, stays in your context; the review material below does not.

What the call can do:
- Rewind: {REWIND_TEXT}
- Drop: remove units from your context. Drops change only what you see in your context, never files. A dropped step disappears with its reasoning, command and output. A dropped output keeps the command and shows "{OUTPUT_PLACEHOLDER}" in place of the output.{DROP_EXTRA}
- Keep: if the call rewinds your context, you can keep units from the steps the rewind undoes. Kept units come back to you in one message right after the rewind{KEEP_EXTRA}.
- Change nothing: set change_nothing to true and leave the other fields empty.

Rewind, drops and keeps can be combined in one call. Whatever you choose, you will still see a list of your submissions so far.

{TOOL_EXAMPLE}
{LEDGER_EXAMPLES}
===== {VIEW_TITLE} =====
{VIEW_TEXT}
===== FILES AND PROCESSES (read-only snapshot) =====
{SNAPSHOT}
===== END OF REVIEW MATERIAL =====

Make exactly one review_decision call.
=== section: rewind:per_channel ===
move any of four parts back to the state at an earlier point, each on its own: the workspace files (workspace), the rest of the environment such as installed packages, services and files outside the workspace (runtime), your context (context), and the command outputs in your context (outputs: the outputs of every step after the point are replaced, the commands stay). Point k is the state right after step k; point 0 is the start of the task. Leave a part as null to keep it as it is now.
=== section: rewind:coupled ===
move the workspace files, the rest of the environment (installed packages, services, files outside the workspace), your context and the command outputs together back to the state at one earlier point (rewind_to). Point k is the state right after step k; point 0 is the start of the task. Use null to rewind nothing.
=== section: view:ledger_flat ===
=== field: VIEW_DESCRIPTION ===
the Ledger of your work so far (your full context, split into numbered steps, with labels on parts of each step)
=== field: VIEW_TITLE ===
LEDGER
=== field: DROP_EXTRA ===
 You can also drop a single labelled item, which removes that exact text from the step it is in, from your reasoning and your messages alike.
=== field: KEEP_EXTRA ===
, with their labels
=== section: view:context ===
=== field: VIEW_DESCRIPTION ===
your full context so far, split into numbered steps
=== field: VIEW_TITLE ===
YOUR CONTEXT
=== field: DROP_EXTRA ===

=== field: KEEP_EXTRA ===

=== section: tool_example ===
An example of a call (the IDs and numbers are made up; use the IDs and step numbers in the material below):
review_decision({EXAMPLE_ARGS})
{EXAMPLE_EXPLANATION}
Another example: review_decision({NOTHING_ARGS}) keeps everything exactly as it is.
=== section: example:per_channel ===
This would return the workspace, the runtime and your context to how they were right after step 14, leave the outputs part as it is, remove step 9 and the output of step 6 from your context, and hand back step 17, which the rewind undoes, in the message after the rewind.
=== section: example:coupled ===
This would return the workspace, the runtime, your context and the command outputs to how they were right after step 14, remove step 9 and the output of step 6 from your context, and hand back step 17, which the rewind undoes, in the message after the rewind.
=== section: ledger_examples ===
How to read the Ledger (labels come from an automatic pass and can be wrong):
- Reasoning and messages are split into fixed units. A unit can be a claim, plan, idea, observation, or have multiple labels.
- A claim can be contradicted later, confirmed later, or unchecked. Status uses only observations available in this record.
- Plans describe intended actions; ideas describe possibilities; observations describe what the agent saw. Commands show which earlier units they rest on. File changes show the commands they carry out.
- Sk selects a whole step; Ok selects all outputs of step k. Labelled reasoning/message IDs such as S18.r1 select the exact text of that item. Other IDs are references only.
- Example: rewind the context to point 14 and keep S18.r1 to carry that labelled item from step 18 forward. Dropping S18.r1 removes its exact text from step 18, never files.
