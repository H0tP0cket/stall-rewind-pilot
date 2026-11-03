<!--
DRAFT review prompt, version v0-draft (track S, 2026-10-07). For the tuning rounds only; nothing here is ratified.
Arms 2 and 3 share every section except `view:*` and `ledger_examples` (arm 2 only); the tool-usage example is the
same for both. The opening line is the REVIEW_PROMPT constant and {OUTPUT_PLACEHOLDER} the OUTPUT_PLACEHOLDER
constant (both from trajectory_retention.experiments.v2_text). {VIEW_TEXT} is the review view's `text` exactly;
{SNAPSHOT} is the read-only files/process snapshot the engine prints. Unit IDs in the example follow the prompt
index's unit_id_format, which must match the IDs the Ledger renderers put in the views.
Lines starting `=== section:` / `=== field:` and these comments are never shown to the agent.
No section may say or imply that the agent is stuck.
The `outputs` channel wording assumes one compiler rule for an independently chosen O point (outputs of steps after
the point are replaced, commands kept); it must match track E's rule before tuning.
-->
=== section: base ===
{REVIEW_PROMPT}

This is a one-turn review. Below is {VIEW_DESCRIPTION}, then a read-only snapshot of the files and processes in your environment. No commands run in this turn: the only action available is one call to the `review_decision` tool. Afterwards this review turn is removed from your context and you continue working on the task.

What the call can do:
- Rewind: {REWIND_TEXT}
- Drop: remove units from your context. Drops change only what you see in your context, never files. A dropped step disappears with its reasoning, command and output. A dropped output keeps the command and shows "{OUTPUT_PLACEHOLDER}" in place of the output.{DROP_EXTRA}
- Keep: if the call rewinds your context, you can keep units from the steps the rewind undoes. Kept units come back to you in one message right after the rewind{KEEP_EXTRA}.
- Change nothing: set change_nothing to true and leave the other fields empty.

Rewind, drops and keeps can be combined in one call. Whatever you choose, you will still see every submission so far with the number of hidden tests that failed at it.

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
=== section: view:ledger ===
=== field: VIEW_DESCRIPTION ===
the Ledger of your work so far (your full context, split into numbered steps and grouped into approaches, with a table of approaches first and labels on parts of each step)
=== field: VIEW_TITLE ===
LEDGER
=== field: DROP_EXTRA ===
 You can also drop a whole approach, or a single labelled item, which removes that exact text from the step it is in, from your reasoning and your messages alike.
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
How to read the Ledger (worked examples; the labels come from an automatic pass over your context and can be wrong):
- The approach table lists each approach with its first and last step, its status, and a short quote from an output that bears on how it ended. To undo one approach and everything after it, rewind to the point just before its first step: for an approach that starts at step 15, that is point 14.
- A step's labels mark exact quotes inside it. A decision is a choice you made about what to do. A fact is something an output showed; "survives rewind: yes" means it stays true after the workspace is rewound (for example, which version of a library is installed), "no" means it describes a state the rewind removes.
- A claim is something you stated about the code or the tests. "Contradicted later" means a later output in your context disagrees with it; "confirmed later" means a later output agrees; "unchecked" means nothing later bears on it. Dropping a claim removes that exact text from the step it is in.
- An idea is something you considered doing. "Never tried" means no later step acted on it; "tried later" means a later step did. Ideas from steps a rewind undoes can be kept.
- Example: if approach {APPROACH_3} runs from step 15 to step 22 and step 18 holds a fact item {ITEM_18} "the parser rejects tabs" marked survives rewind: yes, then rewinding to point 14 and keeping {ITEM_18} undoes {APPROACH_3} and hands the fact back to you.
