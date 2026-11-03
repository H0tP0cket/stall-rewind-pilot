<!--
DRAFT arm 5 note prompt, version v0-draft (track S, 2026-10-07). For the tuning rounds only; nothing here is ratified.
The note turn runs at the stall checkpoint with the agent's full context, no tool, plain-text output; then the arm 4
undo; the note comes back in the hand-back. {UNDONE_SPAN} and {POINT_TEXT} are filled by prompt_texts.render_note
from the frozen rewind point (sections span:* below). No line may say or imply that the agent is stuck.

Source (AgentRewind, ~/.cache/mettlebench/paper/paper.txt):
  l.252-255: "the agent selects a checkpoint and generates a new rewind memory that summarizes useful information from
             the current trajectory, such as falsified hypotheses and alternative strategies."
  l.1194:    "// Tool 2: rewind to a chosen target, carrying a memory note."
  l.1206-1208: "memory_summary": { "type": "string", "description": "Required memory to carry back to the selected
             point. Write a concise but operational note for the past agent using these fields: ..." },
  l.1213-1216: Figure 8 caption: "... the longer natural-language descriptions are abbreviated with "...", and the
             complete definitions are in the released code."
The abbreviated field list, from the released code (~/.cache/mettlebench/harness/replay/backtrack.py l.373-437,
memory_summary description): "failed_check, observed_loop, why_local_repair_failed, root_cause, evidence_observed,
assumptions_to_recheck, rewind_target_reason, selected_candidate_safety_check, do_not_repeat, corrected_strategy,
do_not_call_before, must_satisfy_before_mutation, mutation_preflight, first_3_tool_calls, and success_invariant.
Treat root_cause as a hypothesis, not an unquestionable fact; evidence_observed should name the concrete
verifier/tool/file evidence supporting it, and assumptions_to_recheck should name uncertainties the next fork must
verify before relying on them. ... Include literal filenames, paths, commands, verifier keywords, or required evidence
when known."

Adaptation (neutral wording, our setting):
  - failed_check -> "Submissions" (what each submission reported; the agent only ever sees failing counts).
  - observed_loop / why_local_repair_failed -> "What was tried" / "What the results showed" (no loop or failure framing).
  - root_cause + evidence_observed -> "Cause hypothesis", marked as a hypothesis, with observed evidence.
  - assumptions_to_recheck -> kept. do_not_repeat -> "Already tried", with the result each gave.
  - corrected_strategy + first_3_tool_calls -> "Plan", including the first three commands.
  - success_invariant -> "Done when".
  - Dropped: rewind_target_reason and selected_candidate_safety_check (the point is fixed by the arm, not chosen),
    do_not_call_before / must_satisfy_before_mutation / mutation_preflight (tool-policy fields of that harness).
  - No length cap (contract: arm 5 has no cap; length is logged).
-->
=== section: body ===
{UNDONE_SPAN} will be undone: your workspace files, the rest of your environment and your context will return to how they were {POINT_TEXT}. Before that happens, write a note to yourself. The note will be handed back to you right after the undo. Apart from the record of your submissions and how many hidden tests failed at each, it is the only thing from those turns that you will see again.

Write the note as plain text under these headings (leave a heading empty if it does not apply):
- Submissions: what each submission so far reported, and which checks you believe still fail.
- What was tried: the commands, edits and results in those turns, in order.
- What the results showed: why the changes did or did not change the number of failing tests, as far as you can tell.
- Cause hypothesis: your current hypothesis about what the hidden tests need, marked as a hypothesis, with the evidence you actually observed (file names, commands, outputs).
- Assumptions to recheck: things you have assumed but not verified.
- Already tried: edits or approaches you would not need to try again, with the result each gave.
- Plan: what you would do next from that earlier point, including the first three commands you would run.
- Done when: what must be true for the hidden tests to pass.

Use literal file names, paths, commands and test names where you know them, and base the note only on what you actually observed. No commands run in this turn; reply with the note only.
=== section: span:partial ===
=== field: UNDONE_SPAN ===
Your last {N_UNDONE} turns
=== field: POINT_TEXT ===
just before the first of those turns
=== section: span:all ===
=== field: UNDONE_SPAN ===
All of your turns so far
=== field: POINT_TEXT ===
at the start of the task
