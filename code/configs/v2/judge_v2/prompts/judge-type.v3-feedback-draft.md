<!-- DRAFT v3 feedback compatibility, 2026-10-27. Same stall/type definitions as v2;
only the description of available feedback changes for neutral-feedback sources. Preserve v2 for old records.
2026-11-02: also names the v2 neutral line; votes collected before this edit used the earlier wording. -->
You are reviewing the record of an autonomous coding agent that is working on a software task inside a sandbox.
The record shows everything the agent itself saw up to one moment: the task it was given, each of its steps (its
reasoning, any visible text, and the shell commands it ran), the command outputs exactly as the agent received them,
and the feedback actually delivered after submissions. This may be only a generic line such as "Continue working." or
"The task is not complete yet. Continue working."; historical records may contain failing-test counts. Do not infer private scores from generic feedback. Steps are numbered from 1.
Nothing after that moment is shown.

A routine check has already judged that the agent is stalled at that moment. Classify what kind of stall it is:
- "stuck": nothing it tries is working and it is going round (retrying variants, re-running, oscillating, wandering,
  looping on what the environment cannot provide). It neither believes it is done nor already holds a fix that its
  own evidence supports.
- "over-commit": it believes a wrong or incomplete fix is done. For example it resubmits unchanged or cosmetically
  changed work after a rejection, re-verifies and re-declares completion, or ignores its own evidence that something
  is still wrong.
- "under-commit": it already has a fix that its own evidence supports (it has named it or written it, and its own
  checks or reasoning in the record point to it) but won't apply it or won't finish. For example it keeps hunting for
  hidden tests or the original source, keeps deliberating or re-checking, or leaves the fix unapplied or unsubmitted.
- "unclear": the record does not separate these, or the agent is not in any of these states.

Answer with a single JSON object and nothing else:
{"type": "stuck" | "over-commit" | "under-commit" | "unclear", "evidence": "<one line: the specific behaviour in the record that decides it>"}
---- CLOSING ----
===== THE RECORD ENDS HERE, AFTER STEP {point} =====

Do not continue the record. Answer now with the single JSON object your instructions describe:
{"type": "stuck" | "over-commit" | "under-commit" | "unclear", "evidence": "<one line>"}
