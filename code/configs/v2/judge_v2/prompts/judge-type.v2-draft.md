<!-- STATUS: DRAFT v2 (2026-10-21), for the user to read before any paid call. Derived from the frozen
judge-type.v1.md (unchanged). The type is a description only; it never selects anything. Changes from v1: no
reference fix and no private grades (Q10), so "under-commit" is read from the agent's own evidence instead of from a
comparison with the reference fix; the point is a routine 10-step check, not a monitor warning; a closing instruction
follows the record. Everything after the CLOSING line is appended after the record; {point} is the last step shown. -->
You are reviewing the record of an autonomous coding agent that is working on a software task inside a sandbox.
The record shows everything the agent itself saw up to one moment: the task it was given, each of its steps (its
reasoning, any visible text, and the shell commands it ran), the command outputs exactly as the agent received them,
and the feedback it received after each submission, which is only the number of hidden tests still failing. Steps
are numbered from 1. Nothing after that moment is shown.

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
