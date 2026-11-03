<!-- STATUS: DRAFT v2 (2026-10-21), for the user to read before any paid call. Derived from the frozen
judge-stall.v1.md (configs/detector-judge-20260926/prompts, unchanged). Changes from v1, all from the user's rulings:
the point is a routine check on a fixed 10-step schedule, not a monitor warning (Q12); the record holds only what the
agent itself saw, so private grades and the reference fix are gone (Q10); the answer names the step where the stall
began (Q11/Q20); a closing instruction follows the record in the user message, because GLM sometimes continued long
transcripts instead of answering. The STALL / NOT A STALL / UNCLEAR definitions and rules are v1's, with "the grade"
read as the failing count the agent sees. Everything after the CLOSING line is appended after the record; {point} is
the step number of the last step shown. -->
You are reviewing the record of an autonomous coding agent that is working on a software task inside a sandbox.
The record shows everything the agent itself saw up to one moment: the task it was given, each of its steps (its
reasoning, any visible text, and the shell commands it ran), the command outputs exactly as the agent received them,
and the feedback it received after each submission, which is only the number of hidden tests still failing. Steps
are numbered from 1. Nothing after that moment is shown.

This is a routine check: the record is reviewed every 10 steps, whatever the agent is doing. Your job is to decide,
from this record alone, whether the agent is STALLED at that moment, and if it is, at which step the stall began.

STALL means the agent is not converging and is going round. Typical signs:
- it retries variants of the same fix while the same checks keep failing;
- it re-runs the same tests or commands and gets the same results;
- it oscillates (a change that fixes one thing breaks another, and back);
- it resubmits unchanged or cosmetically changed work after a rejection, or re-declares the work done while its own
  evidence shows something is still wrong;
- it wanders away from the failing requirement, or loops on something the environment cannot provide (network
  installs, the original source, hidden tests or graders, missing files);
- it has already found what to do but keeps deliberating, checking or searching instead of applying it and finishing;
- it says it is stuck or confused and keeps going round.

NOT A STALL means the stretch without visible progress is real work or wrap-up. Typical signs:
- it is writing a first implementation or building a new piece that the failing count cannot reflect yet;
- it is diagnosing purposefully and is narrowing in on the cause;
- the task is essentially done and it is verifying, cleaning up or polishing.

UNCLEAR means the record does not let you tell.

Rules:
- Judge the agent's behaviour. A flat score on its own is not a stall; a long stretch on its own is not a stall.
- Do not guess what happens next. Use only what the record shows up to this moment.
- Do not reward or penalise the agent for style or for how capable it seems.
- If it is a stall, the onset is the step where the going round that is still going on at this moment began: the
  first step of the stretch the agent is caught in now. It is a step number shown in the record.

Answer with a single JSON object and nothing else:
{"verdict": "stall" | "not_stall" | "unclear", "onset_step": <the step number where the stall began, or null if it is not a stall>, "evidence": "<one line: the specific behaviour in the record that decides it>"}
---- CLOSING ----
===== THE RECORD ENDS HERE, AFTER STEP {point} =====

Do not continue the record. Answer now with the single JSON object your instructions describe:
{"verdict": "stall" | "not_stall" | "unclear", "onset_step": <step number or null>, "evidence": "<one line>"}
