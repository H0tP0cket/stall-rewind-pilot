"""Recorded reviews of leak-scan contents and comments findings (leak.py), keyed by finding id.

A finding is resolved only by a "not_a_leak" verdict with its reason; anything else, or no entry, fails the scan. The
reviewer read the shared lines or comment text in the private explain file (artifacts/v2/lane/tasks/<task>/gate/
leak-explain.json). Reviews written during the agent build are marked as such and need a human check before the freeze.
"""
from __future__ import annotations

AGENT = "Claude (v2 build, track L, 2026-10-07); needs a human check"

REVIEWS: dict[str, dict] = {
    "2ebb06858e724e40": {
        "task": "llm-inference-batching-scheduler", "kind": "comments", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "The public baseline packer's comment 'Assign each request to the smallest representative >= its "
                  "aligned seq' describes the baseline's own step; the reference's docstring reuses that phrase for the "
                  "same step. The reference's new content (its bucket choice and plans) is not in the starter."},
    "c12e07420b0436c8": {
        "task": "tune-mjcf", "kind": "contents", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "The shared lines are the public /app/eval.py, which instruction.md names as the iteration aid ('The "
                  "/app/eval.py script can help you iterate'). They implement the checks the instruction states (same "
                  "full physics state within atol=1e-5, no NaN or Inf, at most 60% of the reference time); every "
                  "other hidden check is also stated in the instruction. No tuned model or tuning step is revealed."},
    "e28c4e06eda7585d": {
        "task": "tune-mjcf", "kind": "comments", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "eval.py's comment 'target percentage of reference model time' on its 0.6 constant: the public "
                  "iteration aid of c12e07420b0436c8, restating the instruction's 60% target."},
    "3a3f7468cc1c2522": {
        "task": "tune-mjcf", "kind": "comments", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "eval.py's comment 'number of runs to average timing' on its run-count constant: the public "
                  "iteration aid of c12e07420b0436c8; it says how eval.py times the models, not how to tune them."},
    "5d85d232ad5083c2": {
        "task": "regex-chess", "kind": "contents", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "The shared lines are the move-checking harness of the public /app/check.py, which instruction.md "
                  "names ('You can look at the provided check.py to verify if your solution is correct. You will be "
                  "tested on other positions as well.'). check.py replays one game; the hidden tests replay "
                  "different positions, none of which is in the starter. No regex or replacement is revealed."},
    "81b937c7e0592144": {
        "task": "regex-chess", "kind": "comments", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "check.py's harness comment 'Normalize line endings and remove any BOM' (PGN parsing); the public "
                  "checker of 5d85d232ad5083c2."},
    "b0cac3acf6c4daf7": {
        "task": "regex-chess", "kind": "comments", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "check.py's harness comment 'Test each position in the game'; the public checker of "
                  "5d85d232ad5083c2."},
    "28b73d89641895e2": {
        "task": "regex-chess", "kind": "comments", "verdict": "not_a_leak", "reviewer": AGENT,
        "reason": "check.py's harness comment 'Get FEN before making the move'; the public checker of "
                  "5d85d232ad5083c2."},
}


def reviews_for(task: str) -> dict[str, dict]:
    return {k: v for k, v in REVIEWS.items() if v["task"] == task}
