"""v5 selection: harder calibration instances chosen by published SWE-smith agent outcomes.

Declared (committed) before any v5 instance is packaged. User-ratified direction
(2026-09-26): keep SWE-smith, but pick instances that published agents mostly
failed, using other people's SWE-smith trajectories and pass rates.

Unchanged from v1/v4: dataset pin, E1-E7, R3, R4 (official image <= 4 GB), I1,
the I2 rule text, the family split by frozen family position (even position =
calibration, odd = held_out; held-out families can never be used), the reserve
idea (a held instance is replaced by the next eligible one) and G1 (every
validation grade <= 60 s). Build and gates are the unchanged v4 pipeline steps.

Changed in v5 (all declared here before packaging):

(a) Instances come from a published-trajectory hard band instead of the seeded
    3 + 2 draw. Eligible = at least one published trajectory in
    SWE-bench/SWE-smith-trajectories (``ticks`` split) AND tier C or tier A:
      C: never resolved; >= 1 Claude 3.7 Sonnet attempt; some failed attempt
         wrote >= half of the measurable bug hunks;
      A: resolved in some but not all attempts.
    Order: every C before every A. Within C: failed attempt wrote at least half
    but not all hunks first, then more Claude 3.7 attempts, more bug hunks,
    longer failed trajectories. Within A: smoothed resolve rate (r+1)/(n+2)
    lowest first, then median successful steps highest first. (The scout's
    ranking, with C moved ahead of A.) Ties: instance id.
    Tier B (always resolved, long) is dropped: our only overlap evidence (4 of
    our run tasks that appear in the published runs) puts every observed stall
    on an instance no published attempt resolved; an always-solved instance is
    the profile that gave DeepSeek under-commit ("names the bug, then hunts").
(b) Families already processed in v1-v4 at calibration positions may contribute
    further instances; every instance already drawn (candidate or reserve) or
    walked in v1-v4 is excluded. Families held in v1-v4 for a measured
    family-wide reason are excluded (the pipeline is unchanged, so the same
    failure recurs): see EXCLUDED_FAMILIES.
(c) At most PER_FAMILY_CAP admitted v5 tasks per family (variety). A held
    instance does not use a slot; the next eligible instance of the family in
    rank order takes its place.
(d) Instances on the comment-leak list are excluded: the stall-supply leak
    review ``comment-leak-external-candidates.json`` (bug patch adds comments or
    docstring text that describes or hints at the injected bug). An eligible
    instance that list does not cover is scanned with the same committed rule
    (``comment_rule.py``) and flagged lines are read the same way; the verdict
    is recorded in ``comment-leak-review.json``.

I2 verdict source, in priority: the lane's own v4 review (issue-review.json of
v4), then the external-evidence scout's review (same rule text, recorded
verbatim), then a v5 review performed for this release (issue-review.json of
v5). Every instance's verdict and source are recorded.

Stop rule: walk the ranked list; build eligible instances in rank order (within
the family cap); stop when the admitted v5 tasks reach CALIBRATION_TARGET, or
record where the list ran out.

No model or provider calls; no outcome of any of our model runs enters
selection (the published outcomes are other people's runs).
"""
from __future__ import annotations

import hashlib
import json

from .selection import CRITERIA, SEED
from .selection_v4 import CRITERIA_V4, GRADE_SECONDS_MAX, OFFICIAL_IMAGE_MAX_BYTES

CALIBRATION_TARGET = 20
PER_FAMILY_CAP = 3

TRAJECTORIES = {
    "dataset": "SWE-bench/SWE-smith-trajectories",
    "revision": "08e109b4a59eaeebf80e4675cd125d42e7ac99a4",
    "canonical_split": "ticks",
    "note": ("ticks = 25,826 trajectories on 13,500 instances; xml/tool are mostly re-renderings of the same runs, "
             "so ticks alone avoids counting one attempt several times. 'resolved' = SWE-smith full resolution. "
             "'hunks written' = the lane's I1 disclosure test fired on the agent's own text for that hunk "
             "(a proxy for partial progress)."),
}

EXCLUDED_FAMILIES = {
    "getnikola__nikola.0f4c230e": "v1 HOLD (all 5 drawn): start tree has 37 symlinks; the shared runner cannot "
                                  "restore W symlinks on this host",
    "jd__tenacity.0d40e76f": "v4 family HOLD: pip's vendored tenacity/nap.py in the image is byte-identical to "
                             "the clean source (clean-source image scan)",
    "agronholm__exceptiongroup.0b4f4937": "v4 HOLD (all 5 drawn): pytest itself imports exceptiongroup on "
                                          "Python 3.10, so a broken module makes grades invalid (runner check)",
    "iterative__dvc.1d6ea681": "v4 HOLD (all 5 drawn): export-subst .git_archival.txt breaks the start-equality "
                               "check; reference suite hit the 900 s timeout",
    "Project-MONAI__MONAI.a09c1f08": "R4: official image 17.9 GB > 4 GB",
}

TIER_B_DROPPED = (
    "Tier B (every published attempt resolved, long trajectories) is not used. Our 4 run tasks that appear in the "
    "published runs were all unresolved by Claude 3.7 (tier C profile) and 2 of them gave genuine DeepSeek stalls; "
    "the always-solved profile is the one that gave DeepSeek under-commit (it names the bug early, then hunts for "
    "hidden tests), which is the negative control, not the target population.")

CRITERIA_V5 = {
    "version": "swesmith-long.selection.v5-published-hard-band",
    "extends": CRITERIA_V4["version"],
    "extends_criteria_sha256": hashlib.sha256(json.dumps(CRITERIA_V4, sort_keys=True).encode()).hexdigest(),
    "v1_criteria_sha256": hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest(),
    "unchanged": ["dataset pin (11 shard SHA-256s)", "E1-E7 instance eligibility", "R3 standard profile",
                  f"R4 official image <= {OFFICIAL_IMAGE_MAX_BYTES} bytes", "I1 (selection_v4.issue_disclosure)",
                  "I2 rule text (CRITERIA_V4.instance_filters.I2_issue_reviewed, with its amendment)",
                  f"frozen eligible repository order (seed {SEED}) and split by position: even calibration, odd "
                  "held_out; held-out-position families are never used",
                  "reserve idea: a held instance is replaced by the next eligible instance",
                  f"G1: every validation grade <= {GRADE_SECONDS_MAX} s",
                  "build and gates: the unchanged v4 pipeline steps (v1 env -> v1 package -> v2 native start -> "
                  "v3 git format -> gates -> runner check)"],
    "a_published_hard_band": {
        "evidence": TRAJECTORIES,
        "eligible": "at least one published trajectory AND (tier C OR tier A)",
        "tier_C": "no attempt resolved; >= 1 Claude 3.7 Sonnet attempt; some failed attempt wrote >= 0.5 of the "
                  "measurable bug hunks",
        "tier_A": "resolved in at least one but not all attempts",
        "order": "all C, then all A. C: (failed attempt wrote >= 0.5 but < 1.0 of hunks) first, then Claude 3.7 "
                 "attempts desc, bug hunks desc, median failed steps desc. A: (r+1)/(n+2) asc, then median "
                 "successful steps desc. Ties: instance id asc.",
        "tier_B_dropped": TIER_B_DROPPED,
    },
    "b_family_reuse": {
        "rule": "calibration-position families (any even position of the frozen order, processed before or not) "
                "are eligible; every instance drawn (candidate/reserve) or walked in v1-v4 is excluded",
        "excluded_families": EXCLUDED_FAMILIES,
    },
    "c_family_cap": f"at most {PER_FAMILY_CAP} admitted v5 tasks per family; a held instance frees its slot",
    "d_comment_leak": ("exclude instances on the stall-supply comment-leak list "
                       "(benchmark_tasks/admission/stall-supply-v1/swesmith-leaks/comment-leak-external-candidates.json, "
                       "branch claude/stall-supply; SHA-256 recorded in selection.json); instances it does not cover are "
                       "scanned with the same comment_rule.py (SHA-256 recorded) and flagged lines read by hand: only "
                       "added comment/docstring text that describes or hints at the injected change excludes"),
    "I2_sources": ["lane v4 review (benchmark_tasks/admission/swesmith-long-v4/issue-review.json)",
                   "external-evidence scout review (same rule text; recorded verbatim in external-evidence.json)",
                   "v5 review (benchmark_tasks/admission/swesmith-long-v5/issue-review.json)"],
    "stop_rule": f"build eligible instances in rank order within the family cap; stop when admitted v5 tasks reach "
                 f">= {CALIBRATION_TARGET}; otherwise record that the eligible list was exhausted",
    "decided_before": "any v5 packaging outcome; no model outcome of ours is used",
}


def tier(row: dict) -> str:
    """Tier of one evidence row (scout columns)."""
    n, r = int(row["attempts"]), int(row["resolved_attempts"])
    if n <= 0:
        return ""
    if 0 < r < n:
        return "A"
    if r == n:
        return "B"
    written = row.get("fail_max_hunks_written_frac")
    if r == 0 and int(row["attempts_c37"]) >= 1 and written is not None and float(written) >= 0.5:
        return "C"
    return ""


def _num(value, default=0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v else v  # NaN -> default


def rank_key(row: dict) -> tuple:
    t = tier(row)
    if t == "C":
        written = _num(row.get("fail_max_hunks_written_frac"))
        partial = 0 if 0.5 <= written < 1.0 else 1
        return (0, partial, -int(row["attempts_c37"]), -int(row["bug_hunks"]), -_num(row.get("fail_steps_median")),
                row["instance_id"])
    if t == "A":
        n, r = int(row["attempts"]), int(row["resolved_attempts"])
        return (1, (r + 1) / (n + 2), -_num(row.get("succ_steps_median")), 0, 0, row["instance_id"])
    raise ValueError("only tiers C and A are ranked")


def walk(evidence: list[dict], *, family_positions: dict[str, int], drawn_before: set[str],
         reviews: dict[str, dict], comment_leak: dict[str, dict]) -> list[dict]:
    """Every evidence row in rank order with its filter outcome (no build outcome).

    evidence rows: scout columns (instance_id, repo_key, I1_passes, official_image_bytes, ...).
    reviews: instance_id -> {"verdict", "note", "source"} (already merged in priority order).
    comment_leak: instance_id -> {"leak": bool, "source", "note"}.
    """
    rows = []
    for row in evidence:
        t = tier(row)
        key = row["repo_key"]
        position = family_positions.get(key)
        step = {"instance_id": row["instance_id"], "repo_key": key, "tier": t, "family_position": position}
        if position is None:
            step["filter"] = "not in the frozen eligible family order (R1-R3)"
        elif position % 2 == 1:
            step["filter"] = "held-out-position family (never used)"
        elif t not in ("C", "A"):
            step["filter"] = "tier B (dropped)" if t == "B" else "no tier (not in the hard band)"
        elif key in EXCLUDED_FAMILIES:
            step["filter"] = "family excluded: " + EXCLUDED_FAMILIES[key]
        elif row["instance_id"] in drawn_before:
            step["filter"] = "already drawn or walked in v1-v4"
        elif _num(row.get("official_image_bytes"), float("inf")) > OFFICIAL_IMAGE_MAX_BYTES:
            step["filter"] = "R4 image size"
        elif str(row.get("I1_passes")).lower() not in ("true", "1"):
            step["filter"] = "I1 (fix code in the issue)"
        else:
            step["filter"] = None
        rows.append(step)
    ranked = sorted([r for r in rows if r["tier"] in ("C", "A") and r["filter"] != "held-out-position family (never used)"
                     and r["filter"] != "not in the frozen eligible family order (R1-R3)"],
                    key=lambda s: rank_key(next(e for e in evidence if e["instance_id"] == s["instance_id"])))
    for rank, step in enumerate(ranked, 1):
        step["rank"] = rank
        if step["filter"] is not None:
            step["status"] = "FILTERED"
            continue
        review = reviews.get(step["instance_id"])
        leak = comment_leak.get(step["instance_id"])
        step["I2"] = review
        step["comment_leak"] = leak
        if review is None:
            step["status"] = "PENDING_I2"
        elif review["verdict"] != "pass":
            step["status"] = "REJECTED_I2"
        elif leak is None:
            step["status"] = "PENDING_COMMENT_LEAK"
        elif leak["leak"]:
            step["status"] = "REJECTED_COMMENT_LEAK"
        else:
            step["status"] = "ELIGIBLE"
    return ranked
