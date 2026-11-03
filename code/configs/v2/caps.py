"""Calibration and freeze for the v2 arms (track S): the shared new-token and submission caps, the arm 6 recipe cap,
the main-stall rewind mode and the prompt versions, all in one calibration.json. No model calls.

  screen-caps --study R --screen-json F        caps from the screen branches (rule CAP_RULE: DRAFT, not decided)
  tuning      --study R                        recipe cap = median arm 5 note length on the tuning stalls (ratified);
                                               rewind mode for main stalls from mismatch rates (ratified: coupled
                                               only if more than 25% of tuning review branches mismatched)
  freeze      --study R [--prompts JSON]       stamp the calibration; arms_v2 main/harm/injection need a frozen one
The calibration lists every DRAFT rule it rests on; spec sets built from it inherit that list.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from study_common import (PROMPT_VERSIONS, REVIEW_VIEWS, iter_results, load_specset, read_json, runs_dir,
                          sha256_file, write_json)
from trajectory_retention.analysis import v2_stats as S

CAP_RULE = {"name": "cap_rule", "status": "DRAFT, not decided by the user",
            "source_arm": "1_continue", "population": "admitted stalls (main and tuning)", "quantile": 0.9,
            "round_tokens_to": 1000,
            "rule": "new_token_cap = the quantile of new_tokens over the screen's continue branches of admitted "
                    "stalls, rounded up; submission_cap = the same quantile of their submission counts, rounded up"}
RECIPE_RULE = {"name": "recipe_cap_rule", "status": "ratified (contract: arm 6)",
               "rule": "median arm 5 note length (note_tokens) over the tuning stalls' arm 5 branches"}
COUPLING_RULE = {"name": "coupling_rule", "status": "ratified (contract: arm 2) for the mismatch threshold",
                 "mismatch_share_over": 0.25, "failed_command_share_over": None, "invalid_call_share_over": None,
                 "rule": "main stalls use coupled rewind (one point for W, context, R and O) only if more than 25% "
                         "of the tuning stalls' review branches (arms 2 and 3 pooled) were mismatched; failed-command "
                         "and invalid-call rates are reported, but no threshold for them has been ratified; solve "
                         "rates are never read"}
CURRENT_ARM_SET = "v2"


def cal_path(study: Path) -> Path:
    return study / "calibration.json"


def load(study: Path) -> dict:
    c = read_json(cal_path(study), {"version": "calibration.v2"})
    if cal_path(study).exists() and c.get("arm_set", "v2") != CURRENT_ARM_SET:
        raise SystemExit("calibration belongs to a different arm set")
    c["arm_set"] = CURRENT_ARM_SET
    if c.get("frozen_at"):
        raise SystemExit(f"calibration was frozen at {c['frozen_at']}; refusing to change it")
    return c


def screen_caps(study: Path, screen_json: Path) -> dict:
    scr = read_json(screen_json)
    keep = {s["stall_id"] for s in scr["stalls"] if s.get("admitted")}
    tokens, subs, used = [], [], []
    for m_path in sorted((study / "specsets").glob("screen*.json")):
        m = load_specset(m_path)
        for block, job, res, _ in iter_results(m, runs_dir(study, m["prefix"])):
            if res is None or job["arm"] != CAP_RULE["source_arm"] or block["unit_id"] not in keep:
                continue
            tokens.append(res["new_tokens"])
            subs.append(len(res.get("submissions") or []))
            used.append(job["branch_id"])
    if not tokens:
        raise SystemExit("no finished continue branches of admitted stalls")
    q = CAP_RULE["quantile"]
    step = CAP_RULE["round_tokens_to"]
    caps = {"new_token_cap": int(math.ceil(S.quantile(tokens, q) / step) * step),
            "submission_cap": max(1, int(math.ceil(S.quantile(subs, q))))}
    c = load(study)
    c |= {"caps": caps, "cap_rule": CAP_RULE,
          "cap_inputs": {"screen_json": {"path": str(screen_json.resolve()), "sha256": sha256_file(screen_json)},
                         "branches": len(used), "new_tokens": sorted(tokens), "submissions": sorted(subs)}}
    c["draft_rules_used"] = sorted(set(c.get("draft_rules_used", [])) | ({CAP_RULE["name"]}
                                   if CAP_RULE["status"].startswith("DRAFT") else set()))
    write_json(cal_path(study), c)
    return {"caps": caps, "branches": len(used)}


def tuning(study: Path) -> dict:
    notes, review = [], []
    sets = []
    for m_path in sorted((study / "specsets").glob("tuning-r*.json")):
        m = load_specset(m_path)
        sets.append(m["prefix"])
        for _, job, res, _ in iter_results(m, runs_dir(study, m["prefix"])):
            if res is None:
                continue
            if job["arm"] == "5_rewind_own_summary" and res.get("note_tokens") is not None:
                notes.append(res["note_tokens"])
            if job["arm"] in REVIEW_VIEWS:
                review.append(res.get("flags") or {})
    if (CURRENT_ARM_SET == "v2" and not notes) or not review:
        raise SystemExit("tuning needs finished arm 5 branches and review-arm branches")
    rates = {k: sum(bool(f.get(k)) for f in review) / len(review) for k in ("mismatch", "failed_command", "invalid_call")}
    trips = [k for k, t in (("mismatch", COUPLING_RULE["mismatch_share_over"]),
                            ("failed_command", COUPLING_RULE["failed_command_share_over"]),
                            ("invalid_call", COUPLING_RULE["invalid_call_share_over"])) if t is not None and rates[k] > t]
    c = load(study)
    c |= {"recipe_cap_tokens": int(round(S.quantile(notes, 0.5))) if notes else None, "recipe_rule": RECIPE_RULE if notes else None,
          "recipe_inputs": {"tuning_sets": sets, "arm5_note_tokens": sorted(notes)},
          "rewind_mode_main": {"mode": "coupled" if trips else "per_channel", "tripped_by": trips, "rates": rates,
                               "review_branches": len(review), "rule": COUPLING_RULE}}
    write_json(cal_path(study), c)
    return {"recipe_cap_tokens": c["recipe_cap_tokens"], "rewind_mode_main": c["rewind_mode_main"]}


def freeze(study: Path, prompts: dict | None) -> dict:
    c = load(study)
    required = ("caps", "rewind_mode_main") + (("recipe_cap_tokens",) if CURRENT_ARM_SET == "v2" else ())
    missing = [k for k in required if k not in c]
    if missing:
        raise SystemExit(f"cannot freeze: missing {missing}")
    pv = PROMPT_VERSIONS | (prompts or {})
    draft = set(c.get("draft_rules_used", []))
    if any(v.endswith("-draft") for v in pv.values()):
        draft.add("prompt_drafts")
    c |= {"prompt_versions": pv, "draft_rules_used": sorted(draft),
          "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    write_json(cal_path(study), c)
    return {"frozen_at": c["frozen_at"], "draft_rules_used": c["draft_rules_used"], "caps": c["caps"],
            "recipe_cap_tokens": c.get("recipe_cap_tokens"), "rewind_mode_main": c["rewind_mode_main"]["mode"]}


def pilot_freeze(study: Path, prompts: dict | None) -> dict:
    """Freeze development inputs without claiming that the later tuning gate passed."""
    if CURRENT_ARM_SET != "v3":
        raise SystemExit("pilot-freeze requires --arm-set v3")
    c = load(study)
    if "caps" not in c:
        raise SystemExit("pilot-freeze requires screening-derived caps")
    if "rewind_mode_main" in c:
        raise SystemExit("pilot-freeze refuses to replace an existing tuning decision")
    c.update(development_only=True,
             rewind_mode_main={"mode": "per_channel", "review_branches": 0, "rates": None,
                               "tripped_by": [], "source": "User-approved development pilot, 2026-10-26; independent channel rewinds. Final tuning gate not assessed."})
    write_json(cal_path(study), c)
    return freeze(study, prompts)


def main(argv: list[str] | None = None) -> dict:
    global CURRENT_ARM_SET, PROMPT_VERSIONS, REVIEW_VIEWS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("screen-caps", "tuning", "freeze", "pilot-freeze"))
    ap.add_argument("--study", required=True)
    ap.add_argument("--arm-set", choices=("v2", "v3"), default="v2")
    ap.add_argument("--screen-json")
    ap.add_argument("--prompts", help="JSON {arm: prompt_version} to freeze instead of the current drafts")
    a = ap.parse_args(argv)
    CURRENT_ARM_SET = a.arm_set
    if a.arm_set == "v3":
        from study_common import PROMPT_VERSIONS_V3
        PROMPT_VERSIONS = {k: v.replace(".v1-draft", ".units-v1-draft") for k, v in PROMPT_VERSIONS_V3.items()}
        REVIEW_VIEWS = {k: None for k in ("2_ledger_review", "3_context_review")}
        CAP_RULE.update(quantile=0.95, status="ratified: Q8, 95th percentile")
    study = Path(a.study)
    if a.cmd == "screen-caps":
        r = screen_caps(study, Path(a.screen_json or study / "screen" / "screen.json"))
    elif a.cmd == "tuning":
        r = tuning(study)
    elif a.cmd == "pilot-freeze":
        r = pilot_freeze(study, json.loads(a.prompts) if a.prompts else None)
    else:
        r = freeze(study, json.loads(a.prompts) if a.prompts else None)
    print(json.dumps(r, indent=1))
    return r


if __name__ == "__main__":
    main()
