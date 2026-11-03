"""Healthy checkpoints for the harm and injection tests, and the injection donors at each (track S). No model calls.

Contract: about 20 healthy checkpoints from runs that went on to solve, after real progress, before any stall, at
most 2 per task; 8 injection conditions whose content is real, from failed runs of the same task (or earlier in the
same run), spliced as the most recent steps through a separate injection composition type (track E). How "after real
progress" and each condition's donor are picked is not ratified: the rules below are DRAFT constants, and every
healthy.json / injection_donors.json records that it used them.

  select --store S [--store S2 ...] --healthy-root H [--target 20]
      healthy.json per checkpoint, from the live detector record (P = measurable progress, S = stall warning)
  donors --healthy-root H --stalls-root D
      injection_donors.json per healthy checkpoint: one payload per condition, or the reason it is unavailable.
      Needs ledger.json (track D) in the healthy dir and in the stall dirs of failed runs of the same task.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import steps_v2
from study_common import (INJECTION_CONDITIONS, MAX_HEALTHY_PER_TASK, healthy_dirs, read_json, short_hash,
                          stall_dirs, stall_id_for, write_json)

HEALTHY_RULE = {"name": "healthy_checkpoint_rule", "status": "DRAFT",
                "rule": "one checkpoint per run, from runs whose terminal reason is a passing submission: the "
                        "checkpoint of the last progress (P) annotation that precedes the run's first stall (S) "
                        "annotation and the first full private score (1.0), and is branchable; at most 2 per task "
                        "(runs in run-id order); stop at the target count"}
SOLVED_REASONS = ("native_success", "passing_submission")
DONOR_RULES = {"name": "injection_donor_rules", "status": "DRAFT", "max_splice_steps": 3,
               "small_neutral_tokens": 500, "large_neutral_tokens": 6000,
               "success_claim_regex": r"\b(pass(es|ed|ing)?|fixed|works?|working|succeed\w*|success\w*|done|complete\w*|resolved|correct)\b",
               "rules": {
                   "continue": "nothing is spliced (control at the same checkpoint)",
                   "false_success_claim": "the step holding a claim item marked contradicted later whose quote reads "
                                          "as a success claim (success_claim_regex), from a failed run of the task",
                   "failed_approach_promising": "the last max_splice_steps steps of a non-current approach of a "
                                                "failed run, ending one step before its last step (failure not shown)",
                   "failed_attempt_failure_shown": "the last max_splice_steps steps of the same kind of approach, "
                                                   "ending at its last step (failure shown)",
                   "stale_output": "the latest step of the receiving run before the checkpoint whose output is "
                                   "flagged stale and not noise, spliced again as the most recent step",
                   "workspace_debris": "files a non-current approach of a failed run changed (W only; paths are "
                                       "resolved by the composition type from the donor's W between the two points)",
                   "small_neutral": "one noise-flagged step of a failed run, the one closest to small_neutral_tokens",
                   "large_neutral": "contiguous noise-flagged steps of one failed run, accumulated toward "
                                    "large_neutral_tokens"}}
PAYLOAD_VERSION = "injection-payload.v1"


# ---------- select ----------

def _annotations(run) -> list[tuple[int, dict]]:
    out = []
    for seq, d in run.ev:
        if d["kind"] == "detector" and (d.get("payload") or {}).get("role") == "output":
            o = d["payload"]["output"]
            for x in (o if isinstance(o, list) else [o]):
                if x.get("record") == "annotation":
                    out.append((seq, x))
    return out


def select(stores: list[Path], root: Path, target: int) -> dict:
    per_task, made, considered = defaultdict(int), [], []
    for sp in stores:
        st = steps_v2.store(str(sp))
        for rid in st.natural_runs():
            run = steps_v2.storeio.Run(st, rid, "v2")
            reason = str(run.terminal.get("reason") or run.terminal.get("runner_reason") or "")
            row = {"store": str(sp), "run_id": rid, "task_id": run.task}
            if not any(r in reason for r in SOLVED_REASONS):
                considered.append(row | {"skip": f"run did not end at a passing submission ({reason or 'no terminal'})"})
                continue
            ann = _annotations(run)
            if not ann:
                considered.append(row | {"skip": "no live detector record"})
                continue
            first_s = min((q for q, a in ann if a.get("label") == "S"), default=None)
            first_full = min((q for q, a in ann if a.get("label") == "P" and (a.get("score") or 0) >= 1.0), default=None)
            cut = min(x for x in (first_s, first_full, 10 ** 12) if x is not None)
            ps = [a for q, a in ann if a.get("label") == "P" and q < cut]
            pick = None
            for a in reversed(ps):
                cp = st.checkpoint_json(a["target_checkpoint_id"])
                if cp.get("branchable"):
                    pick = (a, cp)
                    break
            if pick is None:
                considered.append(row | {"skip": "no branchable progress checkpoint before the first stall warning"})
                continue
            if per_task[run.task] >= MAX_HEALTHY_PER_TASK:
                considered.append(row | {"skip": f"task already has {MAX_HEALTHY_PER_TASK} healthy checkpoints"})
                continue
            if len(made) >= target:
                considered.append(row | {"skip": "target reached"})
                continue
            a, cp = pick
            _, k = steps_v2.checkpoint_point(str(sp), rid, a["target_checkpoint_id"])
            hid = "healthy--" + stall_id_for(run.task, rid, k)
            h = {"version": "healthy.v2", "healthy_id": hid, "store": str(Path(sp).resolve()), "run_id": rid,
                 "task_id": run.task, "checkpoint_id": a["target_checkpoint_id"], "step_index": k,
                 "progress_annotation": a.get("annotation_id"), "first_stall_warning_seq": first_s,
                 "selection": HEALTHY_RULE, "draft_rules_used": [HEALTHY_RULE["name"]]}
            write_json(root / hid / "healthy.json", h)
            per_task[run.task] += 1
            made.append(hid)
            considered.append(row | {"picked": hid})
    write_json(root / "selection.json", {"rule": HEALTHY_RULE, "picked": made, "considered": considered})
    return {"picked": len(made), "per_task": dict(per_task), "considered": len(considered)}


# ---------- donors ----------

def _failed_ledgers(stalls_root: Path, task_id: str) -> list[dict]:
    out = []
    for d in stall_dirs(stalls_root):
        s = read_json(d / "stall.json")
        if s["task_id"] != task_id or s.get("later_solved") or not (d / "ledger.json").exists():
            continue
        out.append({"stall": s, "ledger": read_json(d / "ledger.json"), "dir": str(d)})
    return out


def _payload(cond: str, h: dict, scope: dict, *, channel: str | None, donor: dict | None = None,
             debris: dict | None = None, size: dict | None = None) -> dict:
    receiver = {"store": h["store"], "run_id": h["run_id"], "task_id": h["task_id"], "checkpoint_id": h["checkpoint_id"],
                "point": h["step_index"], "task_version": {k: scope.get(k) for k in ("requirement_version",
                                                                                      "evaluator_version")}}
    ns = "inj-" + short_hash(f"{h['healthy_id']}|{cond}|{json.dumps(donor, sort_keys=True)}", 8)
    return {"version": PAYLOAD_VERSION, "condition": cond, "channel": channel, "receiver": receiver, "donor": donor,
            "splice": None if donor is None or channel != "context" else {
                "position": "after_last_step", "content": "worker_visible_only", "include_reasoning": True,
                "rename_tool_call_ids": True, "id_namespace": ns},
            "debris": debris, "size": size or {"target_tokens": None, "measured_tokens": None},
            "record_notice": True, "rules": DONOR_RULES["name"]}


def _donor(f: dict, first: int, last: int, rule: str, evidence: dict, *, same_run: bool = False) -> dict:
    s = f["stall"]
    return {"store": s["store"], "run_id": s["run_id"], "task_id": s["task_id"], "same_run": same_run,
            "steps": [first, last], "cut": {"after_point": first - 1, "through_point": last},
            "task_version": None, "selection": {"rule": rule, "evidence": evidence, "ledger_dir": f.get("dir")}}


def _versions_match(scope: dict, donor_scope: dict) -> bool:
    return all(scope.get(k) == donor_scope.get(k) for k in ("requirement_version", "evaluator_version"))


def donors_for(h: dict, stalls_root: Path) -> dict:
    rules = DONOR_RULES
    scope = steps_v2.run_scope(h["store"], h["run_id"])
    failed = []
    for f in _failed_ledgers(stalls_root, h["task_id"]):
        ds = steps_v2.run_scope(f["stall"]["store"], f["stall"]["run_id"])
        if _versions_match(scope, ds):
            f["scope"] = ds
            failed.append(f)
    out: dict[str, dict] = {"continue": {"payload": _payload("continue", h, scope, channel=None)}}
    none = lambda why: {"unavailable": why}
    claim_rx = re.compile(rules["success_claim_regex"], re.I)
    n = rules["max_splice_steps"]

    def with_version(p: dict, f: dict) -> dict:
        if p.get("donor"):
            p["donor"]["task_version"] = {k: f["scope"].get(k) for k in ("requirement_version", "evaluator_version")}
        return p

    pick = None
    for f in failed:
        for it in f["ledger"].get("items", []):
            if it.get("type") == "claim" and it.get("claim_status") == "contradicted later" and claim_rx.search(it.get("quote", "")):
                pick = (f, it)
                break
        if pick:
            break
    out["false_success_claim"] = none("no contradicted success claim in a failed run of this task") if not pick else {
        "payload": with_version(_payload("false_success_claim", h, scope, channel="context", donor=_donor(
            pick[0], pick[1]["step"], pick[1]["step"], rules["rules"]["false_success_claim"],
            {"item": pick[1].get("id"), "quote": pick[1].get("quote")})), pick[0])}
    approach = None
    for f in failed:
        aps = f["ledger"].get("approaches", [])
        for ap in aps[:-1]:
            if ap.get("last_step", 0) - ap.get("first_step", 0) >= 1:
                approach = (f, ap)
                break
        if approach:
            break
    if approach:
        f, ap = approach
        lo_p, hi_p = max(ap["first_step"], ap["last_step"] - n), ap["last_step"] - 1
        lo_f, hi_f = max(ap["first_step"], ap["last_step"] - n + 1), ap["last_step"]
        ev = {"approach": ap.get("id"), "status": ap.get("status"), "failure_quote": ap.get("failure_quote")}
        out["failed_approach_promising"] = {"payload": with_version(_payload(
            "failed_approach_promising", h, scope, channel="context",
            donor=_donor(f, lo_p, hi_p, rules["rules"]["failed_approach_promising"], ev)), f)}
        out["failed_attempt_failure_shown"] = {"payload": with_version(_payload(
            "failed_attempt_failure_shown", h, scope, channel="context",
            donor=_donor(f, lo_f, hi_f, rules["rules"]["failed_attempt_failure_shown"], ev)), f)}
        out["workspace_debris"] = {"payload": with_version(_payload(
            "workspace_debris", h, scope, channel="workspace",
            donor=_donor(f, ap["first_step"], ap["last_step"], rules["rules"]["workspace_debris"], ev),
            debris={"donor_point_before": ap["first_step"] - 1, "donor_point_after": ap["last_step"],
                    "path_rule": "files created or modified in the donor between the two points, excluding paths "
                                 "that exist in the receiver's workspace", "paths": None}), f)}
    else:
        for c in ("failed_approach_promising", "failed_attempt_failure_shown", "workspace_debris"):
            out[c] = none("no failed run of this task with a finished non-current approach")
    own = Path(h["dir"]) / "ledger.json"
    if own.exists():
        steps = [s for s in read_json(own).get("steps", []) if s["k"] <= h["step_index"]]
        stale = [s for s in steps if not s["flags"].get("output_current", True) and not s["flags"].get("noise")]
        if stale:
            s = stale[-1]
            donor = {"store": h["store"], "run_id": h["run_id"], "task_id": h["task_id"], "same_run": True,
                     "steps": [s["k"], s["k"]], "cut": {"after_point": s["k"] - 1, "through_point": s["k"]},
                     "task_version": {k: scope.get(k) for k in ("requirement_version", "evaluator_version")},
                     "selection": {"rule": rules["rules"]["stale_output"], "evidence": {"step": s["k"]}}}
            out["stale_output"] = {"payload": _payload("stale_output", h, scope, channel="context", donor=donor)}
        else:
            out["stale_output"] = none("no stale, non-noise output before the checkpoint")
    else:
        out["stale_output"] = none("ledger.json missing in the healthy dir")
    sized = []
    for f in failed:
        rs = steps_v2.run_steps(f["stall"]["store"], f["stall"]["run_id"])["steps"]
        noise = {s["k"] for s in f["ledger"].get("steps", []) if s["flags"].get("noise")}
        for s in rs:
            if s["k"] in noise:
                sized.append((f, s["k"], steps_v2.approx_tokens(steps_v2.step_text(s))))
    if sized:
        f, k, tok = min(sized, key=lambda x: abs(x[2] - rules["small_neutral_tokens"]))
        out["small_neutral"] = {"payload": with_version(_payload(
            "small_neutral", h, scope, channel="context", donor=_donor(f, k, k, rules["rules"]["small_neutral"], {}),
            size={"target_tokens": rules["small_neutral_tokens"], "measured_tokens": tok}), f)}
        best = None
        for f in failed:
            run = [(k, t) for ff, k, t in sized if ff is f]
            for i in range(len(run)):
                tot, j = 0, i
                while j < len(run) and (j == i or run[j][0] == run[j - 1][0] + 1) and tot < rules["large_neutral_tokens"]:
                    tot += run[j][1]
                    j += 1
                cand = (abs(tot - rules["large_neutral_tokens"]), f, run[i][0], run[j - 1][0], tot)
                best = cand if best is None or cand[0] < best[0] else best
        _, f, lo, hi, tot = best
        out["large_neutral"] = {"payload": with_version(_payload(
            "large_neutral", h, scope, channel="context", donor=_donor(f, lo, hi, rules["rules"]["large_neutral"], {}),
            size={"target_tokens": rules["large_neutral_tokens"], "measured_tokens": tot}), f)}
    else:
        out["small_neutral"] = out["large_neutral"] = none("no noise-flagged step in a failed run of this task")
    return {"version": "injection-donors.v1", "healthy_id": h["healthy_id"], "rules": rules,
            "draft_rules_used": [rules["name"]], "failed_runs_considered": [f["stall"]["stall_id"] for f in failed],
            "conditions": {c: out[c] for c in INJECTION_CONDITIONS}}


def donors(healthy_root: Path, stalls_root: Path) -> dict:
    summary = {}
    for d in healthy_dirs(healthy_root):
        h = read_json(d / "healthy.json") | {"dir": str(d)}
        r = donors_for(h, stalls_root)
        write_json(d / "injection_donors.json", r)
        summary[h["healthy_id"]] = {c: ("ok" if "payload" in v else v["unavailable"]) for c, v in r["conditions"].items()}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--store", action="append", required=True)
    p.add_argument("--healthy-root", required=True)
    p.add_argument("--target", type=int, default=20)
    p = sub.add_parser("donors")
    p.add_argument("--healthy-root", required=True)
    p.add_argument("--stalls-root", required=True)
    a = ap.parse_args()
    if a.cmd == "select":
        r = select([Path(s) for s in a.store], Path(a.healthy_root), a.target)
    else:
        r = donors(Path(a.healthy_root), Path(a.stalls_root))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
