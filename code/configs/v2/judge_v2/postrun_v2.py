"""Stall judge v2 over a natural-run store: stall records, episodes and branch sources (user rulings Q8, Q10-Q12,
Q19, Q20; rules in judging.py). Beside the earlier postrun (configs/detector-judge-20260926/postrun.py), which stays
as it was; detector v8.1 keeps running live in natural runs as a record-only marker and is not read here.

For every finished natural-v2/v1 run in the store (or --run substrings):
  1. judge points at steps 10, 20, ... up to the last step; the record at each is the worker's own context after
     that step (record.py; never grades, evaluations or references);
  2. 3 votes per point (GLM-5.3 Flash by default, configs/v2/judge_v2/judge-glm.json); onset P, fallback, episodes;
  3. one type call per episode (description only);
  4. a branch source per episode: the checkpoint at the episode's first confirmed judge point (the stall moment,
     where arms 1, 2, 3, 5 and 6 start), plus `stall_onset_point` = P - 1, the point just before the stall's first
     step ran (point 0 = the initial checkpoint when P = 1), which is arm 4's approved rewind target.
     An episode is selected only when both checkpoints are branchable; otherwise it is flagged with the reason.
Outputs in --out: stall_records.json (every run: every point with votes, onsets, P and its source, costs; episodes),
branch_sources.json (the selected sources), records/<run>/pNNNN.txt (exactly the record text each point was judged
on), calls/ (receipts; reused on rerun, so the command is idempotent). With --write-store-records, each completely
judged run also gets durable, immutable store records: one `stall_judge_v2` per run (id = run id) and one
`stall_episode` per episode (id = run id + "/E<n>"); a changed record is refused, never overwritten
(configs/v2/arms_v2.py collect-episodes reads them).

Paid calls need --allow-draft-prompts while the v2 prompts are DRAFT. --scripted <json> replaces the model with
fixed answers (client.Scripted) for engine checks: nothing leaves the machine and nothing is spent.

Usage: python configs/v2/judge_v2/postrun_v2.py --store <store> --out <dir> [--run SUBSTR ...]
         [--judge-settings configs/v2/judge_v2/judge-glm.json] [--cap-usd 5] [--parallel 3]
         [--scripted spec.json] [--allow-draft-prompts] [--write-store-records] [--retry-no-answer]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import client  # noqa: E402
import judging  # noqa: E402
import record as R  # noqa: E402

V2_NATURAL = "natural-v2/v1"
DEFAULT_SETTINGS = HERE / "judge-glm.json"
RUN_KIND, EPISODE_KIND = "stall_judge_v2", "stall_episode"


def run_key(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest()[:12]


def judge_run(run: R.RunView, settings, transport, budget, out: Path, prompts: dict, pool, *, points=None,
              retry_no_answer=False, with_type=True) -> dict:
    """Judge one run at its points (default: every 10 steps); returns the run's stall record."""
    key = run_key(run.run_id)
    calls = out / "calls"
    rows, records = [], {}
    status = "complete"
    for k in (run.points() if points is None else points):
        try:
            rec = run.record(k)
        except LookupError as e:
            rows.append({"point": k, "error": "no_record", "detail": str(e), "confirmed": False})
            continue
        records[k] = rec
        txt = out / "records" / key / f"p{k:04d}.txt"
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(judging.user_message(rec.render(), prompts["stall"][1], k))
        row = judging.judge_point(settings, transport, budget, calls, key, rec, prompts, pool,
                                  retry_no_answer=retry_no_answer)
        row["record_file"] = str(txt.relative_to(out))
        rows.append(row)
        if row["skipped_spend_guard"]:
            status = "spend_guard_stopped"
            break
    subs = []
    for rec in records.values():  # the latest record holds every submission up to its point
        subs = rec.submissions() if len(rec.submissions()) >= len(subs) else subs
    eps = judging.episodes(rows, subs) if status == "complete" else []
    passing = [s for s in run.passing_submission_steps() if s is not None]
    for e in eps:
        e["episode_id"] = f"{run.run_id}/E{e['episode']}"
        first = next(r for r in rows if r["point"] == e["first_point"])
        e["first_point_receipts"] = first["receipts"]
        e["cost_usd"] = round(sum(r.get("cost_usd", 0.0) for r in rows if r["point"] in e["points"]), 6)
        if with_type:
            t = judging.judge_type(settings, transport, budget, calls, key, records[e["first_point"]], prompts)
            e.update(t)
            e["cost_usd"] = round(e["cost_usd"] + t["type_cost_usd"], 6)
        # branch source = the stall moment (the first confirmed judge point); arm 4's rewind target = the state just
        # before step P ran (point P-1; P = 1 -> point 0, the initial checkpoint). Both must be branchable.
        def point_cp(k):
            cp = run.point_checkpoint(k)
            reason = None if cp is not None and cp.branchable else (
                f"no point checkpoint at step {k}" if cp is None else
                (cp.validation or {}).get("not_branchable_reason") or "checkpoint not branchable")
            return (cp.checkpoint_id if cp is not None else None), reason
        e["checkpoint_id"], src_reason = point_cp(e["first_point"])
        e["step_index"] = e["first_point"]
        e["stall_onset_point"] = e["onset"] - 1
        e["stall_onset_checkpoint_id"], onset_reason = point_cp(e["stall_onset_point"])
        e["source_branchable"], e["stall_onset_branchable"] = src_reason is None, onset_reason is None
        e["branchable"] = src_reason is None and onset_reason is None
        if not e["branchable"]:
            e["not_branchable_reason"] = "; ".join(
                f"{what}: {r}" for what, r in (("source (first confirmed point)", src_reason),
                                               ("stall_onset_point (arm 4 rewind target)", onset_reason)) if r)
        e["later_solved_by_submission"] = any(s > e["first_point"] for s in passing)
    for i, e in enumerate(eps):  # two episodes of a run sharing one branch source are flagged, never merged
        same = [o["episode"] for o in eps[:i] if o["checkpoint_id"] and o["checkpoint_id"] == e["checkpoint_id"]]
        e["same_source_as_episodes"] = same
    return {"run_id": run.run_id, "run_key": key, "task_id": run.task, "model": run.model, "policy": run.policy,
            "n_steps": run.n_steps, "terminal_reason": (run.terminal or {}).get("reason"), "status": status,
            "points": rows, "episodes": eps, "submissions_in_record": subs,
            "passing_submission_steps": passing,
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows) + sum(e.get("type_cost_usd", 0.0) for e in eps), 6),
            "votes": {"refused": sum(r.get("refused_votes", 0) for r in rows),
                      "no_answer": sum(r.get("no_answer_votes", 0) for r in rows),
                      "unparseable": sum(r.get("unparseable_votes", 0) for r in rows)}}


def judge_meta(settings, prompts, transport, *, neutral=False) -> dict:
    return {"version": judging.JUDGE_VERSION, "record_version": R.RECORD_VERSION, "model": settings.model,
            "provider": settings.provider, "settings": settings.path, "settings_version": settings.version,
            "transport": settings.transport,
            "votes": judging.VOTES, "attempts": settings.attempts, "point_every": R.POINT_EVERY,
            "prompts": {k: {"path": str((judging.NEUTRAL_PROMPTS if neutral else judging.PROMPTS)[k]),
                            "sha256": v[2]} for k, v in prompts.items()},
            "feedback_description": "neutral/historical-compatible" if neutral else "historical count feedback",
            "prompt_status": ("DRAFT v3 feedback compatibility; behavioural definitions unchanged" if neutral
                              else judging.PROMPT_STATUS),
            "scripted": transport.scripted, "rules": judging.RULES}


def store_records(root: Path, run_rec: dict, meta: dict, out: Path) -> dict:
    """Durable, immutable records in the natural run's own store; returns what was written or refused."""
    from trajectory_retention.store import ArtifactStore, RecordConflict
    live = ArtifactStore(root)
    res = {"written": [], "already": [], "refused": []}
    items = [(RUN_KIND, run_rec["run_id"], {"judge": meta, "receipts_root": str(out), **run_rec})]
    for e in run_rec["episodes"]:
        items.append((EPISODE_KIND, e["episode_id"], {
            "version": judging.JUDGE_VERSION, "run_id": run_rec["run_id"], "task_id": run_rec["task_id"],
            "episode": e, "judge": {k: meta[k] for k in ("model", "provider", "settings_version", "prompts",
                                                         "prompt_status", "scripted", "votes")},
            "receipts_root": str(out)}))
    for kind, rid, value in items:
        try:
            prior = live.get_record(kind, rid)
        except KeyError:
            prior = None
        if prior is not None:
            (res["already"] if prior == value else res["refused"]).append(f"{kind}:{rid}")
            continue
        try:
            live.put_record(kind, rid, value, run_id=run_rec["run_id"])
            res["written"].append(f"{kind}:{rid}")
        except RecordConflict as exc:
            res["refused"].append(f"{kind}:{rid}: {exc}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--judge-settings", default=str(DEFAULT_SETTINGS))
    ap.add_argument("--cap-usd", type=float, default=5.0)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--scripted", metavar="JSON")
    ap.add_argument("--allow-draft-prompts", action="store_true")
    ap.add_argument("--write-store-records", action="store_true")
    ap.add_argument("--retry-no-answer", action="store_true")
    ap.add_argument("--pilot-manifest", type=Path,
                    help="existing-run-pilot.v1: judge only its eligible prefixes; keep records outside source store")
    a = ap.parse_args()
    if not a.scripted and not a.allow_draft_prompts and "DRAFT" in judging.PROMPT_STATUS:
        raise SystemExit("the v2 judge prompts are DRAFT: paid calls need --allow-draft-prompts (the user reads them first)")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = client.Settings.load(Path(a.judge_settings))
    transport = client.make_transport(settings, Path(a.scripted) if a.scripted else None, out / "scripted-requests.jsonl")
    prior_spend = client.receipts_cost(out / "calls")
    budget = client.Budget(a.cap_usd, prior_spend)
    prompts = {k: judging.load_prompt(k) for k in ("stall", "type")}
    meta = judge_meta(settings, prompts, transport)
    root = Path(a.store).resolve()
    pilot = json.loads(a.pilot_manifest.read_text()) if a.pilot_manifest else None
    pilot_runs = None
    if pilot is not None:
        if pilot.get("version") != "existing-run-pilot.v1" or Path(pilot["store"]).resolve() != root:
            raise SystemExit("pilot manifest version/store mismatch")
        if a.write_store_records:
            raise SystemExit("pilot judging keeps derived records outside historical source stores")
        pilot_runs = {r["run_id"]: r for r in pilot["runs"]}
    astore = R.open_readonly(root)
    runs, skipped = [], []
    for rid in R.natural_runs(astore):
        if pilot_runs is not None and rid not in pilot_runs:
            continue
        if a.run and not any(s in rid for s in a.run):
            continue
        run = R.RunView(astore, rid)
        if run.policy != V2_NATURAL:
            skipped.append({"run_id": rid, "reason": f"run policy {run.policy!r} is not {V2_NATURAL}"})
        elif run.terminal is None:
            skipped.append({"run_id": rid, "reason": "run not finished"})
        else:
            runs.append(run)
    # A neutral source cannot be judged with a prompt claiming it received hidden-test counts.
    # Mixed batches use the compatible description; vote rules and behavioural definitions stay unchanged.
    if any(run.submission_feedback is not None for run in runs):
        from trajectory_retention.experiments.submission_feedback import validate
        for run in runs:
            validate(run.submission_feedback)
        prompts = {k: judging.load_prompt(k, neutral=True) for k in ("stall", "type")}
        meta = judge_meta(settings, prompts, transport, neutral=True)
    report, written = [], {}
    with ThreadPoolExecutor(a.parallel) as pool:
        for run in runs:
            points = None
            if pilot_runs is not None:
                entry = pilot_runs[run.run_id]
                points = [p["point"] for p in entry["points"]]
                if any(k not in run.points() or k >= entry["branch_before_step"] for k in points):
                    raise SystemExit(f"ineligible pilot points for {run.run_id}")
                for p in entry["points"]:
                    if hashlib.sha256(run.record(p["point"]).render().encode()).hexdigest() != p["sha256"]:
                        raise SystemExit(f"pilot prefix changed: {run.run_id} point {p['point']}")
            rec = judge_run(run, settings, transport, budget, out, prompts, pool, points=points,
                            retry_no_answer=a.retry_no_answer)
            report.append(rec)
            print(f"{rec['task_id'][:48]:48} {rec['run_key']} steps={rec['n_steps']:3} points={len(rec['points']):2} "
                  f"confirmed={sum(bool(p.get('confirmed')) for p in rec['points']):2} episodes={len(rec['episodes'])} "
                  f"status={rec['status']} ${rec['cost_usd']:.4f}", flush=True)
            if a.write_store_records and rec["status"] == "complete":
                written[rec["run_id"]] = store_records(root, rec, meta, out)
    sources = [{"episode_id": e["episode_id"], "store": str(root), "run_id": r["run_id"], "task_id": r["task_id"],
                "checkpoint_id": e["checkpoint_id"], "step_index": e["step_index"], "onset_step": e["onset"],
                "stall_onset_point": e["stall_onset_point"], "stall_onset_checkpoint_id": e["stall_onset_checkpoint_id"],
                "onset_source": e["onset_source"], "fallback_to": e.get("fallback_to"), "judge_point": e["first_point"],
                "points": e["points"],
                "later_solved_by_submission": e["later_solved_by_submission"], "type": e.get("type"),
                "same_source_as_episodes": e["same_source_as_episodes"]}
               for r in report if r["status"] == "complete" for e in r["episodes"]
               if e["branchable"] and not e["same_source_as_episodes"]]
    summary = {"pipeline": "postrun-stall-judge-v2", "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "store": str(root), "judge": meta, "runs": len(report), "skipped": skipped,
               "points": sum(len(r["points"]) for r in report),
               "confirmed_points": sum(bool(p.get("confirmed")) for r in report for p in r["points"]),
               "episodes": sum(len(r["episodes"]) for r in report), "branch_sources": len(sources),
               "fallback_onsets": sum(e["onset_source"] == "fallback" for r in report for e in r["episodes"]),
               "votes": {k: sum(r["votes"][k] for r in report) for k in ("refused", "no_answer", "unparseable")},
               "spend_usd_this_command": round(budget.spent - prior_spend, 6),
               "spend_usd_receipts_total": round(client.receipts_cost(out / "calls"), 6),
               "store_records": written or None}
    if pilot is not None:
        summary["pilot"] = {"manifest": str(a.pilot_manifest.resolve()),
                            "sha256": hashlib.sha256(a.pilot_manifest.read_bytes()).hexdigest(),
                            "development_only": True, "judge_status": pilot["judge"]["status"]}
    (out / "stall_records.json").write_text(json.dumps({"summary": summary, "runs": report}, indent=1))
    (out / "branch_sources.json").write_text(json.dumps(sources, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "judge"}, indent=1))


if __name__ == "__main__":
    main()
