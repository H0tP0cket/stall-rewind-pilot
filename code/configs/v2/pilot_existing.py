"""Prepare existing natural runs for a development pilot, without model calls.

prepare inventories completed runs and exports every eligible ten-step worker-visible
prefix. restore executes one harmless scripted command after selected saved checkpoints
in isolated Docker branches. Neither command calls a provider or changes source stores.
The manifest is development material, not task admission or confirmed stall evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from judge_v2 import record as R
from trajectory_retention.experiments.branch_v2 import ReadOnlyArtifactStore
from trajectory_retention.experiments.submission_stop import SETTING, progress
from study_common import write_json


def collect(manifest_path: Path, judged_path: Path, out: Path) -> dict:
    from study_common import stall_id_for
    manifest = json.loads(manifest_path.read_text())
    judged = json.loads(judged_path.read_text())
    binding = judged["summary"].get("pilot") or {}
    if binding.get("sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("judge results are not bound to this pilot manifest")
    sources = {r["run_id"]: r for r in manifest["runs"]}
    made, skipped = [], []
    for run in judged["runs"]:
        if run["status"] != "complete":
            skipped.append({"run_id": run["run_id"], "reason": run["status"]})
            continue
        for episode in run["episodes"]:
            if not episode["branchable"] or episode["same_source_as_episodes"]:
                skipped.append({"episode": episode["episode_id"], "reason": "unbranchable or duplicate source"})
                continue
            source = sources[run["run_id"]]
            if episode["step_index"] not in [p["point"] for p in source["points"]]:
                raise ValueError("episode source outside eligible pilot prefixes")
            sid = stall_id_for(run["task_id"], run["run_id"], episode["step_index"])
            stall = {"version": "stall.v2", "stall_id": sid, "store": manifest["store"],
                     "run_id": run["run_id"], "task_id": run["task_id"],
                     "checkpoint_id": episode["checkpoint_id"], "step_index": episode["step_index"],
                     "stall_onset_point": episode["stall_onset_point"],
                     "stall_onset_checkpoint_id": episode["stall_onset_checkpoint_id"],
                     "later_solved": source["later_solved"], "episode": episode,
                     "development_only": True, "context_tokens_at_stall": None,
                     "judge_results": str(judged_path.resolve()), "judge": judged["summary"]["judge"]}
            boundary = {"status": "ok", "version": "judge-onset-boundary.v1",
                        "rewind_point": episode["stall_onset_point"],
                        "source": "judge first-confirmed episode onset; restore checkpoint before onset",
                        "episode_id": episode["episode_id"]}
            for name, value in (("stall.json", stall), ("boundary.json", boundary)):
                path = out / sid / name
                if path.exists() and json.loads(path.read_text()) != value:
                    raise ValueError(f"refusing to change frozen pilot input: {path}")
                write_json(path, value)
            made.append(sid)
    return {"made": made, "skipped": skipped}


def label(directory: Path, kind: str, budget: float) -> dict:
    from trajectory_retention.ledger.units import build_units, scripted_units_labeller
    from trajectory_retention.ledger.arm_views import build_views
    from trajectory_retention.ledger.__main__ import make_client
    from trajectory_retention.store import ArtifactStore
    record_path = directory / ("stall.json" if (directory / "stall.json").exists() else "healthy.json")
    row = json.loads(record_path.read_text())
    src = ReadOnlyArtifactStore(row["store"])
    client = (scripted_units_labeller(ArtifactStore(directory / "labeller")) if kind == "scripted"
              else make_client(kind, directory / "labeller", budget_usd=budget))
    doc = build_units(src, row["run_id"], row["checkpoint_id"], client, out_dir=directory)
    if doc["status"] == "ok":
        build_views(doc, directory)
    return {"status": doc["status"], "fake_labels": doc["fake_labels"], "labeller": doc["labeller"]}


def healthy(manifest_path: Path, judged_path: Path, review_path: Path, out: Path) -> dict:
    """Record a reviewed healthy checkpoint; judge-prefix and later-success gates are mandatory."""
    manifest, judged, review = (json.loads(p.read_text()) for p in (manifest_path, judged_path, review_path))
    if (judged["summary"].get("pilot") or {}).get("sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("judge results are not bound to this pilot manifest")
    run = next(r for r in manifest["runs"] if r["run_id"] == review["run_id"])
    point = review["point"]
    cp = next(p for p in run["points"] if p["point"] == point)
    jr = next(r for r in judged["runs"] if r["run_id"] == run["run_id"])
    expected = list(range(10, point + 1, 10))
    prior = [p for p in jr["points"] if p["point"] <= point]
    if not run["later_solved"] or jr["status"] != "complete" or [p["point"] for p in prior] != expected:
        raise ValueError("healthy checkpoint needs a solved source and complete judging through its point")
    if any(p.get("confirmed") for p in prior):
        raise ValueError("healthy checkpoint must precede any confirmed stall")
    if any(len(p.get("votes", [])) != 3 or
           any(v in ("no_answer", "skipped_spend_guard") for v in p["votes"]) for p in prior):
        raise ValueError("healthy checkpoint cannot be selected with missing technical judge votes")
    if not review.get("reviewer") or not review.get("progress_reason") or not review.get("evidence"):
        raise ValueError("record reviewer, visible progress rationale and evidence quotes")
    src = ReadOnlyArtifactStore(manifest["store"])
    text = R.RunView(src, run["run_id"]).record(point).render()
    if any(not quote or quote not in text for quote in review["evidence"]):
        raise ValueError("healthy progress evidence must occur in its worker-visible prefix")
    hid = f"healthy-{run['key']}-p{point}"
    row = {"version": "healthy.v2", "healthy_id": hid, "store": manifest["store"],
           "run_id": run["run_id"], "task_id": run["task_id"], "checkpoint_id": cp["checkpoint_id"],
           "step_index": point, "development_only": True,
           "selection": "reviewed visible progress; before confirmed stall; source later solved",
           "review": review, "judge_results": str(judged_path.resolve()), "draft_rules_used": []}
    existing = list(out.glob("*/healthy.json"))
    if sum(json.loads(p.read_text())["task_id"] == run["task_id"] and p.parent.name != hid for p in existing) >= 2:
        raise ValueError("at most two healthy checkpoints per task")
    target = out / hid / "healthy.json"
    if target.exists() and json.loads(target.read_text()) != row:
        raise ValueError("refusing to overwrite different healthy selection")
    write_json(target, row)
    return {"healthy_id": hid, "directory": str(target.parent)}


def prepare(source: Path, out: Path, include_interrupted: bool = False) -> dict:
    store = source / "store"
    src = ReadOnlyArtifactStore(store)
    rows, skipped = [], []
    for path in sorted((source / "status").glob("*.json")):
        status = json.loads(path.read_text())
        accepted = {"completed", "accounting_interrupted"} if include_interrupted else {"completed"}
        if status.get("status") not in accepted:
            # Runs stopped without a final status (user stop, disk full) are skipped with their last state.
            skipped.append({"run_id": status.get("run_id") or path.stem,
                            "reason": status.get("status") or f"no final status (state {status.get('state')})"})
            continue
        cutoff = status["steps"]
        stop_at = None
        previous = None
        for sub in (status.get("submission_behaviour") or {}).get("submissions", []):
            n = sub["failing_count"]
            verdict = {"status": "pass" if n == 0 else "fail", "failing_count": n}
            state = progress(SETTING, previous, verdict)
            previous = state
            if n == 0:
                break
            if state["no_improvement_count"] >= SETTING["limit"]:
                stop_at = sub["step"]
                cutoff = min(cutoff, stop_at)
                break
        rid = status["run_id"]
        key = hashlib.sha256(rid.encode()).hexdigest()[:12]
        run = R.RunView(src, rid)
        points, unavailable = [], []
        for k in range(10, cutoff, 10):
            cp = run.point_checkpoint(k)
            if cp is None or not cp.branchable:
                unavailable.append({"point": k, "reason": "no branchable checkpoint"})
                continue
            rec = run.record(k)
            if rec.issues:
                unavailable.append({"point": k, "reason": rec.issues})
                continue
            text = rec.render()
            rel = Path("prefixes") / key / f"p{k:04d}.txt"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_text(text)
            points.append({"point": k, "checkpoint_id": cp.checkpoint_id,
                           "record": str(rel), "sha256": hashlib.sha256(text.encode()).hexdigest()})
        rows.append({"run_id": rid, "task_id": status["task_id"], "key": key,
                     "source_status_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "original_steps": status["steps"], "counterfactual_stop_step": stop_at,
                     "branch_before_step": cutoff, "later_solved": status.get("solved_by_submission") if status["status"] == "completed" else None,
                     "source_status": status["status"],
                     "points": points, "unavailable": unavailable})
    manifest = {"version": "existing-run-pilot.v1", "development_only": True,
                "source": str(source.resolve()), "store": str(store.resolve()),
                "source_protocol": "historical; unchanged", "branch_submission_stop": SETTING,
                "judge": {"settings": str(HERE / "judge_v2/judge-glm.json"),
                          "status": "provisional pilot choice, user 2026-10-26; not human validated"},
                "selection": ("all completed runs" + (" and runs ended by an accounting interruption (later outcome "
                              "unknown, later_solved null; user 2026-11-02)" if include_interrupted else "") +
                              "; every ten steps strictly before actual or counterfactual termination"),
                "runs": rows, "skipped": skipped, "paid_calls": 0}
    write_json(out / "manifest.json", manifest)
    return {"runs": len(rows), "points": sum(len(r["points"]) for r in rows),
            "skipped": len(skipped), "manifest": str(out / "manifest.json")}


def restore(manifest_path: Path, out: Path, selectors: list[str]) -> dict:
    from arms_v2 import arm_spec_v3
    from trajectory_retention.experiments.branch_v2 import run_branch, BranchStore
    from trajectory_retention.execution.model import ScriptedModel
    manifest = json.loads(manifest_path.read_text())
    results = []
    for selector in selectors:
        key, point = selector.rsplit(":", 1)
        matches = [(r, p) for r in manifest["runs"] for p in r["points"]
                   if key in r["run_id"] and p["point"] == int(point)]
        if len(matches) != 1:
            raise ValueError(f"{selector}: expected exactly one eligible checkpoint, got {len(matches)}")
        run, pt = matches[0]
        unit = f"restore-{run['key']}-p{point}"
        dest = out / unit
        if dest.exists():
            raise ValueError(f"refusing to overwrite restoration evidence: {dest}")
        stall = {"stall_id": unit, "store": manifest["store"], "run_id": run["run_id"],
                 "checkpoint_id": pt["checkpoint_id"], "step_index": pt["point"]}
        spec = arm_spec_v3(stall, "1_continue", 0, purpose="tuning", prefix="restore-check",
                           caps=None, onset_point=None)
        spec["budget"]["step_limit"] = 1
        spec["submission_stop"] = manifest["branch_submission_stop"]
        command = "printf 'PILOT_RESTORE_OK\\n'; pwd"
        def factory(config):
            return ScriptedModel([{"content": "Check restored session.", "commands": [command],
                                   "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.0}}],
                                 model_name="scripted/pilot-restore", model_settings=dict(config.model_settings))
        result = run_branch(spec, out_dir=dest, model_factory=factory)
        branch = BranchStore(Path(result["store"]), ReadOnlyArtifactStore(manifest["store"]))
        events = list(branch.iter_events(result["run_id"])) if result.get("run_id") else []
        observed = any(e.kind == "observation_delivery" and e.delivered_ref and
                       "PILOT_RESTORE_OK" in branch.get_bytes(e.delivered_ref).decode() for e in events)
        passed = result["end_reason"] == "step_guard" and observed
        results.append({"selector": selector, "checkpoint_id": pt["checkpoint_id"],
                        "passed": passed, "result": str(dest / "result.json"),
                        "error": result.get("error"), "end_reason": result["end_reason"]})
        print(json.dumps(results[-1]), flush=True)
    report = {"version": "existing-run-restore-check.v1", "paid_calls": 0,
              "scientific_outcomes": False, "all_pass": all(r["passed"] for r in results), "checks": results}
    write_json(out / "restore-report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="cmd", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--include-interrupted", action="store_true",
                      help="also take runs ended by an accounting interruption, up to the interruption")
    rest = subs.add_parser("restore")
    rest.add_argument("--manifest", type=Path, required=True)
    rest.add_argument("--out", type=Path, required=True)
    rest.add_argument("--checkpoint", action="append", required=True, help="unique run substring:point")
    coll = subs.add_parser("collect")
    coll.add_argument("--manifest", type=Path, required=True)
    coll.add_argument("--judged", type=Path, required=True)
    coll.add_argument("--out", type=Path, required=True)
    lab = subs.add_parser("label")
    lab.add_argument("--directory", type=Path, required=True)
    lab.add_argument("--labeller", choices=("scripted", "deepseek", "replay", "terra-pilot"), required=True)
    lab.add_argument("--budget-usd", type=float, default=0)
    hp = subs.add_parser("healthy")
    hp.add_argument("--manifest", type=Path, required=True)
    hp.add_argument("--judged", type=Path, required=True)
    hp.add_argument("--review", type=Path, required=True)
    hp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "prepare":
        result = prepare(a.source, a.out, a.include_interrupted)
    elif a.cmd == "restore":
        result = restore(a.manifest, a.out, a.checkpoint)
    elif a.cmd == "collect":
        result = collect(a.manifest, a.judged, a.out)
    elif a.cmd == "label":
        result = label(a.directory, a.labeller, a.budget_usd)
    else:
        result = healthy(a.manifest, a.judged, a.review, a.out)
    print(json.dumps(result, indent=1))
    if result.get("all_pass") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
