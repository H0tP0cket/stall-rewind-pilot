"""Validation replay of stall judge v2 on the 71 validation runs (three validation-v7-20260924-v1 stores), scored
against the Claude hindsight labels and, once the user has labelled them, against human labels.

  estimate  FREE. Every judge point (steps 10, 20, ... up to the last step; 211 points), its record size, and the
            spend estimate per judge settings file: input tokens from the record size, output from the 2026-09-26 GLM
            judge receipts, and the guard's hard worst case (every call returning a full response).
  judge     PAID unless --scripted. Judges every point of every run with one settings file (GLM-5.3 Flash or the
            second, non-Claude model), 3 votes, the v2 DRAFT prompts, no type call. Writes <out>/<model>/
            stall_records.json (the postrun format) and the receipts. Refuses a Claude judge: the hindsight labels
            were written by Claude Opus, so a Claude judge must not be validated on them.
            --hand-label judges the 20 hand-label moments instead (the checkpoint steps of the packet's items).
  score     FREE. Agreement of each judged model with the hindsight labels: point confusion and Cohen's kappa (a point
            counts as genuine when it lies inside a hindsight GENUINE_STALL stretch), recall of the hindsight genuine
            episodes (any confirmed point inside), precision of confirmed points, the same at or after the hindsight
            onset (strict), episodes formed, and onset error (judge P minus the hindsight onset step) for confirmed
            points inside genuine stretches and for the first confirmed point of each caught episode; by batch and
            by worker model. With --human <labels.csv>, also agreement with the human labels on the hand-label items.
  packet    FREE. A v2 hand-label packet: the same 20 moments as the existing packet, but each item is exactly the
            record the v2 judge sees (no private grades), with an onset column in labels.csv.

Steps and checkpoints: hindsight labels name checkpoints (C<n>); a checkpoint's step is its recorded step_index or
the model-call count of its saved context (mining.points.checkpoint_step_index). A stretch C_a..C_b covers judge points
k with step(C_a) <= k <= step(C_b) (TERMINAL = the last step). A virtual checkpoint (C<n>#..., C<n>+...) takes its base
checkpoint's step.

Usage:
  python configs/v2/judge_v2/replay_v2.py estimate --settings configs/v2/judge_v2/judge-glm.json configs/v2/judge_v2/judge-luna.json
  python configs/v2/judge_v2/replay_v2.py judge --settings configs/v2/judge_v2/judge-glm.json --out DIR --cap-usd C \
      --allow-draft-prompts [--parallel-runs 4] [--scripted spec.json] [--hand-label]
  python configs/v2/judge_v2/replay_v2.py score --out DIR [--human labels.csv]
  python configs/v2/judge_v2/replay_v2.py packet --dest DIR
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OLD = HERE.parents[1] / "detector-judge-20260926"
import client  # noqa: E402
import judging  # noqa: E402
import postrun_v2  # noqa: E402
import record as R  # noqa: E402

MAIN = Path.home() / "projects/llm-trajectories"
CAL = MAIN / "artifacts/calibration"
BATCHES = {  # the stores of the 2026-09-26 validation (configs/detector-judge-20260926/storeio.py)
    "run1r-mettlebench": CAL / "validation-v7-20260924-v1-run1r-mettlebench/store",
    "run1r-swesmith": CAL / "validation-v7-20260924-v1-run1r-swe-swesmith/store",
    "climb1-mettlebench": CAL / "validation-v7-20260924-v1-climb1-mettlebench/store",
}
HAND = CAL / "detector-judge-20260926-v1/hand-label"
OLD_GLM_CALLS = CAL / "detector-judge-20260926-v1/judge/calls"
GLM_CHARS_PER_TOKEN = 3.528  # measured: 132.8M request chars / 37.6M prompt tokens over the 515 GLM receipts


def rid6(run_id: str) -> str:
    return run_id.split(":")[1][:6]


def load_runs() -> list[tuple[str, R.RunView]]:
    out = []
    for batch, root in BATCHES.items():
        a = R.open_readonly(root)
        out += [(batch, R.RunView(a, rid)) for rid in R.natural_runs(a)]
    return out


def cp_step(run: R.RunView, label) -> float | None:
    """Step of a hindsight checkpoint label (C<n>, virtual C<n>#.., TERMINAL)."""
    from trajectory_retention.mining.points import checkpoint_step_index
    if label is None:
        return None
    s = str(label)
    if "TERMINAL" in s.upper():
        return float(run.n_steps)
    n = s.split("#")[0].split("+")[0].strip()
    c = next((c for c in run.checkpoints if c.checkpoint_id.endswith("/" + n)), None)
    return None if c is None else float(checkpoint_step_index(run.store, c))


def hindsight(runs) -> dict[str, dict]:
    """{rid6: {"stretches": [...with step bounds], "episodes": [...]}} from the Claude hindsight reviews."""
    sys.path.insert(0, str(OLD))
    from reference import load  # noqa: E402 - the earlier study's loader, unchanged
    ref = load()
    by = {rid6(r.run_id): r for _, r in runs}
    for k, ent in ref.items():
        run = by.get(k)
        for s in ent["stretches"]:
            s["lo_step"] = cp_step(run, s.get("from_cp")) if run else None
            s["hi_step"] = cp_step(run, s.get("to_cp")) if run else None
            if s["hi_step"] is None and run is not None:
                s["hi_step"] = float(run.n_steps)
            s["onset_step"] = cp_step(run, s.get("onset_cp") or s.get("from_cp")) if run else None
    return ref


def inside(ref_run: dict | None, k: int, genuine_only=True) -> list[dict]:
    if not ref_run:
        return []
    return [s for s in ref_run["stretches"] if s.get("lo_step") is not None and s["lo_step"] <= k <= s["hi_step"]
            and (not genuine_only or s["category"] == "GENUINE_STALL")]


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labels)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def hand_items(runs) -> list[dict]:
    """The 20 hand-label moments: item, run, checkpoint label, step."""
    key = json.loads((HAND / "KEY-judge-answers.json").read_text())
    by = {r.run_id.split(":")[1][:10]: (b, r) for b, r in runs}
    out = []
    for k in key:
        prefix, cp = k["point_id"].split(":")
        batch, run = by[prefix]
        out.append({"item": k["item"], "point_id": k["point_id"], "batch": batch, "run_id": run.run_id,
                    "checkpoint": cp, "step": int(cp_step(run, cp)), "old_judge_confirmed": k["judge_confirmed"],
                    "hindsight_genuine": k["reference_genuine"]})
    return out


# ---------- estimate ----------

def cmd_estimate(a):
    runs = load_runs()
    prompts = {k: judging.load_prompt(k) for k in ("stall", "type")}
    sys_chars = len(prompts["stall"][0])
    pts = []
    for batch, run in runs:
        for k in run.points():
            rec = run.record(k)
            pts.append({"batch": batch, "run_id": run.run_id, "point": k, "source": rec.source["kind"],
                        "issues": rec.issues,
                        "chars": sys_chars + len(judging.user_message(rec.render(), prompts["stall"][1], k))})
    old = []
    for f in OLD_GLM_CALLS.rglob("*.json"):
        for at in json.loads(f.read_text()).get("attempts", []):
            u = (at.get("response") or {}).get("usage") or {}
            if u:
                old.append((u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0, u.get("cost") or 0.0))
    old_cost, old_prompt = sum(c for *_, c in old), sum(p for p, *_ in old)
    mean_out = statistics.mean(c for _, c, _ in old)
    chars = sum(p["chars"] for p in pts)
    rep = {"points": len(pts), "runs": len(runs), "issues": [p for p in pts if p["issues"]],
           "sources": dict(collections.Counter(p["source"] for p in pts)),
           "record_chars": {"total": chars, "max": max(p["chars"] for p in pts),
                            "median": statistics.median(p["chars"] for p in pts)},
           "earlier_glm_judge": {"calls": len(old), "spend_usd": round(old_cost, 4),
                                 "prompt_tokens": old_prompt, "mean_output_tokens": round(mean_out, 1),
                                 "usd_per_call": round(old_cost / len(old), 5),
                                 "usd_per_million_prompt_tokens_all_in": round(old_cost / old_prompt * 1e6, 4)},
           "per_settings": {}}
    for sp in a.settings:
        s = client.Settings.load(Path(sp))
        calls = len(pts) * judging.VOTES
        tok_in = chars / GLM_CHARS_PER_TOKEN * judging.VOTES
        list_price = tok_in * s.price_in + calls * mean_out * s.price_out
        worst = sum(s.worst_case_usd(p["chars"]) for p in pts) * judging.VOTES
        row = {"model": s.model, "calls": calls, "input_tokens_est": int(tok_in),
               "expected_usd_at_list_price": round(list_price, 2),
               "hard_worst_case_usd_guard": round(worst, 2)}
        if "glm" in s.model:
            row["expected_usd_scaled_from_earlier_glm_spend"] = round(tok_in * old_cost / old_prompt, 2)
        rep["per_settings"][s.model] = row
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "estimate.json").write_text(json.dumps(rep | {"point_rows": pts}, indent=1))
    print(json.dumps(rep, indent=1))


# ---------- judge ----------

def cmd_judge(a):
    s = client.Settings.load(Path(a.settings))
    if s.model.startswith("anthropic/") or "claude" in s.model.lower():
        raise SystemExit("a Claude judge must not be validated on the Claude-written hindsight labels")
    if not a.scripted and not a.allow_draft_prompts:
        raise SystemExit("the v2 judge prompts are DRAFT: paid calls need --allow-draft-prompts")
    out = Path(a.out).resolve() / (s.slug + ("-hand-label" if a.hand_label else ""))
    out.mkdir(parents=True, exist_ok=True)
    transport = client.make_transport(s, Path(a.scripted) if a.scripted else None, out / "scripted-requests.jsonl")
    budget = client.Budget(a.cap_usd, client.receipts_cost(out / "calls"))
    prompts = {k: judging.load_prompt(k) for k in ("stall", "type")}
    runs = load_runs()
    plan = []  # (batch, run, points)
    if a.hand_label:
        items = hand_items(runs)
        for batch, run in runs:
            ks = sorted({i["step"] for i in items if i["run_id"] == run.run_id})
            if ks:
                plan.append((batch, run, ks))
    else:
        plan = [(b, r, None) for b, r in runs if not a.run or any(x in r.run_id for x in a.run)]
    vote_pool = ThreadPoolExecutor(judging.VOTES * a.parallel_runs)

    def one(item):
        batch, run, ks = item
        rec = postrun_v2.judge_run(run, s, transport, budget, out, prompts, vote_pool, points=ks,
                                   retry_no_answer=a.retry_no_answer, with_type=False)
        rec["batch"] = batch
        print(f"{batch:20} {rid6(run.run_id)} points={len(rec['points']):2} "
              f"confirmed={sum(bool(p.get('confirmed')) for p in rec['points']):2} status={rec['status']} "
              f"${rec['cost_usd']:.4f}", flush=True)
        return rec
    with ThreadPoolExecutor(a.parallel_runs) as rp:
        report = list(rp.map(one, plan))
    summary = {"judge": postrun_v2.judge_meta(s, prompts, transport), "hand_label": a.hand_label,
               "runs": len(report), "points": sum(len(r["points"]) for r in report),
               "incomplete_runs": [r["run_id"] for r in report if r["status"] != "complete"],
               "spend_usd_receipts_total": round(client.receipts_cost(out / "calls"), 4)}
    if a.hand_label:
        summary["items"] = hand_items(runs)
    (out / "stall_records.json").write_text(json.dumps({"summary": summary, "runs": report}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("judge", "items")}, indent=1))


# ---------- score ----------

def score_model(d: Path, runs, ref) -> dict:
    data = json.loads((d / "stall_records.json").read_text())
    model = data["summary"]["judge"]["model"]
    if model.startswith("anthropic/") or "claude" in model.lower():
        return {"model": model, "refused": "Claude judge: not scored against Claude hindsight labels"}
    by_run = {r.run_id: (b, r) for b, r in runs}
    rows = []
    for rr in data["runs"]:
        batch, run = by_run[rr["run_id"]]
        k6 = rid6(rr["run_id"])
        for p in rr["points"]:
            if p.get("error") or p.get("skipped_spend_guard"):
                continue
            gen = inside(ref.get(k6), p["point"])
            onset_h = gen[0]["onset_step"] if gen else None
            rows.append({"run": k6, "batch": batch, "worker": (run.model or "?").split("/")[-1], "point": p["point"],
                         "confirmed": p["confirmed"], "genuine": bool(gen), "reviewed": k6 in ref,
                         "categories": sorted({s["category"] for s in inside(ref.get(k6), p["point"], False)}),
                         "onset": p.get("onset"), "onset_source": p.get("onset_source"),
                         "hindsight_onset_step": onset_h, "votes": p["votes"],
                         "episode": f"{k6}:{gen[0]['from_cp']}-{gen[0]['to_cp']}" if gen else None})
    rows = [r for r in rows if r["reviewed"]]

    def block(rs, eps_keys):
        pairs = [("stall" if r["confirmed"] else "no", "stall" if r["genuine"] else "no") for r in rs]
        conf = collections.Counter(f"judge={a} hindsight={b}" for a, b in pairs)
        confirmed = [r for r in rs if r["confirmed"]]
        strict = [r for r in confirmed if r["genuine"] and r["point"] >= (r["hindsight_onset_step"] or 0)]
        caught = sorted({r["episode"] for r in confirmed if r["genuine"]})
        strict_caught = sorted({r["episode"] for r in strict})
        errs = [r["onset"] - r["hindsight_onset_step"] for r in confirmed
                if r["genuine"] and r["onset"] is not None and r["hindsight_onset_step"] is not None]
        first_errs = []
        for e in caught:
            f = min((r for r in confirmed if r["episode"] == e), key=lambda r: r["point"])
            if f["onset"] is not None and f["hindsight_onset_step"] is not None:
                first_errs.append(f["onset"] - f["hindsight_onset_step"])

        def dist(xs):
            return None if not xs else {"n": len(xs), "mean_abs": round(statistics.mean(abs(x) for x in xs), 2),
                                        "median_abs": statistics.median(abs(x) for x in xs),
                                        "mean_signed": round(statistics.mean(xs), 2),
                                        "within_5_steps": sum(abs(x) <= 5 for x in xs),
                                        "within_10_steps": sum(abs(x) <= 10 for x in xs)}
        return {"points": len(rs), "confusion": dict(conf), "cohen_kappa": kappa(pairs),
                "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
                "confirmed_points": len(confirmed),
                "precision_inside_genuine": round(sum(r["genuine"] for r in confirmed) / len(confirmed), 3)
                if confirmed else None,
                "precision_strict_at_or_after_onset": round(len(strict) / len(confirmed), 3) if confirmed else None,
                "hindsight_episodes_with_a_point": len(eps_keys),
                "episodes_caught": len(caught), "recall": round(len(caught) / len(eps_keys), 3) if eps_keys else None,
                "episodes_caught_strict": len(strict_caught),
                "missed_episodes": sorted(set(eps_keys) - set(caught)),
                "onset_error_steps_all_confirmed_genuine_points": dist(errs),
                "onset_error_steps_first_point_per_caught_episode": dist(first_errs),
                "onset_from_fallback": sum(r["onset_source"] == "fallback" for r in confirmed),
                "false_confirmed_categories": dict(collections.Counter("/".join(r["categories"]) or "unlabelled"
                                                                       for r in confirmed if not r["genuine"]))}

    def eps_of(rs):  # hindsight genuine episodes that hold at least one judge point
        return sorted({r["episode"] for r in rs if r["genuine"]})
    all_eps = sorted({f"{k}:{s['from_cp']}-{s['to_cp']}" for k, v in ref.items() for s in v["episodes"]})
    res = {"model": model, "overall": block(rows, eps_of(rows)),
           "hindsight_genuine_episodes_total": len(all_eps),
           "hindsight_episodes_without_any_judge_point": sorted(set(all_eps) - set(eps_of(rows))),
           "by_batch": {b: block([r for r in rows if r["batch"] == b], eps_of([r for r in rows if r["batch"] == b]))
                        for b in BATCHES},
           "by_worker": {w: block([r for r in rows if r["worker"] == w], eps_of([r for r in rows if r["worker"] == w]))
                         for w in sorted({r["worker"] for r in rows})},
           "votes": dict(collections.Counter(v for r in rows for v in r["votes"])),
           "episodes_formed": sum(len(r["episodes"]) for r in data["runs"]),
           "spend_usd": data["summary"].get("spend_usd_receipts_total"),
           "point_rows": rows}
    return res


def score_human(d: Path, labels: Path, runs) -> dict:
    data = json.loads((d / "stall_records.json").read_text())
    items = data["summary"].get("items") or hand_items(runs)
    lab = {}
    with open(labels) as f:
        rows = list(csv.reader(f))
    head = [h.lower() for h in rows[0]]
    i_onset = next((i for i, h in enumerate(head) if h.startswith("onset")), None)
    for r in rows[1:]:
        if len(r) > 1 and r[1].strip():
            lab[r[0].strip().zfill(2)] = {"verdict": r[1].strip().lower(),
                                          "onset": (int(r[i_onset]) if i_onset is not None and len(r) > i_onset
                                                    and r[i_onset].strip().isdigit() else None)}
    pts = {(rr["run_id"], p["point"]): p for rr in data["runs"] for p in rr["points"]}
    out, pairs, ref_pairs, old_pairs, errs = [], [], [], [], []
    for it in items:
        h = lab.get(it["item"])
        p = pts.get((it["run_id"], it["step"]))
        if h is None or p is None or p.get("error"):
            continue
        hs = "stall" if h["verdict"] == "stall" else "no"
        pairs.append((hs, "stall" if p["confirmed"] else "no"))
        ref_pairs.append((hs, "stall" if it["hindsight_genuine"] else "no"))
        old_pairs.append((hs, "stall" if it["old_judge_confirmed"] else "no"))
        if hs == "stall" and p["confirmed"] and h["onset"] is not None and p.get("onset") is not None:
            errs.append(p["onset"] - h["onset"])
        out.append(it | {"human": h, "judge_votes": p["votes"], "judge_confirmed": p["confirmed"],
                         "judge_onset": p.get("onset")})
    return {"labelled_items": len(pairs), "kappa_human_vs_judge_v2": kappa(pairs),
            "agreement_human_vs_judge_v2": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
            "kappa_human_vs_hindsight": kappa(ref_pairs), "kappa_human_vs_v1_judge": kappa(old_pairs),
            "onset_error_vs_human": errs, "items": out,
            "caveat": "the existing packet (hand-label/packet) shows private grades, which the v2 judge never sees; "
                      "the v2 packet (replay_v2.py packet) shows exactly the v2 judge's record"}


def cmd_score(a):
    runs = load_runs()
    ref = hindsight(runs)
    base = Path(a.out)
    res = {}
    for d in sorted(p for p in base.iterdir() if (p / "stall_records.json").exists()):
        if d.name.endswith("-hand-label"):
            if a.human:
                res[d.name] = score_human(d, Path(a.human), runs)
            continue
        res[d.name] = score_model(d, runs, ref)
    (base / "scores.json").write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        if "overall" in r:
            o = r["overall"]
            print(f"{name}: points={o['points']} kappa={o['cohen_kappa']} agreement={o['agreement']} "
                  f"recall={o['episodes_caught']}/{o['hindsight_episodes_with_a_point']} "
                  f"precision={o['precision_inside_genuine']} strict={o['precision_strict_at_or_after_onset']} "
                  f"onset={o['onset_error_steps_first_point_per_caught_episode']}")
        else:
            print(name, {k: v for k, v in r.items() if k not in ("items", "point_rows")})


# ---------- packet ----------

def cmd_packet(a):
    runs = load_runs()
    by = {r.run_id: r for _, r in runs}
    items = hand_items(runs)
    dest = Path(a.dest)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty; a packet is never overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    closing = judging.load_prompt("stall")[1]
    for it in items:
        rec = by[it["run_id"]].record(it["step"])
        (dest / f"item-{it['item']}.txt").write_text(judging.user_message(rec.render(), closing, it["step"]))
    with open(dest / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "verdict (stall / not_stall / unclear)", "onset step if stall",
                    "type if stall (stuck / over-commit / under-commit / unclear)", "note"])
        for it in items:
            w.writerow([it["item"], "", "", "", ""])
    (dest / "README.md").write_text(
        "# Hand-label packet v2 (blind)\n\nThe same 20 moments as `../packet`, in the same item order, but each "
        "`item-NN.txt` is exactly the record the v2 stall judge sees: the task, every step (reasoning, commands), "
        "outputs as the agent received them, and the submission feedback the agent saw. No private grades, no "
        "reference fix. The record ends after the step named at its end.\n\nFor each item fill `labels.csv`: verdict "
        "(`stall`, `not_stall`, `unclear`), the step where the stall began if it is a stall, and optionally the type."
        "\n\nScore: `python configs/v2/judge_v2/replay_v2.py score --out <replay dir> --human <this labels.csv>` after "
        "`replay_v2.py judge --hand-label`.\n")
    (dest.parent / f"KEY-{dest.name}.json").write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} items to {dest}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--settings", nargs="+", default=[str(HERE / "judge-glm.json"), str(HERE / "judge-luna.json")])
    e.add_argument("--out")
    j = sub.add_parser("judge")
    j.add_argument("--settings", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--cap-usd", type=float, required=True)
    j.add_argument("--parallel-runs", type=int, default=4)
    j.add_argument("--run", action="append", default=[])
    j.add_argument("--scripted")
    j.add_argument("--allow-draft-prompts", action="store_true")
    j.add_argument("--retry-no-answer", action="store_true")
    j.add_argument("--hand-label", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--human")
    p = sub.add_parser("packet")
    p.add_argument("--dest", default=str(HAND / "packet-v2"))
    a = ap.parse_args()
    {"estimate": cmd_estimate, "judge": cmd_judge, "score": cmd_score, "packet": cmd_packet}[a.cmd](a)


if __name__ == "__main__":
    main()
