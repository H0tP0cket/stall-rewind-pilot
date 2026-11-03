"""End-to-end check of the Ledger on real stalled DeepSeek V4.1 Flash runs, with FAKE labels and no paid call.

  PYTHONPATH=src .venv/bin/python -m trajectory_retention.ledger.e2e [--out artifacts/v2/ledger] [--source-root DIR]

For three stalls from the local MettleBench stores (read-only) it runs both passes with the scripted labeller and
writes, per stall under OUT/stalls/<name>/: boundary.json, ledger-boundary.json, ledger.json, review_view_ledger.json,
review_view_context.json, recipe.json (small cap, so trimming happens), labeller_calls.json, the labeller receipts
(OUT/stalls/<name>/labeller) and checks.json. Then it exports the gate decks for the three stalls (OUT/gate), fills
answers from the key ("self", plus a deterministic perturbed copy as a second annotator), scores them, and writes
OUT/report.json with every check, the replay guard, and the cost estimate for the same passes with the paid labeller.
One stall uses a small call budget so that chunking and later-output elision run.

Checks (see ``checks``): every quote resolves (re-read from the store, raw context included); no content after the
cut and no private evaluator data in any view, recipe or kept label (with planted-leak controls that must be caught);
arm 2 and arm 3 views show the same steps; ``units`` covers every ID in each view's text; the recipe respects its cap
and logs its cuts; kept labels carry no IDs; the frozen boundary is never overwritten; a rebuild from receipts is
byte-identical.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..observers.t2.store_ro import ReadOnlyStore
from ..store import ArtifactStore
from . import checks as C
from . import templates as T
from .__main__ import build_stall
from .gate import export_decks, score, self_answers
from .labeller import SETTINGS_PATH, ScriptedLabeller, study_labeller
from .passes import FrozenBoundaryExists, build_boundaries
from .render import label_text

REPO = Path(__file__).resolve().parents[3]
COMMAND = "PYTHONPATH=src .venv/bin/python -m trajectory_retention.ledger.e2e"
RECIPE_CAP = 60
STALLS = [  # (name, store under the source root, run id, checkpoint, call budget in characters)
    ("climb1-e2d1ab49-C12", "calibration/validation-v7-20260924-v1-climb1-mettlebench/store",
     "validation-v7-20260924-v1-climb1:e2d1ab49a3386c1d5daa63bc:natural", "C12", 300_000),
    ("climb1-fa63deec-C13", "calibration/validation-v7-20260924-v1-climb1-mettlebench/store",
     "validation-v7-20260924-v1-climb1:fa63deecbcb7c2fae09914a9:natural", "C13", 300_000),
    ("run1r-31b994ee-C5", "calibration/validation-v7-20260924-v1-run1r-mettlebench/store",
     "validation-v7-20260924-v1-run1r:31b994ee8b6882daeeef6d70:natural", "C5", 90_000),
]
# StreamLake fp8 prices for deepseek/deepseek-v4.1-flash (the study pin in configs/v2/settings.json), fitted exactly
# from the provider smoke receipt (artifacts/v2/provider-smoke/streamlake-fp8-20261007T184034Z.json): uncached input,
# output (reasoning included). Cache hits (about $0.003/M) are ignored, so the estimate is an upper bound on input.
USD_PER_MTOK = {"input": 0.141, "output": 0.564}
ASSUMED_OUTPUT_TOKENS_PER_CALL = (4_000, 16_000)  # max reasoning effort; not measured for these prompts


def _source_root() -> Path:
    own = REPO / "artifacts"
    if (own / "calibration").is_dir():
        return own
    common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-common-dir"], capture_output=True,
                            text=True, check=True).stdout.strip()
    return (REPO / common).resolve().parent / "artifacts"


def _dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n")


def run_stall(src: ReadOnlyStore, name: str, run_id: str, cp: str, max_chars: int, out: Path) -> dict[str, Any]:
    sdir = out / "stalls" / name
    checkpoint_id = f"{run_id}/{cp}"
    t0 = time.monotonic()
    res = build_stall(src, run_id, checkpoint_id, sdir, ScriptedLabeller(ArtifactStore(sdir / "labeller")),
                      recipe_cap=RECIPE_CAP, max_chars=max_chars)
    seconds = time.monotonic() - t0
    ledger, v2, v3, recipe = res["ledger"], res["ledger_view"], res["context_view"], res["recipe"]
    units = sorted(set(v2["units"]) | set(v3["units"]), key=lambda u: (u[0], int(u[1:])))
    labels = {u: label_text(ledger, u) for u in units}
    texts = {"review_view_ledger": v2["text"], "review_view_context": v3["text"], "recipe": recipe["text"],
             "kept_labels": "\n".join(labels.values())}
    sources = C.leak_sources(src, ledger)
    planted_future = next(iter(sorted(sources["future"])), None)
    planted_private = next(iter(sorted(sources["private"], key=len, reverse=True)), None)
    out_checks: dict[str, Any] = {
        "quotes_resolve": C.check_quotes(src, ledger),
        "no_leaks": C.check_leaks(texts, sources),
        "leak_control_future_caught": None if planted_future is None else not C.check_leaks(
            {"planted": v2["text"] + "\n" + planted_future}, sources)["pass"],
        "leak_control_private_caught": None if planted_private is None else not C.check_leaks(
            {"planted": v3["text"] + "\n" + planted_private}, sources)["pass"],
        "views_same_steps_and_units_cover_text": C.check_views(ledger, v2, v3),
        "recipe": C.check_recipe(ledger, recipe),
        "kept_labels": C.check_labels(ledger, units),
        "no_stuck_wording": C.no_stuck_wording(texts),
        "context_bound_to_events": {"pass": ledger["checks"]["steps_match_responses"]
                                    and ledger["checks"]["steps_match_counters"] and not ledger["checks"]["issues"],
                                    **{k: v for k, v in ledger["checks"].items() if k != "issues"},
                                    "issues": ledger["checks"]["issues"][:10]},
    }
    out_checks["leak_controls"] = {"pass": bool(out_checks.pop("leak_control_future_caught"))
                                   and bool(out_checks.pop("leak_control_private_caught")),
                                   "planted_future_example": (planted_future or "")[:80],
                                   "planted_private_example": (planted_private or "")[:80]}
    _dump(sdir / "checks.json", out_checks)
    calls = ledger["labeller"]["boundary_passes"] + ledger["labeller"]["passes"]
    return {"name": name, "run_id": run_id, "checkpoint_id": checkpoint_id, "dir": str(sdir.relative_to(out)),
            "seconds": round(seconds, 1), "steps": ledger["stall_point"], "status": ledger["status"],
            "boundary": {k: ledger["boundary"][k] for k in ("current_approach_id", "current_approach_first_step",
                                                            "rewind_point", "first_approach_stall", "status")},
            "windows": len(ledger["windows"]), "approaches": [(a["id"], a["first_step"], a["last_step"], a["status"])
                                                              for a in ledger["approaches"]],
            "direct_boundary": {k: ledger["direct_boundary"].get(k) for k in ("first_step", "agrees_with_primary",
                                                                              "difference_steps")},
            "items": {t: sum(1 for i in ledger["items"] if i["type"] == t) for t in ("decision", "fact", "claim", "idea")},
            "dropped": len(ledger["dropped"]), "dropped_reasons": sorted({d["reason"].split(" (")[0].split(" ")[0]
                                                                         for d in ledger["dropped"]}),
            "calls": [{k: c.get(k) for k in ("prompt", "chunk", "status", "request_chars", "later_outputs_elided",
                                             "outputs_elided")} for c in calls],
            "views_chars": {"arm2": len(v2["text"]), "arm3": len(v3["text"])},
            "recipe": {"tokens": recipe["tokens"], "cap": recipe["cap_tokens"], "cuts": len(recipe["cut_log"]),
                       "entries_before_cap": len(recipe["entries_before_cap"])},
            "checks": {k: v["pass"] for k, v in out_checks.items()}}


def frozen_boundary_proof(src: ReadOnlyStore, stall: dict[str, Any], out: Path) -> dict[str, Any]:
    sdir = out / "stalls" / stall["name"]
    probe = out / "proofs" / "frozen-boundary"
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True)
    tampered = json.loads((sdir / "boundary.json").read_text())
    tampered["rewind_point"] += 1
    (probe / "boundary.json").write_text(json.dumps(tampered))
    client = ScriptedLabeller(ArtifactStore(sdir / "labeller"))
    try:
        build_boundaries(src, stall["run_id"], stall["checkpoint_id"], client, out_dir=probe)
        refused = False
    except FrozenBoundaryExists:
        refused = True
    kept = json.loads((probe / "boundary.json").read_text())["rewind_point"] == tampered["rewind_point"]
    shutil.rmtree(probe)
    # rebuilding into the stall's own directory from the stored receipts must reproduce every output byte for byte
    before = {p.name: p.read_bytes() for p in sdir.glob("*.json") if p.name != "checks.json"}
    client = ScriptedLabeller(ArtifactStore(sdir / "labeller"))
    build_stall(src, stall["run_id"], stall["checkpoint_id"], sdir, client, recipe_cap=RECIPE_CAP,
                max_chars=next(s[4] for s in STALLS if s[0] == stall["name"]))
    after = {p.name: p.read_bytes() for p in sdir.glob("*.json") if p.name != "checks.json"}
    raw_diff = sorted(n for n in before if before[n] != after.get(n))
    diff = sorted(n for n in before if _without_cache_flag(before[n]) != _without_cache_flag(after.get(n, b"null")))
    return {"pass": refused and kept and not diff and client.calls == 0,
            "different_boundary_refused": refused, "frozen_file_untouched": kept,
            "rebuild_files_differing": diff, "rebuild_files_differing_only_in_cached_flag": raw_diff,
            "rebuild_new_calls": client.calls, "rebuild_cache_hits": client.cache_hits}


def _without_cache_flag(data: bytes) -> Any:
    """A call record's ``cached`` flag says whether the receipt was reused; everything else must be identical."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "cached"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(json.loads(data))


def replay_guard(out: Path, src: ReadOnlyStore, stall: dict[str, Any]) -> dict[str, Any]:
    """The study client (pin and settings read from configs/v2/settings.json) in replay mode with an empty receipt
    store must refuse instead of calling out, and a boundary pass that could not label must leave no frozen
    boundary.json behind."""
    probe = out / "proofs" / "replay"
    shutil.rmtree(probe, ignore_errors=True)
    client = study_labeller(ArtifactStore(probe / "labeller"), budget_usd=0.0, replay=True)
    body = client.request_body([{"role": "user", "content": "probe"}])
    try:
        client.complete([{"role": "user", "content": "probe"}], purpose="ledger:replay-guard")
        refused = False
    except RuntimeError as exc:
        refused = "unavailable" in str(exc)
    part = build_boundaries(src, stall["run_id"], stall["checkpoint_id"], client, out_dir=probe)
    errors = sorted({c.get("error", "")[:60] for c in part["labeller"]["passes"]})
    not_frozen = not (probe / "boundary.json").exists() and (probe / "boundary.failed.json").exists()
    result = {"pass": refused and not_frozen and part["boundary"]["status"] == "labelling_failed"
              and client.calls == 0 and client.spent == 0.0,
              "refused_without_receipt": refused, "failed_boundary_status": part["boundary"]["status"],
              "failed_boundary_not_frozen": not_frozen, "calls_made": client.calls, "spent_usd": client.spent,
              "call_errors": errors,
              "settings_file": str(SETTINGS_PATH), "request_shape": {k: body[k] for k in (
                  "model", "provider", "reasoning", "max_tokens", "temperature")}}
    shutil.rmtree(probe, ignore_errors=True)
    return result


def cost_estimate(stalls: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [c for s in stalls for c in s["calls"] if c["status"] == "ok" and c.get("repeat", 0) == 0]
    chars = sum(c["request_chars"] or 0 for c in calls)
    tokens_in = round(chars / 3.6)
    lo, hi = ASSUMED_OUTPUT_TOKENS_PER_CALL
    usd = lambda out_tok: round(tokens_in * USD_PER_MTOK["input"] / 1e6 + len(calls) * out_tok * USD_PER_MTOK["output"] / 1e6, 3)
    return {"calls": len(calls), "request_chars": chars, "input_tokens_est": tokens_in,
            "input_tokens_basis": "characters / 3.6 (not the provider tokenizer)",
            "prices_usd_per_mtok": USD_PER_MTOK, "price_basis": "exact fit to the StreamLake fp8 smoke receipt (the study pin)",
            "assumed_output_tokens_per_call": [lo, hi], "usd_low": usd(lo), "usd_high": usd(hi),
            "per_stall_usd": [round(usd(lo) / max(1, len(stalls)), 3), round(usd(hi) / max(1, len(stalls)), 3)],
            "command": ("PYTHONPATH=src .venv/bin/python -m trajectory_retention.ledger build --store <STORE> --run "
                        "<RUN_ID> --checkpoint <CHECKPOINT_ID> --out <DIR> --labeller deepseek "
                        "--recipe-cap <CAP> --budget-usd 1.00  # pin and settings from configs/v2/settings.json")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ledger end-to-end check (fake labels, no paid calls)")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "v2" / "ledger"))
    ap.add_argument("--source-root", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    root = Path(a.source_root).resolve() if a.source_root else _source_root()
    stalls, opened = [], {}
    for name, rel, run_id, cp, max_chars in STALLS:
        src = opened.setdefault(rel, ReadOnlyStore(root / rel))
        stalls.append(run_stall(src, name, run_id, cp, max_chars, out))
        print(f"{name}: steps={stalls[-1]['steps']} rewind={stalls[-1]['boundary']['rewind_point']} "
              f"items={stalls[-1]['items']} checks={stalls[-1]['checks']}", flush=True)
    proofs = {"frozen_boundary_and_rebuild": frozen_boundary_proof(opened[STALLS[0][1]], stalls[0], out),
              "replay_guard": replay_guard(out, opened[STALLS[0][1]], stalls[0])}
    ledgers = [json.loads((out / s["dir"] / "ledger.json").read_text()) for s in stalls]
    gate = out / "gate"
    counts = export_decks(ledgers, gate / "annotator", gate / "key.json")
    (gate / "answers").mkdir(parents=True, exist_ok=True)
    self_answers(gate / "key.json", gate / "answers" / "answers-self.jsonl")
    self_answers(gate / "key.json", gate / "answers" / "answers-self_perturbed.jsonl", perturb=0.25)
    scored = score(gate / "key.json", [gate / "answers" / "answers-self.jsonl",
                                       gate / "answers" / "answers-self_perturbed.jsonl"])
    _dump(gate / "score.json", scored)
    annotator_files = sorted(p.name for p in (gate / "annotator").iterdir())
    blind = not any(s in (gate / "annotator" / f).read_text() for f in annotator_files if f.endswith(".jsonl")
                    for s in ('"predicted"', '"item"', "⟦"))
    self_ok = all(row["kappa"] in (1.0, None) for d, rep in scored["decks"].items() if "agreement" in rep
                  for pair, row in rep["agreement"].items() if pair == "ledger~self")
    f1_self = scored["decks"]["boundaries"]["f1_within_2"]["ledger~self"]["f1"]
    gate_summary = {"cards": counts, "annotator_files": annotator_files, "blind_files_have_no_predictions": blind,
                    "self_kappa_is_1": self_ok, "self_boundary_f1": f1_self,
                    "pass": blind and self_ok and f1_self == 1.0,
                    "note": "answers are copied from the key: a pipeline test of export and scoring, not validation"}
    for s in stalls:
        sdir = out / s["dir"]
        s["checks"] = {k: v["pass"] for k, v in json.loads((sdir / "checks.json").read_text()).items()}
    all_pass = (all(all(s["checks"].values()) for s in stalls) and all(p["pass"] for p in proofs.values())
                and gate_summary["pass"])
    report = {"what": "Ledger v2 end-to-end check on real stalled DeepSeek V4.1 Flash runs (MettleBench), FAKE labels",
              "fake_labels": True, "paid_calls": 0, "command": COMMAND, "source_root": str(root),
              "templates_version": T.TEMPLATES_VERSION, "recipe_cap_tokens": RECIPE_CAP, "all_pass": all_pass,
              "stalls": stalls, "proofs": proofs, "gate": gate_summary, "paid_labeller_estimate": cost_estimate(stalls)}
    _dump(out / "report.json", report)
    for src in opened.values():
        src.close()
    print(json.dumps({"all_pass": all_pass, "proofs": {k: v["pass"] for k, v in proofs.items()},
                      "gate": gate_summary["pass"], "report": str(out / "report.json")}, indent=1))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
