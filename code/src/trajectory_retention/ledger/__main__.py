"""Build one stall's Ledger outputs into its analysis directory.

  python -m trajectory_retention.ledger build --store STORE --run RUN_ID --checkpoint CHECKPOINT_ID --out DIR \
      --labeller {deepseek,replay,scripted} [--stage boundaries|full|all] [--recipe-cap N] [--budget-usd B] \
      [--pin-only SLUG] [--max-chars N]

Writes into DIR: boundary.json (frozen), ledger-boundary.json, and with the full stage ledger.json,
review_view_ledger.json (arm 2), review_view_ledger_flat.json (arm set v3: arms 2 and 5, no approach layer),
review_view_context.json (arm 3; arm set v3: arms 3 and 6), recipe.json (arm 6, with --recipe-cap) and
labeller_calls.json; labeller receipts go to DIR/labeller (an ArtifactStore). The source store is opened read-only.

``deepseek`` makes paid calls (pin and settings from configs/v2/settings.json, or --pin-only plus the worker
settings); ``replay`` serves only existing receipts and never spends; ``scripted`` writes fake labels for pipeline
checks and marks every output ``fake_labels``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..observers.t2.store_ro import ReadOnlyStore
from ..store import ArtifactStore
from .labeller import ScriptedLabeller, study_labeller
from .passes import DEFAULT_MAX_CHARS, build_boundaries, build_full
from .render import render_recipe, render_review_view

WORKER_SETTINGS = {"temperature": 1, "reasoning": {"effort": "max"}, "max_tokens": 262144}  # brief-interfaces


def make_client(kind: str, store_dir: Path, *, budget_usd: float = 0.0, pin_only: str | None = None):
    store = ArtifactStore(store_dir)
    if kind == "scripted":
        return ScriptedLabeller(store)
    if kind == "terra-pilot":
        from .labeller import CodexPilotLabeller
        return CodexPilotLabeller(store, store_dir / "codex-sessions")
    if kind not in ("deepseek", "replay"):
        raise ValueError(f"unknown labeller: {kind}")
    # same shape as the worker pin in configs/v2/provider_smoke.py: one provider, fallbacks off
    pin = ({"only": [pin_only], "order": [pin_only], "allow_fallbacks": False, "require_parameters": True,
            **({"quantizations": [pin_only.split("/")[-1]]} if "/" in pin_only else {})} if pin_only else None)
    settings = WORKER_SETTINGS if pin_only else None
    if kind == "deepseek":
        env = Path("~/.config/trajectory-retention/openrouter.env").expanduser()
        if env.is_file() and "OPENROUTER_API_KEY" not in os.environ:
            for line in env.read_text().splitlines():
                line = line.strip().removeprefix("export ")
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return study_labeller(store, budget_usd=budget_usd, pin=pin, settings=settings, replay=(kind == "replay"))


def build_stall(source: ReadOnlyStore, run_id: str, checkpoint_id: str, out: Path, client, *, stage: str = "all",
                recipe_cap: int | None = None, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    if stage in ("boundaries", "all"):
        result["boundary"] = build_boundaries(source, run_id, checkpoint_id, client, out_dir=out,
                                              max_chars=max_chars)["boundary"]
    if stage in ("full", "all"):
        ledger = build_full(source, run_id, checkpoint_id, client, boundary_dir=out, out_dir=out, max_chars=max_chars)
        result["ledger"] = ledger
        for kind, name in (("ledger", "review_view_ledger.json"), ("ledger_flat", "review_view_ledger_flat.json"),
                           ("context", "review_view_context.json")):
            view = render_review_view(ledger, kind)
            (out / name).write_text(json.dumps(view, indent=1, ensure_ascii=False) + "\n")
            result[kind + "_view"] = view
        if recipe_cap is not None:
            recipe = render_recipe(ledger, cap_tokens=recipe_cap)
            (out / "recipe.json").write_text(json.dumps(recipe, indent=1, ensure_ascii=False) + "\n")
            result["recipe"] = recipe
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--store", required=True)
    b.add_argument("--run", required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--labeller", choices=["deepseek", "replay", "scripted"], required=True)
    b.add_argument("--stage", choices=["boundaries", "full", "all"], default="all")
    b.add_argument("--recipe-cap", type=int)
    b.add_argument("--budget-usd", type=float, default=0.0)
    b.add_argument("--pin-only", help="provider slug while configs/v2/settings.json does not exist")
    b.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser().resolve()
    client = make_client(a.labeller, out / "labeller", budget_usd=a.budget_usd, pin_only=a.pin_only)
    src = ReadOnlyStore(a.store)
    try:
        res = build_stall(src, a.run, a.checkpoint, out, client, stage=a.stage, recipe_cap=a.recipe_cap,
                          max_chars=a.max_chars)
    finally:
        src.close()
    summary = {"out": str(out), "boundary": {k: (res.get("boundary") or {}).get(k) for k in
                                             ("status", "current_approach_id", "rewind_point")}}
    if "ledger" in res:
        summary["ledger_status"] = res["ledger"]["status"]
        summary["items"] = len(res["ledger"]["items"])
        summary["labeller_cost_usd"] = res["ledger"]["labeller"]["cost_usd"]
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
