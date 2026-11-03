"""Offline T2 re-extraction over a saved mining store (read-only), with a metrics report.

Usage:
  python -m trajectory_retention.observers.t2.offline --source STORE --out OUT_DIR --iteration N \
      --budget-usd 5 [--repeat-sample 40] [--runs RUN_SUBSTRING ...]

The source store is opened read-only. Classifier calls, items, links and reports
are written to OUT_DIR (an ArtifactStore plus iteration report files). Completed
identical classifier calls are reused from OUT_DIR's store, so reruns only pay
for changed requests. Worker packets are written for S/L checkpoints and each
run's final checkpoint so they can be read directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import tiktoken

from ...store import ArtifactStore
from .derive import at_checkpoint, kept, pursuit_summary
from .jev import JevClient
from .label import LABEL_VERSION, pass1, pass2
from .render import (PACKET_TOKEN_BUDGET, RENDER_VERSION, drop_pursuit, drop_roles, drop_status, random_matched,
                     removed_by, render, select_for_packet)
from .schema import T2_VERSION, WORKER_SOURCES
from .segment import SEGMENT_VERSION
from .sources import load_run
from .store_ro import ReadOnlyStore

ENC = tiktoken.get_encoding("o200k_base")


def tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def load_env(path: Path) -> None:
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def old_packet_tokens(store: ReadOnlyStore) -> dict[str, dict[str, int]]:
    """Old (v12) primary T typed-packet size at each checkpoint where it completed."""
    from ..render import render_typed
    from ..tstore import TVersion
    out: dict[str, dict[str, int]] = {}
    for _, pair in store.records("observer_scope_pair_v1"):
        try:
            version = TVersion.from_record(store.get_record("t_version", pair["primary_version_id"]))
        except KeyError:
            continue
        text = render_typed(version).text
        out[pair["checkpoint_id"]] = {"packet_tokens": tokens(text), "entries": len(version.entries),
                                      "entry_text_tokens": tokens("\n".join(e.text for e in version.entries))}
    return out


def label_distribution(items) -> dict[str, Any]:
    prose = [i for i in items if i.source_kind in WORKER_SOURCES]
    obs = [i for i in items if i.source_kind in {"observation"}]
    return {
        "prose_items": len(prose), "observation_items": len(obs),
        "role": dict(Counter(i.label("role") for i in prose)),
        "stance": dict(Counter(i.label("stance") for i in prose)),
        "keep_prose": dict(Counter(i.label("keep") for i in prose)),
        "keep_observation": dict(Counter(i.label("keep") for i in obs)),
        "kept_role": dict(Counter(i.label("role") for i in prose if kept(i))),
        "spec_statement": dict(Counter(i.label("spec_statement") for i in prose)),
        "task_copies": sum(i.task_copy for i in items),
        "by_source_kind": dict(Counter(i.source_kind for i in items)),
    }


def reproducibility(src_items, decoded, client, sample: int, seed: int, commands=None) -> dict[str, Any]:
    """Re-ask pass-1 questions for a deterministic sample of events with repeat=1 and compare labels."""
    rng = random.Random(seed)
    events = sorted({(i.seq, i.span.field) for i in src_items if i.source_kind in WORKER_SOURCES | {"observation"}})
    chosen = set(rng.sample(events, min(sample, len(events))))
    originals = [i for i in src_items if (i.seq, i.span.field) in chosen]
    clones = [copy.deepcopy(i) for i in originals]
    for c in clones:
        c.labels = {}
    pass1(clones, decoded, client, repeat=1, commands=commands)
    agree: dict[str, list[int]] = defaultdict(list)
    diffs: dict[str, list[float]] = defaultdict(list)
    for a, b in zip(originals, clones):
        for name in ("role", "stance", "keep"):
            la, lb = a.labels.get(name), b.labels.get(name)
            if la is None or lb is None or la.method.startswith("deterministic"):
                continue
            agree[name].append(int(la.value == lb.value))
            if la.probabilities and lb.probabilities:
                keys = set(la.probabilities) | set(lb.probabilities)
                diffs[name].append(max(abs(la.probabilities.get(k, 0) - lb.probabilities.get(k, 0)) for k in keys))
    return {name: {"n": len(v), "label_agreement": sum(v) / len(v) if v else None,
                   "median_max_prob_diff": statistics.median(diffs[name]) if diffs[name] else None}
            for name, v in agree.items()}


def selection_stability(source, client, args, main_items) -> list[dict[str, Any]]:
    """Relabel whole runs with repeat=1 and compare the S/L checkpoint packet selections (Jaccard)."""
    rows = []
    for run_id in source.run_ids():
        if not any(s in run_id for s in args.selection_repeat) or run_id not in main_items:
            continue
        first_items, _ = main_items[run_id]
        src = load_run(source, run_id)
        commands = {i.tool_call_id: i.text for i in src.items if i.source_kind == "worker_action" and i.tool_call_id}
        pass1(src.items, src.decoded_fields, client, workers=args.workers, repeat=1, commands=commands)
        links, _ = pass2(src.items, client, workers=args.workers, repeat=1)
        first = load_run(source, run_id)
        by_id = {i.item_id: i for i in first_items}
        for item in first.items:
            item.labels = by_id[item.item_id].labels
        first_links = [l for l in _stored_links(run_id, args)]
        for cp in src.checkpoints:
            if not ({"S", "L"} & set(cp.annotations)):
                continue
            a_view = at_checkpoint(first, cp, first.links + first_links)
            b_view = at_checkpoint(src, cp, src.links + links)
            a_sel = select_for_packet(a_view, PACKET_TOKEN_BUDGET, tokens)
            b_sel = select_for_packet(b_view, PACKET_TOKEN_BUDGET, tokens)
            a_kept = {i.item_id for i in a_view.items if kept(i)}
            b_kept = {i.item_id for i in b_view.items if kept(i)}
            jac = lambda x, y: len(x & y) / len(x | y) if x | y else None
            rows.append({"run": run_id.split(":")[1][:8], "checkpoint": f"C{cp.ordinal}", "annotations": list(cp.annotations),
                         "kept_jaccard": jac(a_kept, b_kept), "selected_jaccard": jac(a_sel, b_sel),
                         "kept": [len(a_kept), len(b_kept)], "selected": [len(a_sel), len(b_sel)]})
    return rows


_LINK_CACHE: dict[str, list] = {}


def _stored_links(run_id, args):
    from .schema import Link
    store = ArtifactStore(Path(args.out).expanduser().resolve() / "store")
    record = store.get_record("t2_run", f"{args.iteration}:{run_id}")
    return [Link(**{k: v for k, v in l.items()}) for l in record["links"] if l["relation"] not in {"same_turn", "result_of"}]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--repeat-sample", type=int, default=40)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--selection-repeat", nargs="*", default=[],
                    help="run substrings to relabel fully with repeat=1 and compare stall-checkpoint selections")
    ap.add_argument("--jev-route", choices=("openrouter", "typesafe"), default="openrouter",
                    help="typesafe only to reuse receipts recorded before 2026-09-25")
    args = ap.parse_args(argv)

    load_env(Path("~/.config/trajectory-retention/openrouter.env" if args.jev_route == "openrouter"
                  else "~/.config/trajectory-retention/typesafe.env").expanduser())
    out = Path(args.out).expanduser().resolve()
    report_dir = out / f"iteration-{args.iteration}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "packets").mkdir(exist_ok=True)
    out_store = ArtifactStore(out / "store")
    client = JevClient(out_store, budget_usd=args.budget_usd, route=args.jev_route)
    source = ReadOnlyStore(args.source)
    errors: list[str] = []
    log = lambda msg: (errors.append(msg), print(msg, file=sys.stderr))

    started = time.time()
    old = old_packet_tokens(source)
    report: dict[str, Any] = {
        "versions": {"t2": T2_VERSION, "segment": SEGMENT_VERSION, "label": LABEL_VERSION, "render": RENDER_VERSION,
                     "classifier": client.model},
        "source_store": str(Path(args.source).resolve()), "iteration": args.iteration,
        "prior_classifier_spend_usd": client.spent, "runs": [],
    }
    all_items = []
    runs = source.run_ids()
    if args.runs:
        runs = [r for r in runs if any(s in r for s in args.runs)]
    for run_id in runs:
        t0 = time.time()
        spent0 = client.spent
        src = load_run(source, run_id)
        commands = {i.tool_call_id: i.text for i in src.items if i.source_kind == "worker_action" and i.tool_call_id}
        s1 = pass1(src.items, src.decoded_fields, client, workers=args.workers, on_error=log, commands=commands)
        links, s2 = pass2(src.items, client, workers=args.workers, on_error=log)
        all_links = src.links + links
        spent = client.spent - spent0
        out_store.put_record("t2_run", f"{args.iteration}:{run_id}", {
            "versions": report["versions"], "run_id": run_id, "model": src.model,
            "items": [i.as_record() for i in src.items], "links": [l.as_record() for l in all_links]},
            run_id=run_id)
        cp_rows = []
        for cp in src.checkpoints:
            view = at_checkpoint(src, cp, all_links)
            selected = select_for_packet(view, PACKET_TOKEN_BUDGET, tokens)
            packet = render(view, selected=selected)
            uncapped = render(view, selected=select_for_packet(view, None, tokens), name="uncapped")
            row = {"checkpoint": f"C{cp.ordinal}", "annotations": list(cp.annotations), "stage": cp.stage_id,
                   "items_available": len(view.items), "items_kept": sum(kept(i) for i in view.items),
                   "packet_tokens": tokens(packet.text), "uncapped_tokens": tokens(uncapped.text),
                   "selected_items": len(selected), "pursuits": len(view.pursuits),
                   "statuses": dict(Counter(s for s, _ in view.status.values())),
                   "old": old.get(cp.checkpoint_id)}
            cp_rows.append(row)
            interesting = {"S", "L"} & set(cp.annotations) or cp is src.checkpoints[-1]
            if interesting:
                tag = f"{run_id.split(':')[1][:8]}-C{cp.ordinal}{'-S' if 'S' in cp.annotations else ''}"
                views = {"full": packet, "uncapped": uncapped}
                views["no_plans"] = render(view, drop_roles({"intended_action"}), name="no_plans", selected=selected)
                views["no_approaches"] = render(view, drop_roles({"approach"}), name="no_approaches", selected=selected)
                views["no_dropped"] = render(view, drop_status({"withdrawn", "replaced"}, view), name="no_dropped",
                                             selected=selected)
                summary = pursuit_summary(view, selected)
                if summary:
                    top = summary[0]["root"]
                    inc = drop_pursuit(view, top)
                    views["minus_top_pursuit"] = render(view, inc, name="minus_top_pursuit", selected=selected)
                    views["random_matched"] = render(view, random_matched(view, removed_by(view, inc, selected), 17,
                                                                          selected), name="random_matched",
                                                     selected=selected)
                for name, pk in views.items():
                    (report_dir / "packets" / f"{tag}.{name}.txt").write_text(pk.text)
                row["views_tokens"] = {k: tokens(v.text) for k, v in views.items()}
                row["pursuit_summary"] = summary[:8]
        run_row = {
            "run_id": run_id, "model": src.model, "items": len(src.items),
            "labels": label_distribution(src.items), "pass1": s1, "pass2": s2,
            "links": dict(Counter(f"{l.relation}/{l.tier}" for l in all_links)),
            "cost": {"t2_classifier_usd": spent, "worker_usd": src.worker_cost_usd,
                     "old_observer_usd": src.old_observer_cost_usd},
            "unreplayed_reasoning_events": src.unreplayed_reasoning_events,
            "checkpoints": cp_rows, "seconds": time.time() - t0,
        }
        report["runs"].append(run_row)
        all_items.append((src.items, src.decoded_fields, commands))
        print(f"{run_id.split(':')[1][:8]} {src.model} items={len(src.items)} t2=${spent:.4f} "
              f"worker=${src.worker_cost_usd:.3f} old=${src.old_observer_cost_usd:.3f} {time.time()-t0:.0f}s", flush=True)
        (report_dir / "report.json").write_text(json.dumps(report, indent=1))

    if args.repeat_sample:
        repro = defaultdict(lambda: {"n": 0, "agree": 0.0})
        per_run = max(1, args.repeat_sample // max(1, len(all_items)))
        results = []
        for items, decoded, commands in all_items:
            results.append(reproducibility(items, decoded, client, per_run, seed=11, commands=commands))
        merged: dict[str, Any] = {}
        for name in ("role", "stance", "keep"):
            n = sum(r.get(name, {}).get("n", 0) for r in results)
            agree = sum((r.get(name, {}).get("label_agreement") or 0) * r.get(name, {}).get("n", 0) for r in results)
            diffs = [r[name]["median_max_prob_diff"] for r in results if name in r and r[name]["median_max_prob_diff"] is not None]
            merged[name] = {"n": n, "label_agreement": agree / n if n else None,
                            "median_of_run_median_max_prob_diff": statistics.median(diffs) if diffs else None}
        report["reproducibility"] = merged
    if args.selection_repeat:
        report["selection_stability"] = selection_stability(source, client, args, main_items={
            items[0].run_id: (items, commands) for items, _, commands in all_items if items})
    report["totals"] = {
        "t2_classifier_usd_this_iteration": sum(r["cost"]["t2_classifier_usd"] for r in report["runs"]),
        "worker_usd": sum(r["cost"]["worker_usd"] for r in report["runs"]),
        "old_observer_usd": sum(r["cost"]["old_observer_usd"] for r in report["runs"]),
        "classifier_calls": client.calls, "cache_hits": client.cache_hits,
        "classifier_spend_total_usd": client.spent, "errors": len(errors), "seconds": time.time() - started,
    }
    report["errors_sample"] = errors[:30]
    (report_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["totals"], indent=1))
    if "reproducibility" in report:
        print(json.dumps(report["reproducibility"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
