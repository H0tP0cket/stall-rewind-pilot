"""Offline T2.1 construction and validation over a saved store (read-only source, new output store).

  python -m trajectory_retention.observers.t2.offline21 --source STORE --out OUT --iteration N --budget-usd 4 \
      [--repeat-events 40] [--repeat-runs SUBSTR ...]

Variants per checkpoint:
  T2.1-descriptive: declared inclusion + exact dedup + recency turn packing (no unit reservation)
  T2.1-coherent:    same, with the frozen selector's approach unit reserved first
Budgets 1,500 / 3,000 / 6,000 local tokens are all recorded. Treatment packets (full, minus selected approach,
matched random deletions over 3 seeds, same-content prose) are written for S and L checkpoints.
Independent classifier repeats use new repeat indices (distinct receipts), never cache replays.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import tiktoken

from ...store import ArtifactStore
from .jev import JevClient
from .label21 import LABEL21_VERSION, pass_a, pass_b
from .offline import load_env
from .packet21 import (BUDGETS, DEFAULT_BUDGET, PACKET21_VERSION, build_packet, minus_selected, random_unit_deletion,
                       render21)
from .questions21 import REGISTRY_VERSION, registry_record
from .segment21 import SEGMENT21_VERSION
from .sources21 import load_ledger
from .store_ro import ReadOnlyStore
from .units21 import SELECTOR_VERSION, at_checkpoint21, build_pursuits, select_unit

ENC = tiktoken.get_encoding("o200k_base")
_TOK: dict[str, int] = {}


def tokens(text: str) -> int:
    n = _TOK.get(text)
    if n is None:
        n = _TOK[text] = len(ENC.encode(text, disallowed_special=()))
    return n


def label_counts(items) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("role", "certainty", "specificity", "restatement", "output_kind"):
        c = Counter(i.label(name) for i in items if name in i.labels)
        if c:
            out[name] = dict(c)
    out["minimal_context_labels"] = sum(any(l.method.endswith(":minimal-context") for l in i.labels.values()) for i in items)
    out["task_copies"] = sum(i.task_copy for i in items)
    out["truncated_for_display"] = sum(getattr(i, "truncated_for_display", False) for i in items)
    return out


def checkpoint_rows(ledger, links, report_dir: Path | None, write_packets: bool) -> tuple[list[dict], list[dict]]:
    cuts = [cp.event_sequence for cp in ledger.checkpoints]
    rows, manifest = [], []
    for cp in ledger.checkpoints:
        view = at_checkpoint21(ledger, cp, links)
        pursuits = build_pursuits(view, checkpoint_cuts=cuts)
        inferred = build_pursuits(view, checkpoint_cuts=cuts, tier="observer_inferred")
        sel = select_unit(pursuits)
        unit = next((p for p in pursuits if p.unit_id == sel["selected"]), None)
        row: dict[str, Any] = {"checkpoint": f"C{cp.ordinal}", "annotations": sorted(set(cp.annotations)),
                               "stage": cp.stage_id, "items": len(view.items), "selector": sel,
                               "inferred_multi_member_pursuits": sum(len(p.approach_members) >= 2 for p in inferred),
                               "status_counts": dict(Counter(p.status for p in pursuits if len(p.approach_members) >= 2)),
                               "budgets": {}}
        packets = {}
        for budget in BUDGETS + (None,):
            desc = build_packet(view, None, budget, tokens)
            coh = build_packet(view, unit, budget, tokens)
            packets[budget] = (desc, coh)
            key = "uncapped" if budget is None else str(budget)
            row["budgets"][key] = {
                "descriptive_tokens": tokens(render21(view, desc.item_ids)),
                "coherent_tokens": tokens(render21(view, coh.item_ids)),
                "coherent_unit_reserved": coh.unit_reserved, "coherent_unit_exceeds_budget": coh.unit_exceeds_budget,
                "withheld_items": len(coh.withheld_candidates), "protected_evidence": len(coh.protected_evidence),
                "withheld_tokens": sum(tokens(view.by_id[i].text[:700]) + 4 for i in coh.withheld_candidates),
                "skipped_turns": coh.skipped_turns, "packed_turns": coh.packed_turns,
            }
        row["exclusions"] = packets[None][0].exclusions
        rows.append(row)
        interesting = bool({"S", "L"} & set(cp.annotations))
        if interesting:
            coh = packets[DEFAULT_BUDGET][1]
            manifest.append({
                "run_id": ledger.run_id, "model": ledger.model, "checkpoint": f"C{cp.ordinal}",
                "checkpoint_id": cp.checkpoint_id, "annotations": sorted(set(cp.annotations)),
                "selector": sel, "budget": DEFAULT_BUDGET,
                "selected_unit": None if unit is None else {
                    **unit.as_record(),
                    "approach_texts": [view.by_id[m].text for m in unit.approach_members],
                    "evidence_texts": [view.by_id[e].text[:300] for e in unit.associated_evidence]},
                "packet": coh.as_record(),
                "entangled": [p.unit_id for p in pursuits if p.entangled_mixed_members],
            })
            if write_packets and report_dir is not None:
                tag = f"{ledger.run_id.split(':')[1][:8]}-C{cp.ordinal}-{'-'.join(sorted(set(cp.annotations)))}"
                out = report_dir / "packets"
                out.mkdir(parents=True, exist_ok=True)
                desc = packets[DEFAULT_BUDGET][0]
                (out / f"{tag}.descriptive.full.txt").write_text(render21(view, desc.item_ids))
                (out / f"{tag}.coherent.full.txt").write_text(render21(view, coh.item_ids))
                (out / f"{tag}.coherent.full.prose.txt").write_text(render21(view, coh.item_ids, style="prose"))
                if coh.unit_reserved and coh.withheld_candidates:
                    (out / f"{tag}.coherent.minus_selected.txt").write_text(render21(view, minus_selected(coh)))
                    randoms = []
                    for seed in (1, 2, 3):
                        r = random_unit_deletion(view, coh, seed, tokens)
                        randoms.append({k: v for k, v in r.items() if k != "item_ids"})
                        if r["status"] == "matched":
                            (out / f"{tag}.coherent.random_seed{seed}.txt").write_text(render21(view, r["item_ids"]))
                    manifest[-1]["random_controls"] = randoms
    return rows, manifest


def repeat_pass_a(ledgers, client, n_events: int, seed: int = 7) -> dict[str, Any]:
    """Three independent labelings (repeat 0 is the main run; repeats 1 and 2 are new receipts)."""
    rng = random.Random(seed)
    pool = [(L, key) for L in ledgers for key in sorted({(i.seq, i.span.field) for i in L.items
                                                           if i.source_kind not in {"worker_action", "format_error"}})]
    sample = rng.sample(pool, min(n_events, len(pool)))
    per_field: dict[str, list[int]] = defaultdict(list)
    for L, key in sample:
        originals = [i for i in L.items if (i.seq, i.span.field) == key]
        commands = {i.tool_call_id: i.text for i in L.items if i.source_kind == "worker_action" and i.tool_call_id}
        clones = []
        for r in (1, 2):
            c = [copy.deepcopy(i) for i in originals]
            for i in c:
                i.labels = {}
            pass_a(c, L.decoded_fields, commands, client, repeat=r, workers=4)
            clones.append(c)
        for idx, orig in enumerate(originals):
            for name in ("role", "certainty", "specificity", "restatement", "output_kind"):
                vals = [orig.labels.get(name)] + [c[idx].labels.get(name) for c in clones]
                if any(v is None for v in vals) or vals[0].method.startswith("deterministic"):
                    continue
                per_field[name].append(int(len({v.value for v in vals}) == 1))
    return {name: {"items": len(v), "all_three_agree": sum(v) / len(v) if v else None} for name, v in per_field.items()}


def repeat_units(ledger, client, main_links, repeats=(1, 2)) -> list[dict[str, Any]]:
    """Relabel a whole run independently and compare selected units and packets at S/L checkpoints."""
    commands = {i.tool_call_id: i.text for i in ledger.items if i.source_kind == "worker_action" and i.tool_call_id}
    variants = [(ledger, main_links)]
    for r in repeats:
        L = copy.deepcopy(ledger)
        for i in L.items:
            i.labels = {}
        pass_a(L.items, L.decoded_fields, commands, client, repeat=r)
        links, _ = pass_b(L.items, L.decoded_fields, client, repeat=r)
        variants.append((L, L.links + links))
    cuts = [cp.event_sequence for cp in ledger.checkpoints]
    rows = []
    for cp in ledger.checkpoints:
        if not ({"S", "L"} & set(cp.annotations)):
            continue
        sel_units, packets, withheld = [], [], []
        for L, links in variants:
            view = at_checkpoint21(L, cp, links)
            pursuits = build_pursuits(view, checkpoint_cuts=cuts)
            sel = select_unit(pursuits)
            unit = next((p for p in pursuits if p.unit_id == sel["selected"]), None)
            pk = build_packet(view, unit, DEFAULT_BUDGET, tokens)
            sel_units.append(sel["selected"])
            packets.append(set(pk.item_ids))
            withheld.append(set(pk.withheld_candidates))
        jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else None
        rows.append({"checkpoint": f"C{cp.ordinal}", "annotations": sorted(set(cp.annotations)),
                     "selected_units": sel_units, "same_selected_unit_all": len(set(sel_units)) == 1,
                     "packet_jaccard_vs_main": [jac(packets[0], p) for p in packets[1:]],
                     "withheld_jaccard_vs_main": [jac(withheld[0], w) for w in withheld[1:]]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--repeat-events", type=int, default=0)
    ap.add_argument("--repeat-runs", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--jev-route", choices=("openrouter", "typesafe"), default="openrouter",
                    help="typesafe only to reuse receipts recorded before 2026-09-25")
    args = ap.parse_args(argv)
    load_env(Path("~/.config/trajectory-retention/openrouter.env" if args.jev_route == "openrouter"
                  else "~/.config/trajectory-retention/typesafe.env").expanduser())
    out = Path(args.out).expanduser().resolve()
    report_dir = out / f"iteration-{args.iteration}"
    report_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(out / "store")
    client = JevClient(store, budget_usd=args.budget_usd, route=args.jev_route)
    source = ReadOnlyStore(args.source)
    errors: list[str] = []
    log = lambda m: (errors.append(m), print(m, file=sys.stderr))
    (report_dir / "question_registry.json").write_text(json.dumps(registry_record(), indent=1))
    report: dict[str, Any] = {"versions": {"segment": SEGMENT21_VERSION, "questions": REGISTRY_VERSION,
                                           "label": LABEL21_VERSION, "selector": SELECTOR_VERSION,
                                           "packet": PACKET21_VERSION, "classifier": client.model},
                              "source_store": str(Path(args.source).resolve()), "runs": [],
                              "prior_spend_usd": client.spent}
    manifest_all = []
    ledgers, main_links = [], {}
    started = time.time()
    runs = source.run_ids() if not args.runs else [r for r in source.run_ids() if any(s in r for s in args.runs)]
    for run_id in runs:
        t0, s0 = time.time(), client.spent
        L = load_ledger(source, run_id)
        commands = {i.tool_call_id: i.text for i in L.items if i.source_kind == "worker_action" and i.tool_call_id}
        sa = pass_a(L.items, L.decoded_fields, commands, client, workers=args.workers, on_error=log)
        links, sb = pass_b(L.items, L.decoded_fields, client, workers=args.workers, on_error=log)
        all_links = L.links + links
        store.put_record("t21_run", f"{args.iteration}:{run_id}", {
            "versions": report["versions"], "run_id": run_id, "model": L.model,
            "items": [i.as_record() for i in L.items], "links": [l.as_record() for l in all_links],
            "discards": [d.as_record() for d in L.discards]}, run_id=run_id)
        rows, manifest = checkpoint_rows(L, all_links, report_dir, True)
        manifest_all.extend(manifest)
        spent = client.spent - s0
        report["runs"].append({
            "run_id": run_id, "model": L.model, "items": len(L.items), "labels": label_counts(L.items),
            "discards": dict(Counter(d.reason for d in L.discards)), "passA": sa, "passB": sb,
            "links": dict(Counter(f"{l.relation}/{l.tier}" for l in all_links)),
            "cost": {"t21_classifier_usd": spent, "worker_usd": L.worker_cost_usd, "old_observer_usd": L.old_observer_cost_usd},
            "checkpoints": rows, "seconds": time.time() - t0})
        ledgers.append(L)
        main_links[run_id] = all_links
        print(f"{run_id.split(':')[1][:8]} {L.model} items={len(L.items)} t2.1=${spent:.4f} worker=${L.worker_cost_usd:.3f} "
              f"pairs={sb.get('pairs_asked')} {time.time()-t0:.0f}s", flush=True)
        (report_dir / "report.json").write_text(json.dumps(report, indent=1))
    with (report_dir / "unit_manifest.jsonl").open("w") as fh:
        for row in manifest_all:
            fh.write(json.dumps(row) + "\n")
    if args.repeat_events:
        report["repeat_pass_a"] = repeat_pass_a(ledgers, client, args.repeat_events)
    if args.repeat_runs:
        report["repeat_units"] = {}
        for L in ledgers:
            if any(s in L.run_id for s in args.repeat_runs):
                report["repeat_units"][L.run_id.split(":")[1][:8]] = repeat_units(L, client, main_links[L.run_id])
    report["totals"] = {"t21_classifier_usd": sum(r["cost"]["t21_classifier_usd"] for r in report["runs"]),
                        "worker_usd": sum(r["cost"]["worker_usd"] for r in report["runs"]),
                        "old_observer_usd": sum(r["cost"]["old_observer_usd"] for r in report["runs"]),
                        "calls": client.calls, "cache_hits": client.cache_hits, "spend_total_usd": client.spent,
                        "errors": len(errors), "seconds": time.time() - started}
    report["errors_sample"] = errors[:30]
    (report_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["totals"], indent=1))
    for key in ("repeat_pass_a", "repeat_units"):
        if key in report:
            print(key, json.dumps(report[key], indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
