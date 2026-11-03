"""Offline T2.2 construction and validation over saved stores (read-only sources, new output store).

  python -m trajectory_retention.observers.t2.offline22 --source STORE [--source STORE ...] --out OUT \
      --iteration N --budget-usd 25 [--probe-budget-usd 5] [--runs SUBSTR ...] [--no-probe]

Per run: T2.2 ledger labels (pass A), artifact-candidate retrieval and explicit relations (pass B), and the
analysis-only claim-vs-grader annotation. Per S checkpoint (and the last checkpoint, for size statistics):
approach units, the frozen selector, packets at every budget, candidate filter views (each scope) with
paraphrase removal, borderline reports, matched random controls, workspace-leak flags and recovery probes.
Filter families here are validation candidates, not ratified arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import tiktoken

from ...store import ArtifactStore
from .artifacts22 import artifacts_for
from .claims22 import claim_vs_grader, summarize as summarize_claims
from .filters22 import CANDIDATE_FAMILIES, apply_filter, candidate_family, matched_randoms
from .glm import GlmClient
from .jev import JevClient
from .label22 import (LABEL22_VERSION, paraphrase_check, pass_a22, pass_b22, pass_b22_same_response, pass_c22,
                      pass_c22_same_response)
from .leak22 import workspace_leak
from .offline import load_env
from .packet22 import PACKET22_VERSION, build_packet22, render22
from .probe22 import distractor_pool, probe_family_choice
from .questions22 import REGISTRY_VERSION, registry_record
from .segment21 import SEGMENT21_VERSION
from .sources21 import load_ledger
from .store_ro import ReadOnlyStore
from .units22 import SELECTOR22_VERSION, at_checkpoint21, build_structure, select_current
from .vocab22 import BUDGETS, DEFAULT_BUDGET, T22_VERSION

ENC = tiktoken.get_encoding("o200k_base")
_TOK: dict[str, int] = {}


def tokens(text: str) -> int:
    n = _TOK.get(text)
    if n is None:
        n = _TOK[text] = len(ENC.encode(text, disallowed_special=()))
    return n


def size_stats(approaches) -> dict[str, Any]:
    sizes = [len(a.members) for a in approaches]
    return {"n": len(sizes), "single_member": sum(s == 1 for s in sizes), "multi_member": sum(s >= 2 for s in sizes),
            "median": statistics.median(sizes) if sizes else None, "max": max(sizes, default=None),
            "hist": dict(Counter(min(s, 10) for s in sizes))}


def _slim(record: dict[str, Any]) -> dict[str, Any]:
    """Drop long id lists from a view record for the report (full records go to the output store)."""
    return {k: v for k, v in record.items() if k not in ("item_ids", "paraphrase") or not isinstance(v, (list, dict))}


VARIANTS = {"explicit": False, "explicit+inferred": True}
# Validation families: the ratified default scopes, plus global versions of the role filters as diagnostics
# (whether a role removal is real when the current approach is small); none of these is a ratified arm.
FAMILIES = dict(CANDIDATE_FAMILIES) | {f"{n}@global": candidate_family(n, "global")
                                       for n in ("forget_theories", "forget_plans", "forget_done_claims")}
PROBE_MAX_PAIRS = 36  # removed items probed per view (deterministic sample when more); cost/latency bound  # units joined by explicit links only / plus inferred tier


def checkpoint_block(src, store, L, links, cp, cuts, jev, glm, probe: bool, packets_dir: Path) -> dict[str, Any]:
    view = at_checkpoint21(L, cp, links)
    block: dict[str, Any] = {"checkpoint_id": cp.checkpoint_id, "annotations": list(cp.annotations), "variants": {}}
    for variant, join_inferred in VARIANTS.items():
        block["variants"][variant] = variant_block(src, store, L, view, cp, cuts, jev, glm, probe, packets_dir,
                                                   variant, join_inferred, links)
    return block


def variant_block(src, store, L, view, cp, cuts, jev, glm, probe, packets_dir, variant, join_inferred,
                  all_links) -> dict[str, Any]:
    structure = build_structure(view, checkpoint_cuts=cuts, join_inferred=join_inferred)
    approaches = structure.approaches
    sel = select_current(approaches)
    current = next((a for a in approaches if a.unit_id == sel.get("selected")), None)
    # Diagnostic only (not a ratified selector): the most recently active approach with no size minimum.
    latest = max(approaches, key=lambda a: (a.last_activity_seq, -a.first_seq), default=None)
    block: dict[str, Any] = {"approaches": size_stats(approaches), "clusters": len(structure.clusters),
                             "structure_stats": structure.stats, "selector": sel,
                             "current_size": len(current.members) if current else None,
                             "latest_approach_size": len(latest.members) if latest else None, "packets": {}}
    frozen = None
    for budget in BUDGETS:
        packet = build_packet22(view, approaches, current, budget, tokens)
        text = render22(view, packet.item_ids, approaches=approaches)
        block["packets"][budget] = {"tokens": tokens(text), "packed_approaches": len(packet.packed_approaches),
                                    "skipped_approaches": len(packet.skipped_approaches),
                                    "unit_exceeds_budget": packet.unit_exceeds_budget,
                                    "current_tokens": packet.approach_tokens.get(packet.current or "", None)}
        if budget == DEFAULT_BUDGET:
            frozen, frozen_text = packet, text
    if "S" not in cp.annotations:
        return block
    tag = "explicit" if not join_inferred else "inferred"
    stem = cp.checkpoint_id.replace(":", "_").replace("/", "_") + f"__{tag}"
    (packets_dir / f"{stem}__full.txt").write_text(frozen_text)
    by_id = view.by_id
    para = lambda kept_items, removed_items: paraphrase_check(jev, kept_items, removed_items)
    block["views"] = []
    for name, spec in FAMILIES.items():
        fv = apply_filter(frozen, view, approaches, current, spec, tokens, paraphrase_fn=para)
        randoms = matched_randoms(frozen, view, fv, tokens) if fv.removed_ids else []
        if fv.removed_ids and not any(r["status"] == "matched" for r in randoms) and spec.scope != "global":
            randoms = matched_randoms(frozen, view, fv, tokens, scope="global")
        fv.random_controls = randoms
        if fv.removed_ids:
            # W x T ablation (user, 2026-09-27): roll the workspace back to where the removed content begins, i.e.
            # the latest checkpoint strictly before the earliest removed item became available. Branchability is
            # checked when the experiment is built, not here.
            first = min(by_id[i].available_seq for i in fv.removed_ids)
            before = [c for c in L.checkpoints if c.event_sequence < first and c.ordinal <= cp.ordinal]
            rb = before[-1] if before else None
            fv.rollback_checkpoint = {"checkpoint_id": rb.checkpoint_id if rb else None,
                                      "ordinal": rb.ordinal if rb else None, "first_removed_seq": first,
                                      "rule": "latest checkpoint before the earliest removed item"}
            acts = fv.removed_actions
            texts = [a.get("text") for a in acts if a.get("text")]
            seqs = [by_id[a["action_id"]].seq for a in acts if a.get("action_id") in by_id] or None
            fv.workspace_leak = workspace_leak(src, L.run_id, cp.checkpoint_id, texts, action_seqs=seqs) if texts else \
                {"status": "no_removed_actions"}
            vtext = render22(view, fv.item_ids, approaches=approaches)
            (packets_dir / f"{stem}__{name}.txt").write_text(vtext)
            if probe and not (join_inferred and spec.scope == "global"):  # global views barely differ by variant
                matched = [r for r in randoms if r["status"] == "matched"][:1]
                rviews = {f"seed{r['seed']}": (render22(view, r["item_ids"], approaches=approaches), r["deleted_ids"])
                          for r in matched}
                try:
                    removed = sorted(fv.removed_ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
                    fv.recovery_probe = probe_family_choice(
                        glm, family=name, full_text=frozen_text, view_text=vtext,
                        removed_items=[by_id[i] for i in removed[:PROBE_MAX_PAIRS]], count=tokens,
                        random_views=rviews, distractor_pool=_pool(L, cp, all_links))
                    fv.recovery_probe["sampled_from"] = len(removed)
                except Exception as exc:  # recorded, never silently dropped
                    fv.recovery_probe = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        rec = fv.as_record()
        store.put_record("t22_view", f"{ITERATION}:{cp.checkpoint_id}:{variant}:{name}", rec, run_id=L.run_id)
        block["views"].append(_slim(rec) | {"family": name, "random_controls": [{k: r.get(k) for k in ("status", "seed", "scope",
                                                                  "target_tokens", "deleted_tokens")} for r in randoms]})
    return block


_POOLS: dict[str, Any] = {}
ITERATION = ""  # set by main(); every stored record id carries it, so iterations can share one store


def _pool(L, cp, all_links):
    """Distractors for the forced-choice probe: same-run statements from after the cut (built once per checkpoint)."""
    if cp.checkpoint_id not in _POOLS:
        _POOLS.clear()
        _POOLS[cp.checkpoint_id] = distractor_pool(L.items, cp.event_sequence, all_links)
    return _POOLS[cp.checkpoint_id]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--probe-budget-usd", type=float, default=5.0)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args(argv)
    global ITERATION
    ITERATION = args.iteration
    load_env(Path("~/.config/trajectory-retention/openrouter.env").expanduser())
    out = Path(args.out).expanduser().resolve()
    report_dir = out / f"iteration-{args.iteration}"
    packets_dir = report_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(out / "store")
    jev = JevClient(store, budget_usd=args.budget_usd)
    # The probe is a measuring instrument, not the ratified judge: low effort keeps calls short and bounded.
    glm = GlmClient(store, budget_usd=args.probe_budget_usd, max_tokens=4096, reasoning={"effort": "low"})
    errors: list[str] = []
    log = lambda m: (errors.append(m), print(m, file=sys.stderr))
    (report_dir / "question_registry.json").write_text(json.dumps(registry_record(), indent=1))
    report: dict[str, Any] = {"versions": {"t22": T22_VERSION, "segment": SEGMENT21_VERSION, "questions": REGISTRY_VERSION,
                                           "label": LABEL22_VERSION, "selector": SELECTOR22_VERSION,
                                           "packet": PACKET22_VERSION, "classifier": jev.model, "probe": glm.model},
                              "sources": [str(Path(s).resolve()) for s in args.source], "runs": []}
    started = time.time()
    for source_path in args.source:
        src = ReadOnlyStore(source_path)
        runs = src.run_ids() if not args.runs else [r for r in src.run_ids() if any(s in r for s in args.runs)]
        for run_id in runs:
            t0, j0, g0 = time.time(), jev.spent, glm.spent
            L = load_ledger(src, run_id)
            arts = artifacts_for(L.items, L.links)
            commands = {i.tool_call_id: i.text for i in L.items if i.source_kind == "worker_action" and i.tool_call_id}
            sa = pass_a22(L.items, L.decoded_fields, commands, jev, workers=args.workers, on_error=log)
            rp = pass_b22(L.items, L.decoded_fields, jev, artifacts_by_id=arts, workers=args.workers, on_error=log)
            sb = rp.stats
            rc = pass_c22(L.items, L.decoded_fields, jev, artifacts_by_id=arts, explicit_links=rp.links,
                          workers=args.workers, on_error=log)
            # Same-response passes (recall fix): the 3 nearest preceding usable sentences of the same response, asked
            # in separate requests so the pass B/C requests above stay byte-identical (cached).
            rs = pass_b22_same_response(L.items, L.decoded_fields, jev, asked_pairs=rp.asked_pairs(),
                                        workers=args.workers, on_error=log)
            rcs = pass_c22_same_response(L.items, L.decoded_fields, jev, explicit_links=rp.links + rs.links,
                                         asked_pairs=rc.asked_pairs(), workers=args.workers, on_error=log)
            all_links = L.links + rp.links + rp.candidate_links + rc.links + rs.links + rcs.links
            claims = claim_vs_grader(src, run_id, L.items)
            store.put_record("t22_run", f"{args.iteration}:{run_id}", {
                "versions": report["versions"], "run_id": run_id, "model": L.model,
                "items": [i.as_record() | {"artifacts": sorted(arts[i.item_id])} for i in L.items],
                "links": [l.as_record() for l in all_links], "discards": [d.as_record() for d in L.discards]}, run_id=run_id)
            store.put_record("t22_relation_pass", f"{args.iteration}:{run_id}", rp.as_record(), run_id=run_id)
            store.put_record("t22_same_approach_pass", f"{args.iteration}:{run_id}", rc.as_record(), run_id=run_id)
            store.put_record("t22_relation_pass_same_response", f"{args.iteration}:{run_id}", rs.as_record(), run_id=run_id)
            store.put_record("t22_same_approach_pass_same_response", f"{args.iteration}:{run_id}", rcs.as_record(),
                             run_id=run_id)
            store.put_record("t22_claim_grader", f"{args.iteration}:{run_id}", {"run_id": run_id, "claims": claims},
                             run_id=run_id)
            cuts = [c.event_sequence for c in L.checkpoints]
            targets = [c for c in L.checkpoints if "S" in c.annotations] + L.checkpoints[-1:]
            blocks = []
            for cp in targets:
                try:
                    blocks.append(checkpoint_block(src, store, L, all_links, cp, cuts, jev, glm, not args.no_probe, packets_dir))
                except Exception as exc:  # recorded, never silently dropped
                    log(f"{run_id} {cp.checkpoint_id}: {type(exc).__name__}: {exc}")
            report["runs"].append({
                "run_id": run_id, "model": L.model, "items": len(L.items),
                "roles": dict(Counter(i.label("role") for i in L.items)), "passA": sa, "passB": sb, "passC": rc.stats,
                "passB_same_response": rs.stats, "passC_same_response": rcs.stats,
                "links": dict(Counter(f"{l.relation}/{l.tier}" for l in all_links)),
                "claims": summarize_claims(claims), "checkpoints": blocks,
                "cost": {"jev_usd": jev.spent - j0, "glm_usd": glm.spent - g0, "worker_usd": L.worker_cost_usd},
                "seconds": time.time() - t0})
            print(f"{run_id.split(':')[1][:8]} {L.model} items={len(L.items)} jev=${jev.spent - j0:.4f} "
                  f"glm=${glm.spent - g0:.4f} {time.time() - t0:.0f}s", flush=True)
            (report_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    report["totals"] = {"jev_usd": jev.spent, "glm_usd": glm.spent, "errors": len(errors), "seconds": time.time() - started}
    report["errors_sample"] = errors[:40]
    (report_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["totals"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
