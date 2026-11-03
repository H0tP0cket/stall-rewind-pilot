"""Resource-response curves and matched paired comparisons.

Estimands follow experiment.md: Q(w,t,b) is the fixed-target score after an
additional worker budget b. Curves are per arm; comparisons are paired within
S1 (same source checkpoint), then aggregated with cluster-aware uncertainty
(clusters = task family by default; here the caller passes cluster keys).

Current scores preserve regressions. Qualified terminal results can absorb
within their assigned allowance; the separate actual-compute view ends at
measured use. Every estimate retains missingness and its observation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .continuations import ContinuationRecord


@dataclass(frozen=True)
class CurvePoint:
    horizon: int
    mean: float | None
    n_observed: int
    n_total: int
    n_censored: int  # not observed at this horizon: stopped earlier or accounting unknown
    solve_rate: float | None
    n_technical_failure: int
    n_accounting_incomplete: int
    n_solve_observed: int
    curve: str = "allowance"
    mean_best_observed_score: float | None = None
    n_best_observed: int = 0
    current_solve_rate: float | None = None
    n_current_solve_observed: int = 0
    n_allowance_absorbed: int = 0


def arm_curve(records: Sequence[ContinuationRecord], horizons: Sequence[int], *, curve: str = "allowance") -> list[CurvePoint]:
    if len(records) > 1:
        import json
        from .conditions import condition_identity, execution_problems
        conditions = {json.dumps([r.worker_model, r.allowance, r.lineage.get("model_settings"),
                                  r.lineage.get("runtime_policy"), r.lineage.get("selection_kind"),
                                  condition_identity(r.lineage)], sort_keys=True) for r in records}
        if len(conditions) > 1 or any(execution_problems(r.lineage) for r in records):
            raise ValueError("cannot pool heterogeneous or unknown recovery execution settings")
    out = []
    for h in horizons:
        scores = [r.score_at(h, curve=curve) for r in records]
        solves = [r.solved_by(h, curve=curve) for r in records]
        bests = [v for r in records if (v := r.best_score_at(h, curve=curve)) is not None]
        current_solves = [v for r in records if (v := r.current_solve_at(h, curve=curve)) is not None]
        obs = [s for s in scores if s is not None]
        sol = [s for s in solves if s is not None]
        out.append(CurvePoint(
            horizon=h, mean=(sum(obs) / len(obs)) if obs else None, n_observed=len(obs), n_total=len(records),
            n_censored=len(records) - len(obs), solve_rate=(sum(sol) / len(sol)) if sol else None,
            n_technical_failure=sum(r.technical_failure for r in records),
            n_accounting_incomplete=sum(not r.horizon_complete for r in records), n_solve_observed=len(sol),
            curve=curve, mean_best_observed_score=sum(bests) / len(bests) if bests else None,
            n_best_observed=len(bests), current_solve_rate=sum(current_solves) / len(current_solves) if current_solves else None,
            n_current_solve_observed=len(current_solves),
            n_allowance_absorbed=sum(curve == "allowance" and h > r.observed_horizon
                                    and r.observed_at(h, curve=curve) for r in records),
        ))
    return out


@dataclass(frozen=True)
class PairedEstimate:
    contrast: str
    horizon: int
    mean_difference: float | None
    n_pairs: int
    n_s1: int
    n_clusters: int
    ci_low: float | None
    ci_high: float | None
    per_s1: Mapping[str, float | None]
    note: str
    n_repeat_blocks_total: int = 0
    n_repeat_blocks_missing: int = 0


def _strict_estimate(records: Sequence[ContinuationRecord], weights: Mapping[str, int], *,
                     horizon: int, cluster_of: Callable[[str], str], n_boot: int = 2000,
                     seed: int = 0, ci: float = 0.95) -> PairedEstimate:
    from collections import defaultdict
    import json
    from .effects import summarize_samples
    from .conditions import condition_identity, execution_problems

    relevant = [r for r in records if r.arm_id in weights]
    if any(execution_problems(r.lineage) for r in relevant):
        raise ValueError("unknown or mismatched recovery execution settings; use matched_effects for missingness")
    # This legacy single-estimate API cannot return model/settings strata.
    # Refuse heterogeneous populations; matched_effects exports all strata.
    strata = {json.dumps([r.worker_model, r.allowance, r.lineage.get("model_settings"),
                          r.lineage.get("runtime_policy"), r.lineage.get("selection_kind"),
                          condition_identity(r.lineage)], sort_keys=True)
              for r in relevant}
    if len(strata) > 1:
        raise ValueError("heterogeneous model/settings/allowance population; use matched_effects")
    blocks = defaultdict(dict)
    targets = {}
    for r in relevant:
        target = json.dumps([r.lineage.get("target_id"), r.lineage.get("stage_id"),
                             r.lineage.get("target_manifest_id")], sort_keys=True)
        if r.source_checkpoint_id in targets and targets[r.source_checkpoint_id] != target:
            raise ValueError("source has different target bindings; use matched_effects")
        targets[r.source_checkpoint_id] = target
        key = (r.source_checkpoint_id, target, r.repeat_id)
        if r.arm_id in blocks[key]:
            raise ValueError("duplicate rollout in matched repeat cell")
        blocks[key][r.arm_id] = r
    diffs = defaultdict(list)
    for (source, _, _), cells in sorted(blocks.items()):
        values = {a: cells[a].score_at(horizon) if a in cells else None for a in weights}
        if all(v is not None for v in values.values()):
            diffs[source].append(sum(weights[a] * values[a] for a in weights))
    per_source = {s: sum(diffs[s]) / len(diffs[s]) if diffs[s] else None for s in sorted(targets)}
    samples = {s: d for s, d in per_source.items() if d is not None}
    estimate = summarize_samples(samples, {s: cluster_of(s) for s in samples},
                                 n_boot=n_boot, seed=seed, confidence=ci)
    pairs = sum(len(d) for d in diffs.values())
    return PairedEstimate(
        contrast=" + ".join(f"{w}*{a}" for a, w in weights.items()), horizon=horizon,
        mean_difference=estimate["mean"], n_pairs=pairs, n_s1=len(targets), n_clusters=estimate["n_families"],
        ci_low=estimate["ci_low"], ci_high=estimate["ci_high"], per_s1=per_source,
        note="complete matched repeats averaged within source; " +
             (estimate["interval_unavailable_reason"] or "family cluster bootstrap; descriptive interval"),
        n_repeat_blocks_total=len(blocks), n_repeat_blocks_missing=len(blocks)-pairs)


def paired_difference(records: Sequence[ContinuationRecord], *, arm_a: str, arm_b: str, horizon: int,
                      cluster_of: Callable[[str], str], n_boot: int = 2000, seed: int = 0,
                      ci: float = 0.95) -> PairedEstimate:
    """Strict matched-repeat contrast. Use matched_effects for planned missing cells/strata."""
    if arm_a == arm_b:
        raise ValueError("contrast arms must differ")
    return _strict_estimate(records, {arm_a: 1, arm_b: -1}, horizon=horizon,
                            cluster_of=cluster_of, n_boot=n_boot, seed=seed, ci=ci)


def interaction_2x2(records: Sequence[ContinuationRecord], *, horizon: int, cluster_of: Callable[[str], str],
                    cells: Mapping[str, str] | None = None, **kw: Any) -> dict[str, Any]:
    """Four-cell interaction uses complete four-cell repeats, with family uncertainty."""
    cells = cells or {"S1_T": "t_typed", "S1_noT": "reset_history", "D_T": "recomposition", "D_noT": "rollback_reset"}
    pairs = {"T_effect_at_S1": ("S1_T", "S1_noT"), "T_effect_at_D": ("D_T", "D_noT"),
             "rollback_effect_noT": ("D_noT", "S1_noT"), "rollback_effect_T": ("D_T", "S1_T")}
    result = {"horizon": horizon, "cells": dict(cells)}
    for label, (a, b) in pairs.items():
        result[label] = paired_difference(records, arm_a=cells[a], arm_b=cells[b], horizon=horizon,
                                          cluster_of=cluster_of, **kw).__dict__
    estimate = _strict_estimate(records, {cells["S1_T"]: 1, cells["S1_noT"]: -1,
                                          cells["D_T"]: -1, cells["D_noT"]: 1}, horizon=horizon,
                               cluster_of=cluster_of, **kw)
    result["WxT_interaction"] = {**estimate.__dict__, "mean": estimate.mean_difference,
                                "n_s1_complete": sum(v is not None for v in estimate.per_s1.values())}
    return result
