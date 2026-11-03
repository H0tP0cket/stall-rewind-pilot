"""Deterministic, repeat-matched descriptive contrasts of fixed-target grades.

The contrast signs implement experiment.md. Numerical horizons, bootstrap
settings and main-study inference remain calibration choices. Missing cells
never become zero, and an engineering checkpoint never becomes S1.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import random
from typing import Any, Mapping, Sequence

from ..contracts import TreatmentManifest
from .continuations import ContinuationRecord
from .conditions import manifest_execution_problems


CONTRASTS: dict[str, dict[str, int]] = {
    "H_at_source": {"continue": 1, "reset_history": -1},
    "H_at_donor": {"rollback_later_history": 1, "rollback_reset": -1},
    "W_at_native_H": {"continue": 1, "rollback_later_history": -1},
    "W_at_fresh_H": {"reset_history": 1, "rollback_reset": -1},
    "WxH": {"continue": 1, "reset_history": -1, "rollback_later_history": -1, "rollback_reset": 1},
    "O_retained": {"continue": 1, "o_masked": -1},
    "summary_vs_none": {"summary": 1, "reset_history": -1},
    "prose_vs_none": {"t_prose": 1, "reset_history": -1},
    "typed_vs_none": {"t_typed": 1, "reset_history": -1},
    "typed_vs_summary": {"t_typed": 1, "summary": -1},
    "typed_vs_same_T_prose": {"t_typed": 1, "t_prose": -1},
    "facts_vs_full": {"t_facts_only": 1, "t_typed": -1},
    "facts_lessons_vs_full": {"t_facts_lessons": 1, "t_typed": -1},
    "T_at_donor": {"recomposition": 1, "rollback_reset": -1},
    "rollback_with_T": {"recomposition": 1, "t_typed": -1},
    "WxT": {"t_typed": 1, "reset_history": -1, "recomposition": -1, "rollback_reset": 1},
}
BASELINE_CONTRASTS = {"agentrewind_vs_continue": {"agentrewind": 1, "continue": -1}}


def _json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class AnalysisSpec:
    horizons: tuple[int, ...]
    n_boot: int = 0
    seed: int = 0
    confidence: float = 0.95
    status: str = "development_descriptive; main-study estimator settings pending calibration"

    def __post_init__(self) -> None:
        if any(type(h) is not int or h < 0 for h in self.horizons):
            raise ValueError("horizons must be non-negative integers")
        if type(self.n_boot) is not int or self.n_boot < 0 or not 0 < self.confidence < 1:
            raise ValueError("invalid bootstrap settings")


def _stratum(m: TreatmentManifest) -> dict[str, Any]:
    return {"worker_model": m.worker_model, "model_settings": dict(m.metadata.get("model_settings", {})),
            "allowance": m.generated_token_limit, "runtime_policy": m.runtime_policy,
            "selection_kind": m.selection_kind,
            "recovery_settings": m.metadata.get("recovery_settings")}


def _target(m: TreatmentManifest) -> str:
    # The full target manifest includes denominator/check bindings when supplied.
    return _json({"target_id": m.target_id, "stage_id": m.stage_id,
                  "manifest": m.metadata.get("recovery_target_manifest_id"),
                  "target": m.metadata.get("recovery_target")})


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize_samples(samples: Mapping[str, float], family_of: Mapping[str, str | None], *,
                      n_boot: int, seed: int, confidence: float) -> dict[str, Any]:
    """Source-weighted mean; resample whole families, never repeated branches."""
    families: dict[str, list[float]] = defaultdict(list)
    for key, value in sorted(samples.items()):
        if family_of.get(key) is not None:
            families[str(family_of[key])].append(value)
    mean = sum(samples.values()) / len(samples) if samples else None
    reason = None
    if any(family_of.get(k) is None for k in samples):
        reason = "family_binding_missing"
    elif len(families) < 2:
        reason = "fewer_than_two_families"
    elif not n_boot:
        reason = "bootstrap_not_requested"
    low = high = None
    if reason is None:
        rng, keys, boot = random.Random(seed), sorted(families), []
        for _ in range(n_boot):
            values = [v for _ in keys for v in families[rng.choice(keys)]]
            boot.append(sum(values) / len(values))
        alpha = (1 - confidence) / 2
        low, high = _quantile(boot, alpha), _quantile(boot, 1 - alpha)
    return {"mean": mean, "n_sources_observed": len(samples), "n_families": len(families),
            "ci_low": low, "ci_high": high, "interval_unavailable_reason": reason,
            "family_means": {f: sum(v) / len(v) for f, v in sorted(families.items())}}


def matched_effects(*, manifests: Sequence[TreatmentManifest], records: Sequence[ContinuationRecord],
                    spec: AnalysisSpec, family_by_source: Mapping[str, str] | None = None,
                    contrasts: Mapping[str, Mapping[str, int]] | None = None) -> dict[str, Any]:
    """Export all planned blocks, including missing/failed/censored cells.

    Match on source, target, model/settings, nominal allowance, runtime policy,
    selection kind and repeat ID. Average complete repeat differences within
    each source/target before averaging sources. Four-cell interactions use
    the same complete four-cell repeat population. Distinct conditions sharing
    an arm key are ambiguous and rejected, not arbitrarily pooled.
    """
    if contrasts is None:
        contrasts = {**CONTRASTS, **(BASELINE_CONTRASTS if any(
            m.metadata.get("arm_id") == "agentrewind" for m in manifests) else {})}
    family_by_source = family_by_source or {}
    planned = {m.treatment_id: m for m in manifests}
    if len(planned) != len(manifests):
        raise ValueError("duplicate treatment manifest IDs")
    actual: dict[str, ContinuationRecord] = {}
    branches: set[str] = set()
    for r in records:
        if r.treatment_id not in planned:
            raise ValueError(f"continuation has no planned manifest: {r.treatment_id}")
        if r.treatment_id in actual or r.branch_run_id in branches:
            raise ValueError("duplicate rollout; reused cells are one observation")
        m = planned[r.treatment_id]
        if (r.source_checkpoint_id, r.worker_model, r.repeat_id, r.allowance, r.arm_id) != (
                m.source_checkpoint_id, m.worker_model, m.repeat_id, m.generated_token_limit, m.metadata.get("arm_id")):
            raise ValueError("continuation identity disagrees with manifest")
        expected_lineage = {"target_id": m.target_id, "stage_id": m.stage_id,
            "target_manifest_id": m.metadata.get("recovery_target_manifest_id"),
            "model_settings": dict(m.metadata.get("model_settings", {})),
            "runtime_policy": m.runtime_policy, "selection_kind": m.selection_kind}
        if any(r.lineage.get(k) != v for k, v in expected_lineage.items()):
            raise ValueError("continuation target/condition lineage disagrees with manifest")
        actual[r.treatment_id] = r
        branches.add(r.branch_run_id)
    strata: dict[str, dict[tuple[str, str, str], dict[str, TreatmentManifest]]] = defaultdict(dict)
    for m in sorted(manifests, key=lambda m: m.treatment_id):
        stratum = _json(_stratum(m))
        block = (m.source_checkpoint_id, _target(m), m.repeat_id)
        cells = strata[stratum].setdefault(block, {})
        arm = str(m.metadata.get("arm_id"))
        if arm in cells:
            raise ValueError(f"ambiguous duplicate arm in matched block: {block}, {arm}")
        cells[arm] = m
    estimates, blocks_out = [], []
    for sid, blocks in sorted(strata.items()):
        stratum = json.loads(sid)
        for name, weights in contrasts.items():
            if not weights or sum(weights.values()) != 0:
                raise ValueError(f"contrast {name} must contain zero-sum weights")
            relevant = {b: cells for b, cells in blocks.items() if set(cells) & set(weights)}
            source_t_versions: dict[tuple[str, str], set[str]] = defaultdict(set)
            source_effective_detectors: dict[tuple[str, str], set[str]] = defaultdict(set)
            repeated_conditions: dict[tuple[str, str, str], set[str]] = defaultdict(set)
            for (source, target, _), cells in relevant.items():
                for arm in weights:
                    if arm in cells:
                        m = cells[arm]
                        record = actual.get(m.treatment_id)
                        detector = record.lineage.get("effective_detector_config") if record else None
                        if isinstance(detector, Mapping):
                            source_effective_detectors[(source, target)].add(_json(detector))
                        version_id = m.metadata.get("packet", {}).get("source_version_id")
                        if version_id:
                            source_t_versions[(source, target)].add(version_id)
                        repeated_conditions[(source, target, arm)].add(_json({
                            "component_donors": dict(m.component_donors), "history_policy": m.history_policy,
                            "packet_hashes": [p.sha256 for p in m.packet_refs],
                            "source_scope": dict(m.source_scope), "mask": m.metadata.get("mask"),
                            "baseline_policy": m.metadata.get("baseline_policy")}))
            effective_detectors = set().union(*source_effective_detectors.values()) if source_effective_detectors else set()
            for h in sorted(set(spec.horizons)):
                per_source: dict[str, list[float]] = defaultdict(list)
                solve_per_source: dict[str, list[float]] = defaultdict(list)
                families = {}
                missing = Counter()
                observed_repeats = solve_repeats = 0
                for (source, target, repeat), cells in sorted(relevant.items()):
                    values, solves, detail = {}, {}, {}
                    sample_id = _json([source, target])
                    families[sample_id] = family_by_source.get(source)
                    packet_cells = [cells[a] for a in weights if a in cells and
                                    cells[a].metadata.get("packet", {}).get("kind", "none") != "none"]
                    t_versions = {m.metadata["packet"].get("source_version_id") for m in packet_cells
                                  if m.metadata["packet"].get("source_version_id")}
                    source_scopes = {_json({"session_scope": m.source_scope.get("session_scope", "full_trajectory"),
                                           "include_returned_reasoning": m.source_scope.get("include_returned_reasoning", False)})
                                     for m in packet_cells}
                    binding_error = "different_frozen_T_versions" if len(t_versions) > 1 else (
                        "different_packet_source_scopes" if len(source_scopes) > 1 else None)
                    if any(m.metadata["packet"].get("kind") != "summary" and
                           not m.metadata["packet"].get("source_version_id") for m in packet_cells):
                        binding_error = "frozen_T_identity_missing"
                    if any("session_scope" not in m.source_scope for m in packet_cells):
                        binding_error = "packet_source_scope_missing"
                    if len(source_t_versions[(source, target)]) > 1:
                        binding_error = "source_T_not_frozen_across_repeats"
                    if len(effective_detectors) > 1:
                        binding_error = "different_effective_detector_configs"
                    if not binding_error and any(len(repeated_conditions[(source, target, a)]) > 1 for a in weights):
                        binding_error = "treatment_condition_changed_across_repeats"
                    for arm in weights:
                        m = cells.get(arm)
                        r = actual.get(m.treatment_id) if m else None
                        status = "arm_not_planned" if m is None else "rollout_missing" if r is None else r.horizon_status(h)
                        value = r.score_at(h) if r else None
                        solve = r.solved_by(h) if r else None
                        execution_errors = manifest_execution_problems(m, r.lineage) if m and r else []
                        if execution_errors:
                            status, value, solve = ";".join(execution_errors), None, None
                        if binding_error:
                            status, value, solve = binding_error, None, None
                        detail[arm] = {"status": status, "score": value, "solved": solve,
                                       "treatment_id": m.treatment_id if m else None,
                                       "branch_run_id": r.branch_run_id if r else None,
                                       "execution_binding_problems": execution_errors,
                                       "executed_recovery_settings": r.lineage.get("executed_recovery_settings") if r else None,
                                       "effective_detector_config": r.lineage.get("effective_detector_config") if r else None}
                        if value is None:
                            missing[status] += 1
                        else:
                            values[arm] = value
                        if solve is not None:
                            solves[arm] = int(solve)
                    difference = sum(weights[a] * values[a] for a in weights) if len(values) == len(weights) else None
                    solve_difference = sum(weights[a] * solves[a] for a in weights) if len(solves) == len(weights) else None
                    if difference is not None:
                        per_source[sample_id].append(difference)
                        observed_repeats += 1
                    if solve_difference is not None:
                        solve_per_source[sample_id].append(solve_difference)
                        solve_repeats += 1
                    blocks_out.append({"contrast": name, "horizon": h, **stratum,
                        "source_checkpoint_id": source, "target_binding": json.loads(target), "repeat_id": repeat,
                        "family_id": family_by_source.get(source), "cells": detail,
                        "difference": difference, "solve_difference": solve_difference})
                samples = {k: sum(v) / len(v) for k, v in per_source.items()}
                solve_samples = {k: sum(v) / len(v) for k, v in solve_per_source.items()}
                options = {"n_boot": spec.n_boot, "seed": spec.seed, "confidence": spec.confidence}
                estimates.append({"contrast": name, "weights": dict(weights), "horizon": h, **stratum,
                    "comparison_class": "agent_direct_policy_baseline" if "agentrewind" in weights else "retention_contrast",
                    "workspace_attribution": "W+R" if stratum["runtime_policy"] == "restore_captured" else
                                             "W" if stratum["runtime_policy"] == "irrelevant" else "unsupported",
                    "n_repeat_blocks_total": len(relevant), "n_repeat_blocks_observed": observed_repeats,
                    "n_repeat_blocks_missing": len(relevant) - observed_repeats,
                    "n_sources_total": len({(s, t) for s, t, _ in relevant}),
                    "n_families_total": len({family_by_source[s] for s, _, _ in relevant if s in family_by_source}),
                    "n_sources_missing_family": len({s for s, _, _ in relevant if s not in family_by_source}),
                    "missing_cells_by_reason": dict(sorted(missing.items())),
                    "score_effect": summarize_samples(samples, families, **options),
                    "solve_effect": {**summarize_samples(solve_samples, families, **options),
                                     "n_repeat_blocks_observed": solve_repeats},
                    "interpretation_limits": ["Engineering selections are instrument checks, not primary S1 samples.",
                        "Summary vs T changes selection and computation; token matching and extraction quality require calibration.",
                        "Repeated rollouts are averaged within source; family intervals require at least two known families."],
                    "per_source": samples})
    return {"version": "matched-effects.v3", "analysis_spec": dict(spec.__dict__),
            "population_scope": "planned treatment manifests; all natural runs and failed upstream selections require the natural/campaign denominator tables",
            "planned_manifest_count": len(manifests), "observed_rollout_count": len(records),
            "estimand": "mean across source/target states of mean complete matched-repeat current-score contrasts at assigned allowance horizons",
            "matching_fields": ["source_checkpoint", "fixed_target_manifest", "worker_model", "model_settings",
                                "allowance", "runtime_policy", "selection_kind", "repeat_id"],
            "missingness_policy": "complete contrast blocks; available cases descriptive, no outcome imputation",
            "curve_policy": "current/last graded score at a justified cut; missing-grade barriers; qualified terminal absorption only within assigned allowance",
            "solve_policy": "ever observed solve by the horizon; distinct from current solve state after a regression",
            "estimates": estimates, "matched_blocks": blocks_out}
