"""Statistics for the v2 study (track S): paired-by-stall contrasts with whole-task resampling, Holm, the screen's
beta-binomial models, the blinded sample-size re-estimate and agreement scores.

Pure Python plus numpy; no model calls. Every function takes plain numbers, so analysis code decides what is paired
and what is blinded. Missing values are dropped by the callers, never filled in.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

Z = statistics.NormalDist()


def mean(xs: Iterable[float | None]) -> float | None:
    v = [float(x) for x in xs if x is not None]
    return statistics.fmean(v) if v else None


def _by_cluster(rows: Sequence[tuple[str, float]]) -> dict[str, list[float]]:
    by: dict[str, list[float]] = defaultdict(list)
    for c, v in rows:
        by[c].append(float(v))
    return dict(by)


def cluster_bootstrap(rows: Sequence[tuple[str, float]], *, n: int = 4000, seed: int = 20261007,
                      level: float = 0.95) -> dict:
    """Mean of stall-level values with a percentile interval from resampling whole clusters (tasks) with
    replacement. rows = [(task_id, stall_value)]."""
    by = _by_cluster(rows)
    keys = sorted(by)
    est = mean(v for _, v in rows)
    if len(keys) < 2:
        return {"estimate": est, "ci": None, "clusters": len(keys), "units": len(rows), "boot": 0}
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        xs = [x for k in rng.choices(keys, k=len(keys)) for x in by[k]]
        stats.append(statistics.fmean(xs))
    stats.sort()
    lo, hi = stats[int((1 - level) / 2 * n)], stats[int((1 + level) / 2 * n) - 1]
    return {"estimate": est, "ci": [lo, hi], "clusters": len(keys), "units": len(rows), "boot": n}


def sign_flip_p(rows: Sequence[tuple[str, float]], *, n: int = 20000, seed: int = 20261007) -> float | None:
    """Two-sided p-value for mean difference = 0, flipping the signs of whole tasks' stall differences together
    (exact when there are at most 16 tasks, Monte Carlo otherwise)."""
    by = _by_cluster(rows)
    keys = sorted(by)
    if not keys:
        return None
    sums = np.array([sum(by[k]) for k in keys])
    total = sum(len(by[k]) for k in keys)
    obs = abs(sums.sum()) / total
    if len(keys) <= 16:
        signs = np.array([[1 if (i >> j) & 1 else -1 for j in range(len(keys))] for i in range(2 ** len(keys))])
        stats = np.abs(signs @ sums) / total
        return float(np.mean(stats >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n, len(keys)))
    stats = np.abs(signs @ sums) / total
    return float((1 + np.sum(stats >= obs - 1e-12)) / (n + 1))


def holm(pvals: dict[str, float | None], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down adjusted p-values and rejections; a missing p-value is never rejected."""
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {k: {"p": None, "p_holm": None, "reject": False} for k, p in pvals.items() if p is None}
    running, stop = 0.0, False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        rej = (not stop) and p <= alpha / (m - i)
        stop = stop or not rej
        out[k] = {"p": p, "p_holm": adj, "reject": rej}
    return out


def ols_intercept(y: Sequence[float], X: Sequence[Sequence[float]]) -> float | None:
    """Intercept of y ~ 1 + centred X (the covariate-adjusted mean difference); None if under-determined."""
    y_arr = np.asarray(y, dtype=float)
    if not len(y_arr):
        return None
    if not X or not len(X[0]):
        return float(y_arr.mean())
    x = np.asarray(X, dtype=float)
    x = x - x.mean(axis=0)
    A = np.column_stack([np.ones(len(y_arr)), x])
    if len(y_arr) <= A.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    return float(coef[0])


def adjusted_bootstrap(rows: Sequence[tuple[str, float, Sequence[float]]], *, n: int = 2000,
                       seed: int = 20261007) -> dict:
    """Covariate-adjusted mean stall difference with whole-task resampling. rows = [(task, diff, covariates)]."""
    est = ols_intercept([r[1] for r in rows], [list(r[2]) for r in rows])
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    keys = sorted(by)
    if est is None or len(keys) < 2:
        return {"estimate": est, "ci": None, "clusters": len(keys), "units": len(rows)}
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        rs = [r for k in rng.choices(keys, k=len(keys)) for r in by[k]]
        v = ols_intercept([r[1] for r in rs], [list(r[2]) for r in rs])
        if v is not None:
            stats.append(v)
    if len(stats) < n * 0.5:
        return {"estimate": est, "ci": None, "clusters": len(keys), "units": len(rows),
                "note": "most resamples were under-determined"}
    stats.sort()
    return {"estimate": est, "ci": [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]],
            "clusters": len(keys), "units": len(rows), "boot_used": len(stats)}


# ---------- screen: how much undoing the current approach helps ----------

def beta_difference(pairs: Sequence[tuple[int, int, int, int]], *, draws: int = 20000, seed: int = 20261007,
                    prior: tuple[float, float] = (0.5, 0.5)) -> dict:
    """Mean over stalls of p_rewind - p_continue, each stall and arm with its own Beta(prior + successes, prior +
    failures) posterior (Jeffreys by default). pairs = [(continue_solves, continue_n, rewind_solves, rewind_n)]."""
    if not pairs:
        return {"estimate": None, "ci": None, "stalls": 0}
    rng = np.random.default_rng(seed)
    a0, b0 = prior
    diff = np.zeros(draws)
    for cs, cn, rs, rn in pairs:
        diff += rng.beta(a0 + rs, b0 + rn - rs, draws) - rng.beta(a0 + cs, b0 + cn - cs, draws)
    diff /= len(pairs)
    return {"model": f"independent Beta({a0},{b0}) per stall and arm; mean over stalls", "stalls": len(pairs),
            "estimate": float(np.mean(diff)), "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
            "p_positive": float(np.mean(diff > 0))}


_lgamma = np.vectorize(math.lgamma)


def _bb_grid(counts: Sequence[tuple[int, int]], mu: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    a = mu[:, None] * kappa[None, :]
    b = (1 - mu)[:, None] * kappa[None, :]
    ll = np.zeros_like(a)
    lbeta_ab = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    for k, n in counts:
        ll += (_lgamma(a + k) + _lgamma(b + n - k) - _lgamma(a + b + n)) - lbeta_ab
    return ll


def hierarchical_difference(pairs: Sequence[tuple[int, int, int, int]], *, draws: int = 20000,
                            seed: int = 20261007) -> dict:
    """Population mean solve rate per arm from a beta-binomial (mu ~ U(0,1), log kappa ~ U(log 0.5, log 1000)) on a
    grid, then mu_rewind - mu_continue. The two arms are fitted independently, which ignores the pairing (wider)."""
    if len(pairs) < 2:
        return {"estimate": None, "ci": None, "stalls": len(pairs)}
    mu = np.linspace(0.005, 0.995, 199)
    kappa = np.exp(np.linspace(math.log(0.5), math.log(1000), 60))
    rng = np.random.default_rng(seed)
    out = {}
    for name, counts in (("continue", [(p[0], p[1]) for p in pairs]), ("rewind", [(p[2], p[3]) for p in pairs])):
        ll = _bb_grid(counts, mu, kappa)
        w = np.exp(ll - ll.max()).sum(axis=1)
        w /= w.sum()
        out[name] = rng.choice(mu, size=draws, p=w) + rng.uniform(-0.0025, 0.0025, draws)
    diff = out["rewind"] - out["continue"]
    return {"model": "beta-binomial per arm, grid posterior, arms independent", "stalls": len(pairs),
            "mu_continue": float(np.mean(out["continue"])), "mu_rewind": float(np.mean(out["rewind"])),
            "estimate": float(np.mean(diff)), "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
            "p_positive": float(np.mean(diff > 0))}


# ---------- blinded variance re-estimate (no arm labels ever reach this function) ----------

def blinded_sample_size(rows: Sequence[tuple[str, float]], *, repeats_per_arm: int, target_difference: float,
                        alpha: float, power: float, planned_stalls: int, max_stalls: int) -> dict:
    """rows = [(stall_id, branch outcome)] pooled over every arm, with no arm labels. The per-stall paired difference
    variance is bounded two ways from the pooled within-stall variance s2w: 2*s2w/R (no stall-by-arm interaction) and
    2*s2w*(1+1/R) (interaction as large as the within-stall spread). Returns the stalls each bound needs."""
    by = _by_cluster(rows)
    vals = [v for _, v in rows]
    if len(by) < 2 or len(vals) < 3:
        return {"stalls": len(by), "branches": len(vals), "note": "too little data"}
    dfs = sum(len(v) - 1 for v in by.values() if len(v) > 1)
    s2w = sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in by.values() if len(v) > 1) / dfs if dfs else None
    z = Z.inv_cdf(1 - alpha / 2) + Z.inv_cdf(power)
    out = {"stalls": len(by), "branches": len(vals), "lumped_variance": statistics.variance(vals),
           "within_stall_pooled_variance": s2w, "repeats_per_arm": repeats_per_arm,
           "target_difference": target_difference, "alpha_two_sided": alpha, "power": power,
           "planned_stalls": planned_stalls, "max_stalls": max_stalls}
    if s2w is None:
        return out | {"note": "no stall has two branches"}
    for name, var_d in (("no_interaction", 2 * s2w / repeats_per_arm),
                        ("worst_case_interaction", 2 * s2w * (1 + 1 / repeats_per_arm))):
        need = math.ceil(z * z * var_d / target_difference ** 2) if target_difference else None
        out[name] = {"var_paired_difference": var_d, "stalls_needed": need,
                     "recommended": None if need is None else min(max(need, planned_stalls), max_stalls)}
    return out


# ---------- agreement ----------

def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labels)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def quantile(xs: Sequence[float], q: float) -> float | None:
    """Linear-interpolation quantile (numpy's default)."""
    v = [float(x) for x in xs if x is not None]
    return float(np.quantile(v, q)) if v else None
