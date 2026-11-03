"""Run-to-run agreement of the units labeller (fixed units + classification, labelled step by step; user,
2026-10-22) on the round-2 packets' stall prefixes, beside the find-items numbers of r2b/agreement/report.json.

  set -a; . ~/.config/trajectory-retention/openrouter.env; set +a
  .venv/bin/python configs/v2/linkpilot/units_agreement.py run --runs 1 2 3 --cap-usd 1.50 [--upto K] [--parallel] p6 p1
  .venv/bin/python configs/v2/linkpilot/units_agreement.py analyse p6 p1

The labeller is the study client (configs/v2/settings.json: DeepSeek V4.1 Flash 1017 on Parasail fp8, temperature 1,
max reasoning, 262,144-token cap), through ``ledger.units.build_units``. Each (run, packet) has its own receipt store
(LINKPILOT/units/run<N>/<pid>/labeller), so every run sends its requests as new paid calls; within a run the steps go
one call at a time (step k needs the frozen labels of steps < k). With --parallel the (run, packet) sequences run side
by side under ONE shared budget guard: before every call it holds back the cost of a full-length answer (the call's
input at $0.30 per million tokens plus 262,144 output tokens at $1.20 per million, the provider prices fitted from the
r2b receipts), counting every call in flight and everything already recorded under LINKPILOT/units; a call that does
not fit waits for calls in flight to settle and is refused once none is in flight, which ends that sequence where it
is (a prefix of steps, still usable: labels are per step and prefix-only). An existing document of a (run, packet) is
passed back as frozen labels, so a rerun continues where it stopped and never revises a labelled step.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from trajectory_retention.contracts import canonical_json  # noqa: E402
from trajectory_retention.ledger import units as U  # noqa: E402
from trajectory_retention.ledger.text import cohen_kappa  # noqa: E402
from trajectory_retention.observers.t2.jev import BudgetExceeded, _ref  # noqa: E402
from trajectory_retention.observers.t2.store_ro import ReadOnlyStore  # noqa: E402

LINKPILOT = Path("/Users/sidsri/projects/llm-trajectories/artifacts/calibration/linkpilot")
OUT = LINKPILOT / "units"
OLD_REPORT = LINKPILOT / "r2b" / "agreement" / "report.json"
ROOT = LINKPILOT / "units"  # every receipt store under here counts against the cap, whatever --out is
# Provider prices per token (input, output) for the hold-back: Parasail fp8 (1017) fitted exactly from the r2b
# receipts (agreement_r2b.py); StreamLake fp8 (V4.1 Flash): the provider smoke fit (0.141, 0.564) rounded up, since the first units receipt billed
# about $0.59 per million output tokens.
PRICES = {"parasail/fp8": (0.30 / 1e6, 1.20 / 1e6), "streamlake/fp8": (0.15 / 1e6, 0.60 / 1e6)}
SETTINGS = REPO / "configs" / "v2" / "settings.json"
DRIVER_VERSION = "linkpilot-units-agreement/v2"


def study_client(store_dir: Path, *, budget: float, settings_path: Path, replay: bool = False):
    """The study labeller with the pin and settings of ``settings_path``; the key is read from the env file the way
    ``ledger.__main__.make_client`` reads it (never printed)."""
    import os
    from trajectory_retention.ledger.labeller import study_labeller
    from trajectory_retention.store import ArtifactStore
    env = Path("~/.config/trajectory-retention/openrouter.env").expanduser()
    if not replay and env.is_file() and "OPENROUTER_API_KEY" not in os.environ:
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    client = study_labeller(ArtifactStore(store_dir), budget_usd=budget, settings_path=settings_path, replay=replay)
    client.usd_in, client.usd_out = PRICES[client.pin["only"][0]]
    return client


# ---------------------------------------------------------------- spend

def recorded(store_dir: Path) -> tuple[float, int]:
    """Recorded cost of a receipt store and the number of attempts that may have been billed without a cost."""
    if not (store_dir / "journal.sqlite3").exists():
        return 0.0, 0
    s = ReadOnlyStore(store_dir)
    try:
        recs = [s.get_json(_ref(r["ref"])) for k in ("glm_call", "glm_call_failed") for r in s.list_records(k)]
    finally:
        s.close()
    lost = sum(1 for r in recs for a in r.get("attempts") or [] if a.get("http_status") == 0
               or (a.get("http_status") or 0) >= 500) + sum(1 for r in recs if r.get("cost_usd") is None)
    return sum(r.get("cost_usd") or 0.0 for r in recs), lost


def spent_everywhere() -> float:
    """Everything recorded under LINKPILOT/units, plus a worst case for every call lost without a receipt
    (lost-calls.json files, e.g. calls in flight when a run was stopped)."""
    lost = sum(x.get("worst_usd", 0.0) for f in ROOT.rglob("lost-calls.json") for x in json.loads(f.read_text()))
    return sum(recorded(d)[0] for d in ROOT.rglob("labeller") if d.is_dir()) + lost


class SharedBudget:
    def __init__(self, cap: float, spent: float):
        self.cap, self.spent, self.held, self.inflight = cap, spent, 0.0, 0
        self.cond = threading.Condition()
        self.lost_calls = 0

    def reserve(self, est: float) -> bool:
        with self.cond:
            while self.spent + self.held + est > self.cap:
                if self.inflight == 0:
                    return False
                self.cond.wait()
            self.held += est
            self.inflight += 1
            return True

    def settle(self, est: float, cost: float) -> None:
        with self.cond:
            self.held -= est
            self.spent += cost
            self.inflight -= 1
            self.cond.notify_all()


class Guarded:
    """The study client behind the shared guard. Receipts already on disk are replayed without a reservation."""

    def __init__(self, inner, budget: SharedBudget):
        self.inner, self.budget = inner, budget
        self.store = inner.store

    def bind_context(self, ctx):
        bind = getattr(self.inner, "bind_context", None)
        if bind:
            bind(ctx)

    def describe(self):
        return self.inner.describe()

    def complete(self, messages, *, purpose, repeat=0, max_tokens=None):
        body = self.inner.request_body(messages, max_tokens=max_tokens)
        raw = canonical_json(body)
        record_id = f"{hashlib.sha256(raw).hexdigest()}:r{repeat}"
        try:
            saved = self.inner.store.get_record("glm_call", record_id)
        except KeyError:
            saved = None
        if saved is not None and saved.get("status") == "ok":
            return self.inner.complete(messages, purpose=purpose, repeat=repeat, max_tokens=max_tokens)
        est = len(raw) / 3.2 * self.inner.usd_in + body["max_tokens"] * self.inner.usd_out
        if not self.budget.reserve(est):
            raise BudgetExceeded(f"shared cap ${self.budget.cap:.2f}: spent ${self.budget.spent:.4f}, "
                                 f"held ${self.budget.held:.4f}, this call holds ${est:.4f}")
        try:
            ans = self.inner.complete(messages, purpose=purpose, repeat=repeat, max_tokens=max_tokens)
        except BaseException:
            self.budget.lost_calls += 1
            self.budget.settle(est, est)  # counted as billed in full until the receipts say otherwise
            raise
        self.budget.settle(est, ans.cost_usd if ans.cost_usd is not None else est)
        return ans


# ---------------------------------------------------------------- run

def doc_path(n: int, pid: str) -> Path | None:
    d = OUT / f"run{n}" / pid
    for name in ("ledger-units.json", "ledger-units.partial.json"):
        if (d / name).exists():
            return d / name
    return None


def run_sequence(n, pid, entry, budget, upto, lock, labeller="deepseek", settings_path=SETTINGS):
    d = OUT / f"run{n}" / pid
    if labeller == "scripted":  # FAKE labels, no call: a dry run of this driver and its analysis
        from trajectory_retention.store import ArtifactStore
        client = U.scripted_units_labeller(ArtifactStore(d / "labeller"))
    else:
        # replay: serve existing receipts only (rebuilds a document of a stopped run; never spends)
        inner = study_client(d / "labeller", budget=budget.cap, settings_path=settings_path,
                             replay=(labeller == "replay"))
        client = Guarded(inner, budget)
    prev = doc_path(n, pid)
    frozen = json.loads(prev.read_text()) if prev else None
    t0 = time.monotonic()

    def on_step(lab):
        cost = sum(t.get("cost_usd") or 0.0 for t in lab["tries"])
        with lock:
            print(f"run{n} {pid} step {lab['step']}: {lab['status']} tries={[t.get('reason') or t['status'] for t in lab['tries']]}"
                  f" ${cost:.4f} | shared spent ${budget.spent:.4f} held ${budget.held:.4f} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)

    src = ReadOnlyStore(entry["store"])
    try:
        doc = U.build_units(src, entry["run_id"], entry["stall_checkpoint_id"], client, out_dir=d, frozen=frozen,
                            upto=upto, on_step=on_step)
    finally:
        src.close()
    if doc["status"] != "partial" and (d / "ledger-units.partial.json").exists():
        (d / "ledger-units.partial.json").unlink()
    if doc["stall_point"] != entry["stall_point_step"]:
        raise ValueError(f"{pid}: {doc['stall_point']} steps, packet stall point {entry['stall_point_step']}")
    (d / "view.txt").write_text(U.render_units_view(doc)["text"])
    meta = {"driver": DRIVER_VERSION, "packet_id": pid, "run": n, "store": entry["store"],
            "status": doc["status"], "labelled_steps": doc["labelled_steps"], "stopped": doc["stopped"],
            "reused_frozen_steps": doc["reused_frozen_steps"], "seconds": round(time.monotonic() - t0, 1),
            "model": doc["labeller"]["model"], "provider": doc["labeller"]["provider"],
            "settings_file": str(settings_path),
            "labeller": {k: v for k, v in doc["labeller"].items() if k != "protocol"}}
    (d / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    with lock:
        print(f"run{n} {pid} DONE {doc['status']} steps {len(doc['labelled_steps'])}/{doc['stall_point']} "
              f"stopped={doc['stopped']}", flush=True)


def run(runs, cap, pids, upto, parallel, labeller="deepseek", settings_path=SETTINGS):
    manifest = json.load(open(LINKPILOT / "manifest.json"))["packets"]
    budget = SharedBudget(cap, spent_everywhere())
    print(f"cap ${cap:.2f}; already recorded under {OUT}: ${budget.spent:.4f}", flush=True)
    lock = threading.Lock()
    jobs = [(n, pid) for pid in pids for n in runs]

    def one(n, pid):
        try:
            run_sequence(n, pid, manifest[pid], budget, upto, lock, labeller, settings_path)
        except Exception as exc:
            with lock:
                print(f"run{n} {pid} ERROR {type(exc).__name__}: {exc}", flush=True)

    if parallel:
        ts = [threading.Thread(target=one, args=j) for j in jobs]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    else:
        for j in jobs:
            one(*j)
    print(json.dumps({"shared_spent_usd": round(budget.spent, 4), "recorded_now_usd": round(spent_everywhere(), 4),
                      "calls_lost_counted_in_full": budget.lost_calls}), flush=True)


# ---------------------------------------------------------------- analyse

def fleiss(rows: list[list[str]]) -> float | None:
    """Fleiss' kappa over items each rated by the same number of raters (categories are the strings)."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    m = len(rows[0])
    if m < 2 or any(len(r) != m for r in rows):
        return None
    cats = sorted({c for r in rows for c in r})
    n = len(rows)
    p = {c: sum(r.count(c) for r in rows) / (n * m) for c in cats}
    pbar = sum((sum(r.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for r in rows) / n
    pe = sum(v * v for v in p.values())
    if pe >= 1.0:
        return 1.0 if pbar == 1.0 else None
    return (pbar - pe) / (1 - pe)


def binary_kappa_counts(a: int, b: int, c: int, d: int) -> float | None:
    n = a + b + c + d
    if not n:
        return None
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def fleiss_binary_sets(sets: list[set], universe: int) -> float | None:
    """Fleiss' kappa for 'linked or not' over a universe of items, given each rater's set of linked items."""
    m = len(sets)
    union = set().union(*sets)
    if universe <= 0 or m < 2:
        return None
    pos_total = sum(len(s) for s in sets)
    p1 = pos_total / (universe * m)
    pe = p1 * p1 + (1 - p1) * (1 - p1)
    agree = 0.0
    for x in union:
        k = sum(x in s for s in sets)
        agree += (k * k + (m - k) ** 2 - m) / (m * (m - 1))
    agree += universe - len(union)  # items nobody linked: full agreement
    pbar = agree / universe
    return None if pe >= 1 else (pbar - pe) / (1 - pe)


def r3(x):
    return None if x is None else round(x, 3)


def dice(a: set, b: set):
    return r3(2 * len(a & b) / (len(a) + len(b))) if (a or b) else None


def load_docs(pid):
    out = {}
    for d in sorted(OUT.glob("run[0-9]*"), key=lambda p: int(p.name[3:])):
        p = doc_path(int(d.name[3:]), pid)
        if p:
            out[int(d.name[3:])] = json.loads(p.read_text())
    return out


def call_costs(n, pid, doc):
    store_dir = OUT / f"run{n}" / pid / "labeller"
    s = ReadOnlyStore(store_dir)
    rows = []
    try:
        for k, lab in doc["labels"].items():
            for t in lab["tries"]:
                if not t.get("record_id"):
                    continue
                try:
                    rec = s.get_record("glm_call", t["record_id"])
                except KeyError:  # scripted (fake) labels: no paid receipt
                    continue
                body = s.get_json(_ref(rec["response_ref"]))
                u = body.get("usage") or {}
                rows.append({"step": int(k), "repeat": t["repeat"], "status": t["status"],
                             "cost_usd": rec.get("cost_usd"), "prompt_tokens": u.get("prompt_tokens"),
                             "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                             "completion_tokens": u.get("completion_tokens"),
                             "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                             "finish_reason": rec.get("finish_reason"), "seconds": rec.get("elapsed_seconds")})
    finally:
        s.close()
    return rows


def analyse(pids):
    report = {"driver": DRIVER_VERSION, "what": "run-to-run agreement of the units labeller (fixed units, "
              "classification by unit id, one prefix-only call per step), temperature 1, study settings",
              "packets": {}}
    all_calls = []
    for pid in pids:
        docs = load_docs(pid)
        if not docs:
            report["packets"][pid] = {"runs": 0}
            continue
        runs = sorted(docs)
        base = docs[runs[0]]
        same_units = all(docs[n]["units"] == base["units"] for n in runs)
        common = sorted(set.intersection(*[{int(k) for k, v in docs[n]["labels"].items() if v["status"] == "ok"}
                                           for n in runs]))
        steps = {int(k): v for k, v in base["units"]["steps"].items()}
        P = {"runs": runs, "units_identical_across_runs": same_units, "stall_point": base["stall_point"],
             "labelled_steps": {f"run{n}": len(docs[n]["labelled_steps"]) for n in runs},
             "steps_labelled_ok_in_every_run": len(common), "status": {f"run{n}": docs[n]["status"] for n in runs}}
        taggable = [u["id"] for k in common for u in steps[k] if u["kind"] in U.TAGGABLE]
        tags = {n: {} for n in runs}
        for n in runs:
            for k in common:
                tags[n].update(docs[n]["labels"][str(k)]["tags"])
        # ---- items per type
        P["items_per_type"] = {}
        for n in runs:
            lab = docs[n]["labels"]
            ks = [k for k in docs[n]["labelled_steps"]]
            tg = Counter(t for k in ks for ts in lab[str(k)]["tags"].values() for t in ts)
            tagged_units = sum(len(lab[str(k)]["tags"]) for k in ks)
            units_n = sum(1 for k in ks for u in steps[k] if u["kind"] in U.TAGGABLE)
            P["items_per_type"][f"run{n}"] = {
                **{t: tg[t] for t in U.TAGS}, "untagged_units": units_n - tagged_units, "taggable_units": units_n,
                "multi_tag_units": sum(1 for k in ks for ts in lab[str(k)]["tags"].values() if len(ts) > 1),
                "rests_on_links": sum(len(v) for k in ks for v in lab[str(k)]["rests_on"].values()),
                "verdicts": dict(Counter(v["verdict"] for k in ks for v in lab[str(k)]["claim_status"].values())),
                "claim_status_at_cut": dict(Counter(c["status"] for c in docs[n]["claims"].values())),
                "dropped": sum(len(lab[str(k)]["dropped"]) for k in ks),
                "actions": len(docs[n]["mechanical"]["actions"]), "hunks": len(docs[n]["mechanical"]["hunks"]),
                "submissions": len(docs[n]["mechanical"]["submissions"])}
        # ---- tags
        tagrep = {"units_compared": len(taggable)}
        for t in U.TAGS + ("none",):
            def has(n, u, t=t):
                ts = tags[n].get(u, [])
                return "yes" if (t in ts if t != "none" else not ts) else "no"
            pairs = {}
            for a, b in itertools.combinations(runs, 2):
                A = {u for u in taggable if has(a, u) == "yes"}
                B = {u for u in taggable if has(b, u) == "yes"}
                pairs[f"run{a}~run{b}"] = {"cohen_kappa": r3(cohen_kappa([(has(a, u), has(b, u)) for u in taggable])),
                                           "dice": dice(A, B), "run_a": len(A), "run_b": len(B), "both": len(A & B)}
            in_all = sum(all(has(n, u) == "yes" for n in runs) for u in taggable)
            in_any = sum(any(has(n, u) == "yes" for n in runs) for u in taggable)
            tagrep[t] = {"pairs": pairs, "fleiss_kappa": r3(fleiss([[has(n, u) for n in runs] for u in taggable]))
                         if len(runs) >= 2 else None, "units_tagged_in_every_run": in_all,
                         "units_tagged_in_any_run": in_any}
        tagrep["exact_tag_set_agreement"] = {
            f"run{a}~run{b}": r3(sum(sorted(tags[a].get(u, [])) == sorted(tags[b].get(u, [])) for u in taggable)
                                 / len(taggable)) if taggable else None for a, b in itertools.combinations(runs, 2)}
        P["tags"] = tagrep
        # ---- rests on
        cmds = [(k, u["id"]) for k in common for u in steps[k] if u["kind"] == "command"]
        task_n = len(base["units"]["task"])
        rests = {n: {c: set(docs[n]["labels"][str(k)]["rests_on"].get(c, [])) for k, c in cmds} for n in runs}
        universe = sum(task_n + sum(1 for j in range(1, k) for u in steps[j] if u["kind"] != "command")
                       + sum(1 for u in steps[k] if u["kind"] in U.TAGGABLE) for k, _ in cmds)
        rr = {"commands_compared": len(cmds), "candidate_pairs": universe, "pairs": {}}
        for a, b in itertools.combinations(runs, 2):
            A = {(c, x) for c in rests[a] for x in rests[a][c]}
            B = {(c, x) for c in rests[b] for x in rests[b][c]}
            both = len(A & B)
            rr["pairs"][f"run{a}~run{b}"] = {
                "links_a": len(A), "links_b": len(B), "both": both, "dice": dice(A, B),
                "cohen_kappa": r3(binary_kappa_counts(both, len(A) - both, len(B) - both,
                                                      universe - len(A | B))),
                "same_set_share": r3(sum(rests[a][c] == rests[b][c] for _, c in cmds) / len(cmds)) if cmds else None,
                "mean_jaccard": r3(statistics.mean([len(rests[a][c] & rests[b][c]) / len(rests[a][c] | rests[b][c])
                                                    if rests[a][c] | rests[b][c] else 1.0 for _, c in cmds]))
                if cmds else None}
        rr["fleiss_kappa"] = r3(fleiss_binary_sets([{(c, x) for c in rests[n] for x in rests[n][c]} for n in runs],
                                                   universe)) if len(runs) >= 2 else None
        P["rests_on"] = rr
        # ---- claim status
        claim_sets = {n: {u for u, ts in tags[n].items() if "claim" in ts} for n in runs}
        common_claims = sorted(set.intersection(*claim_sets.values()), key=lambda u: (int(u[1:].split(".")[0]), u))
        cells = []
        for c in common_claims:
            k0 = int(c[1:].split(".")[0])
            for j in common:
                if j >= k0 and any(u["kind"] in U.EVIDENCE for u in steps[j]):
                    cells.append((c, j))

        def verdict(n, c, j):
            v = docs[n]["labels"][str(j)]["claim_status"].get(c)
            return v["verdict"] if v else "nothing"

        def final(n, c):
            vs = [verdict(n, c, j) for cc, j in cells if cc == c]
            return "contradicted_later" if "contradicts" in vs else "confirmed_later" if "confirms" in vs \
                else "unchecked"
        cs = {"claims_tagged_in_every_run": len(common_claims), "cells_compared": len(cells), "pairs": {}}
        for a, b in itertools.combinations(runs, 2):
            cs["pairs"][f"run{a}~run{b}"] = {
                "cell_kappa": r3(cohen_kappa([(verdict(a, c, j), verdict(b, c, j)) for c, j in cells])),
                "cell_agreement": r3(sum(verdict(a, c, j) == verdict(b, c, j) for c, j in cells) / len(cells))
                if cells else None,
                "non_nothing_cells_a": sum(verdict(a, c, j) != "nothing" for c, j in cells),
                "non_nothing_cells_b": sum(verdict(b, c, j) != "nothing" for c, j in cells),
                "final_status_kappa": r3(cohen_kappa([(final(a, c), final(b, c)) for c in common_claims])),
                "final_status_agreement": r3(sum(final(a, c) == final(b, c) for c in common_claims)
                                             / len(common_claims)) if common_claims else None}
        cs["cell_fleiss_kappa"] = r3(fleiss([[verdict(n, c, j) for n in runs] for c, j in cells]))
        cs["final_status_fleiss_kappa"] = r3(fleiss([[final(n, c) for n in runs] for c in common_claims]))
        cs["final_status_counts"] = {f"run{n}": dict(Counter(final(n, c) for c in common_claims)) for n in runs}
        P["claim_status"] = cs
        # ---- cost
        costs = {}
        for n in runs:
            rows = call_costs(n, pid, docs[n])
            all_calls += rows
            pt = sum(r["prompt_tokens"] or 0 for r in rows)
            costs[f"run{n}"] = {
                "calls": len({r["step"] for r in rows}), "tries": len(rows),
                "failed_tries": sum(r["status"] != "ok" for r in rows),
                "cost_usd": r3(sum(r["cost_usd"] or 0 for r in rows) * 1000) / 1000 if rows else 0,
                "per_step_call_usd_mean": round(statistics.mean([sum(x["cost_usd"] or 0 for x in rows if x["step"] == s)
                                                                 for s in {r["step"] for r in rows}]), 5) if rows else None,
                "prompt_tokens": pt, "cached_tokens": sum(r["cached_tokens"] for r in rows),
                "cache_share": r3(sum(r["cached_tokens"] for r in rows) / pt) if pt else None,
                "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
                "reasoning_tokens": sum(r["reasoning_tokens"] or 0 for r in rows),
                "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
                "max_call_usd": max([r["cost_usd"] or 0 for r in rows] or [0]),
                "complete": docs[n]["status"] in ("ok", "incomplete")}
        P["cost"] = costs
        report["packets"][pid] = P
    # ---- the old find-items numbers, beside
    if OLD_REPORT.exists():
        old = json.loads(OLD_REPORT.read_text())
        report["old_find_items_temperature_1"] = {
            pid: {"pairs": {k: {"strict_dice": v["strict"]["dice"],
                                "strict_dice_by_type": {t: x["dice"] for t, x in v["strict"]["by_type"].items()},
                                "not_found_in_other_run": v["unmatched_items_both_sides"].get("not_found_in_other_run"),
                                "type_differs": v["unmatched_items_both_sides"].get("type_differs", 0)}
                            for k, v in p["pairs"].items()},
                  "across_all_runs": p["across_all_runs"], "counts": p["counts"]}
            for pid, p in old["packets"].items() if pid in pids}
        report["old_find_items_temperature_1"]["source"] = str(OLD_REPORT)
    # ---- spend
    stores = sorted(OUT.glob("run*/*/labeller"))
    sp = {str(d.relative_to(OUT)): recorded(d) for d in stores}
    rows = all_calls
    report["spend"] = {"per_store_usd": {k: round(v[0], 4) for k, v in sp.items()},
                       "recorded_usd": round(sum(v[0] for v in sp.values()), 4),
                       "attempts_possibly_billed_without_cost": sum(v[1] for v in sp.values()),
                       "worst_case_usd": round(sum(v[0] for v in sp.values()) + sum(v[1] for v in sp.values())
                                               * max([r["cost_usd"] or 0 for r in rows] or [0]), 4),
                       "per_step_call_usd": {"n": len(rows), "mean": round(statistics.mean([r["cost_usd"] or 0
                                                                                          for r in rows]), 5) if rows else None,
                                             "median": round(statistics.median([r["cost_usd"] or 0 for r in rows]), 5)
                                             if rows else None,
                                             "max": max([r["cost_usd"] or 0 for r in rows] or [0])},
                       "cache_share_all_calls": r3(sum(r["cached_tokens"] for r in rows)
                                                   / sum(r["prompt_tokens"] or 0 for r in rows)) if rows else None,
                       "reasoning_tokens_per_call": {"median": statistics.median([r["reasoning_tokens"] or 0 for r in rows])
                                                     if rows else None,
                                                     "max": max([r["reasoning_tokens"] or 0 for r in rows] or [0])}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main():
    global OUT
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    r.add_argument("--cap-usd", type=float, required=True)
    r.add_argument("--upto", type=int, default=None, help="label only steps 1..K (a smoke prefix)")
    r.add_argument("--parallel", action="store_true")
    r.add_argument("--labeller", choices=["deepseek", "replay", "scripted"], default="deepseek")
    r.add_argument("--settings", default=str(SETTINGS), help="model, pin and settings (default configs/v2/settings.json)")
    r.add_argument("packets", nargs="+")
    s = sub.add_parser("analyse")
    s.add_argument("packets", nargs="+")
    for x in (r, s):
        x.add_argument("--out", default=str(OUT), help="output folder (default LINKPILOT/units)")
    a = ap.parse_args()
    OUT = Path(a.out)
    if a.cmd == "run":
        run(a.runs, a.cap_usd, a.packets, a.upto, a.parallel, a.labeller, Path(a.settings))
    else:
        print(json.dumps(analyse(a.packets), indent=1)[:20000])


if __name__ == "__main__":
    main()
