"""Run-to-run agreement of the Ledger labeller's fixed items (protocol v2.1, study settings, temperature 1).

  .venv/bin/python configs/v2/linkpilot/agreement_r2b.py run --runs 2 3 --cap-usd 1.00 p1 p6
  .venv/bin/python configs/v2/linkpilot/agreement_r2b.py analyse p1 p6

Run 1 is the existing r2b/ledger-v21 labelling. Each further run gets its own receipt store
(r2b/agreement/run<N>/labeller), so the same requests are sent again as new paid calls instead of being replayed.
Runs and packets go one call at a time. The budget guard of each run is given the cap minus everything already
recorded in the other runs' stores, and holds back per call the cost of a full-length answer (max_tokens) at the
provider's own prices, fitted exactly from the r2b receipts ($0.30 per million input tokens, $1.20 per million output
tokens), so the cap holds even if a call returns the longest possible answer.

Matching (per packet, every pair of runs, one-to-one, greedy in context order):
  strict  same type, same step, overlapping quote spans in the same part of the step;
  loose   same type, same step;
  span    same step, overlapping quote spans, any type (used to split drift: an item without a strict match that
          has a span match of another type is a TYPE difference; one with no span match at all is a FOUND difference).
Writes r2b/agreement/report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ledger_items_r2b as LI  # noqa: E402  (puts the repo src on sys.path)

from trajectory_retention.ledger.__main__ import make_client  # noqa: E402
from trajectory_retention.observers.t2.jev import _ref  # noqa: E402
from trajectory_retention.observers.t2.store_ro import ReadOnlyStore  # noqa: E402

R2B = LI.LINKPILOT / "r2b"
AGREE = R2B / "agreement"
PROVIDER_USD_PER_TOKEN = (0.30 / 1e6, 1.20 / 1e6)  # exact fit to 40 Parasail fp8 receipts (r2b/ledger, ledger-v21)
TYPES = ("decision", "claim", "fact", "idea")


def run_dirs():
    return sorted((d for d in AGREE.glob("run[0-9]*") if d.is_dir()), key=lambda d: int(d.name[3:]))


def recorded(store_dir: Path) -> tuple[float, int]:
    if not store_dir.exists():
        return 0.0, 0
    s = ReadOnlyStore(store_dir)
    try:
        recs = [s.get_json(_ref(r["ref"])) for k in ("glm_call", "glm_call_failed") for r in s.list_records(k)]
    finally:
        s.close()
    lost = sum(1 for r in recs for a in r.get("attempts") or [] if a.get("http_status") == 0
               or (a.get("http_status") or 0) >= 500) + sum(1 for r in recs if r.get("cost_usd") is None)
    return sum(r.get("cost_usd") or 0.0 for r in recs), lost


def study_client(store_dir, budget, temperature):
    """The study labeller (pin, reasoning effort, max tokens from configs/v2/settings.json); ``temperature`` replaces
    the settings' temperature in the request (and in the recorded settings) when given."""
    client = make_client("deepseek", store_dir, budget_usd=budget)
    client.usd_in, client.usd_out = PROVIDER_USD_PER_TOKEN
    if temperature is not None:
        client.extra = {**client.extra, "temperature": temperature}
        client.settings = {**client.settings, "temperature": temperature}
    return client


def spent_everywhere():
    """Everything recorded in this folder's receipt stores (runs and probe)."""
    return sum(recorded(d)[0] for d in AGREE.glob("*/labeller") if d.is_dir())


def packet_cost_seen(pid):
    """The highest recorded cost of this packet in any earlier labelling (r2b v2.1 and every agreement run), and the
    number of calls it takes."""
    paths = [R2B / "ledger-v21" / pid / "items.json"] + list(R2B.glob(f"agreement*/run*/{pid}/items.json"))
    docs = [json.loads(p.read_text())["labeller"] for p in paths if p.exists()]
    return max([d["cost_usd"] for d in docs] or [0.0]), max([len(d["passes"]) for d in docs] or [1])


def run(runs, cap, pids, temperature):
    manifest = json.load(open(LI.LINKPILOT / "manifest.json"))["packets"]
    hold = 300_000 / 3.2 * PROVIDER_USD_PER_TOKEN[0] + 262_144 * PROVIDER_USD_PER_TOKEN[1]  # one full-length call
    for n in runs:
        out = AGREE / f"run{n}"
        others = spent_everywhere() - recorded(out / "labeller")[0]
        client = study_client(out / "labeller", cap - others, temperature)
        print(f"run{n}: temperature {client.extra.get('temperature')}; budget ${cap - others:.4f} (cap ${cap:.2f} "
              f"minus ${others:.4f} elsewhere)", flush=True)
        for pid in pids:
            # skip rather than start a packet the cap may not let finish: its calls before the last at the highest
            # cost seen, then the guard's full-length hold-back for the last call (the guard still checks every call)
            cost, ncalls = packet_cost_seen(pid)
            need = cost * (ncalls - 1) / ncalls + hold
            if spent_everywhere() + need > cap:
                print(f"run{n} {pid} SKIPPED: spent ${spent_everywhere():.4f} + ${need:.4f} needed > cap ${cap:.2f}",
                      flush=True)
                continue
            try:
                doc, tries = LI.run_packet(pid, manifest[pid], client, out, "v2.1")
                print(f"run{n} {pid}", doc["status"], dict(Counter(i["type"] for i in doc["items"])),
                      "dropped", len(doc["dropped"]), [(t["prompt"], t["chunk"], t["status"], t["reply"]) for t in tries],
                      "| run spent", round(client.spent, 4), flush=True)
            except Exception as exc:
                print(f"run{n} {pid} ERROR {type(exc).__name__}: {exc}", flush=True)
    print(json.dumps({f"run{n}": recorded(AGREE / f"run{n}" / "labeller") for n in runs}))


def load(pid):
    # the temperature-1 folder's run 1 is the r2b v2.1 labelling; other folders hold all their runs
    runs = {1: R2B / "ledger-v21" / pid / "items.json"} if AGREE.name == "agreement" else {}
    for d in run_dirs():
        p = d / pid / "items.json"
        if p.exists():
            runs[int(d.name[3:])] = p
    out = {}
    for n, p in sorted(runs.items()):
        doc = json.loads(p.read_text())
        out[n] = {"status": doc["status"], "items": doc["items"], "dropped": len(doc["dropped"])}
    return out


def overlap(a, b):
    sa, sb = a["span"], b["span"]
    return sa["field_id"] == sb["field_id"] and sa["start"] < sb["end"] and sb["start"] < sa["end"]


RULES = {
    "strict": lambda a, b: a["type"] == b["type"] and a["step"] == b["step"] and overlap(a, b),
    "loose": lambda a, b: a["type"] == b["type"] and a["step"] == b["step"],
    "span": lambda a, b: a["step"] == b["step"] and overlap(a, b),
}


def match(A, B, rule):
    used, pairs = set(), []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if j not in used and RULES[rule](a, b):
                used.add(j)
                pairs.append((i, j))
                break
    return pairs


def dice(m, na, nb):
    return round(2 * m / (na + nb), 3) if na + nb else None


def pair_report(A, B):
    rep = {}
    for rule in ("strict", "loose"):
        pairs = match(A, B, rule)
        by = {}
        for t in TYPES:
            na, nb = sum(a["type"] == t for a in A), sum(b["type"] == t for b in B)
            m = sum(A[i]["type"] == t for i, _ in pairs)
            by[t] = {"run_a": na, "run_b": nb, "matched": m, "dice": dice(m, na, nb)}
        rep[rule] = {"matched": len(pairs), "dice": dice(len(pairs), len(A), len(B)), "by_type": by}
    # drift split for items without a strict match, from both sides
    strict = match(A, B, "strict")
    sa, sb = {i for i, _ in strict}, {j for _, j in strict}
    drift = Counter()
    type_moves = Counter()
    for X, Y, seen in ((A, B, sa), (B, A, sb)):
        for i, x in enumerate(X):
            if i in seen:
                continue
            other = next((y for y in Y if RULES["span"](x, y) and y["type"] != x["type"]), None)
            if other is not None:
                drift["type_differs"] += 1
                type_moves[f"{x['type']}<->{other['type']}" if x["type"] < other["type"]
                           else f"{other['type']}<->{x['type']}"] += 1
            elif any(RULES["span"](x, y) for y in Y):
                drift["same_type_span_taken"] += 1
            else:
                drift["not_found_in_other_run"] += 1
    rep["unmatched_items_both_sides"] = dict(drift)
    rep["type_differences"] = dict(type_moves)
    return rep


def all_runs(runs):
    """Items of run 1 found (strict) in every other run, per type, and the union size (run 1 items plus items of
    later runs matched to nothing earlier)."""
    ns = sorted(runs)
    base = runs[ns[0]]["items"]
    in_all = Counter()
    for i, a in enumerate(base):
        if all(any(i == x for x, _ in match(base, runs[n]["items"], "strict")) for n in ns[1:]):
            in_all[a["type"]] += 1
    pool = list(base)
    for n in ns[1:]:
        items = runs[n]["items"]
        hit = {j for _, j in match(pool, items, "strict")}
        pool += [b for j, b in enumerate(items) if j not in hit]
    return {"run1_items_found_in_every_run": {t: in_all[t] for t in TYPES},
            "union_items": {t: sum(p["type"] == t for p in pool) for t in TYPES}}


def analyse(pids):
    report = {"rules": {k: v for k, v in {
        "strict": "same type, same step, overlapping quote spans in the same part of the step",
        "loose": "same type, same step",
        "dice": "2 x matched / (items in run a + items in run b)"}.items()}, "packets": {}}
    for pid in pids:
        runs = load(pid)
        counts = {f"run{n}": {**{t: sum(i["type"] == t for i in r["items"]) for t in TYPES},
                              "total": len(r["items"]), "dropped": r["dropped"], "status": r["status"]}
                  for n, r in runs.items()}
        pairs = {f"run{a}~run{b}": pair_report(runs[a]["items"], runs[b]["items"])
                 for a, b in itertools.combinations(sorted(runs), 2)}
        report["packets"][pid] = {"counts": counts, "pairs": pairs,
                                  "across_all_runs": all_runs(runs) if runs else None}
    stores = sorted(d.parent for d in AGREE.glob("*/labeller") if d.is_dir())
    sp = {d.name: recorded(d / "labeller") for d in stores}
    rec = sum(v[0] for v in sp.values())
    top = 0.0
    for d in stores:
        s = ReadOnlyStore(d / "labeller")
        try:
            top = max([top] + [s.get_json(_ref(r["ref"])).get("cost_usd") or 0.0 for r in s.list_records("glm_call")])
        finally:
            s.close()
    lost = sum(v[1] for v in sp.values())
    if (AGREE / "lost-calls.json").exists():  # calls cut before a receipt was written (may have been billed)
        lost += len(json.loads((AGREE / "lost-calls.json").read_text()))
    report["spend"] = {"per_run": {k: round(v[0], 4) for k, v in sp.items()}, "recorded_usd": round(rec, 4),
                       "lost_or_failed_attempts": lost, "worst_case_usd": round(rec + lost * top, 4)}
    report["determinism"] = determinism(pids)
    if (AGREE / "probe.json").exists():
        report["probe"] = json.loads((AGREE / "probe.json").read_text())
    AGREE.mkdir(parents=True, exist_ok=True)
    (AGREE / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def receipt(store_dir, record_id):
    s = ReadOnlyStore(store_dir)
    try:
        rec = s.get_record("glm_call", record_id)
        body = s.get_json(_ref(rec["response_ref"]))
    finally:
        s.close()
    msg = ((body.get("choices") or [{}])[0].get("message") or {})
    text = msg.get("content") or ""
    return {"completion_tokens": rec.get("completion_tokens"), "reasoning_tokens": rec.get("reasoning_tokens"),
            "answer_sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "answer_chars": len(text),
            "reasoning_chars": len(msg.get("reasoning") or "")}


def determinism(pids):
    """The same request across runs: are the answers byte-identical, and are the token counts equal?"""
    out = {}
    for pid in pids:
        per = {}
        for d in run_dirs():
            f = d / pid / "calls.json"
            if not f.exists():
                continue
            for c in json.loads(f.read_text()):
                if c.get("status") == "ok" and c.get("record_id"):
                    per.setdefault(f"{c['prompt']} {c['chunk']}", {})[d.name] = receipt(d / "labeller", c["record_id"])
        out[pid] = {k: {"runs": v, "answers_identical": len({x["answer_sha256"] for x in v.values()}) == 1,
                        "completion_tokens_identical": len({x["completion_tokens"] for x in v.values()}) == 1}
                    for k, v in per.items()}
    return out


PROBE_PROMPT = "Reply with one English word chosen at random. Reply with the word only."


def probe(repeats, cap):
    """The same tiny request ``repeats`` times at temperature 0 and at the settings' temperature (1): does
    temperature 0 make the answer fixed?"""
    rows = {}
    for temp in (0, None):
        client = study_client(AGREE / "probe" / "labeller", cap - (spent_everywhere()
                                                                   - recorded(AGREE / "probe" / "labeller")[0]), temp)
        t = client.extra.get("temperature")
        rows[str(t)] = []
        for r in range(repeats):
            ans = client.complete([{"role": "user", "content": PROBE_PROMPT}], purpose=f"agreement-probe-t{t}",
                                  repeat=r)
            rec = receipt(AGREE / "probe" / "labeller", ans.call_record_id)
            rows[str(t)].append({"answer": ans.text.strip()[:40], **rec})
    res = {"prompt": PROBE_PROMPT, "repeats": repeats, "by_temperature": rows,
           "distinct_answers": {t: len({x["answer"] for x in v}) for t, v in rows.items()},
           "distinct_reasoning_lengths": {t: len({x["reasoning_tokens"] for x in v}) for t, v in rows.items()}}
    (AGREE / "probe.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=int, nargs="+", default=[2, 3])
    r.add_argument("--cap-usd", type=float, default=1.00)
    r.add_argument("--temperature", type=float, default=None, help="default: the settings' temperature")
    r.add_argument("packets", nargs="+")
    s = sub.add_parser("analyse")
    s.add_argument("packets", nargs="+")
    q = sub.add_parser("probe")
    q.add_argument("--repeats", type=int, default=5)
    q.add_argument("--cap-usd", type=float, default=1.00)
    for x in (r, s, q):
        x.add_argument("--dir", default="agreement", help="folder under r2b/ (agreement = temperature 1)")
    a = ap.parse_args()
    global AGREE
    AGREE = R2B / a.dir
    if a.cmd == "run":
        run(a.runs, a.cap_usd, a.packets, a.temperature)
    elif a.cmd == "probe":
        probe(a.repeats, a.cap_usd)
    else:
        rep = analyse(a.packets)
        print(json.dumps(rep, indent=1)[:12000])


if __name__ == "__main__":
    main()
