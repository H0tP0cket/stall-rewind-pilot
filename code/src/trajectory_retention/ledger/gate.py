"""Human gate tooling for the Ledger: blind decks, answer files for two annotators, and scoring.

Export (follows ``observers/t2/labelcheck22`` and ``configs/detector-judge-20260926/hand_label_*``): the annotator
folder holds instructions, one plain log per stall (steps numbered, no labels, IDs or grades) and one JSONL file of
cards per deck; the key (predictions and where each card came from) goes to a separate file that must be outside the
annotator folder. Card order is a seeded shuffle per deck.

Decks:
  boundaries       one card per stall: where does each approach start, and where does the current one start
  tags             agent sentences and output lines, stratified by predicted type: decision / fact / claim / idea / none
  claim_status     predicted claims: contradicted_later / confirmed_later / unchecked
  idea_status      predicted ideas: tried_later / never_tried
  survives_rewind  predicted facts from the undone steps: still true after the rewind, or not
  approach_status  predicted approaches: failed / abandoned / worked / open
  fact_truth       about 50 handed-back facts (those an arm 6 recipe can carry): true at the rewind point?

Answers: ``answers-<annotator>.jsonl``, one line per card: ``{"deck", "card", "answer": {...}}``; "unsure" answers
are counted and left out of agreement. Scoring reports, per label type, Cohen's kappa of the Ledger against each
annotator and between the annotators; approach-boundary F1 within +/-2 steps (boundaries are the starts of approaches
after the first); the share of handed-back facts judged true. The contract's gate thresholds are reported next to
each number (kappa >= 0.6, boundary F1 >= 0.7, facts true >= 0.9); scoring decides nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .render import step_lines
from .text import cohen_kappa, prose_spans21

GATE_VERSION = "ledger-gate/v1"
SEED = 20261007
TOLERANCE = 2
THRESHOLDS = {"kappa": 0.6, "boundary_f1": 0.7, "facts_true": 0.9}
TYPES = ("decision", "fact", "claim", "idea", "none")
OPTIONS = {
    "tags": {"decision": "The agent adopts or commits to an approach, fix or design it will now pursue.",
             "fact": "Something observed or established about the task, code, files or environment.",
             "claim": "The agent asserts its own work succeeded or is correct.",
             "idea": "An option the agent mentions without acting on it at that point.",
             "none": "None of these.", "unsure": "Cannot tell."},
    "claim_status": {"contradicted_later": "A later output or later words of the agent show it was not true.",
                     "confirmed_later": "A later output shows it was true.",
                     "unchecked": "Nothing later in the log settles it.", "unsure": "Cannot tell."},
    "idea_status": {"tried_later": "A later step acts on it.", "never_tried": "No later step acts on it.",
                    "unsure": "Cannot tell."},
    "survives_rewind": {"true": "Still true after files and runtime go back to the rewind point.",
                        "false": "Not true after the rewind.", "unsure": "Cannot tell."},
    "approach_status": {"failed": "Outputs or the agent's words show it did not work.",
                        "abandoned": "Dropped without such evidence.", "worked": "Achieved what it was for.",
                        "open": "Still being pursued at the end with no clear result.", "unsure": "Cannot tell."},
    "fact_truth": {"yes": "True at the rewind point.", "no": "Not true at the rewind point.", "unsure": "Cannot tell."},
}
QUESTIONS = {
    "boundaries": ("An approach is one line of attack on the task, pursued over consecutive steps. Read the log and give "
                   "the first step of every approach (always including 1) and the first step of the current approach "
                   "(the one pursued at the end). Answer {\"approach_starts\": [1, ...], \"current_start\": n}."),
    "tags": "Which label fits the quoted text, read in its step? Answer {\"type\": \"<option>\"}.",
    "claim_status": "Judge only from the log after the quoted step. Answer {\"status\": \"<option>\"}.",
    "idea_status": "Judge only from the log after the quoted step. Answer {\"status\": \"<option>\"}.",
    "survives_rewind": ("Suppose files and runtime go back to how they were at the rewind point (before step "
                        "rewind_point + 1). Answer {\"status\": \"true\" | \"false\" | \"unsure\"}."),
    "approach_status": "Judge the steps given, from the log only. Answer {\"status\": \"<option>\"}.",
    "fact_truth": ("Is the quoted statement true at the rewind point (check the log, before step rewind_point + 1)? "
                   "Answer {\"true\": \"yes\" | \"no\" | \"unsure\"}."),
}


def plain_log(ledger: Mapping[str, Any]) -> str:
    lines = ["TASK"] + [f["text"] for f in ledger["head"] if f["kind"] != "system"] + [""]
    for s in ledger["steps"]:
        lines += [f"=== Step {s['k']} ==="] + step_lines(s, style="plain") + [""]
    return "\n".join(lines)


def _units(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Candidate units for the tags deck: agent sentences and non-empty output lines, with their predicted type."""
    spans = defaultdict(list)
    for i in ledger["items"]:
        spans[i["span"]["field_id"]].append(i)
    out = []
    for s in ledger["steps"]:
        for f in s["fields"]:
            if f["kind"] in ("reasoning", "message"):
                cands = prose_spans21(f["text"])[0]
            elif f["kind"] == "output":
                cands, pos = [], 0
                for line in f["text"].splitlines(keepends=True):
                    a, b = pos, pos + len(line.rstrip("\n"))
                    pos += len(line)
                    if 12 <= len(line.strip()) <= 300:
                        cands.append((a, b))
            else:
                continue
            for a, b in cands:
                hit = [i for i in spans.get(f["id"], ()) if i["span"]["start"] < b and a < i["span"]["end"]]
                text = f["text"]
                out.append({"step": s["k"], "field": f["id"], "kind": f["kind"], "start": a, "end": b,
                            "text": text[a:b], "before": text[max(0, a - 300):a], "after": text[b:b + 150],
                            "predicted": hit[0]["type"] if hit else "none", "item": hit[0]["id"] if hit else None})
    return out


SOURCE = {"reasoning": "agent reasoning", "message": "agent message", "output": "tool output"}


def export_decks(ledgers: Sequence[Mapping[str, Any]], dest: str | Path, key_path: str | Path, *,
                 per_type: int = 8, per_status: int = 12, facts: int = 50, seed: int = SEED) -> dict[str, int]:
    dest, key_path = Path(dest), Path(key_path)
    if dest.resolve() in key_path.resolve().parents:
        raise ValueError("the key file must not be inside the annotator folder")
    (dest / "logs").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    decks: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    stalls = {}
    for n, L in enumerate(ledgers, 1):
        sid = f"stall-{n:02d}"
        log = f"logs/{sid}.txt"
        (dest / log).write_text(plain_log(L))
        r = L["boundary"]["rewind_point"]
        stalls[sid] = {"run_id": L["run_id"], "checkpoint_id": L["checkpoint_id"], "steps": L["stall_point"],
                       "rewind_point": r, "fake_labels": bool(L.get("fake_labels"))}
        decks["boundaries"].append(({"log": log, "steps": L["stall_point"]},
                                    {"stall": sid, "starts": [a["first_step"] for a in L["approaches"]],
                                     "current_start": L["approaches"][-1]["first_step"],
                                     "direct_first_step": (L.get("direct_boundary") or {}).get("first_step")}))
        units = _units(L)
        for t in TYPES:
            pool = [u for u in units if u["predicted"] == t]
            for u in rng.sample(pool, min(per_type, len(pool))):
                decks["tags"].append(({"log": log, "step": u["step"], "source": SOURCE[u["kind"]],
                                       "context_before": u["before"], "text": u["text"], "context_after": u["after"]},
                                      {"stall": sid, "predicted": t, "item": u["item"], "field": u["field"],
                                       "span": [u["start"], u["end"]]}))
        items = L["items"]
        for deck, typ, key in (("claim_status", "claim", "claim_status"), ("idea_status", "idea", "idea_status")):
            pool = [i for i in items if i["type"] == typ and i.get(key)]
            for i in rng.sample(pool, min(per_status, len(pool))):
                decks[deck].append(({"log": log, "step": i["step"], "text": i["quote"]},
                                    {"stall": sid, "item": i["id"], "predicted": i[key]}))
        pool = [i for i in items if i["type"] == "fact" and i.get("survives_rewind") is not None]
        for i in rng.sample(pool, min(per_status, len(pool))):
            decks["survives_rewind"].append(({"log": log, "step": i["step"], "rewind_point": r, "text": i["quote"]},
                                             {"stall": sid, "item": i["id"],
                                              "predicted": "true" if i["survives_rewind"] else "false"}))
        for a in L["approaches"]:
            if a.get("status"):
                decks["approach_status"].append(({"log": log, "steps": [a["first_step"], a["last_step"]],
                                                  "is_current": a is L["approaches"][-1]},
                                                 {"stall": sid, "approach": a["id"], "predicted": a["status"]}))
    handed = [(f"stall-{n:02d}", L, i) for n, L in enumerate(ledgers, 1) for i in L["items"]
              if i["type"] == "fact" and i.get("survives_rewind") is True and i["step"] > L["boundary"]["rewind_point"]]
    for sid, L, i in rng.sample(handed, min(facts, len(handed))):
        decks["fact_truth"].append(({"log": f"logs/{sid}.txt", "step": i["step"],
                                     "rewind_point": L["boundary"]["rewind_point"], "text": i["quote"]},
                                    {"stall": sid, "item": i["id"], "predicted": "yes"}))
    key: dict[str, Any] = {"version": GATE_VERSION, "seed": seed, "stalls": stalls, "decks": {}}
    counts = {}
    for deck, rows in decks.items():
        rng.shuffle(rows)
        prefix = deck[0]
        with (dest / f"{deck}.jsonl").open("w") as fh:
            for m, (card, k) in enumerate(rows, 1):
                cid = f"{prefix}{m:03d}"
                fh.write(json.dumps({"deck": deck, "card": cid, **card}, ensure_ascii=False) + "\n")
                key["decks"].setdefault(deck, {})[cid] = k
        counts[deck] = len(rows)
    (dest / "instructions.json").write_text(json.dumps({
        "version": GATE_VERSION, "decks": {d: {"question": QUESTIONS[d], "options": OPTIONS.get(d)} for d in counts},
        "answers": "one JSON line per card in answers-<your name>.jsonl: {\"deck\": ..., \"card\": ..., \"answer\": {...}}",
        "blind": "Cards carry no model labels, IDs or grades. Do not open the key file until you have finished."},
        indent=1, ensure_ascii=False))
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=1))
    return counts


def _answer_value(deck: str, ans: Mapping[str, Any]) -> Any:
    if deck == "tags":
        return ans.get("type")
    if deck == "fact_truth":
        return ans.get("true")
    if deck == "boundaries":
        return ans
    return ans.get("status")


def _match(pred: Sequence[int], ref: Sequence[int], tol: int) -> int:
    """Greedy one-to-one matching of boundaries within +/- tol (closest pairs first)."""
    pairs = sorted((abs(p - r), p, r) for p in pred for r in ref if abs(p - r) <= tol)
    used_p, used_r, tp = set(), set(), 0
    for _, p, r in pairs:
        if p not in used_p and r not in used_r:
            used_p.add(p)
            used_r.add(r)
            tp += 1
    return tp


def boundary_f1(a: Mapping[str, list[int]], b: Mapping[str, list[int]], tol: int = TOLERANCE) -> dict[str, Any]:
    tp = fp = fn = 0
    for stall in set(a) & set(b):
        pa = sorted({x for x in a[stall] if x > 1})
        pb = sorted({x for x in b[stall] if x > 1})
        m = _match(pa, pb, tol)
        tp, fp, fn = tp + m, fp + len(pa) - m, fn + len(pb) - m
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = (2 * p * r / (p + r) if p and r else 0.0) if p is not None and r is not None else (
        1.0 if tp + fp + fn == 0 and set(a) & set(b) else None)
    return {"stalls": len(set(a) & set(b)), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def load_answers(path: str | Path) -> dict[tuple[str, str], Any]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[(row["deck"], row["card"])] = row["answer"]
    return out


def _kappas(deck: str, cards: Mapping[str, Any], raters: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per pair of raters: kappa over cards both answered (unsure left out); for tags also per type (binary)."""
    out: dict[str, Any] = {}
    names = list(raters)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = raters[names[x]], raters[names[y]]
            pairs = [(a[c], b[c]) for c in cards if a.get(c) not in (None, "unsure") and b.get(c) not in (None, "unsure")]
            row = {"n": len(pairs), "kappa": cohen_kappa(pairs),
                   "agreement": sum(p == q for p, q in pairs) / len(pairs) if pairs else None}
            if deck == "tags":
                row["per_type"] = {t: cohen_kappa([(p == t, q == t) for p, q in pairs]) for t in TYPES if t != "none"}
            out[f"{names[x]}~{names[y]}"] = row
    return out


def score(key_path: str | Path, answer_paths: Sequence[str | Path]) -> dict[str, Any]:
    key = json.loads(Path(key_path).read_text())
    annotators = {Path(p).stem.removeprefix("answers-"): load_answers(p) for p in answer_paths}
    report: dict[str, Any] = {"version": GATE_VERSION, "annotators": sorted(annotators), "thresholds": THRESHOLDS,
                              "decks": {}}
    for deck, cards in key["decks"].items():
        raters = {"ledger": {c: (k["starts"] if deck == "boundaries" else k["predicted"]) for c, k in cards.items()}}
        unsure = {}
        for name, ans in annotators.items():
            got = {c: _answer_value(deck, ans[(deck, c)]) for c in cards if (deck, c) in ans}
            unsure[name] = sum(1 for v in got.values() if v in ("unsure", None))
            raters[name] = got
        if deck == "boundaries":
            by_stall = {r: {cards[c]["stall"]: (v["approach_starts"] if r != "ledger" else v)
                            for c, v in raters[r].items() if v} for r in raters}
            direct = {cards[c]["stall"]: [1, cards[c]["direct_first_step"]] for c in cards
                      if cards[c].get("direct_first_step")}
            cur = {}
            for name in annotators:
                ok = [abs(raters[name][c]["current_start"] - cards[c]["current_start"]) <= TOLERANCE
                      for c in cards if raters[name].get(c)]
                cur[name] = sum(ok) / len(ok) if ok else None
            names = list(by_stall)
            report["decks"][deck] = {
                "cards": len(cards),
                "f1_within_2": {f"{names[x]}~{names[y]}": boundary_f1(by_stall[names[x]], by_stall[names[y]])
                                for x in range(len(names)) for y in range(x + 1, len(names))},
                "direct_pass_vs_annotators": {n: boundary_f1(direct, by_stall[n]) for n in annotators},
                "current_start_within_2": cur}
            continue
        if deck == "fact_truth":
            shares = {}
            for name in annotators:
                vals = [v for v in raters[name].values() if v in ("yes", "no")]
                shares[name] = {"judged": len(vals), "true": sum(v == "yes" for v in vals),
                                "share_true": sum(v == "yes" for v in vals) / len(vals) if vals else None,
                                "unsure": unsure[name]}
            allv = [v for n in annotators for v in raters[n].values() if v in ("yes", "no")]
            report["decks"][deck] = {"cards": len(cards), "per_annotator": shares,
                                     "pooled_share_true": sum(v == "yes" for v in allv) / len(allv) if allv else None}
            continue
        report["decks"][deck] = {"cards": len(cards), "unsure": unsure, "agreement": _kappas(deck, cards, raters)}
    return report


def self_answers(key_path: str | Path, out_path: str | Path, *, perturb: float = 0.0, seed: int = SEED) -> int:
    """Pipeline test only: answers copied from the key (the Ledger rating itself), optionally with a deterministic
    share of answers changed. Agreement against these is not validation."""
    key = json.loads(Path(key_path).read_text())
    rng = random.Random(seed)
    rows = []
    for deck, cards in key["decks"].items():
        opts = [o for o in OPTIONS.get(deck, {}) if o != "unsure"]
        for cid, k in sorted(cards.items()):
            if deck == "boundaries":
                starts = list(k["starts"])
                if perturb and rng.random() < perturb and len(starts) > 1:
                    starts[-1] = starts[-1] + rng.choice([-3, 1])
                ans = {"approach_starts": starts, "current_start": starts[-1]}
            else:
                v = k["predicted"]
                if perturb and rng.random() < perturb:
                    v = rng.choice([o for o in opts if o != v])
                ans = {"type": v} if deck == "tags" else {"true": v} if deck == "fact_truth" else {"status": v}
            rows.append({"deck": deck, "card": cid, "answer": ans})
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ledger gate decks and scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--ledger", action="append", required=True, help="a ledger.json (repeat per stall)")
    ex.add_argument("--dest", required=True)
    ex.add_argument("--key", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--key", required=True)
    sc.add_argument("--answers", action="append", required=True, help="answers-<annotator>.jsonl (one or two)")
    sc.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export_decks([json.loads(Path(p).read_text()) for p in a.ledger], a.dest, a.key)))
    else:
        rep = score(a.key, a.answers)
        text = json.dumps(rep, indent=1)
        if a.out:
            Path(a.out).write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
