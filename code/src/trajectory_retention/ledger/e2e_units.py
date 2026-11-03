"""End-to-end check of the units Ledger (fixed units + classification, step by step), with no paid call.

  PYTHONPATH=src .venv/bin/python -m trajectory_retention.ledger.e2e_units [--out artifacts/v2/ledger-units]
      [--source-root DIR] [--natural-store DIR]

Failure modes looked for, written before the code:
  1. segmentation is not deterministic: the same record gives other unit ids, spans or texts on a second load, or the
     units of step k change when later steps are added (a later cut);
  2. a unit's text is not its span, a non-space character of a shown field lies in no unit or in two, ids repeat;
  3. labels are not prefix-only: the request for step k holds a later step, the labels of step k differ between a build
     at an earlier cut and one at the stall, or resuming from frozen labels revises them;
  4. an answer that is not id-only is accepted: an unknown id, a quote in place of an id, an id of a later step, a
     wrong set of command keys, prose around the JSON; or a valid answer is rejected; or a known id in a place it is
     not allowed is kept instead of dropped and logged; or a call gets more than two tries;
  5. mechanical items are wrong: a command without its action item, a hunk that carries out a command of another
     step or a missing command, hunks that differ from the round-2 packet builder's independent hunk index (p1), a
     natural-v2 run's hunks without attribution;
  6. the flat view shows an id that its unit map lacks (or the reverse), content after the cut or private evaluator
     data, or a labeller request leaks either;
  7. a rebuild from receipts makes new calls or gives a different document.
On the three real stalls of ``ledger.e2e`` (and one natural-v2 run when its store is on disk) it runs ``build_units``
with the scripted labeller (FAKE labels). Writes OUT/report.json; exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..observers.t2.glm import GlmAnswer
from ..observers.t2.store_ro import ReadOnlyStore
from ..store import ArtifactStore
from . import checks as C
from . import units as U
from .e2e import STALLS, _source_root
from .steps import load_context

REPO = Path(__file__).resolve().parents[3]
COMMAND = "PYTHONPATH=src .venv/bin/python -m trajectory_retention.ledger.e2e_units"
LINKPILOT_MANIFEST = "calibration/linkpilot/manifest.json"
NATURAL_STORE = Path("~/.cache/trajectory-retention/v2-screen/tbv2-20261007/screen-stage1-1-r4/store").expanduser()


class _Recording:
    def __init__(self, inner):
        self.inner, self.requests = inner, []

    def bind_context(self, ctx):
        self.inner.bind_context(ctx)

    def describe(self):
        return self.inner.describe()

    def complete(self, messages, **kw):
        self.requests.append((kw.get("repeat"), [dict(m) for m in messages]))
        return self.inner.complete(messages, **kw)


class _Canned:
    def __init__(self, text: str):
        self.text, self.tries = text, 0

    def complete(self, messages, *, purpose, repeat=0, max_tokens=None):
        self.tries += 1
        return GlmAnswer(self.text, None, "canned", "canned", "stop", {}, 0.0, False, f"canned:r{repeat}")


def _strip(x):
    """Labels without receipt bookkeeping (the scripted labeller's receipt id also hashes the whole context, so it
    differs between cuts; the request itself is compared through request_sha256)."""
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in ("cached", "record_id")}
    return [_strip(v) for v in x] if isinstance(x, list) else x


def _earlier_checkpoint(src, run_id: str, n: int) -> tuple[str, int] | None:
    best = None
    for cp in sorted(src.list_checkpoints(run_id), key=lambda c: c.ordinal):
        if cp.state.history is None:
            continue
        m = len(load_context(src, run_id, cp.checkpoint_id).steps)
        if 3 <= m <= n - 2 and (best is None or abs(m - n // 2) < abs(best[1] - n // 2)):
            best = (cp.checkpoint_id, m)
    return best


def segmentation_checks(src, run_id, checkpoint_id, early) -> dict[str, Any]:
    a = load_context(src, run_id, checkpoint_id)
    b = load_context(src, run_id, checkpoint_id)
    sa, sb = U.segment_context(a), U.segment_context(b)
    same = json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)
    cov = U.segmentation_coverage(a, sa)
    prefix = {"pass": True, "skipped": "no earlier checkpoint"}
    if early:
        e = U.segment_context(load_context(src, run_id, early[0]))
        diff = [k for k in e["steps"] if json.dumps(e["steps"][k], sort_keys=True)
                != json.dumps(sa["steps"][k], sort_keys=True)]
        prefix = {"pass": not diff and e["task"] == sa["task"], "earlier_cut_steps": early[1],
                  "steps_differing": diff}
    return {"same_record_same_units": {"pass": same}, "coverage": cov, "earlier_cut_same_units": prefix}


def canned_cases(seg, labels, k, order) -> dict[str, Any]:
    """Answers for a real step: which are accepted, with how many tries, and what is dropped."""
    units = seg["steps"][k]
    r = [u["id"] for u in units if u["kind"] in U.TAGGABLE]
    cmds = [u["id"] for u in units if u["kind"] == "command"]
    outs = [u["id"] for u in units if u["kind"] in U.EVIDENCE]
    later = f"S{k + 1}.r1"
    earlier_out = next(u["id"] for j in range(k - 1, 0, -1) for u in seg["steps"][j] if u["kind"] == "output")
    base = {"tags": {"claim": [r[0]], "plan": [r[1]], "idea": [], "observation": [r[0]]},
            "rests_on": {c: [earlier_out, "T1"] for c in cmds},
            "claim_status": {r[0]: {"verdict": "contradicts", "evidence": outs[0]}}}

    def mod(f):
        x = json.loads(json.dumps(base))
        f(x)
        return x
    quote = units[0]["text"][:40]
    cases = [
        ("valid", json.dumps(base), True, 0),
        ("valid_in_json_fence", "```json\n" + json.dumps(base) + "\n```", True, 0),
        ("unknown_id_in_tags", json.dumps(mod(lambda x: x["tags"]["claim"].append(f"S{k}.r9999"))), False, 0),
        ("quote_instead_of_id", json.dumps(mod(lambda x: x["tags"]["idea"].append(quote))), False, 0),
        ("later_step_id", json.dumps(mod(lambda x: x["rests_on"][cmds[0]].append(later))), False, 0),
        ("unknown_evidence", json.dumps(mod(lambda x: x["claim_status"][r[0]].update(evidence="S0.o1"))), False, 0),
        ("missing_command_key", json.dumps(mod(lambda x: x["rests_on"].pop(cmds[0]))), False, 0),
        ("extra_top_key", json.dumps(mod(lambda x: x.update(notes="x"))), False, 0),
        ("unknown_tag", json.dumps(mod(lambda x: x["tags"].update(fact=[r[0]]))), False, 0),
        ("prose_around_json", "Here are the labels:\n" + json.dumps(base), False, 0),
        ("agent_continuation", f"[S{k + 1}.c1] command 1:\nls -la", False, 0),
        ("known_but_not_allowed_target", json.dumps(mod(lambda x: x["rests_on"][cmds[0]].append(r[1]))), True, 1),
        ("evidence_not_this_step", json.dumps(mod(lambda x: x["claim_status"][r[0]].update(evidence=earlier_out))),
         True, 1),
        ("status_of_non_claim", json.dumps(mod(lambda x: x["claim_status"].update(
            {r[1]: {"verdict": "confirms", "evidence": outs[0]}}))), True, 1),
    ]
    rows, bad = [], []
    for name, text, accept, drops in cases:
        client = _Canned(text)
        lab = U.label_step(client, seg, labels, k, order, U.claims_before(seg, labels, k))
        ok = (lab["status"] == "ok") == accept and client.tries == (1 if accept else 2) and \
            (len(lab["dropped"]) == drops if accept else lab["tags"] == {})
        rows.append({"case": name, "pass": ok, "status": lab["status"], "tries": client.tries,
                     "reasons": [t.get("reason") or t["status"] for t in lab["tries"]], "dropped": lab["dropped"]})
        if not ok:
            bad.append(name)
    return {"pass": not bad, "step": k, "failures": bad, "cases": rows}


def prefix_checks(requests, doc) -> dict[str, Any]:
    """Request for step k: holds no later step, no id of a later step, and its log ends at step k."""
    bad = []
    for _, msgs in requests:
        user = msgs[-1]["content"]
        if not user.startswith(U.UNITS_LOG):
            bad.append({"step": None, "problem": "request does not start with the fixed preamble"})
            continue
        user = user[len(U.UNITS_LOG) - len("<recorded_log>\n"):]  # the preamble's own id examples are not the log
        k = int(re.search(r"Label step (\d+), the", user).group(1))
        log = user[user.index("<recorded_log>\n"):user.rindex("</recorded_log>")]
        later_ids = [x for x in re.findall(r"\bS(\d+)\.[rmcofeu]\d+\b", user) if int(x) > k]
        steps = [int(x) for x in re.findall(r"^<step (\d+)>$", log, re.M)]
        if later_ids or steps != list(range(1, k + 1)) or roles(msgs) != ["system", "user"]:
            bad.append({"step": k, "later_ids": later_ids[:3], "steps": steps[-3:]})
    return {"pass": not bad and bool(requests), "requests": len(requests), "failures": bad[:5]}


def roles(msgs):
    return [m["role"] for m in msgs]


def freezing_checks(src, run_id, checkpoint_id, early, sdir, full) -> dict[str, Any]:
    if not early:
        return {"pass": True, "skipped": "no earlier checkpoint"}
    m = early[1]
    a = U.build_units(src, run_id, early[0], U.scripted_units_labeller(ArtifactStore(sdir / "early" / "labeller")),
                      out_dir=sdir / "early")
    same_early = [k for k in range(1, m + 1) if _strip(a["labels"][str(k)]) != _strip(full["labels"][str(k)])]
    resumed = U.build_units(src, run_id, checkpoint_id,
                            U.scripted_units_labeller(ArtifactStore(sdir / "resumed" / "labeller")),
                            out_dir=sdir / "resumed", frozen=a)
    kept = [k for k in range(1, m + 1) if json.dumps(resumed["labels"][str(k)], sort_keys=True)
            != json.dumps(a["labels"][str(k)], sort_keys=True)]
    rest = [k for k in range(m + 1, full["stall_point"] + 1)
            if _strip(resumed["labels"][str(k)]) != _strip(full["labels"][str(k)])]
    # frozen labels whose step content differs are not reused: a tampered step stops the reuse there
    tampered = json.loads(json.dumps(a))
    tampered["labels"][str(2)]["step_sha256"] = "0" * 64
    t = U.build_units(src, run_id, early[0], U.scripted_units_labeller(ArtifactStore(sdir / "tampered" / "labeller")),
                      frozen=tampered)
    return {"pass": not same_early and not kept and not rest and resumed["reused_frozen_steps"] == list(range(1, m + 1))
            and t["reused_frozen_steps"] == [1],
            "earlier_cut_steps": m, "labels_differ_earlier_cut_vs_stall": same_early,
            "resumed_revised_frozen_steps": kept, "resumed_later_steps_differ": rest,
            "reused_frozen_steps": resumed["reused_frozen_steps"], "tampered_reuse": t["reused_frozen_steps"]}


def mechanical_checks(doc, manifest_entry) -> dict[str, Any]:
    seg_cmds = {u["id"]: int(k) for k, us in doc["units"]["steps"].items() for u in us if u["kind"] == "command"}
    actions = {a["unit"] for a in doc["mechanical"]["actions"]}
    bad = []
    if actions != set(seg_cmds):
        bad.append("commands and action items differ")
    for h in doc["mechanical"]["hunks"]:
        if h["carries_out"] is not None and seg_cmds.get(h["carries_out"]) != h["step"]:
            bad.append(f"{h['id']} carries out {h['carries_out']}")
        if h["carries_out"] is None and h["attribution"] != "step_only":
            bad.append(f"{h['id']} has no command but attribution {h['attribution']}")
    for s in doc["mechanical"]["submissions"]:
        if s["command"] is None or seg_cmds.get(s["command"]) != s["step"]:
            bad.append(f"submission {s['unit']} without its command")
    out = {"actions": len(actions), "hunks": len(doc["mechanical"]["hunks"]),
           "attribution": sorted({h["attribution"] for h in doc["mechanical"]["hunks"]}),
           "submissions": [{k: s[k] for k in ("unit", "failing_count", "command")} for s in doc["mechanical"]["submissions"]],
           "problems": bad[:10]}
    if manifest_entry is not None:
        k_max = manifest_entry["stall_point_step"]
        theirs = sorted((int(h[1:].split("h")[0]), int(h.split("h")[1]), v["path"], v["header"])
                        for h, v in manifest_entry["hunk_index"].items() if int(h[1:].split("h")[0]) <= k_max)
        ours = sorted((h["step"], int(h["id"].split(".h")[1]), h["path"], h["header"]) for h in doc["mechanical"]["hunks"])
        out["same_as_round2_builder"] = theirs == ours
        out["round2_builder_hunks"] = len(theirs)
        if theirs != ours:
            bad.append("hunks differ from the round-2 packet builder")
    out["pass"] = not bad
    return out


def view_checks(src, doc, view, requests) -> dict[str, Any]:
    ids = set(re.findall(r"⟦(T\d+|S\d+(?:\.[rmcofeuh]\d+)?)(?: [^⟧]*)?⟧", view["text"]))
    units = set(view["units"])
    sources = C.leak_sources(src, doc)
    # hunk lines are read from the worker's own workspace captures (never from grades); a file path there can match a
    # string in a grade payload without coming from it, so they are checked apart: only for content after the cut
    hunk_lines = [ln for ln in view["text"].splitlines() if re.match(r"⟦S\d+\.h\d+ ", ln)]
    body = "\n".join(ln for ln in view["text"].splitlines() if not re.match(r"⟦S\d+\.h\d+ ", ln))
    hunk_future = C.check_leaks({"hunk_lines": "\n".join(hunk_lines)}, {"future": sources["future"], "private": set()})
    texts = {"view_without_hunk_lines": body}
    texts.update({f"request{n}": msgs[-1]["content"] for n, (_, msgs) in enumerate(requests)})
    leaks = C.check_leaks(texts, sources)
    planted = C.check_leaks({"planted": view["text"] + "\n" + next(iter(sources["future"] or {"x"}))}, sources)
    hunk_private = sorted({s for s in sources["private"] if s in "\n".join(hunk_lines)})
    return {"pass": ids == units and leaks["pass"] and hunk_future["pass"] and (not sources["future"]
                                                                                or not planted["pass"]),
            "ids_in_text_not_in_map": sorted(ids - units)[:5], "map_not_in_text": sorted(units - ids)[:5],
            "leaks": {k: leaks[k] for k in ("pass", "future_strings", "private_strings", "hits")},
            "hunk_lines": len(hunk_lines), "hunk_lines_future_leak": hunk_future["hits"],
            "hunk_lines_matching_grade_strings (workspace paths, reported only)": hunk_private[:10],
            "planted_future_leak_caught": not planted["pass"] if sources["future"] else None}


def run_stall(src, name, run_id, cp, out, manifest_entry=None) -> dict[str, Any]:
    sdir = out / "stalls" / name
    checkpoint_id = cp if "/" in cp else f"{run_id}/{cp}"
    rec = _Recording(U.scripted_units_labeller(ArtifactStore(sdir / "labeller")))
    doc = U.build_units(src, run_id, checkpoint_id, rec, out_dir=sdir)
    view = U.render_units_view(doc)
    (sdir / "view.txt").write_text(view["text"])
    first = (sdir / "ledger-units.json").read_bytes()
    replay = U.scripted_units_labeller(ArtifactStore(sdir / "labeller"))
    again = U.build_units(src, run_id, checkpoint_id, replay)
    n = doc["stall_point"]
    early = _earlier_checkpoint(src, run_id, n)
    labels = {int(a): b for a, b in doc["labels"].items()}
    seg = U.segment_context(load_context(src, run_id, checkpoint_id))
    order = {u["id"]: i for i, u in enumerate(list(seg["task"]) + [u for k in sorted(seg["steps"])
                                                                    for u in seg["steps"][k]])}
    k_case = next(k for k in range(2, n + 1) if sum(u["kind"] in U.TAGGABLE for u in seg["steps"][k]) >= 2
                  and any(u["kind"] == "command" for u in seg["steps"][k])
                  and any(u["kind"] == "output" for u in seg["steps"][k]))
    planted = [lab for lab in labels.values() if lab["tries"] and lab["tries"][0].get("status") == "rejected"]
    checks = {
        "segmentation": segmentation_checks(src, run_id, checkpoint_id, early),
        "prefix_only_requests": prefix_checks(rec.requests, doc),
        "freezing": freezing_checks(src, run_id, checkpoint_id, early, sdir, doc),
        "id_only_answers": canned_cases(seg, labels, k_case, order),
        "planted_unknown_ids_rejected": {
            "pass": bool(planted) and all(len(lab["tries"]) == 2 and lab["status"] == "ok"
                                          and "unknown_id" in lab["tries"][0]["reason"] for lab in planted),
            "steps": [lab["step"] for lab in planted]},
        "two_tries_at_most": {"pass": all(len(lab["tries"]) <= 2 for lab in labels.values())},
        "mechanical": mechanical_checks(doc, manifest_entry),
        "view": view_checks(src, doc, view, rec.requests),
        "rebuild_from_receipts": {"pass": replay.calls == 0 and _strip(again["labels"]) == _strip(doc["labels"])
                                  and json.loads(first)["units"] == again["units"],
                                  "new_calls": replay.calls, "cache_hits": replay.cache_hits},
    }
    flat = {k: v["pass"] if "pass" in v else all(x["pass"] for x in v.values()) for k, v in checks.items()}
    (sdir / "checks.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False, default=str) + "\n")
    kinds: dict[str, int] = {}
    for us in doc["units"]["steps"].values():
        for u in us:
            kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    return {"name": name, "steps": n, "status": doc["status"], "task_units": len(doc["units"]["task"]),
            "units": kinds, "hunks": len(doc["mechanical"]["hunks"]), "calls": doc["labeller"]["calls"],
            "tries": doc["labeller"]["tries"], "checks": flat}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Units Ledger end-to-end check (fake labels, no paid calls)")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "v2" / "ledger-units"))
    ap.add_argument("--source-root", default=None)
    ap.add_argument("--natural-store", default=str(NATURAL_STORE))
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    root = Path(a.source_root).resolve() if a.source_root else _source_root()
    manifest = {}
    if (root / LINKPILOT_MANIFEST).is_file():
        for e in json.loads((root / LINKPILOT_MANIFEST).read_text())["packets"].values():
            manifest[e["stall_checkpoint_id"]] = e
    stalls, opened = [], {}
    for name, rel, run_id, cp, _ in STALLS:
        src = opened.setdefault(rel, ReadOnlyStore(root / rel))
        stalls.append(run_stall(src, name, run_id, cp, out, manifest.get(f"{run_id}/{cp}")))
        print(name, json.dumps(stalls[-1]), flush=True)
    natural = {"skipped": f"no natural-v2 store at {a.natural_store}"}
    nat = Path(a.natural_store)
    if (nat / "journal.sqlite3").is_file():
        src = opened.setdefault(str(nat), ReadOnlyStore(nat))
        # the run of that store with the most workspace-changing steps, at its last branchable point
        best = None
        for run_id in _runs(src):
            changed = sum(1 for e in src.iter_events(run_id) if e.kind == "mutation" and (e.payload or {}).get("relevant"))
            if best is None or changed > best[1]:
                best = (run_id, changed)
        cps = [c for c in sorted(src.list_checkpoints(best[0]), key=lambda c: c.ordinal) if c.branchable]
        natural = run_stall(src, "natural-v2-" + best[0].split(":")[1][:8], best[0], cps[-1].checkpoint_id, out)
        natural["checks"]["hunks_from_point_checkpoints"] = natural["hunks"] > 0
        print("natural", json.dumps(natural), flush=True)
    for src in opened.values():
        src.close()
    every = stalls + ([natural] if "checks" in natural else [])
    all_pass = all(all(s["checks"].values()) for s in every)
    report = {"what": "units Ledger end to end: segmentation, mechanical items, step-by-step labelling with the "
                      "scripted labeller (FAKE labels) on real stalls, canned answers for the id-only check",
              "fake_labels": True, "paid_calls": 0, "command": COMMAND, "source_root": str(root),
              "all_pass": all_pass, "stalls": stalls, "natural_v2": natural,
              "versions": {"segment": U.SEGMENT_VERSION, "units": U.UNITS_VERSION, "view": U.VIEW_VERSION}}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"all_pass": all_pass, "report": str(out / "report.json")}, indent=1))
    return 0 if all_pass else 1


def _runs(src) -> list[str]:
    import sqlite3
    db = sqlite3.connect(f"file:{src.root}/journal.sqlite3?immutable=1", uri=True)
    try:
        return [r[0] for r in db.execute("select distinct run_id from events order by run_id")]
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
