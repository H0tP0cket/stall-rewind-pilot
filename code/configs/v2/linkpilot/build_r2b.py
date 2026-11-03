"""Linking pilot round 2b packets: round 2's packet builder (linkpilot tools/build_r2.py) run on the items from
ledger_items_r2b.py (labeller protocol v2). The round 1 packet text is unchanged; the labeller's fixed items are shown
at the top of each step up to the stall. Ledger statuses (claim / idea status, refers_to) stay out of packets and
fixed-items files so labellers set claim status blind to them. Round 1 and round 2 files are read, never written.

  .venv/bin/python configs/v2/linkpilot/build_r2b.py [--linkpilot DIR] [p1 ...]

Packets are built only for packets whose items are complete (every labeller call answered); the others are listed
under "excluded" with the reason. Writes, in LINKPILOT/r2b/: packet-pN-r2.md, fixed-items/pN.json and manifest-r2.json
(no researcher_only fields), with the labeller run's tries and spend (recorded, and a worst case that counts every
timed-out or failed HTTP attempt as billed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from trajectory_retention.observers.t2.jev import _ref  # noqa: E402
from trajectory_retention.observers.t2.store_ro import ReadOnlyStore  # noqa: E402

LINKPILOT = Path("/Users/sidsri/projects/llm-trajectories/artifacts/calibration/linkpilot")
VERSION = "linkpilot-build-r2/v2"
LEDGER = ["ledger"]  # items folder under r2b/ (--ledger-dir)
WHERE = {"reasoning": "reasoning", "message": "message", "command": "command", "output": "output",
         "feedback": "feedback", "format_error": "format error", "user": "user message"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def split_steps(prefix):
    """{k: (start, end)} character ranges of '## Step k' sections in the prefix."""
    heads = [(m.start(), m.end(), int(m.group(1))) for m in re.finditer(r"^## Step (\d+)[^\n]*$", prefix, re.M)]
    out = {}
    for i, (s, e, k) in enumerate(heads):
        out[k] = (s, e, heads[i + 1][0] if i + 1 < len(heads) else len(prefix))
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def item_box(k, items, step_text):
    lines = [f"**Fixed items in step {k}** (from the Ledger labeller; do not add, remove or retype):", ""]
    for it in items:
        q = it["quote"]
        found = q in step_text or norm(q) in norm(step_text)
        where = WHERE.get(it["field_kind"], it["field_kind"])
        extra = []
        if not found:
            extra.append("this text sits in a part of an output the packet cuts; it is quoted here in full")
        if (it.get("occurrences_in_step") or 1) > 1:
            extra.append(f"the same text appears {it['occurrences_in_step']} times in this step")
        need = {"claim": " — needs a status", "decision": "", "fact": "", "idea": ""}[it["type"]]
        lines.append(f"- **{it['id']}** {it['type']} (in the {where}){need}:")
        lines += ["  ~~~~~text"] + ["  " + x for x in q.split("\n")] + ["  ~~~~~"]
        if extra:
            lines.append("  *(" + "; ".join(extra) + ")*")
    lines.append("")
    return lines, [it["id"] for it in items if not (it["quote"] in step_text or norm(it["quote"]) in norm(step_text))]


def run_stats(led, tries):
    agent = [t for t in tries if t["reply"] in ("agent_markup",)]
    return {"status": led["status"], "calls": len({(t["prompt"], json.dumps(t["chunk"])) for t in tries}),
            "tries": len(tries), "failed_tries": sum(1 for t in tries if t["status"] != "ok"),
            "tries_by_reply": dict(Counter(t["reply"] for t in tries)),
            "tries_acting_as_agent_markup": len(agent),
            "chunks_failed": sorted({(t["prompt"], tuple(t["chunk"])) for t in tries if t["status"] != "ok"}
                                    - {(t["prompt"], tuple(t["chunk"])) for t in tries if t["status"] == "ok"}),
            "cost_usd": round(sum(t.get("cost_usd") or 0.0 for t in tries), 4),
            "protocol": led["labeller"].get("protocol", {}).get("protocol"), "max_tries_per_call": 2}


def spend(store_dir):
    """Recorded spend from every receipt, and a worst case that adds, for every HTTP attempt that did not come back
    (timeout or connection error, status 0) or failed with 5xx, the highest single-call cost seen in this run."""
    s = ReadOnlyStore(store_dir)
    try:
        recs = [s.get_json(_ref(r["ref"])) for kind in ("glm_call", "glm_call_failed") for r in s.list_records(kind)]
    finally:
        s.close()
    costs = [r.get("cost_usd") for r in recs]
    top = max([c for c in costs if c] or [0.0])
    lost = sum(1 for r in recs for a in r.get("attempts") or [] if a.get("http_status") in (0,) or
               (a.get("http_status") or 0) >= 500)
    missing = sum(1 for c in costs if c is None)
    recorded = sum(c or 0.0 for c in costs)
    # calls that raised before any receipt was written (e.g. a connection cut mid-response): possibly billed
    no_receipt = []
    for f in sorted(Path(store_dir).parent.glob("*/calls.json")):
        for c in json.loads(f.read_text()):
            if c.get("status") == "error" and "BudgetExceeded" not in (c.get("error") or ""):
                no_receipt.append({"packet_dir": f.parent.name, "prompt": c["prompt"], "chunk": c["chunk"],
                                   "error": c.get("error")})
    lost_total = lost + missing + len(no_receipt)
    return {"receipts": len(recs), "failed_receipts": sum(1 for r in recs if r.get("status") != "ok"),
            "recorded_usd": round(recorded, 4), "receipts_without_cost": missing,
            "http_attempts_lost_or_5xx": lost, "calls_lost_without_receipt": no_receipt,
            "max_single_call_usd": round(top, 4),
            "worst_case_usd": round(recorded + lost_total * top, 4),
            "basis": "recorded = provider-reported usage.cost on each receipt; worst case adds the highest single-call "
                     "cost for every lost (timeout/connection) or 5xx HTTP attempt, every receipt without a cost and "
                     "every call that failed before a receipt was written"}


def build(pid, m1, root, here):
    e = m1["packets"][pid]
    src = here / e["file"]
    assert sha(src) == e["sha256"], f"{pid}: round 1 packet changed"
    led_path = root / LEDGER[0] / pid / "items.json"
    led = json.loads(led_path.read_text())
    assert led["run_id"] == e["run_id"] and led["checkpoint_id"] == e["stall_checkpoint_id"]
    assert led["stall_point"] == e["stall_point_step"]
    text = src.read_text()
    cut = text.index("# ===== STALL POINT")
    prefix, rest = text[:cut], text[cut:]
    ranges = split_steps(prefix)
    by_step = {}
    for it in led["items"]:
        by_step.setdefault(it["step"], []).append(it)
    out, pos, not_found, inserted = [], 0, [], []
    for k in sorted(ranges):
        s, e_head, end = ranges[k]
        out.append(prefix[pos:e_head])
        if by_step.get(k):
            box, nf = item_box(k, by_step[k], prefix[e_head:end])
            ins = "\n\n" + "\n".join(box)
            out.append(ins)
            inserted.append(ins)
            not_found += nf
        pos = e_head
    out.append(prefix[pos:])
    body = "".join(out)
    assert set(by_step) <= set(ranges), f"{pid}: items in steps the packet does not show"
    counts = Counter(it["type"] for it in led["items"])
    hunks = [h for h in e["hunk_index"] if int(h[1:].split("h")[0]) <= e["stall_point_step"]]
    head_old = body.split("\n", 1)[0]
    assert head_old == f"# Packet {pid}", head_old
    intro = (f"# Packet {pid} (round 2)\n\n"
             f"Round 2: the items are fixed. They are listed at the top of each step up to the stall point, with "
             f"their ids and types: {counts.get('decision', 0)} decisions, {counts.get('claim', 0)} claims, "
             f"{counts.get('fact', 0)} facts and {counts.get('idea', 0)} ideas. Add a status for every claim, and the "
             f"rests-on and carries-out links, as `guide-r2.md` explains. The same list is in `fixed-items/{pid}.json`. "
             f"Hunks up to the stall: {len(hunks)}.\n")
    body = intro + body.split("\n", 1)[1]
    r2 = root / f"packet-{pid}-r2.md"
    r2.write_text(body + rest)
    stripped = (body + rest)[len(intro):]
    for ins in inserted:
        stripped = stripped.replace(ins, "", 1)
    fixed = {"packet_id": pid, "version": VERSION, "stall_point": e["stall_point_step"],
             "source": {"ledger_items": str(led_path.relative_to(root)), "ledger_items_sha256": sha(led_path),
                        "labeller_model": led["labeller"]["model"],
                        "labeller_provider": (led["labeller"]["provider"] or {}).get("only"),
                        "labeller_protocol": led["labeller"].get("protocol", {}).get("protocol"),
                        "status": led["status"]},
             "items": [{"id": it["id"], "step": it["step"], "type": it["type"], "where": it["field_kind"],
                        "quote": it["quote"]} for it in led["items"]],
             "prefix_hunks": hunks,
             "quotes_not_in_packet_text": not_found}
    (root / "fixed-items").mkdir(parents=True, exist_ok=True)
    fpath = root / "fixed-items" / f"{pid}.json"
    fpath.write_text(json.dumps(fixed, indent=1, ensure_ascii=False) + "\n")
    leaked = [k for k in ("claim_status", "idea_status", "status_evidence", "refers_to", "survives_rewind")
              if f'"{k}"' in fpath.read_text() or k in intro + "".join(inserted)]
    assert not leaked, f"{pid}: Ledger status fields reached a labeller file: {leaked}"
    tries = json.loads((root / LEDGER[0] / pid / "tries.json").read_text())
    return {"packet_id": pid, "file": r2.name, "sha256": sha(r2), "round1_file": f"../{e['file']}",
            "round1_sha256": e["sha256"], "fixed_items": f"fixed-items/{pid}.json", "fixed_items_sha256": sha(fpath),
            "ledger_items": str(led_path.relative_to(root)), "ledger_items_sha256": sha(led_path),
            "ledger_status": led["status"], "items_by_type": dict(counts), "items_total": len(led["items"]),
            "ledger_dropped": len(led["dropped"]), "quotes_not_in_packet_text": not_found,
            "prefix_hunks": len(hunks), "stall_point_step": e["stall_point_step"],
            "run_id": e["run_id"], "store": e["store"], "stall_checkpoint_id": e["stall_checkpoint_id"],
            "family": e["family"], "task_id": e["task_id"], "worker_model_recorded": e["model"],
            "record_unchanged_outside_item_boxes": stripped == text.split("\n", 1)[1],
            "labeller_run": run_stats(led, tries)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linkpilot", default=str(LINKPILOT))
    ap.add_argument("--ledger-dir", default="ledger", help="items folder under r2b/ (e.g. ledger-v21)")
    ap.add_argument("packets", nargs="*")
    a = ap.parse_args()
    LEDGER[0] = a.ledger_dir
    here = Path(a.linkpilot)
    root = here / "r2b"
    m1 = json.loads((here / "manifest.json").read_text())
    pids = a.packets or sorted(m1["packets"])
    rows, excluded, led0 = {}, {}, None
    for pid in pids:
        led = json.loads((root / LEDGER[0] / pid / "items.json").read_text())
        led0 = led0 or led
        if led["status"] != "ok":
            tries = json.loads((root / LEDGER[0] / pid / "tries.json").read_text())
            st = run_stats(led, tries)
            excluded[pid] = {"reason": "not every labeller call answered with one JSON object in its two tries; "
                                       f"failed chunks: {st['chunks_failed']}",
                             "ledger_items": f"{LEDGER[0]}/{pid}/items.json", "labeller_run": st,
                             "items_by_type_partial": dict(Counter(i["type"] for i in led["items"]))}
            continue
        rows[pid] = build(pid, m1, root, here)
    lab = led0["labeller"]
    proto = (led0["labeller"].get("protocol") or {}).get("protocol")
    man = {"name": f"linkpilot round 2b: fixed items from the Ledger labeller with {proto} (label-only prompts, "
                   "strict answer check); labellers add claim status and links (DRAFT, unratified link types)",
           "builder": f"configs/v2/linkpilot/build_r2b.py ({VERSION}); items from "
                      f"configs/v2/linkpilot/ledger_items_r2b.py at commit {led0['code']['commit']}",
           "labeller": {"model": lab["model"], "provider": lab["provider"], "settings": lab["settings"],
                        "prompts": lab["prompt_sha256"], "protocol": lab.get("protocol"),
                        "passes_run": led0["flat_ledger"]["passes_run"],
                        "passes_skipped": led0["flat_ledger"]["passes_skipped"],
                        "tags_rewind_point": "the stall point (survives-rewind not asked)",
                        "max_tries_per_call": 2},
           "files": {"guide": "guide-r2.md", "schema": "schema-r2.json", "scorer": "scoring_r2.py",
                     "scorer_check": "../test-r2/check_r2.py", "smoke": "test-r2/"},
           "privacy": "Packets hold only worker-visible material and workspace contents; items come from the "
                      "worker-visible prefix only. The Ledger's own claim and idea statuses are kept out of packets "
                      f"and fixed-items files (researcher side: r2b/{LEDGER[0]}/). No researcher_only field from "
                      "manifest.json is copied here.",
           "packets": rows, "excluded": excluded,
           "labeller_spend": spend(root / LEDGER[0] / "labeller")}
    (root / "manifest-r2.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    for pid, r in rows.items():
        print(pid, r["items_by_type"], "hunks", r["prefix_hunks"], "not in text", r["quotes_not_in_packet_text"],
              "record unchanged", r["record_unchanged_outside_item_boxes"], "run", json.dumps(r["labeller_run"]))
    for pid, x in excluded.items():
        print(pid, "EXCLUDED", json.dumps(x))
    print("spend", json.dumps(man["labeller_spend"]))


if __name__ == "__main__":
    main()
