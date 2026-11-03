"""Linking pilot round 2b, fixed items: the flat Ledger item pass (decisions + tags) with labeller protocol v2 (user,
2026-10-22) on each pilot packet's stall prefix, through the repo's own ``passes.build_items`` (two tries per call,
v2 answer check). Replaces round 2's tools/ledger_items.py (frozen v1 prompts) for the r2b packets; round 2 outputs are
left as they are.

  set -a; . ~/.config/trajectory-retention/openrouter.env; set +a
  .venv/bin/python configs/v2/linkpilot/ledger_items_r2b.py --labeller deepseek|scripted|replay --budget-usd 4.0 \
      [--linkpilot DIR] [--parallel] [p1 ...]

Writes LINKPILOT/r2b/ledger/<pid>/items.json (build_r2b input; researcher side, holds the Ledger's own statuses),
calls.json and tries.json (every try: accepted or not, and what the reply looked like); receipts in
LINKPILOT/r2b/ledger/labeller (one ArtifactStore for all packets, so the budget guard sees the whole run). Source stores
are opened read-only. The code's git commit is recorded in every items.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from trajectory_retention.ledger import passes as P  # noqa: E402
from trajectory_retention.ledger.__main__ import make_client  # noqa: E402
from trajectory_retention.observers.t2.jev import _ref  # noqa: E402
from trajectory_retention.observers.t2.store_ro import ReadOnlyStore  # noqa: E402

LINKPILOT = Path("/Users/sidsri/projects/llm-trajectories/artifacts/calibration/linkpilot")
DRIVER_VERSION = "linkpilot-ledger-items-r2b/v2"
PROTOCOL = "v2"  # the first r2b run; --protocol picks another (v2.1: step-boundary render)
# Researcher-side description of a reply that is not one JSON object: the agent's tool-call markup or the log's own
# step format (the labeller carrying on the work log as the coding agent).
AGENT_MARKUP = re.compile(r"<｜DSML｜|<tool_call|</?function|\"name\"\s*:\s*\"bash\"|^command \d+:|"
                          r"^output of command \d+|COMPLETE_TASK_AND_SUBMIT|^=== Step \d+ ===|"
                          r"^</?step \d+>|^\[step \d+\] ", re.M)


def git_state():
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"repo": str(REPO), "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [ln for ln in git("status", "--porcelain", "--", "src", "configs").splitlines() if ln]}


def reply_text(client, record_id):
    for kind in ("glm_call", "scripted_call"):
        try:
            rec = client.store.get_record(kind, record_id)
        except KeyError:
            continue
        raw = client.store.get_bytes(_ref(rec["response_ref"])).decode("utf-8")
        try:
            body = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(body, dict) and "choices" in body:
            return ((body["choices"] or [{}])[0].get("message") or {}).get("content") or ""
        return raw
    return None


def describe(text):
    if text is None:
        return "no_receipt"
    if P.parse_json_object_only(text) is not None:
        return "one_json_object"
    if AGENT_MARKUP.search(text):
        return "agent_markup"
    return "prose_or_other"


def run_packet(pid, entry, client, out, protocol=PROTOCOL):
    src = ReadOnlyStore(entry["store"])
    try:
        doc = P.build_items(src, entry["run_id"], entry["stall_checkpoint_id"], client, protocol=protocol)
    finally:
        src.close()
    if doc["stall_point"] != entry["stall_point_step"]:
        raise ValueError(f"{pid}: Ledger sees {doc['stall_point']} steps, packet stall point is "
                         f"{entry['stall_point_step']}")
    calls = doc["labeller"]["passes"]
    tries = []
    for r in calls:
        text = reply_text(client, r["record_id"]) if r.get("record_id") else None
        tries.append({"prompt": r["prompt"], "chunk": r["chunk"], "repeat": r["repeat"], "status": r["status"],
                      "reply": describe(text) if r["status"] != "error" else "call_error",
                      "reply_chars": None if text is None else len(text), "cost_usd": r.get("cost_usd"),
                      "finish_reason": r.get("finish_reason"), "error": r.get("error"),
                      "reply_head": None if text is None else text.strip()[:400],
                      "reply_tail": None if text is None else text.strip()[-200:]})
    doc.update({"driver": DRIVER_VERSION, "packet_id": pid, "store": entry["store"], "code": git_state(),
                "flat_ledger": {"passes_run": ["decisions", "tags"],
                                "passes_skipped": ["grouping", "direct_boundary", "approach_status"],
                                "tags_rewind_point": doc["stall_point"], "protocol": protocol,
                                "why": "flat Ledger, no approach layer; labeller protocol v2 (user, 2026-10-22)"}})
    d = out / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    (d / "calls.json").write_text(json.dumps(calls, indent=1, ensure_ascii=False) + "\n")
    (d / "tries.json").write_text(json.dumps(tries, indent=1, ensure_ascii=False) + "\n")
    return doc, tries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeller", choices=["scripted", "deepseek", "replay"], required=True)
    ap.add_argument("--budget-usd", type=float, default=0.0)
    ap.add_argument("--linkpilot", default=str(LINKPILOT))
    ap.add_argument("--out", default=None, help="default: LINKPILOT/r2b/ledger (scripted: r2b/ledger-scripted)")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--protocol", default=PROTOCOL, choices=sorted(P.PROTOCOLS))
    ap.add_argument("packets", nargs="*")
    a = ap.parse_args()
    here = Path(a.linkpilot)
    out = Path(a.out) if a.out else here / "r2b" / ("ledger-scripted" if a.labeller == "scripted" else "ledger")
    manifest = json.load(open(here / "manifest.json"))["packets"]
    pids = a.packets or sorted(manifest)
    out.mkdir(parents=True, exist_ok=True)
    client = make_client(a.labeller, out / "labeller", budget_usd=a.budget_usd)
    print("labeller:", json.dumps(client.describe()), "| code:", json.dumps(git_state()), flush=True)
    results, lock = {}, threading.Lock()

    def one(pid):
        try:
            doc, tries = run_packet(pid, manifest[pid], client, out, a.protocol)
            c = {}
            for i in doc["items"]:
                c[i["type"]] = c.get(i["type"], 0) + 1
            res = {"status": doc["status"], "items": c, "dropped": len(doc["dropped"]),
                   "cost_usd": round(doc["labeller"]["cost_usd"], 4),
                   "tries": [(t["prompt"], t["chunk"], t["repeat"], t["status"], t["reply"]) for t in tries]}
        except Exception as exc:
            res = {"error": f"{type(exc).__name__}: {exc}"}
        with lock:
            results[pid] = res
            print(pid, json.dumps(res), "| total spent", round(getattr(client, "spent", 0.0), 4), flush=True)

    if a.parallel and a.labeller != "scripted":
        ts = [threading.Thread(target=one, args=(p,)) for p in pids]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    else:
        for p in pids:
            one(p)
    print(json.dumps({"total_spent_usd": round(getattr(client, "spent", 0.0), 4), "results": results}, indent=1))


if __name__ == "__main__":
    main()
