"""Round 2 scorer smoke test on the round 2b packets: the linking pilot's test-r2/smoke_real.py, pointed at r2b.
Two synthetic label files per packet (mechanical, not judgements) are scored by the round 2 scorer (scoring_r2.py, read
only), which parses the real r2b packet, strips its item boxes, and finds LATER-step quotes and task quotes.

  .venv/bin/python configs/v2/linkpilot/smoke_r2b.py [--linkpilot DIR] p2 p3 ...

Writes LINKPILOT/r2b/test-r2/smoke-<pid>/{synthA,synthB,report}.json and r2b/test-r2/smoke-summary.json; exit code 0
only when every packet scores with no problems.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LINKPILOT = Path("/Users/sidsri/projects/llm-trajectories/artifacts/calibration/linkpilot")


def labels(pid, fx, text, name, flip):
    later = text.split("# ===== STALL POINT")[1]
    m = re.search(r"^## Step (\d+)  \(LATER\)\n(.*?)(?=^## Step|\Z)", later, re.M | re.S)
    lk = int(m.group(1))
    line = next(l for l in m.group(2).splitlines() if len(l.strip()) > 20 and not l.startswith(("~", "*")))
    task_line = next(l for l in text.split("\n# TASK\n")[1].splitlines() if len(l.strip()) > 30)
    items = fx["items"]
    dec = [i for i in items if i["type"] == "decision"]
    cl = [i for i in items if i["type"] == "claim"]
    sup = [i for i in items if i["type"] in ("claim", "fact")]
    cs = [{"item": c["id"], "status": "unchecked"} for c in cl]
    if flip and cs:
        cs[-1] = {"item": cl[-1]["id"], "status": "contradicted_later", "status_step": lk, "status_quote": line.strip()}
    rests, carries, unl, conc = [], [], [], []
    for d in dec:
        s = [x for x in sup if x["step"] <= d["step"]]
        if s:
            x = s[-1] if not flip else s[0]
            rests.append({"decision": d["id"], "support": x["id"], "confidence": 2, "reliance_step": d["step"],
                          "reliance_quote": d["quote"]})
        if flip:
            rests.append({"decision": d["id"], "support": "TASK", "task_quote": task_line.strip(), "confidence": 1,
                          "reliance_step": d["step"], "reliance_quote": d["quote"]})
    for h in fx["prefix_hunks"]:
        k = int(h[1:].split("h")[0])
        ds = [d for d in dec if d["step"] <= k]
        if ds:
            carries.append({"decision": ds[-1]["id"], "hunks": [h], "confidence": 2})
        else:
            unl.append({"hunk": h, "why": "no_fixed_decision"})
    for c in cl:
        ds = [d for d in dec if d["step"] <= c["step"]]
        if ds:
            conc.append({"claim": c["id"], "decisions": [ds[-1 if not flip else 0]["id"]], "confidence": 2})
    return {"packet_id": pid, "labeller": name, "guide_version": "linkpilot-guide/r2-draft-1", "claim_status": cs,
            "rests_on": rests, "carries_out": carries, "unlinked_hunks": unl, "concerns": conc}


def smoke(pid, lp, root):
    fx = json.load(open(root / f"fixed-items/{pid}.json"))
    text = open(root / f"packet-{pid}-r2.md").read()
    out = root / "test-r2" / f"smoke-{pid}"
    out.mkdir(parents=True, exist_ok=True)
    for n, f in (("synthA", False), ("synthB", True)):
        json.dump(labels(pid, fx, text, n, f), open(out / f"{n}.json", "w"), indent=1)
    r = subprocess.run([sys.executable, "-I", str(lp / "scoring_r2.py"), "--packet", str(root / f"packet-{pid}-r2.md"),
                        "--items", str(root / f"fixed-items/{pid}.json"),
                        "--ledger", str(root / fx["source"]["ledger_items"]),  # the items the packet was built from
                        "--json", str(out / "report.json"), str(out / "synthA.json"), str(out / "synthB.json")],
                       capture_output=True, text=True)
    rep = json.load(open(out / "report.json")) if (out / "report.json").exists() else {"problems": {"run": [r.stderr]}}
    bad = {k: v for k, v in rep["problems"].items() if v}
    return {"returncode": r.returncode, "problems": bad, "pass": not bad and r.returncode == 0,
            "candidate_pairs": rep.get("candidate_pairs"), "ledger": rep.get("ledger"),
            "stderr_tail": r.stderr[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linkpilot", default=str(LINKPILOT))
    ap.add_argument("packets", nargs="+")
    a = ap.parse_args()
    lp = Path(a.linkpilot)
    root = lp / "r2b"
    res = {pid: smoke(pid, lp, root) for pid in a.packets}
    ok = all(r["pass"] for r in res.values())
    (root / "test-r2").mkdir(exist_ok=True)
    (root / "test-r2" / "smoke-summary.json").write_text(json.dumps({"all_pass": ok, "packets": res}, indent=1) + "\n")
    for pid, r in res.items():
        print(pid, "PASS" if r["pass"] else "FAIL", json.dumps({k: r[k] for k in ("problems", "candidate_pairs")}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
