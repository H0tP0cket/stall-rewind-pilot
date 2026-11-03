"""Opus 5.5 record-only review choice for arms 2 and 3 (contract: one per stall on each view; never executed, never
blocks a branch, never fed back). Same pending/record pattern as configs/detector-judge-20260926/opus_shadow.py.

Track E's review mode writes the exact review request to <branch attempt dir>/opus-review-input.json when the spec
says review.opus_record (arms_v2 sets it on repeat 0 of each main stall). This script makes no model calls: a live
Claude session runs the sub-agent `.claude/agents/opus-review-record.md` on each pending input file.
Identical inputs (same request bytes) are recorded once.

  pending --study R [--prefix main]   write the input files for unrecorded review inputs and print the task list
  record  --study R                   validate each written answer and file it under R/opus-review/records/
  refused --study R --id ID [--note]  mark an input the API refused (recorded as refused, never retried silently)
Input files are hard-wrapped at 1800 characters per line so the Read tool never truncates a line.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from study_common import canonical, read_json, safe, sha256_bytes, validate_decision, write_json

AGENT = "opus-review-record (.claude/agents; model opus)"
END = "===== END OF INPUT FILE ====="
INSTRUCTIONS = """INSTRUCTIONS

You are taking one review turn in place of a software-engineering agent, for a research study. Below is exactly the
request the agent receives at this point: its messages in order and the one tool it may call. Answer as the agent
would be asked to: make the single review_decision call you would make, using only what this request shows. Your
call is recorded for comparison only. It is never executed and nothing is fed back to the agent.

Write your answer to the output path as one JSON object and nothing else:
{"name": "review_decision", "arguments": { ...exactly the tool's parameters... }}"""


def wrap(text: str, width: int = 1800) -> str:
    out = []
    for ln in text.split("\n"):
        while len(ln) > width:
            out.append(ln[:width])
            ln = ln[width:]
        out.append(ln)
    return "\n".join(out)


def find_inputs(study: Path, prefix: str | None) -> list[Path]:
    pat = f"runs/{prefix or '*'}/branches/*/a*/opus-review-input.json"
    return sorted(study.glob(pat))


def render(inp: dict) -> str:
    req = inp.get("request") or {"messages": inp.get("messages"), "tools": inp.get("tools")}
    parts = [INSTRUCTIONS, "", "TOOL (JSON schema)", json.dumps(req["tools"], indent=1), "", "REQUEST MESSAGES", ""]
    for i, m in enumerate(req["messages"], 1):
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"===== MESSAGE {i} ({m.get('role')}) =====\n{content or ''}")
    return wrap("\n".join(parts)) + f"\n{END}\n"


def cmd_pending(study: Path, prefix: str | None) -> list[dict]:
    od = study / "opus-review"
    (od / "inputs").mkdir(parents=True, exist_ok=True)
    tasks, seen = [], {}
    for p in find_inputs(study, prefix):
        inp = read_json(p)
        req = inp.get("request") or {"messages": inp.get("messages"), "tools": inp.get("tools")}
        digest = sha256_bytes(canonical(req))
        stall_id = inp["branch_id"].split("/")[0]
        gid = f"{safe(stall_id)}__{inp['arm']}__{digest[:8]}"
        seen.setdefault(gid, []).append(str(p))
        if len(seen[gid]) > 1 or (od / "records" / f"{gid}.json").exists() or (od / "inputs" / f"{gid}.answer.json").exists():
            continue
        refused = (od / "inputs" / f"{gid}.refused").exists()
        if refused:
            continue
        text = render(inp)
        f = od / "inputs" / f"{gid}.md"
        f.write_text(text)
        tasks.append({"id": gid, "stall_id": stall_id, "arm": inp["arm"], "branch_id": inp["branch_id"],
                      "view_ref": inp.get("view_ref"), "rewind_mode": inp.get("rewind_mode"),
                      "prompt_version": inp.get("prompt_version"), "input": str(f),
                      "answer": str(od / "inputs" / f"{gid}.answer.json"), "chars": len(text),
                      "input_sha256": sha256_bytes(text.encode()), "request_sha256": digest})
    prior = {t["id"]: t for t in read_json(od / "pending.json", [])}
    prior.update({t["id"]: t for t in tasks})
    write_json(od / "pending.json", list(prior.values()))
    write_json(od / "sources.json", seen)
    print(json.dumps([{k: t[k] for k in ("id", "arm", "chars", "input", "answer")} for t in tasks], indent=1))
    return tasks


def cmd_record(study: Path) -> dict:
    od = study / "opus-review"
    pend = {t["id"]: t for t in read_json(od / "pending.json", [])}
    made = []
    for ans in sorted((od / "inputs").glob("*.answer.json")):
        gid = ans.name.removesuffix(".answer.json")
        out = od / "records" / f"{gid}.json"
        if out.exists() or gid not in pend:
            continue
        t = pend[gid]
        raw = ans.read_text()
        try:
            obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except (AttributeError, ValueError):
            obj = None
        args = (obj or {}).get("arguments", obj) if isinstance(obj, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = None
        units = read_json(t["view_ref"])["units"] if t.get("view_ref") and Path(t["view_ref"]).exists() else {}
        errs = ["unparseable answer"] if args is None else validate_decision(args, t["rewind_mode"], units)
        write_json(out, {"id": gid, "stall_id": t["stall_id"], "arm": t["arm"], "rewind_mode": t["rewind_mode"],
                         "prompt_version": t["prompt_version"], "decision": args, "valid": not errs, "errors": errs,
                         "judge": AGENT, "input": t["input"], "input_sha256": t["input_sha256"],
                         "request_sha256": t["request_sha256"], "raw_answer_sha256": sha256_bytes(raw.encode()),
                         "role": "record-only Opus choice; never executed, never blocks, never fed back"})
        made.append(gid)
    refused = sorted(p.name.removesuffix(".refused") for p in (od / "inputs").glob("*.refused"))
    recs = sorted(p.stem for p in (od / "records").glob("*.json"))
    summary = {"records": len(recs), "new": made, "refused": refused, "pending_unanswered":
               sorted(set(pend) - set(recs) - set(refused))}
    write_json(od / "summary.json", summary)
    print(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pending", "record", "refused"))
    ap.add_argument("--study", required=True)
    ap.add_argument("--prefix", default="main")
    ap.add_argument("--id")
    ap.add_argument("--note")
    a = ap.parse_args(argv)
    study = Path(a.study)
    if a.cmd == "pending":
        return cmd_pending(study, a.prefix)
    if a.cmd == "record":
        return cmd_record(study)
    (study / "opus-review" / "inputs" / f"{a.id}.refused").write_text(a.note or "API safeguard refused the input")
    return {"refused": a.id}


if __name__ == "__main__":
    main()
