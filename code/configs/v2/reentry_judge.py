"""Mechanism measure (contract: Outcomes): re-entry into the failed approach, judged by Opus 5.5 blind to arm
(hand-back stripped), with a 10% human check. No model calls here: a live Claude session runs the sub-agent
`.claude/agents/opus-reentry-judge.md` on each pending input file.

Each input shows the task, the stall's current approach (source steps from the frozen boundary's first step through
the stall step) and the branch's own new steps only. The prefix, the hand-back message, review and note turns, arm
names, seeds and model names are never shown; the worker-visible constants (hand-back header, generic line, review
prompt) are redacted if the agent's own text repeats them. Item ids are salted hashes; the key that maps items back
to branches and arms lives outside the inputs folder.

  build   --study R [--prefix main]    inputs + key for every finished branch, then the pending task list
  record  --study R                    file each written answer under R/reentry/records/
  sample  --study R [--fraction 0.1]   blind human-check packet (inputs + labels.csv; answers and arms not included)
  agree   --study R                    agreement and Cohen's kappa, human vs Opus, on the sampled items
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from pathlib import Path

import prompt_texts
import steps_v2
from study_common import ARMS, ARMS_V3, iter_results, load_specset, read_json, runs_dir, sha256_bytes, worker_text, write_json
from trajectory_retention.analysis import v2_stats as S

PROMPT_VERSION = "reentry-judge.v0-draft"
AGENT = "opus-reentry-judge (.claude/agents; model opus)"
TERRA_AGENT = "gpt-6.1-terra (Codex subscription; development pilot)"
END = "===== END OF INPUT FILE ====="
LABELS = ("yes", "no", "unclear")


def wrap(text: str, width: int = 1800) -> str:
    out = []
    for ln in text.split("\n"):
        while len(ln) > width:
            out.append(ln[:width])
            ln = ln[width:]
        out.append(ln)
    return "\n".join(out)


def salt(rd: Path) -> str:
    p = rd / "salt"
    if not p.exists():
        rd.mkdir(parents=True, exist_ok=True)
        p.write_text(os.urandom(16).hex())
    return p.read_text().strip()


def redact(text: str) -> tuple[str, int]:
    consts, _ = worker_text()
    n = 0
    for k in ("HANDBACK_HEADER", "GENERIC_LINE", "REVIEW_PROMPT"):
        n += text.count(consts[k])
        text = text.replace(consts[k], "[redacted]")
    for arm in sorted(set(ARMS) | set(ARMS_V3), key=len, reverse=True):
        n += text.count(arm)
        text = text.replace(arm, "[redacted]")
    return text, n


def build_one(stall_dir: Path, result: dict) -> tuple[str | None, dict]:
    st = read_json(stall_dir / "stall.json")
    b = read_json(stall_dir / "boundary.json", None)
    if b is None:
        return None, {"skip": "boundary.json missing"}
    if not result.get("store") or not result.get("run_id"):
        return None, {"skip": "result.json has no branch store/run_id"}
    consts, _ = worker_text()
    seq, k = steps_v2.checkpoint_point(st["store"], st["run_id"], st["checkpoint_id"])
    src = steps_v2.run_steps(st["store"], st["run_id"], through_sequence=seq)
    first = b.get("current_approach_first_step") or (b["rewind_point"] + 1)
    approach = src["steps"][first - 1:st["step_index"]]
    br = steps_v2.run_steps(result["store"], result["run_id"], handback_header=consts["HANDBACK_HEADER"])
    if not approach:
        return None, {"skip": "no source steps in the current approach"}
    if not br["steps"]:
        return None, {"skip": "branch has no continuation steps"}
    task = next((t["text"] for t in src["task"] if t["role"] == "user"), "")
    body = (f"{prompt_texts.judge_prompt(PROMPT_VERSION)['text']}\n\n\n===== TASK =====\n{steps_v2.mask(task)}\n\n\n"
            f"===== EARLIER APPROACH ({len(approach)} steps) =====\n{steps_v2.render(approach, 'EARLIER STEP')}\n\n\n"
            f"===== CONTINUATION ({len(br['steps'])} steps) =====\n{steps_v2.render(br['steps'], 'CONTINUATION STEP')}")
    body, n = redact(body)
    return wrap(body) + f"\n{END}\n", {"approach_steps": [first, st["step_index"]], "continuation_steps": len(br["steps"]),
                                       "excluded_from_branch": br["excluded"], "redactions": n}


def cmd_build(study: Path, prefix: str, judge: str = "opus") -> dict:
    rd = study / "reentry"
    (rd / "inputs").mkdir(parents=True, exist_ok=True)
    s = salt(rd)
    agent = TERRA_AGENT if judge == "terra-pilot" else AGENT
    key = read_json(rd / "key.json", {"judge": agent, "prompt": prompt_texts.judge_prompt(PROMPT_VERSION) | {"text": None},
                                      "items": {}, "skipped": {}})
    if key["judge"] != agent:
        raise ValueError("cannot mix re-entry judge models in one analysis")
    m = load_specset(study / "specsets" / f"{prefix}.json")
    made = 0
    for block, job, res, _ in iter_results(m, runs_dir(study, prefix)):
        if res is None:
            continue
        item = sha256_bytes(f"{s}|{job['branch_id']}".encode())[:12]
        if item in key["items"]:
            continue
        text, info = build_one(Path(block["stall_dir"]), res)
        if text is None:
            key["skipped"][job["branch_id"]] = info["skip"]
            continue
        (rd / "inputs" / f"{item}.md").write_text(text)
        key["items"][item] = {"branch_id": job["branch_id"], "arm": job["arm"], "stall_id": block["unit_id"],
                              "task_id": block["task_id"], "input_sha256": sha256_bytes(text.encode()), **info}
        made += 1
    write_json(rd / "key.json", key)
    pending = [{"id": i, "input": str(rd / "inputs" / f"{i}.md"), "answer": str(rd / "inputs" / f"{i}.answer.json")}
               for i in sorted(key["items"]) if not (rd / "records" / f"{i}.json").exists()
               and not (rd / "inputs" / f"{i}.answer.json").exists()]
    write_json(rd / "pending.json", pending)
    return {"built": made, "items": len(key["items"]), "pending": len(pending), "skipped": len(key["skipped"])}


def cmd_judge_terra(study: Path) -> dict:
    from trajectory_retention.observers.codex_subscription import CodexSubscription, MODEL
    rd = study / "reentry"
    key = read_json(rd / "key.json")
    if key["judge"] != TERRA_AGENT:
        raise ValueError("build with --judge terra-pilot before running the pilot judge")
    transport = CodexSubscription(rd / "codex-sessions")
    done = 0
    for item, info in sorted(key["items"].items()):
        output = rd / "inputs" / f"{item}.answer.json"
        if output.exists():
            continue
        text = (rd / "inputs" / f"{item}.md").read_text()
        if sha256_bytes(text.encode()) != info["input_sha256"] or not text.rstrip().endswith(END):
            raise ValueError(f"changed or incomplete blinded input: {item}")
        body = {"model": MODEL, "reasoning": {"effort": "max"}, "max_tokens": 128000,
                "messages": [{"role": "system", "content":
                  "You are a judge for a software-engineering research study. Judge only the supplied record. "
                  "Follow its judging instructions and return exactly the requested JSON object. Do not use tools."},
                  {"role": "user", "content": text}]}
        response = transport.post(body, {"purpose": "blinded-reentry", "item": item})
        write_json(rd / "receipts" / f"{item}.json", response)
        output.write_text(response["choices"][0]["message"]["content"])
        done += 1
    return {"new_answers": done, **cmd_record(study)}


def cmd_record(study: Path) -> dict:
    rd = study / "reentry"
    key = read_json(rd / "key.json")
    made = []
    for ans in sorted((rd / "inputs").glob("*.answer.json")):
        item = ans.name.removesuffix(".answer.json")
        out = rd / "records" / f"{item}.json"
        if out.exists() or item not in key["items"]:
            continue
        raw = ans.read_text()
        try:
            obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except (AttributeError, ValueError):
            obj = {}
        v = obj.get("reentry") if obj.get("reentry") in LABELS else "unparseable"
        write_json(out, {"item": item, "reentry": v, "first_reentry_step": obj.get("first_reentry_step"),
                         "evidence": obj.get("evidence"), "judge": key["judge"], "prompt_version": PROMPT_VERSION,
                         "input_sha256": key["items"][item]["input_sha256"]})
        made.append(item)
    return {"new_records": len(made), "records": len(list((rd / "records").glob("*.json")))}


def cmd_sample(study: Path, fraction: float, seed: int) -> dict:
    rd = study / "reentry"
    items = sorted(p.stem for p in (rd / "records").glob("*.json"))
    if not items:
        raise SystemExit("no recorded judgements to sample")
    k = max(1, math.ceil(fraction * len(items)))
    pick = random.Random(seed).sample(items, k)
    hc = rd / "human-check"
    pk = hc / "packet"
    if pk.exists():
        shutil.rmtree(pk)
    pk.mkdir(parents=True)
    keymap = {}
    with open(pk / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "reentry (yes / no / unclear)", "first re-entry continuation step (or blank)", "note"])
        for i, item in enumerate(pick, 1):
            shutil.copy(rd / "inputs" / f"{item}.md", pk / f"item-{i:02d}.md")
            w.writerow([f"{i:02d}", "", "", ""])
            keymap[f"{i:02d}"] = item
    (pk / "README.md").write_text(
        f"# Re-entry human check (blind)\n\n{k} of {len(items)} judged branches, drawn at random (seed {seed}). Each "
        "item shows exactly what the model judge saw: the task, the earlier approach and the continuation. Arms and "
        "the judge's answers are not in this folder; do not open ../KEY.json until you have finished.\n\n"
        "Fill labels.csv with yes / no / unclear (does the continuation re-enter the earlier approach?), then run\n"
        "`python configs/v2/reentry_judge.py agree --study <study>`.\n")
    write_json(hc / "KEY.json", {"seed": seed, "fraction": fraction, "items": keymap})
    return {"sampled": k, "of": len(items), "packet": str(pk)}


def cmd_agree(study: Path) -> dict:
    hc = study / "reentry" / "human-check"
    keymap = read_json(hc / "KEY.json")["items"]
    rows = list(csv.reader(open(hc / "packet" / "labels.csv")))[1:]
    pairs = []
    for r in rows:
        if r and r[1].strip():
            rec = read_json(study / "reentry" / "records" / f"{keymap[r[0]]}.json")
            pairs.append((r[1].strip().lower(), rec["reentry"]))
    res = {"labelled": len(pairs), "agreement": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
           "kappa": S.cohen_kappa(pairs), "pairs": pairs}
    write_json(hc / "agreement.json", res)
    return res


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "record", "sample", "agree", "judge-terra"))
    ap.add_argument("--judge", choices=("opus", "terra-pilot"), default="opus")
    ap.add_argument("--study", required=True)
    ap.add_argument("--prefix", default="main")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args(argv)
    study = Path(a.study)
    r = {"build": lambda: cmd_build(study, a.prefix, a.judge), "record": lambda: cmd_record(study),
         "judge-terra": lambda: cmd_judge_terra(study),
         "sample": lambda: cmd_sample(study, a.fraction, a.seed), "agree": lambda: cmd_agree(study)}[a.cmd]()
    print(json.dumps(r, indent=1))
    return r


if __name__ == "__main__":
    main()
