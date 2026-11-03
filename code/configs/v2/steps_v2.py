"""Worker-visible steps of a run, read from its store (read-only journal copy), for the re-entry judge and the
injection donor search.

Step k (brief-interfaces.md) is the k-th model response of a run with its delivered tool outputs and any submission
feedback delivered right after it. Only worker-visible content is read: task disclosure, model responses
(reasoning, visible text, tool calls), delivered observations and submission feedback. Private grades, detector
output and evaluator data are never read. In branch runs, review and note turns (model responses marked with payload
mode "review"/"note", or a review_decision tool call) and the hand-back message are excluded and listed.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "detector-judge-20260926"))
import storeio  # noqa: E402  read-only journal copy, blobs read in place

MASKS = [(re.compile(r"deepseek[\w.\-]*", re.I), "[MODEL]"), (re.compile(r"\bgpt-?\d[\w.\-]*", re.I), "[MODEL]"),
         (re.compile(r"\bluna\b", re.I), "[MODEL]"), (re.compile(r"\bglm[\w.\-]*", re.I), "[MODEL]"),
         (re.compile(r"\bopenai\b|\banthropic\b|\bclaude\b|\bopus\b", re.I), "[VENDOR]")]
EXCLUDED_MODES = ("review", "note")
SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def mask(text: str) -> str:
    for rx, rep in MASKS:
        text = rx.sub(rep, text)
    return text


@lru_cache(maxsize=64)
def store(path: str) -> "storeio.Store":
    return storeio.Store(Path(path).resolve())


def _command(tc: dict) -> str:
    f = tc.get("function") or tc
    args = f.get("arguments")
    try:
        a = json.loads(args) if isinstance(args, str) else (args or {})
        return a.get("command") if isinstance(a, dict) and "command" in a else json.dumps(a)
    except ValueError:
        return str(args)


def run_scope(store_path: str, run_id: str) -> dict:
    for _, d in store(store_path).events(run_id):
        if d.get("scope"):
            return d["scope"]
    return {}


def run_steps(store_path: str, run_id: str, *, through_sequence: int | None = None,
              handback_header: str | None = None) -> dict:
    st = store(str(store_path))
    task, steps, excluded = [], [], []
    cur = None
    for seq, d in st.events(run_id):
        if through_sequence is not None and seq > through_sequence:
            break
        k, p = d["kind"], d.get("payload") or {}
        delivered = st.ref_text(d.get("delivered_ref")) if d.get("delivered_ref") else None
        if k == "task_disclosure":
            task.append({"role": p.get("role"), "text": delivered or ""})
        elif k == "model_response":
            o = st.ref_json(d.get("output_ref"))
            o = o if isinstance(o, dict) else {"content": str(o or "")}
            calls = o.get("tool_calls") or []
            mode = p.get("mode") or ("review" if any((c.get("function") or c).get("name") == "review_decision"
                                                     for c in calls) else None)
            if mode in EXCLUDED_MODES:
                excluded.append({"seq": seq, "what": f"{mode} turn"})
                cur = None
                continue
            cur = {"k": len(steps) + 1, "seq": seq, "reasoning": o.get("reasoning_content") or o.get("reasoning"),
                   "content": o.get("content"), "commands": [_command(c) for c in calls], "outputs": [],
                   "feedback": [], "usage": d.get("usage") or {}}
            steps.append(cur)
        elif delivered is not None and (d["delivered_ref"].get("visibility") == "worker_visible"):
            if (handback_header and delivered.startswith(handback_header)) or k in ("handback_delivery", "handback"):
                excluded.append({"seq": seq, "what": "hand-back message"})
            elif cur is None:
                excluded.append({"seq": seq, "what": f"{k} outside a step"})
            elif k == "native_submission_feedback":
                cur["feedback"].append(delivered)
            else:
                cur["outputs"].append(delivered)
    return {"task": task, "steps": steps, "excluded": excluded}


def step_text(s: dict, *, include_reasoning: bool = True) -> str:
    parts = []
    if include_reasoning and s.get("reasoning"):
        parts.append(f"[reasoning]\n{s['reasoning']}")
    if s.get("content"):
        parts.append(f"[visible text]\n{s['content']}")
    for c in s["commands"]:
        parts.append(f"[command]\n{c}")
    for o in s["outputs"]:
        parts.append(f"[output]\n{o}")
    for f in s["feedback"]:
        parts.append(f"[submission feedback]\n{f}")
    return "\n".join(parts)


def render(steps: list[dict], title: str = "STEP") -> str:
    return mask("\n\n".join(f"----- {title} {i} -----\n{step_text(s)}" for i, s in enumerate(steps, 1)))


def approx_tokens(text: str) -> int:
    """A size estimate (characters / 4), used only to compare donor sizes; never billed or capped on."""
    return (len(text) + 3) // 4


def checkpoint_point(store_path: str, run_id: str, checkpoint_id: str) -> tuple[int | None, int | None]:
    """(event_sequence, step_index) of a checkpoint; step_index from the record if present, else the number of model
    responses at or before its event sequence."""
    st = store(str(store_path))
    try:
        cp = st.checkpoint_json(checkpoint_id)
    except TypeError:
        return None, None
    seq = cp.get("event_sequence")
    k = cp.get("step_index", (cp.get("validation") or {}).get("step_index"))
    if k is None and seq is not None:
        k = sum(1 for q, d in st.events(run_id) if q <= seq and d["kind"] == "model_response")
    return seq, k
