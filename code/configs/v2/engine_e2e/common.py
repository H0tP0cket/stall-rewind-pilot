"""Shared helpers for the engine end-to-end checks (scripted models only; no provider calls)."""
from __future__ import annotations

import datetime as dt
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MAIN = Path.home() / "projects/llm-trajectories"  # gitignored task packages and artifacts live in the main checkout
REGISTRY_V1 = MAIN / "benchmark_tasks/admission/terminalbench-v1/registry.json"
TASK = "terminalbench-llm-inference-batching-scheduler-v1"
PACKAGE = MAIN / "benchmark_tasks/packages/terminalbench-v1" / TASK
SCRATCH = Path.home() / ".cache/trajectory-retention/v2-engine"
ARTIFACTS = MAIN / "artifacts/v2/engine"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
PARTIAL = "cd /app/task_file/scripts && python3 baseline_packer.py"  # the lane's public partial control (5 of 6 pass)


def step(content, commands, *, tokens=2500, cost=0.0, reasoning=None) -> dict:
    entry = {"content": content, "commands": list(commands),
             "usage": {"prompt_tokens": 1000, "completion_tokens": tokens, "total_tokens": 1000 + tokens, "cost": cost}}
    if reasoning is not None:
        entry["reasoning"] = reasoning
    return entry


def events(store, run_id):
    return list(store.iter_events(run_id))


def lifecycle(evs, transition):
    return [e for e in evs if e.kind == "lifecycle" and e.payload.get("transition") == transition]


def ts(event) -> float:
    return dt.datetime.fromisoformat(event.timestamp).timestamp()


def requests(store, evs):
    return [store.get_bytes(e.input_ref).decode("utf-8") for e in evs if e.kind == "model_request"]


def step_overheads(evs) -> list[float]:
    """Seconds from each accepted or rejected model response to the next model request (tools, capture, grading)."""
    responses = [e for e in evs if e.kind == "model_response"]
    reqs = [e for e in evs if e.kind == "model_request"]
    return [ts(reqs[i + 1]) - ts(responses[i]) for i in range(min(len(responses), len(reqs) - 1))]


def stats(values) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 3), "median": round(statistics.median(values), 3),
            "max": round(max(values), 3), "total": round(sum(values), 3)}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=False, default=str) + "\n")
