"""Few-cent provider smoke test for the v2 worker, through the worker's own guarded model class.

Four calls on one pinned provider (fallbacks off), with the study's worker settings:
  1. task turn                         -> the model answers with a bash tool call
  2. history incl. its past reasoning  -> next turn, after a fixed tool observation
  2b. same history, reasoning removed  -> prompt-token difference = reasoning tokens sent back
  3. exact repeat of call 2            -> prompt-cache hit evidence
Checks: returned provider matches the pin, past reasoning is counted in prompt tokens, cached prompt tokens on the
repeat, throughput. Writes one JSON artifact (no credentials) to artifacts/v2/provider-smoke/.

Usage: python configs/v2/provider_smoke.py --provider digitalocean [--model deepseek/deepseek-v4.1-flash-1017] [--env-file ~/.config/trajectory-retention/openrouter.env]
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages  # noqa: E402
from trajectory_retention.execution.agent import SYSTEM_TEMPLATE, VERIFICATION_INSTANCE_TEMPLATE  # noqa: E402
from trajectory_retention.execution.model import DEFAULT_OBSERVATION_TEMPLATE, OpenRouterChatModel  # noqa: E402
from trajectory_retention.provider_policy import provider_identity_matches  # noqa: E402

MODEL = "deepseek/deepseek-v4.1-flash"
PINS = {"digitalocean": {"only": ["digitalocean"]},
        "streamlake-fp8": {"only": ["streamlake/fp8"], "quantizations": ["fp8"]},
        "gmicloud-fp8": {"only": ["gmicloud/fp8"], "quantizations": ["fp8"]},
        "morph-fp8": {"only": ["morph/fp8"], "quantizations": ["fp8"]},
        "parasail-fp8": {"only": ["parasail/fp8"], "quantizations": ["fp8"]}}
TASK = (ROOT / "benchmark_tasks/packages/terminalbench-v1/terminalbench-path-tracing-v1/public/instruction.md")
OBSERVATION = {"output": "total 8\ndrwxr-xr-x 2 root root 4096 Oct  7 00:00 .\n-rw-r--r-- 1 root root  812 Oct  7 00:00 "
                         "image.ppm\n", "returncode": 0,
               "exception_info": ""}


def load_env(path: pathlib.Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def usage_of(message: dict) -> dict:
    extra = message["extra"]
    u = extra["usage"]
    raw = extra["response"].get("usage") or {}
    att = extra["attempts"][-1]
    out = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "reasoning_tokens": u.reasoning_tokens,
           "cached_input_tokens": u.cached_input_tokens, "cost_usd": u.cost_usd,
           "latency_seconds": att.get("latency_seconds"), "provider": extra["response"].get("provider"),
           "finish_reason": extra["normalized"]["finish_reason"], "raw_usage": raw, "attempts": len(extra["attempts"])}
    if u.output_tokens and att.get("latency_seconds"):
        out["tokens_per_second_incl_ttft"] = round(u.output_tokens / att["latency_seconds"], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PINS), required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--env-file", default=str(pathlib.Path.home() / ".config/trajectory-retention/openrouter.env"))
    ap.add_argument("--max-tokens", type=int, default=16384)
    a = ap.parse_args()
    load_env(pathlib.Path(a.env_file).expanduser())
    pin = PINS[a.provider]
    settings = {"max_tokens": a.max_tokens, "temperature": 1, "reasoning": {"effort": "max"},
                "provider": {"allow_fallbacks": False, "order": pin["only"], "require_parameters": True, **pin}}
    model = OpenRouterChatModel(model_name=a.model, model_settings=settings, max_attempts=2)
    task = VERIFICATION_INSTANCE_TEMPLATE.replace("{{task}}", TASK.read_text())
    base = [{"role": "system", "content": SYSTEM_TEMPLATE}, {"role": "user", "content": task}]

    calls = {}
    m1 = model.query(copy.deepcopy(base))
    calls["1_task_turn"] = usage_of(m1)
    reasoning = m1.get("reasoning") or ""
    obs = format_toolcall_observation_messages(actions=m1["extra"]["actions"], outputs=[OBSERVATION],
                                               observation_template=DEFAULT_OBSERVATION_TEMPLATE)
    with_r = base + [m1] + obs
    no_r = base + [{k: v for k, v in m1.items() if k not in ("reasoning", "reasoning_details")}] + obs
    m2 = model.query(copy.deepcopy(with_r))
    calls["2_with_past_reasoning"] = usage_of(m2)
    m2b = model.query(copy.deepcopy(no_r))
    calls["2b_past_reasoning_removed"] = usage_of(m2b)
    m3 = model.query(copy.deepcopy(with_r))
    calls["3_repeat_of_2"] = usage_of(m3)

    passback = calls["2_with_past_reasoning"]["input_tokens"] - calls["2b_past_reasoning_removed"]["input_tokens"]
    returned = {c["provider"] for c in calls.values()}
    checks = {
        "provider_matches_pin": all(isinstance(p, str) and provider_identity_matches(pin["only"][0], p) for p in returned),
        "returned_providers": sorted(map(str, returned)),
        "past_reasoning_chars": len(reasoning),
        "past_reasoning_has_details": bool(m1.get("reasoning_details")),
        "prompt_tokens_from_past_reasoning": passback,
        "tokens_per_reasoning_char": round(passback / len(reasoning), 3) if reasoning else None,
        "past_reasoning_sent_back": passback > 0,
        "repeat_cached_input_tokens": calls["3_repeat_of_2"]["cached_input_tokens"],
        "repeat_cache_share": (round(calls["3_repeat_of_2"]["cached_input_tokens"] / calls["3_repeat_of_2"]["input_tokens"], 3)
                               if calls["3_repeat_of_2"]["cached_input_tokens"] is not None else None),
        "total_cost_usd": round(sum(c["cost_usd"] or 0 for c in calls.values()), 5),
    }
    out = {"model": a.model, "provider_pin": settings["provider"], "settings": {k: v for k, v in settings.items() if k != "provider"},
           "model_class": "trajectory_retention.execution.model.OpenRouterChatModel", "task_text": str(TASK.relative_to(ROOT)),
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "calls": calls, "checks": checks,
           "precision": "not observable through the API; see the OpenRouter endpoint listing and the provider's docs"}
    dest = ROOT / "artifacts/v2/provider-smoke"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{a.provider}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(checks, indent=2))
    print(path)


if __name__ == "__main__":
    main()
