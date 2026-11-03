"""Does a pinned provider honour the reasoning-effort setting? Same hard prompt at effort max and low.

If reasoning tokens at max are not clearly above low, the provider is ignoring the setting. Also gives throughput on
long generations. Writes one JSON artifact to artifacts/v2/provider-smoke/.

Usage: python configs/v2/provider_effort_probe.py --provider digitalocean [--repeats 2]
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs/v2"))
from provider_smoke import MODEL, PINS, load_env, usage_of  # noqa: E402
from trajectory_retention.execution.agent import SYSTEM_TEMPLATE  # noqa: E402
from trajectory_retention.execution.model import OpenRouterChatModel  # noqa: E402

PROMPT = ("Without running any code first, work out how many positive divisors 7560 has, and the sum of those "
          "divisors. Then call the bash tool exactly once with `echo <count> <sum>`.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=sorted(PINS), required=True)
    ap.add_argument("--env-file", default=str(pathlib.Path.home() / ".config/trajectory-retention/openrouter.env"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=65536)
    a = ap.parse_args()
    load_env(pathlib.Path(a.env_file).expanduser())
    pin = PINS[a.provider]
    rows = []
    for effort in ("max", "low"):
        settings = {"max_tokens": a.max_tokens, "temperature": 1, "reasoning": {"effort": effort},
                    "provider": {"allow_fallbacks": False, "order": pin["only"], "require_parameters": True, **pin}}
        model = OpenRouterChatModel(model_name=MODEL, model_settings=settings, max_attempts=2)
        for r in range(a.repeats):
            try:
                m = model.query([{"role": "system", "content": SYSTEM_TEMPLATE}, {"role": "user", "content": PROMPT}])
                rows.append({"effort": effort, "repeat": r, **usage_of(m)})
            except Exception as exc:  # a truncated or failed call is still evidence
                row = {"effort": effort, "repeat": r, "error": f"{type(exc).__name__}: {exc}"[:500]}
                messages = getattr(exc, "messages", None)
                if messages and isinstance(messages[0].get("extra"), dict) and "usage" in messages[0]["extra"]:
                    row.update(usage_of(messages[0]))  # e.g. no tool call because the response cap was reached
                rows.append(row)
    out = {"model": MODEL, "provider": a.provider, "prompt": PROMPT, "rows": rows,
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dest = ROOT / "artifacts/v2/provider-smoke"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"effort-{a.provider}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    for r in rows:
        print(r["effort"], r.get("reasoning_tokens"), r.get("output_tokens"), r.get("tokens_per_second_incl_ttft"),
              r.get("cost_usd"), r.get("error", ""))
    print(path)


if __name__ == "__main__":
    main()
