"""Engine check of not-sent-zero-cost-retry.v1 (user, 2026-11-02) next to worst-case-charged-retry.v1. No provider calls.

The live worker client (ReceivedResponseGuardV1, charged-retry rule bound from configs/v2/settings.json) is pointed at:
  dns       an unresolvable host: the connection never opens, so every attempt is a known $0 attempt, retried, uncharged
  silent    a local server that accepts and never answers: the request was sent, so every retried attempt keeps its
            worst-case charge
A dummy API key is used; no request reaches any provider.

Usage: python configs/v2/engine_e2e/m14_not_sent.py   (writes artifacts/v2/engine/m14-not-sent.json)
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ["OPENROUTER_API_KEY"] = "dummy-not-a-key"
from trajectory_retention.execution.model import ModelAbort  # noqa: E402
from trajectory_retention.execution.received_response_guard_v1 import ReceivedResponseGuardV1  # noqa: E402
from trajectory_retention.provider_policy import NOT_SENT_RULE  # noqa: E402

SETTINGS = json.loads((ROOT / "configs/v2/settings.json").read_text())


def attempts(url: str, timeout: float) -> list[dict]:
    m = ReceivedResponseGuardV1(model_name="deepseek/deepseek-v4.1-flash", model_settings={"max_tokens": 1000},
                                max_attempts=3, backoff_seconds=0.01, timeout_seconds=timeout, url=url)
    m.unresolved_retry = SETTINGS["unresolved_retry"]
    try:
        m.query([{"role": "user", "content": "hi"}])
    except ModelAbort as e:
        return [{"attempt": a["attempt"], "status": a["status"], "cost_usd": getattr(a.get("usage"), "cost_usd", None),
                 "not_sent": a.get("not_sent"), "charge_usd": (a.get("worst_case_charge") or {}).get("usd"),
                 "decision": a["retry_decision"]["reason"]} for a in e.attempts]
    raise AssertionError("expected the call to fail")


def main() -> None:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held = []
    threading.Thread(target=lambda: [held.append(srv.accept()) for _ in range(8)], daemon=True).start()
    dns = attempts("https://openrouter-dns-check.invalid/api/v1/chat/completions", 5)
    silent = attempts(f"http://127.0.0.1:{srv.getsockname()[1]}/v1", 1)
    checks = {
        "dns_all_attempts_known_zero_and_uncharged": all(
            a["cost_usd"] == 0.0 and a["charge_usd"] is None and (a["not_sent"] or {}).get("rule") == NOT_SENT_RULE
            for a in dns),
        "dns_retried_until_attempt_limit": [a["decision"] for a in dns][-1] == "attempt_limit" and len(dns) == 3,
        "silent_retries_keep_worst_case_charge": all(a["charge_usd"] and a["charge_usd"] > 0 for a in silent[:-1])
        and all(a["not_sent"] is None for a in silent),
        "silent_stops_under_charged_rule": silent[-1]["decision"] == "attempt_limit_unresolved",
    }
    report = {"what": __doc__.split("\n\n")[0], "provider_calls": 0, "checks": checks, "all_pass": all(checks.values()),
              "dns": dns, "silent": silent}
    out = ROOT / "artifacts/v2/engine/m14-not-sent.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"all_pass": report["all_pass"], "checks": checks}, indent=1))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
