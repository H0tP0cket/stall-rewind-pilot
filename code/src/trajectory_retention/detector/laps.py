"""Small, deterministic worker-delivery recurrence rules; no semantic judge.

Only completed, actually delivered tool observations enter this stream. Exact
output equality is deliberately conservative. Shell syntax we cannot interpret
stays opaque; there is no inferred check/region mapping. Counts are provisional
v5 development choices, not calibrated scientific settings.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex

VERSION = "worker-lap-evidence/v1"
RULES = {
    "lap_repeated_failure": "Three distinct delivered attempts of the same action with the same result and a nonzero tool exit; no relevant mutation between attempts.",
    "lap_action_motif": "Two consecutive, nonoverlapping repetitions of a 2-4 action-result sequence containing at least two distinct actions; no relevant mutation between repetitions.",
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def action_identity(tool_name: str, command: str) -> dict:
    """Normalize only single simple invocations; retain unsupported shell bytes."""
    opaque = {"class": "opaque", "tool": tool_name, "command": command}
    if tool_name != "bash" or re.search(r"[\n;|&<>`$*?{}()]", command):
        return opaque
    try:
        argv = shlex.split(command)
    except ValueError:
        return opaque
    if not argv:
        return opaque
    kinds = {"cat": "read", "head": "read", "tail": "read", "rg": "search",
             "grep": "search", "ls": "inspect", "pytest": "test"}
    kind = kinds.get(argv[0])
    if argv[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
        kind = "test"
    if kind is None:
        return opaque
    return {"class": kind, "tool": tool_name, "argv": argv,
            "cwd_policy": "fresh-worker-subshell"}


def observation(*, attempt_id: str, tool_name: str, command: str, delivered: str,
                returncode: int, status: str, event_ids: dict, event_sequence: int,
                relevant_mutation: bool = False) -> dict:
    """Input uses delivered text only; raw/private grader output is not accepted."""
    action = action_identity(tool_name, command)
    # A timeout, missing delivery or transport failure is not a worker loop.
    eligible = tool_name == "bash" and status in {"ok", "command_failed"} and type(returncode) is int and not relevant_mutation
    return {"evidence_version": VERSION, "attempt_id": attempt_id, "action": action,
            "action_key": digest(action), "observation_key": digest(delivered),
            "returncode": returncode, "status": status, "eligible": eligible,
            "relevant_mutation": relevant_mutation,
            "source_event_ids": event_ids, "event_sequence": event_sequence,
            "output_basis": "actual_worker_delivery", "normalization": "exact-delivered-text"}


class LapWindow:
    """Per-observed-window stream, with no durable state outside replay inputs."""
    def __init__(self):
        self.seen: set[str] = set()
        self.actions: list[dict] = []
        self.failures: dict[tuple, list[dict]] = {}
        self.signal: dict | None = None
        self.requested: set[str] = set()

    def clear_evidence(self):
        self.actions.clear()
        self.failures.clear()
        self.signal = None

    def add(self, item: dict, known_tokens: int) -> dict | None:
        if item.get("evidence_version") != VERSION or item.get("output_basis") != "actual_worker_delivery":
            raise ValueError("Lap input requires versioned actual-delivery evidence")
        attempt = item.get("attempt_id")
        if not isinstance(attempt, str) or not attempt or not item.get("source_event_ids"):
            raise ValueError("Lap input requires a distinct attempt and source events")
        if attempt in self.seen:
            return None
        self.seen.add(attempt)
        if item.get("eligible") is not True:
            self.clear_evidence()
            return None
        if (item.get("status") not in {"ok", "command_failed"} or type(item.get("returncode")) is not int
                or item.get("relevant_mutation") is not False
                or item.get("action_key") != digest(item.get("action"))):
            raise ValueError("Contradictory eligible lap evidence")
        row = {**item, "known_tokens": known_tokens}
        self.actions.append(row)
        self.actions = self.actions[-8:]
        selected, name = [], None
        key = (row["action_key"], row["observation_key"], row["returncode"])
        if row["returncode"] != 0 and row["action"]["class"] not in {"read", "search", "inspect"}:
            chain = self.failures.setdefault(key, [])
            chain.append(row)
            del chain[:-3]
            if len(chain) == 3:
                selected, name = list(chain), "lap_repeated_failure"
        if name is None:
            for size in (2, 3, 4):
                if len(self.actions) < 2 * size:
                    continue
                left, right = self.actions[-2 * size:-size], self.actions[-size:]
                symbols = lambda rows: [(x["action_key"], x["observation_key"], x["returncode"]) for x in rows]
                if len({x["action_key"] for x in left}) >= 2 and symbols(left) == symbols(right):
                    selected, name = left + right, "lap_action_motif"
                    break
        if name is None:
            # Pending behavioral recurrence is refreshed only by a new match.
            # Unrelated subsequent actions do not carry a stale candidate forward.
            self.signal = None
            return None
        previous = self.signal
        pattern_key = digest([name, key if name == "lap_repeated_failure" else symbols(left)])
        self.signal = {"signal": name, "pattern_key": pattern_key, "strength": "behavioral", "onset_seq": selected[0]["seq"],
            "onset_interval": [selected[0]["seq"], selected[1]["seq"]],
            "completion_seq": row["seq"], "recognized_seq": row["seq"],
            "recognized_tokens": known_tokens,
            "evidence": {"attempt_ids": [x["attempt_id"] for x in selected],
                         "source_events": [x["source_event_ids"] for x in selected],
                         "worker_event_sequences": [x["event_sequence"] for x in selected],
                         "symbols": [[x["action_key"], x["observation_key"], x["returncode"]] for x in selected],
                         "rule": RULES[name]}}
        if previous is not None and previous["pattern_key"] == pattern_key:
            self.signal["recognized_seq"] = previous["recognized_seq"]
            self.signal["recognized_tokens"] = previous["recognized_tokens"]
        return self.signal
