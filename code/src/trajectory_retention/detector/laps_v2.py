"""Versioned recurrence over delivered attempts, rather than shell spelling.

No private grade, inferred semantics, or fuzzy similarity enters this parser.
Unsupported shell stays exact. Presentation wrappers are removed only around a
single identifiable operation; compound output is usable only with a complete
native pytest failure section. The original action/result digests survive.
"""
from __future__ import annotations

import ast
import json
import re
import shlex

from .laps import digest

VERSION = "worker-lap-evidence/v2"
RULES = {
    "lap_repeated_failure": "Three distinct delivered attempts with the same supported operation and failure identity; intervening edits do not erase identical failures.",
    "lap_action_motif": "Two consecutive repetitions of a 2-4 action/result motif with distinct actions and at least one verification; duplicate inspection alone is insufficient.",
    "lap_repeated_verification": "Three distinct identical verification attempts and equivalent delivered results inside the observed no-progress window; passing smoke checks do not establish task completion.",
    "lap_persistent_check_failure": "One native pytest check has an identical complete diagnostic on three distinct executions of the same operation, even when other failures change.",
}


def _words(command):
    try:
        return shlex.split(command)
    except ValueError:
        return None


def _silent_edit_prefix(prefix: str) -> bool:
    """Recognize only a single edit whose normal output is redirected/silent.

    In particular `pytest A; pytest B` is not an edit-plus-test operation: its
    first failure must never be attributed to the second invocation.
    """
    prefix = prefix.strip()
    heredoc = re.fullmatch(r"cat\s*>{1,2}\s*([^\s;&|<>`$]+)\s*<<\s*(['\"])([A-Za-z_][A-Za-z_0-9]*)\2\s*\n.*\n\3", prefix, re.S)
    if heredoc:
        return True
    if re.search(r"[\n`$]", prefix):
        return False
    try:
        lexer = shlex.shlex(prefix, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        argv = list(lexer)
    except ValueError:
        return False
    return bool(argv and argv[0] == "sed" and "-i" in argv[1:]
                and not any(token and all(c in ";&|<>" for c in token) for token in argv))


def action_identity(tool_name: str, command: str) -> dict:
    opaque = {"class": "opaque", "tool": tool_name, "command": command}
    if tool_name != "bash":
        return opaque
    # A single Python stdin script with assertions is an explicit local check.
    # Its exact program remains part of identity: changed inputs never merge.
    script = re.fullmatch(r"\s*(python3?|[^\s]+/python3?)\s+-\s+<<\s*(['\"]?)([A-Za-z_][A-Za-z_0-9]*)\2\s*\n(.*?)\n\3\s*", command, re.S)
    if script:
        try:
            tree = ast.parse(script[4])
        except SyntaxError:
            return opaque
        if any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
            return {"class": "verification", "tool": tool_name,
                    "interpreter": script[1], "stdin_program": script[4]}
        return opaque
    cwd, compound = None, False
    text = command.strip()
    # Only an explicit leading cd affects the canonical working directory.
    cd = re.match(r"^cd\s+([^\n;&]+)\s*&&\s*(.+)$", text, re.S)
    if cd:
        path = _words(cd[1])
        if not path or len(path) != 1 or re.search(r"[$`*?]", cd[1]):
            return opaque
        cwd, text = path[0], cd[2]
    # A supported final pytest invocation can follow an edit. Never infer a
    # result for this invocation from the outer shell exit: native evidence is
    # required below. Keep the prefix digest as provenance, outside identity.
    final = re.search(r"(?:^|[;\n])\s*((?:python3?\s+-m\s+)?pytest\s+[^\n;]+)\s*$", text)
    if final and final.start() > 0:
        prefix = text[:final.start()]
        if not _silent_edit_prefix(prefix):
            return opaque
        text, compound = final[1], True
    # head/tail only change presentation. An equal native diagnostic is still
    # necessary; missing/truncated failure sections cannot match by prefix.
    pipe = re.fullmatch(r"(.*?)\s*\|\s*(?:head|tail)(?:\s+(?:-n\s+\d+|-\d+))?\s*", text)
    if pipe:
        text = pipe[1]
    text = re.sub(r"\s+2>&1\s*$", "", text)
    if re.search(r"[\n;|&<>`$*?{}()]", text):
        return opaque
    argv = _words(text)
    if not argv:
        return opaque
    kinds = {"cat": "read", "head": "read", "tail": "read", "rg": "search",
             "grep": "search", "ls": "inspect", "pytest": "verification"}
    kind = kinds.get(argv[0])
    if argv[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
        # Executable selection is retained; different environments are not
        # assumed identical just because both run pytest.
        kind = "verification"
    if argv[:3] in (["python", "-m", "unittest"], ["python3", "-m", "unittest"]):
        kind = "verification"
    if kind is None or (compound and kind != "verification"):
        return opaque
    return {"class": kind, "tool": tool_name, "argv": argv,
            "cwd": cwd, "cwd_policy": "fresh-worker-subshell",
            "compound_requires_native_result": compound}


def _delivered_body(delivered: str, returncode: int):
    # This is the JSON the worker actually received, not the raw tool stream.
    try:
        body = json.loads(delivered)
    except (ValueError, TypeError):
        return delivered, False
    if (type(body) is dict and type(body.get("returncode")) is int
            and body["returncode"] == returncode and type(body.get("output")) is str
            and set(body) == {"returncode", "output"}):
        return body["output"], True
    return delivered, False


def _pytest_failure(body: str):
    """Complete native sections only; keep check, exception and all operands."""
    # Collection failures have a stable file identity. A following native
    # section/summary establishes the captured section ended before truncation.
    summaries = re.findall(r"(?m)^(?:FAILED|ERROR) ([^\s]+)(?: - .*)?$", body)
    sections = list(re.finditer(r"(?m)^_{3,}\s+([^\n]+?)\s+_{3,}\s*$", body))
    results = []
    for start in sections:
        tail = body[start.end():]
        end = re.search(r"(?m)^(?:_{3,}\s+.+\s+_{3,}|={3,}\s+.+\s+={3,})\s*$", tail)
        if end is None:
            continue
        diagnostic = tail[:end.start()].strip()
        if not re.search(r"(?m)^E\s+(?:\S+(?:Error|Exception)(?::|\b)|assert\b)", diagnostic):
            continue
        label = start[1].strip()
        if label.startswith("ERROR collecting "):
            check = label.removeprefix("ERROR collecting ")
        else:
            # Match the full native test/class/parameter identity. Ambiguous
            # headings in different files never become an inferred check ID.
            candidates = [node for node in summaries if "::" in node
                          and ".".join(node.split("::")[1:]) == label]
            if len(candidates) != 1:
                continue
            check = candidates[0]
        results.append({"check": check, "diagnostic": diagnostic})
    return results


def observation(*, attempt_id: str, tool_name: str, command: str, delivered: str,
                returncode: int, status: str, event_ids: dict, event_sequence: int,
                relevant_mutation: bool = False, mutation_evidence: dict | None = None) -> dict:
    action = action_identity(tool_name, command)
    body, decoded = _delivered_body(delivered, returncode)
    failures = _pytest_failure(body) if action["class"] == "verification" and "pytest" in action.get("argv", []) else []
    normalization = []
    # For supported pytest the complete observed collection diagnostic supplies
    # identity independently of head limits and the outer pipeline's exit code.
    result = {"delivered": delivered}
    if failures:
        result = {"pytest_collection_failures": failures}
        normalization.append("complete-pytest-failure-sections/v1")
    elif action["class"] == "verification" and not action.get("compound_requires_native_result"):
        # Only the native pytest summary duration is nuisance. Input numbers,
        # traceback text and any other timing values remain exact.
        normalized = re.sub(r"(?m)^(=+\s+.*\b(?:passed|failed|error|errors|skipped)\b.*?\bin )\d+(?:\.\d+)?(s(?: \([^\n]*\))?\s*=+)$", r"\1<elapsed>\2", body)
        normalized = re.sub(r"(?m)^((?:\d+ (?:passed|failed|errors?|skipped|deselected|xfailed|xpassed)(?:, )?)+ in )\d+(?:\.\d+)?s$", r"\1<elapsed>s", normalized)
        result = {"worker_result": normalized, "returncode": returncode}
        if normalized != body:
            normalization.append("pytest-final-summary-duration/v1")
    eligible = (tool_name == "bash" and status in {"ok", "command_failed"}
                and type(returncode) is int)
    if action.get("compound_requires_native_result") and not failures:
        eligible = False
    # Compound and simple attempts match only with the same complete native
    # result. Whether a prefix edited files is retained separately in provenance.
    identity = {k: v for k, v in action.items() if k != "compound_requires_native_result"}
    return {"evidence_version": VERSION, "attempt_id": attempt_id, "action": identity,
            "action_key": digest(identity), "observation_key": digest(result),
            "raw_action_key": digest({"tool": tool_name, "command": command}),
            "raw_observation_key": digest(delivered), "result_identity": result,
            "failure_units": failures,
            "returncode": returncode, "status": status, "eligible": eligible,
            "failure": bool(failures) or (eligible and returncode != 0 and action["class"] not in {"read", "search", "inspect"}),
            "verification": action["class"] == "verification",
            "relevant_mutation": relevant_mutation, "mutation_evidence": mutation_evidence,
            "source_event_ids": event_ids, "event_sequence": event_sequence,
            "output_basis": "actual_worker_delivery", "normalization": normalization,
            "delivered_wrapper_decoded": decoded}


class LapWindow:
    def __init__(self):
        self.seen = set()
        self.requested = set()
        self.clear_evidence()

    def clear_evidence(self):
        self.actions = []
        self.chains = {}
        self.check_chains = {}
        self.signal = None

    def add(self, item: dict, known_tokens: int):
        if item.get("evidence_version") != VERSION or item.get("output_basis") != "actual_worker_delivery":
            raise ValueError("Lap input requires versioned actual-delivery evidence")
        attempt = item.get("attempt_id")
        if not isinstance(attempt, str) or not attempt or not item.get("source_event_ids"):
            raise ValueError("Lap input requires distinct attempt and source events")
        if attempt in self.seen:
            return None
        self.seen.add(attempt)
        if item.get("status") not in {"ok", "command_failed"} or type(item.get("returncode")) is not int:
            self.clear_evidence()  # Actual transport/observation barrier.
            return None
        if item.get("eligible") is not True:
            self.actions.clear()  # Unsupported action is not missing history.
            return None
        if (item.get("action_key") != digest(item.get("action"))
                or item.get("observation_key") != digest(item.get("result_identity"))):
            raise ValueError("Contradictory lap identity")
        row = {**item, "known_tokens": known_tokens}
        key = (row["action_key"], row["observation_key"])
        # A changed result for the SAME operation supersedes its old chain.
        # Unrelated operations and edits leave a persistent failure testable.
        for old in list(self.chains):
            if old[0] == key[0] and old != key:
                del self.chains[old]
        if self.signal and any(s[0] == key[0] and s[1] != key[1] for s in self.signal["evidence"]["symbols"]):
            self.signal = None
        self.actions.append(row)
        self.actions = self.actions[-8:]
        selected, name = [], None
        if row.get("failure") or row.get("verification"):
            chain = self.chains.setdefault(key, [])
            chain.append(row)
            del chain[:-3]
            if len(chain) == 3:
                selected = list(chain)
                name = "lap_repeated_failure" if row["failure"] else "lap_repeated_verification"
        unit_key = None
        present_checks = {unit["check"] for unit in row.get("failure_units", [])}
        for old in list(self.check_chains):
            if old[0] == row["action_key"] and old[1] not in present_checks:
                # Missing a previously failing check is not proof of a pass,
                # but it prevents calling its later return persistent failure.
                del self.check_chains[old]
        for unit in row.get("failure_units", []):
            check_key = (row["action_key"], unit["check"], digest(unit["diagnostic"]))
            for old in list(self.check_chains):
                if old[:2] == check_key[:2] and old != check_key:
                    del self.check_chains[old]
            chain = self.check_chains.setdefault(check_key, [])
            # One native check repeated in a malformed output is still only
            # one worker attempt. It cannot supply several recurrence votes.
            if not chain or chain[-1]["attempt_id"] != attempt:
                chain.append(row)
                del chain[:-3]
            if name is None and len(chain) == 3:
                selected, name, unit_key = list(chain), "lap_persistent_check_failure", check_key
        def symbols(rows):
            return [[x["action_key"], x["observation_key"]] for x in rows]
        if name is None:
            for size in (2, 3, 4):
                if len(self.actions) < 2 * size:
                    continue
                left, right = self.actions[-2 * size:-size], self.actions[-size:]
                if (len({x["action_key"] for x in left}) >= 2
                        and any(x.get("verification") or x.get("failure") for x in left)
                        and symbols(left) == symbols(right)):
                    selected, name = left + right, "lap_action_motif"
                    break
        if name is None:
            return None  # A match remains pending until progress/gap or change.
        previous = self.signal
        pattern_key = digest([name, symbols(selected[:len(selected)//2]) if name == "lap_action_motif" else unit_key or key])
        self.signal = {"signal": name, "pattern_key": pattern_key, "strength": "behavioral",
            "onset_seq": selected[0]["seq"], "onset_interval": [selected[0]["seq"], selected[1]["seq"]],
            "completion_seq": row["seq"], "recognized_seq": row["seq"], "recognized_tokens": known_tokens,
            "evidence": {"attempt_ids": [x["attempt_id"] for x in selected],
                         "source_events": [x["source_event_ids"] for x in selected],
                         "worker_event_sequences": [x["event_sequence"] for x in selected],
                         "symbols": symbols(selected), "failure_unit_key": unit_key, "rule": RULES[name]}}
        if previous is not None and previous["pattern_key"] == pattern_key:
            self.signal["recognized_seq"] = previous["recognized_seq"]
            self.signal["recognized_tokens"] = previous["recognized_tokens"]
        return self.signal
