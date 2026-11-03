"""Mechanical per-step flags (no model): workspace change, outcome, noise, output current/stale.

``approach`` is filled in by the boundary pass. The rules are generic shell/test-output rules and are recorded with
their version in every Ledger, so a change is a new version, not a silent edit.

* workspace_change: the run's mutation events mark the step's response batch as changing the workspace or runtime
  (``relevant``), with the changed paths. A format-error step changes nothing.
* outcome: parsed from the step's outputs. ``tests_fail`` when a pytest or unittest summary reports failures or
  errors, ``tests_pass`` when summaries exist and none fails, otherwise ``error`` when some command exited non-zero,
  ``ok`` when all exited zero, ``none`` without outputs. Counts are in ``detail.outcome``.
* noise: every command of the step only lists or displays (ls, find, cat, head, sed -n, ...), with no write
  redirection, and the step changed nothing.
* output current or stale, per output and per step: an output of step k is ``stale`` at the cut when a later step
  re-ran the same command (whitespace-normalised), or a later step changed the workspace/runtime and either the
  command is not a pure listing/display (its output depends on the whole state) or a changed path matches a path the
  command names. Otherwise ``current``. A step's flag is ``current``, ``stale``, ``mixed`` or ``none``.
"""

from __future__ import annotations

import re

from .text import tags_for

FLAGS_VERSION = "ledger-flags/v1"

_SUMMARY = re.compile(r"^=+ (?P<body>.*?) in [\d.]+s(?: \([^)]*\))? =+\s*$", re.M)
_SHORT = re.compile(r"^(?P<body>(?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)"
                    r"(?:, )?)+) in [\d.]+s\s*$", re.M)
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in [\d.]+s\s*$", re.M)
_UNITTEST_END = re.compile(r"^(OK|FAILED)(?: \((?P<body>[^)]*)\))?\s*$", re.M)
_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
_WRITE_REDIRECT = re.compile(r"(?<![0-9&<>])>{1,2}\s*(?!&|/dev/null)")
DISPLAY = {"cat", "head", "tail", "nl", "less", "more", "bat", "xxd", "od", "hexdump", "strings"}
LISTING = {"ls", "find", "tree", "du", "pwd", "wc", "stat", "file", "which", "realpath", "df", "readlink"}
QUIET = {"echo", "printf", "true", ":", "clear", "cd"}


def parse_tests(text: str) -> dict[str, int] | None:
    """The last pytest or unittest summary in an output, as counts; None when there is none."""
    found = None
    for rx in (_SUMMARY, _SHORT):
        for m in rx.finditer(text):
            counts: dict[str, int] = {}
            for n, kind in _COUNT.findall(m.group("body")):
                kind = "errors" if kind.startswith("error") else kind
                counts[kind] = counts.get(kind, 0) + int(n)
            if counts or "no tests ran" in m.group("body"):
                if found is None or m.start() >= found[0]:
                    found = (m.start(), counts)
    ran = list(_UNITTEST_RAN.finditer(text))
    end = list(_UNITTEST_END.finditer(text))
    if ran and end and end[-1].start() > ran[-1].start() and (found is None or ran[-1].start() > found[0]):
        total = int(ran[-1].group(1))
        body = end[-1].group("body") or ""
        counts = {k: int(v) for k, v in re.findall(r"(failures|errors|skipped)=(\d+)", body)}
        failed = counts.pop("failures", 0)
        counts = {"failed": failed, **counts, "passed": max(0, total - failed - counts.get("errors", 0)
                                                            - counts.get("skipped", 0))}
        found = (ran[-1].start(), counts)
    return None if found is None else found[1]


def _programs(command: str) -> list[tuple[str, str]]:
    out = []
    for seg in _SPLIT.split(command.strip()):
        words = seg.split()
        while words and (_ASSIGN.match(words[0]) or words[0] in ("sudo", "env", "command", "time")):
            words = words[1:]
        if words and words[0] == "timeout" and len(words) > 2:
            words = words[2:]
        if words:
            out.append((words[0].rsplit("/", 1)[-1], seg))
    return out


def readonly_kind(command: str) -> str | None:
    """``display`` / ``listing`` when every part of the command only displays or lists, else None."""
    progs = _programs(command)
    if not progs or _WRITE_REDIRECT.search(command):
        return None
    kinds = set()
    for prog, seg in progs:
        if prog == "sed" and re.search(r"(?:^|\s)-n\b", seg) and not re.search(r"(?:^|\s)-i", seg):
            kinds.add("display")
        elif prog in DISPLAY:
            kinds.add("display")
        elif prog in LISTING and not re.search(r"-delete|-exec", seg):
            kinds.add("listing")
        elif prog in QUIET:
            continue
        else:
            return None
    return "display" if "display" in kinds else "listing" if kinds else None


def _norm_cmd(command: str) -> str:
    return " ".join(command.split())


def _paths_match(a: str, b: str) -> bool:
    a, b = a.strip().lstrip("./"), b.strip().lstrip("./")
    for prefix in ("workspace/",):
        a, b = a.removeprefix(prefix), b.removeprefix(prefix)
    return bool(a) and bool(b) and (a == b or a.endswith("/" + b) or b.endswith("/" + a))


def compute_flags(context) -> None:
    """Fill ``step.flags`` (all but ``approach``) and ``step.detail`` for every step of a Context."""
    steps = context.steps
    for step in steps:
        outs = step.outputs()
        tests = []
        codes = []
        for f in outs:
            if f.part in ("output", "output_tail", None):
                t = parse_tests(f.text)
                if t is not None:
                    tests.append(t)
            rc = f.meta.get("returncode")
            if rc is not None and f.part != "output_tail":
                codes.append(rc)
        failing = any(t.get("failed", 0) + t.get("errors", 0) > 0 for t in tests)
        if not outs:
            outcome = "none"
        elif tests:
            outcome = "tests_fail" if failing else "tests_pass"
        elif any(c not in (0, None) for c in codes):
            outcome = "error"
        else:
            outcome = "ok"
        kinds = [readonly_kind(c["command"]) for c in step.commands]
        noise = bool(step.commands) and all(kinds) and not step.workspace_change
        step.flags.update({"workspace_change": step.workspace_change, "outcome": outcome, "noise": noise})
        step.detail.update({"outcome": {"tests": tests, "exit_codes": codes}, "command_kinds": kinds})
    # staleness: needs later steps
    for step in steps:
        per_output = []
        for c in step.commands:
            outs = [f for f in step.outputs() if f.index == c["index"]]
            if not outs:
                continue
            ro = readonly_kind(c["command"])
            named = tags_for(c["command"])
            reason = None
            for later in steps[step.k:]:
                if any(_norm_cmd(lc["command"]) == _norm_cmd(c["command"]) for lc in later.commands):
                    reason = f"re-ran at step {later.k}"
                    break
                if later.workspace_change:
                    if ro is None:
                        reason = f"state changed at step {later.k}"
                        break
                    hit = [p for p in later.changed_paths if any(_paths_match(p, n) for n in named)]
                    if hit:
                        reason = f"{hit[0]} changed at step {later.k}"
                        break
            per_output.append({"index": c["index"], "status": "stale" if reason else "current", "reason": reason})
        states = {o["status"] for o in per_output}
        step.flags["output_current"] = ("none" if not states else states.pop() if len(states) == 1 else "mixed")
        step.detail["outputs"] = per_output
