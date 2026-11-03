"""Load and render the v2 prompt drafts in configs/v2/prompts (review turn for arms 2 and 3, arm 5 note, re-entry judge;
for arm set v3 also the flat-Ledger review turn of arm 2 and the read-only turns of arms 5 and 6).

A prompt version names a file plus how to assemble it (prompts/index.json). Worker-visible constants (REVIEW_PROMPT,
OUTPUT_PLACEHOLDER) come from trajectory_retention.experiments.v2_text through study_common.worker_text(), so the
wording lives in one place. Rendering is a single placeholder pass, so braces inside a view's text are never touched.

  python configs/v2/prompt_texts.py lint            render every worker-visible prompt and check the wording rules
  python configs/v2/prompt_texts.py show <version> [--mode per_channel|coupled]
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from study_common import HERE, lint_worker_text, sha256_bytes, worker_text

PROMPTS = HERE / "prompts"
_PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


@lru_cache(maxsize=None)
def index() -> dict:
    return json.loads((PROMPTS / "index.json").read_text())


@lru_cache(maxsize=None)
def sections(file: str) -> dict:
    """{section: text | {field: text}} from a prompt file; HTML comments and marker lines are dropped."""
    raw = re.sub(r"<!--.*?-->\s*", "", (PROMPTS / file).read_text(), flags=re.S)
    out: dict = {}
    sec = field = None
    buf: list[str] = []

    def flush():
        if sec is None:
            return
        text = "\n".join(buf).rstrip("\n")
        if field is None:
            if text.strip() or sec not in out:
                out[sec] = text if text.strip() else {}
        else:
            out.setdefault(sec, {})[field] = text

    for line in raw.split("\n"):
        m = re.fullmatch(r"=== section: (\S+) ===", line)
        f = re.fullmatch(r"=== field: (\S+) ===", line)
        if m or f:
            flush()
            if m:
                sec, field = m.group(1), None
            else:
                field = f.group(1)
            buf = []
        else:
            buf.append(line)
    flush()
    return out


def fill(template: str, values: dict[str, str]) -> str:
    missing = sorted({m for m in _PLACEHOLDER.findall(template) if m not in values})
    if missing:
        raise KeyError(f"unfilled placeholders {missing}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def _meta(version: str, kind: str) -> dict:
    p = index()["prompts"].get(version)
    if p is None or p["kind"] != kind:
        raise KeyError(f"no {kind} prompt {version}")
    return p


def _result(version: str, text: str, files: list[str], source: str | None = None) -> dict:
    return {"version": version, "text": text, "sha256": sha256_bytes(text.encode()),
            "template_sha256": {f: sha256_bytes((PROMPTS / f).read_bytes()) for f in files},
            "worker_text_source": source, "status": index()["status"]}


def unit_id(kind: str, n: int, unit_format: dict | None = None) -> str:
    return (unit_format or index()["unit_id_format"])[kind].format(n=n)


def example_args(mode: str, unit_format: dict | None = None) -> tuple[str, str]:
    u = lambda k, n: unit_id(k, n, unit_format)
    common = {"drop": [u("output", 6), u("step", 9)], "keep": [u("step", 17)], "change_nothing": False}
    if mode == "per_channel":
        ex = {"rewind": {"workspace": 14, "context": 14, "runtime": 14, "outputs": None}, **common}
        nothing = {"rewind": None, "drop": [], "keep": [], "change_nothing": True}
    else:
        ex = {"rewind_to": 14, **common}
        nothing = {"rewind_to": None, "drop": [], "keep": [], "change_nothing": True}
    return json.dumps(ex), json.dumps(nothing)


def render_review(version: str, *, mode: str, view_text: str, snapshot: str, unit_format: dict | None = None) -> dict:
    """The review turn's user message for arms 2 and 3, exactly as the agent will read it."""
    if mode not in ("per_channel", "coupled"):
        raise ValueError(mode)
    p = _meta(version, "review")
    s = sections(p["file"])
    consts, source = worker_text()
    ex, nothing = example_args(mode, unit_format)
    tool_example = fill(s["tool_example"], {"EXAMPLE_ARGS": ex, "NOTHING_ARGS": nothing,
                                            "EXAMPLE_EXPLANATION": s[f"example:{mode}"]})
    ledger = ""
    if p.get("ledger_examples"):
        ledger = "\n" + fill(s["ledger_examples"], {"APPROACH_3": unit_id("approach", 3, unit_format),
                                                    "ITEM_18": unit_id("item", 18, unit_format)}) + "\n"
    view = s[f"view:{p['view']}"]
    text = fill(s["base"], {"REVIEW_PROMPT": consts["REVIEW_PROMPT"], "OUTPUT_PLACEHOLDER": consts["OUTPUT_PLACEHOLDER"],
                            "VIEW_DESCRIPTION": view["VIEW_DESCRIPTION"], "VIEW_TITLE": view["VIEW_TITLE"],
                            "DROP_EXTRA": view["DROP_EXTRA"], "KEEP_EXTRA": view["KEEP_EXTRA"],
                            "REWIND_TEXT": s[f"rewind:{mode}"],
                            "TOOL_EXAMPLE": tool_example, "LEDGER_EXAMPLES": ledger, "VIEW_TEXT": view_text,
                            "SNAPSHOT": snapshot})
    return _result(version, text, [p["file"], p["tool"]], source) | {"mode": mode, "arm": p["arm"]}


def render_read_only(version: str, *, view_text: str, snapshot: str) -> dict:
    """The read-only review turn's user message for arms 5 and 6 of arm set v3 (no tool), as the agent reads it."""
    p = _meta(version, "read_only")
    s = sections(p["file"])
    consts, source = worker_text()
    reading = ("\n" + s["ledger_reading"] + "\n") if p.get("ledger_reading") else ""
    view = s[f"view:{p['view']}"]
    text = fill(s["base"], {"READ_ONLY_PROMPT": consts["READ_ONLY_PROMPT"], "VIEW_DESCRIPTION": view["VIEW_DESCRIPTION"],
                            "VIEW_TITLE": view["VIEW_TITLE"], "LEDGER_READING": reading, "VIEW_TEXT": view_text,
                            "SNAPSHOT": snapshot})
    return _result(version, text, [p["file"]], source) | {"mode": None, "arm": p["arm"]}


def tool_schema(version: str, mode: str) -> dict:
    return json.loads((PROMPTS / _meta(version, "review")["tool"]).read_text())[mode]


def render_note(version: str, *, n_undone_turns: int, rewind_point: int) -> dict:
    """The arm 5 note request (appended to the agent's full context at the stall checkpoint)."""
    p = _meta(version, "note")
    s = sections(p["file"])
    span = s["span:all"] if rewind_point == 0 else s["span:partial"]
    text = fill(s["body"], {"UNDONE_SPAN": fill(span["UNDONE_SPAN"], {"N_UNDONE": str(n_undone_turns)}),
                            "POINT_TEXT": span["POINT_TEXT"]})
    return _result(version, text, [p["file"]]) | {"arm": p["arm"], "n_undone_turns": n_undone_turns,
                                                   "rewind_point": rewind_point}


def judge_prompt(version: str) -> dict:
    p = _meta(version, "judge")
    return _result(version, sections(p["file"])["body"], [p["file"]])


def lint_all() -> dict:
    """Render every worker-visible prompt (both rewind modes, both note spans) and check the wording rules."""
    hits, rendered = {}, {}
    view = "S1 [step]\n$ ls\nfile.py\n"
    for version, p in index()["prompts"].items():
        if p["kind"] == "review":
            outs = [render_review(version, mode=m, view_text=view, snapshot="(snapshot)") for m in ("per_channel", "coupled")]
        elif p["kind"] == "read_only":
            outs = [render_read_only(version, view_text=view, snapshot="(snapshot)")]
        elif p["kind"] == "note":
            outs = [render_note(version, n_undone_turns=7, rewind_point=12), render_note(version, n_undone_turns=19,
                                                                                          rewind_point=0)]
        else:
            continue
        for i, o in enumerate(outs):
            rendered[f"{version}#{i}"] = o["sha256"]
            found = lint_worker_text(o["text"])
            if found:
                hits[f"{version}#{i}"] = found
    consts, source = worker_text()
    for k, v in consts.items():
        found = lint_worker_text(v)
        if found:
            hits[f"constant:{k}"] = found
    return {"ok": not hits, "hits": hits, "rendered": rendered, "worker_text_source": source}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lint")
    s = sub.add_parser("show")
    s.add_argument("version")
    s.add_argument("--mode", default="per_channel")
    a = ap.parse_args()
    if a.cmd == "lint":
        r = lint_all()
        print(json.dumps(r, indent=1))
        raise SystemExit(0 if r["ok"] else 1)
    kind = index()["prompts"][a.version]["kind"]
    if kind == "review":
        print(render_review(a.version, mode=a.mode, view_text="<view text>", snapshot="<snapshot>")["text"])
    elif kind == "read_only":
        print(render_read_only(a.version, view_text="<view text>", snapshot="<snapshot>")["text"])
    elif kind == "note":
        print(render_note(a.version, n_undone_turns=7, rewind_point=12)["text"])
    else:
        print(judge_prompt(a.version)["text"])


if __name__ == "__main__":
    main()
