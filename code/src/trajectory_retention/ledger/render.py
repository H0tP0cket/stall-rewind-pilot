"""Renderers over a Ledger dict (``ledger.json``): review views for arms 2 and 3, kept-item labels, the arm 6 recipe,
and the labeller's input text. All strings come from ``templates`` (or ``prompts`` for the labeller).

* ``render_review_view(ledger, "ledger")`` (arm 2): the approach index table, then the full context with step, output,
  approach and item IDs and every tag in place.
* ``render_review_view(ledger, "context")`` (arm 3): the same context with step and output IDs only; no labels.
  Both views show the same steps with the same bytes once tags and headers are set aside (``checks`` proves it).
* ``render_review_view(ledger, "ledger_flat")`` (arm set v3, arms 2 and 5): the flat Ledger, i.e. the arm 2 view with
  no approach layer and facts without the survives-rewind status.
* ``label_text(ledger, unit_id)``: what a kept unit becomes in the hand-back (label kept, IDs dropped).
* ``render_recipe(ledger, cap_tokens=...)`` (arm 6): failure evidence of the current approach, facts from the undone
  steps that survive the rewind, never-tried ideas from anywhere; trimmed whole entries from the end to the cap, with
  a cut log.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from . import templates as T

RENDER_VERSION = "ledger-render/v1"


def _output_meta(f: Mapping[str, Any], first: bool) -> str:
    meta = f.get("meta") or {}
    bits = []
    if first and "returncode" in meta:
        bits.append(f"exit code {meta['returncode']}")
    if meta.get("exception_info"):
        bits.append(f"exception: {meta['exception_info']}")
    part = f.get("part")
    if part == "output_head":
        bits.append("first part")
        if meta.get("elided_chars") is not None:
            bits.append(f"{meta['elided_chars']} characters elided after it"
                        + (f": {meta['warning']}" if meta.get("warning") else ""))
    elif part == "output_tail":
        bits.append("last part")
    return f" ({'; '.join(bits)})" if bits else ""


def _tagged(f: Mapping[str, Any], items: Sequence[Mapping[str, Any]], *, flat: bool = False) -> str:
    text = f["text"]
    marks = sorted((i for i in items if i["span"]["field_id"] == f["id"]), key=lambda i: i["span"]["start"])
    out, pos = [], 0
    for item in marks:
        s, e = item["span"]["start"], item["span"]["end"]
        out += [text[pos:s], T.tag_open(item["id"], item_label(item, flat=flat)), text[s:e], T.tag_close(item["id"])]
        pos = e
    out.append(text[pos:])
    return "".join(out)


def item_label(item: Mapping[str, Any], *, with_step: bool = True, flat: bool = False) -> str:
    """``flat`` (the flat Ledger): a fact carries no survives-rewind status (measured against the approach boundary)."""
    label = T.ITEM_LABELS[item["type"]]
    status = None
    if item["type"] == "fact" and item.get("survives_rewind") is not None and not flat:
        status = T.STATUS_TEXT["survives" if item["survives_rewind"] else "lost"]
    elif item["type"] == "claim" and item.get("claim_status"):
        status = item["claim_status"]
    elif item["type"] == "idea" and item.get("idea_status"):
        status = item["idea_status"]
    if status in T.STATUS_TEXT:
        e = (item.get("status_evidence") or {}).get("step")
        status = (T.STATUS_TEXT[status].format(e=e) if (e is not None and with_step)
                  else T.STATUS_TEXT_NO_STEP.get(status, T.STATUS_TEXT[status]))
    return f"{label} · {status}" if status else label


def step_lines(step: Mapping[str, Any], *, style: str, items: Sequence[Mapping[str, Any]] = (),
               elide_outputs: bool = False, part_prefix: str = "") -> list[str]:
    """Body of one step. ``style``: ``ledger`` / ``context`` (IDs) or ``plain`` / ``labeller`` (no IDs).
    ``part_prefix`` is put before every part header (labeller render v2 only; empty everywhere else)."""
    ids = style in ("ledger", "ledger_flat", "context")
    k = step["k"]
    fields = step["fields"]
    by_index: dict[Any, list] = defaultdict(list)
    for f in fields:
        if f["kind"] == "output":
            by_index[f.get("index")].append(f)
    commands = {f.get("index") for f in fields if f["kind"] == "command"}
    lines: list[str] = []

    def body(f):
        return _tagged(f, items, flat=style == "ledger_flat") if style in ("ledger", "ledger_flat") else f["text"]

    def outputs(index):
        for n, o in enumerate(by_index.get(index, [])):
            oid = T.mark(T.OUTPUT_ID.format(k=k)) + " " if ids else ""
            num = (index + 1) if index is not None else "?"
            lines.append(part_prefix + T.OUTPUT_HEADER.format(oid=oid, n=num, meta=_output_meta(o, n == 0)))
            lines.append(T.LABELLER_ELIDED.format(n=len(o["text"])) if elide_outputs else body(o))

    for f in fields:
        kind = f["kind"]
        if kind == "output":
            if f.get("index") not in commands and f is by_index[f.get("index")][0]:
                outputs(f.get("index"))
            continue
        lines.append(part_prefix + T.FIELD_HEADERS[kind].format(n=(f.get("index") or 0) + 1))
        lines.append(body(f))
        if kind == "command":
            outputs(f.get("index"))
    return lines


def _header(step: Mapping[str, Any], style: str, ledger: Mapping[str, Any] | None) -> str:
    sid = T.mark(T.STEP_ID.format(k=step["k"]))
    if style not in ("ledger", "ledger_flat"):
        return T.STEP_HEADER.format(sid=sid, k=step["k"])
    flags = step.get("flags") or {}
    bits = ""
    if flags.get("workspace_change"):
        bits += T.FLAG_TEXT["workspace_change"]
    outcome = flags.get("outcome")
    if outcome in ("tests_fail", "tests_pass"):
        t = (step.get("detail") or {}).get("outcome", {}).get("tests") or [{}]
        bits += T.OUTCOME_TEXT[outcome].format(tests=", ".join(f"{v} {k}" for k, v in t[-1].items()))
    elif outcome == "error":
        codes = [c for c in (step.get("detail") or {}).get("outcome", {}).get("exit_codes", []) if c not in (0, None)]
        bits += T.OUTCOME_TEXT["error"].format(codes=", ".join(map(str, codes)))
    if flags.get("output_current") in ("stale", "mixed"):
        bits += T.FLAG_TEXT[flags["output_current"]]
    if flags.get("noise"):
        bits += T.FLAG_TEXT["noise"]
    if style == "ledger_flat":
        return T.STEP_HEADER_LEDGER_FLAT.format(sid=sid, k=step["k"], flags=bits)
    aid = flags.get("approach")
    return T.STEP_HEADER_LEDGER.format(sid=sid, k=step["k"], aid=T.mark(aid) if aid else T.NONE, flags=bits)


def _approach_table(ledger: Mapping[str, Any]) -> list[str]:
    items = {i["id"]: i for i in ledger.get("items", [])}
    rows = [T.APPROACHES_HEADER]
    last = ledger["approaches"][-1]["id"] if ledger.get("approaches") else None
    for a in ledger.get("approaches", []):
        start = items.get(a.get("start_item"))
        fq = a.get("failure_evidence")
        rows.append(T.APPROACH_ROW.format(
            aid=T.mark(a["id"]), first=a["first_step"], last=a["last_step"], status=a.get("status") or "status unknown",
            current=T.CURRENT if a["id"] == last else "",
            start=T.APPROACH_START.format(iid=T.mark(start["id"]), quote=start["quote"]) if start else T.NONE,
            failure=T.APPROACH_FAILURE.format(quote=fq["quote"], k=fq["step"]) if fq else T.NONE))
    return rows


def render_review_view(ledger: Mapping[str, Any], kind: str = "ledger") -> dict[str, Any]:
    """Review view for arm 2 (``kind="ledger"``) or arm 3 (``kind="context"``) in the brief-interfaces format.

    ``kind="ledger_flat"`` is the flat Ledger of arm set v3 (arms 2 and 5): the same steps, flags and tagged items,
    with no approach layer (no approach index table, no approach marks or units, no per-approach status, no rewind
    point in the legend) and facts without the survives-rewind status."""
    if kind not in ("ledger", "ledger_flat", "context"):
        raise ValueError(kind)
    flat = kind == "ledger_flat"
    items = ledger.get("items", []) if kind in ("ledger", "ledger_flat") else []
    by_step: dict[int, list] = defaultdict(list)
    for i in items:
        by_step[i["step"]].append(i)
    rewind = (ledger.get("boundary") or {}).get("rewind_point")
    legend = {"ledger": T.LEGEND_LEDGER.format(rewind_point=rewind), "ledger_flat": T.LEGEND_LEDGER_FLAT,
              "context": T.LEGEND_BASE}[kind]
    lines = [legend, ""]
    units: dict[str, dict[str, Any]] = {}
    if kind == "ledger":
        lines += _approach_table(ledger) + [""]
        for a in ledger.get("approaches", []):
            units[a["id"]] = {"kind": "approach", "steps": [a["first_step"], a["last_step"]],
                              "label": a.get("status")}
    lines += [T.TASK_HEADER] + [f["text"] for f in ledger["head"] if f["kind"] != "system"] + ["", T.CONTEXT_HEADER]
    for step in ledger["steps"]:
        k = step["k"]
        lines += ["", _header(step, kind, ledger)] + step_lines(step, style=kind, items=by_step.get(k, ()))
        units[T.STEP_ID.format(k=k)] = {"kind": "step", "steps": [k, k]}
        if any(f["kind"] == "output" for f in step["fields"]):
            units[T.OUTPUT_ID.format(k=k)] = {"kind": "output", "steps": [k, k]}
    for i in items:
        units[i["id"]] = {"kind": "item", "steps": [i["step"], i["step"]], "quote": i["quote"],
                          "label": item_label(i, flat=flat)}
    return {"version": RENDER_VERSION, "templates_version": T.FLAT_TEMPLATES_VERSION if flat else T.TEMPLATES_VERSION,
            "kind": kind,
            "run_id": ledger["run_id"], "checkpoint_id": ledger["checkpoint_id"], "stall_point": ledger["stall_point"],
            "fake_labels": bool((ledger.get("labeller") or {}).get("fake_labels")),
            "text": "\n".join(lines).rstrip() + "\n", "units": units}


def label_text(ledger: Mapping[str, Any], unit_id: str, *, flat: bool = False) -> str:
    """Hand-back text of one kept unit: items keep their label; steps and outputs come back as plain text.
    ``flat``: the label as the flat Ledger shows it (no approach units exist there)."""
    kind, num = unit_id[0], int(unit_id[1:])
    if kind == "A" and flat:
        raise KeyError(f"the flat Ledger has no approach units ({unit_id})")
    if kind == "I":
        item = next(i for i in ledger["items"] if i["id"] == unit_id)
        label = item_label(item, flat=flat)
        name, _, status = label.partition(" · ")
        return T.KEPT_ITEM.format(label=name, k=item["step"], status=f", {status}" if status else "", quote=item["quote"])
    if kind in ("S", "O"):
        step = ledger["steps"][num - 1]
        if kind == "S":
            return T.KEPT_STEP.format(k=num, text="\n".join(step_lines(step, style="plain")))
        outs = [f for f in step["fields"] if f["kind"] == "output"]
        return T.KEPT_OUTPUT.format(k=num, text="\n".join(f["text"] for f in outs))
    if kind == "A":
        a = next(x for x in ledger["approaches"] if x["id"] == unit_id)
        start = next((i for i in ledger["items"] if i["id"] == a.get("start_item")), None)
        fq = a.get("failure_evidence")
        return T.KEPT_APPROACH.format(first=a["first_step"], last=a["last_step"], status=a.get("status") or "unknown",
                                      start=start["quote"] if start else T.NONE,
                                      failure=T.APPROACH_FAILURE.format(quote=fq["quote"], k=fq["step"]) if fq else T.NONE)
    raise KeyError(unit_id)


def recipe_entries(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ordered recipe entries (before the cap): failure evidence, surviving facts from undone steps, never-tried ideas."""
    rewind = ledger["boundary"]["rewind_point"]
    current = ledger["approaches"][-1] if ledger.get("approaches") else {}
    out = []
    fq = current.get("failure_evidence")
    if fq:
        out.append({"kind": "failure", "id": current["id"], "section": None,
                    "line": T.RECIPE_FAILURE.format(quote=fq["quote"], k=fq["step"])})
    for i in ledger.get("items", []):
        if i["type"] == "fact" and i["step"] > rewind and i.get("survives_rewind") is True:
            out.append({"kind": "fact", "id": i["id"], "section": T.RECIPE_FACTS_HEADER,
                        "line": T.RECIPE_FACT.format(quote=i["quote"], k=i["step"])})
    for i in ledger.get("items", []):
        if i["type"] == "idea" and i.get("idea_status") == "never_tried":
            out.append({"kind": "idea", "id": i["id"], "section": T.RECIPE_IDEAS_HEADER,
                        "line": T.RECIPE_IDEA.format(quote=i["quote"], k=i["step"])})
    return out


def _recipe_text(entries: Sequence[Mapping[str, Any]]) -> str:
    lines, section = [], None
    for e in entries:
        if e["section"] and e["section"] != section:
            lines.append(e["section"])
        section = e["section"]
        lines.append(e["line"])
    return "\n".join(lines)


def render_recipe(ledger: Mapping[str, Any], *, cap_tokens: int) -> dict[str, Any]:
    """Arm 6 recipe text capped at ``cap_tokens``: whole entries are cut from the end; a lone first entry that is
    still over the cap is cut to it. Every cut is logged."""
    entries = recipe_entries(ledger)
    kept = list(entries)
    cut_log = []
    while kept and T.count_tokens(_recipe_text(kept)) > cap_tokens and len(kept) > 1:
        e = kept.pop()
        cut_log.append({"kind": e["kind"], "id": e["id"], "tokens": T.count_tokens(e["line"]), "reason": "over_cap"})
    text = _recipe_text(kept)
    if kept and T.count_tokens(text) > cap_tokens:
        before = T.count_tokens(text)
        text = T.truncate_tokens(text, cap_tokens)
        cut_log.append({"kind": kept[0]["kind"], "id": kept[0]["id"], "tokens": before - T.count_tokens(text),
                        "reason": "truncated_to_cap"})
    return {"version": RENDER_VERSION, "templates_version": T.TEMPLATES_VERSION, "text": text,
            "cap_tokens": cap_tokens, "cut_log": cut_log, "tokens": T.count_tokens(text), "tokenizer": T.TOKENIZER,
            "entries_before_cap": [{"kind": e["kind"], "id": e["id"]} for e in entries],
            "fake_labels": bool((ledger.get("labeller") or {}).get("fake_labels"))}


def labeller_text(steps: Sequence[Mapping[str, Any]], head: Sequence[Mapping[str, Any]], *,
                  target: tuple[int, int], earlier: bool = True, later: bool = False,
                  elide_later_outputs: bool = False, elide_outputs: bool = False) -> str:
    """The labeller's input: task, then earlier steps (agent text only), target steps (full unless
    ``elide_outputs``), later steps (outputs elided with ``elide_later_outputs``)."""
    lo, hi = target
    lines = ["TASK"] + [f["text"] for f in head if f["kind"] != "system"] + [""]
    sections = []
    if earlier and lo > 1:
        sections.append(("earlier", [s for s in steps if s["k"] < lo], True))
    sections.append(("target" if (lo > 1 or later) else "all", [s for s in steps if lo <= s["k"] <= hi],
                     elide_outputs))
    if later and hi < len(steps):
        sections.append(("later", [s for s in steps if s["k"] > hi], elide_later_outputs))
    for name, chunk, elide in sections:
        lines += [T.LABELLER_SECTIONS[name], ""]
        for s in chunk:
            lines += [T.LABELLER_STEP_HEADER.format(k=s["k"])] + step_lines(s, style="labeller", elide_outputs=elide) + [""]
    return "\n".join(lines).rstrip() + "\n"


def labeller_text_v2(steps: Sequence[Mapping[str, Any]], head: Sequence[Mapping[str, Any]], *,
                     target: tuple[int, int], earlier: bool = True, later: bool = False,
                     elide_later_outputs: bool = False, elide_outputs: bool = False) -> str:
    """The labeller's input, render v2 (protocol v2.1): the same sections, steps and text as ``labeller_text``, but
    each step sits between ``<step k>`` and ``</step k>`` lines and every part header carries ``[step k]``."""
    lo, hi = target
    lines = ["TASK"] + [f["text"] for f in head if f["kind"] != "system"] + [""]
    sections = []
    if earlier and lo > 1:
        sections.append(("earlier", [s for s in steps if s["k"] < lo], True))
    sections.append(("target" if (lo > 1 or later) else "all", [s for s in steps if lo <= s["k"] <= hi],
                     elide_outputs))
    if later and hi < len(steps):
        sections.append(("later", [s for s in steps if s["k"] > hi], elide_later_outputs))
    for name, chunk, elide in sections:
        lines += [T.LABELLER_SECTIONS[name], ""]
        for s in chunk:
            k = s["k"]
            lines += ([T.LABELLER_V2_STEP_OPEN.format(k=k)]
                      + step_lines(s, style="labeller", elide_outputs=elide,
                                   part_prefix=T.LABELLER_V2_PART_PREFIX.format(k=k))
                      + [T.LABELLER_V2_STEP_CLOSE.format(k=k), ""])
    return "\n".join(lines).rstrip() + "\n"


LABELLER_RENDERERS = {"v1": labeller_text, "v2": labeller_text_v2}
