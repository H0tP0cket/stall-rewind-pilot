"""Bind the fixed-unit Ledger to the existing context-edit interface.

Selectable units retain the approved operations: whole steps, whole step outputs,
and labelled reasoning/message items. Task, command, feedback and file-change IDs
are reference annotations, not new editing operations. Files change by rewind only.
"""
from __future__ import annotations

import json
from pathlib import Path
from .render import render_review_view, label_text
from .units import render_units_view, STATUS_LABEL


def build_views(doc: dict, out: Path) -> dict:
    if doc.get("status") != "ok":
        raise ValueError("review views require a complete fixed-unit Ledger with valid labels")
    ledger = {k: doc[k] for k in ("run_id", "checkpoint_id", "stall_point", "head", "steps", "labeller")}
    ledger.update(version="fixed-unit-arm-binding.v1", approaches=[], items=[])
    context = render_review_view(ledger, "context")
    flat = render_units_view(doc)
    flat["kind"] = "ledger_flat"
    flat["binding_version"] = ledger["version"]
    # Step/output selection has exactly the same meaning as in the plain-context arm.
    selectable = dict(context["units"])
    for k, units in doc["units"]["steps"].items():
        tags = doc["labels"][k].get("tags") or {}
        for unit in units:
            ts = tags.get(unit["id"], [])
            if unit["kind"] not in ("reasoning", "message") or not ts:
                continue
            labels = []
            for tag in ts:
                claim = doc["claims"].get(unit["id"]) if tag == "claim" else None
                labels.append(tag + (" · " + STATUS_LABEL[claim["status"]].format(s=claim["status_step"])
                                     if claim else ""))
            label = "; ".join(labels)
            selectable[unit["id"]] = {"kind": "item", "steps": [int(k), int(k)],
                                       "quote": unit["text"], "label": label,
                                       "kept_text": f"{label} (step {k}): {unit['text']}"}
    for uid, entry in selectable.items():
        if "kept_text" not in entry:
            entry["kept_text"] = label_text(ledger, uid, flat=True)
    flat["units"] = selectable
    flat["text"] = ("Selectable IDs: Sk selects step k; Ok selects all outputs of step k; labelled reasoning and "
                    "message IDs (Sk.rn / Sk.mn) select that exact item. Other IDs in the Ledger are references "
                    "only. Task text and submission records are protected. File changes happen only through "
                    "workspace/runtime rewind.\n\n" + flat["text"])
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("ledger.json", ledger), ("ledger-units.json", doc),
                        ("review_view_ledger_flat.json", flat), ("review_view_context.json", context)):
        path = out / name
        text = json.dumps(value, ensure_ascii=False, indent=1) + "\n"
        if path.exists() and path.read_text() != text:
            raise ValueError(f"refusing to replace different review evidence: {path}")
        path.write_text(text)
    return {"ledger": flat, "context": context}
