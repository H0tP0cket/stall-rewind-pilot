"""T2.2 label checks: blind reference-labelling exports and their scoring.

(i) Role sample: worker-prose sentences stratified by Jev main role
    (``unknown`` included), each with its local context, exported without any
    Jev label. Scored with Cohen's kappa and a confusion matrix.
(ii) Link windows: about 30 consecutive prose and tool-output items ending at
    each S checkpoint's cut (same session, available at the cut), with every
    ordered (later, earlier) pair listed, where the later item is worker prose
    (only prose can state a relation) not written before the earlier one.
    Reference labellers mark the real explicit links; unmarked pairs in a
    reviewed window are negatives. Scored
    by precision and by recall split into misses that were asked but fell
    below threshold and misses that were never asked (not a candidate).

Rules: annotator files never contain predictions, probabilities, item ids
or grades; predictions go to a separate key file. Agreement against a
model-simulated annotator is a pipeline test, not validation.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from ...ledger.text import cohen_kappa  # noqa: F401
from .vocab22 import LINK_TIER_EXPLICIT, RELATION_OPTIONS, ROLE_OPTIONS

LABELCHECK22_VERSION = "t2.2-labelcheck/v1"
PROSE = {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}
SOURCE_NAMES = {"worker_text": "agent message", "worker_reasoning": "agent reasoning (replayed)",
                "provider_summary": "provider summary of agent reasoning",
                "worker_code_comment": "comment in agent command", "observation": "tool output"}
NEGATIVE_RELATIONS = {"no_explicit_relation", "ambiguous", "", None}
UNSURE_ROLES = {"", None, "unsure", "ambiguous"}
WINDOW = 30
CONTEXT_BEFORE, CONTEXT_AFTER = 400, 200


def _order(item) -> tuple:
    return (item.available_seq, item.seq, item.order)


def _context(item, items_by_field: Mapping[tuple, list], decoded: Mapping[tuple[int, str], str] | None) -> tuple[str, str]:
    text = (decoded or {}).get((item.seq, item.span.field))
    if text is not None and text[item.span.start:item.span.end] == item.text:
        return text[max(0, item.span.start - CONTEXT_BEFORE):item.span.start], text[item.span.end:item.span.end + CONTEXT_AFTER]
    same = items_by_field.get((item.seq, item.span.field), [])
    k = next((n for n, i in enumerate(same) if i.item_id == item.item_id), 0)
    return " ".join(i.text for i in same[max(0, k - 2):k]), " ".join(i.text for i in same[k + 1:k + 2])


def role_sample(items: Sequence, *, per_role: int = 8, seed: int = 20260926, role_label: str = "role",
                decoded: Mapping[tuple[int, str], str] | None = None) -> tuple[list[dict], dict[str, Any]]:
    """Blind rows and a key: up to ``per_role`` prose sentences per predicted main role, shuffled."""
    prose = [i for i in items if i.source_kind in PROSE and not i.task_copy]
    by_field: dict[tuple, list] = defaultdict(list)
    for i in sorted(prose, key=lambda i: (i.seq, i.span.field, i.order)):
        by_field[(i.seq, i.span.field)].append(i)
    strata: dict[str, list] = defaultdict(list)
    for i in prose:
        strata[i.label(role_label)].append(i)
    rng = random.Random(seed)
    chosen = []
    for role in sorted(strata):
        pool = sorted(strata[role], key=lambda i: i.item_id)
        chosen.extend(rng.sample(pool, min(per_role, len(pool))))
    rng.shuffle(chosen)
    rows, key = [], {}
    for n, item in enumerate(chosen):
        rid = f"r{n:03d}"
        before, after = _context(item, by_field, decoded)
        rows.append({"id": rid, "source": SOURCE_NAMES.get(item.source_kind, item.source_kind),
                     "context_before": before, "text": item.text, "context_after": after})
        lab = item.labels.get(role_label)
        key[rid] = {"item_id": item.item_id, "run_id": item.run_id, "predicted": item.label(role_label),
                    "probabilities": dict(lab.probabilities or {}) if lab is not None else None}
    return rows, {"version": LABELCHECK22_VERSION, "kind": "roles", "seed": seed, "per_role": per_role,
                  "strata": {r: len(v) for r, v in strata.items()}, "rows": key}


def link_windows(items: Sequence, links: Sequence, checkpoints: Sequence, *, asked_pairs: Collection[tuple[str, str]] = (),
                 window: int = WINDOW, commands: Mapping[str, str] | None = None) -> tuple[list[dict], dict[str, Any]]:
    """Blind windows ending at each S checkpoint, every ordered (later prose, earlier item) pair, and a key.

    Pairs follow window order (available_seq, seq, order), so pairs inside one response are listed too and
    flagged ``same_event`` in the key. ``asked_pairs`` are the (later_id, earlier_id) pairs pass B asked about;
    of ``links`` only explicit-wording links count as predictions."""
    explicit = {(l.source_id, l.target_id): l for l in links if l.tier == LINK_TIER_EXPLICIT}
    asked = set(asked_pairs)
    rows, windows, pairs_key = [], {}, {}
    for cp in checkpoints:
        if "S" not in set(cp.annotations):
            continue
        pool = sorted((i for i in items if i.session_id == cp.session_id and i.available_seq <= cp.event_sequence
                       and not i.task_copy and (i.source_kind in PROSE or i.source_kind == "observation")), key=_order)
        chosen = pool[-window:]
        wid = f"w{len(rows):03d}"
        refs = {i.item_id: f"{wid}.i{n:02d}" for n, i in enumerate(chosen)}
        pairs = []
        for b, later in enumerate(chosen):
            if later.source_kind not in PROSE:
                continue
            for earlier in chosen[:b]:
                if earlier.seq > later.seq:  # written after the later item (reasoning replayed late)
                    continue
                pid = f"{wid}.p{len(pairs):03d}"
                pairs.append({"pair_id": pid, "later": refs[later.item_id], "earlier": refs[earlier.item_id]})
                link = explicit.get((later.item_id, earlier.item_id))
                pairs_key[pid] = {"window": wid, "later_id": later.item_id, "earlier_id": earlier.item_id,
                                  "same_event": earlier.seq == later.seq,
                                  "asked": (later.item_id, earlier.item_id) in asked or link is not None,
                                  "predicted": link.relation if link is not None else None,
                                  "p": link.probability if link is not None else None}
        rows.append({"window_id": wid, "items": [
            {"ref": refs[i.item_id], "source": SOURCE_NAMES.get(i.source_kind, i.source_kind), "text": i.text,
             **({"command": (commands or {}).get(i.tool_call_id or "", "unknown")[:300]} if i.source_kind == "observation" else {})}
            for i in chosen], "pairs": pairs})
        windows[wid] = {"run_id": chosen[0].run_id if chosen else None, "checkpoint_id": cp.checkpoint_id,
                        "n_items": len(chosen), "n_pairs": len(pairs)}
    return rows, {"version": LABELCHECK22_VERSION, "kind": "links", "window": window, "windows": windows,
                  "pairs": pairs_key}


def export(dest: Path, key_path: Path, *, roles: tuple[list, dict] | None = None,
           windows: tuple[list, dict] | None = None) -> dict[str, int]:
    """Write annotator files (no predictions) under ``dest`` and the key to ``key_path`` (keep it outside ``dest``)."""
    dest, key_path = Path(dest), Path(key_path)
    if dest.resolve() in key_path.resolve().parents:
        raise ValueError("key file must not be inside the annotator folder")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "instructions.json").write_text(json.dumps({
        "version": LABELCHECK22_VERSION,
        "roles": {"task": "Choose the one main role the sentence `text` expresses, read in its context. Use 'unsure' "
                          "if you cannot tell.", "options": ROLE_OPTIONS},
        "links": {"task": "For each pair, mark a relation only if the wording of the later item itself explicitly "
                          "connects it to the earlier item. Leave unmarked otherwise. Mark every window you finish "
                          "as reviewed.", "options": RELATION_OPTIONS,
                  "annotation_lines": [{"id": "<pair_id>", "relation": "<option>"}, {"id": "<window_id>", "reviewed": True}]},
    }, indent=1))
    key: dict[str, Any] = {}
    counts = {}
    for name, part in (("roles", roles), ("links", windows)):
        if part is None:
            continue
        rows, part_key = part
        with (dest / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        key[name] = part_key
        counts[name] = len(rows)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=1))
    return counts


def score_roles(key: Mapping[str, Any], annotations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Kappa and confusion (reference -> predicted) over rows the reference labelled with a role."""
    ref = {a["id"]: a.get("role") for a in annotations if "role" in a}
    pairs = [(k["predicted"], ref[rid]) for rid, k in key["rows"].items() if ref.get(rid) not in UNSURE_ROLES]
    committed = [(p, r) for p, r in pairs if p != "unknown"]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, r in pairs:
        confusion[r][p] += 1
    return {"n": len(pairs), "reference_unsure": sum(1 for rid in key["rows"] if rid in ref and ref[rid] in UNSURE_ROLES),
            "kappa": cohen_kappa(pairs), "agreement": sum(p == r for p, r in pairs) / len(pairs) if pairs else None,
            "n_committed": len(committed), "kappa_committed": cohen_kappa(committed),
            "predicted_unknown": len(pairs) - len(committed),
            "confusion": {r: dict(v) for r, v in sorted(confusion.items())}}


def score_links(key: Mapping[str, Any], annotations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Precision and recall of explicit links inside reviewed windows; misses split by whether pass B asked."""
    marked = {a["id"]: a.get("relation") for a in annotations if "relation" in a}
    reviewed = {a["id"] for a in annotations if a.get("reviewed")} | {pid.split(".")[0] for pid in marked}
    tp = fp = 0
    fn = {"asked_below_threshold": 0, "never_asked": 0}
    type_agree = []
    for pid, k in key["pairs"].items():
        if k["window"] not in reviewed:
            continue
        ref_pos = marked.get(pid) not in NEGATIVE_RELATIONS
        pred_pos = k["predicted"] is not None
        if pred_pos and ref_pos:
            tp += 1
            type_agree.append(k["predicted"] == marked[pid])
        elif pred_pos:
            fp += 1
        elif ref_pos:
            fn["asked_below_threshold" if k["asked"] else "never_asked"] += 1
    positives = tp + sum(fn.values())
    return {"windows_reviewed": len(reviewed & set(key["windows"])),
            "tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / positives if positives else None,
            "missed_share_asked_below_threshold": fn["asked_below_threshold"] / positives if positives else None,
            "missed_share_never_asked": fn["never_asked"] / positives if positives else None,
            "relation_agreement_on_tp": sum(type_agree) / len(type_agree) if type_agree else None}
