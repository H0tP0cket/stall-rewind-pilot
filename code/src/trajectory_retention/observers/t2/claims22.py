"""T2.2 claim vs grader: analysis-only comparison of the worker's result claims with its private grades.

For each item whose main role is ``result_claim`` this records the latest
valid private grade at or before the item's event (checkpoint id, score,
full_solve) and a verdict: ``grader_agrees`` (full solve), ``grader_disagrees``
(not solved) or ``no_grade``.

Rules:

* Analysis only. Rows are recorded under kind ``t22_claim_grader`` in the
  analysis output store, never in T, never in a packet, never shown to a
  worker or used by any T2.2 selection.
* A grade counts when its grade event is valid, carries a full_solve value,
  belongs to the same stage as the claim (when both are known), and its
  checkpoint cut is at or before the claim's event.
* The grade measures the state at its checkpoint. Recorded workspace or
  runtime mutations between that checkpoint and the claim are counted, so a
  verdict against a stale grade is visible. The next grade after the claim is
  recorded as context; it does not change the verdict.
"""

from __future__ import annotations

from typing import Any, Sequence

from .leak22 import _cache

CLAIMS22_VERSION = "t2.2-claims/v1"
RECORD_KIND = "t22_claim_grader"
CLAIM_ROLE = "result_claim"
VERDICTS = ("grader_agrees", "grader_disagrees", "no_grade")


def valid_grades(source_store, run_id: str) -> list[dict[str, Any]]:
    """Valid private grades in checkpoint order, from the run's grade events joined to their checkpoints."""
    checkpoints = {cp.checkpoint_id: cp for cp in source_store.list_checkpoints(run_id)}
    grades = []
    for e in source_store.iter_events(run_id):
        p = e.payload or {}
        if e.kind != "grade" or p.get("valid") is not True or p.get("full_solve") is None:
            continue
        cp = checkpoints.get(p.get("checkpoint_id"))
        if cp is None:
            continue
        grades.append({"checkpoint_id": cp.checkpoint_id, "checkpoint_seq": cp.event_sequence, "grade_seq": e.sequence,
                       "evaluation_id": p.get("evaluation_id"), "stage_id": p.get("stage_id") or cp.scope.stage_id,
                       "score": p.get("score"), "full_solve": p.get("full_solve")})
    return sorted(grades, key=lambda g: (g["checkpoint_seq"], g["grade_seq"]))


def _state_changes(events, after_seq: int, through_seq: int) -> int:
    n = 0
    for e in events:
        if e.kind != "mutation" or not after_seq < e.sequence <= through_seq:
            continue
        p = e.payload or {}
        ws = [k for k in p.get("workspace_fidelity_paths_changed") or () if not _cache(k)]
        n += bool(ws or p.get("files") or p.get("runtime_paths_changed"))
    return n


def claim_vs_grader(source_store, run_id: str, items: Sequence, *, role_label: str = "role",
                    out_store=None) -> list[dict[str, Any]]:
    grades = valid_grades(source_store, run_id)
    mutations = [e for e in source_store.iter_events(run_id) if e.kind == "mutation"]
    rows = []
    for item in sorted(items, key=lambda i: (i.seq, i.order)):
        if item.run_id != run_id or item.label(role_label) != CLAIM_ROLE:
            continue
        same_stage = [g for g in grades if not (item.stage_id and g["stage_id"]) or g["stage_id"] == item.stage_id]
        prior = [g for g in same_stage if g["checkpoint_seq"] <= item.seq and g["grade_seq"] <= item.seq]
        later = [g for g in same_stage if g not in prior]
        grade = prior[-1] if prior else None
        verdict = "no_grade" if grade is None else "grader_agrees" if grade["full_solve"] else "grader_disagrees"
        lab = item.labels.get(role_label)
        row = {"version": CLAIMS22_VERSION, "kind": RECORD_KIND, "analysis_only": True, "run_id": run_id,
               "item_id": item.item_id, "seq": item.seq, "stage_id": item.stage_id, "text": item.text,
               "role_p": (lab.probabilities or {}).get(CLAIM_ROLE) if lab is not None else None,
               "grade": grade, "verdict": verdict,
               "state_changes_since_grade": None if grade is None else _state_changes(mutations, grade["checkpoint_seq"], item.seq),
               "next_grade": later[0] if later else None}
        rows.append(row)
        if out_store is not None:
            out_store.put_record(RECORD_KIND, f"{CLAIMS22_VERSION}:{item.item_id}", row, run_id=run_id)
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS}
    stale = sum(1 for r in rows if r["state_changes_since_grade"])
    flips = sum(1 for r in rows if r["grade"] and r["next_grade"] and r["next_grade"]["full_solve"] != r["grade"]["full_solve"])
    return {"claims": len(rows), **counts, "grade_stale_at_claim": stale, "next_grade_differs": flips}
