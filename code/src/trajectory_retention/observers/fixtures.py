"""Development fixture: a synthetic recorded trajectory in a real ArtifactStore.

DEVELOPMENT ONLY. Nothing here is a natural mining run, and the checkpoint it
marks as the development S1 is an engineering checkpoint
(selection_kind="engineering_checkpoint"), never a naturally selected S1.

The fixture follows the event shapes confirmed with the execution session on
2026-09-15 (see observers/prefix.py) so observer code paths exercised here are
the ones real records will exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .tstore import entry_id_for
from ..contracts import (
    ArtifactStoreProtocol,
    Checkpoint,
    EventRecord,
    NativeState,
    RuntimeClass,
    Scope,
    Usage,
    Visibility,
)

RUN_ID = "devfix-etl-001"
TASK_ID = "dev-fixture-etl"


@dataclass(frozen=True)
class FixtureTrajectory:
    run_id: str
    scope: Scope
    checkpoints: tuple[Checkpoint, ...]
    dev_s1_checkpoint_id: str
    dev_donor_checkpoint_id: str
    event_seq_by_label: Mapping[str, int]


TASK_TEXT = (
    "Build a CLI program `etl.py` that reads an ETL pipeline specification from STDIN, "
    "validates and normalizes it, then outputs JSON. Error codes: SCHEMA_VALIDATION_FAILED, "
    "UNKNOWN_OP, MISSING_COLUMN, BAD_EXPR. Exit code 0 on success, 1 on error."
)
SYSTEM_TEXT = "You are a coding agent. Use the bash tool. Finish by writing the deliverable."


def _scope(stage: str = "checkpoint_1", session: str = "sess-1") -> Scope:
    return Scope(
        task_id=TASK_ID, stage_id=stage, target_id=f"{TASK_ID}:{stage}", session_id=session,
        detector_scope_id=f"{stage}:{session}", requirement_version="req-v1", evaluator_version="eval-v1",
    )


def build_fixture(store: ArtifactStoreProtocol, run_id: str = RUN_ID) -> FixtureTrajectory:
    scope = _scope()
    labels: dict[str, int] = {}

    def text_ref(text: str, visibility: Visibility = Visibility.WORKER_VISIBLE):
        return store.put_bytes(text.encode("utf-8"), media_type="text/plain", visibility=visibility)

    def add(kind: str, *, visibility: Visibility, label: str | None = None, **kw: Any) -> EventRecord:
        ev = store.append_event(EventRecord(run_id=run_id, kind=kind, scope=scope, visibility=visibility, **kw))
        if label:
            labels[label] = ev.sequence
        return ev

    def response(label: str, content: str | None, tool_calls: list[dict[str, Any]], call_id: str,
                 reasoning: str | None = None, generated: int = 300) -> EventRecord:
        body = {"content": content, "tool_calls": tool_calls, "reasoning_content": reasoning,
                "finish_reason": "tool_calls" if tool_calls else "stop"}
        return add("model_response", visibility=Visibility.WORKER_VISIBLE, label=label, call_id=call_id,
                   actor="worker", output_ref=store.put_json(body, visibility=Visibility.WORKER_VISIBLE),
                   usage=Usage(output_tokens=generated, generated_tokens=generated,
                               reasoning_accounting="included", measurement_basis="fixture"))

    def tool(label: str, call_id: str, command: str, raw: str, delivered: str | None = None) -> EventRecord:
        delivered = raw if delivered is None else delivered
        add("tool_call", visibility=Visibility.WORKER_VISIBLE, call_id=call_id, actor="worker",
            payload={"command": command})
        return add("tool_result", visibility=Visibility.WORKER_VISIBLE, label=label, call_id=call_id,
                   output_ref=text_ref(raw, Visibility.RESEARCH_PRIVATE), delivered_ref=text_ref(delivered),
                   payload={"returncode": 0, "duration_seconds": 0.1,
                            "raw_equals_delivered": raw == delivered, "exception_info": None,
                            "delivered_to_history": True, "delivery_basis": "synthetic_fixture_append"})

    def private(kind: str, label: str | None = None, **payload: Any) -> EventRecord:
        return add(kind, visibility=Visibility.RESEARCH_PRIVATE, label=label, payload=payload)

    state_ref = store.put_json({"fixture": "state"})
    checkpoints: list[Checkpoint] = []

    def checkpoint(ordinal: int, reasons: tuple[str, ...], cut: int, ws: str) -> Checkpoint:
        state = NativeState(workspace=store.put_json({"ws": ws}), history=state_ref, harness=state_ref,
                            runtime_class=RuntimeClass.IRRELEVANT)
        cp = Checkpoint(run_id=run_id, scope=scope, ordinal=ordinal, event_sequence=cut, state=state,
                        reasons=reasons, checkpoint_id=f"{run_id}:C{ordinal}", branchable=True,
                        parent_checkpoint_id=checkpoints[-1].checkpoint_id if checkpoints else None)
        store.save_checkpoint(cp)
        checkpoints.append(cp)
        return cp

    # --- C0: initial ----------------------------------------------------
    add("task_disclosure", visibility=Visibility.WORKER_VISIBLE, label="system", delivered_ref=text_ref(SYSTEM_TEXT),
        payload={"role": "system", "message_index": 0})
    add("task_disclosure", visibility=Visibility.WORKER_VISIBLE, label="task", delivered_ref=text_ref(TASK_TEXT),
        payload={"role": "user", "message_index": 1})
    private("grade", label="grade0", score=0.0)
    c0 = checkpoint(0, ("initial",), labels["grade0"], "empty")

    # --- turn 1: inspect, hypothesis ------------------------------------
    response("r1", "I'll start by checking the workspace. The spec says exit code 1 on error, so "
             "I think errors should go to stdout as JSON, not stderr.",
             [{"id": "call-1", "name": "bash", "arguments": {"command": "ls -la"}}], "call-1")
    tool("t1", "call-1", "ls -la", "total 0\n-rw-r--r-- 1 u u 0 README.md\n")
    private("mutation", files=[])
    # read-only: no grade, no checkpoint

    # --- turn 2: write file, grade, new best ----------------------------
    response("r2", "Plan: implement parse_steps() first, then expression validation. "
             "Maybe the rename mapping form needs special handling.",
             [{"id": "call-2", "name": "bash", "arguments": {"command": "cat > etl.py <<'EOF'\nimport sys, json\n"
                                                              "def parse_steps(steps):\n    return steps\nEOF"}}],
             "call-2")
    tool("t2", "call-2", "cat > etl.py", "")
    private("mutation", files=["etl.py"])
    private("grade", label="grade1", score=0.25)
    c1 = checkpoint(1, ("graded",), labels["grade1"], "v1")

    # --- turn 3: test run with truncated output -------------------------
    long_raw = "FAILED tests/test_norm.py::test_rename_mapping - KeyError: 'mapping'\n" + ("x" * 5000) + \
               "\n3 passed, 2 failed\n"
    delivered = long_raw[:200] + "\n<output truncated: 5000 more characters>\n"
    response("r3", "Let me run the public diagnostics.",
             [{"id": "call-3", "name": "bash", "arguments": {"command": "python public_diag.py"}}], "call-3")
    tool("t3", "call-3", "python public_diag.py", long_raw, delivered)
    private("mutation", files=[])

    # --- turn 4: edit, grade flat ----------------------------------------
    response("r4", "The rename mapping test fails with KeyError: 'mapping'. I need to handle both the "
             "from/to form and the mapping form. I'll edit parse_steps.",
             [{"id": "call-4", "name": "bash", "arguments": {"command": "sed -i '' 's/return steps/return [normalize(s) for s in steps]/' etl.py"}}],
             "call-4")
    tool("t4", "call-4", "sed -i", "")
    private("mutation", files=["etl.py"])
    private("grade", label="grade2", score=0.25)
    c2 = checkpoint(2, ("graded",), labels["grade2"], "v2")

    # --- turn 5: edit again, regression, detector fires (dev S1) --------
    response("r5", "That did not fix it. I was wrong about the mapping form; it is not a KeyError in "
             "normalize but in the rename handler. Abandoning the sed approach; rewriting the handler.",
             [{"id": "call-5", "name": "bash", "arguments": {"command": "cat > etl.py <<'EOF'\n# rewrite\nEOF"}}],
             "call-5")
    tool("t5", "call-5", "cat > etl.py", "")
    private("mutation", files=["etl.py"])
    private("grade", label="grade3", score=0.10)
    private("detector", label="detector3", fired=True)
    c3 = checkpoint(3, ("graded", "warning"), labels["detector3"], "v3")

    # --- later events that must never be visible to observers at C3 -----
    response("r6", "Found it: the rename handler must accept mapping.",
             [{"id": "call-6", "name": "bash", "arguments": {"command": "echo fixed"}}], "call-6")
    tool("t6", "call-6", "echo fixed", "fixed\n")
    private("mutation", files=["etl.py"])
    private("grade", label="grade4", score=0.6)
    c4 = checkpoint(4, ("graded",), labels["grade4"], "v4")

    return FixtureTrajectory(
        run_id=run_id, scope=scope, checkpoints=tuple(checkpoints),
        dev_s1_checkpoint_id=c3.checkpoint_id, dev_donor_checkpoint_id=c1.checkpoint_id,
        event_seq_by_label=labels,
    )


def scripted_update_ops(fx: FixtureTrajectory) -> dict[int, list[dict[str, Any]]]:
    """Deterministic extractor 'responses' per update index, exercising each validation path.

    tmp_ids are scoped to one update; later updates cite the persisted entry_id
    exactly as a real extractor would after reading the prior T.
    """
    s = fx.event_seq_by_label
    hyp_mapping_id = entry_id_for(fx.run_id, 1, 2)  # third accepted add of update 1
    return {
        0: [
            {"op": "add", "type": "verified_fact", "source_category": "observation_direct",
             "scope": "task_requirement", "tmp_id": "req_exit",
             "text": "Exit code 0 on success, 1 on error.",
             "quotes": [{"seq": s["task"], "field": "delivered_text", "quote": "Exit code 0 on success, 1 on error."}]},
        ],
        1: [
            {"op": "add", "type": "hypothesis", "source_category": "worker_expressed", "scope": "worker_strategy",
             "text": "I think errors should go to stdout as JSON, not stderr.",
             "quotes": [{"seq": s["r1"], "field": "assistant_text",
                         "quote": "I think errors should go to stdout as JSON, not stderr."}]},
            {"op": "add", "type": "plan_step", "source_category": "worker_expressed", "scope": "worker_strategy",
             "tmp_id": "plan1", "text": "implement parse_steps() first, then expression validation",
             "quotes": [{"seq": s["r2"], "field": "assistant_text",
                         "quote": "implement parse_steps() first, then expression validation"}]},
            {"op": "add", "type": "hypothesis", "source_category": "worker_expressed", "scope": "implementation_behavior",
             "tmp_id": "hyp_mapping", "text": "Maybe the rename mapping form needs special handling.",
             "quotes": [{"seq": s["r2"], "field": "assistant_text",
                         "quote": "Maybe the rename mapping form needs special handling."}]},
            {"op": "add", "type": "verified_fact", "source_category": "observation_direct",
             "scope": "implementation_behavior", "text": "README.md exists in the workspace",
             "quotes": [{"seq": s["t1"], "field": "delivered_content", "quote": "README.md"}]},
            # Paraphrase of a worker statement: must be rejected (not an exact quote).
            {"op": "add", "type": "hypothesis", "source_category": "worker_expressed", "scope": "worker_strategy",
             "text": "Errors are printed to stdout.",
             "quotes": [{"seq": s["r1"], "field": "assistant_text",
                         "quote": "I think errors should go to stdout as JSON, not stderr."}]},
            # Extractor's own diagnosis: diverted to advisor.
            {"op": "add", "type": "lesson", "source_category": "extractor_inference", "scope": "worker_strategy",
             "text": "The agent should write tests before editing.", "quotes": []},
        ],
        2: [
            {"op": "add", "type": "verified_fact", "source_category": "observation_direct",
             "scope": "implementation_behavior", "tmp_id": "fact_fail",
             "text": "test_rename_mapping FAILED with KeyError: 'mapping'",
             "quotes": [{"seq": s["t3"], "field": "delivered_content",
                         "quote": "FAILED tests/test_norm.py::test_rename_mapping - KeyError: 'mapping'"}]},
            # Cites content that was only in the raw (truncated-away) output: must be rejected.
            {"op": "add", "type": "verified_fact", "source_category": "observation_direct",
             "scope": "implementation_behavior", "text": "3 passed, 2 failed",
             "quotes": [{"seq": s["t3"], "field": "delivered_content", "quote": "3 passed, 2 failed"}]},
            {"op": "add", "type": "plan_step", "source_category": "worker_expressed", "scope": "worker_strategy",
             "text": "I need to handle both the from/to form and the mapping form.",
             "quotes": [{"seq": s["r4"], "field": "assistant_text",
                         "quote": "I need to handle both the from/to form and the mapping form."}]},
        ],
        3: [
            {"op": "set_status", "entry_id": hyp_mapping_id, "status": "worker_rejected",
             "quotes": [{"seq": s["r5"], "field": "assistant_text", "quote": "I was wrong about the mapping form"}]},
            {"op": "add", "type": "hypothesis", "source_category": "worker_expressed", "scope": "implementation_behavior",
             "tmp_id": "hyp2", "text": "it is not a KeyError in normalize but in the rename handler",
             "quotes": [{"seq": s["r5"], "field": "assistant_text",
                         "quote": "it is not a KeyError in normalize but in the rename handler"}]},
            {"op": "supersede", "entry_id": hyp_mapping_id, "by": "hyp2",
             "quotes": [{"seq": s["r5"], "field": "assistant_text", "quote": "I was wrong about the mapping form"}]},
            {"op": "add", "type": "lesson", "source_category": "worker_expressed", "scope": "worker_strategy",
             "text": "Abandoning the sed approach; rewriting the handler.",
             "quotes": [{"seq": s["r5"], "field": "assistant_text",
                         "quote": "Abandoning the sed approach; rewriting the handler."}]},
            # Future leak attempt: cites r6, which is after the C3 cut. Must be rejected.
            {"op": "add", "type": "verified_fact", "source_category": "worker_expressed", "scope": "implementation_behavior",
             "text": "Found it: the rename handler must accept mapping.",
             "quotes": [{"seq": s["r6"], "field": "assistant_text",
                         "quote": "Found it: the rename handler must accept mapping."}]},
        ],
    }
