"""Natural mining runner: execute -> record -> capture -> grade isolated copy -> detect -> continue.

One engine serves natural mining, resumption from a recorded checkpoint and
recovery treatments. It follows the ratified V1 ordering: complete the valid
tool batch, capture the branchable boundary, grade its isolated copy, record
the result and detector decision, then permit the next model call. Detector
outputs are research records only and never reach the worker.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded

from ..accounting import CostLedger
from ..contracts import (Annotation, ArtifactRef, ArtifactStoreProtocol, Checkpoint, Evaluation, NativeState, RunConfig,
                         RunResult, RuntimeClass, RuntimeExport, Scope, StageSpec, TaskAdapter, TreatmentManifest, Usage,
                         Visibility, new_id, to_jsonable, worker_operational_settings)
from ..detector.core import Detector, DetectorConfig, LEGACY_RULE_VERSION, RULE_VERSION, LAP_RULE_VERSION, ATTEMPT_RULE_VERSION, LAP_RULE_VERSIONS, OBSERVABILITY_RULE_VERSIONS, STALL_CLOCK_RULE_VERSION, STALL_CLOCK_RULE_VERSIONS, STALL_CLOCK_V8_RULE_VERSION, SUBMISSION_VERDICT_RULE_VERSIONS, make_detector
# Rule versions that use the simple, task-agnostic input path (no declared relevance/signature policies).
SIMPLE_INPUT_RULE_VERSIONS = (LEGACY_RULE_VERSION, *STALL_CLOCK_RULE_VERSIONS)
from ..execution.agent import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE, ContextWindowExhausted, RecordingAgent, WorkerBudgetExhausted
from ..execution.context_limit import EXIT_STATUS as CONTEXT_EXIT_STATUS, ContextLimit
from ..provider_policy import unresolved_retry_rule
from ..experiments.submission_stop import validate as validate_submission_stop, progress as submission_progress
from ..execution.archives import extract_directory, pack_directory
from ..execution.environment import (CaptureError, DockerWorkerEnvironment, RestoreError, ToolTransportFailure, comparable,
                                     fidelity_manifest_from_tar, loop_manifest_from_fidelity, manifest_digest)
from ..execution.model import ModelAbort, _BaseModel, history_replay_issue
from ..execution.recorder import Recorder
from ..persistence import preserve_failure
from .input_policies import adapter_binding, resolve_input_policies
from .mutation_policy import evaluate_mutation
from .signatures import build_failure_signature
from .runtime_support import bind_runtime_support

MUTATION_RULE = ("mutation-relevance/v3: W full fidelity manifest changed OR R content manifest (fidelity entries of "
                 "docker-diff paths outside /workspace, plus deleted list) changed")
FAILURE_SIGNATURE_RULE = "failure-signature/v1: sha256 over sorted (check_id,status) of non-PASS checks; None when grade invalid"
FRESH_HISTORY_ORIENTATION = "The workspace may contain prior work. Inspect it as needed."
# mini-SWE-agent exit statuses -> normalized termination reasons handed to TaskAdapter.next_stage.
NORMALIZED_REASONS = {"Submitted": "submitted", "LimitsExceeded": "call_limit", "RepeatedFormatError": "worker_error",
                      "WorkerBudgetExhausted": "budget_exhausted", "WorkerCostLimit": "cost_limit",
                      "WorkerWallClockLimit": "wall_clock_limit", "ContextWindowExhausted": "context_window_exhausted"}
# The engine's step guard is a run-wide call allowance. It is not the
# adapter's optional stage_step_limit / stage_cost_limit release condition.
STAGE_FINISH_REASONS = ("Submitted",)
O_MASK_PLACEHOLDER = "[tool output withheld]"
# v2 study run policies, selected by RunConfig.metadata["v2_run_policy"]; absent keeps every earlier behaviour.
V2_NATURAL = "natural-v2/v1"
V2_BRANCH = "branch-v2/v1"
from .technical_resume import POLICY as V2_TECHNICAL_RESUME
V2_RULES = {
    V2_NATURAL: ("natural-v2/v1: a branchable checkpoint at every point (after each step, including any submission "
                 "feedback), ungraded unless a change, the stall clock or a submission grades it; per-action manifests "
                 "between the actions of a step; a submission verdict is processed before any token, cost or step "
                 "guard; a private full solve is only logged; no wall-clock cap"),
    V2_BRANCH: ("branch-v2/v1: no detector; W and R are captured and graded only at submissions and at the end (a "
                "branch that ends at a submission ends in that submission's graded state); the loop continues past "
                "submissions until a passing submission, 3 identical failing sets, the submission cap, or the "
                "branch-relative token, cost and step guards (intervention turns included, no refunds)"),
}
V2_RULES[V2_TECHNICAL_RESUME] = (V2_RULES[V2_NATURAL] +
    "; technical restart from the latest restorable pre-submission checkpoint; source protocol retained; "
    "only remaining original allowance; separate execution, not an independent mine or arm outcome")
V2_TURN_RULE = ("intervention-turn/v1: one-shot review or note turn of the recording agent at the source checkpoint "
                "(full source context plus the turn prompt; the decision tool only, or no tool); same guarded model "
                "class, request policy and per-attempt accounting; at most 2 calls (an invalid call gets one retry, "
                "then the turn yields nothing); its calls, tokens and cost count against the branch's own allowance; "
                "the turn never enters the continued context")
V3_TURN_RULE = ("intervention-turn/v2 (arm set v3): one-shot review turn of the recording agent at the source checkpoint "
                "whose request holds only the source context's system and task messages plus the turn prompt (the "
                "view); the decision tool only (arms 2, 3) or no tool (arms 5, 6); same guarded model class, request "
                "policy, per-attempt accounting and context-limit rule; at most 2 calls (an invalid call gets one "
                "retry, then the turn yields nothing); its calls, tokens and cost count against the branch's own "
                "allowance; its accepted response is kept in the continued context (branch_v2.KEPT_OUTPUT_RULE)")
V2_DELIVERY_RULE = ("branch-v2-delivery/v1: kept (edited) context, then the hand-back message, then any injected "
                    "steps; the serialized next request is validated before the first continuation call")
V3_DELIVERY_RULE = ("branch-v2-delivery/v2 (arm set v3): kept (edited) context, then the kept review output (when the "
                    "turn kept one), then the hand-back message; the serialized next request is validated before the "
                    "first continuation call")
V2_TOOL_STATE = {V2_NATURAL: "per_action_manifest", V2_BRANCH: "none", V2_TECHNICAL_RESUME: "per_action_manifest"}
MUTATION_RULE_V2 = ("mutation-relevance/v2-points: per-action manifest differences (W fidelity, R entries, deleted, "
                    "mounts) between consecutive actions, or the step endpoint difference against the previous point")


class TechnicalFailure(RuntimeError):
    pass


@dataclass
class _Session:
    """Live state for one worker session (one stage under native SCB lifecycle)."""
    stage: StageSpec
    scope: Scope
    env: DockerWorkerEnvironment
    agent: RecordingAgent
    w_manifest: dict[str, str]
    r_manifest: dict[str, Any]
    w_fidelity: dict[str, Any] = field(default_factory=dict)
    last_graded_checkpoint: Checkpoint | None = None
    last_graded_state: tuple[str, str] | None = None  # (w digest, r digest) at last grade
    last_evaluation: Evaluation | None = None
    last_evaluation_ref: ArtifactRef | None = None
    usage_cursor: int = 0
    mutation_source_events_seen: set[str] = field(default_factory=set)


@dataclass
class _Run:
    task: TaskAdapter
    config: RunConfig
    store: ArtifactStoreProtocol
    recorder: Recorder
    ledger: CostLedger
    detector: Detector
    scratch: Path
    checkpoints: list[Checkpoint] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    first_stall_id: str | None = None
    session: _Session | None = None
    stage_index: int = 0
    session_index: int = 0
    detector_seq: int = 0
    terminated: bool = False
    fixed_target: bool = False  # treatments: never advance to later native stages
    terminal_failure: str | None = None
    checkpoint_failure: BaseException | None = None
    transition_failure: BaseException | None = None
    baseline: Any = None
    native_exit_status: str | None = None
    termination: dict[str, Any] = field(default_factory=dict)
    mutation_policy: Any = None
    signature_policies: dict[str, Any] = field(default_factory=dict)
    checkpoint_published: Callable[[Checkpoint], None] | None = None
    # Native submission verdicts (first-failure criterion per submission) for
    # adapters declaring native_submission_protocol; restored on resume.
    native_submissions: list[dict[str, Any]] = field(default_factory=list)
    started_monotonic: float = field(default_factory=time.monotonic)
    v2: str | None = None  # V2_NATURAL / V2_BRANCH / None (earlier behaviour)
    detector_disabled: bool = False  # v2 branches run no detector
    branch_origin_event: Any = None  # the run's branch_started event, kept instead of rescanning the journal
    last_grade_event: Any = None
    submission_cap: int | None = None  # v2 branches only; counted from the intervention
    v2_branch_plan: Any = None  # v2 branches: the branch plan (experiments/branch_v2.py BranchPlan)
    v2_compiled: Any = None  # v2 branches: the compiled intervention (donors, edits, hand-back, splice, turn totals)
    v2_end_checkpoint: Any = None  # v2 branches: the submission checkpoint a branch ended at (its end state)
    context_limit: Any = None  # v2 runs: the bound context-limit rule (execution/context_limit.py), or None
    v2_stop_before_first_call: str | None = None  # v2 branches: the branch cannot send its first request
    capture_gap_pending: bool = False
    capture_gap_count: int = 0


NATIVE_SUBMISSION_RULE = ("native-submission/v1: on Submitted (natural runs only) grade the submitted W/R at a "
                          "native_submission checkpoint (reusing a grade taken after the submitting step), ask the "
                          "adapter for the native verdict, end on native success or on the adapter-declared number of "
                          "consecutive identical first failures, otherwise drop the exit marker, append the interrupted "
                          "batch's executed tool outputs (explicit not-executed results for any later call in it), then "
                          "only the native feedback string as an ordinary user message in the same session")

# Declared, versioned harness change to native-submission/v1 (user decision 2026-09-27, mb10 batch): the native
# feedback string is delivered inside a fixed wrapper, because a Luna audit found the bare grader string read as a user
# follow-up. Opt-in per run via RunConfig.metadata["native_submission_feedback_wrapper"]; absent = unchanged v1.
FEEDBACK_WRAPPER_VERSIONS = {"native-submission-feedback-wrapper/v1"}


def feedback_wrapper(config: RunConfig) -> dict[str, str] | None:
    spec = config.metadata.get("native_submission_feedback_wrapper")
    if spec is None:
        return None
    if not isinstance(spec, Mapping) or spec.get("version") not in FEEDBACK_WRAPPER_VERSIONS:
        raise ValueError("Unsupported native submission feedback wrapper")
    template = spec.get("template")
    if not isinstance(template, str) or template.count("{feedback}") != 1:
        raise ValueError("Feedback wrapper template must contain exactly one {feedback}")
    return {"version": spec["version"], "template": template,
            "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest()}


def _declared_detector_override(start: dict, config: RunConfig) -> bool:
    """Stall resampling (2026-09-27): a natural continuation may run a different LIVE detector rule when it declares
    metadata.detector_override {recorded, live}; only rule_version may differ, every threshold equals the source."""
    o = (config.metadata or {}).get("detector_override") or {}
    src = dict(start["config"].get("detector_settings") or {})
    new = dict(config.detector_settings or {})
    return (o.get("recorded") == src.get("rule_version") and o.get("live") == new.get("rule_version")
            and {k: v for k, v in src.items() if k != "rule_version"} == {k: v for k, v in new.items() if k != "rule_version"})


class MiningRunner:
    def __init__(self, *, model_factory: Callable[[RunConfig], _BaseModel], worker_image: str,
                 detector_config: DetectorConfig, scratch_root: Path, system_template: str = SYSTEM_TEMPLATE,
                 instance_template: str = INSTANCE_TEMPLATE, network: str = "bridge") -> None:
        self.model_factory = model_factory
        self.worker_image = worker_image
        self.detector_config = detector_config
        self.scratch_root = Path(scratch_root)
        self.system_template = system_template
        self.instance_template = instance_template
        self.network = network

    def _instance_template(self, config: RunConfig) -> str:
        choice = config.metadata.get("instance_template")
        if choice is not None:
            from ..experiments.v2_text import INSTANCE_TEMPLATES
            if (choice not in INSTANCE_TEMPLATES or config.metadata.get("verification_prompt") is not None
                    or self.instance_template != INSTANCE_TEMPLATE):
                raise ValueError("Unsupported instance template selection")
            return INSTANCE_TEMPLATES[choice]
        if config.metadata.get("verification_prompt") is not None:
            from ..execution.agent import VERIFICATION_INSTANCE_TEMPLATE, VERIFICATION_PROMPT_VERSION
            if config.metadata["verification_prompt"] != VERIFICATION_PROMPT_VERSION:
                raise ValueError("Unsupported verification prompt policy")
            return VERIFICATION_INSTANCE_TEMPLATE
        return self.instance_template

    def _v2_policy(self, task: TaskAdapter, config: RunConfig, resume: Checkpoint | None,
                   treatment: TreatmentManifest | None) -> str | None:
        """Validate a declared v2 run policy before any execution evidence exists."""
        policy = config.metadata.get("v2_run_policy")
        if policy is None:
            return None
        if policy not in V2_RULES:
            raise ValueError("Unsupported v2 run policy")
        from .capture_gaps import POLICY as CAPTURE_GAP_POLICY
        if config.metadata.get("capture_gap_policy") is not None and (
                config.metadata["capture_gap_policy"] != CAPTURE_GAP_POLICY or policy != V2_TECHNICAL_RESUME):
            raise ValueError("Background capture gaps require the explicit technical-continuation policy")
        if config.wall_clock_limit_seconds is not None:
            raise ValueError("v2 runs have no wall-clock cap; the per-command timeout stays")
        if self.detector_config.rule_version not in STALL_CLOCK_RULE_VERSIONS:
            raise ValueError("v2 runs use the stall-clock detector family")
        if getattr(task, "native_submission_protocol", None) is None:
            raise ValueError("v2 runs need a task-declared native submission protocol")
        if policy == V2_NATURAL and (resume is not None or treatment is not None):
            raise ValueError("natural-v2 runs start fresh")
        if policy == V2_TECHNICAL_RESUME and (resume is None or treatment is not None):
            raise ValueError("technical resumes restore an aligned checkpoint with no treatment")
        if policy == V2_BRANCH and (treatment is None or "branch_v2" not in treatment.metadata):
            raise ValueError("branch-v2 runs start from a compiled branch-v2 treatment")
        self._context_limit(config)  # a malformed binding is refused before any evidence
        unresolved_retry_rule(config.metadata.get("unresolved_retry"))  # likewise the charged-retry rule
        validate_submission_stop(config.metadata.get("submission_stop"))
        if config.metadata.get("context_limit") is None and not config.model.startswith("scripted/"):
            raise ValueError("v2 runs with a live worker need the context-limit rule bound to a confirmed context "
                             "window (metadata.context_limit; context_limit.window_tokens in configs/v2/settings.json)")
        return policy

    @staticmethod
    def _context_limit(config: RunConfig) -> ContextLimit | None:
        from ..experiments.v2_text import OUTPUT_PLACEHOLDER
        return ContextLimit.from_metadata(config.metadata.get("context_limit"), OUTPUT_PLACEHOLDER)

    @staticmethod
    def _context_anchor(store: ArtifactStoreProtocol, checkpoint: Checkpoint, limit: ContextLimit) -> dict | None:
        """The size anchor of a restored context: the latest worker request at or before the checkpoint whose provider
        prompt count is known (its local measure from its own record, or measured again from its stored body)."""
        requests, anchor = {}, None
        for e in store.iter_events(checkpoint.run_id, through_sequence=checkpoint.event_sequence):
            if e.kind == "model_request" and not (e.payload or {}).get("mode"):
                requests[e.event_id] = e
            elif (e.kind == "model_response" and not (e.payload or {}).get("mode") and e.usage is not None
                  and e.usage.input_tokens is not None and e.parent_event_id in requests):
                anchor = (requests[e.parent_event_id], e)
        if anchor is None:
            return None
        request, response = anchor
        local = ((request.payload or {}).get("context_limit") or {}).get("local_tokens")
        if local is None and request.input_ref is not None:
            local = limit.measure(json.loads(store.get_bytes(request.input_ref)))["local_tokens"]
        if not local:
            return None
        return {"source": "recorded", "run_id": checkpoint.run_id, "request_event_sequence": request.sequence,
                "response_event_sequence": response.sequence, "provider_prompt_tokens": response.usage.input_tokens,
                "local_tokens": local, "ratio": response.usage.input_tokens / local}

    def _bind_context_limit(self, run: _Run, agent: RecordingAgent, anchor_checkpoint: Checkpoint | None) -> None:
        agent.context_limit = run.context_limit
        if run.context_limit is None or anchor_checkpoint is None:
            return
        anchor = self._context_anchor(run.store, anchor_checkpoint, run.context_limit)
        if anchor is not None:
            agent.context_ratio, agent.context_anchor = anchor["ratio"], anchor

    # ================================================================== entry
    def run(self, task: TaskAdapter, config: RunConfig, *, store: ArtifactStoreProtocol,
            resume: Checkpoint | None = None, treatment: TreatmentManifest | None = None,
            checkpoint_published: Callable[[Checkpoint], None] | None = None, branch_plan: Any = None) -> RunResult:
        # A run identity names one physical execution, including interrupted
        # attempts. Campaign phase receipts provide resumption by reuse; calling
        # Runner again must never append another execution to the same history.
        if next(iter(store.iter_events(config.run_id)), None) is not None:
            raise ValueError("Run identity already has execution evidence; use a new run identity")
        from ..experiments.submission_feedback import delivery
        delivery(config, "", feedback_wrapper(config))  # reject incompatible policies before execution
        self._instance_template(config)
        v2 = self._v2_policy(task, config, resume, treatment)
        if v2 == V2_TECHNICAL_RESUME:
            from .technical_resume import validate
            validate(store, config, resume, treatment)
        if (v2 == V2_BRANCH) != (branch_plan is not None):
            raise ValueError("A branch plan is given exactly for branch-v2 runs")
        recorder = Recorder(store, config.run_id)
        claim = {"claim_id": new_id("execution_claim")}
        recorder.persist(lambda: store.put_record("run_claim", config.run_id, claim, run_id=config.run_id), evidence=claim)
        run_started_at, run_clock = time.time(), time.monotonic()
        if v2:  # a checkpoint at every point: identical snapshots without re-reading the whole journal each time
            from .ledger_cache import IncrementalCostLedger
            ledger = IncrementalCostLedger(store)
        else:
            ledger = CostLedger(store)  # type: ignore[arg-type]
        scratch = self.scratch_root / hashlib.sha256(config.run_id.encode()).hexdigest()
        scratch.mkdir(parents=True, exist_ok=False)
        detector = make_detector(self.detector_config, run_id=config.run_id)
        run = _Run(task=task, config=config, store=store, recorder=recorder, ledger=ledger, detector=detector,
                   scratch=scratch, checkpoint_published=checkpoint_published, v2=v2,
                   detector_disabled=v2 == V2_BRANCH, v2_branch_plan=branch_plan,
                   submission_cap=branch_plan.submission_cap if branch_plan is not None else None,
                   context_limit=self._context_limit(config) if v2 else None)
        record = task.describe()
        prompt_policy = config.metadata.get("verification_prompt")
        if prompt_policy is not None:
            from ..execution.agent import VERIFICATION_INSTANCE_TEMPLATE, VERIFICATION_PROMPT_VERSION
            if prompt_policy != VERIFICATION_PROMPT_VERSION:
                raise ValueError("Unsupported verification prompt policy")
            if self.instance_template != INSTANCE_TEMPLATE:
                raise ValueError("Verification prompt requires the shared native-task template")
        recorder.emit("lifecycle", payload={"transition": "run_started", "task": to_jsonable(record), "config": to_jsonable(config),
                                            "detector_config": self.detector_config.to_record(), "worker_image": self.worker_image,
                                            "mutation_rule": MUTATION_RULE, "failure_signature_rule": FAILURE_SIGNATURE_RULE,
                                            "resume_from": resume.checkpoint_id if resume else None,
                                            "treatment": to_jsonable(treatment) if treatment else None,
                                            **({"v2_run_policy": v2, "v2_rule": V2_RULES[v2]} if v2 else {})})
        status, reason = "completed", "unknown"
        try:
            if self.detector_config.rule_version not in SIMPLE_INPUT_RULE_VERSIONS:
                run.mutation_policy, run.signature_policies = resolve_input_policies(task, record)
                recorder.emit("lifecycle", payload={"transition": "detector_input_policies_resolved",
                    "adapter": adapter_binding(task, record), "mutation": to_jsonable(run.mutation_policy),
                    "failure_signatures": to_jsonable(run.signature_policies),
                    "missing_mutation_policy": run.mutation_policy is None,
                    "missing_signature_policy": "lossless available detail; status-only identity unavailable unless declared"})
            if len(record.stages) > 1:
                self._lifecycle_operations(record.lifecycle.stage_operations)
            if config.detector_settings and DetectorConfig(**config.detector_settings) != self.detector_config:
                raise TechnicalFailure("Declared detector settings differ from the active detector")
            if config.tool_timeout_seconds <= 0:
                raise TechnicalFailure("Tool timeout must be positive")
            if resume is None and treatment is None:
                stage = task.stage(config.stage_id) if config.stage_id else record.stages[0]
                self._start_natural(run, stage)
            else:
                self._start_from_checkpoint(run, resume, treatment)
            reason = self._mine_v2(run) if run.v2 else self._mine(run)
            if reason == "accounting_interruption":
                status = "accounting_interrupted"
        except CaptureError as e:
            recorder.capture_failure = e
            status, reason = "technical_failure", f"capture_failure:{e}"
            try:
                recorder.incident("capture_failure", error=str(e), traceback=traceback.format_exc())
            except BaseException as cleanup:
                e.add_note(f"Capture failure incident persistence also failed: {type(cleanup).__name__}")
                raise e
            if recorder.persistence_failure is not None or hasattr(e, "capture_diagnostic_persistence_failure"):
                raise  # Preserve the original capture error and its in-memory diagnostics.
        except TechnicalFailure as e:
            status, reason = "technical_failure", f"technical_failure:{e}"
            try:
                recorder.incident("technical_failure", error=str(e), traceback=traceback.format_exc())
            except BaseException as cleanup:
                e.add_note(f"Failure incident persistence also failed: {type(cleanup).__name__}")
                raise e
            if hasattr(e, "persistence_evidence"):
                raise
        except Exception as e:  # unexpected: preserve evidence, never fabricate a checkpoint
            status, reason = "technical_failure", f"unexpected:{type(e).__name__}:{e}"
            try:
                recorder.incident("unexpected_exception", error=str(e), traceback=traceback.format_exc())
            except BaseException as cleanup:
                e.add_note(f"Failure incident persistence also failed: {type(cleanup).__name__}")
                raise e
            if hasattr(e, "persistence_evidence"):
                # The caller needs the recoverable in-memory receipt. Do not
                # turn persistent storage failure into a returned completion.
                raise
        except BaseException as e:
            # Cancellation still propagates to the phase journal, leaving its
            # external effects uncertain. The terminal event must not call an
            # interrupted execution completed just because no RunResult follows.
            status, reason = "interrupted", f"interrupted:{type(e).__name__}"
            try:
                recorder.incident("run_interrupted", error_type=type(e).__name__)
            except BaseException as cleanup:
                e.add_note(f"Interruption incident persistence also failed: {type(cleanup).__name__}")
            raise
        finally:
            primary = sys.exception()
            if primary is not None and hasattr(primary, "persistence_evidence"):
                recorder.persistence_failure = primary
            cleanup_error = None
            try:
                self._terminal(run, status, reason)
            except BaseException as exc:
                cleanup_error = exc
            if run.session is not None:
                try:
                    self._timed_call(run, "environment_close", run.session.env.close, role="infrastructure")
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        cleanup_error.add_note(f"Environment cleanup also failed: {type(exc).__name__}")
            if cleanup_error is not None:
                if primary is None:
                    raise cleanup_error
                primary.add_note(f"Terminal/close cleanup also failed: {type(cleanup_error).__name__}")
        pre_terminal_reason = reason
        if run.terminal_failure is not None:
            status, reason = "technical_failure", f"terminal_recording_failure:{run.terminal_failure}"
        normalized = run.termination.get("normalized_reason", reason)
        if normalized not in {"technical_failure", "worker_error"}:
            reason = normalized
        acc = ledger.snapshot(config.run_id)
        result = RunResult(run_id=config.run_id, status=status, termination_reason=reason,
                           checkpoint_ids=tuple(c.checkpoint_id for c in run.checkpoints),
                           first_stall_id=run.first_stall_id,
                           final_evaluation_ref=run.session.last_evaluation_ref if run.session else None,
                           accounting_ref=acc,
                           details={"annotations": [a.annotation_id for a in run.annotations],
                                    "tool_execution_uncertain": bool(run.session and run.session.agent.tool_execution_uncertain),
                                    "pre_terminal_reason": pre_terminal_reason,
                                    "terminal_recording_failure": run.terminal_failure,
                                    "termination": run.termination,
                                    "worker_allowance": {"assigned_generated_tokens": config.generated_token_limit,
                                        "known_generated_tokens_this_run": (run.session.agent.known_generated_tokens -
                                            run.session.agent.generated_token_offset) if run.session else 0,
                                        "unknown_usage_calls_this_run": (run.session.agent.unknown_usage_calls -
                                            run.session.agent.unknown_usage_offset) if run.session else 0,
                                        "exact_accounting": not run.session.agent.accounting_interrupted() if run.session else True},
                                    "stages_visited": run.stage_index + 1, "sessions": run.session_index,
                                    "started_at": run_started_at, "finished_at": time.time(),
                                    "elapsed_seconds": time.monotonic() - run_clock,
                                    "elapsed_scope": "whole worker run including grading, capture and environment overhead"})
        recorder.persist(lambda: store.put_record("run_result", config.run_id, result, run_id=config.run_id), evidence=result)
        return result

    # ============================================================= start paths
    def _timed_call(self, run: _Run, name: str, action: Callable, *, role: str = "restore") -> Any:
        start, status = time.monotonic(), "error"
        try:
            result = action()
            status = "ok"
            return result
        finally:
            usage = Usage(elapsed_seconds=time.monotonic() - start, measurement_basis="measured_local_operation")
            primary = sys.exception()
            try:
                event = run.recorder.emit("operation", usage=usage, payload={"name": name, "status": status})
                run.ledger.record(run_id=run.config.run_id, operation_id=event.event_id,
                    role=role, kind="restore" if role == "restore" else "infrastructure", usage=usage,
                    status=status, source_event_id=event.event_id, source_event_sequence=event.sequence)
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"Operation timing persistence also failed: {type(cleanup).__name__}")

    def _new_env(self, run: _Run) -> DockerWorkerEnvironment:
        env = None
        def create():
            nonlocal env
            # Opt-in, benchmark-native container settings declared by the task (user, dropped capabilities,
            # resources). Tasks that declare none get the unchanged defaults.
            options = dict(getattr(run.task, "worker_environment", None) or {})
            factory = getattr(run.task, "create_worker_environment", DockerWorkerEnvironment)
            env = factory(image=self.worker_image, timeout=run.config.tool_timeout_seconds, network=self.network,
                          **options)
            return env
        try:
            return self._timed_call(run, "environment_create", create, role="infrastructure")
        except BaseException as e:
            # Construction may have returned a live container before its timing
            # receipt failed. Ownership has not yet transferred to a session.
            if env is not None:
                self._close_after_failure(run, env, e)
            if not isinstance(e, Exception) or hasattr(e, "persistence_evidence"):
                raise
            raise TechnicalFailure(f"environment start failed: {e}") from e

    def _close_after_failure(self, run: _Run, env: DockerWorkerEnvironment, primary: BaseException) -> None:
        try:
            self._timed_call(run, "environment_close", env.close, role="infrastructure")
        except BaseException as cleanup:
            primary.add_note(f"Environment cleanup also failed: {type(cleanup).__name__}")
            if hasattr(cleanup, "persistence_evidence"):
                preserve_failure(primary, {"environment_id": env.container_id,
                    "prior_evidence": getattr(primary, "persistence_evidence", None),
                    "cleanup_evidence": cleanup.persistence_evidence})
                run.recorder.persistence_failure = primary

    def _make_scope(self, run: _Run, stage: StageSpec, *, session_id: str | None = None) -> Scope:
        run.session_index += 1
        sid = session_id or f"{run.config.run_id}/session{run.session_index}"
        return Scope(task_id=run.task.describe().task_id, stage_id=stage.stage_id, target_id=stage.target_id, session_id=sid,
                     detector_scope_id=f"{run.config.run_id}/scope{run.session_index}",
                     requirement_version=stage.requirement_version, evaluator_version=stage.evaluator_version)

    def _worker_task_prompt(self, run: _Run, stage: StageSpec) -> str:
        from ..tasks.prompts import prompt_for_task
        prompt = prompt_for_task(run.task, stage)
        run.recorder.emit("lifecycle", payload={"transition": "worker_task_prompt_bound",
            **prompt.binding(), "prompt_ref": to_jsonable(
                run.recorder.text_ref(prompt.text, Visibility.WORKER_VISIBLE))})
        return prompt.text

    def _new_agent(self, run: _Run, env: DockerWorkerEnvironment, *, token_limit: int) -> RecordingAgent:
        model = self.model_factory(run.config)
        if model.effective_request_policy() != run.config.worker_request_policy:
            raise TechnicalFailure("Executed worker request policy differs from declared configuration")
        if model.config.get("provider") != "scripted" and (model.model_name != run.config.model
                or model.model_settings != dict(run.config.model_settings)):
            raise TechnicalFailure("Executed worker model/route settings differ from declared configuration")
        model.bind_execution(run.config.worker_request_policy)
        model.unresolved_retry = unresolved_retry_rule(run.config.metadata.get("unresolved_retry")) if run.v2 else None
        agent = RecordingAgent(model, env, recorder=run.recorder, ledger=run.ledger, generated_token_limit=token_limit,
                              max_model_calls=run.config.max_model_calls, system_template=self.system_template,
                              instance_template=self._instance_template(run.config),
                              max_consecutive_format_errors=run.config.max_consecutive_format_errors)
        agent.tool_state_policy = V2_TOOL_STATE.get(getattr(run, "v2", None), "full")
        limit = getattr(run, "context_limit", None)
        agent.context_limit = limit
        template = run.config.metadata.get("instance_template")
        run.recorder.emit("lifecycle", payload={"transition": "worker_policy_bound",
            "tool_state_policy": agent.tool_state_policy,
            **({"context_limit": limit.binding()} if limit is not None else {}),
            "instance_template": {"version": template or "shared", "sha256": hashlib.sha256(
                agent.config.instance_template.encode("utf-8")).hexdigest()},
            "operational_settings": worker_operational_settings(run.config),
            "effective_request_policy": to_jsonable(model.effective_request_policy()),
            "effective_format_error_limit": agent.config.max_consecutive_format_errors,
            "effective_model_call_limit": agent.config.step_limit or None,
            "effective_model": model.model_name, "effective_model_settings": model.model_settings,
            "provider": model.config.get("provider"), "scripted_fixture": model.config.get("provider") == "scripted",
            "allowance_policy": model.config["allowance_policy"],
            **({"unresolved_retry": model.unresolved_retry} if model.unresolved_retry is not None else {})})
        return agent

    def _start_natural(self, run: _Run, stage: StageSpec) -> None:
        env = self._new_env(run)
        try:
            self._start_natural_in(run, stage, env)
        except BaseException as error:
            if run.session is None or run.session.env is not env:
                self._close_after_failure(run, env, error)
            raise

    def _start_natural_in(self, run: _Run, stage: StageSpec, env: DockerWorkerEnvironment) -> None:
        host_ws = run.scratch / "initial-workspace"
        def prepare_workspace():
            native_archive = getattr(run.task, "initial_workspace_archive", None)
            if native_archive is not None:
                # The benchmark's own /workspace, byte-for-byte with numeric ownership and modes (e.g. a root-owned,
                # execute-only reference binary), instead of a host re-pack that would hand every file to the worker.
                data = native_archive(stage_id=stage.stage_id)
            else:
                run.task.materialize(host_ws, stage_id=stage.stage_id)
                data = pack_directory(host_ws, default_owner=(env.worker_uid, env.worker_gid))
            manifest = loop_manifest_from_fidelity(fidelity_manifest_from_tar(data))
            return data, manifest

        data, manifest = self._timed_call(run, "initial_workspace_prepare", prepare_workspace, role="infrastructure")
        try:
            self._timed_call(run, "initial_workspace_restore",
                lambda: env.restore_workspace(data, fidelity_manifest_from_tar(data)))
        except RestoreError as e:
            raise TechnicalFailure(f"initial workspace materialization failed: {e}") from e
        scope = self._make_scope(run, stage)
        run.recorder.scope = scope
        run.recorder.emit("lifecycle", payload={"transition": "stage_started", "stage": to_jsonable(stage), "scope": to_jsonable(scope),
                                                "initial_workspace_manifest_digest": manifest_digest(manifest), "session_index": run.session_index,
                                                "environment": env.binding()})
        def initialize_session():
            agent = self._new_agent(run, env, token_limit=run.config.generated_token_limit)
            agent.start(self._worker_task_prompt(run, stage))
            return _Session(stage=stage, scope=scope, env=env, agent=agent, w_manifest=env.workspace_manifest(),
                                   r_manifest=env.runtime_manifest(), w_fidelity=env.workspace_fidelity_manifest())

        run.session = self._timed_call(run, "natural_session_prepare", initialize_session, role="infrastructure")
        self._detector_input(run, {"type": "scope_start", "scope_id": scope.detector_scope_id, "stage_id": stage.stage_id,
                                   "reason": "stage_start"})
        self._checkpoint(run, reasons=("initial", "graded"), labels=("Initial", "StageStart"), grade=True, scope_baseline=True)

    def _start_from_checkpoint(self, run: _Run, resume: Checkpoint | None, treatment: TreatmentManifest | None) -> None:
        store = run.store
        branch_v2 = getattr(run, "v2", None) == V2_BRANCH  # older run doubles carry no v2 field
        if branch_v2:
            # Before any paid turn: the uncompiled treatment (every channel from the source checkpoint) and, below,
            # the task, settings and recovery-source bindings. The review or note turn runs after those checks; its
            # choice compiles into ordinary same-run donors, context edits, a hand-back and any injection, validated
            # again and then guarded exactly like any other treatment.
            from ..experiments.treatments import validate_manifest
            source_run = store.get_checkpoint(treatment.source_checkpoint_id).run_id
            problems = validate_manifest(treatment, store, source_run)
            if problems:
                raise TechnicalFailure("Branch-v2 manifest invalid before the turn: " + "; ".join(problems))
        manual = treatment is not None and "manual_experiment" in treatment.metadata
        if treatment is not None:
            if manual:
                from ..experiments.manual import validate_manual_manifest
                problems = validate_manual_manifest(treatment, store)
                if problems:
                    raise TechnicalFailure("Manual experiment manifest invalid: " + "; ".join(problems))
            declared_recovery = treatment.metadata.get("recovery_settings")
            if declared_recovery is not None:
                actual_recovery = worker_operational_settings(run.config)
                if dict(declared_recovery) != actual_recovery:
                    raise TechnicalFailure("Treatment recovery settings differ from executed configuration")
            run.fixed_target = True
            source = store.get_checkpoint(treatment.source_checkpoint_id)
            w_donor = store.get_checkpoint(treatment.component_donors.get("workspace", source.checkpoint_id))
            r_donor_id = treatment.component_donors.get("runtime")
            r_donor = store.get_checkpoint(r_donor_id) if r_donor_id else (w_donor if treatment.runtime_policy == "restore_captured" else None)
            history_policy = treatment.history_policy
            h_donor_id = treatment.component_donors.get("history", source.checkpoint_id)
            h_donor = store.get_checkpoint(h_donor_id) if h_donor_id != "fresh" else source
            stage = run.task.stage(treatment.stage_id)
            token_limit = treatment.generated_token_limit
        else:
            assert resume is not None
            source = w_donor = r_donor = store.get_checkpoint(resume.checkpoint_id)
            h_donor = source
            if source != resume:
                raise TechnicalFailure("Resume object differs from immutable checkpoint")
            history_policy = "native"
            stage = run.task.stage(resume.scope.stage_id)
            token_limit = run.config.generated_token_limit
            if getattr(run.task, "native_submission_protocol", None) is not None:
                # The consecutive-failure count is part of the natural prefix being resumed.
                keys = ("ordinal", "status", "criterion_id", "consecutive_same", "checkpoint_id", "evaluation_id",
                        "grade_reused_from_submitting_step")
                run.native_submissions = [{k: e.payload.get(k) for k in keys}
                    for e in store.iter_events(source.run_id, through_sequence=source.event_sequence)
                    if e.kind == "lifecycle" and e.payload.get("transition") == "native_submission_verdict"]
        if resume is not None and resume != source:
            raise TechnicalFailure("Resume checkpoint differs from treatment source")
        if (stage.stage_id, stage.target_id, stage.requirement_version, stage.evaluator_version) != (
                source.scope.stage_id, source.scope.target_id, source.scope.requirement_version, source.scope.evaluator_version):
            raise TechnicalFailure("Recovery target differs from selected checkpoint scope")
        if source.scope.task_id != run.task.describe().task_id:
            raise TechnicalFailure("Recovery task differs from selected checkpoint")
        start = next((e.payload for e in store.iter_events(source.run_id)
                      if e.kind == "lifecycle" and e.payload.get("transition") == "run_started"), None)
        if not start or start.get("task") != to_jsonable(run.task.describe()):
            raise TechnicalFailure("Recovery task binding differs from the recorded natural source")
        from ..resample_source import recovery_source_ok, store_start_lookup
        if start and not recovery_source_ok(start, store_start_lookup(store)):
            from .technical_resume import chain_source_ok
            if treatment is not None or not chain_source_ok(start, store_start_lookup(store), run.config.metadata):
                raise TechnicalFailure("Recovery cannot inherit another recovery branch")
        if start and (start["config"]["model"] != run.config.model or
                      start["config"]["model_settings"] != dict(run.config.model_settings)):
            raise TechnicalFailure("Recovery worker differs from natural source settings")
        if start:
            for key in ("tool_timeout_seconds", "detector_settings", "worker_request_policy", "max_consecutive_format_errors"):
                if start["config"].get(key) != to_jsonable(run.config).get(key):
                    if key == "detector_settings" and treatment is None and _declared_detector_override(start, run.config):
                        continue  # declared resample detector override (rule version only; never reaches the worker)
                    raise TechnicalFailure(f"Recovery execution settings differ from natural source: {key}")
        if treatment is not None:
            if treatment.target_id != source.scope.target_id or treatment.stage_id != source.scope.stage_id:
                raise TechnicalFailure("Treatment changes fixed recovery target")
            if treatment.worker_model != run.config.model or token_limit != run.config.generated_token_limit:
                raise TechnicalFailure("Treatment worker/allowance differs from execution configuration")
            if not (manual or branch_v2) and treatment.component_donors.get("history", source.checkpoint_id) not in (
                    source.checkpoint_id, "fresh"):
                raise TechnicalFailure("Primary history donor must be S or fresh")
        if branch_v2:
            treatment = self._v2_intervene(run, treatment)
            problems = validate_manifest(treatment, store, source.run_id)
            if problems:
                raise TechnicalFailure("Branch-v2 manifest invalid: " + "; ".join(problems))
            if treatment.worker_model != run.config.model or treatment.generated_token_limit != token_limit:
                raise TechnicalFailure("Branch-v2 compile changed the worker or the allowance")
            w_donor = store.get_checkpoint(treatment.component_donors.get("workspace", source.checkpoint_id))
            r_donor_id = treatment.component_donors.get("runtime")
            r_donor = store.get_checkpoint(r_donor_id) if r_donor_id else (
                w_donor if treatment.runtime_policy == "restore_captured" else None)
            history_policy = treatment.history_policy
            h_donor = store.get_checkpoint(treatment.component_donors.get("history", source.checkpoint_id))
        for donor in (w_donor, r_donor, h_donor):
            if donor is None:
                continue
            if donor.run_id != source.run_id or donor.ordinal > source.ordinal or donor.event_sequence > source.event_sequence:
                raise TechnicalFailure("Component donor is outside the selected natural prefix")
            if not donor.branchable:
                raise TechnicalFailure("Component donor is not branchable")
            if (donor.scope.task_id, donor.scope.stage_id, donor.scope.target_id,
                    donor.scope.requirement_version, donor.scope.evaluator_version) != (
                    source.scope.task_id, source.scope.stage_id, source.scope.target_id,
                    source.scope.requirement_version, source.scope.evaluator_version):
                raise TechnicalFailure("Component donor is outside the fixed target")
        if not source.branchable or not w_donor.branchable:
            raise TechnicalFailure("source or donor checkpoint is not branchable")
        env = self._new_env(run)
        try:
            self._restore_into(run, env, source, w_donor, r_donor, history_policy, stage, token_limit, treatment,
                               h_donor=h_donor)
        except BaseException as error:
            if run.session is None or run.session.env is not env:
                self._close_after_failure(run, env, error)
            raise

    def _restore_into(self, run: _Run, env: DockerWorkerEnvironment, source: Checkpoint, w_donor: Checkpoint,
                      r_donor: Checkpoint | None, history_policy: str, stage: StageSpec, token_limit: int,
                      treatment: TreatmentManifest | None, *, h_donor: Checkpoint | None = None) -> None:
        store = run.store
        h_donor = h_donor or source
        def load_workspace():
            for checkpoint in (source, w_donor, r_donor, h_donor):
                if checkpoint is not None:
                    self._verify_environment_binding(run, checkpoint, env)
            comp = (treatment.metadata or {}).get("workspace_composition") if treatment is not None else None
            if comp:  # declared selective revert: the composed W recorded at manifest build time
                if comp.get("base_checkpoint_id") != w_donor.checkpoint_id:
                    raise TechnicalFailure("workspace composition base differs from the workspace donor")
                return store.get_bytes(_ref(comp["workspace_ref"])), store.get_json(_ref(comp["manifest_ref"]))
            w_data = store.get_bytes(w_donor.state.workspace)
            w_manifest = store.get_json(_ref(w_donor.validation["workspace_manifest_ref"]))
            return w_data, w_manifest

        w_data, w_manifest = self._timed_call(run, "workspace_restore_input", load_workspace, role="infrastructure")
        try:
            w_result = self._timed_call(run, "workspace_restore", lambda: env.restore_workspace(w_data, w_manifest))
        except RestoreError as e:
            raise TechnicalFailure(f"workspace restore failed: {e}") from e
        r_result: dict[str, Any] = {"policy": "fresh_container", "validated": True}
        if r_donor is not None and r_donor.state.runtime_class == RuntimeClass.CAPTURED and r_donor.state.runtime is not None:
            def load_runtime():
                bundle = store.get_json(r_donor.state.runtime)
                bundle["archive"] = store.get_bytes(_ref(bundle["archive_ref"])) if bundle.get("archive_ref") else None
                return bundle

            bundle = self._timed_call(run, "runtime_restore_input", load_runtime, role="infrastructure")
            try:
                r_result = self._timed_call(run, "runtime_restore", lambda: env.restore_runtime(bundle))
                r_result["policy"] = "restore_captured"
            except RestoreError as e:
                raise TechnicalFailure(f"runtime restore failed: {e}") from e
        elif r_donor is not None and r_donor.state.runtime_class == RuntimeClass.UNSUPPORTED:
            raise TechnicalFailure("donor runtime is R_UNSUPPORTED; faithful restore impossible")
        history = self._timed_call(run, "history_restore_input",
            lambda: store.get_json(h_donor.state.history), role="infrastructure")
        carried = ("native", "o_masked", "edited")  # policies that continue the donor's own context and counters
        if history_policy in carried and history_replay_issue(history["messages"]):
            raise TechnicalFailure(f"Native history restoration unsupported: {history_replay_issue(history['messages'])}")
        session_id = h_donor.scope.session_id if history_policy in carried else None
        scope = self._make_scope(run, stage, session_id=session_id)
        run.recorder.scope = scope
        run.branch_origin_event = run.recorder.emit("lifecycle", payload={
            "transition": "branch_started", "source_checkpoint_id": source.checkpoint_id, "source_run_id": source.run_id,
            "workspace_donor": w_donor.checkpoint_id, "runtime_donor": r_donor.checkpoint_id if r_donor else None,
            "history_donor": h_donor.checkpoint_id if history_policy != "fresh" else "fresh",
            "history_policy": history_policy, "workspace_restore": w_result, "runtime_restore": r_result,
            "stage": to_jsonable(stage), "scope": to_jsonable(scope), "environment": env.binding(),
            "token_offset": history["counters"].get("known_generated_tokens", 0), "additional_allowance": token_limit,
            "treatment": to_jsonable(treatment) if treatment else None})
        offset = int(history["counters"].get("known_generated_tokens", 0)) if history_policy in carried else 0
        def initialize_session():
            agent = self._new_agent(run, env, token_limit=offset + token_limit)
            if history_policy == "edited":
                self._restore_edited(run, agent, history, h_donor, treatment)
            elif history_policy in ("native", "o_masked"):
                messages = history["messages"]
                mask_report = None
                if history_policy == "o_masked":
                    from ..experiments.observation_mask import apply_observation_mask, validate_observation_mask
                    spec = treatment.metadata.get("mask", {}) if treatment else {}
                    problems = validate_observation_mask(store, h_donor, spec,
                        selection_kind=treatment.selection_kind if treatment else "engineering_checkpoint")
                    if problems:
                        raise TechnicalFailure("Observation mask invalid: " + "; ".join(problems))
                    messages, mask_report = apply_observation_mask(messages, spec)
                agent.restore(messages, history["counters"], history["cursor"], task=history["task"])
                restored_counters = agent.counters()
                inherited = ("n_calls", "cost", "known_generated_tokens", "unknown_usage_calls",
                             "n_consecutive_format_errors")
                restore_checks = {"revision": "native-history-restore.v1",
                    "messages_exact": agent.messages == messages, "cursor_exact": agent.cursor() == history["cursor"],
                    "task_exact": agent.extra_template_vars.get("task") == history["task"],
                    "inherited_counters_exact": all(restored_counters.get(k) == history["counters"].get(k, 0)
                                                    for k in inherited),
                    "new_run_offsets_exact": (restored_counters["generated_token_offset"] == history["counters"].get("known_generated_tokens", 0)
                        and restored_counters["unknown_usage_offset"] == history["counters"].get("unknown_usage_calls", 0)
                        and restored_counters["calls_this_run"] == 0),
                    "observations_exact": [store.get_bytes(ref).decode("utf-8") for ref in agent.observation_refs]
                        == [m["content"] for m in messages if m.get("role") == "tool" and isinstance(m.get("content"), str)]}
                if not all(value for key, value in restore_checks.items() if key != "revision"):
                    run.recorder.emit("lifecycle", payload={"transition": "history_restore_failed", "validation": restore_checks})
                    raise TechnicalFailure("Native history/cursor/counters/O restoration did not round trip")
                delivered_history_ref = run.recorder.put_json(agent.messages, visibility=Visibility.WORKER_VISIBLE)
                run.recorder.emit("lifecycle", payload={"transition": "history_restored", "message_count": len(messages),
                                                        "counters": history["counters"], "cursor": history["cursor"],
                                                        "history_ref": to_jsonable(h_donor.state.history), "history_policy": history_policy,
                                                        "mask": mask_report, "validation": restore_checks,
                                                        "restored_counters": restored_counters,
                                                        "observations": to_jsonable(agent.observation_refs),
                                                        "delivered_history_ref": to_jsonable(delivered_history_ref)})
                if treatment is not None and treatment.packet_refs:
                    if "manual_experiment" not in treatment.metadata:
                        raise TechnicalFailure("Native history packet delivery requires a validated manual intervention")
                    for ref in treatment.packet_refs:
                        agent.append_user_message(store.get_bytes(ref).decode("utf-8"), kind="treatment_packet",
                            payload={"position": "after_restored_native_history", "packet_ref": ref.sha256,
                                     "treatment_id": treatment.treatment_id})
            elif history_policy == "fresh":
                agent.start(self._worker_task_prompt(run, stage) + "\n\n" + FRESH_HISTORY_ORIENTATION)
                for ref in (treatment.packet_refs if treatment else ()):
                    packet = store.get_bytes(ref).decode("utf-8")
                    agent.append_user_message(packet, kind="treatment_packet",
                                              payload={"position": "after_target_and_orientation", "packet_ref": ref.sha256})
            elif history_policy != "edited":
                raise TechnicalFailure(f"unsupported history policy {history_policy!r}")
            if getattr(run, "context_limit", None) is not None:
                # the restored context keeps the outputs hidden in it; the size anchor comes from its own record
                self._bind_context_limit(run, agent, h_donor if history_policy in carried else None)
            if getattr(run, "v2", None) == V2_BRANCH:
                self._v2_deliver(run, agent, history)
            return _Session(stage=stage, scope=scope, env=env, agent=agent, w_manifest=env.workspace_manifest(),
                                   r_manifest=env.runtime_manifest(), w_fidelity=env.workspace_fidelity_manifest())

        run.session = self._timed_call(run, "restored_session_prepare", initialize_session, role="infrastructure")
        if treatment is not None and treatment.metadata.get("arm_id") == "agentrewind":
            if history_policy != "native" or w_donor != source or r_donor != source or treatment.packet_refs:
                raise TechnicalFailure("AgentRewind baseline starts from aligned S W/R/H without external packets")
            self._install_baseline(run, source)
        if getattr(run, "v2", None) == V2_BRANCH:
            # submissions_and_end recording: no branch-start capture or grade; W and R are saved at submissions/end.
            agent = run.session.agent
            run.recorder.emit("lifecycle", payload={"transition": "branch_v2_ready", "rule": V2_RULES[V2_BRANCH],
                "call_offset": agent.call_offset, "token_offset": agent.generated_token_offset,
                "token_limit": agent.generated_token_limit, "intervention_calls": agent.intervention_calls,
                "counters": agent.counters(), "attempt_cost_known": agent.attempt_cost_known,
                "attempt_cost_unknown": agent.attempt_cost_unknown, "submission_cap": run.submission_cap})
            return
        self._detector_input(run, {"type": "scope_start", "scope_id": scope.detector_scope_id, "stage_id": stage.stage_id,
                                   "reason": "branch_start"})
        self._checkpoint(run, reasons=("initial", "graded", "branch_start"), labels=("Initial",), grade=True, scope_baseline=True)

    def _restore_edited(self, run: _Run, agent: RecordingAgent, history: dict[str, Any], h_donor: Checkpoint,
                        treatment: TreatmentManifest | None) -> None:
        """The `edited` history policy: the donor's context with the declared edits; counters kept, cursor and O rebuilt."""
        from ..experiments.context_edit import EDITS_VERSION, EditError, apply_edits
        spec = (treatment.metadata or {}).get("context_edits") if treatment is not None else None
        if not isinstance(spec, Mapping) or spec.get("version") != EDITS_VERSION or not spec.get("edits"):
            raise TechnicalFailure("Edited history needs declared context edits")
        try:
            messages, report = apply_edits(history["messages"], list(spec["edits"]))
        except EditError as e:
            raise TechnicalFailure(f"Context edits do not apply: {e}") from e
        cursor = {"message_count": len(messages), "pending_actions": [], "completed_in_batch": []}
        agent.restore(messages, history["counters"], cursor, task=history["task"])
        restored = agent.counters()
        inherited = ("n_calls", "cost", "known_generated_tokens", "unknown_usage_calls", "n_consecutive_format_errors")
        checks = {"revision": "edited-history-restore.v1", "messages_exact": agent.messages == messages,
            "cursor_rebuilt": agent.cursor() == cursor,
            "task_exact": agent.extra_template_vars.get("task") == history["task"],
            "inherited_counters_exact": all(restored.get(k) == history["counters"].get(k, 0) for k in inherited),
            "new_run_offsets_exact": (restored["generated_token_offset"] == history["counters"].get("known_generated_tokens", 0)
                and restored["unknown_usage_offset"] == history["counters"].get("unknown_usage_calls", 0)
                and restored["calls_this_run"] == 0),
            "observations_rebuilt": [run.store.get_bytes(ref).decode("utf-8") for ref in agent.observation_refs]
                == [m["content"] for m in messages if m.get("role") == "tool" and isinstance(m.get("content"), str)]}
        if not all(value for key, value in checks.items() if key != "revision"):
            run.recorder.emit("lifecycle", payload={"transition": "history_restore_failed", "validation": checks})
            raise TechnicalFailure("Edited history/cursor/counters/O restoration did not round trip")
        delivered = run.recorder.put_json(agent.messages, visibility=Visibility.WORKER_VISIBLE)
        run.recorder.emit("lifecycle", payload={"transition": "history_restored", "history_policy": "edited",
            "message_count": len(messages), "counters": history["counters"], "cursor": cursor,
            "history_ref": to_jsonable(h_donor.state.history), "context_edits": report, "validation": checks,
            "restored_counters": restored, "observations": to_jsonable(agent.observation_refs),
            "delivered_history_ref": to_jsonable(delivered)})
        if treatment is not None and treatment.packet_refs:
            if "manual_experiment" not in treatment.metadata:
                raise TechnicalFailure("Edited history packet delivery requires a validated manual intervention")
            for ref in treatment.packet_refs:
                agent.append_user_message(run.store.get_bytes(ref).decode("utf-8"), kind="treatment_packet",
                    payload={"position": "after_restored_edited_history", "packet_ref": ref.sha256,
                             "treatment_id": treatment.treatment_id})

    # ================================================================== mining
    def _mine(self, run: _Run) -> str:
        while True:
            s = run.session
            assert s is not None
            # Recovery has a fixed evaluated target. Natural stage release
            # continues to follow the benchmark's own worker exit semantics.
            if run.fixed_target and s.last_evaluation is not None and s.last_evaluation.valid and s.last_evaluation.full_solve:
                return "fixed_target_complete"
            # Stall resampling (user 2026-09-27): a natural continuation may declare an early stop on a valid full
            # solve (RunConfig.metadata["stop_on_full_solve"]); absent = unchanged natural behaviour.
            if (run.config.metadata.get("stop_on_full_solve") is True and s.last_evaluation is not None
                    and s.last_evaluation.valid and s.last_evaluation.full_solve):
                return "resample_full_solve"
            before_checkpoints = len(run.checkpoints)
            run.native_exit_status = None
            exit_status = self._step(run)
            run.native_exit_status = exit_status
            self._drain_usage(run)
            mutated = self._timed_call(run, "post_step_mutation_measurement",
                lambda: self._mutation(run), role="infrastructure")
            lap_request = self._timed_call(run, "lap_recurrence",
                lambda: self._observe_laps(run, relevant_mutation=mutated), role="infrastructure") \
                if self.detector_config.rule_version in LAP_RULE_VERSIONS else None
            if mutated or lap_request is not None:
                reasons = ("graded", "lap_recurrence") if lap_request is not None else ("graded",)
                self._checkpoint(run, reasons=reasons, labels=(), grade=True)
            elif run.baseline is not None and exit_status is None and len(run.checkpoints) == before_checkpoints:
                self._checkpoint(run, reasons=("baseline_decision",), labels=(), grade=False)
            if self.detector_config.rule_version in STALL_CLOCK_RULE_VERSIONS:
                # Check the stall clock after every completed turn, including turns without edits.
                outputs = self._detector_input(run, {"type": "turn_boundary", "scope_id": s.scope.detector_scope_id})
                if any(o.get("record") == "grade_request" for o in outputs):
                    self._checkpoint(run, reasons=("graded", "stall_clock"), labels=(), grade=True)
            if s.agent.accounting_interrupted():
                run.recorder.incident("worker_accounting_interruption", further_requests_permitted=False,
                    counters=s.agent.counters(), final_state_retained=True)
                return "accounting_interruption"
            if exit_status is None and s.agent.budget_exhausted():
                exit_status = "WorkerBudgetExhausted"
                run.native_exit_status = exit_status
                s.agent.add_messages({"role": "exit", "content": exit_status, "extra": {"exit_status": exit_status, "submission": ""}})
                run.recorder.emit("lifecycle", payload={"transition": "budget_boundary", "known_generated_tokens": s.agent.known_generated_tokens,
                                                        "unknown_usage_calls": s.agent.unknown_usage_calls,
                                                        "nominal_allowance": s.agent.generated_token_limit})
            if exit_status is None:
                guard = self._run_guard(run)
                if guard is not None:
                    exit_status = guard
                    run.native_exit_status = exit_status
                    s.agent.add_messages({"role": "exit", "content": exit_status, "extra": {"exit_status": exit_status, "submission": ""}})
            if exit_status is None:
                continue
            normalized = NORMALIZED_REASONS.get(exit_status, f"worker_exit:{exit_status}")
            run.recorder.emit("lifecycle", payload={"transition": "worker_exit_normalized",
                "raw_exit": exit_status, "normalized": normalized,
                "limit_scope": "run_wide", "stage_release_eligible": exit_status in STAGE_FINISH_REASONS})
            # Even a submission at the allowance boundary cannot disclose a
            # later target. Both counters carry through native stage resets.
            if exit_status == "WorkerBudgetExhausted" or s.agent.budget_exhausted():
                return "budget_exhausted"
            if exit_status in ("WorkerCostLimit", "WorkerWallClockLimit"):
                return NORMALIZED_REASONS[exit_status]
            if exit_status == "LimitsExceeded" or (run.config.max_model_calls is not None
                    and s.agent.n_calls - s.agent.call_offset >= run.config.max_model_calls):
                return "call_limit"
            if exit_status == "RepeatedFormatError":
                return "worker_error:RepeatedFormatError"
            if exit_status not in STAGE_FINISH_REASONS:
                return f"worker_exit:{exit_status}"
            if run.fixed_target:
                # Q16: recovery branches solve the frozen S1 target only; no later-stage disclosure.
                return f"fixed_target_finished:{normalized}"
            if getattr(run.task, "native_submission_protocol", None) is not None:
                # Benchmark-declared submission feedback (explicit reopen policy).
                outcome = self._native_submission(run, before_checkpoints)
                if outcome is None:
                    continue
                return f"sequence_complete:{outcome}"
            next_stage = run.task.next_stage(s.stage.stage_id, termination_reason=normalized)
            if next_stage is None:
                return f"sequence_complete:{normalized}"
            # Capture the outgoing state now; advancement labels become known
            # only after the corresponding incoming boundary is validated.
            stage_end = self._checkpoint(run, reasons=("stage_transition_boundary", "graded"), labels=(), grade=True)
            self._transition(run, stage_end, next_stage, exit_status)

    def _run_guard(self, run: _Run) -> str | None:
        """Runaway guards (dollar / wall clock). Checked between turns; never shown to the worker."""
        s = run.session
        limit = run.config.cost_limit_usd
        if limit is not None and s.agent.cost >= limit:
            run.recorder.emit("lifecycle", payload={"transition": "run_guard_boundary", "guard": "cost_limit_usd",
                "limit": limit, "observed": s.agent.cost, "basis": "SDK known accepted-response cost"})
            return "WorkerCostLimit"
        limit = run.config.wall_clock_limit_seconds
        elapsed = time.monotonic() - run.started_monotonic
        if limit is not None and elapsed >= limit:
            run.recorder.emit("lifecycle", payload={"transition": "run_guard_boundary", "guard": "wall_clock_limit_seconds",
                "limit": limit, "observed": elapsed})
            return "WorkerWallClockLimit"
        return None

    def _native_submission(self, run: _Run, before_checkpoints: int) -> str | None:
        """Apply an adapter-declared native submission protocol. None means the session continues."""
        s = run.session
        assert s is not None
        protocol = dict(run.task.native_submission_protocol)
        limit = protocol.get("repeated_failure_termination")
        if type(limit) is not int or limit <= 0:
            raise TechnicalFailure("Native submission protocol lacks a positive repeated-failure limit")
        if len(run.checkpoints) > before_checkpoints and s.last_graded_checkpoint is run.checkpoints[-1]:
            checkpoint, evaluation, reused = run.checkpoints[-1], s.last_evaluation, True
        else:
            checkpoint = self._checkpoint(run, reasons=("native_submission", "graded"), labels=(), grade=True)
            evaluation, reused = s.last_evaluation, False
        verdict = dict(run.task.native_submission_verdict(evaluation))
        status = verdict.get("status")
        if status not in {"pass", "fail", "error"}:
            run.recorder.emit("lifecycle", payload={"transition": "native_submission_unavailable",
                "rule": NATIVE_SUBMISSION_RULE, "checkpoint_id": checkpoint.checkpoint_id, "verdict": verdict})
            raise TechnicalFailure("Native submission verdict unavailable for the submitted state")
        criterion = verdict.get("criterion_id")
        previous = run.native_submissions[-1] if run.native_submissions else None
        consecutive = (previous["consecutive_same"] + 1 if previous and status != "pass"
                       and previous.get("criterion_id") == criterion else 1)
        record = {"ordinal": len(run.native_submissions) + 1, "status": status, "criterion_id": criterion,
                  "consecutive_same": consecutive if status != "pass" else 0, "checkpoint_id": checkpoint.checkpoint_id,
                  "evaluation_id": evaluation.evaluation_id if evaluation is not None else None,
                  "grade_reused_from_submitting_step": reused}
        run.native_submissions.append(record)
        if status == "pass":
            outcome = "native_success"
        elif consecutive >= limit:
            outcome = "native_repeated_failure"
        else:
            outcome = None
        run.recorder.emit("lifecycle", payload={"transition": "native_submission_verdict", "rule": NATIVE_SUBMISSION_RULE,
            "protocol": protocol, **record, "outcome": outcome or "continue"})
        if outcome is not None:
            return outcome
        feedback = verdict.get("feedback")
        if not isinstance(feedback, str):
            raise TechnicalFailure("Native submission verdict lacks a feedback string")
        messages = s.agent.messages
        if not messages or messages[-1].get("role") != "exit":
            raise TechnicalFailure("Native reopen expected a trailing submission exit marker")
        exit_message = messages.pop()
        # The submitting batch was interrupted before its observations were appended.
        # A tool-call conversation needs a result per call: deliver the executed
        # outputs (the submission command's real output) and an explicit
        # not-executed result for any later call in that batch.
        batch_message, batch_outputs = getattr(s.agent, "batch_outputs", (None, None))
        if batch_message is None or not messages or messages[-1] is not batch_message:
            raise TechnicalFailure("Native reopen could not bind the interrupted submission batch")
        actions = batch_message.get("extra", {}).get("actions", [])
        outputs = list(batch_outputs) + [{"output": "", "returncode": -1,
            "exception_info": "not executed: the batch ended at the submission command"}
            for _ in actions[len(batch_outputs):]]
        observations = s.agent.model.format_observation_messages(batch_message, outputs, s.agent.get_template_vars())
        s.agent.add_messages(*observations)
        for observation in observations:
            delivered = run.recorder.put_bytes(observation["content"].encode("utf-8"),
                media_type="text/plain; charset=utf-8", visibility=Visibility.WORKER_VISIBLE)
            s.agent.observation_refs.append(delivered)
            run.recorder.emit("observation_delivery", call_id=observation.get("tool_call_id"), actor="harness",
                delivered_ref=delivered, visibility=Visibility.WORKER_VISIBLE,
                payload={"role": observation.get("role"), "delivery_status": "appended_to_history",
                         "native_submission_reopen": True, "rule": NATIVE_SUBMISSION_RULE})
        issue = history_replay_issue(messages)
        if issue is not None:
            raise TechnicalFailure(f"Native reopen left an incomplete conversation: {issue}")
        wrapper = feedback_wrapper(run.config)
        from ..experiments.submission_feedback import delivery
        delivered_feedback, feedback_policy = delivery(run.config, feedback, wrapper)
        s.agent.append_user_message(delivered_feedback, kind="native_submission_feedback",
            payload={"rule": NATIVE_SUBMISSION_RULE, "submission_ordinal": record["ordinal"],
                     "feedback_wrapper": None if wrapper is None else {k: wrapper[k] for k in ("version", "template_sha256")},
                     "submission_feedback": feedback_policy,
                     "native_feedback_sha256": hashlib.sha256(feedback.encode("utf-8")).hexdigest(),
                     "criterion_id": criterion, "checkpoint_id": checkpoint.checkpoint_id,
                     "reopened_exit": exit_message.get("extra", {}).get("exit_status")})
        if self.detector_config.rule_version in SUBMISSION_VERDICT_RULE_VERSIONS:
            # After the feedback reaches the worker the conversation is complete again, so a checkpoint
            # here is branchable (the submission checkpoint itself ends in the exit marker).
            outputs = self._detector_input(run, {"type": "submission_verdict", "scope_id": s.scope.detector_scope_id,
                "checkpoint_id": checkpoint.checkpoint_id, "status": status, "criterion_id": criterion})
            if any(o.get("record") == "grade_request" for o in outputs):
                self._checkpoint(run, reasons=("graded", "repeated_rejection"), labels=(), grade=True)
        return None

    # ============================================================ v2 policies
    def _mine_v2(self, run: _Run) -> str:
        """natural-v2/v1 and branch-v2/v1 (see V2_RULES); the earlier _mine stays unchanged for every other run."""
        branch = run.v2 == V2_BRANCH
        first = branch  # a branch's intervention turn may already have used its allowance: check before any call
        while True:
            s = run.session
            assert s is not None
            run.native_exit_status = None
            if first:
                first = False
                if run.v2_stop_before_first_call is not None:  # the turn's or the first request does not fit
                    return self._v2_stop(run, run.v2_stop_before_first_call)
                guard = "WorkerBudgetExhausted" if s.agent.budget_exhausted() else self._run_guard_v2(run)
                if guard is not None:
                    return self._v2_stop(run, guard)
            exit_status = self._step(run)
            run.native_exit_status = exit_status
            self._drain_usage(run)
            submitted = exit_status == "Submitted"
            graded = capture = None
            point_taken = False
            if not branch:
                capture = self._capture(run, reasons=("step_end",))
                if capture is None:
                    # A recorded unsupported point is not a checkpoint. Advance
                    # the detector clock, record missing grades, and enforce the
                    # same guards before allowing the next worker request.
                    outputs = self._detector_input(run, {"type": "turn_boundary", "scope_id": s.scope.detector_scope_id})
                    run.recorder.emit("lifecycle", payload={"transition": "checkpoint_grade_unavailable",
                        "step_index": s.agent.n_calls, "reason": "background_capture_gap",
                        "grade_required_by_detector": any(o.get("record") == "grade_request" for o in outputs),
                        "mutation_status": "unknown", "score": None, "branchable": False})
                    if s.agent.accounting_interrupted():
                        return "accounting_interruption"
                    guard = "WorkerBudgetExhausted" if s.agent.budget_exhausted() else self._run_guard_v2(run)
                    if guard is not None:
                        return self._v2_stop(run, guard)
                    continue
                if self._mutation_v2(run, capture):
                    graded = self._checkpoint_from(run, capture, labels=(), grade=True,
                                                   reasons=("graded",) if submitted else ("graded", "point"))
                    point_taken = not submitted
                outputs = self._detector_input(run, {"type": "turn_boundary", "scope_id": s.scope.detector_scope_id})
                if any(o.get("record") == "grade_request" for o in outputs):
                    if graded is None:
                        graded = self._checkpoint_from(run, capture, labels=(), grade=True, reasons=(
                            ("graded", "stall_clock") if submitted else ("graded", "stall_clock", "point")))
                        point_taken = not submitted
                    else:  # unreachable while the clock sees this turn's grade; kept explicit, never merged
                        graded = self._checkpoint(run, reasons=("graded", "stall_clock"), labels=(), grade=True)
                if graded is not None and s.last_evaluation is not None and s.last_evaluation.valid \
                        and s.last_evaluation.full_solve and not submitted:
                    run.recorder.emit("lifecycle", payload={"transition": "private_full_solve_logged",
                        "checkpoint_id": graded.checkpoint_id, "evaluation_id": s.last_evaluation.evaluation_id,
                        "step_index": s.agent.n_calls, "rule": V2_RULES[V2_NATURAL]})
            if submitted:
                run.recorder.emit("lifecycle", payload={"transition": "worker_exit_normalized", "raw_exit": exit_status,
                    "normalized": NORMALIZED_REASONS[exit_status], "limit_scope": "run_wide",
                    "stage_release_eligible": False, "v2_run_policy": run.v2})
                outcome, point_taken = self._native_submission_v2(run, graded, capture)
                if outcome is not None:
                    if branch:  # the branch's end state is exactly this graded submission state
                        run.v2_end_checkpoint = run.checkpoints[-1]
                    return f"sequence_complete:{outcome}"
                exit_status = run.native_exit_status = None
                if not branch and not point_taken:
                    self._checkpoint(run, reasons=("point", "post_feedback"), labels=(), grade=False)
            elif not branch and not point_taken:
                self._checkpoint_from(run, capture, reasons=("point",), labels=(), grade=False)
            if s.agent.accounting_interrupted():
                run.recorder.incident("worker_accounting_interruption", further_requests_permitted=False,
                    counters=s.agent.counters(), final_state_retained=True)
                return "accounting_interruption"
            if exit_status is None:
                guard = ("WorkerBudgetExhausted" if s.agent.budget_exhausted() else self._run_guard_v2(run))
                if guard is None:
                    continue
                return self._v2_stop(run, guard)
            return self._v2_exit(run, exit_status)

    def _v2_stop(self, run: _Run, guard: str) -> str:
        """End a v2 run at a token, cost or step guard between turns (never worker-visible)."""
        s = run.session
        assert s is not None
        if guard == "WorkerBudgetExhausted":
            run.recorder.emit("lifecycle", payload={"transition": "budget_boundary",
                "known_generated_tokens": s.agent.known_generated_tokens,
                "unknown_usage_calls": s.agent.unknown_usage_calls,
                "nominal_allowance": s.agent.generated_token_limit})
        run.native_exit_status = guard
        s.agent.add_messages({"role": "exit", "content": guard, "extra": {"exit_status": guard, "submission": ""}})
        return self._v2_exit(run, guard)

    def _v2_exit(self, run: _Run, exit_status: str) -> str:
        normalized = NORMALIZED_REASONS.get(exit_status, f"worker_exit:{exit_status}")
        run.recorder.emit("lifecycle", payload={"transition": "worker_exit_normalized",
            "raw_exit": exit_status, "normalized": normalized, "limit_scope": "run_wide",
            "stage_release_eligible": False, "v2_run_policy": run.v2})
        if exit_status == "WorkerBudgetExhausted":
            return "budget_exhausted"
        if exit_status == "LimitsExceeded":
            return "call_limit"
        if exit_status == "WorkerCostLimit":
            return "cost_limit"
        if exit_status == "RepeatedFormatError":
            return "worker_error:RepeatedFormatError"
        if exit_status == CONTEXT_EXIT_STATUS:
            return "context_window_exhausted"
        return f"worker_exit:{exit_status}"

    @staticmethod
    def _state_change(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        bw, aw = comparable(before["workspace"]), comparable(after["workspace"])
        br, ar = comparable(before["runtime"]["entries"]), comparable(after["runtime"]["entries"])
        w = sorted(p for p in set(bw) | set(aw) if bw.get(p) != aw.get(p))
        r = sorted({p for p in set(br) | set(ar) if br.get(p) != ar.get(p)}
                   | (set(before["runtime"]["deleted"]) ^ set(after["runtime"]["deleted"])))
        mounts = before["runtime"].get("mounts") != after["runtime"].get("mounts")
        return {"workspace_changed": w, "runtime_changed": r, "mounts_changed": mounts, "changed": bool(w or r or mounts)}

    def _mutation_v2(self, run: _Run, capture: dict[str, Any]) -> bool:
        """Per-action change evidence plus the endpoint difference; never 'unchanged' from one endpoint scan."""
        s = run.session
        assert s is not None
        runtime = capture["runtime"]
        start = {"workspace": s.w_fidelity, "runtime": s.r_manifest}
        end = {"workspace": capture["w_fidelity"],
               "runtime": {"entries": runtime["entries"], "deleted": runtime["deleted"], "mounts": runtime["mounts"]}}
        per_action, previous = [], start
        for index, state, event in s.agent.batch_action_states:
            per_action.append({"tool_index": index, "source": "manifest_scan", "state_event_id": event.event_id,
                               **self._state_change(previous, state)})
            previous = state
        executed = len(s.agent.completed_in_batch)
        if executed:
            per_action.append({"tool_index": executed - 1, "source": "step_capture",
                               **self._state_change(previous, end)})
        endpoint = self._state_change(start, end)
        gap = run.capture_gap_pending
        relevant = gap or endpoint["changed"] or any(a["changed"] for a in per_action)
        before_w, after_w = s.w_manifest, loop_manifest_from_fidelity(capture["w_fidelity"])
        files = [{"path": p, "before": before_w.get(p), "after": after_w.get(p)}
                 for p in sorted(set(before_w) | set(after_w)) if before_w.get(p) != after_w.get(p)]
        ev = run.recorder.emit("mutation", payload={"rule": MUTATION_RULE_V2, "relevant": relevant,
            "workspace_before": manifest_digest(before_w), "workspace_after": manifest_digest(after_w), "files": files,
            "workspace_fidelity_paths_changed": endpoint["workspace_changed"],
            "runtime_paths_changed": endpoint["runtime_changed"], "mounted_state_changed": endpoint["mounts_changed"],
            "per_action": per_action, "actions_executed": executed,
            "within_batch_mutation": any(a["changed"] for a in per_action) and not endpoint["changed"],
            "response_event_sequence": s.agent.last_response_event_sequence,
            **({"prior_capture_gap": True, "intermediate_evidence_complete": False,
                "grade_forced_after_gap": True} if gap else {})})
        self._detector_input(run, {"type": "mutation", "relevant": relevant, "workspace_before": manifest_digest(before_w),
                                   "workspace_after": manifest_digest(after_w), "files": files,
                                   "event_sequence": ev.sequence})
        s.w_manifest, s.r_manifest, s.w_fidelity = after_w, end["runtime"], capture["w_fidelity"]
        run.capture_gap_pending = False
        return relevant

    def _run_guard_v2(self, run: _Run) -> str | None:
        """Cost and step guards counted from this run's start (a branch: its intervention). Never worker-visible."""
        s = run.session
        assert s is not None
        agent = s.agent
        limit = run.config.cost_limit_usd
        spent = agent.attempt_cost_known + agent.attempt_cost_worst_case
        if limit is not None and spent >= limit:
            run.recorder.emit("lifecycle", payload={"transition": "run_guard_boundary", "guard": "cost_limit_usd",
                "limit": limit, "observed": spent, "unknown_cost_attempts": agent.attempt_cost_unknown,
                "observed_known": agent.attempt_cost_known, "observed_worst_case_charged": agent.attempt_cost_worst_case,
                "charged_unknown_attempts": agent.charged_unknown_attempts,
                "basis": "provider-reported cost of every finished worker attempt in this run, failed attempts and "
                         "intervention turns included, plus the worst-case charge of each retried unresolved attempt "
                         "(worst-case-charged-retry.v1)"})
            return "WorkerCostLimit"
        calls = agent.n_calls - agent.call_offset + agent.intervention_calls
        if run.config.max_model_calls is not None and calls >= run.config.max_model_calls:
            run.recorder.emit("lifecycle", payload={"transition": "run_guard_boundary", "guard": "max_model_calls",
                "limit": run.config.max_model_calls, "observed": calls,
                "intervention_calls": agent.intervention_calls,
                "basis": "model calls in this run, intervention turns included"})
            return "LimitsExceeded"
        return None

    def _native_submission_v2(self, run: _Run, graded: Checkpoint | None,
                              capture: dict[str, Any] | None) -> tuple[str | None, bool]:
        """The verdict of a submission, processed before any guard: (outcome or None, post-feedback point taken)."""
        s = run.session
        assert s is not None
        protocol = dict(run.task.native_submission_protocol)
        limit = protocol.get("repeated_failure_termination")
        if type(limit) is not int or limit <= 0:
            raise TechnicalFailure("Native submission protocol lacks a positive repeated-failure limit")
        if graded is not None:
            checkpoint, reused = graded, True
        elif capture is not None:
            checkpoint, reused = self._checkpoint_from(run, capture, reasons=("native_submission", "graded"),
                                                       labels=(), grade=True), False
        else:
            checkpoint, reused = self._checkpoint(run, reasons=("native_submission", "graded"), labels=(),
                                                  grade=True), False
        evaluation = s.last_evaluation
        grade_event = run.last_grade_event
        if grade_event is None or grade_event.payload.get("checkpoint_id") != checkpoint.checkpoint_id:
            raise TechnicalFailure("A submission verdict must come from the grade of its own submitted state")
        verdict = dict(run.task.native_submission_verdict(evaluation))
        status = verdict.get("status")
        if status not in {"pass", "fail", "error"}:
            run.recorder.emit("lifecycle", payload={"transition": "native_submission_unavailable",
                "rule": NATIVE_SUBMISSION_RULE, "checkpoint_id": checkpoint.checkpoint_id, "verdict": verdict})
            raise TechnicalFailure("Native submission verdict unavailable for the submitted state")
        criterion = verdict.get("criterion_id")
        previous = run.native_submissions[-1] if run.native_submissions else None
        consecutive = (previous["consecutive_same"] + 1 if previous and status != "pass"
                       and previous.get("criterion_id") == criterion else 1)
        record = {"ordinal": len(run.native_submissions) + 1, "status": status, "criterion_id": criterion,
                  "consecutive_same": consecutive if status != "pass" else 0, "checkpoint_id": checkpoint.checkpoint_id,
                  "evaluation_id": evaluation.evaluation_id if evaluation is not None else None,
                  "grade_reused_from_submitting_step": reused, "step_index": s.agent.n_calls,
                  "failing_count": verdict.get("failing_count"), "share_passed": verdict.get("share_passed"),
                  "grade_event_id": grade_event.event_id if grade_event is not None else None}
        try:
            record.update(submission_progress(run.config.metadata.get("submission_stop"), previous, verdict))
        except ValueError as exc:
            raise TechnicalFailure(str(exc)) from exc
        run.native_submissions.append(record)
        if status == "pass":
            outcome = "native_success"
        elif consecutive >= limit:
            outcome = "native_repeated_failure"
        elif record.get("no_improvement_count", 0) >= 3:
            outcome = "native_no_improvement"
        elif run.submission_cap is not None and len(run.native_submissions) >= run.submission_cap:
            outcome = "submission_cap"
        else:
            outcome = None
        run.recorder.emit("lifecycle", payload={"transition": "native_submission_verdict", "rule": NATIVE_SUBMISSION_RULE,
            "v2_run_policy": run.v2, "protocol": protocol, **record, "submission_cap": run.submission_cap,
            "outcome": outcome or "continue"})
        if outcome is not None:
            return outcome, False
        feedback = verdict.get("feedback")
        if not isinstance(feedback, str):
            raise TechnicalFailure("Native submission verdict lacks a feedback string")
        messages = s.agent.messages
        if not messages or messages[-1].get("role") != "exit":
            raise TechnicalFailure("Native reopen expected a trailing submission exit marker")
        exit_message = messages.pop()
        batch_message, batch_outputs = getattr(s.agent, "batch_outputs", (None, None))
        if batch_message is None or not messages or messages[-1] is not batch_message:
            raise TechnicalFailure("Native reopen could not bind the interrupted submission batch")
        actions = batch_message.get("extra", {}).get("actions", [])
        outputs = list(batch_outputs) + [{"output": "", "returncode": -1,
            "exception_info": "not executed: the batch ended at the submission command"}
            for _ in actions[len(batch_outputs):]]
        observations = s.agent.model.format_observation_messages(batch_message, outputs, s.agent.get_template_vars())
        s.agent.add_messages(*observations)
        for observation in observations:
            delivered = run.recorder.put_bytes(observation["content"].encode("utf-8"),
                media_type="text/plain; charset=utf-8", visibility=Visibility.WORKER_VISIBLE)
            s.agent.observation_refs.append(delivered)
            run.recorder.emit("observation_delivery", call_id=observation.get("tool_call_id"), actor="harness",
                delivered_ref=delivered, visibility=Visibility.WORKER_VISIBLE,
                payload={"role": observation.get("role"), "delivery_status": "appended_to_history",
                         "native_submission_reopen": True, "rule": NATIVE_SUBMISSION_RULE})
        issue = history_replay_issue(messages)
        if issue is not None:
            raise TechnicalFailure(f"Native reopen left an incomplete conversation: {issue}")
        wrapper = feedback_wrapper(run.config)
        from ..experiments.submission_feedback import delivery
        delivered_feedback, feedback_policy = delivery(run.config, feedback, wrapper)
        s.agent.append_user_message(delivered_feedback, kind="native_submission_feedback",
            payload={"rule": NATIVE_SUBMISSION_RULE, "submission_ordinal": record["ordinal"],
                     "feedback_wrapper": None if wrapper is None else {k: wrapper[k] for k in ("version", "template_sha256")},
                     "submission_feedback": feedback_policy,
                     "native_feedback_sha256": hashlib.sha256(feedback.encode("utf-8")).hexdigest(),
                     "criterion_id": criterion, "checkpoint_id": checkpoint.checkpoint_id,
                     "evaluation_id": record["evaluation_id"], "step_index": record["step_index"],
                     "reopened_exit": exit_message.get("extra", {}).get("exit_status")})
        if self.detector_config.rule_version in SUBMISSION_VERDICT_RULE_VERSIONS and not run.detector_disabled:
            outputs = self._detector_input(run, {"type": "submission_verdict", "scope_id": s.scope.detector_scope_id,
                "checkpoint_id": checkpoint.checkpoint_id, "status": status, "criterion_id": criterion})
            if any(o.get("record") == "grade_request" for o in outputs):
                self._checkpoint(run, reasons=("graded", "repeated_rejection", "point", "post_feedback"), labels=(),
                                 grade=True)
                return None, True
        return None, False

    # ======================================================= v2 branch intervention
    def _v2_intervene(self, run: _Run, treatment: TreatmentManifest) -> TreatmentManifest:
        """Run the branch's review or note turn (if any) at the source checkpoint, then compile the branch.

        The plan (experiments/branch_v2.py) turns the spec and the turn's outcome into ordinary same-run donors per
        channel, context edits, the hand-back and any injection; this records the choice durably and returns the
        compiled treatment, which then goes through the unchanged donor guards."""
        from ..experiments.context_edit import EDITS_VERSION, OUTPUTS_RULE
        plan = run.v2_branch_plan
        store = run.store
        source = store.get_checkpoint(treatment.source_checkpoint_id)
        history = store.get_json(source.state.history)
        from ..experiments.branch_v2 import BranchError
        spec = plan.turn()
        turn = self._v2_turn(run, spec, history, source) if spec is not None else None
        if turn is not None and turn.get("stopped") == CONTEXT_EXIT_STATUS:
            # the turn's request does not fit even with every tool output hidden: the branch ends before its first
            # continuation call with the context-limit end reason (never a silent change-nothing)
            run.v2_stop_before_first_call = CONTEXT_EXIT_STATUS
        try:
            compiled = plan.compile(turn)
        except BranchError as e:  # a rule-valid choice the engine or the Ledger files cannot carry out
            raise TechnicalFailure(f"Branch-v2 choice could not be compiled: {e}") from e
        composition = compiled.get("workspace_composition")
        if composition is not None:
            data = composition.pop("archive")
            manifest = fidelity_manifest_from_tar(data)
            composition.update(workspace_ref=to_jsonable(run.recorder.put_bytes(data, media_type="application/x-tar")),
                               manifest_ref=to_jsonable(run.recorder.put_json(manifest)), entries=len(manifest))
        choice = compiled.get("choice")
        if choice is not None:
            run.recorder.persist(lambda: store.put_record("choice", run.config.run_id, choice, run_id=run.config.run_id),
                                 evidence=choice)
        compiled["turn"] = turn
        run.v2_compiled = compiled
        record = compiled["record"]
        run.recorder.emit("lifecycle", payload={"transition": "branch_v2_compiled", **record, "choice": choice,
                                                "workspace_composition": composition})
        metadata = dict(treatment.metadata)
        metadata["branch_v2"] = {**dict(metadata["branch_v2"]), "compiled": record}
        edits = list(compiled["context_edits"])
        if edits:
            metadata["context_edits"] = {"version": EDITS_VERSION, "edits": edits, "outputs_rule": OUTPUTS_RULE}
        if composition is not None:
            metadata["workspace_composition"] = composition
        return replace(treatment, component_donors=dict(compiled["donors"]),
                       history_policy="edited" if edits else "native", metadata=metadata,
                       intended_mismatches=tuple(compiled["intended_mismatches"]))

    def _v2_turn(self, run: _Run, spec: dict[str, Any], history: dict[str, Any],
                 source: Checkpoint | None = None) -> dict[str, Any]:
        """One-shot review/note turn of the recording agent (V2_TURN_RULE, or V3_TURN_RULE when the turn asks for the
        head-only request); its history is ephemeral."""
        mode = spec["mode"]
        from ..experiments.context_edit import steps_of
        request_context = spec.get("request", "full")
        if request_context == "full":
            start, rule = history["messages"], V2_TURN_RULE
        elif request_context == "head":  # Q24(a): the source context's step 0 (system and task) only
            start, rule = [history["messages"][i] for i in steps_of(history["messages"])[0]], V3_TURN_RULE
            if [m.get("role") for m in start] != ["system", "user"]:
                raise TechnicalFailure("The source context does not start with exactly a system and a task message")
        else:
            raise TechnicalFailure(f"unknown intervention turn request context {request_context!r}")
        model = self.model_factory(run.config)
        if model.effective_request_policy() != run.config.worker_request_policy:
            raise TechnicalFailure("Intervention turn request policy differs from declared configuration")
        if model.config.get("provider") != "scripted" and (model.model_name != run.config.model
                or model.model_settings != dict(run.config.model_settings)):
            raise TechnicalFailure("Intervention turn model/route settings differ from declared configuration")
        model.bind_execution(run.config.worker_request_policy)
        model.unresolved_retry = unresolved_retry_rule(run.config.metadata.get("unresolved_retry"))
        saved = (model.config["tools"], model.action_parser)
        model.config["tools"], model.action_parser = list(spec["tools"]), spec["parser"]
        calls: list[dict[str, Any]] = []
        result = stopped = None
        try:
            agent = RecordingAgent(model, None, recorder=run.recorder, ledger=run.ledger,
                                   generated_token_limit=spec["token_limit"], max_model_calls=None,
                                   system_template=self.system_template,
                                   instance_template=self._instance_template(run.config),
                                   max_consecutive_format_errors=0)
            agent.tool_state_policy, agent.turn_mode = "none", mode
            self._bind_context_limit(run, agent, source)  # the turn's request falls under the same rule
            agent.messages = deepcopy(start)
            agent.extra_template_vars |= {"task": history["task"]}
            run.recorder.emit("lifecycle", payload={"transition": "intervention_turn_started", "rule": rule,
                "mode": mode, "prompt": spec["binding"], "tools": [t["function"]["name"] for t in spec["tools"]],
                "max_calls": 2, "token_limit": spec["token_limit"], "context_messages": len(start),
                "source_context_messages": len(history["messages"]), "request_context": request_context,
                "settings": model.model_settings})
            agent.append_user_message(spec["text"], kind="intervention_prompt",
                                      payload={"mode": mode, "prompt_version": spec["binding"].get("version"),
                                               "prompt_sha256": spec["binding"].get("sha256")})
            for number in (1, 2):
                try:
                    message = agent.query()
                except FormatError as e:
                    agent.add_messages(*e.messages)  # the error message, for the one retry
                    calls.append({"call": number, "valid": False, "error": e.messages[0]["content"],
                                  "request_ref": to_jsonable(agent.last_request_ref)})
                    continue
                except LimitsExceeded as e:  # the branch allowance or exact accounting ran out in the turn
                    stopped = type(e).__name__
                    break
                problem = spec["check"](message) if spec.get("check") else None
                if problem is not None:
                    agent.append_user_message(problem, kind="intervention_invalid", payload={"mode": mode, "call": number})
                    calls.append({"call": number, "valid": False, "error": problem,
                                  "request_ref": to_jsonable(agent.last_request_ref)})
                    continue
                calls.append({"call": number, "valid": True, "request_ref": to_jsonable(agent.last_request_ref)})
                result = {"actions": message["extra"]["actions"], "content": message.get("content"),
                          "reasoning": message.get("reasoning") or message.get("reasoning_content")}
                break
        except ModelAbort as e:
            raise TechnicalFailure(f"intervention turn model call failed: {e}") from e
        finally:
            model.config["tools"], model.action_parser = saved
        responses = [m for m in agent.messages[len(start):] if m.get("role") == "assistant"]
        totals = {"calls": agent.n_calls, "known_generated_tokens": agent.known_generated_tokens,
                  "unknown_usage_calls": agent.unknown_usage_calls, "attempt_cost_known": agent.attempt_cost_known,
                  "attempt_cost_unknown": agent.attempt_cost_unknown,
                  "attempt_cost_worst_case": agent.attempt_cost_worst_case,
                  "charged_unknown_attempts": agent.charged_unknown_attempts,
                  "worst_case_generated_tokens": agent.worst_case_generated_tokens}
        accepted = None  # the valid response as it entered the turn's context (slim provenance, no request copy)
        if result is not None and agent.messages and agent.messages[-1].get("role") == "assistant":
            last = agent.messages[-1]
            accepted = {k: deepcopy(v) for k, v in last.items() if k != "extra"}
            accepted["extra"] = {k: (last.get("extra") or {}).get(k) for k in ("call_id", "response_event_sequence")}
        outcome = {"mode": mode, "valid": result is not None, "calls": calls, "stopped": stopped, "result": result,
                   "accepted_message": accepted, "request_context": request_context,
                   "totals": totals, "prompt": spec["binding"], "prompt_text": spec["text"],
                   "turn_texts": [spec["text"]] + [c["error"] for c in calls if not c["valid"]],
                   "response_texts": [t for m in responses for t in (m.get("content"), m.get("reasoning"),
                                                                     m.get("reasoning_content")) if isinstance(t, str)]}
        run.recorder.emit("lifecycle", payload={"transition": "intervention_turn_finished", "rule": rule,
            "mode": mode, "valid": outcome["valid"], "calls": calls, "stopped": stopped, "totals": totals})
        return outcome

    def _v2_deliver(self, run: _Run, agent: RecordingAgent, history: dict[str, Any]) -> None:
        """After the kept context: the hand-back, any injected steps, the turn's spend, then the request check."""
        from ..experiments.context_edit import apply_edits, validate_request
        from ..experiments.v2_text import REVIEW_TOOL_NAME
        compiled = run.v2_compiled
        kept = list(agent.messages)
        _, report = apply_edits(history["messages"], list(compiled["context_edits"]))
        appended: list[dict[str, Any]] = []
        kept_output = compiled.get("kept_output")
        if kept_output is not None:  # arm set v3 (Q1): the review turn's own output, right after the kept context
            from ..experiments.branch_v2 import KEPT_OUTPUT_RULE
            agent.add_messages(deepcopy(kept_output))
            sent_form = {k: v for k, v in kept_output.items() if k != "extra"}
            run.recorder.emit("lifecycle", payload={"transition": "review_output_kept", "rule": KEPT_OUTPUT_RULE,
                "message_index": len(agent.messages) - 1, "mode": kept_output["extra"].get("mode"),
                "source_call_id": kept_output["extra"].get("call_id"),
                "source_response_event_sequence": kept_output["extra"].get("response_event_sequence"),
                "message_ref": to_jsonable(run.recorder.put_json(sent_form, visibility=Visibility.WORKER_VISIBLE)),
                "sha256": hashlib.sha256(json.dumps(sent_form, sort_keys=True).encode("utf-8")).hexdigest()})
            appended.append(agent.messages[-1])
        handback = compiled.get("handback")
        if handback is not None:
            agent.append_user_message(handback["text"], kind="handback_delivery",
                payload={"rule": handback["rule"], "parts": handback["parts"],
                         "sha256": hashlib.sha256(handback["text"].encode("utf-8")).hexdigest()})
            appended.append(agent.messages[-1])
        injection = compiled.get("injection")
        if injection is not None and injection.get("messages"):
            spliced = deepcopy(injection["messages"])
            agent.add_messages(*spliced)
            refs = []
            for message in spliced:
                if message.get("role") == "tool" and isinstance(message.get("content"), str):
                    ref = run.recorder.put_bytes(message["content"].encode("utf-8"), media_type="text/plain; charset=utf-8",
                                                 visibility=Visibility.WORKER_VISIBLE)
                    agent.observation_refs.append(ref)
                    refs.append(to_jsonable(ref))
            run.recorder.emit("lifecycle", payload={"transition": "injection_spliced", **injection["record"],
                "messages_ref": to_jsonable(run.recorder.put_json(spliced, visibility=Visibility.WORKER_VISIBLE)),
                "observation_refs": refs, "position": "after_last_step"})
            appended += spliced
        totals = (compiled.get("turn") or {}).get("totals") or {}
        agent.known_generated_tokens += totals.get("known_generated_tokens", 0)
        agent.unknown_usage_calls += totals.get("unknown_usage_calls", 0)
        agent.attempt_cost_known += totals.get("attempt_cost_known", 0.0)
        agent.attempt_cost_unknown += totals.get("attempt_cost_unknown", 0)
        agent.attempt_cost_worst_case += totals.get("attempt_cost_worst_case", 0.0)
        agent.charged_unknown_attempts += totals.get("charged_unknown_attempts", 0)
        agent.worst_case_generated_tokens += totals.get("worst_case_generated_tokens", 0)
        agent.intervention_calls = totals.get("calls", 0)
        # the context-limit rule first, exactly as the first continuation call applies it (hidden outputs are logged
        # and stay in the context); a context that cannot fit ends the branch before that call
        hidden: list[int] = []
        limit_record = None
        if agent.context_limit is not None and run.v2_stop_before_first_call is None:
            try:
                fitted = agent.fit_context()
                hidden = [m["message_index"] for m in fitted["masked"]]
                limit_record = {"fits": True, "outputs_hidden_now": len(hidden), "size": fitted["size_after"],
                                "window_tokens": fitted["window_tokens"], "ratio": fitted["ratio"]}
            except ContextWindowExhausted:
                run.v2_stop_before_first_call = CONTEXT_EXIT_STATUS
                limit_record = {"fits": False}
        # serialized exactly as the next worker call will send it, the remaining allowance included
        agent.model.allowance_remaining = agent._remaining_allowance
        try:
            body = agent.model.prepare_request(agent.messages).body
        finally:
            agent.model.allowance_remaining = None
        validation = validate_request(body, history["messages"], kept, report, appended=appended,
                                      forbidden=tuple(compiled.get("forbidden") or ()),
                                      tools=agent.model.config["tools"], tool_name=REVIEW_TOOL_NAME,
                                      hidden=hidden, kept_output=kept_output)
        ref = run.recorder.put_bytes(body, media_type="application/json", visibility=Visibility.WORKER_VISIBLE)
        rule = V3_DELIVERY_RULE if (compiled.get("record") or {}).get("spec_version") == "branch-spec.v3" \
            else V2_DELIVERY_RULE
        run.recorder.emit("lifecycle", payload={"transition": "next_request_validated", "rule": rule,
                                                **validation, "request_ref": to_jsonable(ref),
                                                **({"context_limit": limit_record} if limit_record else {})})
        if not validation["valid"]:
            failed = sorted(k for k, v in validation["checks"].items() if v is False)
            raise TechnicalFailure(f"The serialized next request failed validation: {failed}")

    def _step(self, run: _Run) -> str | None:
        """One mini-SWE-agent step with the pinned exception semantics. Returns exit status if any."""
        s = run.session
        assert s is not None
        agent = s.agent
        from ..baselines.agentrewind import RewindRequested
        try:
            agent.step()
            agent.n_consecutive_format_errors = 0
        except RewindRequested as exc:
            self._apply_baseline_rewind(run, exc.plan)
        except FormatError as e:
            agent.n_consecutive_format_errors += 1
            if 0 < agent.config.max_consecutive_format_errors <= agent.n_consecutive_format_errors:
                agent.add_messages(*e.messages, {"role": "exit", "content": "RepeatedFormatError",
                                                 "extra": {"exit_status": "RepeatedFormatError", "submission": ""}})
            else:
                agent.add_messages(*e.messages)
        except WorkerBudgetExhausted as e:
            agent.add_messages(*e.messages)
        except InterruptAgentFlow as e:  # Submitted, LimitsExceeded
            agent.add_messages(*e.messages)
        except ModelAbort as e:
            raise TechnicalFailure(f"model call failed: {e}") from e
        except ToolTransportFailure as e:
            raise TechnicalFailure(f"tool transport failed ({e.result.status})") from e
        last = agent.messages[-1]
        return last.get("extra", {}).get("exit_status") if last.get("role") == "exit" else None

    def _install_baseline(self, run: _Run, source: Checkpoint) -> None:
        from ..baselines.agentrewind import (AgentRewindBaseline, BaselinePolicyError, POLICY,
                                            SYSTEM_PROMPT, RewindRequested, parse_actions)
        s = run.session
        assert s is not None
        controller = AgentRewindBaseline(run.store, source_checkpoint_id=source.checkpoint_id, run_id=run.config.run_id)
        run.baseline = controller
        s.agent.model.action_parser = parse_actions
        s.agent.model.config["tools"] = [*s.agent.model.config["tools"], *controller.tools]
        s.agent.model.config["recovery_baseline"] = POLICY
        s.agent.add_messages({"role": "system", "content": SYSTEM_PROMPT,
                              "extra": {"agentrewind_policy": POLICY["policy_id"]}})
        s.agent.add_messages({"role": "user", "content": self._worker_task_prompt(run, s.stage),
                              "extra": {"agentrewind_fixed_target": True}})
        run.recorder.emit("lifecycle", payload={"transition": "baseline_tools_installed", "policy": POLICY,
            "disclosed_messages_ref": to_jsonable(run.recorder.put_json(s.agent.messages[-2:], visibility=Visibility.WORKER_VISIBLE))})

        def control(action: dict, sequence: int) -> dict:
            try:
                args = action["arguments"]
                if action["tool_name"] == "backtrack_candidates":
                    return controller.candidates(response_event_sequence=sequence, **args)
                if action["tool_name"] == "backtrack_commit":
                    raise RewindRequested(controller.prepare_commit(response_event_sequence=sequence, **args))
                raise BaselinePolicyError("Unknown recovery tool")
            except BaselinePolicyError as exc:
                return {"error": str(exc), "state_restored": False}
        s.agent.control_handler = control

    @staticmethod
    def _verify_environment_binding(run: _Run, donor: Checkpoint, env: DockerWorkerEnvironment) -> None:
        original = run.store.get_json(donor.state.harness)["environment"]
        actual = env.binding()
        if not original.get("image_id") or original["image_id"] != actual.get("image_id"):
            raise TechnicalFailure("Restore base-image binding differs from captured donor")
        for key in ("architecture", "env", "network", "limits", "cwd", "interpreter", "worker_uid", "worker_gid",
                    "container_user", "cap_drop"):
            if key in original and original[key] != actual.get(key):
                raise TechnicalFailure(f"Restore environment binding differs from donor: {key}")
        if original.get("command_execution") != actual.get("command_execution"):
            raise TechnicalFailure("Restore command execution policy differs from donor or is unrecorded")

    def _apply_baseline_rewind(self, run: _Run, plan: Any) -> None:
        s = run.session
        assert s is not None and run.baseline is not None
        self._drain_usage(run)  # commit inference belongs to the abandoned suffix's clock

        def restore(donor: Checkpoint) -> dict:
            env = self._new_env(run)
            try:
                def load_workspace():
                    self._verify_environment_binding(run, donor, env)
                    w_data = run.store.get_bytes(donor.state.workspace)
                    fidelity = run.store.get_json(_ref(donor.validation["workspace_manifest_ref"]))
                    return w_data, fidelity

                w_data, fidelity = self._timed_call(run, "baseline_workspace_input", load_workspace, role="infrastructure")
                wr = self._timed_call(run, "baseline_workspace_restore", lambda: env.restore_workspace(w_data, fidelity))
                if donor.state.runtime_class == RuntimeClass.CAPTURED:
                    def load_runtime():
                        runtime = run.store.get_json(donor.state.runtime)
                        runtime["archive"] = run.store.get_bytes(_ref(runtime["archive_ref"])) if runtime.get("archive_ref") else None
                        return runtime

                    runtime = self._timed_call(run, "baseline_runtime_input", load_runtime, role="infrastructure")
                    rr = self._timed_call(run, "baseline_runtime_restore", lambda: env.restore_runtime(runtime))
                elif donor.state.runtime_class == RuntimeClass.IRRELEVANT:
                    rr = {"validated": True, "policy": "adapter_declared_irrelevant"}
                else:
                    raise TechnicalFailure("Baseline donor runtime unsupported")
            except BaseException as error:
                self._close_after_failure(run, env, error)
                raise
            old_env = s.env
            try:
                self._timed_call(run, "environment_close", old_env.close, role="infrastructure")
            except BaseException as error:
                self._close_after_failure(run, env, error)
                raise
            s.env = s.agent.env = env
            return {"workspace_donor": donor.checkpoint_id, "runtime_donor": donor.checkpoint_id,
                    "workspace": wr, "runtime": rr}
        try:
            run.baseline.apply(plan, s.agent, restore)
            s.scope = self._make_scope(run, s.stage)
            run.recorder.scope = s.scope
            self._detector_input(run, {"type": "scope_start", "scope_id": s.scope.detector_scope_id,
                "stage_id": s.stage.stage_id, "reason": "baseline_rewind"})
            def refresh_state():
                s.w_manifest = s.env.workspace_manifest()
                s.w_fidelity = s.env.workspace_fidelity_manifest()
                s.r_manifest = s.env.runtime_manifest()
                return None

            self._timed_call(run, "baseline_state_probe", refresh_state, role="infrastructure")
            s.agent.batch_mutated = False
            self._checkpoint(run, reasons=("baseline_rewind", "graded"), labels=(), grade=True, scope_baseline=True)
        except Exception as exc:
            if hasattr(exc, "persistence_evidence"):
                raise
            raise TechnicalFailure(f"Baseline rewind failed: {exc}") from exc

    def _drain_usage(self, run: _Run) -> None:
        s = run.session
        assert s is not None
        log = s.agent.usage_log
        while s.usage_cursor < len(log):
            seq, usage = log[s.usage_cursor]
            s.usage_cursor += 1
            self._detector_input(run, {"type": "usage", "generated_tokens": usage.generated_tokens, "response_event_sequence": seq})

    def _mutation(self, run: _Run) -> bool:
        s = run.session
        assert s is not None
        before_w, before_r = s.w_manifest, s.r_manifest
        after_fidelity = s.env.workspace_fidelity_manifest()
        after_w = loop_manifest_from_fidelity(after_fidelity)
        after_r = s.env.runtime_manifest()
        changed = sorted(set(before_w.items()) ^ set(after_w.items()))
        files = []
        for path in sorted({p for p, _ in changed}):
            files.append({"path": path, "before": before_w.get(path), "after": after_w.get(path)})
        b_entries, a_entries = comparable(before_r["entries"]), comparable(after_r["entries"])
        r_changed = sorted({p for p in set(b_entries) | set(a_entries) if b_entries.get(p) != a_entries.get(p)}
                           | (set(before_r["deleted"]) ^ set(after_r["deleted"])))
        before_full, after_full = comparable(s.w_fidelity), comparable(after_fidelity)
        fidelity_changes = sorted(p for p in set(before_full) | set(after_full)
                                  if before_full.get(p) != after_full.get(p))
        mounts_changed = before_r.get("mounts") != after_r.get("mounts")
        relevant = bool(fidelity_changes or r_changed or mounts_changed or s.agent.batch_mutated)
        s.w_manifest, s.r_manifest = after_w, after_r
        s.w_fidelity = after_fidelity
        if self.detector_config.rule_version not in SIMPLE_INPUT_RULE_VERSIONS:
            return self._mutation_v3(run, before_w, before_r, before_full, after_w, after_r, after_full)
        ev = run.recorder.emit("mutation", payload={"rule": MUTATION_RULE, "relevant": relevant,
                                                    "workspace_before": manifest_digest(before_w), "workspace_after": manifest_digest(after_w),
                                                    "files": files, "workspace_fidelity_paths_changed": fidelity_changes,
                                                    "runtime_paths_changed": r_changed,
                                                    "mounted_state_changed": mounts_changed,
                                                    "within_batch_mutation": s.agent.batch_mutated,
                                                    "response_event_sequence": s.agent.last_response_event_sequence})
        self._detector_input(run, {"type": "mutation", "relevant": relevant, "workspace_before": manifest_digest(before_w),
                                   "workspace_after": manifest_digest(after_w), "files": files, "event_sequence": ev.sequence})
        return relevant

    def _mutation_v3(self, run: _Run, before_w: dict, before_r: dict, before_full: dict,
                     after_w: dict, after_r: dict, after_full: dict) -> bool:
        """Feed retained per-action states once, then any measured post-action gap."""
        s = run.session
        assert s is not None
        record = run.task.describe()
        state_events = list(getattr(s.agent, "batch_state_events", []))
        complete = len(state_events) == len(getattr(s.agent, "completed_in_batch", []))
        if s.agent.batch_mutated and not state_events:
            complete = False
        all_w, all_r, mounts = set(), set(), False

        def emit_change(bw, br, bf, aw, ar, af, source_event=None):
            nonlocal mounts
            w_paths = sorted(p for p in set(bf) | set(af) if bf.get(p) != af.get(p))
            be, ae = comparable(br["entries"]), comparable(ar["entries"])
            r_paths = sorted({p for p in set(be) | set(ae) if be.get(p) != ae.get(p)}
                | (set(br["deleted"]) ^ set(ar["deleted"])))
            mount_change = br.get("mounts") != ar.get("mounts")
            all_w.update(w_paths); all_r.update(r_paths); mounts |= mount_change
            if not (w_paths or r_paths or mount_change):
                return
            decision = evaluate_mutation(task_id=record.task_id, task_version=record.task_version,
                workspace_paths=w_paths, runtime_paths=r_paths, mounts_changed=mount_change,
                within_batch_mutation=False, intermediate_evidence_complete=True, policy=run.mutation_policy)
            files = [{"path":p,"before":bw.get(p),"after":aw.get(p)}
                     for p in sorted(set(bw) | set(aw)) if bw.get(p) != aw.get(p)]
            payload = {"rule":"mutation-input/v4", "policy_result":decision,
                "relevant":decision["task_relevant"], "workspace_before":manifest_digest(bw),
                "workspace_after":manifest_digest(aw), "files":files,
                "source_tool_state_event_id":source_event.event_id if source_event else None,
                "source_tool_state_event_sequence":source_event.sequence if source_event else None,
                "response_event_sequence":s.agent.last_response_event_sequence}
            event = run.recorder.emit("mutation", payload=payload)
            self._detector_input(run,{"type":"mutation",**payload,"event_sequence":event.sequence,
                                      "failure_relevance":decision["failure_relevance"]})

        last_w, last_r, last_full = before_w, before_r, before_full
        for source in state_events:
            if source.event_id in s.mutation_source_events_seen:
                continue
            if source.scope != s.scope or source.payload.get("status") != "captured" or not source.input_ref or not source.output_ref:
                raise TechnicalFailure("Intermediate mutation evidence has an incompatible scope or missing state")
            before, after = run.store.get_json(source.input_ref), run.store.get_json(source.output_ref)
            bw, aw = loop_manifest_from_fidelity(before["workspace"]), loop_manifest_from_fidelity(after["workspace"])
            bf, af = comparable(before["workspace"]), comparable(after["workspace"])
            emit_change(last_w,last_r,last_full,bw,before["runtime"],bf)
            emit_change(bw,before["runtime"],bf,aw,after["runtime"],af,source)
            last_w,last_r,last_full = aw,after["runtime"],af
            s.mutation_source_events_seen.add(source.event_id)
        emit_change(last_w,last_r,last_full,after_w,after_r,after_full)
        decision = evaluate_mutation(task_id=record.task_id,task_version=record.task_version,
            workspace_paths=all_w,runtime_paths=all_r,mounts_changed=mounts,
            within_batch_mutation=s.agent.batch_mutated,intermediate_evidence_complete=complete,
            policy=run.mutation_policy)
        run.recorder.emit("mutation_summary",payload={"rule":"mutation-input/v4","policy_result":decision,
            "source_tool_state_event_ids":[e.event_id for e in state_events],"duplicate_detector_mutation":False,
            "response_event_sequence":s.agent.last_response_event_sequence})
        return decision["grade_required"]

    # ============================================================== checkpoint
    def _failed_checkpoint_capture(self, run: _Run, error: CaptureError, *, checkpoint_id: str,
            attempt: int, reasons: tuple[str, ...], capture_started: float,
            workspace_bytes: bytes | None, workspace_manifest: dict | None) -> None:
        """Preserve a failed attempt, without certifying or publishing a checkpoint.

        The existing second capture attempt is allowed only after this receipt
        succeeds. Storage failure retains the original capture exception and
        available bytes; it must not trigger another environment capture.
        """
        capture = dict(getattr(error, "capture_evidence", {}))
        error.capture_evidence = {"checkpoint_id": checkpoint_id, "attempt": attempt,
            "capture": capture, "workspace_bytes": workspace_bytes,
            "workspace_manifest": workspace_manifest}
        diagnostic = {"schema": "failed-capture-diagnostic.v1", "restorable": False,
            "branchable": False, "validated_snapshot": False,
            "checkpoint_id": checkpoint_id, "attempt": attempt,
            "capture": {k: v for k, v in capture.items() if k != "archive_bytes"},
            "workspace_manifest": workspace_manifest, "failed_archive_ref": None,
            "workspace_diagnostic_archive_ref": None}
        event = None
        try:
            if capture.get("archive_bytes") is not None:
                diagnostic["failed_archive_ref"] = to_jsonable(run.recorder.put_bytes(
                    capture["archive_bytes"], media_type="application/x-tar"))
            if workspace_bytes is not None:
                diagnostic["workspace_diagnostic_archive_ref"] = to_jsonable(run.recorder.put_bytes(
                    workspace_bytes, media_type="application/x-tar"))
            ref = run.recorder.put_json(diagnostic)
            event = run.recorder.emit("capture", output_ref=ref, payload={
                "status": "failed", "failure_kind": "capture_failure", "checkpoint_id": checkpoint_id,
                "attempt": attempt, "reasons": list(reasons), "error": str(error),
                "branchable_checkpoint": False, "diagnostic_complete": True})
        except BaseException as cleanup:
            error.capture_diagnostic_persistence_failure = cleanup
            run.recorder.capture_failure = error
            error.add_note(f"Checkpoint capture diagnostic persistence also failed: {type(cleanup).__name__}")
            raise error
        finally:
            elapsed = time.monotonic() - capture_started
            error.capture_evidence["capture_duration_seconds"] = elapsed
            if event is not None:
                try:
                    run.ledger.record(run_id=run.config.run_id, operation_id=f"capture:{event.event_id}",
                        role="infrastructure", kind="local", status="error",
                        usage=Usage(elapsed_seconds=elapsed, measurement_basis="measured_capture_seconds"),
                        source_event_id=event.event_id, source_event_sequence=event.sequence)
                except BaseException as cleanup:
                    error.capture_diagnostic_persistence_failure = cleanup
                    run.recorder.capture_failure = error
                    run.recorder.persistence_failure = cleanup
                    error.add_note(f"Checkpoint capture cost persistence also failed: {type(cleanup).__name__}")
                    raise error

    def _checkpoint(self, run: _Run, *, reasons: tuple[str, ...], labels: tuple[str, ...], grade: bool,
                    scope_baseline: bool = False) -> Checkpoint:
        capture = self._capture(run, reasons=reasons)
        return self._checkpoint_from(run, capture, reasons=reasons, labels=labels, grade=grade,
                                     scope_baseline=scope_baseline)

    def _capture(self, run: _Run, *, reasons: tuple[str, ...]) -> dict[str, Any] | None:
        """Capture W and R for the next checkpoint ordinal (retry once); nothing is certified here."""
        s = run.session
        assert s is not None
        env = s.env
        ordinal = len(run.checkpoints)
        checkpoint_id = f"{run.config.run_id}/C{ordinal}"
        if run.config.metadata.get("capture_gap_policy") is not None:
            # Failed point occurrences cannot share an identity with a later
            # successful checkpoint merely because no checkpoint was published.
            checkpoint_id += f"-S{s.agent.n_calls}"
        # ---- capture native state (retry once on capture error)
        attempt_errors: list[str] = []
        last_capture_error = None
        for attempt in (1, 2):
            capture_started = time.monotonic()
            # Never attach a prior attempt's workspace to a later failed one.
            w_data = w_fidelity = None
            try:
                w_data, w_fidelity = env.capture_workspace()
                runtime = env.capture_runtime()
                break
            except CaptureError as e:
                last_capture_error = e
                attempt_errors.append(str(e))
                self._failed_checkpoint_capture(run, e, checkpoint_id=checkpoint_id, attempt=attempt,
                    reasons=reasons, capture_started=capture_started,
                    workspace_bytes=w_data, workspace_manifest=w_fidelity)
        else:
            from .capture_gaps import POLICY as CAPTURE_GAP_POLICY, diagnose
            if (run.config.metadata.get("capture_gap_policy") == CAPTURE_GAP_POLICY
                    and run.v2 == V2_TECHNICAL_RESUME and reasons == ("step_end",)
                    and run.native_exit_status is None):
                diagnostic = diagnose(env, attempt_errors)
                if diagnostic is not None:
                    history = {"messages": s.agent.messages, "counters": s.agent.counters(),
                               "cursor": s.agent.cursor(), "checkpoint_eligible": False}
                    run.recorder.emit("checkpoint_gap", output_ref=run.recorder.put_json(history), payload={
                        "rule": CAPTURE_GAP_POLICY, "step_index": s.agent.n_calls,
                        "occurrence": run.capture_gap_count + 1, "errors": attempt_errors,
                        "diagnostic": diagnostic, "branchable": False, "grade": None,
                        "workspace_and_runtime_snapshot": None,
                        "previous_checkpoint_id": run.checkpoints[-1].checkpoint_id if run.checkpoints else None,
                        "worker_context_changed": False})
                    run.capture_gap_count += 1
                    run.capture_gap_pending = True
                    return None
            error = TechnicalFailure(f"capture failed twice at {checkpoint_id}: {attempt_errors[-1]}")
            run.checkpoint_failure = error
            run.recorder.capture_failure = last_capture_error
            try:
                run.recorder.emit("capture", payload={"checkpoint_id": checkpoint_id, "status": "failed", "reasons": list(reasons),
                                                      "errors": attempt_errors})
            except BaseException as cleanup:
                last_capture_error.capture_diagnostic_persistence_failure = cleanup
                last_capture_error.add_note(f"Capture failure summary persistence also failed: {type(cleanup).__name__}")
                raise last_capture_error
            raise error from last_capture_error
        return {"checkpoint_id": checkpoint_id, "ordinal": ordinal, "w_data": w_data, "w_fidelity": w_fidelity,
                "runtime": runtime, "capture_started": capture_started, "attempt_errors": attempt_errors}

    def _checkpoint_from(self, run: _Run, capture: dict[str, Any], *, reasons: tuple[str, ...],
                         labels: tuple[str, ...], grade: bool, scope_baseline: bool = False) -> Checkpoint:
        """Certify one checkpoint occurrence from one capture; a capture is never reused for a second occurrence."""
        s = run.session
        assert s is not None
        agent = s.agent
        checkpoint_id, ordinal = capture["checkpoint_id"], capture["ordinal"]
        w_data, w_fidelity, runtime = capture["w_data"], capture["w_fidelity"], capture["runtime"]
        if ordinal != len(run.checkpoints) or capture.get("used"):
            raise TechnicalFailure("A capture certifies exactly one checkpoint occurrence at its own ordinal")
        capture["used"] = True
        try:
            return self._checkpoint_from_capture(run, checkpoint_id=checkpoint_id, ordinal=ordinal,
                reasons=reasons, labels=labels, grade=grade, scope_baseline=scope_baseline,
                w_data=w_data, w_fidelity=w_fidelity, runtime=runtime,
                capture_started=capture["capture_started"], attempt_errors=capture["attempt_errors"])
        except BaseException as error:
            run.checkpoint_failure = error
            if (run.recorder.persistence_failure is not None or hasattr(error, "persistence_evidence")
                    or isinstance(error, (OSError, sqlite3.OperationalError))):
                prior = getattr(error, "persistence_evidence", None)
                preserve_failure(error, {"checkpoint_id": checkpoint_id, "workspace_bytes": w_data,
                    "workspace_fidelity": w_fidelity, "runtime": runtime,
                    "history": {"messages": agent.messages, "cursor": agent.cursor(), "counters": agent.counters()},
                    "prior_evidence": prior})
                run.recorder.persistence_failure = error
            raise

    def _checkpoint_from_capture(self, run: _Run, *, checkpoint_id: str, ordinal: int,
            reasons: tuple[str, ...], labels: tuple[str, ...], grade: bool, scope_baseline: bool,
            w_data: bytes, w_fidelity: dict[str, Any], runtime: dict[str, Any],
            capture_started: float, attempt_errors: list[str]) -> Checkpoint:
        # This continuation may grade and feed the detector; never retry it as
        # a unit. Only immutable writes within it may be retried.
        s = run.session
        assert s is not None
        agent = s.agent
        store = run.store
        task_record = run.task.describe()
        runtime = bind_runtime_support(task_record, s.stage, runtime, store=store)
        w_manifest = loop_manifest_from_fidelity(w_fidelity)
        w_ref = run.recorder.put_bytes(w_data, media_type="application/x-tar")
        w_manifest_ref = run.recorder.put_json(w_fidelity)
        w_loop_ref = run.recorder.put_json(w_manifest)
        archive_ref = run.recorder.put_bytes(runtime["archive"], media_type="application/x-tar") if runtime.get("archive") else None
        r_record = {k: v for k, v in runtime.items() if k != "archive"}
        r_record["archive_ref"] = to_jsonable(archive_ref) if archive_ref else None
        r_ref = run.recorder.put_json(r_record)
        runtime_class = RuntimeClass(runtime["runtime_class"])
        cursor = agent.cursor()
        history = {"messages": agent.messages, "counters": agent.counters(), "cursor": cursor, "task": agent.extra_template_vars.get("task"),
                   "stage_id": s.stage.stage_id, "session_id": s.scope.session_id}
        h_ref = run.recorder.put_json(history)
        harness_ref = run.recorder.put_json(agent.harness_binding())
        observations = tuple(agent.observation_refs)
        w_digest = manifest_digest(w_manifest)
        r_digest = manifest_digest({"entries": comparable(runtime["entries"]), "deleted": sorted(runtime["deleted"])})
        # Point index: model calls in this history so far (format-error responses count; restored calls included).
        step_index = agent.n_calls
        capture_ev = run.recorder.emit("capture", payload={
            "checkpoint_id": checkpoint_id, "status": "captured", "reasons": list(reasons), "labels": list(labels),
            "workspace_manifest_digest": w_digest, "workspace_fidelity_digest": manifest_digest(comparable(w_fidelity)),
            "workspace_files": len(w_manifest), "workspace_entries": len(w_fidelity), "workspace_bytes": len(w_data),
            "runtime_class": runtime_class.value, "runtime_support": runtime["runtime_support"],
            "runtime_paths": len(runtime["changed"]) + len(runtime["deleted"]),
            "foreign_processes": runtime["foreign_processes"], "cursor": cursor, "capture_errors": attempt_errors,
            "step_index": step_index})
        event_cut = capture_ev.sequence
        run.ledger.record(run_id=run.config.run_id, operation_id=f"capture:{capture_ev.event_id}",
            role="infrastructure", kind="local", status="ok",
            usage=Usage(elapsed_seconds=time.monotonic()-capture_started, measurement_basis="measured_capture_seconds"),
            source_event_id=capture_ev.event_id, source_event_sequence=capture_ev.sequence)
        # Every declared official grading occurrence is a new measurement,
        # even for unchanged state. Content-addressed blobs may still be shared.
        evaluation = evaluation_ref = None
        reused = False
        if grade:
            evaluation, evaluation_ref = self._grade(run, checkpoint_id, w_data, w_manifest, runtime,
                                                     capture_event=capture_ev)
            s.last_graded_state = (manifest_digest(comparable(w_fidelity)), r_digest)
            s.last_evaluation, s.last_evaluation_ref = evaluation, evaluation_ref
        accounting_ref = run.ledger.snapshot(run.config.run_id, through_sequence=event_cut)
        replay_issue = history_replay_issue(agent.messages)
        history_round_trip = store.get_json(h_ref) == to_jsonable(history)
        branchable = (not cursor["pending_actions"] and runtime_class != RuntimeClass.UNSUPPORTED
                      and replay_issue is None and history_round_trip)
        validation = {"workspace_archive_validated_against_live_manifest": True, "workspace_manifest_ref": to_jsonable(w_manifest_ref),
                      "workspace_loop_manifest_ref": to_jsonable(w_loop_ref), "workspace_manifest_digest": w_digest,
                      "workspace_fidelity_digest": manifest_digest(comparable(w_fidelity)), "runtime_digest": r_digest, "runtime_coverage": runtime["coverage"],
                      "runtime_rule": runtime["rule"], "runtime_support": runtime["runtime_support"],
                      "foreign_processes": runtime["foreign_processes"],
                      "pending_actions": len(cursor["pending_actions"]), "history_round_trip": history_round_trip,
                      "evaluation_reused": reused, "capture_errors": attempt_errors,
                      "history_replay_issue": replay_issue, "step_index": step_index,
                      "not_branchable_reason": None if branchable else ("pending_actions" if cursor["pending_actions"]
                          else "runtime_unsupported" if runtime_class == RuntimeClass.UNSUPPORTED
                          else "history_round_trip_failed" if not history_round_trip else replay_issue)}
        # The run's one branch_started event, kept when emitted (a journal rescan here is O(steps) per checkpoint).
        branch_origin = run.branch_origin_event
        capture_binding = {"revision": "checkpoint-source-binding.v1", "capture_event_id": capture_ev.event_id,
            "capture_event_sequence": event_cut, "session_id": s.scope.session_id, "scope": to_jsonable(s.scope),
            "task_record_ref": to_jsonable(run.recorder.put_json(task_record)),
            "target_ref": to_jsonable(run.recorder.put_json(s.stage)),
            "harness_ref": to_jsonable(harness_ref), "runtime_support": runtime["runtime_support"],
            "within_run_parent_checkpoint_id": run.checkpoints[-1].checkpoint_id if run.checkpoints else None,
            "branch_origin_event_id": branch_origin.event_id if branch_origin else None,
            "branch_origin": dict(branch_origin.payload) if branch_origin else None}
        state = NativeState(workspace=w_ref, history=h_ref, harness=harness_ref, runtime_class=runtime_class, runtime=r_ref,
                            observations=observations, execution_cursor=cursor,
                            capture_evidence=capture_binding)
        checkpoint = Checkpoint(run_id=run.config.run_id, scope=s.scope, ordinal=ordinal, event_sequence=event_cut, state=state,
                                reasons=reasons, checkpoint_id=checkpoint_id,
                                parent_checkpoint_id=run.checkpoints[-1].checkpoint_id if run.checkpoints else None,
                                evaluation_ref=evaluation_ref, accounting_ref=accounting_ref, branchable=branchable, validation=validation)
        run.recorder.persist(lambda: store.save_checkpoint(checkpoint), evidence=checkpoint)
        run.checkpoints.append(checkpoint)
        if run.checkpoint_published is not None:
            # A wakeup only: immutable checkpoints are the durable external
            # observer outbox. Never execute extraction or await T here. A lost
            # notification is recovered by the external checkpoint scan.
            try:
                run.checkpoint_published(checkpoint)
            except Exception as exc:
                preserve_failure(exc, {"checkpoint_id": checkpoint.checkpoint_id,
                    "operation": "external_observer_wakeup", "recovery": "scan_immutable_checkpoints"})
        if run.baseline is not None and checkpoint.branchable:
            run.baseline.register_checkpoint(checkpoint)
        if grade:
            s.last_graded_checkpoint = checkpoint
        for label in labels:
            self._annotate(run, checkpoint, label, {"origin": "structural", "reasons": list(reasons)}, evidence_cutoff=event_cut)
        # ---- detector
        self._detector_input(run, {"type": "checkpoint", "checkpoint_id": checkpoint_id, "valid": branchable})
        if grade and evaluation is not None:
            self._detector_grade(run, checkpoint, evaluation, w_digest, scope_baseline=scope_baseline)
        return checkpoint

    def _grade(self, run: _Run, checkpoint_id: str, w_data: bytes, w_manifest: dict[str, str],
               runtime: dict[str, Any] | None = None, *, capture_event: Any = None) -> tuple[Evaluation, ArtifactRef]:
        s = run.session
        assert s is not None
        grade_dir = run.scratch / "grades" / checkpoint_id.replace("/", "_")
        ws = grade_dir / "workspace"
        source_binding: dict[str, Any] = {}
        if runtime is not None:
            runtime = bind_runtime_support(run.task.describe(), s.stage, runtime, store=run.store)
        def prepare():
            ws.mkdir(parents=True, exist_ok=True)
            extract_directory(w_data, ws)
            if _manifest_dir(ws) != w_manifest:
                raise TechnicalFailure("isolated grading copy does not match captured workspace manifest")
            # Shared RuntimeExport: read-only host export of the captured out-of-workspace R
            # from the same capture boundary, plus the exact worker image binding.
            export = None
            if runtime is not None:
                runtime_dir = grade_dir / "runtime"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_sha = None
                if runtime.get("archive"):
                    archive_path = runtime_dir / "runtime.tar"
                    archive_path.write_bytes(runtime["archive"])
                    archive_path.chmod(0o400)
                    archive_sha = hashlib.sha256(runtime["archive"]).hexdigest()
                export = RuntimeExport(base_image_id=s.env.image_id, runtime_class=RuntimeClass(runtime["runtime_class"]),
                                       snapshot={"rule": runtime["rule"], "entries": runtime["entries"], "changed": runtime["changed"],
                                                 "deleted": runtime["deleted"], "processes": runtime["processes"],
                                                 "foreign_processes": runtime["foreign_processes"], "coverage": runtime["coverage"],
                                                 "runtime_class": runtime["runtime_class"],
                                                 "runtime_support": runtime["runtime_support"],
                                                 "irrelevance_evidence": runtime.get("irrelevance_evidence"),
                                                 "mounts": runtime.get("mounts"), "initial_mounts": runtime.get("initial_mounts"),
                                                 "mount_state_changed": runtime.get("mount_state_changed"),
                                                 "workspace_fidelity_manifest": fidelity_manifest_from_tar(w_data),
                                                 "workspace_default_owner": [s.env.worker_uid, s.env.worker_gid],
                                                 "process_restore_supported": False},
                                       archive=archive_path, archive_sha256=archive_sha,
                                       workspace_snapshot_sha256=hashlib.sha256(w_data).hexdigest(), environment=dict(s.env.env))
            grade_kwargs: dict[str, Any] = {"stage_id": s.stage.stage_id, "private_output_dir": grade_dir / "private"}
            if export is not None and "runtime" in inspect.signature(run.task.grade).parameters:
                grade_kwargs["runtime"] = export
            archive_ref = (run.recorder.put_bytes(runtime["archive"], media_type="application/x-tar")
                           if runtime is not None and runtime.get("archive") else None)
            runtime_ref = (run.recorder.put_json({**{k: v for k, v in runtime.items() if k != "archive"},
                "archive_ref": to_jsonable(archive_ref) if archive_ref else None}) if runtime is not None else None)
            capture = capture_event if capture_event is not None else next(
                (e for e in reversed(list(run.store.iter_events(run.config.run_id)))
                 if e.kind == "capture" and e.payload.get("checkpoint_id") == checkpoint_id
                 and e.payload.get("status") == "captured"), None)
            source_binding.update({"revision": "evaluation-source-binding.v1", "run_id": run.config.run_id,
                "checkpoint_id": checkpoint_id, "scope": to_jsonable(s.scope),
                "capture_event_id": capture.event_id if capture else None,
                "capture_event_sequence": capture.sequence if capture else None,
                "workspace_ref": to_jsonable(run.recorder.put_bytes(w_data, media_type="application/x-tar")),
                "runtime_ref": to_jsonable(runtime_ref) if runtime_ref else None,
                "runtime_export_consumed": "runtime" in grade_kwargs, "base_image_id": s.env.image_id,
                "task_record_ref": to_jsonable(run.recorder.put_json(run.task.describe())),
                "target_ref": to_jsonable(run.recorder.put_json(s.stage)),
                "harness_ref": to_jsonable(run.recorder.put_json(s.agent.harness_binding()))})
            return grade_kwargs

        grade_kwargs = self._timed_call(run, "grade_prepare", prepare, role="infrastructure")
        started = time.time()
        t0 = time.monotonic()
        evaluation = None
        error = None
        try:
            evaluation = run.task.grade(ws, **grade_kwargs)
            if not isinstance(evaluation, Evaluation):
                raise TypeError("grader must return an Evaluation")
            if (evaluation.task_id, evaluation.stage_id, evaluation.target_id, evaluation.evaluator_version) != (
                    s.scope.task_id, s.stage.stage_id, s.stage.target_id, s.stage.evaluator_version):
                raise ValueError("grader returned an evaluation for a different task, target or evaluator version")
        except BaseException as exc:
            error = exc
        elapsed = time.monotonic() - t0
        try:
            return self._record_grade(run, checkpoint_id, w_data, grade_dir, ws, grade_kwargs,
                                      evaluation, error, started, elapsed, source_binding=source_binding)
        except BaseException as recording_error:
            if recording_error is error or (isinstance(recording_error, TechnicalFailure)
                                            and recording_error.__cause__ is error and error is not None):
                raise  # The recorded grader failure, not a subsequent storage failure.
            # A completed evaluator result and its original private files are
            # still known even if any measurement/receipt write fails.
            primary = error if error is not None else recording_error
            preserve_failure(primary, {"checkpoint_id": checkpoint_id, "evaluation": evaluation,
                "grader_error": error, "workspace_bytes": w_data,
                "private_receipt_directory": str(grade_dir / "private"),
                "started_at": started, "elapsed_seconds": elapsed,
                "recording_error": recording_error,
                "prior_evidence": getattr(recording_error, "persistence_evidence", None)})
            run.recorder.persistence_failure = primary
            if error is not None:
                error.add_note(f"Grader receipt persistence also failed: {type(recording_error).__name__}")
            raise primary

    def _record_grade(self, run: _Run, checkpoint_id: str, w_data: bytes, grade_dir: Path, ws: Path,
            grade_kwargs: dict[str, Any], evaluation: Evaluation | None, error: BaseException | None,
            started: float, elapsed: float, *, source_binding: dict[str, Any]) -> tuple[Evaluation, ArtifactRef]:
        # The evaluator has already run. Only individual prepared writes retry.
        s = run.session
        assert s is not None
        # Commit evaluator time before ingesting its potentially large receipts.
        # A failed/interrupted ingestion must not erase work already performed.
        status = "ok" if error is None and evaluation.valid else "error"
        measured = run.recorder.emit("operation", payload={"name": "grade_evaluate",
            "checkpoint_id": checkpoint_id, "status": status, "started_at": started,
            "elapsed_seconds": elapsed, "error": str(error) if error is not None else None})
        operation_id = f"grade:{evaluation.evaluation_id}" if error is None else f"grade-failed:{checkpoint_id}"
        run.ledger.record(run_id=run.config.run_id, operation_id=operation_id,
            role="grading", kind="grading", status=status,
            usage=Usage(elapsed_seconds=elapsed, measurement_basis="measured_wall_clock"),
            source_event_id=measured.event_id, source_event_sequence=measured.sequence)
        if error is not None:
            def persist_failure():
                receipt_dir = grade_dir / "private"
                return run.store.put_directory(receipt_dir) if receipt_dir.is_dir() else None
            receipts = self._timed_call(run, "grade_persist", persist_failure, role="infrastructure")
            run.recorder.emit("grade", output_ref=receipts, payload={"checkpoint_id": checkpoint_id,
                "valid": False, "status": "failed", "error": str(error),
                "source_binding": source_binding, "returned_evaluation": (to_jsonable(evaluation)
                    if evaluation is None or isinstance(evaluation, Evaluation) else {"type": type(evaluation).__name__, "repr": repr(evaluation)}),
                "started_at": started, "elapsed_seconds": elapsed})
            # Retain the failed grading workspace, as before. No cleanup occurred.
            if isinstance(error, Exception):
                raise TechnicalFailure(f"grader raised: {error}") from error
            raise error

        def persist_success():
            raw_ref = None
            results = grade_dir / "private" / "results.json"
            if results.exists():
                raw_ref = run.recorder.put_bytes(results.read_bytes(), media_type="application/json")
            receipts = run.store.put_directory(grade_dir / "private") if (grade_dir / "private").is_dir() else None
            recorded = Evaluation(**{**evaluation.__dict__, "workspace_ref": run.recorder.put_bytes(w_data, media_type="application/x-tar"),
                "runtime_ref": _ref(source_binding["runtime_ref"]) if source_binding["runtime_ref"] else None,
                "raw_output_ref": raw_ref, "details": {**evaluation.details,
                    "source_binding_ref": to_jsonable(run.recorder.put_json(source_binding)),
                    "private_receipt_tree": to_jsonable(receipts) if receipts else None}})
            ref = run.recorder.persist(lambda: run.store.put_record("evaluation", recorded.evaluation_id,
                recorded, run_id=run.config.run_id), evidence=recorded)
            return recorded, ref

        evaluation, ref = self._timed_call(run, "grade_persist", persist_success, role="infrastructure")
        # Grade availability follows persistence, so its quality/cost point pays
        # for preparation, evaluation and storage. Cleanup belongs to later time.
        run.last_grade_event = run.recorder.emit("grade", payload={"checkpoint_id": checkpoint_id, "evaluation_id": evaluation.evaluation_id,
            "evaluation_ref": to_jsonable(ref), "valid": evaluation.valid, "score": evaluation.score,
            "source_binding_ref": evaluation.details["source_binding_ref"],
            "full_solve": evaluation.full_solve, "checks": len(evaluation.checks),
            "started_at": started, "elapsed_seconds": elapsed, "stage_id": s.stage.stage_id,
            "target_id": s.stage.target_id, "runtime_export_consumed": "runtime" in grade_kwargs})
        self._timed_call(run, "grade_cleanup", lambda: shutil.rmtree(ws, ignore_errors=True), role="infrastructure")
        return evaluation, ref

    # ================================================================ detector
    def _observe_laps(self, run: _Run, *, relevant_mutation: bool) -> dict | None:
        from ..detector.laps import observation
        if self.detector_config.rule_version == ATTEMPT_RULE_VERSION:
            from ..detector.laps_v2 import observation
        session = run.session
        assert session is not None
        rows = getattr(session.agent, "batch_lap_observations", ())
        states = {e.parent_event_id: e for e in getattr(session.agent, "batch_state_events", ())}
        for row in rows:
            source = dict(row)
            source["event_ids"] = source.pop("source_event_ids")
            extra = {}
            action_mutation = relevant_mutation
            if self.detector_config.rule_version == ATTEMPT_RULE_VERSION:
                state = states.get(source["event_ids"]["tool_result"])
                if state is None or state.payload.get("status") != "captured":
                    raise TechnicalFailure("Attempt recurrence lacks paired per-tool state evidence")
                record = run.task.describe()
                decision = evaluate_mutation(task_id=record.task_id, task_version=record.task_version,
                    workspace_paths=state.payload["workspace_changed"],
                    runtime_paths=sorted(set(state.payload["runtime_changed"]) | (
                        set(state.payload["runtime_deleted_before"]) ^ set(state.payload["runtime_deleted_after"]))),
                    mounts_changed=state.payload["mounted_state_changed"],
                    within_batch_mutation=False, intermediate_evidence_complete=True, policy=run.mutation_policy)
                action_mutation = decision["task_relevant"]
                extra["mutation_evidence"] = {"tool_state_event_id": state.event_id,
                    "tool_state_event_sequence": state.sequence, "policy_result": decision}
            payload = observation(**source, relevant_mutation=action_mutation, **extra)
            self._detector_input(run, {"type": "lap", "scope_id": session.scope.detector_scope_id, **payload})
        if not rows:
            return None
        outputs = self._detector_input(run, {"type": "lap_boundary",
            "scope_id": session.scope.detector_scope_id, "worker_event_sequence": rows[-1]["event_sequence"]})
        return next((item for item in outputs if item.get("record") == "grade_request"), None)

    def _detector_input(self, run: _Run, item: dict[str, Any]) -> list[dict[str, Any]]:
        if run.detector_disabled:
            return []  # v2 branches: no detector inputs, outputs or grade requests at all
        run.detector_seq += 1
        item = dict(item, seq=run.detector_seq)
        ev = run.recorder.emit("detector", payload={"role": "input", "input": item})
        try:
            outputs = run.detector.feed(item)
        except ValueError as e:
            raise TechnicalFailure(f"detector rejected input: {e}") from e
        for out in outputs:
            run.recorder.emit("detector", payload={"role": "output", "input_seq": item["seq"], "input_event_sequence": ev.sequence,
                                                   "output": out})
            if out.get("record") == "annotation":
                self._persist_detector_annotation(run, out, ev.sequence)
        return outputs

    def _detector_grade(self, run: _Run, checkpoint: Checkpoint, evaluation: Evaluation, w_digest: str, *, scope_baseline: bool) -> None:
        # Signature normalization and persisted detector evidence may traverse
        # large native result vectors. Attribute that work separately from the
        # completed private grade; do not hide it in worker/model usage.
        return self._timed_call(run, "detector_grade",
            lambda: self._detector_grade_observed(run, checkpoint, evaluation, w_digest,
                                                  scope_baseline=scope_baseline),
            role="infrastructure")

    def _detector_grade_observed(self, run: _Run, checkpoint: Checkpoint, evaluation: Evaluation, w_digest: str, *, scope_baseline: bool) -> None:
        s = run.session
        assert s is not None
        non_pass = sorted((c.check_id, c.status.value) for c in evaluation.checks if c.status.value != "PASS")
        signature = hashlib.sha256(json.dumps(non_pass).encode()).hexdigest() if evaluation.valid else None
        digest = hashlib.sha256(json.dumps(sorted((c.check_id, c.status.value) for c in evaluation.checks)).encode()).hexdigest() if evaluation.valid else None
        detail = None
        if self.detector_config.rule_version not in SIMPLE_INPUT_RULE_VERSIONS:
            record = run.task.describe()
            policy = run.signature_policies.get(s.stage.stage_id)
            signature_builder = build_failure_signature
            if self.detector_config.rule_version == ATTEMPT_RULE_VERSION:
                from .lap_signature_policy_v2 import build_attempt_failure_signature
                signature_builder = build_attempt_failure_signature
            elif self.detector_config.rule_version == LAP_RULE_VERSION:
                from .lap_signature_policy_v1 import extend_lap_signature_policy
                policy = extend_lap_signature_policy(policy)
            detail = signature_builder(evaluation, task_version=record.task_version,
                **adapter_binding(run.task, record), policy=policy)
            signature, digest = detail["signature"], detail["status_digest"]
        incomplete = any(c.reachable is None for c in evaluation.capabilities)
        known_reach = (list(evaluation.reachable_milestones) if evaluation.reachable_milestones else (
            [c.capability_id for c in evaluation.capabilities if c.reachable is True] if evaluation.capabilities else None))
        # A partial vector can prove a particular new milestone while remaining
        # insufficient to prove absence of advancement elsewhere. Historical
        # versions retain their exact all-or-unknown input representation.
        reach = known_reach if self.detector_config.rule_version in OBSERVABILITY_RULE_VERSIONS else (None if incomplete else known_reach)
        frontier_complete = not incomplete and known_reach is not None
        self._detector_input(run, {"type": "grade", "checkpoint_id": checkpoint.checkpoint_id, "scope_id": s.scope.detector_scope_id,
                                   "valid": evaluation.valid, "score": evaluation.score, "workspace_hash": w_digest,
                                   "failure_signature": signature, "check_status_digest": digest, "reachable": reach,
                                   "frontier_observability_complete": frontier_complete,
                                   **({"frontier_positive_evidence_valid": evaluation.details.get(
                                       "frontier_positive_evidence_valid") is True}
                                      if self.detector_config.rule_version in OBSERVABILITY_RULE_VERSIONS else {}),
                                   **({"full_solve": evaluation.full_solve} if self.detector_config.rule_version in (*LAP_RULE_VERSIONS, *STALL_CLOCK_RULE_VERSIONS) else {}),
                                   **({"passing_checks": sum(1 for c in evaluation.checks if c.status.value == "PASS"),
                                       "executed_checks": sum(1 for c in evaluation.checks if c.status.value in ("PASS", "FAIL"))}
                                      if self.detector_config.rule_version in STALL_CLOCK_RULE_VERSIONS else {}),
                                   "scope_baseline": scope_baseline, "evaluation_id": evaluation.evaluation_id,
                                   "signature_rule": detail["rule_version"] if detail else FAILURE_SIGNATURE_RULE,
                                   **({"signature_evidence":detail,"failed_check_ids":detail["failed_check_ids"]} if detail else {})})

    def _persist_detector_annotation(self, run: _Run, out: dict[str, Any], input_event_sequence: int) -> None:
        target = out.get("target_checkpoint_id")
        if target is None:
            if out.get("label") != "L" or out.get("predecessor_status") != "no_valid_predecessor" or run.session is None:
                raise TechnicalFailure("Detector annotation has no valid target or explicit L localization outcome")
            evidence = {**out, "localization_reason": out.get("localization_reason") or
                        "no_valid_checkpoint_strictly_before_onset"}
            ann = Annotation(run_id=run.config.run_id, checkpoint_id=None, kind="L", scope=run.session.scope,
                rule_version=out["rule_version"], evidence_cutoff=input_event_sequence,
                annotation_id=out["annotation_id"], linked_stall_id=out.get("linked_s_annotation_id"), evidence=to_jsonable(evidence))
            run.recorder.persist(lambda: run.store.save_annotation(ann), evidence=ann)
            run.annotations.append(ann)
            run.recorder.emit("annotation", payload={"annotation_id": ann.annotation_id, "kind": "L",
                "checkpoint_id": None, "unbound": True, "evidence_cutoff": input_event_sequence})
            return
        checkpoint = next(c for c in run.checkpoints if c.checkpoint_id == target)
        label = out["label"]
        ann = self._annotate(run, checkpoint, label, out, evidence_cutoff=input_event_sequence,
                             annotation_id=out["annotation_id"], linked=out.get("linked_s_annotation_id"))
        if label == "S" and run.first_stall_id is None:
            run.first_stall_id = ann.annotation_id

    def _annotate(self, run: _Run, checkpoint: Checkpoint, kind: str, evidence: dict[str, Any], *, evidence_cutoff: int,
                  annotation_id: str | None = None, linked: str | None = None) -> Annotation:
        ann = Annotation(run_id=run.config.run_id, checkpoint_id=checkpoint.checkpoint_id, kind=kind, scope=checkpoint.scope,
                         rule_version=str(evidence.get("rule_version", "structural/v1")), evidence_cutoff=evidence_cutoff,
                         annotation_id=annotation_id or new_id("annotation"), origin="online", linked_stall_id=linked,
                         evidence=to_jsonable(evidence))
        run.recorder.persist(lambda: run.store.save_annotation(ann), evidence=ann)
        run.annotations.append(ann)
        run.recorder.emit("annotation", payload={"annotation_id": ann.annotation_id, "kind": kind, "checkpoint_id": checkpoint.checkpoint_id,
                                                 "evidence_cutoff": evidence_cutoff})
        return ann

    # ============================================================== transitions
    @staticmethod
    def _lifecycle_operations(declared: Any) -> dict[str, str]:
        """Accept explicit adapter operations; never guess from benchmark name."""
        raw = dict(declared)
        w, r, h, o = raw.get("W"), raw.get("R", raw.get("R_outside_workspace")), raw.get("H"), raw.get("O")
        if w not in {"carry", "carry_with_stage_materialization"}:
            raise TechnicalFailure(f"Unsupported declared workspace lifecycle: {w!r}")
        if r in {"reset", "reset_to_fresh_container", "reset_fresh_container"}:
            r = "reset_fresh_container"
        elif r in {"carry", "carry_same_container", "carry_container"}:
            r = "carry_same_container"
        else:
            raise TechnicalFailure(f"Unsupported declared runtime lifecycle: {r!r}")
        if h in {"reset", "reset_to_current_stage_prompt", "reset_live_context"}:
            h = "reset_to_current_stage_prompt"
            if o not in {"reset_with_history", "reset_with_H", "new_substream", "reset"}:
                raise TechnicalFailure("History reset requires a declared fresh O substream")
            o = "new_substream"
        elif h == "carry" and o == "carry":
            pass
        else:
            raise TechnicalFailure(f"Unsupported declared history/observation lifecycle: {h!r}/{o!r}")
        return {"W": w, "R": r, "H": h, "O": o}

    def _transition(self, run: _Run, stage_end: Checkpoint, next_stage: StageSpec, exit_status: str) -> None:
        try:
            self._transition_impl(run, stage_end, next_stage, exit_status)
        except BaseException as error:
            run.transition_failure = error
            try:
                run.recorder.emit("lifecycle", payload={"transition": "stage_transition_failed",
                    "rule_version": "stage-boundary.v2", "before_checkpoint": stage_end.checkpoint_id,
                    "from_stage": stage_end.scope.stage_id, "to_stage": next_stage.stage_id,
                    "exit_status": exit_status, "error": str(error), "error_type": type(error).__name__,
                    "continuation_valid": False})
            except BaseException as cleanup:
                error.add_note(f"Transition failure receipt also failed: {type(cleanup).__name__}")
            raise

    def _transition_impl(self, run: _Run, stage_end: Checkpoint, next_stage: StageSpec, exit_status: str) -> None:
        """Apply the adapter's native state lifecycle and versioned stage material."""
        s = run.session
        assert s is not None
        old_env = s.env
        operations = self._lifecycle_operations(run.task.describe().lifecycle.stage_operations)
        plan = {"transition": "stage_transition_planned", "from_stage": s.stage.stage_id, "to_stage": next_stage.stage_id,
                "before_checkpoint": stage_end.checkpoint_id, "exit_status": exit_status,
                "operations": operations, "declared_operations": dict(run.task.describe().lifecycle.stage_operations)}
        run.recorder.emit("lifecycle", payload=plan)
        old_binding = old_env.binding()
        env = None
        try:
            def export_stage():
                w_data = run.store.get_bytes(stage_end.state.workspace)
                # The task adapter edits only legitimate stage disclosure files on
                # a copy of carried W. This never mounts the live worker in a grader.
                disclosure_workspace = run.scratch / "stage-material" / str(run.stage_index + 1)
                extract_directory(w_data, disclosure_workspace)
                return disclosure_workspace

            disclosure_workspace = self._timed_call(run, "stage_workspace_export", export_stage, role="infrastructure")
            returned_stage = self._timed_call(run, "stage_materialize",
                lambda: run.task.materialize(disclosure_workspace, stage_id=next_stage.stage_id), role="infrastructure")
            if returned_stage != next_stage:
                raise TechnicalFailure("Stage materialization returned a different target")
            def pack_stage():
                prior_ownership = run.store.get_json(_ref(stage_end.validation["workspace_manifest_ref"]))
                w_data = pack_directory(disclosure_workspace, ownership=prior_ownership,
                                       default_owner=(old_env.worker_uid, old_env.worker_gid))
                w_manifest = fidelity_manifest_from_tar(w_data)
                return w_data, w_manifest

            w_data, w_manifest = self._timed_call(run, "stage_workspace_pack", pack_stage, role="infrastructure")
            env = old_env if operations["R"] == "carry_same_container" else self._new_env(run)
            w_result = self._timed_call(run, "stage_workspace_restore", lambda: env.restore_workspace(w_data, w_manifest))
            if operations["R"] == "reset_fresh_container" and env.container_id == old_binding["container_id"]:
                raise TechnicalFailure("container identity did not change")
            if env is not old_env:
                self._timed_call(run, "environment_close", old_env.close, role="infrastructure")
        except BaseException as e:
            if env is not None and env is not old_env:
                self._close_after_failure(run, env, e)
            if not isinstance(e, Exception) or hasattr(e, "persistence_evidence"):
                raise
            raise TechnicalFailure(f"stage transition failed: {e}") from e
        run.stage_index += 1
        scope = self._make_scope(run, next_stage)
        run.recorder.scope = scope
        def initialize_session():
            agent = self._new_agent(run, env, token_limit=run.config.generated_token_limit)
            # Run-wide counters carry; the new agent gets explicit offsets.
            agent.known_generated_tokens = s.agent.known_generated_tokens
            agent.unknown_usage_calls = s.agent.unknown_usage_calls
            agent.generated_token_offset = s.agent.generated_token_offset
            agent.unknown_usage_offset = s.agent.unknown_usage_offset
            agent.n_calls = s.agent.n_calls
            agent.cost = s.agent.cost
            agent.call_offset = s.agent.call_offset
            task_prompt = self._worker_task_prompt(run, next_stage)
            if operations["H"] == "reset_to_current_stage_prompt":
                agent.start(task_prompt)
            else:
                messages = [m for m in s.agent.messages if m.get("role") != "exit"]
                if history_replay_issue(messages):
                    raise TechnicalFailure("Carry-history lifecycle ended at an incomplete tool batch")
                agent.restore(messages, s.agent.counters(), {"message_count": len(messages), "pending_actions": [],
                    "completed_in_batch": []}, task=task_prompt)
                agent.call_offset = s.agent.call_offset
                agent.generated_token_offset = s.agent.generated_token_offset
                agent.unknown_usage_offset = s.agent.unknown_usage_offset
                agent.append_user_message(task_prompt, payload={"lifecycle": "native_stage_disclosure"})
            return _Session(stage=next_stage, scope=scope, env=env, agent=agent, w_manifest=env.workspace_manifest(),
                                   r_manifest=env.runtime_manifest(), w_fidelity=env.workspace_fidelity_manifest())

        try:
            run.session = self._timed_call(run, "stage_session_prepare", initialize_session, role="infrastructure")
        except BaseException as error:
            # Session ownership transfers only after all initialization probes
            # succeed. The outer run still owns the old session on this path.
            if env is not old_env:
                self._close_after_failure(run, env, error)
            raise
        self._detector_input(run, {"type": "scope_start", "scope_id": scope.detector_scope_id, "stage_id": next_stage.stage_id,
                                   "reason": "native_stage_transition"})
        stage_start = self._checkpoint(run, reasons=("stage_start", "graded"), labels=(), grade=True, scope_baseline=True)
        pair = {**plan, "rule_version": "stage-boundary.v2", "transition": "stage_transition_validated",
            "after_checkpoint": stage_start.checkpoint_id, "old_environment": old_binding,
            "new_environment": env.binding(), "workspace_restore": w_result,
            "scope": to_jsonable(scope), "stage": to_jsonable(next_stage),
            "counter_offsets": run.session.agent.counters(), "runtime_before_ref": to_jsonable(stage_end.state.runtime),
            "before_scope": to_jsonable(stage_end.scope), "after_scope": to_jsonable(scope)}
        validated = run.recorder.emit("lifecycle", payload=pair)
        evidence = {"rule_version": "stage-boundary.v2", "transition_event_id": validated.event_id,
                    "before_checkpoint": stage_end.checkpoint_id, "after_checkpoint": stage_start.checkpoint_id,
                    "before_scope": to_jsonable(stage_end.scope), "after_scope": to_jsonable(scope)}
        self._annotate(run, stage_end, "StageEnd", evidence, evidence_cutoff=validated.sequence)
        self._annotate(run, stage_start, "StageStart", evidence, evidence_cutoff=validated.sequence)
        run.recorder.emit("lifecycle", payload={**pair, "transition": "stage_transition_completed",
                                               "validated_event_id": validated.event_id})

    def _terminal(self, run: _Run, status: str, reason: str) -> None:
        if run.terminated:
            return
        run.terminated = True
        captured = None
        skipped = ("capture_failure" if run.recorder.capture_failure is not None else
                   "persistence_failure" if run.recorder.persistence_failure is not None else
                   "run_interrupted" if status == "interrupted" else
                   "tool_execution_uncertain" if run.session is not None and getattr(run.session.agent, "tool_execution_uncertain", False) else
                   "baseline_failed" if run.baseline is not None and run.baseline._failed else
                   "transition_incomplete" if getattr(run, "transition_failure", None) is not None else
                   "checkpoint_incomplete" if run.checkpoint_failure is not None else None)
        if skipped == "persistence_failure":
            run.terminal_failure = str(run.recorder.persistence_failure)
        end = run.v2_end_checkpoint
        if (skipped is None and run.v2 == V2_BRANCH and end is not None and run.checkpoints
                and run.checkpoints[-1] is end and status == "completed"):
            # A branch that ended at a submission ends in that submission's captured and graded state; nothing ran
            # after it, so the end grade is the submission's grade (no second capture of the same state).
            captured = end
            run.recorder.emit("lifecycle", payload={"transition": "end_state_is_submission",
                "checkpoint_id": end.checkpoint_id, "rule": V2_RULES[V2_BRANCH]})
        elif skipped is None and run.session is not None and run.session.env.container_id:
            try:
                self._drain_usage(run)
                captured = self._checkpoint(run, reasons=("terminal",), labels=("Terminal",), grade=True)
            except Exception as e:
                run.terminal_failure = str(e)
                if hasattr(e, "persistence_evidence") or run.recorder.persistence_failure is not None:
                    raise  # Surface the in-memory capture/grade evidence to the caller.
                run.recorder.incident("terminal_capture_failed", error=str(e))
        try:
            self._detector_input(run, {"type": "terminal", "reason": reason})
        except Exception as e:
            run.terminal_failure = str(e)
            run.recorder.incident("detector_terminal_failed", error=str(e))
        run.termination = self._termination_evidence(run, status, reason, captured)
        run.recorder.emit("terminal", payload={"status": "technical_failure" if run.terminal_failure else status,
                                               "reason": run.termination["normalized_reason"], "runner_reason": reason,
                                               "termination": run.termination,
                                               "terminal_recording_failure": run.terminal_failure,
                                               "terminal_capture_skipped": skipped,
                                               "capture_failure": str(run.recorder.capture_failure) if run.recorder.capture_failure is not None else None,
                                               "terminal_checkpoint_id": captured.checkpoint_id if captured else None,
                                               "checkpoints": len(run.checkpoints), "first_stall_id": run.first_stall_id,
                                               "counters": run.session.agent.counters() if run.session else None})

    @staticmethod
    def _termination_evidence(run: _Run, status: str, reason: str, captured: Checkpoint | None) -> dict[str, Any]:
        """Describe the actual stop, with fresh terminal-grade identity or explicit absence."""
        native = getattr(run, "native_exit_status", None)
        if status == "technical_failure" or run.terminal_failure is not None:
            normalized = "technical_failure"
        elif status == "interrupted":
            normalized = "interrupted"
        elif reason == "accounting_interruption":
            normalized = "accounting_interruption"
        elif reason == "budget_exhausted":
            normalized = "budget_exhausted"
        elif reason == "call_limit":
            normalized = "call_limit"
        elif reason in ("cost_limit", "wall_clock_limit"):
            normalized = reason
        elif reason == "worker_error:RepeatedFormatError":
            normalized = "worker_error"
        elif reason in ("fixed_target_complete", "resample_full_solve"):
            normalized = "verified_complete"
        elif native == "Submitted" and reason in {"sequence_complete:submitted", "fixed_target_finished:submitted"}:
            normalized = "voluntary_finish"
        elif native == "IdleTermination":
            normalized = "idle_termination"
        else:
            normalized = reason
        session = run.session
        agent = session.agent if session else None
        known = agent.known_generated_tokens - agent.generated_token_offset if agent else 0
        unknown = agent.unknown_usage_calls - agent.unknown_usage_offset if agent else 0
        exact = not agent.accounting_interrupted() if agent else True
        voluntary = normalized == "voluntary_finish"
        ev = run.store.get_json(captured.evaluation_ref) if captured and captured.evaluation_ref else None
        valid = bool(ev and ev.get("valid") is True)
        # The grade receipt is the availability boundary, after persistence.
        grade_events = [e for e in run.store.iter_events(run.config.run_id) if captured and ev and e.kind == "grade"
            and e.payload.get("checkpoint_id") == captured.checkpoint_id
            and e.payload.get("evaluation_id") == ev.get("evaluation_id")]
        receipt = grade_events[0] if len(grade_events) == 1 else None
        verified = ev.get("full_solve") if valid and type(ev.get("full_solve")) is bool else None
        return {"policy_version": "termination.v1", "normalized_reason": normalized,
            "native_exit_status": native, "runner_reason": reason,
            "voluntary_finish": voluntary, "early_finish": (known < run.config.generated_token_limit if exact else None) if voluntary else False,
            "verified_complete": verified,
            "outcome": ("verified_complete" if verified is True else "voluntary_finish_unsolved" if voluntary and verified is False
                        else "voluntary_finish_unverified" if voluntary else normalized),
            "terminal_checkpoint_id": captured.checkpoint_id if captured else None,
            "terminal_evaluation_ref": to_jsonable(captured.evaluation_ref) if captured else None,
            "terminal_evaluation_id": ev.get("evaluation_id") if ev else None,
            "terminal_grade_event_id": receipt.event_id if receipt else None,
            "terminal_grade_event_sequence": receipt.sequence if receipt else None,
            "terminal_scope": to_jsonable(captured.scope) if captured else None,
            "terminal_grade_valid": valid, "terminal_score": ev.get("score") if valid else None,
            "assigned_generated_tokens": run.config.generated_token_limit,
            "known_generated_tokens_this_run": known, "unknown_usage_calls_this_run": unknown, "exact_accounting": exact}


# ------------------------------------------------------------------ helpers
def _mask_observations(messages: list[dict[str, Any]], placeholder: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Study A O-masking: replace delivered tool-result contents, keep structure and pairing."""
    masked, count = [], 0
    for m in messages:
        if m.get("role") == "tool":
            mm = dict(m)
            mm["content"] = placeholder
            mm["extra"] = {**m.get("extra", {}), "masked": True}
            masked.append(mm)
            count += 1
        else:
            masked.append(dict(m))
    return masked, {"policy": "o_masked", "placeholder": placeholder, "masked_tool_results": count,
                    "preserved": ["assistant turns", "tool calls", "tool_call_id pairing", "system/user messages"]}


def _ref(value: Any) -> ArtifactRef:
    return ArtifactRef(**value) if isinstance(value, dict) else value


def _tar_dir(path: Path) -> bytes:
    """Host directory -> archive with root ownership (the worker runs as root in its container)."""
    return pack_directory(path)


def _manifest_dir(path: Path) -> dict[str, str]:
    from ..execution.environment import loop_excluded, sha256_bytes
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = p.relative_to(path).as_posix()
            if not loop_excluded(rel):
                out[rel] = sha256_bytes(p.read_bytes())
    return dict(sorted(out.items()))
