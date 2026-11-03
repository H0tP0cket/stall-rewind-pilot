"""Continuation records: what actually ran on a branch.

A ContinuationRecord is built from the branch run's recorded grades (not from
observer opinion). It keeps every graded point with cumulative generated
tokens, the stop reason, the technical-failure flag and the observed horizon,
so later curve estimators can censor correctly instead of inventing zeros.

`from_run_records` reads a branch run through the shared store: graded
checkpoints plus the shared CostLedger's worker model records (one per
provider attempt, including rejected/failed attempts, each bound to its
model_attempt event sequence). The budget clock is the cumulative
generated_tokens of those records. An attempt with unknown usage makes every
later position unknown: such points are kept but flagged, never placed at a
guessed budget, and the record's horizon is marked incomplete.
`from_points` builds one directly (tests, or an execution session that emits
the points itself).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..accounting import CostLedger
from ..contracts import ArtifactStoreProtocol, TreatmentManifest, to_jsonable

STOP_REASONS = ("fixed_target_complete", "legitimate_stop", "idle", "token_boundary", "technical_failure",
                "max_calls", "unknown", "verified_complete", "voluntary_finish", "voluntary_finish_unsolved",
                "idle_termination", "budget_exhausted", "accounting_interruption", "call_limit", "interrupted", "worker_error")


def grade_value_problems(grade: Mapping[str, Any]) -> list[str]:
    """Validate read-only historical bodies without trusting constructor coercions."""
    problems = []
    if grade.get("valid") is not True:
        problems.append("grade_invalid_or_missing")
    score = grade.get("score")
    if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
        problems.append("grade_score_malformed")
    if grade.get("full_solve") is not None and type(grade["full_solve"]) is not bool:
        problems.append("grade_full_solve_malformed")
    return problems


@dataclass(frozen=True)
class GradedPoint:
    checkpoint_id: str
    generated_tokens: int  # cumulative known generated tokens at this grade (lower bound if not complete)
    score: float | None
    valid: bool
    full_solve: bool | None
    check_vector: Mapping[str, str]
    cost_usd: float | None = None
    elapsed_seconds: float | None = None
    tokens_complete: bool = True  # False: >=1 earlier worker attempt had unknown usage
    event_sequence: int | None = None
    available_at_sequence: int | None = None
    evaluation_id: str | None = None
    grade_event_id: str | None = None
    exclusions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContinuationRecord:
    treatment_id: str
    branch_run_id: str
    source_checkpoint_id: str
    arm_id: str
    repeat_id: str
    allowance: int
    points: tuple[GradedPoint, ...]
    stop_reason: str
    technical_failure: bool
    observed_horizon: int  # known generated tokens actually observed (censoring boundary)
    baseline_score: float | None  # fixed-target score of the restored state at branch start
    worker_model: str
    horizon_complete: bool = True  # False: budget accounting has unknown gaps; horizon is a lower bound
    # Known tokens at the first attempt with unknown usage. Horizons beyond it
    # and at the boundary are unobservable: unknown usage may be zero.
    unknown_from_tokens: int | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)
    baseline_full_solve: bool | None = None
    termination: Mapping[str, Any] = field(default_factory=dict)
    termination_binding_problems: tuple[str, ...] = ("termination_metadata_unverified",)
    quality_barriers: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(f"unknown stop reason {self.stop_reason}")
        toks = [p.generated_tokens for p in self.points]
        cuts = [p.event_sequence for p in self.points]
        if all(c is not None for c in cuts):
            if cuts != sorted(cuts):
                raise ValueError("graded points must follow checkpoint event order")
        elif toks != sorted(toks):
            raise ValueError("graded points must be in non-decreasing token order")
        if self.points and self.observed_horizon < max(toks):
            raise ValueError("observed horizon precedes last graded point")

    def valid_points(self) -> list[GradedPoint]:
        return [p for p in self.points if p.valid and p.score is not None]

    def placeable_points(self) -> list[GradedPoint]:
        """Valid points whose budget position is fully known."""
        return [p for p in self.valid_points() if p.tokens_complete]

    def placeable_outcomes(self) -> list[GradedPoint]:
        """Solve missingness is independent of scalar-score missingness."""
        return [p for p in self.points if p.valid and p.tokens_complete]

    def _check_horizon(self, horizon: int, curve: str) -> None:
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if curve not in {"allowance", "actual_compute"}:
            raise ValueError("curve must be allowance or actual_compute")

    def absorption_point(self) -> GradedPoint | None:
        """A qualified final result, never an earlier solve or best score."""
        t = self.termination
        if (self.termination_binding_problems or self.technical_failure or not self.horizon_complete
                or self.stop_reason not in {"verified_complete", "fixed_target_complete", "voluntary_finish", "voluntary_finish_unsolved"}
                or t.get("policy_version") != "termination.v1"
                or t.get("normalized_reason") not in {"verified_complete", "voluntary_finish", "voluntary_finish_unsolved"}
                or not (t.get("voluntary_finish") is True or t.get("verified_complete") is True)
                or t.get("exact_accounting") is not True
                or t.get("assigned_generated_tokens") != self.allowance
                or t.get("known_generated_tokens_this_run") != self.observed_horizon):
            return None
        point = self.points[-1] if self.points else None
        if (point is None or point.checkpoint_id != t.get("terminal_checkpoint_id")
                or not point.valid or not point.tokens_complete or point.score is None
                or point.score != t.get("terminal_score") or t.get("terminal_grade_valid") is not True
                or any(b["event_sequence"] > (point.available_at_sequence or point.event_sequence or -1)
                       for b in self.quality_barriers)):
            return None
        return point

    def observed_at(self, horizon: int, *, curve: str = "allowance") -> bool:
        """Resource coverage; score missingness is checked independently."""
        self._check_horizon(horizon, curve)
        if curve == "allowance" and horizon > self.allowance:
            return False
        if horizon > self.observed_horizon:
            return curve == "allowance" and self.absorption_point() is not None
        if not self.horizon_complete:
            # Past the first accounting gap we cannot say which grades precede the horizon.
            return self.unknown_from_tokens is not None and horizon < self.unknown_from_tokens
        return True

    def completion_tokens(self) -> int | None:
        if self.baseline_full_solve is True:
            return 0
        return next((p.generated_tokens for p in self.placeable_outcomes() if p.full_solve is True), None)

    def horizon_status(self, horizon: int, *, curve: str = "allowance") -> str:
        self._check_horizon(horizon, curve)
        if curve == "allowance" and horizon > self.allowance:
            return "outside_assigned_allowance"
        if self.observed_at(horizon, curve=curve):
            if self.score_at(horizon, curve=curve) is None:
                return "grade_measurement_missing"
            if horizon > self.observed_horizon:
                return "absorbing_verified_completion" if self.termination.get("verified_complete") is True else "absorbing_voluntary_finish"
            return "observed_current_grade"
        if not self.horizon_complete and (self.unknown_from_tokens is None or horizon >= self.unknown_from_tokens):
            return "accounting_unknown"
        return "technical_failure" if self.technical_failure else "stopped_before_horizon"

    def last_graded_score_at(self, horizon: int, *, curve: str = "allowance") -> float | None:
        """Current/last grade at the cut, respecting intervening missing grades."""
        if not self.observed_at(horizon, curve=curve):
            return None
        return self._current_measurement(horizon, "score", self.baseline_score)

    def _current_measurement(self, horizon: int, field: str, baseline: Any) -> Any:
        measurements = [(p.available_at_sequence if p.available_at_sequence is not None else
                         p.event_sequence if p.event_sequence is not None else i,
                         getattr(p, field) if p.valid else None)
                        for i, p in enumerate(self.points) if p.tokens_complete and p.generated_tokens <= horizon]
        measurements.extend((b["event_sequence"], None) for b in self.quality_barriers
                            if b["tokens_complete"] and b["generated_tokens"] <= horizon)
        return sorted(measurements, key=lambda x: x[0])[-1][1] if measurements else baseline

    def score_at(self, horizon: int, *, curve: str = "allowance") -> float | None:
        """Primary current fixed-target score; no best-score substitution."""
        return self.last_graded_score_at(horizon, curve=curve)

    def best_score_at(self, horizon: int, *, curve: str = "allowance") -> float | None:
        """Separate best *observed* grade; missing grades may hide a higher best."""
        if not self.observed_at(horizon, curve=curve):
            return None
        best: float | None = self.baseline_score
        for p in self.placeable_points():
            if p.generated_tokens <= horizon:
                best = p.score if best is None else max(best, p.score)
        return best

    def current_solve_at(self, horizon: int, *, curve: str = "allowance") -> bool | None:
        if not self.observed_at(horizon, curve=curve):
            return None
        return self._current_measurement(horizon, "full_solve", self.baseline_full_solve)

    def solved_by(self, horizon: int, *, curve: str = "allowance") -> bool | None:
        """Whether a solve was ever observed; distinct from current solve state."""
        self._check_horizon(horizon, curve)
        if (curve == "allowance" and horizon > self.allowance
                or curve == "actual_compute" and horizon > self.observed_horizon):
            return None
        completion = self.completion_tokens()
        if completion is not None and completion <= horizon:
            return True
        if not self.observed_at(horizon, curve=curve):
            return None
        if self.baseline_full_solve is True:
            return True
        measured = [p.full_solve for p in self.placeable_outcomes() if p.generated_tokens <= horizon]
        if any(v is True for v in measured):
            return True
        if any(v is False for v in measured) or self.baseline_full_solve is False:
            return False
        return None

    def regressions(self) -> int:
        prev = self.baseline_score
        count = 0
        for p in self.valid_points():
            if prev is not None and p.score is not None and p.score < prev:
                count += 1
            prev = p.score if p.score is not None else prev
        return count

    def as_record(self) -> dict[str, Any]:
        return {
            "version": "continuation.v2",
            "treatment_id": self.treatment_id, "branch_run_id": self.branch_run_id,
            "source_checkpoint_id": self.source_checkpoint_id, "arm_id": self.arm_id, "repeat_id": self.repeat_id,
            "allowance": self.allowance,
            "points": [dict(p.__dict__, check_vector=dict(p.check_vector)) for p in self.points],
            "stop_reason": self.stop_reason, "technical_failure": self.technical_failure,
            "observed_horizon": self.observed_horizon, "horizon_complete": self.horizon_complete,
            "unknown_from_tokens": self.unknown_from_tokens,
            "baseline_score": self.baseline_score, "worker_model": self.worker_model, "lineage": dict(self.lineage),
            "baseline_full_solve": self.baseline_full_solve,
            "termination": dict(self.termination),
            "termination_binding_problems": list(self.termination_binding_problems),
            "quality_barriers": [dict(b) for b in self.quality_barriers],
            "allowance_absorption_qualified": self.absorption_point() is not None,
        }


def from_points(manifest: TreatmentManifest, branch_run_id: str, points: Iterable[GradedPoint], *,
                stop_reason: str, observed_horizon: int, baseline_score: float | None,
                technical_failure: bool = False, horizon_complete: bool = True,
                unknown_from_tokens: int | None = None,
                baseline_full_solve: bool | None = None,
                termination: Mapping[str, Any] | None = None,
                termination_binding_problems: Iterable[str] | None = None,
                quality_barriers: Iterable[Mapping[str, Any]] = (),
                lineage: Mapping[str, Any] | None = None) -> ContinuationRecord:
    return ContinuationRecord(
        treatment_id=manifest.treatment_id, branch_run_id=branch_run_id,
        source_checkpoint_id=manifest.source_checkpoint_id, arm_id=str(manifest.metadata.get("arm_id", "?")),
        repeat_id=manifest.repeat_id, allowance=manifest.generated_token_limit, points=tuple(points),
        stop_reason=stop_reason, technical_failure=technical_failure, observed_horizon=observed_horizon,
        baseline_score=baseline_score, worker_model=manifest.worker_model, horizon_complete=horizon_complete,
        unknown_from_tokens=unknown_from_tokens, baseline_full_solve=baseline_full_solve,
        termination=dict(termination or {}),
        termination_binding_problems=tuple(termination_binding_problems) if termination_binding_problems is not None
            else ("termination_metadata_unverified" if termination else "termination_metadata_missing",),
        quality_barriers=tuple(dict(b) for b in quality_barriers),
        lineage={"target_id": manifest.target_id, "stage_id": manifest.stage_id,
                 "target_manifest_id": manifest.metadata.get("recovery_target_manifest_id"),
                 "model_settings": dict(manifest.metadata.get("model_settings", {})),
                 "declared_worker_model": manifest.worker_model,
                 "declared_recovery_settings": manifest.metadata.get("recovery_settings"),
                 "runtime_policy": manifest.runtime_policy, "selection_kind": manifest.selection_kind,
                 **dict(lineage or {})},
    )


@dataclass(frozen=True)
class BudgetClock:
    """Cumulative worker generated tokens by event sequence, from the shared CostLedger.

    Uses role="worker", kind="model" records: one per provider attempt, so
    rejected and failed attempts count and nothing is counted twice. Records
    without a source_event_sequence cannot be placed on the clock and are
    reported as unplaced (their tokens are excluded from every position).
    """

    seqs: tuple[int, ...]
    cums: tuple[int, ...]
    unknown_from_seq: int | None  # first seq at which an attempt had unknown generated_tokens
    total_known: int
    attempts: int
    unknown_attempts: int
    unplaced_records: int
    identity_unverified_records: int = 0

    @staticmethod
    def from_ledger(store: ArtifactStoreProtocol, run_id: str) -> "BudgetClock":
        records = [r for r in CostLedger(store).records(run_id) if r.role == "worker" and r.kind == "model"]  # type: ignore[arg-type]
        events = {e.sequence: e for e in store.iter_events(run_id)}
        def placed_record(r: Any) -> bool:
            event = events.get(r.source_event_sequence)
            return event is not None and (r.source_event_id is None or r.source_event_id == event.event_id)
        placed = sorted((r for r in records if placed_record(r)), key=lambda r: r.source_event_sequence)
        unplaced = len(records) - len(placed)
        seqs, cums = [], []
        cum = 0
        unknown_from = None
        unknown = 0
        for r in placed:
            g = r.usage.generated_tokens
            if g is None:
                unknown += 1
                if unknown_from is None:
                    unknown_from = r.source_event_sequence
            else:
                cum += g
            seqs.append(r.source_event_sequence)
            cums.append(cum)
        return BudgetClock(tuple(seqs), tuple(cums), unknown_from,
                           sum(r.usage.generated_tokens or 0 for r in records), len(records),
                           sum(r.usage.generated_tokens is None for r in records), unplaced,
                           sum(not placed_record(r) or r.source_event_id is None for r in records))

    def at(self, seq: int) -> tuple[int, bool]:
        """(known cumulative tokens at or before seq, complete?)."""
        i = bisect.bisect_right(self.seqs, seq)
        known = self.cums[i - 1] if i else 0
        complete = self.unplaced_records == 0 and (self.unknown_from_seq is None or self.unknown_from_seq > seq)
        return known, complete

    @property
    def complete(self) -> bool:
        return self.unknown_attempts == 0 and self.unplaced_records == 0


def _termination_binding(store: ArtifactStoreProtocol, manifest: TreatmentManifest, run_id: str,
                         events: list[Any], cps: list[Any], points: list[GradedPoint],
                         clock: BudgetClock) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Verify final outcome evidence independently of native exit labels."""
    terminals = [e for e in events if e.kind == "terminal"]
    if len(terminals) != 1:
        return {}, ("terminal_event_missing_or_ambiguous",)
    terminal = terminals[0]
    raw = terminal.payload.get("termination")
    if not isinstance(raw, Mapping):
        return {}, ("termination_metadata_missing",)
    t, problems = dict(raw), []
    if t.get("policy_version") != "termination.v1":
        problems.append("termination_policy_unsupported")
    try:
        result = store.get_record("run_result", run_id)
    except KeyError:
        result = {}
        problems.append("run_result_missing")
    if result.get("details", {}).get("termination") != t:
        problems.append("terminal_run_result_metadata_mismatch")
    if result.get("status") != "completed" or terminal.payload.get("status") != "completed":
        problems.append("terminal_run_not_completed")
    if (t.get("normalized_reason") not in {"verified_complete", "voluntary_finish", "voluntary_finish_unsolved"}
            or not (t.get("voluntary_finish") is True or t.get("verified_complete") is True)):
        problems.append("terminal_stop_not_absorbing")
    if result.get("termination_reason") != t.get("normalized_reason"):
        problems.append("terminal_run_result_reason_mismatch")
    if (t.get("normalized_reason") in {"voluntary_finish", "voluntary_finish_unsolved"} and t.get("voluntary_finish") is not True
            or t.get("normalized_reason") == "verified_complete" and
               (t.get("verified_complete") is not True or t.get("voluntary_finish") is not False
                or t.get("runner_reason") != "fixed_target_complete")):
        problems.append("terminal_stop_category_mismatch")
    starts = [e for e in events if e.kind == "lifecycle" and e.payload.get("transition") == "run_started"]
    executed_allowance = starts[0].payload.get("config", {}).get("generated_token_limit") if len(starts) == 1 else None
    if type(executed_allowance) is not int or executed_allowance != manifest.generated_token_limit:
        problems.append("terminal_executed_allowance_unverified")
    if type(t.get("assigned_generated_tokens")) is not int or t["assigned_generated_tokens"] != manifest.generated_token_limit:
        problems.append("terminal_allowance_mismatch")
    if (type(t.get("known_generated_tokens_this_run")) is not int
            or t["known_generated_tokens_this_run"] != clock.total_known
            or type(t.get("unknown_usage_calls_this_run")) is not int
            or t["unknown_usage_calls_this_run"] != clock.unknown_attempts
            or t.get("exact_accounting") is not True or not clock.complete or clock.identity_unverified_records):
        problems.append("terminal_worker_accounting_unverified")
    if t.get("voluntary_finish") is True and t.get("early_finish") is not (clock.total_known < manifest.generated_token_limit):
        problems.append("terminal_early_finish_mismatch")
    cp = next((c for c in cps if c.checkpoint_id == t.get("terminal_checkpoint_id")), None)
    if cp is None or cp != (cps[-1] if cps else None) or "terminal" not in cp.reasons:
        problems.append("terminal_checkpoint_missing_or_not_final")
    if cp is None:
        return t, tuple(problems)
    if (cp.run_id != run_id or to_jsonable(cp.scope) != t.get("terminal_scope")
            or cp.scope.stage_id != manifest.stage_id or cp.scope.target_id != manifest.target_id):
        problems.append("terminal_scope_mismatch")
    try:
        source = store.get_checkpoint(manifest.source_checkpoint_id)
    except KeyError:
        source = None
    if source is None or any(getattr(cp.scope, k) != getattr(source.scope, k)
            for k in ("task_id", "stage_id", "target_id", "requirement_version", "evaluator_version")):
        problems.append("terminal_frozen_target_unverified")
    if t.get("voluntary_finish") is True:
        exits = [e for e in events if e.kind == "lifecycle"
                 and e.payload.get("transition") == "worker_exit_normalized" and e.sequence < terminal.sequence]
        last_exit = exits[-1] if exits else None
        if (t.get("native_exit_status") != "Submitted"
                or t.get("runner_reason") not in {"sequence_complete:submitted", "fixed_target_finished:submitted"}
                or last_exit is None or last_exit.scope != cp.scope
                or last_exit.payload.get("raw_exit") != "Submitted"
                or last_exit.payload.get("normalized") != "submitted"):
            problems.append("terminal_voluntary_exit_unverified")
    if to_jsonable(cp.evaluation_ref) != t.get("terminal_evaluation_ref") or cp.evaluation_ref is None:
        problems.append("terminal_evaluation_reference_mismatch")
    if result.get("final_evaluation_ref") != to_jsonable(cp.evaluation_ref):
        problems.append("terminal_run_result_evaluation_mismatch")
    ev = store.get_json(cp.evaluation_ref) if cp.evaluation_ref else {}
    if (ev.get("evaluation_id") != t.get("terminal_evaluation_id") or ev.get("valid") is not True
            or t.get("terminal_grade_valid") is not True or ev.get("score") != t.get("terminal_score")
            or type(ev.get("score")) is not type(t.get("terminal_score"))
            or ev.get("score") is None or ev.get("full_solve") != t.get("verified_complete")
            or type(ev.get("full_solve")) is not type(t.get("verified_complete"))
            or ev.get("workspace_ref") != to_jsonable(cp.state.workspace)):
        problems.append("terminal_grade_evidence_mismatch")
    point = next((p for p in points if p.checkpoint_id == cp.checkpoint_id), None)
    receipt = next((e for e in events if e.event_id == t.get("terminal_grade_event_id")), None)
    if (point is None or not point.valid or not point.tokens_complete
            or point.grade_event_id != t.get("terminal_grade_event_id")
            or point.available_at_sequence != t.get("terminal_grade_event_sequence")
            or point.available_at_sequence is None
            or point.generated_tokens != clock.total_known
            or not cp.event_sequence <= point.available_at_sequence <= terminal.sequence
            or any(seq > point.available_at_sequence for seq in clock.seqs)
            or receipt is None or receipt.scope != cp.scope
            or receipt.payload.get("evaluation_ref") != to_jsonable(cp.evaluation_ref)):
        problems.append("terminal_grade_availability_unverified")
    return t, tuple(problems)


def quality_barriers(events: Iterable[Any], clock: BudgetClock) -> tuple[dict[str, Any], ...]:
    """Recorded failed measurements/executions, with event identity and no invented checkpoint."""
    barriers = []
    for event in events:
        payload, reason = event.payload, None
        if event.kind == "capture" and payload.get("status") == "failed":
            reason = "capture_failed"
        elif event.kind == "tool_result" and payload.get("status") in {"transport_error", "transport_timeout", "interrupted", "error"}:
            reason = "tool_execution_unmeasured"
        elif event.kind == "incident" and payload.get("what") == "terminal_capture_failed":
            reason = "terminal_capture_failed"
        elif event.kind == "terminal" and (payload.get("terminal_capture_skipped") or payload.get("terminal_recording_failure")
                or isinstance(payload.get("termination"), Mapping) and payload["termination"].get("policy_version") == "termination.v1"
                   and payload["termination"].get("terminal_checkpoint_id") is None):
            reason = "terminal_measurement_missing"
        if reason:
            tokens, complete = clock.at(event.sequence)
            barriers.append({"source_event_id": event.event_id, "event_sequence": event.sequence,
                "event_kind": event.kind, "reason": reason, "generated_tokens": tokens, "tokens_complete": complete})
    return tuple(barriers)


def from_run_records(store: ArtifactStoreProtocol, manifest: TreatmentManifest, branch_run_id: str, *,
                     stop_reason: str, technical_failure: bool = False) -> ContinuationRecord:
    """Build from the shared store: budget from the CostLedger, grades from checkpoints."""
    from .costs import grade_availability

    clock = BudgetClock.from_ledger(store, branch_run_id)
    events = list(store.iter_events(branch_run_id))
    starts = [e for e in events
              if e.kind == "lifecycle" and e.payload.get("transition") == "run_started"]
    start = starts[0] if len(starts) == 1 else None
    config = start.payload.get("config", {}) if start else {}
    from ..contracts import worker_operational_settings
    executed_settings = worker_operational_settings(config)
    bindings = [e for e in events
                if e.kind == "lifecycle" and e.payload.get("transition") == "worker_policy_bound"]
    attempts = [e for e in events
                if e.kind == "model_attempt" and e.payload.get("phase") == "started"]
    bindings_verified = bool(bindings) and all(
        e.payload.get("operational_settings") == executed_settings
        and e.payload.get("effective_request_policy") == config.get("worker_request_policy")
        and e.payload.get("effective_format_error_limit") == config.get("max_consecutive_format_errors")
        and e.payload.get("effective_model_call_limit") == config.get("max_model_calls")
        and (e.payload.get("scripted_fixture") is True or (
            e.payload.get("effective_model") == config.get("model")
            and e.payload.get("effective_model_settings") == config.get("model_settings")))
        for e in bindings) and all(any(b.sequence < a.sequence for b in bindings) for a in attempts)
    cps = sorted(store.list_checkpoints(branch_run_id), key=lambda c: c.ordinal)  # type: ignore[attr-defined]
    points: list[GradedPoint] = []
    baseline: float | None = None
    baseline_solve: bool | None = None
    ungraded = 0
    incompatible = []
    reused = []
    try:
        source_scope = store.get_checkpoint(manifest.source_checkpoint_id).scope
    except KeyError:
        source_scope = None
    for c in cps:
        exclusions = []
        if c.evaluation_ref is None:
            ungraded += 1
            exclusions.append("checkpoint_ungraded")
        if c.validation.get("evaluation_reused") and c.ordinal != 0:
            reused.append(c.checkpoint_id)
            exclusions.append("evaluation_reused")
        ev = store.get_json(c.evaluation_ref) if c.evaluation_ref else {}
        target = manifest.metadata.get("recovery_target", {})
        compatible = (ev.get("target_id") == manifest.target_id and ev.get("stage_id") == manifest.stage_id
                      and all(ev.get(k) == getattr(c.scope, k) for k in ("task_id", "target_id", "stage_id", "evaluator_version"))
                      and (not source_scope or all(getattr(c.scope, k) == getattr(source_scope, k)
                          for k in ("task_id", "stage_id", "target_id", "requirement_version", "evaluator_version")))
                      and (not target.get("evaluator_version") or ev.get("evaluator_version") == target["evaluator_version"]))
        if not compatible:
            incompatible.append(c.checkpoint_id)
            exclusions.append("grade_scope_incompatible")
        exclusions.extend(grade_value_problems(ev))
        availability = grade_availability(store, c, events)
        available_at = availability["available_at_sequence"]
        if available_at is None:
            exclusions.append(availability["status"])
        checks = {chk["check_id"]: chk["status"] for chk in ev.get("checks", [])}
        tokens, complete = clock.at(available_at if available_at is not None else c.event_sequence)
        point = GradedPoint(c.checkpoint_id, tokens, ev.get("score"), not exclusions, ev.get("full_solve"),
                            checks, tokens_complete=complete, event_sequence=c.event_sequence,
                            available_at_sequence=available_at, evaluation_id=ev.get("evaluation_id"),
                            grade_event_id=availability.get("grade_event_id"), exclusions=tuple(exclusions))
        if c.ordinal == 0 and "initial" in c.reasons and tokens == 0 and complete and point.valid:
            baseline = point.score if point.valid else None
            baseline_solve = point.full_solve if point.valid else None
            continue
        points.append(point)
    unknown_from = None if clock.unknown_from_seq is None else clock.at(clock.unknown_from_seq)[0]
    if clock.unplaced_records:
        unknown_from = 0  # tokens of unplaceable records could precede any position
    termination, termination_problems = _termination_binding(store, manifest, branch_run_id, events, cps, points, clock)
    return from_points(manifest, branch_run_id, points, stop_reason=stop_reason, observed_horizon=clock.total_known,
                       baseline_score=baseline, technical_failure=technical_failure, horizon_complete=clock.complete,
                       baseline_full_solve=baseline_solve,
                       termination=termination, termination_binding_problems=termination_problems,
                       quality_barriers=quality_barriers(events, clock),
                       unknown_from_tokens=unknown_from,
                       lineage={"source": "store+cost_ledger", "checkpoint_count": len(cps), "ungraded_checkpoints": ungraded,
                                "execution_settings_source": "run_started" if start else "run_started_missing_or_ambiguous",
                                "run_started_event_id": start.event_id if start else None,
                                "run_started_sequence": start.sequence if start else None,
                                "executed_recovery_settings": executed_settings,
                                "worker_policy_bindings_verified": bindings_verified,
                                "worker_policy_binding_event_ids": [e.event_id for e in bindings],
                                "effective_detector_config": start.payload.get("detector_config") if start else None,
                                "executed_model_settings": config.get("model_settings"),
                                "executed_worker_model": config.get("model"),
                                "worker_model_attempts": clock.attempts, "attempts_unknown_usage": clock.unknown_attempts,
                                "incompatible_grade_checkpoints": incompatible,
                                "reused_evaluation_checkpoints": reused,
                                "ledger_records_without_event_sequence": clock.unplaced_records,
                                "ledger_worker_source_identities_unverified": clock.identity_unverified_records})
