"""Shared version-1 wire records and integration Protocols.

Records preserve unknown values; they never infer observation, admission or
restoration from a missing field. Scientific numeric settings are supplied by
the campaign, not defaults in this module. Integrators may use adapters around
these Protocols; there must be one canonical persisted representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
import math
from pathlib import Path
import collections.abc
import types
from typing import Any, Iterator, Mapping, Protocol, Sequence, Union, get_args, get_origin, get_type_hints, runtime_checkable
from uuid import uuid4

SCHEMA_VERSION = "trajectory-retention.v1"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    # Exact builtin types cannot override dataclass/container behavior. Keep
    # the original generic path for subclasses, custom mappings and records.
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("Non-finite measurements must be recorded as missing")
        return value
    if kind is dict:
        if any(not isinstance(k, str) for k in value):
            raise TypeError("JSON record keys must be strings")
        return {k: to_jsonable(v) for k, v in value.items()}
    if kind is list or kind is tuple:
        return [to_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("JSON record keys must be strings")
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite measurements must be recorded as missing")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def from_jsonable(record_type: Any, value: Any) -> Any:
    """Strictly reconstruct wire records, rejecting silently dropped fields."""
    if record_type is Any:
        return to_jsonable(value)
    if record_type is Path and isinstance(value, str):
        return Path(value)
    origin, args = get_origin(record_type), get_args(record_type)
    if origin in (Union, types.UnionType):
        for member in args:
            try:
                return from_jsonable(member, value)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"Value does not satisfy {record_type}")
    if record_type is type(None):
        if value is not None:
            raise TypeError("Expected null")
        return None
    if origin in (tuple, list, Sequence, collections.abc.Sequence):
        if not isinstance(value, (tuple, list)):
            raise TypeError("Expected sequence")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(value):
                raise ValueError("Tuple length does not match its schema")
            converted = [from_jsonable(t, item) for t, item in zip(args, value)]
        else:
            converted = [from_jsonable(args[0], item) for item in value]
        return tuple(converted) if origin is tuple else converted
    if origin in (dict, Mapping, collections.abc.Mapping):
        if not isinstance(value, Mapping):
            raise TypeError("Expected object")
        return {from_jsonable(args[0], k): from_jsonable(args[1], v) for k, v in value.items()}
    if isinstance(record_type, type) and issubclass(record_type, StrEnum):
        return record_type(value)
    if isinstance(record_type, type) and is_dataclass(record_type):
        if not isinstance(value, Mapping):
            raise TypeError("Expected record object")
        hints = get_type_hints(record_type)
        unknown = set(value) - {f.name for f in fields(record_type)}
        if unknown:
            raise ValueError(f"Unknown {record_type.__name__} fields: {sorted(unknown)}")
        if "schema_version" in value and value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported record schema version")
        return record_type(**{k: from_jsonable(hints[k], v) for k, v in value.items()})
    if record_type is float and type(value) in (float, int):
        if not math.isfinite(value):
            raise ValueError("Non-finite float")
        return float(value)
    if record_type in (str, int, bool) and type(value) is record_type:
        return value
    raise TypeError(f"Expected {record_type}, received {type(value).__name__}")


class Visibility(StrEnum):
    WORKER_VISIBLE = "worker_visible"
    OBSERVER_ALLOWED = "observer_allowed"
    RESEARCH_PRIVATE = "research_private"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR_TASK = "ERROR_TASK"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"
    INFRA_ERROR = "INFRA_ERROR"


class RuntimeClass(StrEnum):
    IRRELEVANT = "R_IRRELEVANT"
    CAPTURED = "R_CAPTURED"
    UNSUPPORTED = "R_UNSUPPORTED"


class AdmissionStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    HOLD = "HOLD"
    ADMITTED = "ADMITTED"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True, kw_only=True)
class ArtifactRef:
    sha256: str
    size_bytes: int
    media_type: str = "application/json"
    visibility: Visibility = Visibility.RESEARCH_PRIVATE

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("Artifact digest must be lowercase SHA-256")
        if self.size_bytes < 0:
            raise ValueError("Negative artifact size")


@dataclass(frozen=True, kw_only=True)
class Scope:
    task_id: str
    stage_id: str
    target_id: str
    session_id: str
    detector_scope_id: str
    requirement_version: str
    evaluator_version: str


@dataclass(frozen=True, kw_only=True)
class Usage:
    """Provider fields may overlap. generated_tokens is the declared budget unit."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_tokens: int | None = None
    generated_tokens: int | None = None
    reasoning_accounting: str = "unknown"
    cost_usd: float | None = None
    elapsed_seconds: float | None = None
    measurement_basis: str = "unknown"
    raw_usage: ArtifactRef | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens",
                     "total_tokens", "generated_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"Invalid {name}")
        for name in ("cost_usd", "elapsed_seconds"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {name}")


@dataclass(frozen=True, kw_only=True)
class EventRecord:
    run_id: str
    kind: str
    event_id: str = field(default_factory=lambda: new_id("event"))
    sequence: int = 0  # Assigned atomically by append_event, per run.
    timestamp: str = field(default_factory=utc_now)
    actor: str = "harness"
    scope: Scope | None = None
    parent_event_id: str | None = None
    call_id: str | None = None
    attempt: int | None = None
    visibility: Visibility = Visibility.RESEARCH_PRIVATE
    input_ref: ArtifactRef | None = None
    output_ref: ArtifactRef | None = None
    delivered_ref: ArtifactRef | None = None
    usage: Usage | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, kw_only=True)
class LifecyclePolicy:
    policy_id: str
    version: str
    origin: str  # benchmark_native / harness_policy / explicit_adaptation
    stage_operations: Mapping[str, str]
    detector_scope_rule: str
    stage_release_rule: str
    source_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class StageSpec:
    stage_id: str
    target_id: str
    instructions: str
    requirement_version: str
    evaluator_version: str
    ordinal: int
    prerequisites: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TaskRecord:
    task_id: str
    family_id: str
    source: str
    source_revision: str
    task_version: str
    environment_version: str
    stages: tuple[StageSpec, ...]
    lifecycle: LifecyclePolicy
    admission: AdmissionStatus = AdmissionStatus.HOLD
    runtime_class: RuntimeClass = RuntimeClass.UNSUPPORTED
    evidence: Mapping[str, Any] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, kw_only=True)
class CheckResult:
    check_id: str
    status: CheckStatus
    raw_status: str | None = None
    capability_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CapabilityResult:
    capability_id: str
    completion: float | None
    reachable: bool | None
    requirement_group: str
    milestone_ids: tuple[str, ...] = ()
    check_ids: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Evaluation:
    task_id: str
    stage_id: str
    target_id: str
    evaluator_version: str
    score: float | None
    full_solve: bool | None
    valid: bool
    checks: tuple[CheckResult, ...]
    capabilities: tuple[CapabilityResult, ...] = ()
    group_scores: Mapping[str, float | None] = field(default_factory=dict)
    reachable_milestones: tuple[str, ...] = ()
    raw_score: float | None = None
    raw_max_score: float | None = None
    evaluation_id: str = field(default_factory=lambda: new_id("grade"))
    workspace_ref: ArtifactRef | None = None
    runtime_ref: ArtifactRef | None = None
    raw_output_ref: ArtifactRef | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.score is not None and (not math.isfinite(self.score) or not 0 <= self.score <= 1):
            raise ValueError("Normalized score must be in [0, 1] or unknown")
        if not self.valid and (self.score is not None or self.full_solve is not None):
            raise ValueError("Invalid evaluation cannot assert score or solve")


@dataclass(frozen=True, kw_only=True)
class NativeState:
    workspace: ArtifactRef
    history: ArtifactRef
    harness: ArtifactRef
    runtime_class: RuntimeClass
    runtime: ArtifactRef | None = None
    observations: tuple[ArtifactRef, ...] = ()
    execution_cursor: Mapping[str, Any] = field(default_factory=dict)
    capture_evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class RuntimeExport:
    """Execution-owned R export for a disposable private grader.

    archive is a read-only host file outside exported W, never a live worker
    mount. snapshot is the declared restoration manifest (changed/deleted
    paths, process support and content verification). Both W and R must come
    from the same capture boundary unless a treatment explicitly declares
    different donors. The adapter must reject unsupported restoration instead
    of quietly substituting its frozen baseline environment.
    """
    base_image_id: str
    runtime_class: RuntimeClass
    snapshot: Mapping[str, Any]
    archive: Path | None = None
    archive_sha256: str | None = None
    workspace_snapshot_sha256: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.base_image_id:
            raise ValueError("Runtime export requires an exact base image binding")
        if (self.archive is None) != (self.archive_sha256 is None):
            raise ValueError("Runtime archive path and digest must be provided together")


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    run_id: str
    scope: Scope
    ordinal: int
    event_sequence: int
    state: NativeState
    reasons: tuple[str, ...]
    checkpoint_id: str = field(default_factory=lambda: new_id("checkpoint"))
    parent_checkpoint_id: str | None = None
    evaluation_ref: ArtifactRef | None = None
    accounting_ref: ArtifactRef | None = None
    branchable: bool = False
    validation: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.ordinal < 0 or self.event_sequence < 0:
            raise ValueError("Invalid checkpoint boundary")
        if self.branchable and self.state.runtime_class == RuntimeClass.UNSUPPORTED:
            raise ValueError("Unsupported required runtime cannot be branchable")
        if self.branchable and self.state.runtime_class == RuntimeClass.CAPTURED and self.state.runtime is None:
            raise ValueError("Captured runtime must have a restoration artifact")


@dataclass(frozen=True, kw_only=True)
class Annotation:
    run_id: str
    checkpoint_id: str | None = None
    kind: str
    scope: Scope
    rule_version: str
    evidence_cutoff: int
    annotation_id: str = field(default_factory=lambda: new_id("annotation"))
    assigned_at: str = field(default_factory=utc_now)
    origin: str = "online"
    linked_stall_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.checkpoint_id is None:
            reason = self.evidence.get("localization_reason")
            if self.kind != "L" or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Only an explicitly unlocalized L may omit its checkpoint")


@dataclass(frozen=True, kw_only=True)
class WorkerRequestPolicy:
    """Operational safeguards, recorded explicitly; not calibrated scientific budgets."""
    version: str = "worker-request-policy.v1"
    max_attempts: int = 5
    backoff_seconds: float = 4.0
    timeout_seconds: float = 600.0
    usage_recovery_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.version != "worker-request-policy.v1":
            raise ValueError("Unsupported worker request policy")
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("Worker request max_attempts must be a positive integer")
        for key in ("backoff_seconds", "timeout_seconds", "usage_recovery_timeout_seconds"):
            value = getattr(self, key)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (key != "backoff_seconds" and value == 0)):
                raise ValueError(f"Invalid worker request {key}")


WORKER_OPERATIONAL_FIELDS = ("max_model_calls", "tool_timeout_seconds", "detector_settings",
                             "worker_request_policy", "max_consecutive_format_errors")


def worker_operational_settings(config: Any) -> dict[str, Any]:
    """Project recorded fields without supplying defaults to historical records."""
    value = to_jsonable(config)
    return {key: value[key] for key in WORKER_OPERATIONAL_FIELDS if key in value}


def worker_request_policy_known(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(WorkerRequestPolicy)}:
        return False
    try:
        from_jsonable(WorkerRequestPolicy, value)
    except (TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    run_id: str
    campaign_id: str
    purpose: str  # development / calibration / main
    model: str
    model_settings: Mapping[str, Any]
    generated_token_limit: int
    stage_id: str | None = None
    max_model_calls: int | None = None
    tool_timeout_seconds: float = 120.0  # Operational guard; recorded, not scientific calibration.
    worker_request_policy: WorkerRequestPolicy = field(default_factory=WorkerRequestPolicy)
    max_consecutive_format_errors: int = 3
    detector_settings: Mapping[str, Any] = field(default_factory=dict)
    repeat_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Runaway guards for natural mining (never shown to the worker). A hit ends the run as
    # censored ('cost_limit' / 'wall_clock_limit'), not as a worker decision.
    cost_limit_usd: float | None = None
    wall_clock_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("cost_limit_usd", "wall_clock_limit_seconds"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when supplied")
        if type(self.generated_token_limit) is not int or self.generated_token_limit <= 0:
            raise ValueError("Declare a positive generated-token allowance")
        if self.max_model_calls is not None and (type(self.max_model_calls) is not int or self.max_model_calls <= 0):
            raise ValueError("A call cap must be positive when supplied")
        if (type(self.tool_timeout_seconds) not in (int, float) or not math.isfinite(self.tool_timeout_seconds)
                or self.tool_timeout_seconds <= 0):
            raise ValueError("Tool timeout must be finite and positive")
        if not isinstance(self.worker_request_policy, WorkerRequestPolicy):
            raise ValueError("Worker request policy must be a validated record")
        if type(self.max_consecutive_format_errors) is not int or self.max_consecutive_format_errors < 0:
            raise ValueError("Format-error guard must be a nonnegative integer; zero explicitly disables it")


@dataclass(frozen=True, kw_only=True)
class TreatmentManifest:
    treatment_id: str
    source_checkpoint_id: str
    target_id: str
    stage_id: str
    selection_kind: str  # primary_s1 / engineering_checkpoint / secondary
    component_donors: Mapping[str, str]
    history_policy: str
    runtime_policy: str
    worker_model: str
    generated_token_limit: int
    repeat_id: str
    packet_refs: tuple[ArtifactRef, ...] = ()
    source_scope: Mapping[str, Any] = field(default_factory=dict)
    intended_mismatches: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, kw_only=True)
class RunResult:
    run_id: str
    status: str
    termination_reason: str
    checkpoint_ids: tuple[str, ...]
    first_stall_id: str | None = None
    final_evaluation_ref: ArtifactRef | None = None
    accounting_ref: ArtifactRef | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ArtifactStoreProtocol(Protocol):
    def put_bytes(self, data: bytes, *, media_type: str = "application/octet-stream",
                  visibility: Visibility = Visibility.RESEARCH_PRIVATE) -> ArtifactRef: ...
    def put_json(self, value: Any, *, visibility: Visibility = Visibility.RESEARCH_PRIVATE) -> ArtifactRef: ...
    def get_bytes(self, ref: ArtifactRef) -> bytes: ...
    def get_json(self, ref: ArtifactRef) -> Any: ...
    def put_directory(self, directory: Path) -> ArtifactRef: ...
    def append_event(self, event: EventRecord) -> EventRecord: ...
    def iter_events(self, run_id: str, *, through_sequence: int | None = None) -> Iterator[EventRecord]: ...
    def save_checkpoint(self, checkpoint: Checkpoint) -> ArtifactRef: ...
    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint: ...
    def list_checkpoints(self, run_id: str) -> list[Checkpoint]: ...
    def save_annotation(self, annotation: Annotation) -> ArtifactRef: ...
    def put_record(self, kind: str, record_id: str, value: Any, *, run_id: str | None = None) -> ArtifactRef: ...
    def get_record(self, kind: str, record_id: str) -> Any: ...
    def list_records(self, kind: str, *, run_id: str | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class TaskAdapter(Protocol):
    def describe(self) -> TaskRecord: ...
    def stage(self, stage_id: str) -> StageSpec: ...
    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        """Write only legitimate worker starter material; never a reference solution."""
        ...
    def grade(self, workspace: Path, *, stage_id: str,
              private_output_dir: Path | None = None,
              runtime: RuntimeExport | None = None) -> Evaluation:
        """Grade an already isolated copy. The caller owns live-worker isolation."""
        ...
    def next_stage(self, stage_id: str, *, termination_reason: str) -> StageSpec | None:
        """Apply native release semantics; a stop reason is not automatically advancement."""
        ...


@runtime_checkable
class ScoringAdapter(Protocol):
    def score(self, checks: Sequence[CheckResult], *, stage: StageSpec) -> Evaluation: ...


@runtime_checkable
class Runner(Protocol):
    def run(self, task: TaskAdapter, config: RunConfig, *, store: ArtifactStoreProtocol,
            resume: Checkpoint | None = None,
            treatment: TreatmentManifest | None = None) -> RunResult: ...
