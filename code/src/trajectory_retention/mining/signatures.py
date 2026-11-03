"""Version-bound failure identity with explicit diagnostic missingness.

All supplied check detail participates unless a policy names a string field and
declares its nuisance substitution. This deliberately does not guess which
paths, numbers or frames a benchmark considers irrelevant. Raw check records are
retained alongside normalization so later parser calibration can replay them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any

from ..contracts import Evaluation, canonical_json, to_jsonable

VERSION = "failure-signature/v2"
_DETAIL_KEYS = {
    "error_class": ("error_class", "exception_type", "exception_kind", "error_type"),
    "frames": ("frames", "traceback_frames", "traceback"),
    "message": ("message", "error_message", "exception_message", "error", "longrepr", "stderr", "stdout"),
    "expected": ("expected", "expected_value", "expected_stdout", "expected_stdout_hex", "expected_returncode"),
    "actual": ("actual", "actual_value", "actual_stdout", "actual_stdout_hex", "actual_returncode"),
}


def _nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


@dataclass(frozen=True)
class NuisanceRule:
    """An exact check-detail path, optionally using '*' for one collection level.

    Substitution is restricted to a named string value; numeric values, whole
    diagnostic objects and unnamed fields cannot disappear through this rule.
    """
    path: tuple[str, ...]
    pattern: str
    replacement: str
    reason: str
    action: str = "substitute"

    def __post_init__(self) -> None:
        if not self.path or any(type(part) is not str or not part for part in self.path):
            raise ValueError("Nuisance rule requires a nonempty detail path")
        if self.action not in ("substitute", "omit_field"):
            raise ValueError("Unknown nuisance normalization action")
        if type(self.pattern) is not str or type(self.replacement) is not str or (self.action == "substitute" and not self.pattern):
            raise ValueError("Nuisance rule requires a string pattern and replacement")
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("Nuisance rule requires its declared reason")
        re.compile(self.pattern)
        if self.action == "omit_field" and (self.path[-1] in ("*", "$raw_status") or self.path[-1] in {
                key for aliases in _DETAIL_KEYS.values() for key in aliases}):
            raise ValueError("Cannot omit an unnamed field or a meaningful diagnostic field")

    @classmethod
    def omit_field(cls, *, path: tuple[str, ...], reason: str) -> NuisanceRule:
        """Declare a named nuisance measurement, including a numeric timestamp."""
        return cls(path=path, pattern="", replacement="", reason=reason, action="omit_field")


@dataclass(frozen=True)
class FailureSignaturePolicy:
    policy_id: str
    version: str
    task_id: str
    task_version: str
    adapter_id: str
    adapter_version: str
    evaluator_version: str
    nuisance_rules: tuple[NuisanceRule, ...] = ()
    allow_status_only: bool = False
    status_only_reason: str | None = None

    def __post_init__(self) -> None:
        for field in ("policy_id", "version", "task_id", "task_version", "adapter_id", "adapter_version", "evaluator_version"):
            if type(getattr(self, field)) is not str or not getattr(self, field).strip():
                raise ValueError(f"Failure signature policy needs {field}")
        if type(self.allow_status_only) is not bool:
            raise ValueError("allow_status_only must be boolean")
        if self.allow_status_only and (type(self.status_only_reason) is not str or not self.status_only_reason.strip()):
            raise ValueError("A status-only signature policy requires a reason")
        if any(not isinstance(rule, NuisanceRule) for rule in self.nuisance_rules):
            raise ValueError("Malformed nuisance rule")

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _normalize(value: Any, rule: NuisanceRule, *, check_id: str,
               changes: list[dict[str, Any]], path: tuple[str, ...] = ()) -> Any:
    remaining = rule.path[len(path):]
    if not remaining:
        if rule.action == "omit_field":
            changes.append({"check_id": check_id, "path": list(path), "rule": to_jsonable(asdict(rule)),
                            "before": value, "action": "omitted_declared_nuisance"})
            return {"$declared_nuisance": True}
        if value is None:
            return None  # Explicitly unavailable diagnostics remain unavailable.
        if type(value) is not str:
            raise ValueError(f"Nuisance path {rule.path!r} must name a string value")
        normalized, count = re.subn(rule.pattern, rule.replacement, value)
        if count:
            changes.append({"check_id": check_id, "path": list(path), "substitutions": count,
                            "rule": to_jsonable(asdict(rule)), "before": value, "after": normalized})
        return normalized
    selector = remaining[0]
    if isinstance(value, dict):
        return {key: _normalize(child, rule, check_id=check_id, changes=changes, path=(*path, key))
                if selector in ("*", key) else child for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize(child, rule, check_id=check_id, changes=changes, path=(*path, str(index)))
                if selector in ("*", str(index)) else child for index, child in enumerate(value)]
    return value  # An absent path is unavailable, never synthesized.


def _observed_details(details: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    observed: dict[str, list[dict[str, Any]]] = {name: [] for name in _DETAIL_KEYS}

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in sorted(value.items()):
                for category, keys in _DETAIL_KEYS.items():
                    if key in keys and _nonempty(child):
                        observed[category].append({"path": list((*path, key)), "value": child})
                visit(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(details)
    return observed


def build_failure_signature(evaluation: Evaluation, *, task_version: str, adapter_id: str,
                            adapter_version: str, policy: FailureSignaturePolicy | None = None) -> dict[str, Any]:
    """Return replayable identity inputs and eligibility for failure recurrence.

    Without an adapter policy, use lossless details and require some diagnostic
    detail for each failed check. An adapter can explicitly justify coarse binary
    identities using allow_status_only. Neither path limits workspace recurrence.
    """
    binding = {"task_id": evaluation.task_id, "task_version": task_version,
               "adapter_id": adapter_id, "adapter_version": adapter_version,
               "evaluator_version": evaluation.evaluator_version}
    if policy is None:
        policy = FailureSignaturePolicy(policy_id="lossless-available-check-detail", version="1", **binding)
    if any(getattr(policy, key) != value for key, value in binding.items()):
        raise ValueError("Failure signature policy binding does not match the graded adapter/task/evaluator")
    ids = [check.check_id for check in evaluation.checks]
    if len(ids) != len(set(ids)):
        raise ValueError("Failure signature rejects duplicate native check IDs")
    raw = [to_jsonable(check) for check in sorted(evaluation.checks, key=lambda check: check.check_id)]
    changes: list[dict[str, Any]] = []
    normalized = []
    unavailable, observed = {}, {}
    detail_missing = []
    for check in raw:
        if check["status"] == "PASS":
            continue
        details = to_jsonable(check["details"])
        raw_status = check["raw_status"]
        for rule in policy.nuisance_rules:
            if rule.path[0] == "$raw_status":
                normalized_status = _normalize({"$raw_status": raw_status}, rule,
                    check_id=check["check_id"], changes=changes)
                raw_status = normalized_status["$raw_status"]
            else:
                details = _normalize(details, rule, check_id=check["check_id"], changes=changes)
        detail = _observed_details(details)
        observed[check["check_id"]] = detail
        unavailable[check["check_id"]] = [name for name, values in detail.items() if not values]
        if not any(detail.values()):
            detail_missing.append(check["check_id"])
        # Evidence refs retain original diagnostics but may hash nuisance bytes.
        # Do not reintroduce those hashes into normalized identity. Raw status is
        # retained because an adapter can expose meaningful failure detail there.
        normalized.append({"check_id": check["check_id"], "status": check["status"],
                           "raw_status": raw_status, "details": details})
    identity = {"rule_version": VERSION, "binding": binding, "policy": policy.to_record(),
                "stage_id": evaluation.stage_id, "target_id": evaluation.target_id, "checks": normalized}
    status_digest = _digest({"binding": binding, "stage_id": evaluation.stage_id, "target_id": evaluation.target_id,
                             "checks": [[check["check_id"], check["status"]] for check in raw]}) if evaluation.valid else None
    if not evaluation.valid:
        status = "invalid_grade"
    elif not normalized:
        status = "no_failures"
    elif detail_missing and not policy.allow_status_only:
        status = "failure_detail_unavailable"
    else:
        status = "available"
    digest = _digest(identity)
    return {"rule_version": VERSION, "binding": binding, "policy": policy.to_record(),
            "evaluation_id": evaluation.evaluation_id, "signature": digest if status == "available" else None,
            "normalized_digest": digest, "status_digest": status_digest, "recurrence_status": status,
            "granularity": "check_status_only_declared" if detail_missing and policy.allow_status_only else (
                "available_check_detail" if not detail_missing else "check_status_only_unavailable"),
            "failed_check_ids": [check["check_id"] for check in normalized],
            "checks_without_semantic_detail": detail_missing, "unavailable_detail": unavailable,
            "observed_detail": observed, "raw_checks": raw, "normalized_checks": normalized,
            "normalization_changes": changes}
