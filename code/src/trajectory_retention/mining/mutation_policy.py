"""Task/version declarations for grading and recurring-failure relevance.

The recording layer must still retain every change. This layer decides whether
that evidence warrants an optional post-batch grade or a failure-recurrence edit;
it never removes archives, suppresses mandatory grades, or guesses shell effects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from typing import Any, Iterable

from ..contracts import to_jsonable

VERSION = "mutation-policy/v2"


def _patterns(values: tuple[str, ...]) -> None:
    if any(type(value) is not str or not value for value in values):
        raise ValueError("Mutation patterns must be nonempty strings")


@dataclass(frozen=True)
class CheckMutationRule:
    check_id: str
    workspace_patterns: tuple[str, ...] = ()
    runtime_patterns: tuple[str, ...] = ()
    mounted_state_relevant: bool = False
    reason: str = "Declared task/check path dependency."
    unmatched_relevance: str = "irrelevant"  # historical closed-world policy; open code uses unknown

    def __post_init__(self) -> None:
        if type(self.check_id) is not str or not self.check_id or type(self.reason) is not str or not self.reason.strip():
            raise ValueError("Check mutation rule requires check identity and reason")
        _patterns(self.workspace_patterns)
        _patterns(self.runtime_patterns)
        if type(self.mounted_state_relevant) is not bool:
            raise ValueError("mounted_state_relevant must be boolean")
        if self.unmatched_relevance not in {"irrelevant", "unknown"}:
            raise ValueError("Unmatched relevance must be irrelevant or unknown")


@dataclass(frozen=True)
class MutationPolicy:
    policy_id: str
    version: str
    task_id: str
    task_version: str
    workspace_patterns: tuple[str, ...]
    runtime_patterns: tuple[str, ...]
    mounted_state_relevant: bool
    check_rules: tuple[CheckMutationRule, ...] = ()
    reason: str = "Declared task state dependencies; full recording is unchanged."
    unmatched_relevance: str = "irrelevant"

    def __post_init__(self) -> None:
        for field in ("policy_id", "version", "task_id", "task_version", "reason"):
            if type(getattr(self, field)) is not str or not getattr(self, field).strip():
                raise ValueError(f"Mutation policy requires {field}")
        _patterns(self.workspace_patterns)
        _patterns(self.runtime_patterns)
        if type(self.mounted_state_relevant) is not bool:
            raise ValueError("mounted_state_relevant must be boolean")
        if any(not isinstance(rule, CheckMutationRule) for rule in self.check_rules):
            raise ValueError("Malformed check mutation rule")
        ids = [rule.check_id for rule in self.check_rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate check mutation mapping")
        if self.unmatched_relevance not in {"irrelevant", "unknown"}:
            raise ValueError("Unmatched relevance must be irrelevant or unknown")

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def _matches(paths: list[str], patterns: tuple[str, ...], *, runtime: bool = False) -> list[str]:
    # Retain every recorded path verbatim. R entries are root-relative while
    # deletions are absolute; matching treats both as container-root paths.
    # There is no filesystem resolution, case folding or nuisance exclusion.
    key = (lambda path: "/" + path.lstrip("/")) if runtime else (lambda path: path)
    return [path for path in paths if any(fnmatchcase(key(path), key(pattern)) for pattern in patterns)]


def evaluate_mutation(*, task_id: str, task_version: str, workspace_paths: Iterable[str],
                      runtime_paths: Iterable[str], mounts_changed: bool,
                      within_batch_mutation: bool, policy: MutationPolicy | None = None,
                      intermediate_evidence_complete: bool = False) -> dict[str, Any]:
    """Evaluate a measured boundary (or each measured intermediate action).

    Unknown policy/localization keeps a conservative grade trigger and explicitly
    unavailable recurrence relevance. For net-zero batches the caller supplies
    the union of retained intermediate paths and sets evidence_complete only when
    all intermediate changes are represented; individual W states still feed the
    detector in their original order.
    """
    workspace, runtime = sorted(set(workspace_paths)), sorted(set(runtime_paths))
    if any(type(path) is not str or not path for path in (*workspace, *runtime)):
        raise ValueError("Measured mutation paths must be nonempty strings")
    if any(type(value) is not bool for value in (mounts_changed, within_batch_mutation, intermediate_evidence_complete)):
        raise ValueError("Mutation observations require explicit booleans")
    if policy is not None and (policy.task_id, policy.task_version) != (task_id, task_version):
        raise ValueError("Mutation policy binding does not match task/version")
    observed = {"workspace_paths": workspace, "runtime_paths": runtime,
                "runtime_path_matching": "container_root_absolute.v1; original observed strings retained",
                "mounts_changed": mounts_changed, "within_batch_mutation": within_batch_mutation,
                "intermediate_evidence_complete": intermediate_evidence_complete}
    changed = bool(workspace or runtime or mounts_changed or within_batch_mutation)
    incomplete = within_batch_mutation and not intermediate_evidence_complete
    failure = {"rule_version": VERSION, "task_id": task_id, "task_version": task_version,
               "status": "unavailable", "policy_id": policy.policy_id if policy else None,
               "version": policy.version if policy else None, "mapped_check_ids": [],
               "relevant_check_ids": None, "matches": {}}
    if not changed:
        relevant, reason = False, "No observed mutation."
    elif policy is None:
        relevant, reason = None, "No task/version mutation policy; relevance unavailable."
    elif incomplete:
        relevant, reason = None, "Recorded intermediate mutation lacks complete intermediate path evidence."
    else:
        relevant = bool(_matches(workspace, policy.workspace_patterns) or _matches(runtime, policy.runtime_patterns, runtime=True)
                        or mounts_changed and policy.mounted_state_relevant)
        reason = "Task/version policy evaluated all observed changes."
        if not relevant and policy.unmatched_relevance == "unknown":
            relevant, reason = None, "Observed changes are outside known dependencies; open dependency scope remains unknown."
    if policy is not None and not incomplete:
        matches = {}
        for rule in policy.check_rules:
            w = _matches(workspace, rule.workspace_patterns)
            r = _matches(runtime, rule.runtime_patterns, runtime=True)
            mounts = mounts_changed and rule.mounted_state_relevant
            matches[rule.check_id] = {"workspace_paths": w, "runtime_paths": r,
                "mounts_changed": mounts, "reason": rule.reason,
                "relevance": True if w or r or mounts else None if changed and rule.unmatched_relevance == "unknown" else False}
        failure.update(status="available", mapped_check_ids=sorted(matches), matches=matches,
            relevant_check_ids=sorted(check_id for check_id, match in matches.items()
                if match["relevance"] is True),
            unknown_check_ids=sorted(check_id for check_id, match in matches.items() if match["relevance"] is None))
        if failure["unknown_check_ids"]:
            failure["status"] = "partially_available"
        if failure["relevant_check_ids"]:
            # A declared check dependency also establishes a task dependency,
            # even when the broader task-level path patterns omit that file.
            relevant = True
            reason = "At least one declared native check depends on an observed change."
    return {"rule_version": VERSION, "task_id": task_id, "task_version": task_version,
            "policy": policy.to_record() if policy else None, "observed": observed,
            "grade_required": changed and relevant is not False, "task_relevant": relevant,
            "reason": reason, "failure_relevance": failure}
