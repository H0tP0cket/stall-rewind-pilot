"""Opt-in frontier aggregation from adapter-validated per-check certificates.

This is an integration seam, not a new behavioral oracle. A task's versioned
translator must first validate its native boundary evidence and issue each
certificate. This module checks the certificate's exact check/evidence binding
and applies the declared capability mapping. A raw FAIL, exception, import or
tool invocation alone never becomes positive or negative frontier evidence.

No current adapter imports this module. Adopting it requires a new task/scoring
binding and task-specific boundary controls. Historical evaluations are intact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping

from trajectory_retention.contracts import (
    ArtifactRef, CheckResult, CheckStatus, Evaluation, to_jsonable,
)

VERSION = "certified-check-frontier/v1"
_POSITIVE_BASES = {"meaningful_behavior_pass", "validated_execution_boundary"}
_NEGATIVE_BASES = {"validated_unreached_boundary"}


def check_digest(check: CheckResult) -> str:
    """Bind status, diagnostics and raw evidence refs, without rewriting them."""
    return hashlib.sha256(json.dumps(to_jsonable(check), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class CheckFrontierCertificate:
    check_id: str
    check_digest: str
    reachable: bool | None
    basis: str
    validator_version: str
    evidence_refs: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CapabilityFrontierRule:
    capability_id: str
    check_ids: tuple[str, ...]
    meaningful_pass_check_ids: tuple[str, ...]
    milestone_id: str


@dataclass(frozen=True, kw_only=True)
class CertifiedFrontierPolicy:
    policy_id: str
    task_id: str
    stage_id: str
    target_id: str
    evaluator_version: str
    validator_version: str
    capabilities: tuple[CapabilityFrontierRule, ...]
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.version != VERSION:
            raise ValueError("Unsupported certified frontier policy version")
        if not all(isinstance(value, str) and value.strip() for value in (
                self.policy_id, self.task_id, self.stage_id, self.target_id,
                self.evaluator_version, self.validator_version)):
            raise ValueError("Declare exact task, target, evaluator and validator bindings")
        if not self.capabilities:
            raise ValueError("Declare a nonempty capability frontier mapping")
        ids, milestones, checks = [], [], []
        for rule in self.capabilities:
            ids.append(rule.capability_id)
            milestones.append(rule.milestone_id)
            checks.extend(rule.check_ids)
            if (not rule.capability_id or not rule.milestone_id or not rule.check_ids
                    or len(rule.check_ids) != len(set(rule.check_ids))
                    or len(rule.meaningful_pass_check_ids) != len(set(rule.meaningful_pass_check_ids))
                    or not set(rule.meaningful_pass_check_ids) <= set(rule.check_ids)):
                raise ValueError("Invalid capability frontier rule")
        if (len(ids) != len(set(ids)) or len(milestones) != len(set(milestones))
                or len(checks) != len(set(checks))):
            raise ValueError("Capabilities, milestones and primary check assignments must be unique")


def apply_certified_frontier(evaluation: Evaluation, *, policy: CertifiedFrontierPolicy,
                             certificates: tuple[CheckFrontierCertificate, ...]) -> Evaluation:
    """Retain certified positives, known negatives, and explicit unknowns.

One certified check reaching the declared capability boundary establishes that
capability's reachability. A negative requires every mapped check to certify
not reaching it. Otherwise the capability is unknown. This is capability-level
reachability; it does not claim every test of a reached capability executed.

Missing certificates are unknown. Malformed, stale or foreign certificates are
rejected. An explicitly invalid envelope/check suppresses its claims. Completion
validity, raw statuses, score, groups and all private diagnostic bytes remain
unchanged. A valid partial positive may therefore survive an unrelated missing
scalar result without certifying the absence of further frontier advancement.
"""
    for field in ("task_id", "stage_id", "target_id", "evaluator_version"):
        if getattr(evaluation, field) != getattr(policy, field):
            raise ValueError(f"Frontier policy {field} does not match evaluation")
    checks = {check.check_id: check for check in evaluation.checks}
    caps = {cap.capability_id: cap for cap in evaluation.capabilities}
    rules = {rule.capability_id: rule for rule in policy.capabilities}
    if len(checks) != len(evaluation.checks) or len(caps) != len(evaluation.capabilities):
        raise ValueError("Duplicate checks or capabilities cannot certify frontier")
    if set(caps) != set(rules) or set(checks) != {c for r in rules.values() for c in r.check_ids}:
        raise ValueError("Evaluation does not cover the declared frontier mapping")
    for cap_id, rule in rules.items():
        if (set(caps[cap_id].check_ids) != set(rule.check_ids)
                or len(caps[cap_id].check_ids) != len(rule.check_ids)):
            raise ValueError("Capability check mapping differs from frozen frontier rule")
        if any(checks[c].capability_id not in (None, cap_id) for c in rule.check_ids):
            raise ValueError("Check primary capability contradicts the frontier mapping")

    claims = {cert.check_id: cert for cert in certificates}
    if len(claims) != len(certificates) or not set(claims) <= set(checks):
        raise ValueError("Duplicate or unknown per-check frontier certificate")
    positive_pass = {cid for rule in rules.values() for cid in rule.meaningful_pass_check_ids}
    observed: dict[str, bool | None] = {}
    reasons: dict[str, str] = {}
    for check_id, check in checks.items():
        cert = claims.get(check_id)
        if cert is None:
            observed[check_id], reasons[check_id] = None, "certificate_missing"
            continue
        if (cert.validator_version != policy.validator_version
                or cert.check_digest != check_digest(check)):
            raise ValueError("Frontier certificate validator or exact check binding differs")
        if cert.reachable is not None and type(cert.reachable) is not bool:
            raise ValueError("Frontier reachability must be boolean or unknown")
        if not set(cert.evidence_refs) <= set(check.evidence_refs):
            raise ValueError("Frontier certificate cites evidence outside its exact check")
        if cert.reachable is None:
            if cert.basis != "unobserved":
                raise ValueError("Unknown frontier requires an unobserved basis")
        elif not cert.evidence_refs:
            raise ValueError("Known frontier requires retained per-check evidence")
        elif cert.reachable is True:
            if cert.basis not in _POSITIVE_BASES:
                raise ValueError("Positive frontier needs a validated behavioral boundary")
            if (check.status not in (CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.ERROR_TASK)
                    or check.details.get("generic_task_crash") is True):
                raise ValueError("Unexecuted or generic-crash check cannot certify positive frontier")
            if cert.basis == "meaningful_behavior_pass" and (
                    check.status != CheckStatus.PASS or check_id not in positive_pass):
                raise ValueError("PASS reachability is not declared for this check")
        elif (cert.basis not in _NEGATIVE_BASES
                or check.status not in (CheckStatus.FAIL, CheckStatus.ERROR_TASK, CheckStatus.BLOCKED)):
            raise ValueError("Negative frontier needs validated pre-boundary failure evidence")
        if (evaluation.details.get("evidence_valid") is False
                or check.details.get("evidence_valid") is False):
            observed[check_id], reasons[check_id] = None, "evidence_invalid"
        else:
            observed[check_id], reasons[check_id] = cert.reachable, cert.basis

    new_caps = []
    for cap in evaluation.capabilities:
        rule = rules[cap.capability_id]
        values = [observed[cid] for cid in rule.check_ids]
        reach = True if any(v is True for v in values) else False if all(v is False for v in values) else None
        new_caps.append(replace(cap, reachable=reach,
            milestone_ids=(rule.milestone_id,) if reach is True else ()))
    unknown = [cap.capability_id for cap in new_caps if cap.reachable is None]
    return replace(evaluation, capabilities=tuple(new_caps),
        reachable_milestones=tuple(m for cap in new_caps for m in cap.milestone_ids),
        details={**evaluation.details, "frontier_policy_version": VERSION,
            "frontier_policy_id": policy.policy_id,
            "frontier_positive_evidence_valid": any(c.reachable is True for c in new_caps),
            "frontier_complete": not unknown, "unknown_reachability": unknown,
            "frontier_certification": {"policy": to_jsonable(policy),
                "certificates": to_jsonable(certificates), "check_reachable": observed,
                "check_observation_reason": reasons}})
