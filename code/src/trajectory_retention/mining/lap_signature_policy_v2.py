"""Additional v12-demonstrated nuisance repairs, opt-in with detector v6.

Every substitution retains raw bytes and before/after evidence in the existing
signature record. Truncated private paths require an exact nonce witnessed in
the same evaluation; arbitrary paths, hex operands and object classes survive.
"""
from dataclasses import replace
import hashlib
import re

from ..contracts import canonical_json
from .lap_signature_policy_v1 import extend_lap_signature_policy as previous
from .signatures import FailureSignaturePolicy, NuisanceRule, build_failure_signature

VERSION = "v13-demonstrated-nuisance/v1"


def extend_lap_signature_policy(policy: FailureSignaturePolicy | None, evaluation=None):
    policy = previous(policy)
    if policy is None or policy.version.endswith(VERSION):
        return policy
    rules = []
    if re.fullmatch(r"scb-circuit-eval-adapted-v[1-9][0-9]*", policy.task_id):
        rules.append(NuisanceRule(path=("native_phases", "*", "longrepr"),
            pattern=r"(?m)^(self = <json\.decoder\.JSONDecoder object at )0x[0-9a-fA-F]+(>, s = )",
            replacement=r"\1<process-allocation>\2",
            reason="Saved v12 Circuit C21/C22 differ only in stdlib JSONDecoder receiver address; input string and exception remain exact."))
    if re.fullmatch(r"scb-(?:circuit-eval|migrate-configs)-adapted-v[1-9][0-9]*", policy.task_id):
        # Full structured native properties establish the private evaluator's
        # generated nonce before its reprlib-truncated rendering is normalized.
        roots = set()
        def visit(value):
            if isinstance(value, str):
                roots.update(re.findall(r"/tmp/scb-eval-([0-9a-f]{12})(?=/)", value))
            elif isinstance(value, dict):
                for key, child in value.items():
                    if key != "longrepr":
                        visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
        if evaluation is not None:
            for check in evaluation.checks:
                visit(check.details.get("native_phases", []))
        for nonce in sorted(roots):
            for prefix, token, replacement in (
                (r"(?<=\.\.\.)tmp/scb-eval-", nonce, "tmp/scb-eval-<private-root>"),
                (r"(?<=\.\.\.)/scb-eval-", nonce, "/scb-eval-<private-root>"),
                (r"(?<=\.\.\.)-eval-", nonce, "-eval-<private-root>"),
                (r"(?<=/tm)\.\.\.", nonce[-6:], "...<private-root>"),
            ):
                rules.append(NuisanceRule(path=("native_phases", "*", "longrepr"),
                    pattern=prefix + re.escape(token) + r"(?=/test_[A-Za-z0-9_])",
                    replacement=replacement,
                    reason="Exact private evaluator nonce from same-grade native properties; only observed reprlib path truncation, preserving test/case/file and operands."))
    if not rules:
        return policy
    return replace(policy, policy_id=policy.policy_id + ":" + VERSION,
                   version=policy.version + ":" + VERSION,
                   nuisance_rules=policy.nuisance_rules + tuple(rules))


def build_attempt_failure_signature(evaluation, *, policy=None, **binding):
    """Nonce-specific substitutions are evidence, not a changing policy identity.

    Hash the frozen derivation procedure and input task policy, rather than
    reintroducing the random nonce through its generated literal regex. Keep
    that exact regex, its source evaluation and every substitution in the record.
    """
    applied = extend_lap_signature_policy(policy, evaluation)
    result = build_failure_signature(evaluation, policy=applied, **binding)
    identity_policy = {"algorithm": VERSION,
        "input_policy": policy.to_record() if policy is not None else None}
    identity = {"rule_version": "failure-signature/v3", "binding": result["binding"],
        "policy": identity_policy, "stage_id": evaluation.stage_id,
        "target_id": evaluation.target_id, "checks": result["normalized_checks"]}
    digest = hashlib.sha256(canonical_json(identity)).hexdigest()
    result.update(rule_version="failure-signature/v3", identity_policy=identity_policy,
                  signature=digest if result["recurrence_status"] == "available" else None,
                  normalized_digest=digest)
    return result
