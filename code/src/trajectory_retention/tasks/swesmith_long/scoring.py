"""Frozen lane score: fraction of FAIL_TO_PASS tests passing; PASS_TO_PASS tracked separately.

* Every FAIL_TO_PASS test is one capability in group ``fail_to_pass``
  (completion 1 iff its check is PASS). score = mean completion of that group.
* PASS_TO_PASS tests are raw checks grouped into one capability per test file in
  group ``pass_to_pass``; that group is reported (group_scores, details) but has
  zero weight in ``score``. Regressions are listed explicitly.
* full_solve is the native SWE-smith "resolved" rule: every FAIL_TO_PASS and
  every retained PASS_TO_PASS check passes.
* FAIL, ERROR_TASK, BLOCKED and NOT_RUN earn no credit; the denominator never
  shrinks. Any INFRA_ERROR invalidates the observation (score None).
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

from trajectory_retention.contracts import CapabilityResult, CheckResult, CheckStatus, Evaluation, StageSpec

VERSION = "swesmith-long.f2p-fraction.v1"


class F2PFractionScorer:
    def __init__(self, checks_path: Path, *, task_id: str, binding: str):
        data = json.loads(Path(checks_path).read_text())
        self.task_id = task_id
        self.f2p = list(data["fail_to_pass"])
        self.p2p = list(data["pass_to_pass"])
        if not self.f2p or len(set(self.f2p)) != len(self.f2p) or len(set(self.p2p)) != len(self.p2p):
            raise ValueError("Frozen check lists must be non-empty and unique")
        if set(self.f2p) & set(self.p2p):
            raise ValueError("A check cannot be both FAIL_TO_PASS and PASS_TO_PASS")
        self.owner = {cid: "f2p::" + cid for cid in self.f2p}
        self.p2p_capabilities: dict[str, list[str]] = {}
        for cid in self.p2p:
            capability = "p2p::" + cid.split("::", 1)[0]
            self.owner[cid] = capability
            self.p2p_capabilities.setdefault(capability, []).append(cid)
        self.expected = self.f2p + self.p2p
        self.version = VERSION + ":" + hashlib.sha256(
            Path(checks_path).read_bytes() + Path(__file__).read_bytes() + binding.encode()).hexdigest()

    def score(self, checks: Sequence[CheckResult], *, stage: StageSpec) -> Evaluation:
        if stage.evaluator_version != self.version:
            raise ValueError("Evaluator identity changed")
        rows: dict[str, CheckResult] = {}
        for row in checks:
            if row.check_id not in self.owner or row.check_id in rows:
                raise ValueError("Unknown or duplicate check result: " + row.check_id)
            if row.capability_id not in (None, self.owner[row.check_id]):
                raise ValueError("Primary capability cannot change")
            rows[row.check_id] = replace(row, capability_id=self.owner[row.check_id])
        for cid in self.expected:
            rows.setdefault(cid, CheckResult(check_id=cid, capability_id=self.owner[cid], status=CheckStatus.NOT_RUN,
                                             details={"reachable": None, "blocking_reason": "missing_observation"}))
        ordered = tuple(rows[cid] for cid in self.expected)
        valid = not any(c.status == CheckStatus.INFRA_ERROR for c in ordered)

        def reach(members):
            values = [m.details.get("reachable") for m in members]
            if not valid:
                return None
            return True if True in values else False if values and all(v is False for v in values) else None

        capabilities = []
        for cid in self.f2p:
            row = rows[cid]
            reachable = reach([row])
            capabilities.append(CapabilityResult(capability_id=self.owner[cid], requirement_group="fail_to_pass",
                completion=float(row.status == CheckStatus.PASS) if valid else None, reachable=reachable,
                check_ids=(cid,), milestone_ids=(f"{self.task_id}:{self.owner[cid]}:observable.v1",) if reachable else ()))
        for capability, members in self.p2p_capabilities.items():
            observed = [rows[m] for m in members]
            reachable = reach(observed)
            capabilities.append(CapabilityResult(capability_id=capability, requirement_group="pass_to_pass",
                completion=float(all(r.status == CheckStatus.PASS for r in observed)) if valid else None,
                reachable=reachable, check_ids=tuple(members),
                milestone_ids=(f"{self.task_id}:{capability}:observable.v1",) if reachable else ()))
        f2p_pass = sum(rows[c].status == CheckStatus.PASS for c in self.f2p)
        p2p_pass = sum(rows[c].status == CheckStatus.PASS for c in self.p2p)
        p2p_caps = [c for c in capabilities if c.requirement_group == "pass_to_pass"]
        groups = {"fail_to_pass": f2p_pass / len(self.f2p) if valid else None,
                  "pass_to_pass": (sum(c.completion for c in p2p_caps) / len(p2p_caps) if p2p_caps else 1.0) if valid else None}
        score = f2p_pass / len(self.f2p) if valid else None
        return Evaluation(task_id=self.task_id, stage_id=stage.stage_id, target_id=stage.target_id,
            evaluator_version=self.version, score=score,
            full_solve=(f2p_pass == len(self.f2p) and p2p_pass == len(self.p2p)) if valid else None,
            valid=valid, checks=ordered, capabilities=tuple(capabilities), group_scores=groups,
            reachable_milestones=tuple(m for c in capabilities for m in c.milestone_ids),
            raw_score=float(f2p_pass) if valid else None, raw_max_score=float(len(self.f2p)),
            details={"score_rule": "passing FAIL_TO_PASS / all FAIL_TO_PASS; PASS_TO_PASS has zero score weight",
                     "native_resolved_rule": "all FAIL_TO_PASS and retained PASS_TO_PASS pass",
                     "fail_to_pass_passed": f2p_pass if valid else None, "fail_to_pass_total": len(self.f2p),
                     "pass_to_pass_passed": p2p_pass if valid else None, "pass_to_pass_total": len(self.p2p),
                     "pass_to_pass_regressions": [c for c in self.p2p if rows[c].status != CheckStatus.PASS] if valid else None,
                     "status_counts": {s.value: sum(r.status == s for r in ordered) for s in CheckStatus},
                     "missing_checks_keep_denominator": True, "mapping_frozen": True})
