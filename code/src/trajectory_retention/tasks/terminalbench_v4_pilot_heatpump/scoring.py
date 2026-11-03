"""Frozen lane score: share of the task's hidden tests (native case checks) that pass (terminalbench-v4-pilot-heatpump).

Copied from the terminalbench-v2 lane (lanes never import each other) with one change: the native TB4 result is Harbor's
reward file, not pytest's exit status (production-planning's test.sh always exits 0 and only writes reward.txt).

* Every native case id enumerated from tests/expected.json and reported by native tests/test.sh (frozen,
  ``private/checks.json``) is one raw check and one atomic capability in requirement group ``native_tests``;
  score = passing checks / all checks.
* ``full_solve`` is the native Terminal-Bench reward (reward.json or reward.txt = 1, read as Harbor reads it) and every
  frozen check passing.
* ``details.failing_tests`` (sorted frozen check ids whose status is not PASS) and ``details.share_passed`` are
  research-private.
* FAIL, ERROR_TASK, BLOCKED and NOT_RUN earn no credit; the denominator never shrinks. Any INFRA_ERROR invalidates the
  observation (score None).
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

from trajectory_retention.contracts import CapabilityResult, CheckResult, CheckStatus, Evaluation, StageSpec

VERSION = "terminalbench-v4-pilot-heatpump.per-test-share.v1"
GROUP = "native_tests"


class NativeCheckFractionScorer:
    def __init__(self, checks_path: Path, *, task_id: str, binding: str, metrics_path: Path | None = None):
        data = json.loads(Path(checks_path).read_text())
        self.task_id = task_id
        self.expected = list(data["checks"])
        if not self.expected or len(set(self.expected)) != len(self.expected):
            raise ValueError("Frozen check list must be non-empty and unique")
        self.owner = {cid: "native::" + cid.split("::", 1)[-1] for cid in self.expected}
        self.metrics = json.loads(Path(metrics_path).read_text()) if metrics_path and Path(metrics_path).is_file() else {}
        payload = Path(checks_path).read_bytes() + Path(__file__).read_bytes() + binding.encode()
        if metrics_path and Path(metrics_path).is_file():
            payload += Path(metrics_path).read_bytes()
        self.version = VERSION + ":" + hashlib.sha256(payload).hexdigest()

    def native_metrics(self, checks: Sequence[CheckResult]) -> dict:
        found = {}
        for spec in self.metrics.get("metrics", []):
            text = next((c.details.get("diagnostic_tail", "") for c in checks if c.check_id == spec["check_id"]), "")
            match = re.search(spec["regex"], text or "")
            found[spec["name"]] = float(match.group(1)) if match else None
        return found

    def score(self, checks: Sequence[CheckResult], *, stage: StageSpec, native_reward: float | None,
              test_sh_exit: int | None, extra: dict | None = None) -> Evaluation:
        if stage.evaluator_version != self.version:
            raise ValueError("Evaluator identity changed")
        rows: dict[str, CheckResult] = {}
        for row in checks:
            if row.check_id not in self.owner or row.check_id in rows:
                raise ValueError("Unknown or duplicate check result: " + row.check_id)
            rows[row.check_id] = replace(row, capability_id=self.owner[row.check_id])
        for cid in self.expected:
            rows.setdefault(cid, CheckResult(check_id=cid, capability_id=self.owner[cid], status=CheckStatus.NOT_RUN,
                                             details={"reachable": None, "blocking_reason": "missing_observation"}))
        ordered = tuple(rows[cid] for cid in self.expected)
        valid = not any(c.status == CheckStatus.INFRA_ERROR for c in ordered)
        capabilities = []
        for cid in self.expected:
            row = rows[cid]
            reachable = row.details.get("reachable") if valid else None
            capabilities.append(CapabilityResult(capability_id=self.owner[cid], requirement_group=GROUP,
                completion=float(row.status == CheckStatus.PASS) if valid else None, reachable=reachable,
                check_ids=(cid,), milestone_ids=(f"{self.task_id}:{self.owner[cid]}:observable.v1",) if reachable else ()))
        passed = sum(rows[c].status == CheckStatus.PASS for c in self.expected)
        score = passed / len(self.expected) if valid else None
        native_reward = (1 if native_reward == 1.0 else 0) if valid and native_reward is not None else None
        full = (native_reward == 1 and passed == len(self.expected)) if valid else None
        details = {"score_rule": "passing native checks / all frozen native checks (one per native case id)",
                   "native_reward_rule": "Harbor reward: /logs/verifier/reward.json, else reward.txt; 1 = pass",
                   "native_reward": native_reward, "test_sh_exit": test_sh_exit,
                   "checks_passed": passed if valid else None, "checks_total": len(self.expected),
                   "status_counts": {s.value: sum(r.status == s for r in ordered) for s in CheckStatus},
                   "native_metrics": self.native_metrics(ordered) if valid else None,
                   "native_metrics_rule": "declared regex over the native failure message; absent when the check passes "
                                          "or the message does not print it; never part of the score",
                   "missing_checks_keep_denominator": True, "mapping_frozen": True,
                   "failing_tests": sorted(c.check_id for c in ordered if c.status != CheckStatus.PASS) if valid else None,
                   "share_passed": score, **(extra or {})}
        return Evaluation(task_id=self.task_id, stage_id=stage.stage_id, target_id=stage.target_id,
            evaluator_version=self.version, score=score, full_solve=full, valid=valid, checks=ordered,
            capabilities=tuple(capabilities), group_scores={GROUP: score},
            reachable_milestones=tuple(m for c in capabilities for m in c.milestone_ids),
            raw_score=float(passed) if valid else None, raw_max_score=float(len(self.expected)), details=details)
