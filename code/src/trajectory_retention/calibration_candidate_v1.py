"""Explicit opt-in mining candidate; no production factory imports this module.

Task-specific native witness validators are supplied explicitly. This integration
does not derive certificates from check status or qualify any sealed task.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from .campaign import validate_live_observer, _observer_counter
from .contracts import to_jsonable
from .detector.threshold_policy_v1 import validate_detector_assignments
from .execution.received_response_guard_v1 import ReceivedResponseGuardV1, RULE
from .mining.runner import MiningRunner
from .observers.extractor import ExtractorConfig
from .observers.live import LiveObserverService
from .observers.readiness_v1 import inspect_live_t_queue, source_binding
from .registry import validate_task_purpose
from .scoring.certified_frontier_v1 import apply_certified_frontier

VERSION = "calibration-integration-candidate.v1"


class CandidateMiningRunnerV1(MiningRunner):
    def __init__(self, task, config, *, threshold_profile, assignments,
                 frontier_policies, certificate_provider, scratch, api_key_env):
        self.assignment_report = validate_detector_assignments(threshold_profile, assignments=assignments)
        record = task.describe()
        rows = [row for row in self.assignment_report["assignments"]
                if (row["task_id"], row["model"]) == (record.task_id, config.model)]
        if len(rows) != 1 or rows[0]["detector_settings"] != dict(config.detector_settings):
            raise ValueError("Candidate requires an exact executed task/model threshold assignment")
        self.policies = dict(frontier_policies)
        if set(self.policies) != {stage.stage_id for stage in record.stages}:
            raise ValueError("Candidate requires one explicit frontier policy per native stage")
        for stage in record.stages:
            policy = self.policies[stage.stage_id]
            if (policy.task_id, policy.stage_id, policy.target_id, policy.evaluator_version) != (
                    record.task_id, stage.stage_id, stage.target_id, stage.evaluator_version):
                raise ValueError("Candidate frontier policy differs from task/stage/evaluator")
        if (not inspect.isfunction(certificate_provider) or certificate_provider.__closure__
                or not inspect.getsourcefile(certificate_provider)):
            raise ValueError("Candidate requires a source-backed native certificate function")
        self.certificate_provider = certificate_provider
        self.validator_path = Path(inspect.getsourcefile(certificate_provider)).resolve()
        self.validator_bytes = self.validator_path.read_bytes()
        self.bound_task, self.bound_config = to_jsonable(record), to_jsonable(config)

        def model_factory(executed):
            policy = executed.worker_request_policy
            return ReceivedResponseGuardV1(model_name=executed.model,
                model_settings=dict(executed.model_settings), api_key_env=api_key_env,
                max_attempts=policy.max_attempts, backoff_seconds=policy.backoff_seconds,
                timeout_seconds=policy.timeout_seconds,
                usage_recovery_timeout_seconds=policy.usage_recovery_timeout_seconds)

        super().__init__(model_factory=model_factory, worker_image=record.environment_version,
            detector_config=threshold_profile.config_for(rows[0]["category"]),
            scratch_root=Path(scratch), network="none")

    def run(self, task, config, *, store, checkpoint_published=None, resume=None, treatment=None):
        if resume is not None or treatment is not None:
            raise ValueError("Candidate v1 is fresh mining only; recovery remains deferred")
        if to_jsonable(task.describe()) != self.bound_task or to_jsonable(config) != self.bound_config:
            raise ValueError("Candidate task/config changed after explicit binding")
        if self.validator_path.read_bytes() != self.validator_bytes:
            raise ValueError("Candidate native certificate source changed")
        for ref in self.assignment_report["profile"]["evidence_refs"]:
            from .contracts import ArtifactRef, from_jsonable
            store.get_bytes(from_jsonable(ArtifactRef, ref))
        binding = {"version": VERSION, "task": self.bound_task, "config": self.bound_config,
            "threshold_assignments": self.assignment_report,
            "frontier_policies": to_jsonable(self.policies), "received_response_guard": RULE,
            "native_validator": {"module": self.certificate_provider.__module__,
                "qualname": self.certificate_provider.__qualname__,
                "source_ref": to_jsonable(store.put_bytes(self.validator_bytes)),
                "source_sha256": hashlib.sha256(self.validator_bytes).hexdigest()},
            "integration_source_ref": to_jsonable(store.put_bytes(Path(__file__).read_bytes()))}
        store.put_record("calibration_candidate_binding_v1", config.run_id, binding, run_id=config.run_id)
        return super().run(task, config, store=store, checkpoint_published=checkpoint_published)

    def _record_grade(self, run, checkpoint_id, w_data, grade_dir, ws, grade_kwargs,
                      evaluation, error, started, elapsed, *, source_binding):
        # This is after native grading and before canonical evaluation persistence.
        # Keep the native evaluation independently, including unknown raw fields.
        if error is None:
            original = evaluation
            original_ref = run.recorder.persist(lambda: run.store.put_record(
                "candidate_native_evaluation_v1", original.evaluation_id, original,
                run_id=run.config.run_id), evidence=original)
            try:
                def certify():
                    if self.validator_path.read_bytes() != self.validator_bytes:
                        raise ValueError("Candidate native certificate source changed")
                    certs = tuple(self.certificate_provider(original, private_output_dir=grade_dir / "private",
                                                           store=run.store))
                    for certificate in certs:
                        for ref in certificate.evidence_refs:
                            run.store.get_bytes(ref)
                    if self.validator_path.read_bytes() != self.validator_bytes:
                        raise ValueError("Candidate native certificate source changed during validation")
                    return apply_certified_frontier(original,
                        policy=self.policies[original.stage_id], certificates=certs)
                evaluation = self._timed_call(run, "candidate_frontier_certify", certify, role="grading")
                from dataclasses import replace
                evaluation = replace(evaluation, details={**evaluation.details,
                    "candidate_version": VERSION, "native_evaluation_ref": to_jsonable(original_ref)})
            except BaseException as exc:
                # The parent records grading failure; it never retries the evaluator.
                error = exc
        return super()._record_grade(run, checkpoint_id, w_data, grade_dir, ws, grade_kwargs,
            evaluation, error, started, elapsed, source_binding=source_binding)


def run_candidate_mining(task, config, *, store, observer_client, threshold_profile,
                         assignments, frontier_policies, certificate_provider,
                         scratch, api_key_env):
    """Run only when explicitly called; uses the real runner and ordered live T.

    This candidate entry point is not the public campaign launcher. It returns a
    readiness report, not admission or permission for a T-dependent recovery.
    """
    if config.mode != "mine" or config.source_natural_run_id is not None:
        raise ValueError("Candidate requires fresh mining with recovery deferred")
    if config.mining.purpose == "main":
        raise ValueError("Candidate v1 is not a qualified main-campaign entry point")
    if next(iter(store.iter_events(config.mining.run_id)), None) is not None:
        raise ValueError("Candidate requires a fresh run identity")
    validate_task_purpose(task.describe(), config.mining.purpose)
    validate_live_observer(config, require_live=True)
    runner = CandidateMiningRunnerV1(task, config.mining, threshold_profile=threshold_profile,
        assignments=assignments, frontier_policies=frontier_policies,
        certificate_provider=certificate_provider, scratch=scratch, api_key_env=api_key_env)
    extractor = ExtractorConfig(model=config.mining.model, settings=config.observer_settings,
                               context_policy=config.observer_context_policy)
    counter = _observer_counter(config)
    sources = source_binding()
    observer = LiveObserverService(store, config.mining.run_id, observer_client, extractor,
        namespace=config.campaign_id, include_returned_reasoning=config.observer_include_returned_reasoning,
        policy=config.live_observer, context_counter=counter)
    observer.start()
    try:
        result = runner.run(task, config.mining, store=store,
                            checkpoint_published=observer.checkpoint_published)
    finally:
        coverage = observer.finish()
    readiness = inspect_live_t_queue(store, config.mining.run_id, namespace=config.campaign_id,
        config=extractor, include_returned_reasoning=config.observer_include_returned_reasoning,
        context_counter=counter, expected_sources=sources)
    report = {"version": VERSION, "run_result": to_jsonable(result),
              "live_observer": coverage, "readiness": readiness, "recovery": "deferred"}
    store.put_record("calibration_candidate_result_v1", config.mining.run_id, report,
                     run_id=config.mining.run_id)
    return report
