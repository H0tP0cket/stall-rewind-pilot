"""Resolve source-bound task declarations for detector v3 inputs.

The optional bundle lives in TaskRecord.evidence['detector_input_policies']; it
is part of the immutable task declaration, not an inferred shell-effect policy.
"""
from __future__ import annotations
from typing import Any

from ..contracts import TaskRecord, from_jsonable
from .mutation_policy import MutationPolicy
from .signatures import FailureSignaturePolicy

VERSION = 'detector-input-policies.v1'


def adapter_binding(task: Any, record: TaskRecord) -> dict[str, str]:
    return {'adapter_id': f'{type(task).__module__}.{type(task).__qualname__}',
            'adapter_version': record.task_version}


def resolve_input_policies(task: Any, record: TaskRecord) -> tuple[MutationPolicy | None, dict[str, FailureSignaturePolicy]]:
    raw = record.evidence.get('detector_input_policies')
    if raw is None:
        return None, {}
    if not isinstance(raw, dict) or set(raw) != {'version','adapter_id','adapter_version','mutation','failure_signatures'}:
        raise ValueError('Detector input policy bundle requires the explicit v1 fields')
    binding = adapter_binding(task, record)
    if raw['version'] != VERSION or any(raw[key] != value for key, value in binding.items()):
        raise ValueError('Detector input policy adapter/version binding mismatch')
    mutation = from_jsonable(MutationPolicy, raw['mutation']) if raw['mutation'] is not None else None
    if mutation is not None and (mutation.task_id, mutation.task_version) != (record.task_id, record.task_version):
        raise ValueError('Detector mutation policy task/version binding mismatch')
    if not isinstance(raw['failure_signatures'], dict):
        raise ValueError('Failure signature declarations must be keyed by stage identity')
    stages = {s.stage_id: s for s in record.stages}
    signatures = {}
    for stage_id, value in raw['failure_signatures'].items():
        if stage_id not in stages:
            raise ValueError('Failure signature policy has an unknown stage')
        policy = from_jsonable(FailureSignaturePolicy, value)
        expected = {'task_id':record.task_id,'task_version':record.task_version,
                    'evaluator_version':stages[stage_id].evaluator_version,**binding}
        if any(getattr(policy,key) != actual for key,actual in expected.items()):
            raise ValueError('Failure signature policy task/adapter/evaluator binding mismatch')
        signatures[stage_id] = policy
    return mutation, signatures
