"""Opt-in campaign gate over durable live T and existing worker accounting.

The runner remains authoritative for attempts, caps and stopping; continuation
analysis remains authoritative for voluntary-finish allowance absorption.
"""
from __future__ import annotations

from .analysis.continuations import BudgetClock
from .observers.readiness_v1 import inspect_live_t_queue

VERSION = "campaign-launch-contract.v1"
METADATA_KEY = "campaign_launch_contract"


def requested(config):
    value = config.mining.metadata.get(METADATA_KEY)
    if value is not None and value != VERSION:
        raise ValueError("Unsupported explicit campaign launch contract")
    return value == VERSION


def inspect_contract(store, config, natural, *, extractor, context_counter, expected_sources):
    live_t = inspect_live_t_queue(store, natural.run_id, namespace=config.campaign_id,
        config=extractor, include_returned_reasoning=config.observer_include_returned_reasoning,
        context_counter=context_counter, expected_sources=expected_sources)
    clock = BudgetClock.from_ledger(store, natural.run_id)
    exact = clock.complete and clock.identity_unverified_records == 0
    return {"version": VERSION, "run_id": natural.run_id, "live_t": live_t,
        "recovery_source_ready": live_t["complete_at_recorded_cut"],
        "recovery_gate_scope": "all recorded checkpoints; no packet or branch work until exact live-T coverage",
        "worker_allowance": {"assigned_generated_tokens": config.mining.generated_token_limit,
            "known_generated_tokens": clock.total_known, "accounting_complete": exact,
            "unknown_attempts": clock.unknown_attempts, "unplaced_records": clock.unplaced_records,
            "identity_unverified_records": clock.identity_unverified_records,
            "remaining_generated_tokens": (max(0, config.mining.generated_token_limit - clock.total_known)
                                           if exact else None),
            "run_status": natural.status, "termination_reason": natural.termination_reason,
            "termination": dict(natural.details.get("termination", {})),
            "curve_policy": "existing ContinuationRecord absorption; never extend actual-compute coverage",
            "cost_policy": "existing CostLedger actual charges and missing prices; no allowance charge"}}
