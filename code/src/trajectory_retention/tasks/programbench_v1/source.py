"""Pinned ProgramBench sources and the native lifecycle this lane reproduces.

Everything a package binds is named here: the harness and wheel releases, the hidden-test dataset revision,
the mini-swe-agent config the native prompt comes from, and the native container/budget settings. Our x86
guards (1,000 calls / 12 h / $10, user-ratified 2026-09-28) are the binding limits; native budgets are recorded.
"""
from __future__ import annotations

HARNESS_URL = "https://github.com/facebookresearch/programbench"
HARNESS_REVISION = "b08d8621031f5f5abc4d3ffc2950256c83fbfe42"
WHEEL_NAME = "programbench-1.2.4-py3-none-any.whl"
WHEEL_SHA256 = "bc24d2879a9d0249ae7e4f7770a7df6c069767f491444d846fe0a75070c572b8"
TESTS_DATASET = "programbench/ProgramBench-Tests"
TESTS_DATASET_REVISION = "de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5"
AGENT_CONFIG_URL = ("https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/"
                    "src/minisweagent/config/benchmarks/programbench.yaml")
AGENT_CONFIG_REVISION = "04d809ceab9df28f9adaed044884180159172930"
IMAGE_TAG = "task_cleanroom_v6"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Native container settings (programbench.yaml `environment.run_args`) and budgets (`agent`).
NATIVE_ENVIRONMENT = {"network": "none", "cpus": "20", "memory": "60g", "user": "agent", "cap_drop": ["SYS_PTRACE"],
                      "command_timeout_seconds": 180}
NATIVE_BUDGET = {"step_limit": 1000, "wall_time_limit_seconds": 21600, "cost_limit": 0}
# What this lane runs. User/capabilities/network are native; CPU and memory are below native (declared deviation,
# to be re-set from measured box capacity before mining).
WORKER_ENVIRONMENT = {"user": "agent", "cap_drop": ("SYS_PTRACE",), "cpus": "4", "memory": "16g"}

LIFECYCLE = {
    "policy_id": "programbench.native-single-submission.v1",
    "stage_release_rule": ("single native assignment; the run ends on the first voluntary submission "
                           "(`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` alone) or on a guard; the submitted or final "
                           "W is graded once; no feedback reaches the worker"),
    "context_overflow": ("native: full history, observation head/tail 5,000 characters over 10,000; a context-window "
                         "overflow ends the run and the partial W is graded"),
    "source_evidence": (AGENT_CONFIG_URL, HARNESS_URL + "/blob/" + HARNESS_REVISION + "/src/programbench/eval/eval.py"),
}

DEVIATIONS = (
    "Worker container 4 CPUs / 16 GB instead of native 20 CPUs / 60 GB (to be re-set from measured box capacity).",
    "Binding budgets are the study's x86 guards (1,000 model calls / 12 h / $10); native 1,000 steps / 6 h recorded.",
    "Command timeout is the study's tool timeout, not native 180 s.",
    "The native near-limit warnings in the observation template (<20 steps, <10 minutes left) are not rendered.",
    "The native system template's task framing and RULES are delivered at the head of the task message; the "
    "shared system message is the native first line unchanged.",
    "The native <system_information> block is omitted (host-specific uname).",
    "Tamper guard: a submitted file byte-identical to the reference executable scores 0 and is flagged (upstream "
    "eval_clean_hashes is empty for some tasks, e.g. seqtk); a built binary equal to the reference is only recorded.",
)
