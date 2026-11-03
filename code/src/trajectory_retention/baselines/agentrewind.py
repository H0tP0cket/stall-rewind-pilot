"""Agent-directed aligned rewind, adapted to the fixed-S retention experiment.

This is a controller for RecordingAgent/MiningRunner, not another worker loop.
The runner records model/tool calls normally and registers coherent checkpoints.
Only the worker calls candidates/commit; private grades and observer T never
enter this module's candidate view. A restore callback uses the runner's ordinary
verified workspace and paired-runtime restoration. No retained action is replayed.

The public source is a cleaned alpha, not a certified paper revision. POLICY
pins both references and declares consequential differences. In particular the
study's source feedback, fixed target, finite additional allowance, supported
checkpoint subset and explicit runtime restoration replace MettleBench defaults.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions

from ..contracts import (ArtifactRef, ArtifactStoreProtocol, Checkpoint, EventRecord,
                         RuntimeClass, Visibility, from_jsonable, new_id, to_jsonable)
from ..execution.model import history_replay_issue
from ..execution.environment import comparable, fidelity_manifest_from_tar
from ..observers.prefix import _delivered_tool_events, source_event_from_record
from ..persistence import persist


POLICY = {
    "policy_id": "agentrewind-fixed-s/1",
    "paper": "https://arxiv.org/html/2608.14380v1",
    "paper_version_date": "2026-08-14",
    "public_source": "https://github.com/Futuresis/replay-agent-recorder",
    "public_revision": "6661046e2b30fcf5f17c93d51acb56babdea8c53",
    "source_status": "cleaned public alpha; not certified paper-experiment revision",
    "selection_policy": "worker_agent_direct",
    "candidate_limit": 80,
    "memory_policy": "cumulative_worker_generated",
    "workspace_history_policy": "aligned_checkpoint_restore",
    "runtime_policy": "paired_with_selected_workspace; captured_or_demonstrated_irrelevant",
    "adaptations": [
        "Start from the same prospectively selected S, not a terminal repeated-failure endpoint.",
        "Freeze S task/target/stage and requirement/evaluator versions; no later-stage release.",
        "Only validated complete engine checkpoints are eligible; no fabricated per-call snapshots.",
        "Use source-native worker-visible feedback only; do not inject private evaluator results.",
        "Finite additional generated-worker allowance and call cap never reset after rewind.",
        "Restore declared captured runtime with W, or require R_IRRELEVANT; not filesystem-only attribution.",
        "Candidate previews use accepted delivered worker decisions and delivered tool observations only.",
        "All memory text is retained without alpha's 20-item/900-char/8000-char rendering truncation.",
        "No probabilistic selection, automatic summary model, memory-discipline veto or hidden diagnostic.",
    ],
}

SYSTEM_PROMPT = """Additional recovery tools are available for this fixed-target continuation.
For this continuation a response may use bash OR one recovery tool. Recovery
tools must be the only tool call in that response. All decisions, investigation
and memory writing consume your existing allowance; rewinding does not renew it.

Use backtrack_candidates when concrete observations or repeated unsuccessful
repairs suggest abandoning the current approach. Inspect the candidate decision
and tool previews. Then, if rewinding is preferable, call backtrack_commit with
a returned record_uid identifying a point before the approach you wish to change.
You may continue in place, and may make further recovery decisions after a rewind.

backtrack_commit restores that checkpoint's aligned workspace and conversation,
and its declared runtime. Later conversation is discarded except for your
cumulative memory notebook. Do not assume you can inspect the discarded branch.
Write memory_summary for your earlier self: failed_check, observed_loop,
why_local_repair_failed, root_cause, evidence_observed, assumptions_to_recheck,
rewind_target_reason, selected_candidate_safety_check, do_not_repeat,
corrected_strategy, first_3_tool_calls, and success_invariant. Treat a root cause
as a hypothesis, distinguish observation from inference, and name a concrete
strategy change. Base evidence only on information you actually received.
The fixed target remains the same after every rewind. Research-private grades,
other branches and external observer representations are not available tools.
"""

TOOL_DEFINITIONS = (
    {"type": "function", "function": {"name": "backtrack_candidates",
        "description": "List the latest 80 eligible earlier aligned checkpoints and visible decision/tool previews.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                       "required": ["reason"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "backtrack_commit",
        "description": "Restore a checkpoint from the latest candidate list, carrying your cumulative operational memory.",
        "parameters": {"type": "object", "properties": {
            "record_uid": {"type": "string"}, "memory_summary": {"type": "string"},
            "reason": {"type": "string"}},
            "required": ["record_uid", "memory_summary"], "additionalProperties": False}}},
)

_POLICY_MARKER = "agentrewind_policy"
_MEMORY_MARKER = "agentrewind_memory"


class BaselinePolicyError(ValueError):
    """Invalid worker decision; return a tool error without restoring anything."""


@dataclass(frozen=True)
class RewindPlan:
    plan_id: str
    donor: Checkpoint
    response_event_sequence: int
    memory_summary: str
    reason: str
    generation: int
    memory_ordinal: int


class RewindRequested(Exception):
    """A recorded worker control action requests a runner-owned restore."""

    def __init__(self, plan: RewindPlan) -> None:
        super().__init__(f"Aligned rewind requested: {plan.plan_id}")
        self.plan = plan


def parse_actions(tool_calls: list, *, format_error_template: str,
                  template_kwargs: dict | None = None) -> list[dict]:
    """Optional model parser hook; ordinary bash parsing stays upstream-owned."""
    if not tool_calls or all(c.function.name == "bash" for c in tool_calls):
        return parse_toolcall_actions(tool_calls, format_error_template=format_error_template,
                                      template_kwargs=template_kwargs)
    error = None
    if len(tool_calls) != 1:
        error = "A recovery tool must be the only tool call in its response."
    else:
        call = tool_calls[0]
        try:
            args = json.loads(call.function.arguments)
        except (ValueError, TypeError):
            args = None
        name = call.function.name
        allowed = {"backtrack_candidates": ({"reason"}, {"reason"}),
                   "backtrack_commit": ({"record_uid", "memory_summary"}, {"record_uid", "memory_summary", "reason"})}
        if name not in allowed:
            error = f"Unknown tool {name!r}."
        elif (not isinstance(args, dict) or not allowed[name][0].issubset(args)
              or not set(args).issubset(allowed[name][1]) or any(not isinstance(v, str) for v in args.values())):
            error = "Recovery arguments must contain exactly the declared string fields."
        elif not call.id:
            error = "Recovery tool call needs a provider call ID."
        else:
            return [{"tool_name": name, "arguments": args, "tool_call_id": call.id}]
    raise FormatError({"role": "user", "content": error + " Use bash or one valid recovery tool.",
                       "extra": {"interrupt_type": "FormatError"}})


class AgentRewindBaseline:
    """Prefix-scoped controller owned by one run, with explicit active ancestry.

    register_checkpoint is a trusted runner hook, never a worker tool. Checkpoints
    from this run are eligible only after registration. A successful rewind trims
    the candidate ancestry, retaining the notebook but excluding abandoned sibling
    suffixes. Natural source checkpoints are frozen through S at construction.
    """

    def __init__(self, store: ArtifactStoreProtocol, *, source_checkpoint_id: str,
                 run_id: str) -> None:
        self.store = store
        self.source = store.get_checkpoint(source_checkpoint_id)
        self.run_id = run_id
        if run_id == self.source.run_id:
            raise BaselinePolicyError("Baseline must use a separate recovery run")
        if any(e.kind == "baseline" for e in store.iter_events(run_id)):
            raise BaselinePolicyError("Baseline already has execution records; do not restart uncertain work")
        self._validate_checkpoint(self.source)
        starts = [e for e in store.iter_events(self.source.run_id, through_sequence=self.source.event_sequence)
                  if e.kind == "lifecycle" and e.payload.get("transition") == "run_started"]
        if any(e.payload.get("resume_from") or e.payload.get("treatment") for e in starts):
            raise BaselinePolicyError("Source must be a natural trajectory, not another recovery branch")
        self._active: list[Checkpoint] = []
        for cp in store.list_checkpoints(self.source.run_id):
            if cp.ordinal <= self.source.ordinal and cp.event_sequence <= self.source.event_sequence:
                try:
                    self._validate_checkpoint(cp)
                except (BaselinePolicyError, KeyError, TypeError, ValueError):
                    continue
                self._active.append(cp)
        if not self._active or self._active[-1] != self.source:
            raise BaselinePolicyError("Source must close its eligible natural prefix")
        self._registered: set[str] = set()
        self._last_registered_ordinal = -1
        self._current_cutoff = 0
        self._last_decision_sequence = 0
        self._preview_starts: dict[str, tuple[str, int]] = {}
        self._generation = 0
        self._offered: dict[str, Checkpoint] = {}
        self._pending: RewindPlan | None = None
        self._memory: list[dict[str, Any]] = []
        self._failed = False
        self._persist("initialized", {"source_checkpoint_id": source_checkpoint_id, "policy": POLICY,
                      "source_workspace_timestamp_fidelity": self._timestamp_fidelity(self.source)})

    @property
    def tools(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(TOOL_DEFINITIONS))

    @property
    def memory(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._memory))

    def _persist(self, transition: str, payload: Mapping[str, Any]) -> EventRecord:
        event = EventRecord(run_id=self.run_id, kind="baseline",
            payload={"policy_id": POLICY["policy_id"], "transition": transition, **payload})
        return persist(lambda: self.store.append_event(event), evidence=event)

    def _timestamp_fidelity(self, checkpoint: Checkpoint) -> str:
        expected = self.store.get_json(from_jsonable(ArtifactRef, checkpoint.validation["workspace_manifest_ref"]))
        return "legacy_seconds" if expected and all("mtime_ns" not in e for e in expected.values()) else "nanoseconds"

    def _validate_checkpoint(self, cp: Checkpoint) -> dict[str, Any]:
        if self.store.get_checkpoint(cp.checkpoint_id) != cp:
            raise BaselinePolicyError("Checkpoint differs from immutable store record")
        if not cp.branchable or cp.state.runtime_class == RuntimeClass.UNSUPPORTED:
            raise BaselinePolicyError("Checkpoint is not branchable")
        source = self.source.scope
        if any(getattr(cp.scope, key) != getattr(source, key) for key in
               ("task_id", "stage_id", "target_id", "requirement_version", "evaluator_version")):
            raise BaselinePolicyError("Checkpoint changes the frozen target or versions")
        history = self.store.get_json(cp.state.history)
        messages = history["messages"]
        cursor = history["cursor"]
        if (history_replay_issue(messages) or cursor.get("pending_actions")
                or cursor.get("message_count") != len(messages)
                or cursor != dict(cp.state.execution_cursor)):
            raise BaselinePolicyError("Checkpoint history/cursor cannot be restored coherently")
        if not cp.validation.get("workspace_archive_validated_against_live_manifest"):
            raise BaselinePolicyError("Checkpoint lacks validated workspace capture")
        manifest = from_jsonable(ArtifactRef, cp.validation["workspace_manifest_ref"])
        expected = self.store.get_json(manifest)
        # Verify blob integrity here, before any destructive environment mutation.
        legacy_seconds = self._timestamp_fidelity(cp) == "legacy_seconds"
        if comparable(fidelity_manifest_from_tar(self.store.get_bytes(cp.state.workspace),
                                                 legacy_seconds=legacy_seconds)) != comparable(expected):
            raise BaselinePolicyError("Workspace archive differs from declared fidelity manifest")
        if cp.state.runtime_class == RuntimeClass.CAPTURED:
            if cp.state.runtime is None:
                raise BaselinePolicyError("Captured runtime lacks a snapshot")
            runtime = self.store.get_json(cp.state.runtime)
            if runtime.get("runtime_class") != RuntimeClass.CAPTURED.value:
                raise BaselinePolicyError("Runtime manifest contradicts checkpoint class")
            if runtime.get("archive_ref"):
                self.store.get_bytes(from_jsonable(ArtifactRef, runtime["archive_ref"]))
        return history

    def register_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Register a newly captured boundary on this controller's current branch."""
        if self._failed:
            raise RuntimeError("Failed rewind invalidated this baseline continuation")
        self._validate_checkpoint(checkpoint)
        if checkpoint.run_id != self.run_id:
            raise BaselinePolicyError("Cannot register a natural, foreign or sibling run checkpoint")
        if checkpoint.checkpoint_id in self._registered:
            if self._active[-1].checkpoint_id == checkpoint.checkpoint_id:
                return
            raise BaselinePolicyError("Cannot reactivate a discarded checkpoint")
        if (checkpoint.ordinal <= self._last_registered_ordinal
                or checkpoint.event_sequence <= self._current_cutoff):
            raise BaselinePolicyError("Checkpoint registration must advance the execution boundary")
        if self._pending is not None:
            raise BaselinePolicyError("Apply or abort the pending rewind before registering a boundary")
        self._active.append(checkpoint)
        self._registered.add(checkpoint.checkpoint_id)
        self._last_registered_ordinal = checkpoint.ordinal
        self._current_cutoff = checkpoint.event_sequence
        self._persist("checkpoint_registered", {"checkpoint_id": checkpoint.checkpoint_id,
                      "active_parent": self._active[-2].checkpoint_id, "generation": self._generation})

    def _worker_call(self, sequence: int, name: str) -> tuple[EventRecord, dict[str, Any]]:
        if self._failed:
            raise RuntimeError("Failed rewind invalidated this baseline continuation")
        event = next((e for e in self.store.iter_events(self.run_id, through_sequence=sequence)
                      if e.sequence == sequence), None)
        if (event is None or event.kind != "model_response" or event.actor != "worker"
                or event.visibility != Visibility.WORKER_VISIBLE or event.payload.get("accepted") is not True
                or event.output_ref is None or sequence <= max(self._current_cutoff, self._last_decision_sequence)):
            raise BaselinePolicyError("Recovery decision must be a new accepted worker model response")
        output = self.store.get_json(event.output_ref)
        calls = output.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("name") != name:
            raise BaselinePolicyError("Recovery tools must be the response's only tool call")
        try:
            arguments = calls[0]["arguments"]
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (ValueError, TypeError) as exc:
            raise BaselinePolicyError("Invalid recovery arguments") from exc
        if not isinstance(arguments, dict):
            raise BaselinePolicyError("Recovery arguments must be an object")
        return event, arguments

    def candidates(self, *, response_event_sequence: int, reason: str = "") -> dict[str, Any]:
        """Worker tool handler; previews never read grades, raw output or later runs."""
        _, args = self._worker_call(response_event_sequence, "backtrack_candidates")
        if args != {"reason": reason} or not isinstance(reason, str):
            raise BaselinePolicyError("Candidate arguments differ from worker response")
        if self._pending is not None:
            raise BaselinePolicyError("A rewind is pending")
        self._offered = {cp.checkpoint_id: cp for cp in self._active[:-1][-80:]}
        rendered = [self._candidate_view(cp, response_event_sequence) for cp in self._offered.values()]
        self._persist("candidates_offered", {"response_event_sequence": response_event_sequence,
            "candidate_ids": list(self._offered), "generation": self._generation,
            "natural_cutoff": self.source.event_sequence, "branch_cutoff": response_event_sequence,
            "rendered_ref": to_jsonable(self.store.put_json(rendered, visibility=Visibility.WORKER_VISIBLE))})
        self._last_decision_sequence = response_event_sequence
        return {"candidates": rendered, "selection_policy": "agent-direct",
                "scope": "fixed target; active prefix only"}

    def _candidate_view(self, cp: Checkpoint, decision_cutoff: int) -> dict[str, Any]:
        preview_run, lower = self._preview_starts.get(cp.checkpoint_id, (cp.run_id, cp.event_sequence))
        cutoff = self.source.event_sequence if preview_run == self.source.run_id else decision_cutoff
        # A candidate preview covers only its next accepted decision, bounded
        # by the next active checkpoint. Discarded same-run suffixes never enter.
        index = self._active.index(cp)
        successor = self._active[index + 1]
        if successor.run_id == preview_run:
            cutoff = min(cutoff, successor.event_sequence)
        events = [e for e in self.store.iter_events(preview_run, through_sequence=cutoff) if e.sequence > lower]
        # Share the recorder/observer's native delivery proof, without reading T.
        # Restrict legacy membership snapshots to active ancestry, so a later
        # sibling checkpoint cannot prove delivery in this candidate segment.
        store = self.store
        active = self._active

        class DeliveryStore:
            def get_json(self, ref):
                return store.get_json(ref)

            def list_checkpoints(self, run_id):
                return [c for c in active if c.run_id == run_id and lower <= c.event_sequence <= cutoff]

        delivered = _delivered_tool_events(DeliveryStore(), preview_run, events, through_sequence=cutoff)
        by_id = {e.event_id: e for e in events}
        response = None
        observations = []
        for event in events:
            if event.sequence <= lower or event.visibility != Visibility.WORKER_VISIBLE:
                continue
            if event.kind == "model_response" and event.payload.get("accepted"):
                if response is not None:
                    break
                response = self.store.get_json(event.output_ref) if event.output_ref else None
            elif response is not None and event.kind in {"observation_delivery", "tool_result"}:
                visible = source_event_from_record(event, self.store,
                    verified_observation_events=delivered, records_by_id=by_id)
                if "delivered_content" in visible.texts:
                    observations.append(visible.texts["delivered_content"][:400])
        response = response or {}
        return {"record_uid": cp.checkpoint_id, "record_kind": "validated_checkpoint",
                "decision_hint": str(response.get("content") or "")[:400],
                "tool_calls_preview": [{"name": call.get("name"),
                    "arguments": str(call.get("arguments") or "")[:400]}
                    for call in response.get("tool_calls") or []],
                "tool_records_preview": observations, "source_mode": "deterministic_delivered_prefix"}

    def prepare_commit(self, record_uid: str, memory_summary: str, *,
                       response_event_sequence: int, reason: str = "") -> RewindPlan:
        """Validate a worker commit against the most recently exposed candidates."""
        _, args = self._worker_call(response_event_sequence, "backtrack_commit")
        expected = {"record_uid": record_uid, "memory_summary": memory_summary}
        if "reason" in args or reason:
            expected["reason"] = reason
        if args != expected or not isinstance(memory_summary, str) or not memory_summary.strip():
            raise BaselinePolicyError("Commit arguments must match the worker's nonempty memory decision")
        if not isinstance(reason, str):
            raise BaselinePolicyError("Reason must be text")
        if self._pending is not None:
            raise BaselinePolicyError("A rewind is already pending")
        if record_uid not in self._offered or record_uid not in {c.checkpoint_id for c in self._active[:-1]}:
            raise BaselinePolicyError("Target was not offered as an earlier active-prefix checkpoint")
        donor = self._offered[record_uid]
        self._validate_checkpoint(donor)
        plan = RewindPlan(new_id("rewind"), donor, response_event_sequence, memory_summary,
                          reason, self._generation, len(self._memory) + 1)
        self._pending = plan
        self._last_decision_sequence = response_event_sequence
        self._persist("rewind_prepared", {**to_jsonable(plan),
                      "donor_workspace_timestamp_fidelity": self._timestamp_fidelity(donor)})
        return plan

    def apply(self, plan: RewindPlan, agent: Any,
              restore: Callable[[Checkpoint], Mapping[str, Any]]) -> Mapping[str, Any]:
        """Restore using the engine callback, then adopt H without refunding spend.

        Callback returns {workspace: {validated: True}, runtime: {validated: True},
        workspace_donor: id, runtime_donor: id}; it must restore into a fresh
        supported runtime (or otherwise verify deletion of abandoned R changes).
        The runner must treat any exception here as a technical failure; continuing
        on a partly restored environment is invalid. Costs belong to its normal
        timed restoration instrumentation. This method never issues shell calls.
        """
        if self._failed:
            raise RuntimeError("Failed rewind invalidated this baseline continuation")
        if self._pending != plan or plan.generation != self._generation:
            raise BaselinePolicyError("Unknown, replayed or stale rewind plan")
        if getattr(agent.recorder, "run_id", None) != self.run_id:
            raise BaselinePolicyError("Agent is outside this baseline run")
        history = self._validate_checkpoint(plan.donor)
        counters = dict(agent.counters())
        offset, allowance = agent.call_offset, agent.generated_token_limit
        generated_offset, unknown_offset = agent.generated_token_offset, agent.unknown_usage_offset
        original_messages = copy.deepcopy(agent.messages)
        if history_replay_issue(history["messages"]):
            raise BaselinePolicyError("Selected history is not a complete decision boundary")
        item = {"ordinal": plan.memory_ordinal, "record_uid": plan.donor.checkpoint_id,
                "response_event_sequence": plan.response_event_sequence,
                "memory_summary": plan.memory_summary, "reason": plan.reason,
                "source_run_id": self.run_id, "plan_id": plan.plan_id}
        new_memory = [*self._memory, item]
        messages = self._restored_messages(history["messages"], new_memory, original_messages)
        cursor = {"message_count": len(messages), "pending_actions": [], "completed_in_batch": []}
        try:
            receipt = restore(plan.donor)
            if (receipt.get("workspace_donor") != plan.donor.checkpoint_id
                    or receipt.get("runtime_donor") != plan.donor.checkpoint_id
                    or receipt.get("workspace", {}).get("validated") is not True
                    or receipt.get("runtime", {}).get("validated") is not True):
                raise RuntimeError("Aligned W/R restoration did not produce validated donor-bound receipts")
            agent.restore(messages, counters, cursor, task=history["task"])
            agent.call_offset = offset
            agent.generated_token_offset, agent.unknown_usage_offset = generated_offset, unknown_offset
            agent.generated_token_limit = allowance
            agent.pending_actions = []
            agent.completed_in_batch = []
            # The step's control flow and expenditure remain on this run.
            agent.batch_mutated = True
            self._memory = new_memory
            self._active = self._active[:self._active.index(plan.donor) + 1]
            self._generation += 1
            self._offered = {}
            self._pending = None
            history_ref = persist(lambda: self.store.put_json(messages, visibility=Visibility.WORKER_VISIBLE),
                evidence={"plan_id": plan.plan_id, "restoration_receipt": dict(receipt),
                          "delivered_history": messages, "counters_after": agent.counters()})
            event = self._persist("rewind_applied", {"plan_id": plan.plan_id,
                "donor_checkpoint_id": plan.donor.checkpoint_id, "receipt": dict(receipt),
                "donor_workspace_timestamp_fidelity": self._timestamp_fidelity(plan.donor),
                "generation": self._generation, "memory": self._memory,
                "delivered_history_ref": to_jsonable(history_ref),
                "counters_before": counters, "counters_after": agent.counters(),
                "additional_allowance_ceiling": allowance, "call_offset": offset,
                "retained_prefix_reexecuted": False})
            self._preview_starts[plan.donor.checkpoint_id] = (self.run_id, event.sequence)
            return {"event_id": event.event_id, "plan_id": plan.plan_id, "generation": self._generation}
        except BaseException as exc:
            self._failed = True
            try:
                self._persist("rewind_failed", {"plan_id": plan.plan_id, "error": f"{type(exc).__name__}: {exc}",
                                                "continuation_valid": False})
            except BaseException as cleanup:
                exc.add_note(f"Rewind failure receipt also failed: {type(cleanup).__name__}")
            raise

    @staticmethod
    def _restored_messages(history: list[dict], memory: list[dict], current: list[dict]) -> list[dict]:
        # Remove only harness-marked notebook/policy messages, never text that
        # merely resembles the prefix. Provider messages retain exact bytes.
        messages = copy.deepcopy([m for m in history if not
            (m.get("role") == "system" and any(m.get("extra", {}).get(k) for k in (_POLICY_MARKER, _MEMORY_MARKER)))])
        messages.append({"role": "system", "content": SYSTEM_PROMPT,
                         "extra": {_POLICY_MARKER: POLICY["policy_id"]}})
        # Fixed target disclosure is common runner-owned context. Copy only its
        # specifically marked message, never arbitrary later user/grade material.
        target = [m for m in current if m.get("role") in {"user", "system"}
                  and m.get("extra", {}).get("agentrewind_fixed_target")]
        for m in target:
            if m not in messages:
                messages.append(copy.deepcopy(m))
        notebook = "Backtrack memory notebook from abandoned future branches:\n" + json.dumps(memory, ensure_ascii=False)
        messages.append({"role": "system", "content": notebook,
                         "extra": {_MEMORY_MARKER: POLICY["policy_id"]}})
        return messages
