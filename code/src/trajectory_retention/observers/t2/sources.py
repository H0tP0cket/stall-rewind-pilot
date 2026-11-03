"""Load one recorded run into unlabeled T2 items plus deterministic structure.

Eligibility reuses ``observers.prefix`` (worker visibility, delivery proof,
forbidden kinds). Two additions:

* Returned reasoning is eligible only after it was actually replayed to the
  worker in a later recorded request, and only from that request's sequence
  (``available_seq``). Its source kind is ``provider_summary`` when the replay
  carries provider summary items and ``worker_reasoning`` for plaintext text.
* Tool-call identities come from the recorded response body so observations
  bind structurally to the action that produced them.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..prefix import source_events_from_store
from .schema import Item, Link
from .segment import TaskText, candidates_for_event, decode, tags_for, tool_call_index


@dataclass
class CheckpointInfo:
    checkpoint_id: str
    ordinal: int
    event_sequence: int
    session_id: str | None
    stage_id: str | None
    annotations: tuple[str, ...]


@dataclass
class RunSource:
    run_id: str
    model: str | None
    items: list[Item]
    links: list[Link]
    checkpoints: list[CheckpointInfo]
    task_text_by_session: dict[str, str]
    worker_cost_usd: float
    old_observer_cost_usd: float
    unreplayed_reasoning_events: list[int] = field(default_factory=list)
    decoded_fields: dict[tuple[int, str], str] = field(default_factory=dict)


def _replay_index(store, run_id: str) -> dict[str, tuple[int, str]]:
    """Map sha256(reasoning text) -> (first replay request seq, kind)."""
    index: dict[str, tuple[int, str]] = {}
    for event in store.iter_events(run_id):
        if event.kind != "model_request" or event.input_ref is None:
            continue
        body = store.get_json(event.input_ref)
        for message in body.get("messages") or ():
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            text = message.get("reasoning")
            if not isinstance(text, str) or not text:
                continue
            types = {d.get("type") for d in message.get("reasoning_details") or () if isinstance(d, dict)}
            kind = "provider_summary" if "reasoning.summary" in types and "reasoning.text" not in types else "worker_reasoning"
            key = hashlib.sha256(text.encode()).hexdigest()
            if key not in index:
                index[key] = (event.sequence, kind)
    return index


def _tool_call_ids(store, run_id: str) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for event in store.iter_events(run_id):
        if event.kind == "model_response" and event.output_ref is not None:
            body = store.get_json(event.output_ref)
            for i, call in enumerate(body.get("tool_calls") or ()):
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    out[(event.sequence, i)] = call["id"]
    return out


def load_run(store, run_id: str) -> RunSource:
    last = max((e.sequence for e in store.iter_events(run_id)), default=0)
    events = source_events_from_store(store, run_id, through_sequence=last)
    replay = _replay_index(store, run_id)
    call_ids = _tool_call_ids(store, run_id)

    task_texts: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if ev.kind == "task_disclosure" and "delivered_text" in ev.texts and ev.worker_visible:
            task_texts[ev.session_id or ""].append(ev.texts["delivered_text"])
    detectors = {sid: TaskText(texts) for sid, texts in task_texts.items()}

    items: list[Item] = []
    links: list[Link] = []
    unreplayed: list[int] = []
    action_by_call: dict[str, Item] = {}
    decoded_fields: dict[tuple[int, str], str] = {}
    for ev in events:
        if not ev.worker_visible or not ev.texts:
            continue
        reasoning_kind, available = None, ev.seq
        reasoning = ev.texts.get("returned_reasoning_text")
        if reasoning:
            hit = replay.get(hashlib.sha256(reasoning.encode()).hexdigest())
            if hit is None:
                unreplayed.append(ev.seq)
            else:
                available, reasoning_kind = hit
        turn_prose: list[Item] = []
        turn_actions: list[Item] = []
        for cand in candidates_for_event(ev, reasoning_kind=reasoning_kind):
            if (ev.seq, cand.field) not in decoded_fields:
                decoded_fields[(ev.seq, cand.field)] = decode(ev.texts[cand.field], cand.decoder)
            idx = tool_call_index(cand.field)
            tool_id = call_ids.get((ev.seq, idx)) if idx is not None else cand.tool_call_id
            item = Item(
                item_id=f"{run_id}:i{ev.seq}.{cand.order}", run_id=run_id, session_id=ev.session_id,
                stage_id=ev.stage_id, seq=ev.seq, order=cand.order,
                available_seq=available if cand.field == "returned_reasoning_text" else ev.seq,
                source_kind=cand.source_kind, text=cand.text, span=cand.span, tool_call_id=tool_id,
                call_group=ev.tool_call_id if ev.kind == "model_response" else None,
                task_copy=bool(detectors.get(ev.session_id or "") and detectors[ev.session_id or ""].is_copy(cand.text)),
                tags=tags_for(cand.text),
            )
            items.append(item)
            if item.source_kind == "worker_action" and tool_id:
                action_by_call.setdefault(tool_id, item)
                turn_actions.append(item)
            elif item.source_kind in {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}:
                turn_prose.append(item)
        for prose in turn_prose:
            for action in turn_actions:
                links.append(Link(prose.item_id, action.item_id, "same_turn", "structural", None,
                                  max(prose.available_seq, action.available_seq), "deterministic:same_response"))
    for item in items:
        if item.source_kind == "observation" and item.tool_call_id in action_by_call:
            action = action_by_call[item.tool_call_id]
            links.append(Link(item.item_id, action.item_id, "result_of", "structural", None, item.seq,
                              "deterministic:tool_call_id"))

    annotations: dict[str, list[str]] = defaultdict(list)
    for _, record in store.records("annotation", run_id=run_id):
        if record.get("checkpoint_id"):
            annotations[record["checkpoint_id"]].append(record["kind"])
    checkpoints = [CheckpointInfo(cp.checkpoint_id, cp.ordinal, cp.event_sequence, cp.scope.session_id,
                                  cp.scope.stage_id, tuple(sorted(annotations.get(cp.checkpoint_id, ()))))
                   for cp in store.list_checkpoints(run_id)]

    worker_cost = observer_cost = 0.0
    for _, record in store.records("cost", run_id=run_id):
        usd = ((record.get("usage") or {}).get("cost_usd")) or 0.0
        if record.get("role") == "worker":
            worker_cost += usd
        elif record.get("role") in {"extraction", "audit"}:
            observer_cost += usd
    model = None
    for _, campaign in store.records("campaign"):
        if campaign.get("mining", {}).get("run_id") == run_id:
            model = campaign["mining"].get("model")
    return RunSource(run_id, model, items, links, checkpoints,
                     {k: "\n\n".join(v) for k, v in task_texts.items()},
                     worker_cost, observer_cost, unreplayed, decoded_fields)


def items_at(source: RunSource, checkpoint: CheckpointInfo, *, session_scoped: bool = True) -> list[Item]:
    """Items that are allowed sources at the checkpoint's event cut."""
    return [i for i in source.items
            if i.available_seq <= checkpoint.event_sequence
            and (not session_scoped or i.session_id == checkpoint.session_id)]


def summarize(source: RunSource) -> dict[str, Any]:
    by_kind: dict[str, int] = defaultdict(int)
    for item in source.items:
        by_kind[item.source_kind] += 1
    return {"run_id": source.run_id, "model": source.model, "items": len(source.items), "by_kind": dict(by_kind),
            "task_copies": sum(i.task_copy for i in source.items), "links": len(source.links),
            "checkpoints": len(source.checkpoints), "unreplayed_reasoning_events": source.unreplayed_reasoning_events}
