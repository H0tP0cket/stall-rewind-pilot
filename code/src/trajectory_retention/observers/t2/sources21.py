"""Load a recorded run into the T2.1 candidate ledger (read-only source store).

Eligibility, replay verification and tool-call binding are shared with T2
(``sources.py``). Segmentation uses ``segment21``; discarded spans are kept in
the ledger with reasons. Structural links are the same mechanical relations:
an observation is ``result_of`` the call that produced it, and prose and
actions from one response are ``same_turn``.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..prefix import source_events_from_store
from .schema import Item, Link, Span
from .segment import TaskText, decode, tags_for, tool_call_index
from .segment21 import Discard, segment_event
from .sources import CheckpointInfo, _replay_index, _tool_call_ids

WORKER_PROSE = {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}


@dataclass
class Item21(Item):
    truncated_for_display: bool = False

    def as_record(self) -> dict[str, Any]:
        record = super().as_record()
        record["truncated_for_display"] = self.truncated_for_display
        return record


@dataclass
class RunLedger:
    run_id: str
    model: str | None
    items: list[Item21]
    links: list[Link]
    discards: list[Discard]
    checkpoints: list[CheckpointInfo]
    task_text_by_session: dict[str, str]
    decoded_fields: dict[tuple[int, str], str]
    worker_cost_usd: float
    old_observer_cost_usd: float
    unreplayed_reasoning_events: list[int] = field(default_factory=list)


def load_ledger(store, run_id: str) -> RunLedger:
    last = max((e.sequence for e in store.iter_events(run_id)), default=0)
    events = source_events_from_store(store, run_id, through_sequence=last)
    replay = _replay_index(store, run_id)
    call_ids = _tool_call_ids(store, run_id)
    task_texts: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if ev.kind == "task_disclosure" and "delivered_text" in ev.texts and ev.worker_visible:
            task_texts[ev.session_id or ""].append(ev.texts["delivered_text"])
    detectors = {sid: TaskText(texts) for sid, texts in task_texts.items()}

    items: list[Item21] = []
    links: list[Link] = []
    discards: list[Discard] = []
    unreplayed: list[int] = []
    decoded: dict[tuple[int, str], str] = {}
    action_by_call: dict[str, Item21] = {}
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
        seg = segment_event(ev, reasoning_kind=reasoning_kind)
        discards.extend(seg.discards)
        prose, actions = [], []
        for cand in seg.candidates:
            if (ev.seq, cand.field) not in decoded:
                decoded[(ev.seq, cand.field)] = decode(ev.texts[cand.field], cand.decoder)
            idx = tool_call_index(cand.field)
            tool_id = call_ids.get((ev.seq, idx)) if idx is not None else ev.tool_call_id if ev.kind != "model_response" else None
            detector = detectors.get(ev.session_id or "")
            item = Item21(
                item_id=f"{run_id}:j{ev.seq}.{cand.order}", run_id=run_id, session_id=ev.session_id, stage_id=ev.stage_id,
                seq=ev.seq, order=cand.order,
                available_seq=available if cand.field == "returned_reasoning_text" else ev.seq,
                source_kind=cand.source_kind, text=cand.text,
                span=Span(ev.seq, cand.field, cand.decoder, cand.start, cand.end),
                tool_call_id=tool_id, call_group=ev.tool_call_id if ev.kind == "model_response" else None,
                task_copy=bool(detector and cand.source_kind in WORKER_PROSE and detector.is_copy(cand.text)),
                tags=tags_for(cand.text), truncated_for_display=cand.truncated_for_display,
            )
            items.append(item)
            if item.source_kind == "worker_action" and tool_id:
                action_by_call.setdefault(tool_id, item)
                actions.append(item)
            elif item.source_kind in WORKER_PROSE:
                prose.append(item)
        for p in prose:
            for a in actions:
                links.append(Link(p.item_id, a.item_id, "same_turn", "structural", None,
                                  max(p.available_seq, a.available_seq), "deterministic:same_response"))
    for item in items:
        if item.source_kind == "observation" and item.tool_call_id in action_by_call:
            links.append(Link(item.item_id, action_by_call[item.tool_call_id].item_id, "result_of", "structural", None,
                              item.seq, "deterministic:tool_call_id"))

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
    return RunLedger(run_id, model, items, links, discards, checkpoints,
                     {k: "\n\n".join(v) for k, v in task_texts.items()}, decoded, worker_cost, observer_cost, unreplayed)
