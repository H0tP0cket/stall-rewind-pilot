"""Cost ledger for runs that checkpoint at every point: same results as CostLedger, without re-reading the journal.

CostLedger.records reads every cost record and every event of the run at each checkpoint, so a run that checkpoints at
every point pays a cost that grows with the square of its length (measured 0.2 s per snapshot at step 5 and 0.56 s at
step 120). Journaled events and records are immutable, so this ledger keeps the parsed ones of each run in memory and
reads only those added since its last call. The unchanged CostLedger.records logic runs on top of the cached reads, so
snapshots stay identical.
"""
from __future__ import annotations

from typing import Any, Iterator

from ..accounting import CostLedger
from ..contracts import EventRecord, from_jsonable


class _CachedReads:
    """The store, with run event and cost-record reads served from immutable caches."""

    def __init__(self, store: Any, events: dict, costs: dict) -> None:
        self._store, self._events, self._costs = store, events, costs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def iter_events(self, run_id: str, *, through_sequence: int | None = None) -> Iterator[EventRecord]:
        cached = self._events.setdefault(run_id, [])
        after = cached[-1].sequence if cached else 0
        with self._store._connect() as db:
            rows = db.execute("SELECT ref_json FROM events WHERE run_id=? AND sequence>? ORDER BY sequence",
                              (run_id, after)).fetchall()
        for row in rows:
            cached.append(from_jsonable(EventRecord, self._store.get_json(self._store._decode_ref(row[0]))))
        for event in cached:
            if through_sequence is not None and event.sequence > through_sequence:
                break
            yield event

    def get_record(self, kind: str, record_id: str) -> Any:
        if kind != "cost":
            return self._store.get_record(kind, record_id)
        if record_id not in self._costs:
            self._costs[record_id] = self._store.get_record(kind, record_id)
        return self._costs[record_id]


class IncrementalCostLedger(CostLedger):
    def __init__(self, store: Any) -> None:
        super().__init__(store)
        self._events: dict[str, list[EventRecord]] = {}
        self._costs: dict[str, Any] = {}

    def records(self, run_id: str, *, through_sequence: int | None = None):
        if not (hasattr(self.store, "_connect") and hasattr(self.store, "_decode_ref")):
            return super().records(run_id, through_sequence=through_sequence)
        real = self.store
        self.store = _CachedReads(real, self._events, self._costs)
        try:
            return super().records(run_id, through_sequence=through_sequence)
        finally:
            self.store = real
