"""Stores for v2 branches: a read-only view of the source store, and a branch's own store that reads through to it.

A branch writes only into its own store (one per branch attempt), so branches stay isolated from one another and the
natural run's store is never written. Reads of anything the branch store does not hold (the source run's checkpoints,
events, records and blobs) fall through to the source store, opened read-only. Everything the branch records (events,
checkpoints, grades, costs, the choice record) is in its own store; references in its events to source evidence
(donor checkpoint ids, the donor history blob) resolve in the source store, whose path the branch records.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..contracts import ArtifactRef, Checkpoint, EventRecord
from ..store import ArtifactStore


class ReadOnlyArtifactStore(ArtifactStore):
    """An existing store opened without initializing or changing it; every write raises."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.blobs = self.root / "blobs"
        self.database = self.root / "journal.sqlite3"
        if not self.database.is_file() or not self.blobs.is_dir():
            raise FileNotFoundError(f"no artifact store at {self.root}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            yield db
        finally:
            db.close()

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("the source store is read-only for branches")

    put_bytes = put_json = put_directory = append_event = put_record = save_checkpoint = save_annotation = _refuse


class BranchStore(ArtifactStore):
    """The branch's own store; reads it does not hold fall through to the read-only source store."""

    def __init__(self, root: str | Path, source: ArtifactStore):
        super().__init__(root)
        self.source = source

    def _owns_run(self, run_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM events WHERE run_id=? LIMIT 1", (run_id,)).fetchone() is not None

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        if self._blob_path(ref.sha256).exists():
            return super().get_bytes(ref)
        return self.source.get_bytes(ref)

    def iter_events(self, run_id: str, *, through_sequence: int | None = None) -> Iterator[EventRecord]:
        owner = super() if self._owns_run(run_id) else self.source
        return owner.iter_events(run_id, through_sequence=through_sequence)

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        try:
            return super().get_checkpoint(checkpoint_id)
        except KeyError:
            return self.source.get_checkpoint(checkpoint_id)

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        own = super().list_checkpoints(run_id)
        return own if own or self._owns_run(run_id) else self.source.list_checkpoints(run_id)

    def get_record(self, kind: str, record_id: str) -> Any:
        try:
            return super().get_record(kind, record_id)
        except KeyError:
            return self.source.get_record(kind, record_id)

    def list_records(self, kind: str, *, run_id: str | None = None) -> list[dict[str, Any]]:
        own = super().list_records(kind, run_id=run_id)
        if run_id is not None and self._owns_run(run_id):
            return own
        seen = {r["record_id"] for r in own}
        return own + [r for r in self.source.list_records(kind, run_id=run_id) if r["record_id"] not in seen]


class ReadOnlyBranchStore(ReadOnlyArtifactStore):
    """A finished branch or continuation store opened read-only, falling through to its own source like BranchStore
    (used to resume an interrupted technical continuation from its own points)."""

    def __init__(self, root: str | Path, source: ArtifactStore):
        super().__init__(root)
        self.source = source

    def _owns_run(self, run_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM events WHERE run_id=? LIMIT 1", (run_id,)).fetchone() is not None

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        if self._blob_path(ref.sha256).exists():
            return ArtifactStore.get_bytes(self, ref)
        return self.source.get_bytes(ref)

    def iter_events(self, run_id: str, *, through_sequence: int | None = None) -> Iterator[EventRecord]:
        if self._owns_run(run_id):
            return ArtifactStore.iter_events(self, run_id, through_sequence=through_sequence)
        return self.source.iter_events(run_id, through_sequence=through_sequence)

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        try:
            return ArtifactStore.get_checkpoint(self, checkpoint_id)
        except KeyError:
            return self.source.get_checkpoint(checkpoint_id)

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        own = ArtifactStore.list_checkpoints(self, run_id)
        return own if own or self._owns_run(run_id) else self.source.list_checkpoints(run_id)

    def get_record(self, kind: str, record_id: str) -> Any:
        try:
            return ArtifactStore.get_record(self, kind, record_id)
        except KeyError:
            return self.source.get_record(kind, record_id)

    def list_records(self, kind: str, *, run_id: str | None = None) -> list[dict[str, Any]]:
        own = ArtifactStore.list_records(self, kind, run_id=run_id)
        if run_id is not None and self._owns_run(run_id):
            return own
        seen = {r["record_id"] for r in own}
        return own + [r for r in self.source.list_records(kind, run_id=run_id) if r["record_id"] not in seen]
