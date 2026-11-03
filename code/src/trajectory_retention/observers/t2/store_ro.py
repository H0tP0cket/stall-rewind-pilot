"""Read-only access to a recorded experiment store.

T2 re-extraction reads historical mining stores and must never modify them.
``ArtifactStore`` creates tables, changes journal mode and chmods its database
on open, so it is not used here. The journal is copied together with its
``-wal``/``-shm`` files into a private temporary directory and the copy is
opened read-only, so committed events still held in the WAL are included
(``immutable=1`` on the original silently skipped them) and the source is never
touched, even while a live batch is writing it. Blobs are content-addressed and
read in place, hash-verified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from ...contracts import ArtifactRef, Checkpoint, EventRecord, from_jsonable


class ReadOnlyStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.database = self.root / "journal.sqlite3"
        if not self.database.is_file():
            raise FileNotFoundError(self.database)
        self._copy_dir = Path(tempfile.mkdtemp(prefix="t2-ro-"))
        for suffix in ("", "-wal", "-shm"):
            src = self.database.with_name(self.database.name + suffix)
            try:  # a live writer can checkpoint the WAL away between listing and copying
                shutil.copy2(src, self._copy_dir / src.name)
            except FileNotFoundError:
                if not suffix:
                    raise
        self._db = sqlite3.connect(f"file:{self._copy_dir / self.database.name}?mode=ro", uri=True,
                                   check_same_thread=False)

    def close(self) -> None:
        self._db.close()
        shutil.rmtree(self._copy_dir, ignore_errors=True)

    def _blob(self, digest: str) -> bytes:
        return (self.root / "blobs" / digest[:2] / digest[2:]).read_bytes()

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        data = self._blob(ref.sha256)
        if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError(f"Corrupt artifact {ref.sha256}")
        return data

    def get_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.get_bytes(ref))

    def _ref(self, raw: str) -> ArtifactRef:
        return from_jsonable(ArtifactRef, json.loads(raw))

    def run_ids(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id")]

    def iter_events(self, run_id: str, *, through_sequence: int | None = None) -> Iterator[EventRecord]:
        query, args = "SELECT ref_json FROM events WHERE run_id=?", [run_id]
        if through_sequence is not None:
            query += " AND sequence<=?"
            args.append(through_sequence)
        for (raw,) in self._db.execute(query + " ORDER BY sequence", args).fetchall():
            yield from_jsonable(EventRecord, self.get_json(self._ref(raw)))

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        rows = self._db.execute("SELECT ref_json FROM checkpoints WHERE run_id=? ORDER BY ordinal", (run_id,)).fetchall()
        return [from_jsonable(Checkpoint, self.get_json(self._ref(r[0]))) for r in rows]

    def list_records(self, kind: str, *, run_id: str | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT record_id, ref_json FROM records WHERE kind=?", [kind]
        if run_id is not None:
            query += " AND run_id=?"
            args.append(run_id)
        rows = self._db.execute(query + " ORDER BY record_id", args).fetchall()
        return [{"record_id": r[0], "ref": json.loads(r[1])} for r in rows]

    def get_record(self, kind: str, record_id: str) -> Any:
        row = self._db.execute("SELECT ref_json FROM records WHERE kind=? AND record_id=?", (kind, record_id)).fetchone()
        if not row:
            raise KeyError((kind, record_id))
        return self.get_json(self._ref(row[0]))

    def records(self, kind: str, *, run_id: str | None = None) -> Iterator[tuple[str, Any]]:
        for row in self.list_records(kind, run_id=run_id):
            yield row["record_id"], self.get_json(from_jsonable(ArtifactRef, row["ref"]))
