"""Private content-addressed artifacts and a durable per-run event journal.

Model/tool effects cannot be made exactly-once by this store. It guarantees
immutable local record identities and ordered durable receipts, allowing an
uncertain external attempt to be preserved rather than silently retried.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import Any, Iterator

from .contracts import (Annotation, ArtifactRef, Checkpoint, EventRecord,
                        Visibility, canonical_json, from_jsonable)
from .persistence import persist


class IntegrityError(RuntimeError):
    pass


class RecordConflict(IntegrityError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(exist_ok=True, mode=0o700)
        self.database = self.root / "journal.sqlite3"
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, record_id TEXT NOT NULL,
                    run_id TEXT, ref_json TEXT NOT NULL,
                    PRIMARY KEY(kind, record_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL, ref_json TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, ref_json TEXT NOT NULL,
                    UNIQUE(run_id, ordinal)
                );
            """)
        os.chmod(self.database, 0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _blob_path(self, digest: str) -> Path:
        return self.blobs / digest[:2] / digest[2:]

    def put_bytes(self, data: bytes, *, media_type: str = "application/octet-stream",
                  visibility: Visibility = Visibility.RESEARCH_PRIVATE) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise TypeError("put_bytes requires bytes")
        ref = ArtifactRef(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
                          media_type=media_type, visibility=visibility)
        target = self._blob_path(ref.sha256)
        target.parent.mkdir(exist_ok=True, mode=0o700)
        if not target.exists():
            fd, name = tempfile.mkstemp(prefix=".writing-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(name, target)
                except FileExistsError:
                    pass
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                Path(name).unlink(missing_ok=True)
        self.get_bytes(ref)  # Verify existing blobs too; never bless corruption.
        return ref

    def put_json(self, value: Any, *, visibility: Visibility = Visibility.RESEARCH_PRIVATE) -> ArtifactRef:
        return self.put_bytes(canonical_json(value), media_type="application/json", visibility=visibility)

    def put_directory(self, directory: Path) -> ArtifactRef:
        """Persist private receipts without following links out of their tree."""
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Receipt directory must be a real directory")
        entries = []
        for path in sorted(directory.rglob("*")):
            info = path.lstat()
            row: dict[str, Any] = {"path": path.relative_to(directory).as_posix(),
                                   "mode": stat.S_IMODE(info.st_mode)}
            if stat.S_ISLNK(info.st_mode):
                row.update(type="symlink", target=os.readlink(path))
            elif stat.S_ISREG(info.st_mode):
                # Capture once; retries persist these exact bytes without
                # rereading a completed grader's receipt or rerunning it.
                data = path.read_bytes()
                ref = persist(lambda: self.put_bytes(data), evidence={
                    "private_receipt_directory": str(directory), "entry": dict(row),
                    "bytes": data, "saved_entries": list(entries)})
                row.update(type="file", content_ref=ref)
            elif stat.S_ISDIR(info.st_mode):
                row.update(type="directory")
            else:
                row.update(type="unsupported", stat_mode=info.st_mode)
            entries.append(row)
        tree = {"kind": "private-receipt-tree.v1", "entries": entries}
        return persist(lambda: self.put_json(tree), evidence=tree)

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        data = self._blob_path(ref.sha256).read_bytes()
        if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise IntegrityError(f"Corrupt artifact {ref.sha256}")
        return data

    def get_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.get_bytes(ref))

    @staticmethod
    def _decode_ref(raw: str) -> ArtifactRef:
        return from_jsonable(ArtifactRef, json.loads(raw))

    def append_event(self, event: EventRecord) -> EventRecord:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT ref_json FROM events WHERE event_id=?", (event.event_id,)).fetchone()
            if prior:
                existing = from_jsonable(EventRecord, self.get_json(self._decode_ref(prior[0])))
                if canonical_json(replace(event, sequence=existing.sequence)) != canonical_json(existing):
                    raise RecordConflict(f"Event identity reused with different content: {event.event_id}")
                return existing
            last = db.execute("SELECT COALESCE(MAX(sequence),0) FROM events WHERE run_id=?", (event.run_id,)).fetchone()[0]
            sequence = last + 1
            if event.sequence not in (0, sequence):
                raise RecordConflict("Event sequence must be assigned by the journal")
            saved = replace(event, sequence=sequence)
            ref = self.put_json(saved)
            db.execute("INSERT INTO events VALUES (?,?,?,?)",
                       (saved.event_id, saved.run_id, sequence, canonical_json(ref).decode()))
        return saved

    def iter_events(self, run_id: str, *, through_sequence: int | None = None) -> Iterator[EventRecord]:
        query = "SELECT ref_json FROM events WHERE run_id=?"
        args: list[Any] = [run_id]
        if through_sequence is not None:
            query += " AND sequence<=?"
            args.append(through_sequence)
        query += " ORDER BY sequence"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        for row in rows:
            yield from_jsonable(EventRecord, self.get_json(self._decode_ref(row[0])))

    def put_record(self, kind: str, record_id: str, value: Any, *, run_id: str | None = None) -> ArtifactRef:
        ref = self.put_json(value)
        encoded = canonical_json(ref).decode()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT ref_json, run_id FROM records WHERE kind=? AND record_id=?", (kind, record_id)).fetchone()
            if row and (row[0] != encoded or row[1] != run_id):
                raise RecordConflict(f"Immutable {kind} record changed: {record_id}")
            if not row:
                db.execute("INSERT INTO records VALUES (?,?,?,?)", (kind, record_id, run_id, encoded))
        return ref

    def get_record(self, kind: str, record_id: str) -> Any:
        with self._connect() as db:
            row = db.execute("SELECT ref_json FROM records WHERE kind=? AND record_id=?", (kind, record_id)).fetchone()
        if not row:
            raise KeyError((kind, record_id))
        return self.get_json(self._decode_ref(row[0]))

    def list_records(self, kind: str, *, run_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT record_id, ref_json FROM records WHERE kind=?"
        args = [kind]
        if run_id is not None:
            query += " AND run_id=?"
            args.append(run_id)
        query += " ORDER BY record_id"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [{"record_id": row[0], "ref": json.loads(row[1])} for row in rows]

    def save_checkpoint(self, checkpoint: Checkpoint) -> ArtifactRef:
        for ref in (checkpoint.state.workspace, checkpoint.state.history, checkpoint.state.harness,
                    checkpoint.state.runtime, checkpoint.evaluation_ref, checkpoint.accounting_ref,
                    *checkpoint.state.observations):
            if ref is not None:
                self.get_bytes(ref)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT COALESCE(MAX(sequence),0) FROM events WHERE run_id=?", (checkpoint.run_id,)).fetchone()[0]
            if checkpoint.event_sequence > last:
                raise IntegrityError("Checkpoint references events not durably recorded")
            prior = db.execute("SELECT ref_json FROM checkpoints WHERE checkpoint_id=?", (checkpoint.checkpoint_id,)).fetchone()
            ref = self.put_json(checkpoint)
            encoded = canonical_json(ref).decode()
            if prior and prior[0] != encoded:
                raise RecordConflict(f"Checkpoint identity changed: {checkpoint.checkpoint_id}")
            if not prior:
                try:
                    db.execute("INSERT INTO checkpoints VALUES (?,?,?,?)", (checkpoint.checkpoint_id, checkpoint.run_id, checkpoint.ordinal, encoded))
                except sqlite3.IntegrityError as exc:
                    raise RecordConflict("Checkpoint ordinal already used in this run") from exc
        return ref

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        with self._connect() as db:
            row = db.execute("SELECT ref_json FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)).fetchone()
        if not row:
            raise KeyError(checkpoint_id)
        return from_jsonable(Checkpoint, self.get_json(self._decode_ref(row[0])))

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        with self._connect() as db:
            rows = db.execute("SELECT ref_json FROM checkpoints WHERE run_id=? ORDER BY ordinal", (run_id,)).fetchall()
        return [from_jsonable(Checkpoint, self.get_json(self._decode_ref(row[0]))) for row in rows]

    def save_annotation(self, annotation: Annotation) -> ArtifactRef:
        if annotation.checkpoint_id is not None:
            checkpoint = self.get_checkpoint(annotation.checkpoint_id)
            if checkpoint.run_id != annotation.run_id:
                raise IntegrityError("Annotation must refer to a checkpoint in the same run")
            if annotation.evidence_cutoff < checkpoint.event_sequence:
                raise IntegrityError("Annotation evidence predates its checkpoint")
            if annotation.scope != checkpoint.scope:
                raise IntegrityError("Annotation scope must match its target checkpoint")
        else:
            # Absence is a first-class localization outcome, not an invented
            # checkpoint. Its scope must still join to durable source evidence.
            if annotation.kind != "L" or not annotation.evidence.get("localization_reason"):
                raise IntegrityError("Unbound annotation must explain L localization")
            with self._connect() as db:
                row = db.execute("SELECT ref_json FROM events WHERE run_id=? AND sequence=?",
                    (annotation.run_id, annotation.evidence_cutoff)).fetchone()
            if row is None:
                raise IntegrityError("Unbound annotation lacks durable evidence at its cutoff")
            evidence = from_jsonable(EventRecord, self.get_json(self._decode_ref(row[0])))
            if evidence.scope != annotation.scope:
                raise IntegrityError("Unbound annotation scope must match its evidence cutoff")
        with self._connect() as db:
            last = db.execute("SELECT COALESCE(MAX(sequence),0) FROM events WHERE run_id=?", (annotation.run_id,)).fetchone()[0]
        if annotation.evidence_cutoff > last:
            raise IntegrityError("Annotation refers to evidence not durably recorded")
        return self.put_record("annotation", annotation.annotation_id, annotation, run_id=annotation.run_id)
