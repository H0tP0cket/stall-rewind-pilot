"""Bounded preservation retry for immutable, already prepared local writes.

Never wrap an external call, a state capture, or an operation that allocates a
new record ID in this helper. A journal retry must reuse the original record.
When storage stays unavailable, the original exception retains the in-memory
evidence for its caller. This cannot make evidence durable after process loss
when all storage fails; a durable start then remains honestly uncertain.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Callable, TypeVar

T = TypeVar("T")
PERSISTENCE_ATTEMPTS = 3


def preserve_failure(error: BaseException, evidence: Any) -> None:
    error.persistence_evidence = evidence
    error.add_note("Local persistence incomplete; received evidence is attached in memory. "
                   "Do not replay the external operation. All-storage failure cannot guarantee durability.")


def persist(operation: Callable[[], T], *, evidence: Any) -> T:
    first = None
    for _ in range(PERSISTENCE_ATTEMPTS):
        try:
            return operation()
        except (OSError, sqlite3.OperationalError) as error:
            if first is None:
                first = error
    preserve_failure(first, evidence)
    raise first
