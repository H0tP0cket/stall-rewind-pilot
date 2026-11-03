"""Canonical JSON and content hashing for observer artifacts.

Observer artifacts are immutable; their identity is the SHA-256 of their
canonical JSON bytes. The shared store may re-wrap these payloads, but must
not change the bytes that were hashed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_id(value: Any) -> str:
    return "sha256:" + sha256_hex(canonical_bytes(value))
