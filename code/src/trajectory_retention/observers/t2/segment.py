"""Deterministic segmentation of allowed source events into verbatim candidate items.

Rules here are generic text/format rules. They must never encode knowledge of a
particular task, benchmark or expected solution. Every candidate is an exact
span of a deterministically decoded source field, so provenance holds by
construction and no quote matching is needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from ...ledger.text import (_ABBREV, _FENCE, _HEADING_ONLY, _INLINE_HEADING, _LIST_MARKER, _NORM, _PATH,  # noqa: F401
                            _SENTENCE_END, _TEST_NAME, _TOOL_ARG, _field_order, _strip_span, decode, normalize,
                            tags_for, tool_call_index)
from ..prefix import SourceEvent
from .schema import Span

SEGMENT_VERSION = "t2-segment/v2"

MIN_ITEM_CHARS = 12
MAX_ITEM_CHARS = 600
SHORT_OUTPUT_CHARS = 240
MAX_OBSERVATION_LINES = 12

_CLAUSE_SPLIT = re.compile(
    r"(?:;\s+|,\s+so\s+|\s+so\s+(?=(?:I'll|I will|let me|let's|we'll|we will|I need|I should|I'm going)\b))",
    re.IGNORECASE)
# Generic informative-output signals (errors, test outcomes, explicit failures).
_INFORMATIVE = re.compile(
    r"(error|exception|traceback|fail|failed|failure|passed|assert|warning|not found|no such|denied|refused|"
    r"timed? ?out|invalid|expected|mismatch|differ|cannot|can't|could not|unable|missing|undefined|"
    r"\b\d+ (?:passed|failed|errors?|skipped)\b|^E\s)", re.IGNORECASE)
_COMMENT = re.compile(r"^\s*#\s+(.{20,})$")


@dataclass(frozen=True)
class Candidate:
    seq: int
    order: int
    field: str
    decoder: str
    source_kind: str
    text: str
    start: int
    end: int
    tool_call_id: str | None

    @property
    def span(self) -> Span:
        return Span(self.seq, self.field, self.decoder, self.start, self.end)


def _decoder_for(kind: str, field_name: str, text: str) -> str:
    if _TOOL_ARG.match(field_name):
        try:
            return "json:command" if isinstance(json.loads(text), dict) else "raw"
        except ValueError:
            return "raw"
    if kind == "observation_delivery" and field_name == "delivered_content":
        try:
            value = json.loads(text)
            return "json:returncode+output" if isinstance(value, dict) and "output" in value else "raw"
        except ValueError:
            return "raw"
    return "raw"


def prose_spans(text: str) -> list[tuple[int, int]]:
    """Sentence/clause spans of worker prose, skipping fenced code blocks."""
    spans: list[tuple[int, int]] = []
    pos, in_fence = 0, False
    for line in text.splitlines(keepends=True):
        line_start, line_end = pos, pos + len(line)
        pos = line_end
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip() or _HEADING_ONLY.match(line):
            continue
        cursor = line_start
        pieces = []
        segments = []
        seg_start = line_start
        for m in _INLINE_HEADING.finditer(text, line_start, line_end):
            segments.append((seg_start, m.start()))
            seg_start = m.end()
        segments.append((seg_start, line_end))
        for seg_s, seg_e in segments:
            cursor = seg_s
            for m in _SENTENCE_END.finditer(text, seg_s, seg_e):
                if _ABBREV.search(text[max(seg_s, m.start() - 8):m.start()]):
                    continue
                pieces.append((cursor, m.start()))
                cursor = m.end()
            pieces.append((cursor, seg_e))
        for s, e in pieces:
            sub = s
            for m in _CLAUSE_SPLIT.finditer(text, s, e):
                spans.append((sub, m.start()))
                sub = m.end()
            spans.append((sub, e))
    out = []
    for s, e in spans:
        s, e = _strip_span(text, s, e)
        if e - s >= MIN_ITEM_CHARS and len(re.findall(r"[A-Za-z]{2,}", text[s:e])) >= 3:
            out.append((s, min(e, s + MAX_ITEM_CHARS)))
    return out


def observation_spans(text: str) -> list[tuple[int, int]]:
    """Informative verbatim lines of a delivered tool output (generic signals only)."""
    stripped = text.strip()
    if stripped in {"returncode: 0", "returncode: None"}:
        return []
    if len(stripped) <= SHORT_OUTPUT_CHARS and stripped.count("\n") <= 3:
        s = text.index(stripped) if stripped else 0
        return [(s, s + len(stripped))] if stripped else []
    lines: list[tuple[int, int]] = []
    pos = 0
    seen: set[str] = set()
    for line in text.splitlines(keepends=True):
        s, e = _strip_span(text, pos, pos + len(line))
        pos += len(line)
        if e - s < 4:
            continue
        body = text[s:e]
        if _INFORMATIVE.search(body) and body not in seen:
            seen.add(body)
            lines.append((s, min(e, s + MAX_ITEM_CHARS)))
    header = text.splitlines()[0] if text else ""
    if lines and len(lines) > MAX_OBSERVATION_LINES:
        half = MAX_OBSERVATION_LINES // 2
        lines = lines[:half] + lines[-half:]
    result = []
    if header.startswith("returncode:") and header.strip() not in {"returncode: 0", "returncode: None"}:
        result.append((0, len(header)))
    result.extend(lines)
    if not lines:
        # Nothing informative: keep the final non-empty line as the observed end state.
        tail = [ln for ln in text.splitlines() if ln.strip()]
        if tail:
            last = tail[-1].strip()
            idx = text.rindex(last)
            if (idx, idx + len(last)) not in result:
                result.append((idx, min(idx + len(last), idx + MAX_ITEM_CHARS)))
    return result


def action_span(command: str) -> tuple[int, int] | None:
    stripped = command.strip()
    if not stripped:
        return None
    s = command.index(stripped)
    limit = min(len(stripped), 300)
    if limit < len(stripped):
        cut = stripped.rfind("\n", 0, limit)
        limit = cut if cut > 40 else limit
    return s, s + limit


def comment_spans(command: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for line in command.splitlines(keepends=True):
        m = _COMMENT.match(line.rstrip("\n"))
        if m and not line.lstrip().startswith("#!") and len(re.findall(r"[A-Za-z]{2,}", m.group(1))) >= 3:
            s = pos + m.start(1)
            out.append((s, min(pos + m.end(1), s + MAX_ITEM_CHARS)))
        pos += len(line)
    return out


def candidates_for_event(event: SourceEvent, *, reasoning_kind: str | None) -> list[Candidate]:
    """All candidates of one eligible source event, in source order.

    ``reasoning_kind`` is ``worker_reasoning`` or ``provider_summary`` when the
    event's returned reasoning was verified as replayed to the worker, and None
    when it is not an eligible source.
    """
    out: list[Candidate] = []
    order = 0

    def add(field_name: str, decoder: str, kind: str, text: str, spans: Iterable[tuple[int, int]], call_id: str | None):
        nonlocal order
        for s, e in spans:
            out.append(Candidate(event.seq, order, field_name, decoder, kind, text[s:e], s, e, call_id))
            order += 1

    if event.kind == "model_response":
        for field_name in sorted(event.texts, key=_field_order):
            raw = event.texts[field_name]
            if field_name == "returned_reasoning_text":
                if reasoning_kind is not None:
                    add(field_name, "raw", reasoning_kind, raw, prose_spans(raw), None)
            elif field_name == "assistant_text":
                add(field_name, "raw", "worker_text", raw, prose_spans(raw), None)
            elif _TOOL_ARG.match(field_name):
                decoder = _decoder_for(event.kind, field_name, raw)
                text = decode(raw, decoder)
                span = action_span(text)
                if span:
                    add(field_name, decoder, "worker_action", text, [span], None)
                add(field_name, decoder, "worker_code_comment", text, comment_spans(text), None)
    elif event.kind in {"observation_delivery", "tool_result"} and "delivered_content" in event.texts:
        raw = event.texts["delivered_content"]
        decoder = _decoder_for(event.kind, "delivered_content", raw)
        text = decode(raw, decoder)
        add("delivered_content", decoder, "observation", text, observation_spans(text), event.tool_call_id)
    elif event.kind == "format_error" and "delivered_text" in event.texts:
        raw = event.texts["delivered_text"]
        stripped = raw.strip()
        if stripped:
            s = raw.index(stripped)
            first = stripped.split("\n\n")[0]
            add("delivered_text", "raw", "format_error", raw, [(s, s + min(len(first), MAX_ITEM_CHARS))], None)
    return out


class TaskText:
    """Exact-copy detector against the task text delivered in the same session."""

    def __init__(self, texts: Iterable[str]):
        self.norm = " ".join(normalize(t) for t in texts)
        words = self.norm.split()
        self.shingles = {" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))}

    def is_copy(self, text: str) -> bool:
        n = normalize(text)
        if len(n) >= 25 and n in self.norm:
            return True
        words = n.split()
        if len(words) < 8:
            return False
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        return sum(g in self.shingles for g in grams) / len(grams) >= 0.8
