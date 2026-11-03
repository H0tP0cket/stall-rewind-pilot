"""T2.1 candidate ledger segmentation (guide §4.1, §7.1).

Differences from T2 ``segment.py`` (which is kept unchanged for T2-frozen):

* Prose splits only at sentence boundaries. No clause splitting, so a
  hypothesis and the action that tests it, a condition and its proposition, or
  a negation and its claim stay in one verbatim unit.
* Long spans are not silently cut: a span over the display limit is kept whole
  in the ledger with ``truncated_for_display`` recorded (rendering may shorten).
* Tool-output lines keep their error-detail continuation lines (``E ``, ``+``/``-``
  diff lines, ``where``/``and`` detail, carets, traceback frames) so expected/actual pairs survive, and
  matches beyond the display cap are recorded rather than silently dropped.
* Every discarded span (code fence, heading, fragment) is recorded with a reason.

All rules are generic text/format rules; none encode task knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ...ledger.text import MIN_CHARS, MIN_WORDS, _words, prose_spans21  # noqa: F401
from ..prefix import SourceEvent
from .segment import (_ABBREV, _COMMENT, _FENCE, _HEADING_ONLY, _INLINE_HEADING, _INFORMATIVE, _SENTENCE_END,
                      _TOOL_ARG, _decoder_for, _field_order, _strip_span, action_span, decode)

SEGMENT21_VERSION = "t2.1-segment/v2"
DISPLAY_CHARS = 700
MAX_CONTINUATION_LINES = 4
# Error-detail continuation formats only (pytest E-lines, diff lines, where/and explanations, caret markers,
# traceback frame lines). Plain indentation is not a continuation: it is usually displayed code.
_CONTINUATION = re.compile(r"^(?:E\s|[+-]\s|\s*(?:where|and)\s|\s*\^+\s*$|\s+File\s\")")


@dataclass(frozen=True)
class Candidate21:
    seq: int
    order: int
    field: str
    decoder: str
    source_kind: str
    text: str
    start: int
    end: int
    truncated_for_display: bool = False


@dataclass
class Discard:
    seq: int
    field: str
    start: int
    end: int
    reason: str

    def as_record(self) -> dict:
        return {"seq": self.seq, "field": self.field, "start": self.start, "end": self.end, "reason": self.reason}


@dataclass
class Segmented:
    candidates: list[Candidate21] = field(default_factory=list)
    discards: list[Discard] = field(default_factory=list)


def observation_spans21(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
    """Informative lines plus their continuation lines, merged into contiguous blocks."""
    stripped = text.strip()
    if stripped in {"returncode: 0", "returncode: None"} or not stripped:
        return [], []
    if len(stripped) <= 240 and stripped.count("\n") <= 3:
        s = text.index(stripped)
        return [(s, s + len(stripped))], []
    lines = []
    pos = 0
    for line in text.splitlines(keepends=True):
        lines.append((pos, pos + len(line), line.rstrip("\n")))
        pos += len(line)
    blocks: list[tuple[int, int]] = []
    header = lines[0][2] if lines else ""
    if header.startswith("returncode:") and header.strip() not in {"returncode: 0", "returncode: None"}:
        blocks.append((0, len(header)))
    i = 1 if header.startswith("returncode:") else 0
    seen: set[str] = set()
    while i < len(lines):
        s, e, body = lines[i]
        if body.strip() and _INFORMATIVE.search(body) and body.strip() not in seen:
            seen.add(body.strip())
            end_i = i
            while (end_i + 1 < len(lines) and end_i - i < MAX_CONTINUATION_LINES
                   and _CONTINUATION.match(lines[end_i + 1][2])):
                end_i += 1
            bs, be = _strip_span(text, s, lines[end_i][1])
            if blocks and bs <= blocks[-1][1] + 1:
                blocks[-1] = (blocks[-1][0], max(be, blocks[-1][1]))
            else:
                blocks.append((bs, be))
            i = end_i + 1
            continue
        i += 1
    if len(blocks) <= (1 if header.startswith("returncode:") and blocks and blocks[0][0] == 0 else 0):
        tail = [ln for ln in lines if ln[2].strip()]
        if tail:
            s, e = _strip_span(text, tail[-1][0], tail[-1][1])
            if (s, e) not in blocks:
                blocks.append((s, e))
    return blocks, []


def comment_spans21(command: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for line in command.splitlines(keepends=True):
        m = _COMMENT.match(line.rstrip("\n"))
        if m and not line.lstrip().startswith("#!") and _words(m.group(1)) >= MIN_WORDS:
            out.append((pos + m.start(1), pos + m.end(1)))
        pos += len(line)
    return out


def segment_event(event: SourceEvent, *, reasoning_kind: str | None) -> Segmented:
    result = Segmented()
    order = 0

    def add(field_name, decoder, kind, text, spans, discards=()):
        nonlocal order
        for s, e in spans:
            result.candidates.append(Candidate21(event.seq, order, field_name, decoder, kind, text[s:e], s, e,
                                                 truncated_for_display=(e - s) > DISPLAY_CHARS))
            order += 1
        for s, e, reason in discards:
            result.discards.append(Discard(event.seq, field_name, s, e, reason))

    if event.kind == "model_response":
        for field_name in sorted(event.texts, key=_field_order):
            raw = event.texts[field_name]
            if field_name == "returned_reasoning_text":
                if reasoning_kind is None:
                    result.discards.append(Discard(event.seq, field_name, 0, len(raw), "reasoning_not_replayed"))
                    continue
                spans, discards = prose_spans21(raw)
                add(field_name, "raw", reasoning_kind, raw, spans, discards)
            elif field_name == "assistant_text":
                spans, discards = prose_spans21(raw)
                add(field_name, "raw", "worker_text", raw, spans, discards)
            elif _TOOL_ARG.match(field_name):
                decoder = _decoder_for(event.kind, field_name, raw)
                text = decode(raw, decoder)
                span = action_span(text)
                if span:
                    add(field_name, decoder, "worker_action", text, [span])
                add(field_name, decoder, "worker_code_comment", text, comment_spans21(text))
    elif event.kind in {"observation_delivery", "tool_result"} and "delivered_content" in event.texts:
        raw = event.texts["delivered_content"]
        decoder = _decoder_for(event.kind, "delivered_content", raw)
        text = decode(raw, decoder)
        spans, discards = observation_spans21(text)
        add("delivered_content", decoder, "observation", text, spans, discards)
    elif event.kind == "format_error" and "delivered_text" in event.texts:
        raw = event.texts["delivered_text"]
        stripped = raw.strip()
        if stripped:
            s = raw.index(stripped)
            first = stripped.split("\n\n")[0]
            add("delivered_text", "raw", "format_error", raw, [(s, s + len(first))])
    return result
