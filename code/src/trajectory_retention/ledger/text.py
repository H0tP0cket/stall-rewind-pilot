"""Shared text primitives for the Ledger, moved here from the T2 code (Astra finding 9).

The definitions below were moved verbatim from ``observers/t2/segment.py`` (field decoders, sentence regexes,
path/test tags), ``observers/t2/segment21.py`` (sentence-only prose spans), ``observers/tstore.py`` (the exact-quote
locator) and ``observers/t2/labelcheck22.py`` (Cohen's kappa). The old modules import them back, so their behaviour
is unchanged. ``json_string_offsets`` is new: it maps a decoded JSON string value back to its raw characters, so a
quote in a decoded command or tool output can be located in the raw context field.

All rules here are generic text/format rules; none encode task knowledge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Sequence

# ---- moved from observers/t2/segment.py ----

_TOOL_ARG = re.compile(r"^tool_call\[(\d+)\]\.arguments$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(\[])")
# Abbreviations after which a period does not end a sentence.
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|cf|approx|incl|resp|Mr|Dr|No|Fig|Eq)\.)$", re.IGNORECASE)
# Summary headings glued into running text, e.g. "...minute.**Addressing timezone issues**".
_INLINE_HEADING = re.compile(r"\*\*[^*\n]{1,80}\*\*")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*```")
_PATH = re.compile(r"(?<![\w/.-])((?:/[\w.-]+)+|[\w.-]+(?:/[\w.-]+)+|[\w-]+\.(?:py|json|ya?ml|toml|txt|md|cfg|ini|sh|js|ts|csv|sql|conf))(?![\w/])")
_TEST_NAME = re.compile(r"\btest_\w+")
_HEADING_ONLY = re.compile(r"^\s*(?:#+\s*)?\*\*[^*]{1,80}\*\*\s*:?\s*$")
_NORM = re.compile(r"[^a-z0-9]+")


def decode(field_text: str, decoder: str) -> str:
    """Deterministic decoders; the recorded span indexes the returned string."""
    if decoder == "raw":
        return field_text
    if decoder == "json:command":
        value = json.loads(field_text)
        if isinstance(value, dict):
            parts = [v for k, v in sorted(value.items()) if isinstance(v, str)]
            if isinstance(value.get("command"), str):
                return value["command"]
            return "\n".join(parts)
        return field_text
    if decoder == "json:returncode+output":
        value = json.loads(field_text)
        if isinstance(value, dict) and "output" in value:
            return f"returncode: {value.get('returncode')}\n{value.get('output') or ''}"
        return field_text
    raise ValueError(f"unknown decoder {decoder!r}")


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in " \t\r\n-*•#>_":
        start += 1
    m = _LIST_MARKER.match(text[start:end])
    if m:
        start += m.end()
    while end > start and text[end - 1] in " \t\r\n*_:":
        end -= 1
    return start, end


def _field_order(name: str) -> tuple[int, int]:
    # Reasoning precedes visible text precedes tool calls, matching production order.
    if name == "returned_reasoning_text":
        return (0, 0)
    if name == "assistant_text":
        return (1, 0)
    m = _TOOL_ARG.match(name)
    return (2, int(m.group(1)) if m else 0)


def tool_call_index(field_name: str) -> int | None:
    m = _TOOL_ARG.match(field_name)
    return int(m.group(1)) if m else None


def normalize(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def tags_for(text: str) -> tuple[str, ...]:
    found = {m.group(1) for m in _PATH.finditer(text) if not m.group(1).startswith(("//", "/dev", "/proc"))}
    found.update(m.group(0) for m in _TEST_NAME.finditer(text))
    return tuple(sorted(t for t in found if len(t) >= 4))


# ---- moved from observers/t2/segment21.py ----

MIN_WORDS = 3
MIN_CHARS = 12


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text))


def prose_spans21(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
    spans: list[tuple[int, int]] = []
    discards: list[tuple[int, int, str]] = []
    pos, in_fence, fence_start = 0, False, 0
    for line in text.splitlines(keepends=True):
        line_start, line_end = pos, pos + len(line)
        pos = line_end
        if _FENCE.match(line):
            if in_fence:
                discards.append((fence_start, line_end, "code_fence"))
            else:
                fence_start = line_start
            in_fence = not in_fence
            continue
        if in_fence or not line.strip():
            continue
        if _HEADING_ONLY.match(line):
            discards.append((line_start, line_end, "heading_only"))
            continue
        segments, seg_start = [], line_start
        for m in _INLINE_HEADING.finditer(text, line_start, line_end):
            segments.append((seg_start, m.start()))
            discards.append((m.start(), m.end(), "inline_heading"))
            seg_start = m.end()
        segments.append((seg_start, line_end))
        for seg_s, seg_e in segments:
            cursor = seg_s
            for m in _SENTENCE_END.finditer(text, seg_s, seg_e):
                if _ABBREV.search(text[max(seg_s, m.start() - 8):m.start()]):
                    continue
                spans.append((cursor, m.start()))
                cursor = m.end()
            spans.append((cursor, seg_e))
    if in_fence:
        discards.append((fence_start, len(text), "code_fence_unclosed"))
    out = []
    for s, e in spans:
        s2, e2 = _strip_span(text, s, e)
        if e2 <= s2:
            continue
        if e2 - s2 < MIN_CHARS or _words(text[s2:e2]) < MIN_WORDS:
            discards.append((s2, e2, "fragment"))
            continue
        out.append((s2, e2))
    return out, discards


# ---- moved from observers/tstore.py ----

def locate_quote(text: str, quote: str, occurrence: int = 0) -> tuple[int, int] | None:
    """Deterministically locate the n-th exact occurrence of quote in text."""
    if not quote or type(occurrence) is not int or occurrence < 0:
        return None
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start < 0:
            return None
    return start, start + len(quote)


# ---- moved from observers/t2/labelcheck22.py ----

def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


# ---- new: raw offsets of a decoded JSON string value ----

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_WS = " \t\r\n"


def _scan_string(raw: str, i: int) -> tuple[str, list[int], int]:
    """Decode the JSON string literal whose opening quote is at ``raw[i]``.

    Returns (value, offsets, end): ``offsets[n]`` is the raw index where decoded character n starts, with one extra
    entry for the end of the value, and ``end`` is the index just after the closing quote."""
    assert raw[i] == '"'
    out: list[str] = []
    offsets: list[int] = []
    j = i + 1
    while True:
        c = raw[j]
        if c == '"':
            offsets.append(j)
            return "".join(out), offsets, j + 1
        offsets.append(j)
        if c != "\\":
            out.append(c)
            j += 1
            continue
        e = raw[j + 1]
        if e != "u":
            out.append(_ESCAPES[e])
            j += 2
            continue
        code = int(raw[j + 2:j + 6], 16)
        j += 6
        if 0xD800 <= code < 0xDC00 and raw[j:j + 2] == "\\u":
            low = int(raw[j + 2:j + 6], 16)
            if 0xDC00 <= low < 0xE000:
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                j += 6
        out.append(chr(code))


def _skip_value(raw: str, i: int) -> int:
    depth = 0
    while i < len(raw):
        c = raw[i]
        if c == '"':
            i = _scan_string(raw, i)[2]
            if depth == 0:
                return i
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                return i
        elif depth == 0 and c in ",":
            return i
        i += 1
    return i


def json_string_offsets(raw: str, key: str) -> tuple[str, list[int]] | None:
    """For a raw JSON object, the top-level string value of ``key`` and its decoded-to-raw index map.

    ``None`` when ``raw`` is not a JSON object, the key is missing or not a string, or the scan disagrees with
    ``json.loads`` (the map is only returned when it is provably exact)."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get(key), str):
        return None
    i = raw.find("{")
    i += 1
    while i < len(raw):
        while i < len(raw) and raw[i] in _WS + ",":
            i += 1
        if i >= len(raw) or raw[i] == "}":
            return None
        name, _, i = _scan_string(raw, i)
        while raw[i] in _WS + ":":
            i += 1
        if name == key and raw[i] == '"':
            value, offsets, _ = _scan_string(raw, i)
            return (value, offsets) if value == parsed[key] else None
        i = _skip_value(raw, i)
    return None
