"""T2.2 named artifacts: the files, functions, tests and errors a text names.

``named_artifacts(text)`` returns type-prefixed, normalized strings:

* ``file:<basename>`` - paths matched by ``segment._PATH`` whose last component
  has a file extension (and a non-empty stem, so ``.cache`` or ``.git`` are
  not files), reduced to that last component so ``/workspace/pkg/mod.py``,
  ``./pkg/mod.py`` and ``mod.py`` agree (same-named files in different
  directories therefore collide; that only adds a candidate pair).
* ``func:<name>`` - identifiers called or defined: ``name(``, ``obj.method(``
  (last component), ``def name``, ``class Name(`` / ``class Name:`` (code
  form only, so English "class" in prose does not name a class), traceback
  frames ``line N, in name``, backticked ``the `name` function|method|class``,
  and any backticked single token that looks like code: snake_case, CamelCase,
  a dotted.name, or containing ``_`` or ``.`` (a plain English word in
  backticks is not). ``func:`` therefore means "a code identifier named as
  called, defined or quoted as code", not strictly a function, so a quoted
  `parse_offset` and a call ``parse_offset(`` match. Dotted names reduce to
  their last component like calls do; a dotted token ending in a common file
  extension is a file name, not an identifier. Stop-identifiers (builtins,
  keywords, ubiquitous container/string methods), dunder names and names
  shorter than 3 characters are dropped everywhere.
* ``test:<name>`` - ``test_*`` names and ``Test*`` classes, including the parts
  of pytest node ids (``path::Class::test_name[param]``, parameters dropped).
  A ``test_*`` name that is part of a file path is a ``file:``, not a ``test:``.
* ``error:<Name>`` - ``XxxError`` / ``XxxException`` names (last dotted
  component). Assertion failures also give ``error:assert <expression>`` from
  pytest ``E  assert ...`` lines, ``AssertionError: assert ...`` lines, a
  traceback source line directly followed by ``AssertionError``, and
  backticked ``assert ...`` spans (whitespace collapsed, at most 120 chars).

``artifacts_for(items, links)`` gives each item its own names, and each tool
output additionally inherits the names of its producing command (its
``result_of`` action), so ``93 passed`` from ``pytest tests/test_x.py`` names
``test_x.py``. The inheritance is for retrieval only.

Rules are generic Python/pytest/shell text formats; none encode task knowledge.
A shared artifact only chooses which pairs the classifier is asked about
(``LINK_TIER_CANDIDATE``); it never joins items into an approach.
"""

from __future__ import annotations

import re
from typing import Iterable

from .segment import _PATH

ARTIFACTS22_VERSION = "t2.2-artifacts/v2"
ARTIFACT_TYPES = ("file", "func", "test", "error")

_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,5}$")
_SKIP_PATH = ("//", "/dev/", "/proc/")
_DEF = re.compile(r"\bdef\s+([A-Za-z_]\w*)|\bclass\s+([A-Za-z_]\w*)\s*[(:]")
_CALL = re.compile(r"\b([A-Za-z_]\w*)\((?!s\))")          # name( / obj.method( ; not the English "file(s)"
_FRAME = re.compile(r"\bline \d+, in ([A-Za-z_]\w*)")
_NAMED = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`\s+(?:function|method|class|constructor)\b")
_TICKED = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\(\))?`")
_CAMEL = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")
_FILE_EXTS = frozenset("py pyi pyc cfg ini toml json yaml yml txt md rst lock html css js ts sh csv sql log xml conf "
                       "so whl gz zip tar in out".split())
_TEST = re.compile(r"\b(test_\w+|Test[A-Z]\w*)")
_ERROR = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b")
_ASSERTS = (re.compile(r"^E\s+(?:AssertionError:\s*)?(assert\s.+)$", re.MULTILINE),
            re.compile(r"^\s*AssertionError:\s*(assert\s.+)$", re.MULTILINE),
            re.compile(r"^\s*(assert\s.+)\n\s*AssertionError\b", re.MULTILINE),
            re.compile(r"`(assert\s[^`\n]+)`"))
ASSERT_CHARS = 120

STOP_IDENTIFIERS = frozenset("""
self cls super print len str int float bool bytes bytearray list dict set frozenset tuple range open type object
isinstance issubclass getattr setattr hasattr delattr sorted reversed enumerate zip map filter repr format min max
sum abs any all iter next id input vars dir round hash callable staticmethod classmethod property exit quit
append extend insert remove pop get items keys values update join split rsplit splitlines strip rstrip lstrip
startswith endswith lower upper replace encode decode copy read write close readlines writelines
if elif else while for return not and or in is assert lambda yield with except raise del import from def class
await async pass try finally global nonlocal None True False
""".split())


def _file_spans(text: str) -> tuple[set[str], list[tuple[int, int]]]:
    files, spans = set(), []
    for m in _PATH.finditer(text):
        path = m.group(1).rstrip(".-")
        if path.startswith(_SKIP_PATH):
            continue
        base = path.rsplit("/", 1)[-1]
        if _EXT.search(base) and not base.startswith(".") and not base.replace(".", "").isdigit():
            files.add(base)
            spans.append((m.start(1), m.start(1) + len(path)))
    return files, spans


def _is_func(name: str) -> bool:
    return (len(name) >= 3 and name not in STOP_IDENTIFIERS and not (name.startswith("__") and name.endswith("__"))
            and not name.startswith(("test_", "Test")) and not name.endswith(("Error", "Exception")))


def _ticked_identifier(token: str) -> str | None:
    """The normalized name of a backticked token that looks like code, else None."""
    if not ("_" in token or "." in token or _CAMEL.search(token)):
        return None  # a plain word, e.g. `translate` or `Markup`
    parts = token.split(".")
    if len(parts) > 1 and parts[-1].lower() in _FILE_EXTS:
        return None
    return parts[-1] if _is_func(parts[-1]) else None


def named_artifacts(text: str) -> frozenset[str]:
    """Type-prefixed normalized names of files, functions, tests and errors in ``text``."""
    files, file_spans = _file_spans(text)
    inside_file = lambda pos: any(s <= pos < e for s, e in file_spans)
    out = {f"file:{f}" for f in files}
    for pattern in (_DEF, _CALL, _FRAME):
        for m in pattern.finditer(text):
            name = m.group(m.lastindex)
            if _is_func(name):
                out.add(f"func:{name}")
    for m in _NAMED.finditer(text):
        name = m.group(1).rsplit(".", 1)[-1]
        if _is_func(name):
            out.add(f"func:{name}")
    for m in _TICKED.finditer(text):
        name = _ticked_identifier(m.group(1))
        if name:
            out.add(f"func:{name}")
    out.update(f"test:{m.group(1)}" for m in _TEST.finditer(text) if not inside_file(m.start(1)))
    out.update(f"error:{m.group(1)}" for m in _ERROR.finditer(text))
    for pattern in _ASSERTS:
        for m in pattern.finditer(text):
            expr = " ".join(m.group(1).split()).rstrip(":,")[:ASSERT_CHARS]
            out.add(f"error:{expr}")
    return frozenset(out)


def artifacts_for(items: Iterable, links: Iterable | None = None) -> dict[str, frozenset[str]]:
    """Names of every item's own text; a tool output also inherits its producing command's names.

    The producer comes from ``result_of`` links when ``links`` is given, otherwise
    from the same rule ``sources21`` uses to create them (the first action item
    with the output's tool-call id).
    """
    items = list(items)
    out = {i.item_id: named_artifacts(i.text) for i in items}
    if links is not None:
        producer = {l.source_id: l.target_id for l in links if l.relation == "result_of"}
    else:
        action_by_call: dict[str, str] = {}
        for i in items:
            if i.source_kind == "worker_action" and i.tool_call_id:
                action_by_call.setdefault(i.tool_call_id, i.item_id)
        producer = {i.item_id: action_by_call[i.tool_call_id] for i in items
                    if i.source_kind == "observation" and i.tool_call_id in action_by_call}
    for output_id, action_id in producer.items():
        if output_id in out and action_id in out:
            out[output_id] = out[output_id] | out[action_id]
    return out
