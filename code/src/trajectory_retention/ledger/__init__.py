"""The Ledger (v2): the full context of a stalled run, segmented into steps and labelled; nothing compressed or cut.

Two passes, both prefix-only and frozen:
  build_boundaries(store, run_id, checkpoint_id, client, out_dir=...)  before screening: steps, mechanical flags,
      decision tags, implemented-decision windows, one LLM grouping pass into approaches, a separate direct-boundary
      pass (comparison only), and the frozen ``boundary.json``.
  build_full(store, run_id, checkpoint_id, client, boundary=..., out_dir=...)  after admission: exact-quote facts
      (+ survives-rewind), claims (+ later status), ideas (+ tried later / never tried), per-approach status and
      failure-evidence quote; writes ``ledger.json``.
Renderers (pure functions of a ledger dict): render_review_view (arm 2 "ledger", arm 3 "context"), label_text,
render_recipe (arm 6). Gate tooling is in ``ledger.gate``; shared text primitives in ``ledger.text``.

The package imports lazily so that old modules can import ``ledger.text`` without loading the rest.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "build_boundaries": "passes", "build_full": "passes", "build_items": "passes", "load_steps": "steps",
    "render_review_view": "render", "render_recipe": "render", "label_text": "render",
    "study_labeller": "labeller", "ScriptedLabeller": "labeller",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(name)
