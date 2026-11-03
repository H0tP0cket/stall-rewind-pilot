"""The worker-visible record of a natural run up to a judge point, and the judge points themselves (stall judge v2).

Judge points (user Q12): steps 10, 20, 30, ... of each natural run, up to its last step, and nothing else. Step k is
the k-th model response with its delivered outputs and any submission feedback delivered right after it (the Ledger's
step definition, ledger/steps.py); a rejected response (format error) is a step too.

The record at point k is the context the worker held after step k, read from what the store saved as the worker's
own context, never from grades or evaluations (user Q10):
  1. the context saved with the point-k checkpoint (v2 natural runs checkpoint every step; validation.step_index = k),
     with any tool output the context-limit rule hid read back from the record, as the Ledger does (the worker saw
     that output in its earlier requests); otherwise
  2. the serialized request of step k+1 (model_request input_ref, worker-visible), which is exactly the context after
     step k (runs from earlier lanes, which checkpoint only some steps); otherwise
  3. the context saved with the run's last checkpoint when it holds exactly k steps and no pending actions (the last
     step of a run from an earlier lane has no next request).
Messages are split into the task head and steps; a user message after the first step is a format-error step, the
submission feedback of the step before it, or another harness message, decided by matching it against the run's
format_error and native_submission_feedback deliveries up to the cut. Nothing else in the store is read here.

Rendered text: the task messages, then "STEP k" blocks (reasoning, visible text, commands, outputs exactly as
delivered, feedback). The worker model's name is masked as in v1. No token counts, grades, scores, check names,
references or harness-private notes. Steps are numbered so the judge can name the onset.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from trajectory_retention.execution.context_limit import unmasked_messages  # noqa: E402
from trajectory_retention.store import ArtifactStore  # noqa: E402

RECORD_VERSION = "judge-v2-record.v1"
POINT_EVERY = 10
SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
FAILING_RX = re.compile(r"Hidden tests still failing:\s*(\d+)\s*\.")
MASKS = [  # the v1 renderer's blinding of the worker model (configs/detector-judge-20260926/render.py), unchanged
    (re.compile(r"deepseek[\w.\-]*", re.I), "[MODEL]"),
    (re.compile(r"\bgpt-?\d[\w.\-]*", re.I), "[MODEL]"),
    (re.compile(r"\bluna\b", re.I), "[MODEL]"),
    (re.compile(r"\bopenai\b", re.I), "[VENDOR]"),
    (re.compile(r"\banthropic\b|\bclaude\b", re.I), "[VENDOR]"),
]
# Event kinds this module reads, and the refs it reads from them. Everything here is worker-visible.
READS = {"model_request": "input_ref", "model_response": "count only", "format_error": "delivered_ref",
         "native_submission_feedback": "delivered_ref", "checkpoint": "state.history (the worker's saved context)"}


def mask(text: str) -> str:
    for rx, rep in MASKS:
        text = rx.sub(rep, text)
    return text


def open_readonly(root: Path, scratch: Path | None = None) -> ArtifactStore:
    """The production store API over a scratch copy of the journal (blobs read in place); the store is never written."""
    tmp = Path(tempfile.mkdtemp(prefix="judge-v2-store-", dir=scratch))
    for suf in ("", "-wal", "-shm"):
        src = root / f"journal.sqlite3{suf}"
        try:
            shutil.copy2(src, tmp / src.name)
        except FileNotFoundError:
            pass
    (tmp / "blobs").symlink_to(root.resolve() / "blobs")
    return ArtifactStore(tmp)


def natural_runs(astore: ArtifactStore) -> list[str]:
    with sqlite3.connect(astore.database) as db:
        return [r[0] for r in db.execute(
            "select distinct run_id from events where run_id like '%:natural' order by run_id")]


@dataclass
class StepView:
    k: int
    fields: list[tuple[str, str]] = field(default_factory=list)  # (kind, text) in context order
    accepted: bool = True
    submission: bool = False
    failing: int | None = None  # from the worker-visible feedback text

    def commands(self) -> list[str]:
        return [t for kind, t in self.fields if kind == "command"]


@dataclass
class Record:
    run_id: str
    point: int
    cut_sequence: int
    source: dict[str, Any]
    head: list[tuple[str, str]]
    steps: list[StepView]
    issues: list[str]
    restored_outputs: list[int]

    def submissions(self) -> list[dict[str, Any]]:
        """Submissions visible in the record: a step with submission feedback after it, or whose command submits."""
        return [{"step": s.k, "failing": s.failing, "feedback": s.submission} for s in self.steps
                if s.submission or any(SUBMIT_MARK in c for c in s.commands())]

    def texts(self) -> list[str]:
        """Every unmasked text piece the rendering is built from (for the provenance check)."""
        return [t for _, t in self.head] + [t for s in self.steps for _, t in s.fields]

    def render(self) -> str:
        out = []
        for role, text in self.head:
            out.append(f"===== {role.upper()} MESSAGE TO THE AGENT =====\n{text}")
        for s in self.steps:
            parts = [f"===== STEP {s.k} ====="]
            n_cmd = 0
            for kind, text in s.fields:
                if kind == "reasoning":
                    parts.append(f"[reasoning]\n{text}")
                elif kind == "message":
                    parts.append(f"[visible text]\n{text}")
                elif kind == "command":
                    n_cmd += 1
                    parts.append(f"[command {n_cmd}]\n{text}")
                elif kind == "output":
                    parts.append(f"----- output delivered to the agent -----\n{text}")
                elif kind == "format_error":
                    parts.append(f"[the harness rejected this response as a format error; message delivered to the "
                                 f"agent]\n{text}")
                elif kind == "feedback":
                    parts.append(f"----- submission feedback delivered to the agent -----\n{text}")
                else:
                    parts.append(f"----- message delivered to the agent -----\n{text}")
            out.append("\n".join(parts))
        return mask("\n\n".join(out))


def _reasoning(m: dict) -> str | None:
    r = m.get("reasoning") or m.get("reasoning_content")
    if isinstance(r, str) and r:
        return r
    texts = [d.get("text") or d.get("summary") for d in m.get("reasoning_details") or () if isinstance(d, dict)]
    texts = [t for t in texts if isinstance(t, str) and t]
    return "\n".join(texts) or None


def _command(call: dict) -> str:
    f = (call or {}).get("function") or call or {}
    args = f.get("arguments")
    try:
        a = json.loads(args) if isinstance(args, str) else (args or {})
        return a["command"] if isinstance(a, dict) and isinstance(a.get("command"), str) else json.dumps(a)
    except ValueError:
        return str(args)


def parse_messages(messages: list[dict], fe_texts: Counter, fb_texts: Counter) -> tuple[list, list[StepView], list[str]]:
    head: list[tuple[str, str]] = []
    steps: list[StepView] = []
    issues: list[str] = []
    for i, m in enumerate(messages):
        role, extra = m.get("role"), m.get("extra") or {}
        content = m.get("content") if isinstance(m.get("content"), str) else None
        if role == "assistant":
            s = StepView(len(steps) + 1)
            steps.append(s)
            r = _reasoning(m)
            if r:
                s.fields.append(("reasoning", r))
            if content:
                s.fields.append(("message", content))
            for call in m.get("tool_calls") or ():
                s.fields.append(("command", _command(call)))
            continue
        if role == "tool":
            if not steps:
                issues.append(f"tool message {i} before any step")
                continue
            steps[-1].fields.append(("output", content or ""))
            continue
        if role == "exit":  # the agent loop's own exit marker: saved in the context, never sent to the model
            continue
        text = content or ""
        if not steps:
            head.append((role or "user", text))
            continue
        if extra.get("interrupt_type") == "FormatError" or fe_texts[text] > 0:
            if fe_texts[text] > 0:
                fe_texts[text] -= 1
            steps.append(StepView(len(steps) + 1, [("format_error", text)], accepted=False))
            continue
        if fb_texts[text] > 0:
            fb_texts[text] -= 1
            s = steps[-1]
            s.fields.append(("feedback", text))
            s.submission = True
            hit = FAILING_RX.search(text)
            s.failing = int(hit.group(1)) if hit else None
            continue
        steps[-1].fields.append(("user", text))
        issues.append(f"step {steps[-1].k}: user message {i} matches no feedback or format-error delivery")
    return head, steps, issues


class RunView:
    """One natural run, read once: worker-visible events and point checkpoints."""

    def __init__(self, astore: ArtifactStore, run_id: str):
        self.store, self.run_id = astore, run_id
        self.events = list(astore.iter_events(run_id))
        self.requests = [e for e in self.events if e.kind == "model_request"]
        self.n_steps = sum(1 for e in self.events if e.kind == "model_response")
        self.checkpoints = astore.list_checkpoints(run_id)
        self.terminal = next((e.payload for e in reversed(self.events) if e.kind == "terminal"), None)
        self.task = next((e.scope.task_id for e in self.events if e.scope is not None and e.scope.task_id), None)
        cfg = next((e.payload.get("config") for e in self.events if e.kind == "lifecycle" and "config" in e.payload), {})
        self.policy = ((cfg or {}).get("metadata") or {}).get("v2_run_policy")
        self.submission_feedback = ((cfg or {}).get("metadata") or {}).get("submission_feedback")
        self.model = next((e.payload.get("model") for e in self.requests), None)

    def points(self) -> list[int]:
        return list(range(POINT_EVERY, self.n_steps + 1, POINT_EVERY))

    def point_checkpoint(self, k: int):
        """The v2 point checkpoint of point k (validation.step_index == k): the last branchable one, as
        mining.points.point_checkpoints maps points; else the last one recorded; else None."""
        hits = [c for c in self.checkpoints if (c.validation or {}).get("step_index") == k]
        branchable = [c for c in hits if c.branchable]
        return (branchable or hits or [None])[-1]

    def passing_submission_steps(self) -> list[int]:
        """Steps of passing submissions (the run's outcome; never shown to the judge)."""
        out = []
        for e in self.events:
            if e.kind == "lifecycle" and e.payload.get("transition") == "native_submission_verdict" \
                    and e.payload.get("outcome") == "native_success":
                cp = e.payload.get("checkpoint_id")
                c = next((c for c in self.checkpoints if c.checkpoint_id == cp), None)
                out.append((c.validation or {}).get("step_index") if c is not None else None)
        return out

    def _texts(self, kind: str, cut: int) -> Counter:
        return Counter(self.store.get_bytes(e.delivered_ref).decode("utf-8") for e in self.events
                       if e.kind == kind and e.sequence <= cut and e.delivered_ref is not None)

    def record(self, k: int) -> Record:
        """The record after step k (see the module docstring for the source order)."""
        if not 1 <= k <= self.n_steps:
            raise ValueError(f"point {k} outside 1..{self.n_steps}")
        messages, restored, source, cut = None, [], None, None
        cp = self.point_checkpoint(k)
        if cp is not None and cp.state.history is not None:
            history = self.store.get_json(cp.state.history)
            messages, restored = unmasked_messages(history["messages"], self.store)
            source, cut = {"kind": "point_checkpoint_context", "checkpoint_id": cp.checkpoint_id}, cp.event_sequence
        elif k < len(self.requests):
            req = self.requests[k]
            body = self.store.get_json(req.input_ref)
            messages = body["messages"] if isinstance(body, dict) else body
            source = {"kind": "next_request", "event_sequence": req.sequence, "request_index": k + 1}
            cut = req.sequence
        else:
            for c in reversed(self.checkpoints):
                if c.state.history is None:
                    continue
                h = self.store.get_json(c.state.history)
                pending = (c.validation or {}).get("pending_actions")
                if (h.get("counters") or {}).get("n_calls") == k and not pending:
                    messages, restored = unmasked_messages(h["messages"], self.store)
                    source, cut = {"kind": "last_checkpoint_context", "checkpoint_id": c.checkpoint_id}, c.event_sequence
                break
        if messages is None:
            raise LookupError(f"{self.run_id}: no worker context recorded after step {k}")
        head, steps, issues = parse_messages(messages, self._texts("format_error", cut),
                                             self._texts("native_submission_feedback", cut))
        if len(steps) != k:
            issues.append(f"the context holds {len(steps)} steps, not {k}")
        return Record(self.run_id, k, cut, source, head, steps, issues, restored)


def worker_visible_corpus(run: RunView, cut: int) -> list[str]:
    """Every worker-visible text the run delivered or sent up to the cut (for the provenance check in the E2E):
    request bodies, delivered texts, model response outputs, and saved contexts of checkpoints up to the cut."""
    out = []
    for e in run.events:
        if e.sequence > cut:
            break
        for ref in (e.input_ref, e.delivered_ref, e.output_ref):
            if ref is not None and getattr(ref.visibility, "value", ref.visibility) == "worker_visible":
                out.append(run.store.get_bytes(ref).decode("utf-8", "replace"))
    for c in run.checkpoints:
        if c.event_sequence <= cut and c.state.history is not None:
            msgs, _ = unmasked_messages(run.store.get_json(c.state.history)["messages"], run.store)
            out.append(json.dumps(msgs, ensure_ascii=False))
            out.extend(m.get("content") for m in msgs if isinstance(m.get("content"), str))
            out.extend(_reasoning(m) or "" for m in msgs)
    return out
