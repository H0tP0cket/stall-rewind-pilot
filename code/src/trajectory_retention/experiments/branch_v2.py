"""v2 branches: ``run_branch(spec, *, out_dir, model_factory=None) -> dict`` (branch-spec.v2, brief-interfaces.md).

One branch = one MiningRunner run under the branch-v2/v1 policy, from a source checkpoint of a natural v2 run. The
engine pieces it uses:

- Per-channel restore. W, R and the context come from independent same-run points (any point 0..s, where s is the
  source checkpoint's point; point s is the source checkpoint itself, an earlier point k is the run's checkpoint for
  point k). They are ordinary component donors, so the runner's donor guards apply unchanged.
- The outputs channel (OUTPUTS_RULE in context_edit.py): with the context at point c and the outputs at point o < c,
  the outputs of steps o+1..c are replaced by OUTPUT_PLACEHOLDER and the commands stay; o >= c changes nothing.
- Context edits (the `edited` history policy): drop_steps, drop_outputs, drop_quote, validated on the actual next
  request.
- Review (arms 2 and 3) and note (arm 5) turns: a one-shot mode of the recording agent at the source checkpoint
  (runner._v2_turn). A review call is valid when it is one review_decision call whose arguments pass the interface
  rules (configs/v2/study_common.validate_decision) and compile against the real context; an invalid call gets one
  retry, then the branch changes nothing (flag invalid_call). The choice compiles through the view's units:
  step S{k} and approach A{n} -> drop_steps (clipped to the context point), output O{k} -> drop_outputs, item I{n}
  -> drop_quote at its step; kept units come back in the hand-back as ledger.render.label_text gives them.
- The hand-back: one message right after the kept context (HANDBACK_RULE).
- Arm set v3 (branch-spec.v3, user 2026-10-16/17), beside the v2 arms, whose behaviour is unchanged: 1 continue;
  2 and 3 the review turn on the flat Ledger or the plain context; 4 forced rewind of every channel to P, the judge's
  stall onset point (spec.stall_onset_point); 5 and 6 a read-only turn on the flat Ledger or the plain context (no
  tool). Every v3 review turn's request is the system and task messages plus the view (REVIEW_REQUEST_RULE), and its
  accepted response stays in the continuing context before the hand-back (KEPT_OUTPUT_RULE).
- Injection (injection-payload.v1, configs/v2/healthy_v2.py): a separate composition type. Worker-visible donor
  messages of the cut steps are spliced after the last step with renamed tool-call ids, or debris files enter W
  through a recorded workspace composition. The receiver's W, R and context stay ordinary donors.

Stores: the branch writes only into its own store under ``out_dir/store``; reads of the source run fall through to the
source store, opened read-only (mining/branch_store.py). result.json carries that store and the run id.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import traceback
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from minisweagent.exceptions import FormatError

from ..contracts import ArtifactRef, RunConfig, RuntimeClass, TreatmentManifest, WorkerRequestPolicy, from_jsonable
from ..mining.branch_store import BranchStore, ReadOnlyArtifactStore
from ..mining.points import checkpoint_step_index, point_checkpoints
from . import v2_text as T
from .context_edit import EditError, OUTPUTS_RULE, outputs_edit, steps_of, validate_edits

SPEC_VERSION = "branch-spec.v2"
RESULT_VERSION = "branch-result.v2"
ENGINE_VERSION = "branch-v2-engine/1"
INJECTION_VERSION = "injection-payload.v1"
# experiments/v2_text.SUBMISSION_FEEDBACK as delivered by count-feedback runs
NATIVE_FEEDBACK = re.compile(r"Hidden tests still failing: \d+\.")
ARMS = ("1_continue", "2_ledger_review", "3_context_review", "4_forced_rewind", "5_rewind_own_summary",
        "6_rewind_ledger_recipe")
REVIEW_ARMS = ("2_ledger_review", "3_context_review")
REWIND_ARMS = ("4_forced_rewind", "5_rewind_own_summary", "6_rewind_ledger_recipe")
CHANNELS = ("workspace", "runtime", "context", "outputs")
SPEC_KEYS = {"version", "branch_id", "purpose", "source", "arm", "repeat", "seed", "rewind", "context_edits", "review",
             "note", "handback", "budget", "injection"}
PURPOSES = ("screen", "tuning", "main", "injection", "harm")
NO_TOKEN_CAP = 10**9  # a null new-token cap: only the $3 runaway-spend guard (engine allowance set beyond reach)
ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs" / "v2"
PRICE_CARD = CONFIGS / "price_card.json"
SCRATCH = Path(os.environ.get("TR_V2_BRANCH_SCRATCH", "~/.cache/trajectory-retention/v2-branches")).expanduser()
HANDBACK_RULE = ("handback/v1 (provisional formatting): one user message right after the kept context, sent when "
                 "anything was undone or removed or anything is carried: HANDBACK_HEADER; GENERIC_LINE when anything "
                 "was undone or removed; one GRADER_RECORD_LINE per submission of the source run up to the source "
                 "checkpoint (a fixed slot no drop can reach); then the kept items, the note or the recipe, each "
                 "after a blank line")
FAILED_COMMAND_RULE = ("failed-command/provisional-1: a command of the first continuation step ended with a status "
                       "other than ok (nonzero exit, timeout or exception)")
MISMATCH_RULE = ("mismatch/v1: the effective points of W, R (only when the source's R is captured and restored), the "
                 "context and the outputs (min(outputs, context)) are not all equal")
NO_CHANGE_RULE = ("no-change/v1 (arms 2 and 3): the review left everything as it was: no channel undone, no context "
                  "edit and nothing carried back, whether by change_nothing, an empty decision or the fallback")
TOKEN_DEFINITIONS = {
    "review_tokens": "generated tokens (reasoning included) of every attempt of the review turn, provider-reported",
    "note_tokens": "length of the note text in tiktoken o200k_base tokens (the recipe cap's tokenizer); the note "
                   "turn's generated tokens are in details.note_turn_generated_tokens",
    "handback_tokens": "length of the hand-back message in tiktoken o200k_base tokens (0 when none was sent)",
    "new_tokens": "generated tokens counted against the new-token cap: review or note turn plus continuation",
}
END_REASONS = {"sequence_complete:native_success": "passing_submission",
               "sequence_complete:native_repeated_failure": "identical_failures",
               "sequence_complete:native_no_improvement": "no_improvement",
               "sequence_complete:submission_cap": "submission_cap", "budget_exhausted": "token_cap",
               "cost_limit": "cost_guard", "call_limit": "step_guard",
               "context_window_exhausted": "context_window_exhausted"}
RESTORE_OPERATIONS = ("environment_create", "workspace_restore_input", "workspace_restore", "runtime_restore_input",
                      "runtime_restore", "history_restore_input", "restored_session_prepare")
FORBIDDEN_MIN_CHARS = 40  # turn texts shorter than this are not searched for in the next request

# ---- arm set v3 (user, 2026-10-16/17), beside the branch-spec.v2 arms above, which keep their behaviour unchanged
SPEC_VERSION_V3 = "branch-spec.v3"
ENGINE_VERSION_V3 = "branch-v2-engine/2 (arm set v3)"
ARMS_V3 = ("1_continue", "2_ledger_review", "3_context_review", "4_forced_rewind_to_p", "5_ledger_read_only",
           "6_context_read_only")
REVIEW_ARMS_V3 = ("2_ledger_review", "3_context_review")
READ_ONLY_ARMS_V3 = ("5_ledger_read_only", "6_context_read_only")
TURN_ARMS_V3 = REVIEW_ARMS_V3 + READ_ONLY_ARMS_V3
SPEC_KEYS_V3 = SPEC_KEYS | {"stall_onset_point"}
REVIEW_REQUEST_RULE = ("review-request/v2 (arm set v3, user Q24(a)): the request of a review turn (arms 2, 3, 5, 6) "
                       "holds only the source context's system and task messages (its step 0) and then the review "
                       "message (the view, built from the full record, and the files/process snapshot); never the "
                       "running context. The context-limit rule applies to it unchanged (it holds no tool output, so "
                       "a view that does not fit ends the branch with context_window_exhausted)")
KEPT_OUTPUT_VERSION = "review-output-kept/v1"
KEPT_OUTPUT_RULE = (KEPT_OUTPUT_VERSION + " (arm set v3, user Q1; representation DRAFT): the review turn's accepted "
                    "response stays in the continuing context as one assistant message right after the kept (edited) "
                    "context and before the hand-back: every field as returned (its reasoning fields included) except "
                    "its tool calls; content = its own text, then for each review_decision call REVIEW_DECISION_KEPT "
                    "with the call's arguments exactly as written (after a blank line when there is text). A turn "
                    "without an accepted response keeps nothing; rejected attempts, format-error messages, the review "
                    "message and the view never enter the context")
HANDBACK_RULE_V3 = (HANDBACK_RULE + "; arm set v3: also sent after every review turn (arms 2, 3, 5 and 6, change "
                    "nothing included), so that the kept review output is followed by a user message")
FORCED_REWIND_RULE_V3 = ("forced-rewind-to-p/v1 (arm set v3, arm 4): W, R, the context and the outputs all return to "
                         "point P = spec.stall_onset_point (the judge's stall onset point, taken as given); P equal to "
                         "the source point undoes nothing")
VIEW_KINDS_V3 = {"2_ledger_review": "ledger_flat", "5_ledger_read_only": "ledger_flat",
                 "3_context_review": "context", "6_context_read_only": "context"}


class BranchError(ValueError):
    pass


def _study():
    """Track S's prompt renderer and decision rules (configs/v2), the single source of the review and note texts."""
    if str(CONFIGS) not in sys.path:
        sys.path.insert(0, str(CONFIGS))
    import prompt_texts
    import study_common
    return prompt_texts, study_common


def _engine(spec: dict) -> str:
    return ENGINE_VERSION_V3 if spec.get("version") == SPEC_VERSION_V3 else ENGINE_VERSION


def _sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.~" else "~" for c in text)


def _ref(value: Any) -> ArtifactRef:
    return from_jsonable(ArtifactRef, value) if isinstance(value, dict) else value


def check_spec(spec: Any) -> None:
    """Refuse a malformed branch spec before anything runs."""
    if not isinstance(spec, dict) or spec.get("version") not in (SPEC_VERSION, SPEC_VERSION_V3):
        raise BranchError(f"spec must be a {SPEC_VERSION} or {SPEC_VERSION_V3} object")
    if spec["version"] == SPEC_VERSION_V3:
        return _check_spec_v3(spec)
    if set(spec) - {"submission_stop", "unresolved_retry", "submission_feedback"} != SPEC_KEYS:
        raise BranchError(f"spec keys {sorted(set(spec) ^ SPEC_KEYS)} differ from the interface")
    if "submission_stop" in spec:
        from .submission_stop import validate
        if validate(spec["submission_stop"]) is None:
            raise BranchError("an explicit submission_stop must enable the supported rule")
    arm = spec["arm"]
    if arm not in ARMS and not (isinstance(arm, str) and arm.startswith("injection:")):
        raise BranchError(f"unknown arm {arm!r}")
    _check_common(spec)
    rw = spec["rewind"]
    hb = spec["handback"]
    review, note, injection = spec["review"], spec["note"], spec["injection"]
    if (review is not None) != (arm in REVIEW_ARMS):
        raise BranchError("a review is given exactly for arms 2 and 3")
    if review is not None and (set(review) != {"view_ref", "prompt_version", "rewind_mode", "opus_record"}
                               or review["rewind_mode"] not in ("per_channel", "coupled")):
        raise BranchError("review must be {view_ref, prompt_version, rewind_mode: per_channel|coupled, opus_record}")
    if review is not None and (rw is not None or spec["context_edits"]):
        raise BranchError("a review arm takes its rewind and drops from the review call only")
    if (note is not None) != (arm == "5_rewind_own_summary") or (note is not None and set(note) != {"prompt_version"}):
        raise BranchError("a note {prompt_version} is given exactly for arm 5")
    if arm in REWIND_ARMS and rw is None:
        raise BranchError(f"{arm} needs a rewind")
    if note is not None and rw["context"] is None:
        raise BranchError("the arm 5 note needs a context rewind point")
    if (hb["recipe"] is not None) != (arm == "6_rewind_ledger_recipe"):
        raise BranchError("a recipe is handed back exactly in arm 6")
    if (injection is not None) != arm.startswith("injection:"):
        raise BranchError("an injection payload is given exactly for injection arms")
    if injection is not None and (rw is not None or spec["context_edits"]):
        raise BranchError("an injection branch keeps the receiving checkpoint's W, R and context")


def _check_spec_v3(spec: dict) -> None:
    """branch-spec.v3: the six arms of arm set v3 only (no note, recipe or injection)."""
    if set(spec) - {"submission_stop", "unresolved_retry", "submission_feedback"} != SPEC_KEYS_V3:
        raise BranchError(f"spec keys {sorted(set(spec) ^ SPEC_KEYS_V3)} differ from the v3 interface")
    if "submission_stop" in spec:
        from .submission_stop import validate
        if validate(spec["submission_stop"]) is None:
            raise BranchError("an explicit submission_stop must enable the supported rule")
    arm = spec["arm"]
    if arm not in ARMS_V3:
        raise BranchError(f"unknown arm {arm!r} for {SPEC_VERSION_V3}")
    _check_common(spec)
    rw, review, p = spec["rewind"], spec["review"], spec["stall_onset_point"]
    if spec["note"] is not None or spec["injection"] is not None or spec["handback"]["recipe"] is not None:
        raise BranchError(f"{SPEC_VERSION_V3} has no note, recipe or injection")
    if (review is not None) != (arm in TURN_ARMS_V3):
        raise BranchError("a review is given exactly for arms 2, 3, 5 and 6")
    if review is not None:
        if set(review) != {"view_ref", "prompt_version", "rewind_mode", "opus_record"}:
            raise BranchError("review must be {view_ref, prompt_version, rewind_mode, opus_record}")
        if arm in REVIEW_ARMS_V3 and review["rewind_mode"] not in ("per_channel", "coupled"):
            raise BranchError("arms 2 and 3 need rewind_mode per_channel or coupled")
        if arm in READ_ONLY_ARMS_V3 and (review["rewind_mode"] is not None or review["opus_record"]):
            raise BranchError("a read-only arm has no rewind mode and no Opus record (it makes no decision)")
    if arm == "4_forced_rewind_to_p":
        if type(p) is not int or p < 0:
            raise BranchError("arm 4 needs stall_onset_point, a point (non-negative integer)")
        if rw != dict.fromkeys(CHANNELS, p) or spec["context_edits"]:
            raise BranchError("arm 4 rewinds every channel to stall_onset_point and edits nothing else")
    elif p is not None or rw is not None or spec["context_edits"]:
        raise BranchError(f"{arm} takes no stall_onset_point, rewind or context edits in the spec")


def _check_common(spec: dict) -> None:
    if "submission_feedback" in spec:
        from .submission_feedback import validate
        if validate(spec["submission_feedback"]) is None:
            raise BranchError("an explicit submission_feedback must enable the supported rule")
    if "unresolved_retry" in spec:
        from ..provider_policy import unresolved_retry_rule
        if unresolved_retry_rule(spec["unresolved_retry"]) is None:
            raise BranchError("an explicit unresolved_retry must enable the supported rule")
    if spec["purpose"] not in PURPOSES:
        raise BranchError(f"unknown purpose {spec['purpose']!r}")
    src = spec["source"]
    if not isinstance(src, dict) or set(src) != {"store", "run_id", "checkpoint_id"}:
        raise BranchError("source must be {store, run_id, checkpoint_id}")
    b = spec["budget"]
    if not isinstance(b, dict) or set(b) != {"new_token_cap", "submission_cap", "cost_limit_usd", "step_limit"}:
        raise BranchError("budget must be {new_token_cap, submission_cap, cost_limit_usd, step_limit}")
    for k in ("new_token_cap", "submission_cap"):
        if b[k] is not None and (type(b[k]) is not int or b[k] <= 0):
            raise BranchError(f"budget.{k} must be null or a positive integer")
    if type(b["step_limit"]) is not int or b["step_limit"] <= 0 or not isinstance(b["cost_limit_usd"], (int, float)) \
            or b["cost_limit_usd"] <= 0:
        raise BranchError("budget needs a positive cost_limit_usd and step_limit")
    rw = spec["rewind"]
    if rw is not None and (not isinstance(rw, dict) or set(rw) != set(CHANNELS)
                           or any(v is not None and (type(v) is not int or v < 0) for v in rw.values())):
        raise BranchError("rewind must be null or {workspace, context, runtime, outputs} of points or nulls")
    if not isinstance(spec["context_edits"], list):
        raise BranchError("context_edits must be a list")
    hb = spec["handback"]
    if not isinstance(hb, dict) or set(hb) != {"items", "recipe"} or not isinstance(hb["items"], list) \
            or not all(isinstance(i, str) for i in hb["items"]) or not (hb["recipe"] is None or isinstance(hb["recipe"], str)):
        raise BranchError("handback must be {items: [text], recipe: text | null}")


def run_started(store: Any, run_id: str) -> dict:
    for event in store.iter_events(run_id):
        if event.kind == "lifecycle" and event.payload.get("transition") == "run_started":
            return event.payload
    raise BranchError(f"run {run_id} has no run_started record")


def _first_payload(store: Any, run_id: str, transition: str) -> dict | None:
    for event in store.iter_events(run_id):
        if event.kind == "lifecycle" and event.payload.get("transition") == transition:
            return event.payload
    return None


# ----------------------------------------------------------------------- turn parsers
def _review_error(error: str) -> FormatError:
    return FormatError({"role": "user", "content": T.REVIEW_FORMAT_ERROR.format(error=error),
                        "extra": {"interrupt_type": "FormatError"}})


def _note_parser(tool_calls: list, *, format_error_template: str, template_kwargs: dict | None = None) -> list:
    if tool_calls:
        raise FormatError({"role": "user", "content": T.NOTE_FORMAT_ERROR, "extra": {"interrupt_type": "FormatError"}})
    return []


def _read_only_parser(tool_calls: list, *, format_error_template: str, template_kwargs: dict | None = None) -> list:
    """Arms 5 and 6 (arm set v3): no tool is offered; a response with tool calls gets one retry (as the note turn)."""
    if tool_calls:
        raise FormatError({"role": "user", "content": T.NOTE_FORMAT_ERROR, "extra": {"interrupt_type": "FormatError"}})
    return []


def _note_check(message: dict) -> str | None:
    return None if isinstance(message.get("content"), str) and message["content"].strip() else T.NOTE_FORMAT_ERROR


# ----------------------------------------------------------------------- the plan
class BranchPlan:
    """Everything the runner needs to run one branch: its turn, then the compile of the turn's outcome."""

    def __init__(self, spec: dict, store: Any, start: dict) -> None:
        self.spec = spec
        self.store = store
        self.start = start
        from .submission_feedback import validate
        self.submission_feedback = validate(spec.get("submission_feedback",
            (start.get("config", {}).get("metadata") or {}).get("submission_feedback")))
        self.v3 = spec["version"] == SPEC_VERSION_V3
        self.read_only = self.v3 and spec["arm"] in READ_ONLY_ARMS_V3
        src = spec["source"]
        self.run_id = src["run_id"]
        try:
            self.source = store.get_checkpoint(src["checkpoint_id"])
        except KeyError as exc:
            raise BranchError(f"source checkpoint {src['checkpoint_id']} not found") from exc
        if self.source.run_id != self.run_id:
            raise BranchError("source checkpoint belongs to another run")
        if not self.source.branchable:
            raise BranchError("source checkpoint is not branchable")
        self.s = checkpoint_step_index(store, self.source)
        self.points = point_checkpoints(store, self.run_id)
        self.history = store.get_json(self.source.state.history)
        budget = spec["budget"]
        self.submission_cap = budget["submission_cap"]
        self.token_limit = budget["new_token_cap"] or NO_TOKEN_CAP
        self._histories: dict[str, list] = {self.source.checkpoint_id: self.history["messages"]}
        self.view = self.ledger = None
        review = spec["review"]
        if review is not None:
            view_path = Path(review["view_ref"]).expanduser()
            self.view = json.loads(view_path.read_text())
            if not isinstance(self.view.get("text"), str) or not isinstance(self.view.get("units"), dict):
                raise BranchError("review view lacks text or units")
            for key, want in (("run_id", self.run_id), ("checkpoint_id", self.source.checkpoint_id),
                              ("stall_point", self.s)):
                if key in self.view and self.view[key] != want:
                    raise BranchError(f"review view {key} {self.view[key]!r} differs from the source ({want!r})")
            if self.v3 and self.view.get("kind") != VIEW_KINDS_V3[spec["arm"]]:
                raise BranchError(f"{spec['arm']} needs a {VIEW_KINDS_V3[spec['arm']]!r} view, not "
                                  f"{self.view.get('kind')!r} (arm set v3: the flat Ledger has no approach layer)")
            self._boundary_ok(view_path.parent)
            ledger_path = view_path.parent / "ledger.json"
            if not ledger_path.exists():
                raise BranchError("ledger.json missing next to the review view (kept units are labelled from it)")
            self.ledger = json.loads(ledger_path.read_text())
        self.grader_record = self._grader_record()

    @staticmethod
    def _boundary_ok(directory: Path) -> None:
        if (directory / "boundary.failed.json").exists():
            raise BranchError("the stall's boundary pass failed (boundary.failed.json); it is not branched")
        if (directory / "boundary.json").exists():
            status = json.loads((directory / "boundary.json").read_text()).get("status")
            if status != "ok":
                raise BranchError(f"boundary.json status is {status!r}, not 'ok'")

    # ---- points and contexts
    def checkpoint_for(self, k: Any) -> str:
        if type(k) is not int or not 0 <= k <= self.s:
            raise BranchError(f"point {k!r} is outside 0..{self.s}")
        if k == self.s:
            return self.source.checkpoint_id
        if k not in self.points:
            raise BranchError(f"point {k} has no branchable checkpoint")
        return self.points[k]

    def messages_at(self, k: int) -> list:
        cid = self.checkpoint_for(k)
        if cid not in self._histories:
            self._histories[cid] = self.store.get_json(self.store.get_checkpoint(cid).state.history)["messages"]
        return self._histories[cid]

    def _grader_record(self) -> list[dict]:
        """Every submission of the source run up to the source checkpoint, from its submission verdict events."""
        out = []
        for e in self.store.iter_events(self.run_id, through_sequence=self.source.event_sequence):
            if e.kind != "lifecycle" or e.payload.get("transition") != "native_submission_verdict":
                continue
            p = e.payload
            n, k = p.get("failing_count"), p.get("step_index")
            if self.submission_feedback is not None:
                n = None  # new hand-backs identify submissions, never disclose private scores
            elif type(n) is not int:  # historical count-feedback protocol
                raise BranchError(f"source submission {p.get('ordinal')} records no failing count shown to the worker")
            if k is None and p.get("checkpoint_id"):
                k = checkpoint_step_index(self.store, self.store.get_checkpoint(p["checkpoint_id"]))
            out.append({"i": p["ordinal"], "k": k, "n": n, "status": p.get("status"), "event_sequence": e.sequence})
        return out

    # ---- the read-only snapshot shown in the review turn
    def snapshot(self) -> dict:
        cp = self.source
        w = self.store.get_json(_ref(cp.validation["workspace_manifest_ref"]))
        r = self.store.get_json(cp.state.runtime) if cp.state.runtime is not None else {}

        def files(entries: dict, prefix: str) -> list[str]:
            rows = []
            for path, e in sorted(entries.items()):
                if e.get("type") == "f":
                    rows.append(f"  {prefix}{path} ({e.get('size')} bytes)")
                elif e.get("type") == "l":
                    rows.append(f"  {prefix}{path} -> {e.get('link')}")
            return rows

        wrows = files(w, "/workspace/")
        rrows = files(r.get("entries") or {}, "/")
        drows = [f"  /{p}" for p in sorted(r.get("deleted") or [])]
        prows = [f"  {p.get('cmdline')}" for p in r.get("foreign_processes") or []]
        lines = []
        remaining = 200 if self.v3 else None
        omitted = 0
        for title, rows in ((T.SNAPSHOT_WORKSPACE, wrows), (T.SNAPSHOT_RUNTIME, rrows), (T.SNAPSHOT_DELETED, drows),
                            (T.SNAPSHOT_PROCESSES, prows)):
            lines.append(title.format(n=len(rows)))
            shown = rows if remaining is None else rows[:remaining]
            omitted += len(rows) - len(shown)
            if remaining is not None:
                remaining -= len(shown)
            lines += shown or ([f"  {T.SNAPSHOT_NONE}"] if not rows else [])
        if omitted:
            lines.append(f"({omitted} entries omitted by the 200-entry snapshot limit.)")
        text = "\n".join(lines)
        return {"text": text, "record": {"checkpoint_id": cp.checkpoint_id, "sha256": _sha(text),
                "workspace_files": len(wrows), "runtime_files": len(rrows), "runtime_deleted": len(drows),
                "processes": len(prows), "entry_limit": 200 if self.v3 else None, "omitted_entries": omitted,
                "rule": "snapshot/v1: files and symlinks of W and of R outside the workspace, "
                "R deletions, and processes beyond the harness's own (none at a branchable checkpoint)"}}

    # ---- the turn
    def turn(self) -> dict | None:
        prompts, _ = _study()
        if self.read_only:  # arms 5 and 6 of arm set v3: the view for one turn, no tool
            review = self.spec["review"]
            snap = self.snapshot()
            rendered = prompts.render_read_only(review["prompt_version"], view_text=self.view["text"],
                                                snapshot=snap["text"])
            if rendered.get("arm") != self.spec["arm"]:
                raise BranchError(f"prompt {review['prompt_version']} is for {rendered.get('arm')}, not {self.spec['arm']}")
            binding = {k: rendered.get(k) for k in ("version", "sha256", "template_sha256", "worker_text_source",
                                                    "status", "arm")}
            binding.update(read_only=True, request=REVIEW_REQUEST_RULE, kept_output=KEPT_OUTPUT_RULE,
                           snapshot=snap["record"], view_ref=review["view_ref"], view_sha256=_sha(self.view["text"]),
                           view_kind=self.view.get("kind"))
            return {"mode": "review", "text": rendered["text"], "binding": binding, "tools": [],
                    "parser": _read_only_parser, "check": None, "token_limit": self.token_limit, "request": "head"}
        if self.spec["review"] is not None:
            review = self.spec["review"]
            mode, version = review["rewind_mode"], review["prompt_version"]
            snap = self.snapshot()
            rendered = prompts.render_review(version, mode=mode, view_text=self.view["text"], snapshot=snap["text"])
            if self.v3 and rendered.get("arm") != self.spec["arm"]:
                raise BranchError(f"prompt {version} is for {rendered.get('arm')}, not {self.spec['arm']}")
            tool = prompts.tool_schema(version, mode)
            if (tool.get("function") or {}).get("name") != T.REVIEW_TOOL_NAME:
                raise BranchError("the review tool schema does not name review_decision")
            binding = {k: rendered.get(k) for k in ("version", "sha256", "template_sha256", "worker_text_source",
                                                    "status", "mode", "arm")}
            binding.update(snapshot=snap["record"], view_ref=review["view_ref"], view_sha256=_sha(self.view["text"]),
                           tool_sha256=_sha(json.dumps(tool, sort_keys=True)))
            turn = {"mode": "review", "text": rendered["text"], "binding": binding, "tools": [tool],
                    "parser": self._review_parser(mode), "check": None, "token_limit": self.token_limit}
            if self.v3:  # Q24(a): system + task + the view; Q1: the turn's output is kept
                binding.update(request=REVIEW_REQUEST_RULE, kept_output=KEPT_OUTPUT_RULE,
                               view_kind=self.view.get("kind"))
                turn["request"] = "head"
            return turn
        if self.spec["note"] is not None:
            rp = self.spec["rewind"]["context"]
            rendered = prompts.render_note(self.spec["note"]["prompt_version"], n_undone_turns=self.s - rp,
                                           rewind_point=rp)
            binding = {k: rendered.get(k) for k in ("version", "sha256", "template_sha256", "status", "arm",
                                                    "n_undone_turns", "rewind_point")}
            return {"mode": "note", "text": rendered["text"], "binding": binding, "tools": [], "parser": _note_parser,
                    "check": _note_check, "token_limit": self.token_limit}
        return None

    def _review_parser(self, mode: str) -> Callable:
        _, study = _study()
        units = self.view["units"]

        def parse(tool_calls: list, *, format_error_template: str, template_kwargs: dict | None = None) -> list:
            if len(tool_calls) != 1:
                raise _review_error(f"it had {len(tool_calls)} tool calls.")
            call = tool_calls[0]
            if call.function.name != T.REVIEW_TOOL_NAME:
                raise _review_error(f"the tool {call.function.name!r} is not available in this turn.")
            try:
                args = json.loads(call.function.arguments)
            except (TypeError, ValueError):
                raise _review_error("its arguments are not valid JSON.") from None
            # Only the interface rules make a call invalid (and reach the worker). A rule-valid call the engine then
            # cannot compile (no branchable checkpoint at a point, a Ledger quote missing from its step) is an engine
            # or Ledger fault: compile() raises and the branch ends as a technical error, never charged to the worker.
            problems = study.validate_decision(args, mode, units)
            if problems:
                raise _review_error("; ".join(problems) + ".")
            return [{"tool_name": T.REVIEW_TOOL_NAME, "arguments": args, "tool_call_id": call.id}]
        return parse

    # ---- compile
    def decision_plan(self, args: dict, mode: str) -> dict:
        """The channels, edits and kept texts of one valid review decision (raises BranchError if it cannot apply)."""
        s = self.s
        if args.get("change_nothing"):
            points = dict.fromkeys(CHANNELS, s)
        elif mode == "per_channel":
            rw = args.get("rewind") or {}
            points = {ch: rw.get(ch) if rw.get(ch) is not None else s for ch in CHANNELS}
        else:
            p = args.get("rewind_to")
            points = dict.fromkeys(CHANNELS, p if p is not None else s)
        units = self.view["units"]
        drops = [] if args.get("change_nothing") else list(args.get("drop") or [])
        keeps = [] if args.get("change_nothing") else list(args.get("keep") or [])
        plan = self.channel_plan(points, drops=[(u, units[u]) for u in drops], extra_edits=[])
        c = points["context"]
        for u in keeps:
            if min(units[u]["steps"]) <= c:
                raise BranchError(f"kept unit {u} is not in a step the context rewind undoes")
        from ..ledger.render import label_text
        flat = self.view.get("kind") == "ledger_flat"  # kept items carry their labels as the flat Ledger shows them
        plan["kept"] = [{"unit": u, "kind": units[u]["kind"], "label": units[u].get("label"),
                         "text": units[u]["kept_text"] if "kept_text" in units[u]
                         else label_text(self.ledger, u, flat=flat)} for u in keeps]
        return plan

    def channel_plan(self, points: dict, *, drops: list, extra_edits: list) -> dict:
        """Donor checkpoints per channel, then the context edits on the context at its point (raises BranchError)."""
        donors = {ch: self.checkpoint_for(points[ch]) for ch in ("workspace", "runtime", "context")}
        if points["outputs"] is None or type(points["outputs"]) is not int or not 0 <= points["outputs"] <= self.s:
            raise BranchError(f"outputs point {points['outputs']!r} is outside 0..{self.s}")
        c = points["context"]
        edits, skipped = [], []
        out_edit = outputs_edit(points["outputs"], c)
        if out_edit is not None:
            edits.append(out_edit)
        for unit_id, unit in drops:
            first, last = unit["steps"]
            kind = unit["kind"]
            if first > c:
                skipped.append({"unit": unit_id, "reason": "already undone by the context rewind"})
            elif kind in ("step", "approach"):
                edits.append({"op": "drop_steps", "first": first, "last": min(last, c)})
            elif kind == "output":
                edits.append({"op": "drop_outputs", "steps": [first]})
            elif kind == "item":
                if self._hidden_by_context_limit(c, first, unit["quote"]):
                    skipped.append({"unit": unit_id, "reason": "already hidden by the context-limit rule"})
                    continue
                edits.append({"op": "drop_quote", "step": first, "quote": unit["quote"]})
            else:
                raise BranchError(f"unit {unit_id} has unknown kind {kind!r}")
        edits += [dict(e) for e in extra_edits]
        problems = validate_edits(self.messages_at(c), edits)
        if problems:
            raise BranchError("the edits do not apply to the context at point " + f"{c}: " + "; ".join(problems))
        return {"points": dict(points), "donors": donors, "edits": edits, "skipped_drops": skipped}

    def _hidden_by_context_limit(self, c: int, k: int, quote: str) -> bool:
        """The quote (from the Ledger, built on the full record) is not in step k of the context at point c because
        the context-limit rule hid the tool output that held it: the drop has nothing left to remove."""
        from ..execution.context_limit import is_masked
        from .context_edit import visible_texts
        messages = self.messages_at(c)
        steps = steps_of(messages)
        if not 1 <= k < len(steps):
            return False
        step = [messages[i] for i in steps[k]]
        return (not any(quote in t for m in step for t in visible_texts(m))
                and any(is_masked(m) for m in step))

    def compile(self, turn: dict | None) -> dict:
        spec, s = self.spec, self.s
        flags = {"mismatch": False, "failed_command": False, "invalid_call": False, "no_change": False}
        choice = None
        carried: list[tuple[str, str]] = []
        note_text = None
        forbidden: list[str] = []
        kept_output = None
        if turn is not None:
            forbidden = [t for t in turn["turn_texts"] + turn["response_texts"] if len(t) >= FORBIDDEN_MIN_CHARS]
        if self.v3 and turn is not None:
            # Q1: the accepted response stays (its texts are expected in the next request); the review message and
            # the view itself must still appear nowhere in it
            kept_output = self._kept_output(turn)
            if kept_output is not None:
                own = {turn["accepted_message"].get(k) for k in ("content", "reasoning", "reasoning_content")}
                forbidden = [t for t in forbidden if t not in own]
            if len(self.view["text"]) >= FORBIDDEN_MIN_CHARS:
                forbidden.append(self.view["text"])
        if self.read_only:
            valid = turn is not None and turn["valid"]
            flags["invalid_call"] = any(not c["valid"] for c in (turn or {}).get("calls", []))
            plan = self.channel_plan(dict.fromkeys(CHANNELS, s), drops=[], extra_edits=[])
            choice = {"version": "read-only-turn.v1", "branch_id": spec["branch_id"], "arm": spec["arm"],
                      "read_only": True, "calls": len(turn["calls"]) if turn else 0, "valid": valid,
                      "stopped": turn.get("stopped") if turn else None,
                      "invalid_reasons": [c["error"] for c in (turn or {}).get("calls", []) if not c["valid"]],
                      "kept_output": kept_output is not None, "view_ref": spec["review"]["view_ref"],
                      "prompt": turn["prompt"] if turn else None}
        elif spec["review"] is not None:
            mode = spec["review"]["rewind_mode"]
            valid = turn is not None and turn["valid"]
            if valid:
                args = turn["result"]["actions"][0]["arguments"]
            else:  # invalid twice, or the turn stopped (allowance or accounting): change nothing
                flags["invalid_call"] = any(not c["valid"] for c in (turn or {}).get("calls", []))
                args = ({"rewind": None} if mode == "per_channel" else {"rewind_to": None}) | {
                    "drop": [], "keep": [], "change_nothing": True}
            plan = self.decision_plan(args, mode)
            carried = [("kept", k["text"]) for k in plan["kept"]]
            choice = {"version": "choice.v1", "branch_id": spec["branch_id"], "arm": spec["arm"], "mode": mode,
                      "calls": len(turn["calls"]) if turn else 0, "valid": valid, "fallback_change_nothing": not valid,
                      "stopped": turn.get("stopped") if turn else None,
                      "invalid_reasons": [c["error"] for c in (turn or {}).get("calls", []) if not c["valid"]],
                      "decision": args, "kept": [{k: v for k, v in x.items() if k != "text"} for x in plan["kept"]],
                      "dropped": list(args.get("drop") or []), "skipped_drops": plan["skipped_drops"],
                      "compiled": {"points": plan["points"], "context_edits": plan["edits"]},
                      "view_ref": spec["review"]["view_ref"], "prompt": turn["prompt"] if turn else None}
        else:
            rw = spec["rewind"] or {}
            points = {ch: rw.get(ch) if rw.get(ch) is not None else s for ch in CHANNELS}
            plan = self.channel_plan(points, drops=[], extra_edits=spec["context_edits"])
            if spec["note"] is not None:
                if turn is not None and turn["valid"]:
                    note_text = turn["result"]["content"]
                    carried.append(("note", note_text))
                    forbidden = [t for t in forbidden if t != note_text]
                else:  # invalid twice, or the turn stopped: no note
                    flags["invalid_call"] = any(not c["valid"] for c in (turn or {}).get("calls", []))
            if spec["handback"]["recipe"] is not None:
                carried.append(("recipe", spec["handback"]["recipe"]))
            carried += [("item", text) for text in spec["handback"]["items"]]
        points = plan["points"]
        effective = {**points, "outputs": min(points["outputs"], points["context"])}
        captured = self.source.state.runtime_class == RuntimeClass.CAPTURED
        if not captured:  # R is not restored from any point (fresh or irrelevant): it takes no part in a mismatch
            effective.pop("runtime")
        flags["mismatch"] = len(set(effective.values())) > 1
        undone = any(v < s for v in effective.values())
        removed = bool(plan["edits"])
        if spec["review"] is not None and not self.read_only:  # nothing undone, removed or carried back
            flags["no_change"] = not (undone or removed or any(body for _, body in carried))
        # arm set v3: a hand-back follows every review turn, so the kept review output is followed by a user message
        send = undone or removed or bool(carried) or (self.v3 and spec["review"] is not None)
        handback = self._handback(undone or removed, carried) if send else None
        injection = composition = None
        if spec["injection"] is not None:
            injection, composition = self._injection(spec["injection"])
        donors = {"workspace": plan["donors"]["workspace"], "history": plan["donors"]["context"]}
        if captured:
            donors["runtime"] = plan["donors"]["runtime"]
        mismatches = []
        if effective["workspace"] < s:
            mismatches.append("workspace_from_earlier_point")
        if captured and effective["runtime"] != effective["workspace"]:
            mismatches.append("runtime_and_workspace_from_different_points")
        if effective["context"] < s:
            mismatches.append("context_from_earlier_point")
        if plan["edits"]:
            mismatches.append("context_edited")
        if injection and injection.get("messages"):
            mismatches.append("injected_steps")
        if composition is not None:
            mismatches.append("workspace_debris")
        record = {"engine": ENGINE_VERSION_V3 if self.v3 else ENGINE_VERSION, "spec_version": spec["version"],
                  "source_point": s, "points": points, "effective_points": effective,
                  "donors": donors, "context_edits": plan["edits"], "skipped_drops": plan["skipped_drops"],
                  "outputs_rule": OUTPUTS_RULE, "flags": dict(flags), "mismatch_rule": MISMATCH_RULE,
                  "undone": undone, "removed": removed, "grader_record": self.grader_record,
                  "handback": None if handback is None else {k: handback[k] for k in ("parts", "rule")} | {
                      "sha256": _sha(handback["text"]), "chars": len(handback["text"])},
                  "note": None if note_text is None else {"sha256": _sha(note_text), "chars": len(note_text)},
                  "injection": None if injection is None else injection["record"],
                  "forbidden_texts": len(forbidden)}
        if self.v3:
            record["review_output_kept"] = None if kept_output is None else {
                "rule": KEPT_OUTPUT_RULE, "sha256": _sha(json.dumps({k: v for k, v in kept_output.items()
                                                                     if k != "extra"}, sort_keys=True)),
                "content_chars": len(kept_output.get("content") or ""),
                "reasoning_chars": len(kept_output.get("reasoning") or kept_output.get("reasoning_content") or ""),
                "source_response_event_sequence": kept_output["extra"].get("response_event_sequence")}
            if spec["review"] is not None:
                record["review_request"] = REVIEW_REQUEST_RULE
            if spec["arm"] == "4_forced_rewind_to_p":
                record["stall_onset_point"] = spec["stall_onset_point"]
                record["forced_rewind_rule"] = FORCED_REWIND_RULE_V3
        return {"donors": donors, "context_edits": plan["edits"], "handback": handback, "injection": injection,
                "workspace_composition": composition, "choice": choice, "flags": flags, "note_text": note_text,
                "intended_mismatches": mismatches, "record": record, "forbidden": forbidden,
                "kept_output": kept_output}

    def _kept_output(self, turn: dict) -> dict | None:
        """The assistant message the review turn leaves in the continuing context (KEPT_OUTPUT_RULE), or None."""
        accepted = turn.get("accepted_message")
        if not turn.get("valid") or not isinstance(accepted, dict) or accepted.get("role") != "assistant":
            return None
        message = {k: deepcopy(v) for k, v in accepted.items() if k not in ("tool_calls", "extra")}
        text = accepted.get("content") if isinstance(accepted.get("content"), str) else ""
        for call in accepted.get("tool_calls") or []:
            decision = T.REVIEW_DECISION_KEPT.format(arguments=(call.get("function") or {}).get("arguments"))
            text = f"{text}\n\n{decision}" if text.strip() else decision
        message["content"] = text
        message["extra"] = {"review_output_kept": KEPT_OUTPUT_VERSION, "mode": turn["mode"],
                            "call_id": (accepted.get("extra") or {}).get("call_id"),
                            "response_event_sequence": (accepted.get("extra") or {}).get("response_event_sequence")}
        return message

    def _handback(self, generic: bool, carried: list[tuple[str, str]]) -> dict:
        lines = [T.HANDBACK_HEADER] + ([T.GENERIC_LINE] if generic else [])
        from .submission_feedback import SUBMISSION_RECORD, binding
        template = SUBMISSION_RECORD if self.submission_feedback is not None else T.GRADER_RECORD_LINE
        lines += [template.format(i=g["i"], k=g["k"], n=g["n"]) for g in self.grader_record]
        text = "\n".join(lines)
        items = [T.HANDBACK_ITEM.format(text=body) for _, body in carried if body]
        if items:
            text += "\n\n" + "\n\n".join(items)
        parts = {"header": T.HANDBACK_HEADER, "generic_line": generic,
                 "grader_record": [{k: g[k] for k in ("i", "k", "n")} for g in self.grader_record],
                 "carried": [{"kind": kind, "chars": len(body), "sha256": _sha(body)} for kind, body in carried if body]}
        if self.submission_feedback is not None:
            parts["submission_feedback"] = binding(self.submission_feedback)
        return {"text": text, "parts": parts, "rule": HANDBACK_RULE_V3 if self.v3 else HANDBACK_RULE}

    # ---- injection composition type
    def _injection(self, payload: dict) -> tuple[dict, dict | None]:
        if not isinstance(payload, dict) or payload.get("version") != INJECTION_VERSION:
            raise BranchError(f"injection payload must be {INJECTION_VERSION}")
        receiver = payload["receiver"]
        scope = self.source.scope
        version = {"requirement_version": scope.requirement_version, "evaluator_version": scope.evaluator_version}
        if (receiver.get("run_id"), receiver.get("checkpoint_id")) != (self.run_id, self.source.checkpoint_id):
            raise BranchError("the injection receiver is not this branch's source checkpoint")
        if receiver.get("point") not in (None, self.s) or receiver.get("task_id") != scope.task_id \
                or receiver.get("task_version") not in (None, version):
            raise BranchError("the injection receiver's point, task or version differs from the source")
        if f"injection:{payload['condition']}" != self.spec["arm"]:
            raise BranchError("the injection condition differs from the arm")
        record: dict[str, Any] = {"version": INJECTION_VERSION, "condition": payload["condition"],
                                  "channel": payload.get("channel"), "rules": payload.get("rules"),
                                  "record_notice": payload.get("record_notice"), "size": payload.get("size")}
        channel = payload.get("channel")
        if channel is None:
            return {"messages": [], "record": record | {"spliced": False}}, None
        donor = payload["donor"]
        same_store = Path(donor["store"]).expanduser().resolve() == Path(self.spec["source"]["store"]).expanduser().resolve()
        dstore = self.store if same_store else ReadOnlyArtifactStore(donor["store"])
        dstart = run_started(dstore, donor["run_id"])
        from ..resample_source import recovery_source_ok, store_start_lookup
        if not recovery_source_ok(dstart, store_start_lookup(dstore)):
            raise BranchError("the injection donor is not a natural run")
        if dstart.get("task") != self.start.get("task"):
            raise BranchError("the injection donor ran a different task record")
        if donor.get("same_run") and donor["run_id"] != self.run_id:
            raise BranchError("a same-run donor names another run")
        first, last = donor["steps"]
        cut = donor.get("cut") or {}
        if cut and (cut.get("after_point") != first - 1 or cut.get("through_point") != last):
            raise BranchError("the injection cut does not match the donor steps")
        if donor["run_id"] == self.run_id and last > self.s:
            raise BranchError("a same-run injection donor must be in the receiver's prefix")
        dpoints = point_checkpoints(dstore, donor["run_id"])
        record.update(donor_store=str(donor["store"]), donor_run_id=donor["run_id"], donor_steps=[first, last],
                      same_run=bool(donor.get("same_run")), selection=donor.get("selection"))

        def donor_checkpoint(k: int):
            if k not in dpoints:
                raise BranchError(f"the injection donor has no branchable checkpoint for point {k}")
            cp = dstore.get_checkpoint(dpoints[k])
            if (cp.scope.task_id, cp.scope.requirement_version, cp.scope.evaluator_version) != (
                    scope.task_id, scope.requirement_version, scope.evaluator_version):
                raise BranchError("the injection donor's task or version differs from the receiver")
            return cp

        if channel == "context":
            splice = payload.get("splice") or {}
            through = donor_checkpoint(last)
            # the donor steps as they were delivered: an output the context-limit rule later hid in the donor's
            # context is read back from the donor's record
            from ..ledger.steps import full_messages
            messages, read_back = full_messages(dstore, dstore.get_json(through.state.history))
            record["donor_outputs_read_back_from_record"] = len(read_back)
            steps = steps_of(messages)
            if not 1 <= first <= last <= len(steps) - 1:
                raise BranchError("the injection donor steps are outside its context")
            keep = ["role", "content", "tool_calls", "tool_call_id"] + (
                ["reasoning", "reasoning_content", "reasoning_details"] if splice.get("include_reasoning", True) else [])
            ns = splice.get("id_namespace") or f"inj-{_sha(json.dumps(donor, sort_keys=True))[:8]}"
            receiver_ids = {c.get("id") for m in self.history["messages"] for c in m.get("tool_calls") or []}
            renamed: dict[str, str] = {}
            out = []
            for k in range(first, last + 1):
                for i in steps[k]:
                    m = messages[i]
                    if m.get("role") == "exit":
                        raise BranchError("the injection donor steps end in an exit marker")
                    new = {key: deepcopy(m[key]) for key in keep if key in m}
                    for call in new.get("tool_calls") or []:
                        renamed[call["id"]] = call["id"] = f"{ns}-{call['id']}"
                    if new.get("tool_call_id") is not None:
                        new["tool_call_id"] = renamed.get(new["tool_call_id"], f"{ns}-{new['tool_call_id']}")
                    extra = {"injected": True, "condition": payload["condition"], "donor_step": k}
                    if (m.get("extra") or {}).get("interrupt_type") == "FormatError":
                        extra["interrupt_type"] = "FormatError"
                    new["extra"] = extra
                    if (new.get("role") == "user" and "interrupt_type" not in extra
                            and self.submission_feedback is not None
                            and NATIVE_FEEDBACK.fullmatch(str(new.get("content") or "").strip())):
                        # Injection test only (user 2026-11-02): a donor's count feedback line is replaced by the
                        # line this branch delivers after its own failed submissions, so both read the same.
                        from .submission_feedback import text as feedback_text
                        extra["feedback_rewritten_from"] = new["content"]
                        new["content"] = feedback_text(self.submission_feedback)
                    out.append(new)
            if set(renamed.values()) & receiver_ids:
                raise BranchError("renamed injected tool-call ids collide with the receiver's")
            record.update(spliced=True, donor_checkpoint_id=through.checkpoint_id, messages=len(out),
                          tool_call_ids_renamed=len(renamed), id_namespace=ns, position="after_last_step",
                          content="worker_visible_only", include_reasoning="reasoning" in keep,
                          messages_sha256=_sha(json.dumps(out, sort_keys=True)),
                          feedback_rewritten=sum(1 for m in out if "feedback_rewritten_from" in m["extra"]),
                          feedback_rewrite_rule=None if self.submission_feedback is None else
                              "injected donor count feedback replaced by the branch's bound submission_feedback text",
                          submission_feedback_messages=sum(1 for m in out if m.get("role") == "user"
                                                           and "interrupt_type" not in m["extra"]))
            return {"messages": out, "record": record}, None
        if channel == "workspace":
            debris = payload.get("debris") or {}
            before = donor_checkpoint(debris["donor_point_before"])
            after = donor_checkpoint(debris["donor_point_after"])
            archive, paths, candidates = compose_debris(self.store.get_bytes(self.source.state.workspace),
                                                        dstore.get_bytes(before.state.workspace),
                                                        dstore.get_bytes(after.state.workspace))
            composition = {"archive": archive, "rule": DEBRIS_RULE, "base_checkpoint_id": self.source.checkpoint_id,
                           "donor_checkpoints": [before.checkpoint_id, after.checkpoint_id], "paths": paths,
                           "candidates_in_receiver": sorted(set(candidates) - set(paths))}
            record.update(spliced=False, debris_paths=paths, debris_empty=not paths,
                          donor_checkpoints=[before.checkpoint_id, after.checkpoint_id])
            return {"messages": [], "record": record}, composition
        raise BranchError(f"unknown injection channel {channel!r}")


DEBRIS_RULE = ("workspace-debris/v1: the receiver's W tar plus every regular file created or changed (content or mode) "
               "in the donor's W between the two points whose path does not exist in the receiver's W, with any "
               "missing parent directories from the donor")


def compose_debris(receiver: bytes, before: bytes, after: bytes) -> tuple[bytes, list[str], list[str]]:
    norm = lambda n: (n[2:] if n.startswith("./") else n.lstrip("/")).rstrip("/")

    def members(data: bytes) -> dict[str, tuple[tarfile.TarInfo, bytes | None]]:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
            return {norm(m.name): (m, t.extractfile(m).read() if m.isfile() else None) for m in t.getmembers()}

    rec, bef, aft = members(receiver), members(before), members(after)
    candidates = sorted(n for n, (m, data) in aft.items() if m.isfile() and (
        n not in bef or not bef[n][0].isfile() or bef[n][1] != data or bef[n][0].mode != m.mode))
    paths = [n for n in candidates if n not in rec]
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as ot:
        for m, data in rec.values():
            ot.addfile(m, io.BytesIO(data) if data is not None else None)
        present = set(rec)
        for path in paths:
            parts = path.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                if parent not in present and parent in aft and aft[parent][0].isdir():
                    ot.addfile(aft[parent][0])
                    present.add(parent)
            m, data = aft[path]
            ot.addfile(m, io.BytesIO(data))
            present.add(path)
    return out.getvalue(), paths, candidates


# ----------------------------------------------------------------------- execution
def _load_task(start: dict, scratch: Path):
    from ..registry import read_registry
    from .v2_fixture import FIXTURES
    meta = (start.get("config") or {}).get("metadata") or {}
    registry_path, task_id = meta.get("registry"), meta.get("task_id") or (start.get("task") or {}).get("task_id")
    if not registry_path or not task_id:
        raise BranchError("the source run does not record its task registry and task id")
    registry = read_registry(Path(registry_path))
    if task_id not in registry:
        raise BranchError(f"{task_id} is not in {registry_path}")
    task = registry[task_id].load(scratch_root=scratch, purpose=start["config"]["purpose"])
    if meta.get("submission_fixture"):
        task = FIXTURES[meta["submission_fixture"]](task)
    return task


def live_model_factory(env_file: str = "~/.config/trajectory-retention/openrouter.env") -> Callable:
    """The worker's guarded model class (as configs/v2/natural.py); credentials are read, never printed."""
    path = Path(env_file).expanduser()
    if not os.environ.get("OPENROUTER_API_KEY") and path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.removeprefix("export ").split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    from ..execution.received_response_guard_v1 import ReceivedResponseGuardV1

    def factory(config: RunConfig):
        policy = config.worker_request_policy
        return ReceivedResponseGuardV1(model_name=config.model, model_settings=dict(config.model_settings),
            max_attempts=policy.max_attempts, backoff_seconds=policy.backoff_seconds,
            timeout_seconds=policy.timeout_seconds, usage_recovery_timeout_seconds=policy.usage_recovery_timeout_seconds)
    return factory


def _treatment(plan: BranchPlan, spec: dict, start: dict) -> TreatmentManifest:
    from ..observers._canonical import content_id
    from .treatments import RecoveryTarget
    src = plan.source
    runtime_policy = {RuntimeClass.CAPTURED: "restore_captured", RuntimeClass.IRRELEVANT: "irrelevant"}.get(
        src.state.runtime_class, "unsupported")
    stage = next((s for s in (start.get("task") or {}).get("stages") or () if s.get("stage_id") == src.scope.stage_id),
                 None)
    if stage is None or type(stage.get("instructions")) is not str:
        raise BranchError("the source task record has no stage instructions for the source checkpoint")
    text = stage["instructions"].encode("utf-8")
    target = RecoveryTarget(target_id=stage["target_id"], stage_id=stage["stage_id"],
                            requirement_version=stage["requirement_version"],
                            evaluator_version=stage["evaluator_version"],
                            prerequisites=tuple(stage.get("prerequisites") or ()),
                            instructions_ref=ArtifactRef(sha256=_sha(text), size_bytes=len(text), media_type="text/plain"))
    donors = {"workspace": src.checkpoint_id, "history": src.checkpoint_id}
    if runtime_policy == "restore_captured":
        donors["runtime"] = src.checkpoint_id
    metadata = {"arm_id": f"branch_v2:{spec['arm']}",
                "branch_v2": {"engine": _engine(spec), "branch_id": spec["branch_id"], "spec": spec,
                              "spec_sha256": _sha(json.dumps(spec, sort_keys=True))},
                "recovery_target": target.as_record(), "recovery_target_manifest_id": target.manifest_id(),
                "isolation": "own branch store; source run read-only; no grades, Ledger or other branches visible"}
    selection = "primary_s1" if spec["purpose"] in ("screen", "tuning", "main") else "secondary"
    return TreatmentManifest(
        treatment_id=content_id({"revision": "branch-v2-treatment.v1", "spec": spec}),
        source_checkpoint_id=src.checkpoint_id, target_id=src.scope.target_id, stage_id=src.scope.stage_id,
        selection_kind=selection, component_donors=donors, history_policy="native", runtime_policy=runtime_policy,
        worker_model=start["config"]["model"], generated_token_limit=plan.token_limit, repeat_id=str(spec["repeat"]),
        source_scope={"through_checkpoint_id": src.checkpoint_id}, metadata=metadata)


def context_limit_for(source_metadata: dict, model: str) -> dict | None:
    """A branch's context-limit binding: the source run's own; for a source that predates the rule, the window in
    configs/v2/settings.json when it is set for the same model; otherwise none (the runner refuses a live v2 run
    without one)."""
    if source_metadata.get("context_limit") is not None:
        return dict(source_metadata["context_limit"])
    settings = json.loads((CONFIGS / "settings.json").read_text())
    limit = settings.get("context_limit") or {}
    if settings.get("model") != model or limit.get("window_tokens") is None:
        return None
    return {"rule": limit["rule"], "window_tokens": limit["window_tokens"],
            "source": "configs/v2/settings.json (the source run predates the context-limit rule)"}


def _config(plan: BranchPlan, spec: dict, start: dict, run_id: str, *, store_path: Path) -> RunConfig:
    from ..mining.runner import V2_BRANCH
    from .v2_text import binding as text_binding
    src = start["config"]
    meta = src.get("metadata") or {}
    budget = spec["budget"]
    metadata = {"v2_run_policy": V2_BRANCH, "v2_text": text_binding(), "engine": _engine(spec),
                "branch_id": spec["branch_id"], "arm": spec["arm"], "purpose": spec["purpose"], "repeat": spec["repeat"],
                "seed": spec["seed"], "spec_sha256": _sha(json.dumps(spec, sort_keys=True)),
                "source_store": str(Path(spec["source"]["store"]).expanduser().resolve()),
                "source_run_id": plan.run_id, "source_checkpoint_id": plan.source.checkpoint_id,
                "source_point": plan.s, "branch_store": str(store_path), "registry": meta.get("registry"),
                "task_id": meta.get("task_id"), "fixture": bool(meta.get("fixture")),
                "token_cap": budget["new_token_cap"], "submission_cap": budget["submission_cap"],
                "token_cap_rule": "null new-token cap: only the $3 runaway-spend guard (engine allowance set beyond reach)"}
    limit = context_limit_for(meta, src["model"])
    if limit is not None:  # the same window as the source run, in every arm
        metadata["context_limit"] = limit
    if meta.get("instance_template"):
        metadata["instance_template"] = meta["instance_template"]
    if meta.get("unresolved_retry"):  # the same charged-retry rule as the source run, in every arm
        metadata["unresolved_retry"] = meta["unresolved_retry"]
    if "unresolved_retry" in spec:
        metadata["unresolved_retry"] = dict(spec["unresolved_retry"])
        metadata["unresolved_retry_provenance"] = {
            "source": "explicit branch spec", "source_run_setting": meta.get("unresolved_retry")}
    if meta.get("submission_stop"):
        metadata["submission_stop"] = meta["submission_stop"]
    if plan.submission_feedback is not None:
        metadata["submission_feedback"] = dict(plan.submission_feedback)
        metadata["submission_feedback_provenance"] = {
            "source": "explicit branch spec" if "submission_feedback" in spec else "source run",
            "source_run_setting": meta.get("submission_feedback"),
            "historical_worker_visible_feedback": "retained unchanged in source context"}
    if "submission_stop" in spec:
        metadata["submission_stop"] = dict(spec["submission_stop"])
        metadata["submission_stop_provenance"] = {
            "source": "explicit branch spec", "source_run_setting": meta.get("submission_stop"),
            "baseline": "fresh at intervention; historical source unchanged"}
    if meta.get("submission_fixture"):
        metadata["submission_fixture"] = meta["submission_fixture"]
    return RunConfig(run_id=run_id, campaign_id=f"branch-v2:{spec['purpose']}", purpose=src["purpose"],
                     model=src["model"], model_settings=dict(src["model_settings"]),
                     generated_token_limit=plan.token_limit, max_model_calls=budget["step_limit"],
                     cost_limit_usd=float(budget["cost_limit_usd"]), tool_timeout_seconds=src["tool_timeout_seconds"],
                     worker_request_policy=from_jsonable(WorkerRequestPolicy, src["worker_request_policy"]),
                     max_consecutive_format_errors=src["max_consecutive_format_errors"],
                     detector_settings=dict(src["detector_settings"]), repeat_id=f"r{spec['repeat']}",
                     stage_id=src.get("stage_id"), metadata=metadata)


def price_card(path: Path = PRICE_CARD) -> dict:
    return json.loads(Path(path).read_text())


def uncached_cost(records: list, prices: dict) -> tuple[float, int]:
    """(USD at the card's uncached prices over worker model attempts with known counts, attempts with unknown counts)."""
    total, unknown = 0.0, 0
    for r in records:
        if r.role != "worker" or r.kind != "model":
            continue
        u = r.usage
        if u is None or u.input_tokens is None or u.generated_tokens is None:
            unknown += 1
            continue
        total += u.input_tokens * prices["input"] / 1e6 + u.generated_tokens * prices["output"] / 1e6
    return total, unknown


def run_branch(spec: dict, *, out_dir: Path, model_factory: Callable | None = None) -> dict:
    """Run one v2 branch; write out_dir/result.json and return the same dict (brief-interfaces.md)."""
    t0 = time.monotonic()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempt = out.name
    base = {"branch_id": spec.get("branch_id") if isinstance(spec, dict) else None,
            "arm": spec.get("arm") if isinstance(spec, dict) else None,
            "repeat": spec.get("repeat") if isinstance(spec, dict) else None}
    store_path = out / "store"
    run_id = None
    # One scratch per invocation: a failed attempt keeps its scratch for inspection, and a rerun must not collide.
    scratch = SCRATCH / f"{_sha(str(out))[:16]}-{uuid.uuid4().hex[:8]}"
    try:
        check_spec(spec)
        run_id = f"branch-v2:{_safe(spec['branch_id'])}:{_safe(attempt)}"
        source = ReadOnlyArtifactStore(spec["source"]["store"])
        store = BranchStore(store_path, source)
        start = run_started(store, spec["source"]["run_id"])
        plan = BranchPlan(spec, store, start)
        config = _config(plan, spec, start, run_id, store_path=store_path)
        if model_factory is None:
            if config.model.startswith("scripted/"):
                raise BranchError("a scripted source run needs an explicit model_factory")
            model_factory = live_model_factory()
        scratch.mkdir(parents=True, exist_ok=True)
        task = _load_task(start, scratch / "task")
        from ..detector.core import DetectorConfig
        from ..mining.runner import MiningRunner
        stage_started = _first_payload(store, plan.run_id, "stage_started") or {}
        network = (stage_started.get("environment") or {}).get("network") or "none"
        if network in SERVICE_NETWORK_LABELS:
            # A service task records a label for the private bridge it builds itself; the runner was given "none".
            network = "none"
        runner = MiningRunner(model_factory=model_factory, worker_image=task.describe().environment_version,
                              detector_config=DetectorConfig(**config.detector_settings),
                              scratch_root=scratch / "runner", network=network)
        result = runner.run(task, config, store=store, treatment=_treatment(plan, spec, start), branch_plan=plan)
        res = _result(spec, store, run_id, result, task, base | {"store": str(store_path), "run_id": run_id})
    except Exception as exc:
        res = base | {"store": str(store_path), "run_id": run_id, "end_reason": "error", "solved": False,
                      "submissions": [], "best_share_passed": None, "final_share_passed": None, "choice": None,
                      "review_tokens": 0, "note_tokens": 0, "handback_tokens": 0, "new_tokens": 0, "steps": 0,
                      "cost_usd": 0.0, "cost_usd_uncached": None, "restore_seconds": 0.0,
                      "flags": {"mismatch": False, "failed_command": False, "invalid_call": False, "no_change": False},
                      "error": f"{type(exc).__name__}: {exc}"[:4000],
                      "details": {"traceback": traceback.format_exc()[-8000:], "error_kind": "technical",
                                  "error_kind_rule": ERROR_KIND_RULE}}
        if store_path.exists() and run_id is not None:
            try:  # whatever the branch recorded before failing (turn, costs) is still reported
                res = _partial(res, BranchStore(store_path, ReadOnlyArtifactStore(spec["source"]["store"])), run_id)
            except Exception as again:
                res["details"]["partial_error"] = f"{type(again).__name__}: {again}"
    res["wall_seconds"] = round(time.monotonic() - t0, 3)
    res["version"] = RESULT_VERSION
    _write_opus_input(spec, out, res)
    (out / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    if res["end_reason"] != "error":
        shutil.rmtree(scratch, ignore_errors=True)
    return res


def _events(store: Any, run_id: str) -> list:
    return list(store.iter_events(run_id))


def _lifecycle(events: list, transition: str) -> list[dict]:
    return [e.payload for e in events if e.kind == "lifecycle" and e.payload.get("transition") == transition]


def _share(task: Any, store: Any, evaluation_id: str | None) -> float | None:
    if not evaluation_id:
        return None
    from ..contracts import Evaluation
    ev = from_jsonable(Evaluation, store.get_record("evaluation", evaluation_id))
    verdict = task.native_submission_verdict(ev)
    share = verdict.get("share_passed")
    if share is None and ev.valid and ev.checks:
        share = sum(1 for c in ev.checks if c.status.value == "PASS") / len(ev.checks)
    return share


def _turn_tokens(events: list) -> dict[str, dict]:
    return {p["mode"]: p for p in _lifecycle(events, "intervention_turn_finished")}


def _result(spec: dict, store: Any, run_id: str, result: Any, task: Any, base: dict) -> dict:
    from ..accounting import CostLedger
    from ..ledger.templates import TOKENIZER, count_tokens
    events = _events(store, run_id)
    ready = (_lifecycle(events, "branch_v2_ready") or [{}])[0]
    offset = ready.get("call_offset") or 0
    submissions = []
    for v in _lifecycle(events, "native_submission_verdict"):
        failing = v.get("failing_count")
        share = v.get("share_passed")
        if share is None:
            share = _share(task, store, v.get("evaluation_id"))
        submissions.append({"step": (v.get("step_index") or 0) - offset, "failing_count": failing,
                            "failing_hash": v.get("criterion_id") or (_sha(json.dumps([])) if failing == 0 else None),
                            "share_passed": share, "grade_id": v.get("evaluation_id"),
                            "checkpoint_id": v.get("checkpoint_id"), "status": v.get("status")})
    termination = (result.details or {}).get("termination") or {}
    end_reason = END_REASONS.get(result.termination_reason, "error") if result.status == "completed" else "error"
    error_kind = _error_kind(end_reason, result)
    if error_kind == "worker_outcome":  # the worker ended the branch itself: an outcome of the arm, never retried
        end_reason = "worker_ended"
    compiled = (_lifecycle(events, "branch_v2_compiled") or [{}])[0]
    flags = dict(compiled.get("flags") or {"mismatch": False, "failed_command": False, "invalid_call": False,
                                          "no_change": False})
    flags["failed_command"] = _failed_command(events)
    turns = _turn_tokens(events)
    handback = next((e for e in events if e.kind == "handback_delivery"), None)
    handback_text = store.get_bytes(handback.delivered_ref).decode("utf-8") if handback else ""
    note_text = None
    choice = compiled.get("choice")
    if "note" in turns and compiled.get("note"):
        note_text = _note_text(store, events)
    snap = store.get_json(result.accounting_ref)
    worker = snap["roles"].get("worker", {})
    records = CostLedger(store).records(run_id)
    card = price_card()
    prices = card["models"].get(result_model(events))
    uncached, unknown = uncached_cost(records, prices) if prices else (None, None)
    final = None
    if termination.get("terminal_grade_valid"):
        final = _share(task, store, termination.get("terminal_evaluation_id"))
    terminal = next((e.payload for e in reversed(events) if e.kind == "terminal"), {})
    counters = terminal.get("counters") or {}
    restore = sum(e.usage.elapsed_seconds or 0.0 for e in events if e.kind == "operation" and e.usage
                  and e.payload.get("name") in RESTORE_OPERATIONS)
    first_request = next((e for e in events if e.kind == "model_request" and not e.payload.get("mode")), None)
    shares = [s["share_passed"] for s in submissions if s["share_passed"] is not None]
    allowance = (result.details or {}).get("worker_allowance") or {}
    res = base | {
        "end_reason": end_reason, "solved": end_reason == "passing_submission", "submissions": submissions,
        "best_share_passed": max(shares) if shares else (0.0 if not submissions and end_reason != "error" else None),
        "final_share_passed": final, "choice": choice,
        "review_tokens": (turns.get("review") or {}).get("totals", {}).get("known_generated_tokens", 0),
        "note_tokens": count_tokens(note_text) if note_text else 0,
        "handback_tokens": count_tokens(handback_text) if handback_text else 0,
        "new_tokens": allowance.get("known_generated_tokens_this_run", 0),
        "steps": counters.get("calls_this_run", 0), "cost_usd": round(worker.get("cost_usd", {}).get("known_sum", 0.0), 8),
        "cost_usd_uncached": None if uncached is None else round(uncached, 8),
        "restore_seconds": round(restore, 3), "flags": flags,
        "details": {"status": result.status, "termination_reason": result.termination_reason,
                    "end_detail": result.termination_reason if end_reason in ("error", "worker_ended") else None,
                    "error_kind": error_kind, "error_kind_rule": ERROR_KIND_RULE,
                    "source_point": compiled.get("source_point"), "points": compiled.get("points"),
                    "context_edits": compiled.get("context_edits"), "skipped_drops": compiled.get("skipped_drops"),
                    "grader_record": compiled.get("grader_record"), "handback_sha256": _sha(handback_text) if handback_text else None,
                    "injection": compiled.get("injection"), "spec_version": spec["version"],
                    "review_output_kept": compiled.get("review_output_kept"),
                    "stall_onset_point": compiled.get("stall_onset_point"), "note_turn_generated_tokens":
                        (turns.get("note") or {}).get("totals", {}).get("known_generated_tokens", 0),
                    "intervention_calls": ready.get("intervention_calls"), "unknown_cost_attempts":
                        worker.get("cost_usd", {}).get("unknown_count"), "uncached_unknown_attempts": unknown,
                    "price_card": {"version": card["version"], "model": result_model(events),
                                   "matched": prices is not None},
                    "tokenizer": TOKENIZER, "token_definitions": TOKEN_DEFINITIONS,
                    "failed_command_rule": FAILED_COMMAND_RULE, "mismatch_rule": MISMATCH_RULE,
                    "no_change_rule": NO_CHANGE_RULE,
                    "step_counting": "steps and submissions[].step count worker calls from the branch start and "
                                     "exclude the review or note turn; the step guard counts that turn too",
                    "setup_seconds": None if first_request is None else round(
                        _ts(first_request) - _ts(events[0]), 3),
                    "terminal_checkpoint_id": termination.get("terminal_checkpoint_id"),
                    "context_limit": context_limit_summary(events),
                    "grades": sum(1 for e in events if e.kind == "grade"),
                    "checkpoints": len(result.checkpoint_ids)}}
    if end_reason == "error":
        res["error"] = f"{result.status}: {result.termination_reason}"
    return res


def context_limit_summary(events: list) -> dict | None:
    """How often the context-limit rule hid outputs in this run (continuation and turns), from its log."""
    bound = next((e.payload["context_limit"] for e in events if e.kind == "lifecycle"
                  and e.payload.get("transition") == "worker_policy_bound" and e.payload.get("context_limit")), None)
    if bound is None:
        return None
    masked = _lifecycle(events, "context_limit_masked")
    return {"window_tokens": bound["window_tokens"], "masking_events": len(masked),
            "outputs_hidden": sum(len(p["outputs"]) for p in masked),
            "outputs_hidden_in_turns": sum(len(p["outputs"]) for p in masked if p.get("mode")),
            "exhausted": bool(_lifecycle(events, "context_window_exhausted"))}


# Environment network labels recorded by service tasks (terminalbench_v4_pilot_heatpump/services.py) for the per-run
# internal bridge they create; their natural runs were started with network "none".
SERVICE_NETWORK_LABELS = frozenset({"per-run-internal-isolated/v1"})
ERROR_KIND_RULE = ("error-kind/v2: a branch the worker itself ended outside the other end reasons (repeated format "
                   "errors or another worker exit, e.g. worker_error:RepeatedFormatError) has end_reason "
                   "'worker_ended' and error_kind 'worker_outcome': an unsolved result of the arm, never retried. "
                   "end_reason 'error' with error_kind 'technical' is every engine, provider, restore or accounting "
                   "failure; error_kind is null otherwise")


def _error_kind(end_reason: str, result: Any) -> str | None:
    if end_reason != "error":
        return None
    reason = result.termination_reason or ""
    if result.status == "completed" and reason.startswith(("worker_error:", "worker_exit:")):
        return "worker_outcome"
    return "technical"


def _partial(res: dict, store: Any, run_id: str) -> dict:
    events = _events(store, run_id)
    if not events:
        return res
    turns = _turn_tokens(events)
    compiled = (_lifecycle(events, "branch_v2_compiled") or [{}])[0]
    res["choice"] = compiled.get("choice")
    res["review_tokens"] = (turns.get("review") or {}).get("totals", {}).get("known_generated_tokens", 0)
    if compiled.get("flags"):
        res["flags"] = dict(compiled["flags"])
    # spend already recorded is reported, never zeroed (the driver's spend stop and the cost tables read it)
    from ..accounting import CostLedger
    records = [r for r in CostLedger(store).records(run_id) if r.role == "worker" and r.kind == "model"]
    known = [r.usage.cost_usd for r in records if r.usage is not None and r.usage.cost_usd is not None]
    generated = [r.usage.generated_tokens for r in records if r.usage is not None
                 and r.usage.generated_tokens is not None]
    res["cost_usd"] = round(sum(known), 8)
    res["new_tokens"] = sum(generated)
    res["steps"] = sum(1 for e in events if e.kind == "model_response" and not (e.payload or {}).get("mode"))
    card = price_card()["models"].get(result_model(events))
    if card is not None:
        uncached, _ = uncached_cost(records, card)
        res["cost_usd_uncached"] = round(uncached, 8)
    res["details"].update(events=len(events), unknown_cost_attempts=len(records) - len(known),
                          partial_rule="counted from the branch store's cost records and events after the failure")
    return res


def result_model(events: list) -> str | None:
    start = next((e.payload for e in events if e.kind == "lifecycle"
                  and e.payload.get("transition") == "run_started"), {})
    return (start.get("config") or {}).get("model")


def _ts(event: Any) -> float:
    import datetime as dt
    return dt.datetime.fromisoformat(event.timestamp).timestamp()


def _note_text(store: Any, events: list) -> str | None:
    response = next((e for e in reversed(events) if e.kind == "model_response" and e.payload.get("mode") == "note"
                     and e.payload.get("accepted")), None)
    if response is None:
        return None
    return (store.get_json(response.output_ref) or {}).get("content")


def _failed_command(events: list) -> bool:
    """FAILED_COMMAND_RULE over the first continuation step."""
    first = next((e for e in events if e.kind == "model_response" and not e.payload.get("mode")), None)
    if first is None:
        return False
    calls = {e.event_id for e in events if e.kind == "tool_call" and e.payload.get("model_call_id") == first.call_id}
    return any(e.payload.get("status") not in ("ok", None) for e in events
               if e.kind == "tool_result" and e.parent_event_id in calls)


def _write_opus_input(spec: dict, out: Path, res: dict) -> None:
    """With review.opus_record, the identical review request goes to a pending file for track S (never blocks)."""
    review = spec.get("review") if isinstance(spec, dict) else None
    if not review or not review.get("opus_record") or not res.get("run_id"):
        return
    try:
        store = BranchStore(Path(res["store"]), ReadOnlyArtifactStore(spec["source"]["store"]))
        request = next((e for e in store.iter_events(res["run_id"])
                        if e.kind == "model_request" and e.payload.get("mode") == "review"), None)
        if request is None:
            return
        body = json.loads(store.get_bytes(request.input_ref))
        (out / "opus-review-input.json").write_text(json.dumps({
            "version": "opus-review-input.v1", "branch_id": spec["branch_id"], "arm": spec["arm"],
            "view_ref": review["view_ref"], "prompt_version": review["prompt_version"],
            "rewind_mode": review["rewind_mode"], "request_sha256": _sha(store.get_bytes(request.input_ref)),
            "request": {"messages": body["messages"], "tools": body.get("tools", [])}}, indent=1) + "\n")
    except Exception as exc:  # record-only: a failure here never changes the branch
        res.setdefault("details", {})["opus_record_error"] = f"{type(exc).__name__}: {exc}"
