"""Deterministic stagnation-plus-recurrence detector.

Implements the original stagnation-plus-recurrence detector (contract on branch v1):
one effective generated-token stagnation clock since the latest strict new best
or genuine first-time frontier advancement, plus one qualifying recurrence
signal inside that window. Numerical settings are configuration and remain
explicitly unfrozen until calibration.

The detector is a pure function of an ordered stream of input records. Live
mining persists every input and output; offline replay of the same inputs must
produce byte-identical outputs. Outputs are research records only. Nothing in
this module can stop, extend, advise or otherwise change the natural worker.
"""

from __future__ import annotations

import hashlib
import math
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .laps import LapWindow, RULES as LAP_SIGNAL_RULES
from .laps_v2 import LapWindow as AttemptWindow, RULES as ATTEMPT_SIGNAL_RULES

LEGACY_RULE_VERSION = "stagnation-recurrence/v2-provisional"
PREVIOUS_RULE_VERSION = "stagnation-recurrence/v3-provisional"
RULE_VERSION = "stagnation-recurrence/v4-provisional"
LAP_RULE_VERSION = "stagnation-recurrence/v5-laps-provisional"
ATTEMPT_RULE_VERSION = "stagnation-recurrence/v6-attempts-provisional"
LAP_RULE_VERSIONS = (LAP_RULE_VERSION, ATTEMPT_RULE_VERSION)
from .stall_clock_v7 import STALL_CLOCK_RULE_VERSION, STALL_CLOCK_RULE_VERSIONS, STALL_CLOCK_V8_RULE_VERSION, STALL_CLOCK_V81_RULE_VERSION, SUBMISSION_VERDICT_RULE_VERSIONS, StallClockDetector  # noqa: E402
OBSERVABILITY_RULE_VERSIONS = (RULE_VERSION, *LAP_RULE_VERSIONS)

# Exact parsers, onset rules and region granularity are implementation choices
# to validate before freezing. The text is stored with every signal record.
SIGNAL_RULES = {
    "workspace_return": (
        "Exact W content-manifest hash sequence over relevant mutations in the window, "
        "starting from the anchor state. First completed A->B->A return; tightest cycle. "
        "Onset is the mutation leaving A."
    ),
    "edit_revert_repeat": (
        "File-level exact content states X->Y->X->Y on one path in the window "
        "(file granularity; region granularity not yet validated). Onset is the first X->Y mutation."
    ),
    "persistent_failure": (
        "Same normalized failure signature at N window grades, each consecutive member separated "
        "by at least one relevant mutation; unchanged regrades are skipped. Onset is the first "
        "relevant edit of the pattern: the edit producing the first member, or the edit leaving "
        "the window anchor when the anchor state is the first member."
    ),
    "repeated_region_failure": (
        "Chain of N window grades with identical failure signature and identical check-status "
        "digest whose inter-member relevant edits share at least one file path (file=region v1). "
        "Greedy chaining. Onset as for persistent_failure."
    ),
}
SIGNAL_RULES.update(LAP_SIGNAL_RULES)
SIGNAL_STRENGTH = {
    "workspace_return": "strong",
    "edit_revert_repeat": "strong",
    "persistent_failure": "strong",
    "repeated_region_failure": "weak",
}


@dataclass(frozen=True)
class DetectorConfig:
    stagnation_token_threshold: int
    persistent_failure_grades: int = 3
    weak_region_grades: int = 3
    calibration_status: str = "unfrozen"
    # stall-clock/v7 only: smallest completion-score gain counted as progress for this task
    # category (declared from measured grading noise before runs; 0 = any strict gain).
    score_min_gain: float = 0.0
    # Runner integration must opt into v3 only once it emits the bound signature
    # and mutation-policy evidence. Historical/default callers retain v2.
    rule_version: str = LEGACY_RULE_VERSION

    def __post_init__(self) -> None:
        if self.stagnation_token_threshold <= 0:
            raise ValueError("stagnation_token_threshold must be positive")
        if self.persistent_failure_grades < 2 or self.weak_region_grades < 2:
            raise ValueError("recurrence counts must be at least 2")
        if (type(self.score_min_gain) not in (int, float) or not math.isfinite(self.score_min_gain)
                or self.score_min_gain < 0):
            raise ValueError("score_min_gain must be a finite nonnegative number")
        if self.score_min_gain and self.rule_version not in STALL_CLOCK_RULE_VERSIONS:
            raise ValueError("score_min_gain applies to stall-clock rule versions only")
        if self.calibration_status not in ("unfrozen", "frozen"):
            raise ValueError("calibration_status must be 'unfrozen' or 'frozen'")
        if self.rule_version not in (*STALL_CLOCK_RULE_VERSIONS, *LAP_RULE_VERSIONS, RULE_VERSION, PREVIOUS_RULE_VERSION,
                                     LEGACY_RULE_VERSION):
            raise ValueError("Unsupported detector rule version")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if not record["score_min_gain"]:
            del record["score_min_gain"]  # Default omitted: pre-v7 settings/records stay byte-identical.
        return record


@dataclass
class _Anchor:
    checkpoint_id: str
    seq: int
    known_tokens: int
    unknown_calls: int

    def record(self) -> dict[str, Any]:
        return asdict(self)


class _Scope:
    def __init__(self, scope_id: str, stage_id: str | None, seq: int, *, attempt_laps: bool = False) -> None:
        self.attempt_laps = attempt_laps
        self.scope_id = scope_id
        self.stage_id = stage_id
        self.start_seq = seq
        self.baseline_done = False
        self.best_score: float | None = None
        self.best_checkpoint: str | None = None
        self.armed = True
        self.active_s: dict[str, Any] | None = None
        self.anchor: _Anchor | None = None
        self.best_anchor: _Anchor | None = None
        self.frontier_anchor: _Anchor | None = None
        self.observation_anchor: _Anchor | None = None
        self.observation_gap = False
        self.verified_complete = False
        self.reset_window()

    def reset_window(self) -> None:
        self.w_states: list[tuple[int, str]] = []
        self.file_states: dict[str, list[tuple[int, str]]] = {}
        self.window_grades: list[dict[str, Any]] = []
        self.relevant_mutations = 0
        self.edited_since_grade: set[str] = set()
        self.first_mutation_since_grade: int | None = None
        self.recognized: dict[str, str] = {}
        self.mutations: list[dict[str, Any]] = []
        self.laps = AttemptWindow() if self.attempt_laps else LapWindow()


class Detector:
    """Online detector. Feed input records in strictly increasing seq order."""

    def __init__(self, config: DetectorConfig, *, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self._last_seq: int | None = None
        self.known_tokens = 0
        self.unknown_calls = 0
        self.ever_reached: dict[str, str] = {}
        self.checkpoints: list[tuple[int, str, bool]] = []
        self.counters = {"P": 0, "F": 0, "L": 0, "S": 0}
        self.scope: _Scope | None = None
        self.seen_lap_attempts: dict[str, str] = {}

    # ------------------------------------------------------------------ input
    def feed(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        seq = item["seq"]
        if self._last_seq is not None and seq <= self._last_seq:
            raise ValueError(f"detector inputs must be strictly ordered: {seq} after {self._last_seq}")
        self._last_seq = seq
        handler = getattr(self, f"_on_{item['type']}", None)
        if handler is None:
            raise ValueError(f"unknown detector input type {item['type']!r}")
        outputs = handler(item)
        if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS and item["type"] == "grade":
            for output in outputs:
                if output.get("record") == "detector_decision":
                    self._add_eligibility_snapshot(item, output)
        return outputs

    def _add_eligibility_snapshot(self, grade: dict[str, Any], decision: dict[str, Any]) -> None:
        """Expose every gate even when an earlier one already prohibits S.

        This is descriptive output, not a second decision path. In particular,
        recurrence can be present while a missing observation forbids using it.
        Historical rule versions never gain fields during replay.
        """
        sc = self.scope
        assert sc is not None
        window = sc.window_grades
        stagnation = self._stagnation(sc)
        conditions = {
            "valid_grade": bool(grade["valid"]),
            "nonbaseline_checkpoint": decision["decision"] != "scope_baseline" and sc.baseline_done,
            "no_new_best": not decision.get("new_best", False),
            "no_first_time_frontier": not decision.get("frontier_advance", False),
            "episode_armed": decision.get("armed_before", sc.armed),
            "score_window_observed": bool(window) and all(g["score_observed"] for g in window),
            "frontier_window_observed": bool(window) and all(g["frontier_observed"] for g in window),
            "token_threshold_met": stagnation["threshold_met"],
            "qualifying_recurrence": bool(self._signals(sc)),
        }
        if self.config.rule_version in LAP_RULE_VERSIONS:
            conditions.update(branchable_checkpoint=self._checkpoint_valid(grade["checkpoint_id"]),
                              not_previously_verified_complete=not sc.verified_complete)
        decision["eligibility"] = conditions
        decision["failed_eligibility_conditions"] = [key for key, value in conditions.items() if not value]
        decision.setdefault("stagnation", stagnation)

    def _on_usage(self, u: dict[str, Any]) -> list[dict[str, Any]]:
        tokens = u["generated_tokens"]
        if tokens is None:
            self.unknown_calls += 1
        else:
            if tokens < 0:
                raise ValueError("generated_tokens cannot be negative")
            self.known_tokens += tokens
        return []

    def _on_lap(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        if self.config.rule_version not in LAP_RULE_VERSIONS:
            raise ValueError("Lap evidence requires the v5 detector")
        sc = self.scope
        if sc is None or item.get("scope_id") != sc.scope_id:
            raise ValueError("Lap received outside its scope")
        if not sc.baseline_done:
            return []
        identity = item.get("attempt_id")
        content = json.dumps({k: v for k, v in item.items() if k != "seq"}, sort_keys=True)
        if identity in self.seen_lap_attempts:
            if self.seen_lap_attempts[identity] != content:
                raise ValueError("Conflicting repeated lap attempt")
            return []
        if not isinstance(identity, str) or not identity:
            raise ValueError("Lap requires a source attempt identity")
        self.seen_lap_attempts[identity] = content
        signal = sc.laps.add(item, self.known_tokens)
        if signal is None:
            return []
        return [{"record": "recurrence_candidate", "seq": item["seq"],
                 "scope_id": sc.scope_id, "rule_version": self.config.rule_version,
                 "signal": signal, "pending_recurrence": True}]

    def _on_lap_boundary(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        if self.config.rule_version not in LAP_RULE_VERSIONS:
            raise ValueError("Lap boundaries require the v5 detector")
        sc = self.scope
        if sc is None or item.get("scope_id") != sc.scope_id:
            raise ValueError("Lap boundary received outside its scope")
        signal = sc.laps.signal
        if not sc.baseline_done or not sc.armed or sc.verified_complete or signal is None:
            return []
        phase = "threshold" if self._stagnation(sc)["threshold_met"] else "recognition"
        if phase in sc.laps.requested or (phase == "recognition" and "threshold" in sc.laps.requested):
            return []
        sc.laps.requested.add(phase)
        return [{"record": "grade_request", "seq": item["seq"], "scope_id": sc.scope_id,
                 "rule_version": self.config.rule_version, "reason": "lap_recurrence",
                 "phase": phase, "signal": signal, "known_tokens": self.known_tokens,
                 "note": "Private isolated grade at the completed tool-batch boundary; not an S or worker feedback."}]

    def _on_checkpoint(self, c: dict[str, Any]) -> list[dict[str, Any]]:
        self.checkpoints.append((c["seq"], c["checkpoint_id"], bool(c["valid"])))
        return []

    def _on_scope_start(self, s: dict[str, Any]) -> list[dict[str, Any]]:
        out = self._close_followup(s["seq"], "scope_closed")
        previous = self.scope.scope_id if self.scope else None
        self.scope = _Scope(s["scope_id"], s.get("stage_id"), s["seq"],
                            attempt_laps=self.config.rule_version == ATTEMPT_RULE_VERSION)
        out.append(
            {
                "record": "detector_scope",
                "event": "scope_initialized",
                "seq": s["seq"],
                "scope_id": s["scope_id"],
                "stage_id": s.get("stage_id"),
                "previous_scope_id": previous,
                "reason": s.get("reason"),
                "note": "Scope initialization is not progress, escape or new-best evidence.",
                "rule_version": self.config.rule_version,
            }
        )
        return out

    def _on_terminal(self, t: dict[str, Any]) -> list[dict[str, Any]]:
        return self._close_followup(t["seq"], f"trajectory_terminal:{t.get('reason')}")

    def _on_mutation(self, m: dict[str, Any]) -> list[dict[str, Any]]:
        sc = self.scope
        if sc is None or not sc.baseline_done or not m["relevant"]:
            return []
        sc.mutations.append({"seq": m["seq"], "failure_relevance": m.get("failure_relevance"),
                             "files": m.get("files", []), "event_sequence": m.get("event_sequence")})
        sc.relevant_mutations += 1
        if sc.first_mutation_since_grade is None:
            sc.first_mutation_since_grade = m["seq"]
        before, after = m.get("workspace_before"), m.get("workspace_after")
        if before is not None and before != sc.w_states[-1][1]:
            if self.config.rule_version == LEGACY_RULE_VERSION:
                sc.w_states.append((m["seq"], before))
            else:
                sc.w_states = [(m["seq"], before)]  # Unobserved gap is not a linked edit.
        if after is not None and after != sc.w_states[-1][1]:
            sc.w_states.append((m["seq"], after))
        for change in m.get("files", []):
            states = sc.file_states.setdefault(change["path"], [])
            if not states or change["before"] != states[-1][1]:
                if states and self.config.rule_version != LEGACY_RULE_VERSION:
                    states.clear()
                states.append((m["seq"], change["before"]))
            if change["after"] != states[-1][1]:
                states.append((m["seq"], change["after"]))
            sc.edited_since_grade.add(change["path"])
        return []

    def _on_grade(self, g: dict[str, Any]) -> list[dict[str, Any]]:
        sc = self.scope
        if sc is None or g["scope_id"] != sc.scope_id:
            raise ValueError("grade received outside its declared detector scope")
        checkpoint_valid = self._checkpoint_valid(g["checkpoint_id"])
        lap_confirmation = None
        if self.config.rule_version in LAP_RULE_VERSIONS:
            if g["valid"] and g.get("full_solve") is True:
                sc.verified_complete = True
            if sc.laps.signal is not None:
                sig = sc.laps.signal
                lap_confirmation = {"recognized_seq": sig["recognized_seq"],
                    "recognized_tokens": sig["recognized_tokens"],
                    "confirmation_delay_tokens": self.known_tokens - sig["recognized_tokens"],
                    "confirmation_delay_events": g["seq"] - sig["recognized_seq"],
                    "supporting_events": sig["evidence"]}
                sc.laps.requested.add("recognition")
                if self._stagnation(sc)["threshold_met"]:
                    sc.laps.requested.add("threshold")
        base = {
            "record": "detector_decision",
            "seq": g["seq"],
            "checkpoint_id": g["checkpoint_id"],
            "checkpoint_valid": checkpoint_valid,
            "scope_id": sc.scope_id,
            "stage_id": sc.stage_id,
            "rule_version": self.config.rule_version,
            "calibration_status": self.config.calibration_status,
        }
        certified_partial_frontier = (self.config.rule_version in OBSERVABILITY_RULE_VERSIONS
            and g.get("frontier_positive_evidence_valid") is True)
        if not g["valid"] and not certified_partial_frontier:
            if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
                sc.observation_gap = True
                if self.config.rule_version in LAP_RULE_VERSIONS:
                    sc.laps.clear_evidence()
            if sc.baseline_done:
                # Missing official observation is a barrier, not an absent
                # row. Otherwise A -> unmeasured B -> A falsely certifies a
                # continuously observed stagnant window.
                sc.window_grades.append(self._window_entry(sc, g))
                sc.edited_since_grade = set()
                sc.first_mutation_since_grade = None
            decision = dict(base, decision="invalid_grade", note="Invalid grading certifies neither progress nor detector conditions.")
            if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
                decision.update(blocked_reason="invalid_grade", blocked_reasons=["invalid_grade",
                    "frontier_observability_unknown", "score_observability_unknown"],
                    observation_gap=True, observation_recovered=False)
            return [decision]

        reach = None if g.get("reachable") is None else sorted(set(g["reachable"]))
        # A validated completed check can establish a positive frontier event
        # despite an unrelated missing grade suffix. It cannot rescue an
        # invalid scalar or certify that nothing else advanced.
        score = g.get("score") if g["valid"] else None
        frontier_observed = self._frontier_observed(g)
        trustworthy = bool(g["valid"]) and score is not None and frontier_observed
        observation_recovered = False
        if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
            observation_recovered = sc.baseline_done and sc.observation_gap and trustworthy
            if observation_recovered:
                sc.observation_anchor = self._anchor(g)
            sc.observation_gap = not trustworthy
            if self.config.rule_version in LAP_RULE_VERSIONS and not trustworthy:
                sc.laps.clear_evidence()

        if not sc.baseline_done:
            sc.baseline_done = True
            added = self._reach(reach, g["checkpoint_id"])
            sc.best_score = score
            sc.best_checkpoint = g["checkpoint_id"] if score is not None else None
            sc.anchor = sc.best_anchor = sc.frontier_anchor = self._anchor(g)
            if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS and trustworthy:
                sc.observation_anchor = self._anchor(g)
            self._start_window(sc, g)
            baseline = [
                dict(
                    base,
                    decision="scope_baseline",
                    baseline_reason="declared_scope_baseline" if g.get("scope_baseline") else "first_valid_grade_in_scope",
                    score=score,
                    capabilities_reached_at_baseline=added,
                    note="Baseline capabilities enter the trajectory-wide ledger without progress labels.",
                )
            ]
            if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
                baseline[0].update(frontier_observability_complete=frontier_observed,
                                   observation_gap=sc.observation_gap,
                                   blocked_reasons=["scope_baseline"])
            return baseline

        records: list[dict[str, Any]] = []
        new_best = score is not None and sc.best_score is not None and score > sc.best_score
        best_initialized = score is not None and sc.best_score is None
        newly = [c for c in reach if c not in self.ever_reached] if reach is not None else []
        frontier = bool(newly)
        stagnation = self._stagnation(sc)

        if new_best:
            self.counters["P"] += 1
            records.append(
                self._annotation("P", self.counters["P"], g["checkpoint_id"], g["seq"], sc,
                                 score=score, previous_best_score=sc.best_score,
                                 previous_best_checkpoint_id=sc.best_checkpoint)
            )
        if frontier:
            for cap in newly:
                self.ever_reached[cap] = g["checkpoint_id"]
            self.counters["F"] += 1
            records.append(
                self._annotation("F", self.counters["F"], g["checkpoint_id"], g["seq"], sc, newly_reached=newly)
            )
        if best_initialized:
            sc.best_score, sc.best_checkpoint = score, g["checkpoint_id"]

        if new_best or frontier:
            if sc.active_s is not None:
                s = sc.active_s
                escape = "both" if new_best and frontier else ("new_best" if new_best else "frontier_only")
                records.append(
                    self._annotation(
                        "N", s["s_index"], g["checkpoint_id"], g["seq"], sc,
                        linked_s_annotation_id=s["annotation_id"],
                        s_checkpoint_id=s["checkpoint_id"],
                        escape_type=escape,
                        best_score_before_s=s["best_score_before_s"],
                        score=score,
                        newly_reached=newly,
                        known_generated_tokens_since_s=self.known_tokens - s["known_tokens"],
                        unknown_usage_calls_since_s=self.unknown_calls - s["unknown_calls"],
                        seq_since_s=[s["seq"], g["seq"]],
                    )
                )
                sc.active_s = None
            sc.armed = True
            anchor = self._anchor(g)
            sc.anchor = anchor
            if new_best:
                sc.best_score, sc.best_checkpoint, sc.best_anchor = score, g["checkpoint_id"], anchor
            if frontier:
                sc.frontier_anchor = anchor
            self._start_window(sc, g)
            decision = dict(base, decision="progress", new_best=new_best, frontier_advance=frontier,
                            newly_reached=newly, score=score, stagnation_before_reset=stagnation,
                            armed=True)
            if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
                decision.update(frontier_observability_complete=frontier_observed,
                                observation_gap=sc.observation_gap,
                                observation_recovered=observation_recovered,
                                blocked_reasons=["genuine_progress"] + ([] if frontier_observed else ["frontier_observability_unknown"])
                                + ([] if score is not None else ["score_observability_unknown"]))
            return [decision] + records

        if observation_recovered:
            # A new observed interval is not a new episode. In particular,
            # preserve armed=False after S and preserve all historical maxima.
            self._start_window(sc, g)
            stagnation = self._stagnation(sc)

        if g["workspace_hash"] != sc.w_states[-1][1]:
            # Mutation events normally carry the hashes; a grade is a second
            # observation of the same W state and must not contradict it.
            if self.config.rule_version == LEGACY_RULE_VERSION:
                sc.w_states.append((g["seq"], g["workspace_hash"]))
            else:
                sc.w_states = [(g["seq"], g["workspace_hash"])]
        if not observation_recovered:
            sc.window_grades.append(self._window_entry(sc, g))
        sc.edited_since_grade = set()
        sc.first_mutation_since_grade = None
        signals = self._signals(sc)
        for signal in signals:
            if signal["signal"] in sc.recognized:
                continue
            self.counters["L"] += 1
            target = self._predecessor(signal["onset_seq"])
            ann = self._annotation(
                "L", self.counters["L"], target, g["seq"], sc,
                signal=signal["signal"], strength=signal["strength"], signal_rule=(ATTEMPT_SIGNAL_RULES.get(signal["signal"], SIGNAL_RULES.get(signal["signal"]))
                    if self.config.rule_version == ATTEMPT_RULE_VERSION else SIGNAL_RULES[signal["signal"]]),
                onset_seq=signal["onset_seq"], onset_interval=signal["onset_interval"],
                contributing=signal["evidence"],
                predecessor_status="selected" if target else "no_valid_predecessor",
                note="Mechanically observed recurrence onset; not a proven causal mistake.",
            )
            if self.config.rule_version != LEGACY_RULE_VERSION:
                ann["localization_reason"] = ("latest_valid_checkpoint_strictly_before_onset" if target else
                                               "no_valid_checkpoint_strictly_before_onset")
            sc.recognized[signal["signal"]] = ann["annotation_id"]
            records.append(ann)

        armed_before = sc.armed
        if not armed_before:
            blocked = "episode_not_escaped"
        elif any(not g["frontier_observed"] for g in sc.window_grades):
            blocked = "frontier_observability_unknown"
        elif score is None or sc.best_score is None or any(not x["score_observed"] for x in sc.window_grades):
            blocked = "score_observability_unknown"
        elif not stagnation["threshold_met"]:
            blocked = "clock_undetermined" if stagnation["clock_status"] == "undetermined" else "below_threshold"
        elif not signals:
            blocked = "no_recurrence_in_window"
        else:
            blocked = None
        blocked_reasons = []
        if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
            if not armed_before:
                blocked_reasons.append("episode_not_escaped")
            if any(not x["frontier_observed"] for x in sc.window_grades):
                blocked_reasons.append("frontier_observability_unknown")
            if score is None or sc.best_score is None or any(not x["score_observed"] for x in sc.window_grades):
                blocked_reasons.append("score_observability_unknown")
            if not stagnation["threshold_met"]:
                blocked_reasons.append("clock_undetermined" if stagnation["clock_status"] == "undetermined" else "below_threshold")
            if not signals:
                blocked_reasons.append("no_recurrence_in_window")
            if self.config.rule_version in LAP_RULE_VERSIONS:
                if not checkpoint_valid:
                    blocked_reasons.append("checkpoint_not_branchable")
                if sc.verified_complete:
                    blocked_reasons.append("target_previously_verified_complete")
            blocked = blocked_reasons[0] if blocked_reasons else None
        fired = blocked is None
        if fired:
            self.counters["S"] += 1
            index = self.counters["S"]
            s_ann = self._annotation(
                "S", index, g["checkpoint_id"], g["seq"], sc,
                stagnation=stagnation,
                signals=[s["signal"] for s in signals],
                l_annotation_ids=[sc.recognized[s["signal"]] for s in signals],
                best_score_before_s=sc.best_score,
                best_checkpoint_before_s=sc.best_checkpoint,
                anchor=self._effective_anchor(sc).record() if self._effective_anchor(sc) else None,
                checkpoint_valid=checkpoint_valid,
                trajectory_first_s=index == 1,
                note="Warning is a research annotation; mining continues unchanged.",
            )
            if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
                s_ann.update(progress_anchor=sc.anchor.record() if sc.anchor else None,
                             observation_anchor=sc.observation_anchor.record() if sc.observation_anchor else None)
            if self.config.rule_version in LAP_RULE_VERSIONS:
                s_ann.update(lap_confirmation=lap_confirmation, score=score,
                             frontier_state=reach, confirming_grade_seq=g["seq"])
            records.append(s_ann)
            sc.armed = False
            sc.active_s = {
                "s_index": index, "annotation_id": s_ann["annotation_id"], "checkpoint_id": g["checkpoint_id"],
                "seq": g["seq"], "known_tokens": self.known_tokens, "unknown_calls": self.unknown_calls,
                "best_score_before_s": sc.best_score,
            }
        decision = dict(base, decision="stall_warning" if fired else "no_warning", blocked_reason=blocked,
                        armed_before=armed_before, score=score, best_score=sc.best_score,
                        stagnation=stagnation, signals=signals, window_start_seq=sc.window_grades[0]["seq"])
        if self.config.rule_version != LEGACY_RULE_VERSION:
            decision["failure_relevance_observation"] = self._failure_relevance_observation(sc)
        if self.config.rule_version in OBSERVABILITY_RULE_VERSIONS:
            decision.update(blocked_reasons=blocked_reasons,
                            frontier_observability_complete=frontier_observed,
                            observation_gap=sc.observation_gap, observation_recovered=observation_recovered,
                            progress_anchor=sc.anchor.record() if sc.anchor else None,
                            observation_anchor=sc.observation_anchor.record() if sc.observation_anchor else None)
        return [decision] + records

    # --------------------------------------------------------------- helpers
    def _checkpoint_valid(self, checkpoint_id: str) -> bool:
        for _, cid, valid in self.checkpoints:
            if cid == checkpoint_id:
                return valid
        raise ValueError(f"grade references unrecorded checkpoint {checkpoint_id}")

    def _reach(self, reach: list[str] | None, checkpoint_id: str) -> list[str]:
        added = []
        for cap in reach or []:
            if cap not in self.ever_reached:
                self.ever_reached[cap] = checkpoint_id
                added.append(cap)
        return added

    def _anchor(self, g: dict[str, Any]) -> _Anchor:
        return _Anchor(g["checkpoint_id"], g["seq"], self.known_tokens, self.unknown_calls)

    def _effective_anchor(self, sc: _Scope) -> _Anchor | None:
        if self.config.rule_version not in OBSERVABILITY_RULE_VERSIONS:
            return sc.anchor
        anchors = [a for a in (sc.anchor, sc.observation_anchor) if a is not None]
        return max(anchors, key=lambda a: a.seq) if anchors else None

    def _frontier_observed(self, g: dict[str, Any]) -> bool:
        if self.config.rule_version not in OBSERVABILITY_RULE_VERSIONS:
            return g.get("reachable") is not None
        return g.get("reachable") is not None and g.get("frontier_observability_complete", True) is True

    def _start_window(self, sc: _Scope, g: dict[str, Any]) -> None:
        sc.reset_window()
        sc.w_states = [(g["seq"], g["workspace_hash"])]
        sc.window_grades = [self._window_entry(sc, g)]

    def _window_entry(self, sc: _Scope, g: dict[str, Any]) -> dict[str, Any]:
        return {
            "seq": g["seq"],
            "checkpoint_id": g["checkpoint_id"],
            "signature": g.get("failure_signature"),
            "status_digest": g.get("check_status_digest"),
            "mutations_cum": sc.relevant_mutations,
            "edited": sorted(sc.edited_since_grade),
            "first_mutation_seq": sc.first_mutation_since_grade,
            "frontier_observed": bool(g.get("valid")) and self._frontier_observed(g),
            "score_observed": bool(g.get("valid")) and g.get("score") is not None,
            "failed_check_ids": g.get("failed_check_ids"),
        }

    def _stagnation(self, sc: _Scope) -> dict[str, Any]:
        def since(anchor: _Anchor | None) -> dict[str, Any]:
            known = self.known_tokens - anchor.known_tokens if anchor else None
            unknown = self.unknown_calls - anchor.unknown_calls if anchor else None
            return {"anchor_checkpoint_id": anchor.checkpoint_id if anchor else None,
                    "known_generated_tokens": known, "unknown_usage_calls": unknown}

        effective = since(self._effective_anchor(sc))
        known, unknown = effective["known_generated_tokens"] or 0, effective["unknown_usage_calls"] or 0
        met = known >= self.config.stagnation_token_threshold
        status = "exact" if unknown == 0 else ("lower_bound" if met else "undetermined")
        return {
            "effective": effective,
            "since_new_best": since(sc.best_anchor),
            "since_frontier": since(sc.frontier_anchor),
            "threshold": self.config.stagnation_token_threshold,
            "threshold_met": met,
            "clock_status": status,
        }

    def _predecessor(self, onset_seq: int) -> str | None:
        target = None
        for seq, cid, valid in self.checkpoints:
            current_scope = (self.config.rule_version == LEGACY_RULE_VERSION or self.scope is not None
                             and seq >= self.scope.start_seq)
            if seq < onset_seq and valid and current_scope:
                target = cid
        return target

    def _signals(self, sc: _Scope) -> list[dict[str, Any]]:
        found = [
            s for s in (
                self._workspace_return(sc),
                self._edit_revert_repeat(sc),
                self._failure_chain(sc, weak=False),
                self._failure_chain(sc, weak=True),
            ) if s is not None
        ]
        if self.config.rule_version in LAP_RULE_VERSIONS and sc.laps.signal is not None:
            found.append(sc.laps.signal)
        return sorted(found, key=lambda s: (s["completion_seq"], s["signal"]))

    def _signal(self, name: str, onset: int, interval: list[int], completion: int, evidence: dict[str, Any]) -> dict[str, Any]:
        return {"signal": name, "strength": SIGNAL_STRENGTH[name], "onset_seq": onset,
                "onset_interval": interval, "completion_seq": completion, "evidence": evidence}

    def _workspace_return(self, sc: _Scope) -> dict[str, Any] | None:
        states = sc.w_states
        for k in range(2, len(states)):
            matches = [i for i in range(k - 1) if states[i][1] == states[k][1]]
            if matches:
                i = matches[-1]
                return self._signal(
                    "workspace_return", states[i + 1][0], [states[i][0], states[i + 1][0]], states[k][0],
                    {"state_seqs": [s[0] for s in states[i:k + 1]], "workspace_hashes": [s[1] for s in states[i:k + 1]]},
                )
        return None

    def _edit_revert_repeat(self, sc: _Scope) -> dict[str, Any] | None:
        best = None
        for path in sorted(sc.file_states):
            st = sc.file_states[path]
            for i in range(len(st) - 3):
                x, y = st[i][1], st[i + 1][1]
                if x != y and st[i + 2][1] == x and st[i + 3][1] == y:
                    cand = self._signal(
                        "edit_revert_repeat", st[i + 1][0], [st[i][0], st[i + 1][0]], st[i + 3][0],
                        {"path": path, "mutation_seqs": [s[0] for s in st[i + 1:i + 4]], "content_hashes": [x, y]},
                    )
                    if best is None or cand["completion_seq"] < best["completion_seq"]:
                        best = cand
                    break
        return best

    def _failure_chain(self, sc: _Scope, *, weak: bool) -> dict[str, Any] | None:
        if self.config.rule_version != LEGACY_RULE_VERSION:
            return self._bound_failure_chain(sc, weak=weak)
        need = self.config.weak_region_grades if weak else self.config.persistent_failure_grades
        grades = sc.window_grades
        chains: dict[tuple, dict[str, Any]] = {}
        for idx, g in enumerate(grades):
            if g["signature"] is None or (weak and g["status_digest"] is None):
                continue
            key = (g["signature"], g["status_digest"]) if weak else (g["signature"],)
            chain = chains.get(key)
            if chain is None:
                chains[key] = {"members": [idx], "common": None}
                continue
            last = chain["members"][-1]
            if g["mutations_cum"] <= grades[last]["mutations_cum"]:
                continue  # unchanged regrade: not an intervening relevant edit
            if weak:
                paths = set().union(*(set(grades[j]["edited"]) for j in range(last + 1, idx + 1)))
                common = paths if chain["common"] is None else chain["common"] & paths
                if not common:
                    chains[key] = {"members": [idx], "common": None}
                    continue
                chain["common"] = common
            chain["members"].append(idx)
            if len(chain["members"]) >= need:
                members = [grades[j] for j in chain["members"]]
                first_idx = chain["members"][0]
                # Onset is the first relevant edit of the pattern: the edit that produced
                # the first member, or, when the first member is the window anchor state,
                # the edit leaving that anchor toward the second member.
                onset_member = grades[chain["members"][1]] if first_idx == 0 else grades[first_idx]
                onset = onset_member["first_mutation_seq"]
                lower = grades[max(first_idx - 1, 0)]["seq"]
                if onset is None:  # edits recorded without seq: localize between grades only
                    onset = onset_member["seq"]
                name = "repeated_region_failure" if weak else "persistent_failure"
                evidence = {"failure_signature": g["signature"], "grade_seqs": [m["seq"] for m in members],
                            "checkpoint_ids": [m["checkpoint_id"] for m in members]}
                if weak:
                    evidence["status_digest"] = g["status_digest"]
                    evidence["common_paths"] = sorted(chain["common"])
                return self._signal(name, onset, [lower, onset_member["seq"]], g["seq"], evidence)
        return None

    @staticmethod
    def _failure_relevance_observation(sc: _Scope) -> dict[str, Any]:
        """Availability is evidence, not a fallback to 'all edits relevant'."""
        unavailable = [m["seq"] for m in sc.mutations
            if not isinstance(m["failure_relevance"], dict)
            or m["failure_relevance"].get("status") != "available"]
        return {"observed_mutation_seqs": [m["seq"] for m in sc.mutations],
                "unavailable_mutation_seqs": unavailable,
                "grades_without_failed_check_ids": [g["seq"] for g in sc.window_grades
                                                       if g["signature"] is not None and not g["failed_check_ids"]],
                "rule": "Only declared check-specific intervening mutation evidence supports failure recurrence."}

    @staticmethod
    def _failure_edits(sc: _Scope, checks: set[str], after_seq: int, through_seq: int) -> list[dict[str, Any]]:
        selected = []
        for mutation in sc.mutations:
            if not after_seq < mutation["seq"] <= through_seq:
                continue
            relevance = mutation["failure_relevance"]
            if (not isinstance(relevance, dict) or relevance.get("status") not in {"available", "partially_available"}
                    or not relevance.get("policy_id") or not relevance.get("version")):
                continue
            matched = checks.intersection(relevance.get("relevant_check_ids") or [])
            if not matched:
                continue
            matches = relevance.get("matches") or {}
            paths = set()
            for check in matched:
                match = matches.get(check, {})
                paths.update("W:" + path for path in match.get("workspace_paths", []))
                paths.update("R:" + path for path in match.get("runtime_paths", []))
                if match.get("mounts_changed"):
                    paths.add("R:<mounted_state>")
            selected.append({"seq": mutation["seq"], "event_sequence": mutation["event_sequence"],
                             "matched_check_ids": sorted(matched), "paths": sorted(paths),
                             "policy": {key: relevance.get(key) for key in (
                                 "rule_version", "policy_id", "version", "task_id", "task_version")}})
        return selected

    def _bound_failure_chain(self, sc: _Scope, *, weak: bool) -> dict[str, Any] | None:
        """Same normalized failure plus independently declared relevant edits.

        A changed file alone establishes neither failure relevance nor a shared
        region. The raw mapping and source mutation sequence accompany the signal.
        """
        need = self.config.weak_region_grades if weak else self.config.persistent_failure_grades
        grades = sc.window_grades
        chains: dict[tuple, dict[str, Any]] = {}
        for index, grade in enumerate(grades):
            if not grade["signature"] or not grade["failed_check_ids"] or (weak and not grade["status_digest"]):
                continue
            checks = set(grade["failed_check_ids"])
            key = (grade["signature"], grade["status_digest"] if weak else None, tuple(sorted(checks)))
            chain = chains.get(key)
            if chain is None:
                chains[key] = {"members": [index], "edits": [], "common": None}
                continue
            last = grades[chain["members"][-1]]
            edits = self._failure_edits(sc, checks, last["seq"], grade["seq"])
            if not edits:
                continue
            paths = set().union(*(set(edit["paths"]) for edit in edits))
            common = paths if chain["common"] is None else chain["common"] & paths
            if weak and not common:
                chains[key] = {"members": [index], "edits": [], "common": None}
                continue
            chain["members"].append(index)
            chain["edits"].append(edits)
            chain["common"] = common
            if len(chain["members"]) < need:
                continue
            members = [grades[member] for member in chain["members"]]
            first_index = chain["members"][0]
            previous_seq = grades[first_index - 1]["seq"] if first_index else grades[0]["seq"]
            incoming = self._failure_edits(sc, checks, previous_seq, members[0]["seq"])
            onset_edit = incoming[0] if incoming else chain["edits"][0][0]
            onset_upper = members[0]["seq"] if incoming else members[1]["seq"]
            evidence = {"failure_signature": grade["signature"], "failed_check_ids": sorted(checks),
                        "grade_seqs": [member["seq"] for member in members],
                        "checkpoint_ids": [member["checkpoint_id"] for member in members],
                        "intervening_failure_relevant_edits": chain["edits"],
                        "onset_edit": onset_edit}
            if weak:
                evidence["status_digest"] = grade["status_digest"]
                evidence["common_paths"] = sorted(common)
            return self._signal("repeated_region_failure" if weak else "persistent_failure",
                                onset_edit["seq"], [previous_seq, onset_upper], grade["seq"], evidence)
        return None

    def _annotation(self, label: str, index: int, target: str | None, assigned_seq: int, sc: _Scope, **details: Any) -> dict[str, Any]:
        return {
            "record": "annotation",
            "annotation_id": f"{self.run_id}/{label}{index}",
            "label": label,
            "label_index": index,
            "target_checkpoint_id": target,
            "assigned_at_seq": assigned_seq,
            "evidence_cutoff_seq": assigned_seq,
            "scope_id": sc.scope_id,
            "stage_id": sc.stage_id,
            "rule_version": self.config.rule_version,
            "calibration_status": self.config.calibration_status,
            **details,
        }

    def _close_followup(self, seq: int, reason: str) -> list[dict[str, Any]]:
        sc = self.scope
        if sc is None or sc.active_s is None:
            return []
        s = sc.active_s
        sc.active_s = None
        return [
            {
                "record": "followup_closed",
                "seq": seq,
                "s_annotation_id": s["annotation_id"],
                "scope_id": sc.scope_id,
                "reason": reason,
                "escaped": False,
                "known_generated_tokens_observed_since_s": self.known_tokens - s["known_tokens"],
                "unknown_usage_calls_since_s": self.unknown_calls - s["unknown_calls"],
                "note": "No N label: no natural escape observed within compatible follow-up; censored, not proven failure.",
            }
        ]


def make_detector(config: DetectorConfig, *, run_id: str) -> "Detector | StallClockDetector":
    """Select the detector implementation named by the rule version."""
    if config.rule_version in STALL_CLOCK_RULE_VERSIONS:
        return StallClockDetector(config, run_id=run_id)
    return Detector(config, run_id=run_id)


def replay(inputs: Iterable[dict[str, Any]], config: DetectorConfig, *, run_id: str) -> list[dict[str, Any]]:
    detector = make_detector(config, run_id=run_id)
    outputs: list[dict[str, Any]] = []
    for item in inputs:
        outputs.extend(detector.feed(item))
    return outputs


def canonical_json(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def records_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(records).encode()).hexdigest()
