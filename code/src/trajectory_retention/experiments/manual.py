"""Manual, source-bound experiment configuration (manual-experiment.v1).

A researcher declares one decision checkpoint and, per state component, an
explicit legal donor instead of picking a predefined arm. The builder resolves
the declaration against the immutable store and emits the existing
``TreatmentManifest`` vocabulary so the shared runner, common validator and
exports consume it unchanged. It never runs a worker, never regenerates T, and
never invents defaults for recovery settings, targets or packets.

Legal donor set: saved checkpoints of the same natural source run, at or
before the decision checkpoint, in the same task/target/requirement/evaluator
scope; restorable components (W, R, H) must also be branchable. Branch
checkpoints, later checkpoints, other runs and private grading never enter a
manifest. Native Continue means the exact decision-checkpoint W/R/H with no
packet; T injection is an explicit intervention. O overlaps H: an observation
mask is compiled from the selected history donor's actual retained,
provider-visible deliveries through root's mask helper: the primary (A, S1]
window, the explicit all-history policies, or the manual-secondary checkpoint
window (after C_i, through C_j or the H cut] that the helper itself validates.

T may come from the frozen primary version at any eligible prefix checkpoint,
as a frozen packet or as a deterministic caller-selected view (types and/or
entry IDs) rendered from that exact version with the shared renderer and no
regeneration or refill. Study B built-in arms keep their shared-object rule.

An external research packet (``{"kind": "research_packet", "packet_id", "label"}``)
resolves an immutable ``research_packet`` record registered beforehand
(research_packets.py): exact worker-visible bytes, kind label, source run and
checkpoint, source/version identities, construction metadata and measured token
count. It must be bound to the same natural run at or before the decision
checkpoint and enters through the same packet injection path and placement as
every other packet; nothing is regenerated, re-extracted or refilled here.

Dispatch: ``metadata["manual_experiment"]["version"] == MANUAL_VERSION``.
``metadata.manual_experiment.execution_support`` names each predefined-arm rule
an explicit declaration replaced (``declared_manual_intervention``). Legality
is conferred only by the deterministic structural rebuild plus the common
validator; no validator message is ever suppressed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from ..contracts import (ArtifactRef, ArtifactStoreProtocol, Checkpoint, RuntimeClass, TreatmentManifest,
                         WORKER_OPERATIONAL_FIELDS, canonical_json, from_jsonable, to_jsonable,
                         worker_operational_settings)
from ..observers import artifacts
from ..observers._canonical import content_id
from ..observers.pipeline import FrozenPackets
from ..observers.render import approx_tokens, render_prose, render_typed
from ..observers.tstore import ENTRY_TYPES, TVersion
from .observation_mask import MASK_PLACEHOLDER, MaskError, build_observation_mask
from .research_packets import PACKET_KIND as RESEARCH_PACKET_KIND, ResearchPacketError, load_research_packet
from .selection import first_stall_annotation
from .treatments import (HISTORY_POLICIES, ORIENTATION_TEXT, BranchSpec, RecoveryTarget, TreatmentError, _packet,
                         _validate_manifest_common, manifest_digest, recovery_settings_known, select_donor)

MANUAL_VERSION = "manual-experiment.v1"
TREATMENT_IDENTITY_REVISION = "manual-treatment-identity.v2"
SELECTION_KINDS = ("primary_s1", "engineering_checkpoint", "secondary")
RUNTIME_POLICIES = ("follow_workspace", "restore_captured", "irrelevant", "discard_captured")
DECISION = "decision"
WORKSPACE = "workspace"
SELECTED_PACKET_KIND = "t_selected"
SELECTION_FILTER_REVISION = "manual-selection/v1"
# Root's mask helper policy for a manual-secondary checkpoint window (after C_i,
# through C_j or the H cut]; the helper validates delivery membership and cuts.
MANUAL_WINDOW_POLICY = "o-mask-manual-checkpoint-window.v1"
SECONDARY_DECLARATIONS = ("cross_session_information", "history_donor_differs_from_decision",
                          "runtime_donor_differs_from_workspace", "history_from_other_session",
                          "packet_on_native_history", "packet_from_earlier_checkpoint", "manual_observation_cutoff")
PACKET_AFTER_FRESH_SCAFFOLD = "after_target_and_orientation (provisional, unratified)"
PACKET_AFTER_NATIVE_HISTORY = "after_restored_native_history"
DECLARED = "declared_manual_intervention"
# Operational fields the shared runner requires to equal the natural source.
FIXED_TO_SOURCE = ("tool_timeout_seconds", "detector_settings", "worker_request_policy", "max_consecutive_format_errors")
VALID_T_STATUSES = ("valid", "valid_with_rejections")


class ManualExperimentError(ValueError):
    pass


@dataclass(frozen=True)
class ManualExperimentSpec:
    """Researcher declaration. Every field is explicit; unknown keys are rejected."""

    run_id: str
    decision_checkpoint_id: str
    selection_kind: str
    label: str
    recovery: Mapping[str, Any]
    workspace_donor: str = DECISION
    runtime: Mapping[str, Any] = field(default_factory=lambda: {"policy": "follow_workspace"})
    history: Mapping[str, Any] = field(default_factory=lambda: {"policy": "native", "donor": DECISION})
    observation_mask: Mapping[str, Any] | None = None
    packet: Mapping[str, Any] = field(default_factory=lambda: {"kind": "none"})
    declared_secondary: tuple[str, ...] = ()
    intended_mismatches: tuple[str, ...] = ()
    notes: str = ""
    version: str = MANUAL_VERSION
    # Selective revert (user C9, 2026-09-27): W = decision W with `paths` restored to their content at `restore_from`
    # (absent there -> deleted). Omitted from the record when unused, so earlier spec digests are unchanged.
    workspace_edit: Mapping[str, Any] | None = None
    # Edited history (history.policy "edited"): the history donor's context with these context edits applied
    # (experiments/context_edit.py). Omitted from the record when unused, so earlier spec digests are unchanged.
    context_edits: tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if self.version != MANUAL_VERSION:
            raise ManualExperimentError(f"unsupported manual experiment version {self.version!r}")
        for name in ("run_id", "decision_checkpoint_id", "label"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ManualExperimentError(f"{name} must be a non-empty string")
        if self.selection_kind not in SELECTION_KINDS:
            raise ManualExperimentError(f"selection_kind must be one of {SELECTION_KINDS}")
        if self.workspace_edit is not None:
            _check_workspace_edit(self)
        if type(self.workspace_donor) is not str or not self.workspace_donor:
            raise ManualExperimentError("workspace_donor must be 'decision' or a checkpoint id")
        for name in ("runtime", "history", "packet", "recovery"):
            if not isinstance(getattr(self, name), Mapping):
                raise ManualExperimentError(f"{name} must be an object")
        if self.runtime.get("policy", "follow_workspace") not in RUNTIME_POLICIES:
            raise ManualExperimentError(f"runtime.policy must be one of {RUNTIME_POLICIES}")
        if set(self.runtime) - {"policy", "donor"}:
            raise ManualExperimentError("runtime accepts only policy and donor")
        if self.history.get("policy") not in HISTORY_POLICIES:
            raise ManualExperimentError(f"history.policy must be one of {HISTORY_POLICIES}")
        if set(self.history) - {"policy", "donor"}:
            raise ManualExperimentError("history accepts only policy and donor")
        if self.history["policy"] == "fresh" and "donor" in self.history:
            raise ManualExperimentError("fresh history has no donor")
        if self.history["policy"] == "edited":
            if (not isinstance(self.context_edits, (list, tuple)) or not self.context_edits
                    or not all(isinstance(e, Mapping) for e in self.context_edits)):
                raise ManualExperimentError("edited history requires a non-empty context_edits list")
        elif self.context_edits is not None:
            raise ManualExperimentError("context_edits are only meaningful with history.policy == 'edited'")
        self._check_mask()
        self._check_packet()
        if set(self.recovery) != {"generated_token_limit", "repeat_id", "settings"}:
            raise ManualExperimentError("recovery requires exactly generated_token_limit, repeat_id and settings")
        limit = self.recovery["generated_token_limit"]
        if type(limit) is not int or limit <= 0:
            raise ManualExperimentError("recovery.generated_token_limit must be a positive integer")
        if type(self.recovery["repeat_id"]) is not str or not self.recovery["repeat_id"]:
            raise ManualExperimentError("recovery.repeat_id must be a non-empty string")
        if self.recovery["settings"] != "source" and not isinstance(self.recovery["settings"], Mapping):
            raise ManualExperimentError("recovery.settings must be 'source' or an explicit operational settings object")
        unknown = set(self.declared_secondary) - set(SECONDARY_DECLARATIONS)
        if unknown:
            raise ManualExperimentError(f"unknown secondary declarations: {sorted(unknown)}")
        if any(type(m) is not str or not m for m in self.intended_mismatches):
            raise ManualExperimentError("intended_mismatches must be non-empty strings")

    def _check_mask(self) -> None:
        mask = self.observation_mask
        if self.history["policy"] == "o_masked" and not isinstance(mask, Mapping):
            raise ManualExperimentError("o_masked history requires an explicit observation_mask policy")
        if self.history["policy"] != "o_masked" and mask is not None:
            raise ManualExperimentError("observation_mask is only meaningful with history.policy == 'o_masked'")
        if mask is None:
            return
        if type(mask.get("policy")) is not str:
            raise ManualExperimentError("observation_mask.policy must be a mask policy id")
        if mask["policy"] == MANUAL_WINDOW_POLICY:
            if (set(mask) - {"policy", "after_checkpoint_id", "through_checkpoint_id"} or "after_checkpoint_id" not in mask
                    or any(type(mask[k]) is not str or not mask[k] for k in mask)):
                raise ManualExperimentError("manual observation window requires policy, after_checkpoint_id and an optional "
                                            "through_checkpoint_id (checkpoint ids)")
        elif set(mask) != {"policy"}:
            raise ManualExperimentError("observation_mask must be {'policy': <mask policy id>}; window fields are only for "
                                        f"{MANUAL_WINDOW_POLICY}")

    def _check_packet(self) -> None:
        p = self.packet
        kind = p.get("kind")
        if type(kind) is not str or not kind:
            raise ManualExperimentError("packet.kind is required (use 'none' for no packet)")
        if kind == "none":
            if set(p) != {"kind"}:
                raise ManualExperimentError("packet 'none' accepts no other fields")
            return
        if kind == SELECTED_PACKET_KIND:
            allowed = {"kind", "source_checkpoint_id", "version_id", "chain_namespace", "types", "entry_ids", "render"}
            if set(p) - allowed or "source_checkpoint_id" not in p or "render" not in p:
                raise ManualExperimentError("t_selected requires source_checkpoint_id and render, optional version_id, "
                                            "chain_namespace, types, entry_ids")
            if "chain_namespace" in p and (type(p["chain_namespace"]) is not str or not p["chain_namespace"]):
                raise ManualExperimentError("t_selected.chain_namespace must be the retained primary chain namespace")
            if type(p["source_checkpoint_id"]) is not str or not p["source_checkpoint_id"]:
                raise ManualExperimentError("t_selected.source_checkpoint_id must be 'decision' or a checkpoint id")
            if p["render"] not in ("typed", "prose"):
                raise ManualExperimentError("t_selected.render must be 'typed' or 'prose'")
            if "version_id" in p and (type(p["version_id"]) is not str or not p["version_id"]):
                raise ManualExperimentError("t_selected.version_id must be a T version id")
            if "types" not in p and "entry_ids" not in p:
                raise ManualExperimentError("t_selected requires types and/or entry_ids; an unfiltered view is a frozen t_typed/t_prose packet")
            for name in ("types", "entry_ids"):
                if name in p and (not isinstance(p[name], (list, tuple)) or not p[name]
                                  or any(type(x) is not str or not x for x in p[name]) or len(set(p[name])) != len(p[name])):
                    raise ManualExperimentError(f"t_selected.{name} must be a non-empty list of unique strings")
            if "types" in p and set(p["types"]) - set(ENTRY_TYPES):
                raise ManualExperimentError(f"t_selected.types must be among {ENTRY_TYPES}")
            return
        if kind == RESEARCH_PACKET_KIND:
            if set(p) != {"kind", "packet_id", "label"} or any(type(p[k]) is not str or not p[k] for k in ("packet_id", "label")):
                raise ManualExperimentError("research_packet requires exactly kind, packet_id and label (non-empty strings)")
            return
        if "entries" in p or "entry_ids" in p:
            raise ManualExperimentError("entry-level selection uses kind 't_selected'; frozen packet kinds take only freeze_id")
        if set(p) != {"kind", "freeze_id"} or type(p["freeze_id"]) is not str:
            raise ManualExperimentError("a frozen packet requires exactly kind and the frozen packet freeze_id")

    def as_record(self) -> dict[str, Any]:
        rec = to_jsonable(self)
        for optional in ("workspace_edit", "context_edits"):
            if rec.get(optional) is None:
                rec.pop(optional, None)
        return rec

    def digest(self) -> str:
        return content_id(self.as_record())


SELECTIVE_REVERT = "selective_revert"
SELECTIVE_REVERT_MISMATCH = "workspace_selectively_reverted"
SELECTIVE_REVERT_RULE = "selective-revert/v1: decision W tar minus declared paths, plus those paths' members from the restore_from W tar"


def _check_workspace_edit(spec: "ManualExperimentSpec") -> None:
    e = spec.workspace_edit
    if not isinstance(e, Mapping) or e.get("policy") != SELECTIVE_REVERT or set(e) - {"policy", "restore_from", "paths", "provenance"}:
        raise ManualExperimentError("workspace_edit must be {policy: 'selective_revert', restore_from, paths[, provenance]}")
    if type(e.get("restore_from")) is not str or not e["restore_from"]:
        raise ManualExperimentError("workspace_edit.restore_from must be a checkpoint id")
    paths = e.get("paths")
    if (not isinstance(paths, (list, tuple)) or not paths
            or any(type(p) is not str or not p or p.startswith("/") or ".." in p.split("/") for p in paths)):
        raise ManualExperimentError("workspace_edit.paths must be non-empty relative workspace paths (runtime paths are not W)")
    if spec.workspace_donor != DECISION:
        raise ManualExperimentError("selective revert edits the decision workspace; workspace_donor must be 'decision'")
    if SELECTIVE_REVERT_MISMATCH not in spec.intended_mismatches:
        raise ManualExperimentError(f"declare intended mismatch {SELECTIVE_REVERT_MISMATCH!r} for a selective revert")


def _compose_selective_revert(store: ArtifactStoreProtocol, base: Checkpoint, restore: Checkpoint,
                              paths: list[str]) -> dict[str, Any]:
    """Build the composed W tar and its fidelity manifest; store both; return the recorded composition."""
    import io
    import tarfile
    from ..execution.environment import fidelity_manifest_from_tar

    wanted = [p.rstrip("/") for p in paths]
    norm = lambda n: n[2:] if n.startswith("./") else n.lstrip("/")
    hit = lambda n: any(norm(n) == p or norm(n).startswith(p + "/") for p in wanted)
    out = io.BytesIO()
    restored, removed_only = set(), set()
    with tarfile.open(fileobj=io.BytesIO(store.get_bytes(base.state.workspace)), mode="r:*") as bt, \
            tarfile.open(fileobj=io.BytesIO(store.get_bytes(restore.state.workspace)), mode="r:*") as rt, \
            tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as ot:
        base_names = set()
        for m in bt.getmembers():
            base_names.add(norm(m.name).rstrip("/"))
            if not hit(m.name):
                ot.addfile(m, bt.extractfile(m) if m.isfile() else None)
        rmembers = rt.getmembers()
        for m in rmembers:  # parent directories that no longer exist in the decision W
            n = norm(m.name).rstrip("/")
            if m.isdir() and n and n not in base_names and any(p.startswith(n + "/") for p in wanted):
                ot.addfile(m)
        for m in rmembers:
            if hit(m.name):
                ot.addfile(m, rt.extractfile(m) if m.isfile() else None)
                restored.add(norm(m.name))
        removed_only = {p for p in wanted if p in base_names and not any(r == p or r.startswith(p + "/") for r in restored)}
    data = out.getvalue()
    manifest = fidelity_manifest_from_tar(data)
    w_ref = store.put_bytes(data, media_type="application/x-tar")
    m_ref = store.put_json(manifest)
    return {"rule": SELECTIVE_REVERT_RULE, "base_checkpoint_id": base.checkpoint_id,
            "restore_from_checkpoint_id": restore.checkpoint_id, "paths": sorted(wanted),
            "paths_restored": sorted(restored), "paths_deleted": sorted(removed_only),
            "paths_absent_in_both": sorted(p for p in wanted if p not in base_names and
                                           not any(r == p or r.startswith(p + "/") for r in restored)),
            "workspace_ref": to_jsonable(w_ref), "manifest_ref": to_jsonable(m_ref), "entries": len(manifest)}


def parse_manual_spec(value: Mapping[str, Any] | ManualExperimentSpec) -> ManualExperimentSpec:
    """Strict parse: unknown keys and wrong shapes are errors, never ignored."""
    if isinstance(value, ManualExperimentSpec):
        return value
    if not isinstance(value, Mapping):
        raise ManualExperimentError("manual experiment spec must be an object")
    known = {f.name for f in fields(ManualExperimentSpec)}
    unknown = set(value) - known
    if unknown:
        raise ManualExperimentError(f"unknown manual experiment fields: {sorted(unknown)}")
    data = dict(value)
    for name in ("declared_secondary", "intended_mismatches", "context_edits"):
        if name in data and not (name == "context_edits" and data[name] is None):
            if not isinstance(data[name], (list, tuple)):
                raise ManualExperimentError(f"{name} must be a list")
            data[name] = tuple(data[name])
    try:
        return ManualExperimentSpec(**data)
    except TypeError as exc:
        raise ManualExperimentError(f"malformed manual experiment spec: {exc}") from exc


# ------------------------------------------------------------------ resolution
@dataclass(frozen=True)
class _Source:
    s1: Checkpoint
    start: Mapping[str, Any]          # run_started lifecycle payload
    prefix: tuple[Checkpoint, ...]    # same run, ordinal <= s1, by ordinal
    first_online_s: Mapping[str, Any] | None


def _load_source(store: ArtifactStoreProtocol, spec: ManualExperimentSpec) -> _Source:
    try:
        s1 = store.get_checkpoint(spec.decision_checkpoint_id)
    except KeyError as exc:
        raise ManualExperimentError(f"decision checkpoint {spec.decision_checkpoint_id} missing") from exc
    if s1.run_id != spec.run_id:
        raise ManualExperimentError("decision checkpoint belongs to a different run")
    if not s1.branchable:
        raise ManualExperimentError("decision checkpoint is not branchable")
    start = None
    for event in store.iter_events(spec.run_id):
        if event.kind == "lifecycle" and event.payload.get("transition") == "run_started":
            start = event.payload
            break
    if start is None:
        raise ManualExperimentError("source run has no recorded run_started binding; target and worker cannot be bound")
    from ..resample_source import recovery_source_ok, store_start_lookup
    if not recovery_source_ok(start, store_start_lookup(store)):
        raise ManualExperimentError("source run is itself a branch/resume; manual experiments start from natural runs "
                                    "or natural resamples of natural runs only")
    prefix = tuple(c for c in sorted(store.list_checkpoints(spec.run_id), key=lambda c: c.ordinal)  # type: ignore[attr-defined]
                   if c.ordinal <= s1.ordinal and c.event_sequence <= s1.event_sequence)
    annotations = [store.get_record("annotation", item["record_id"])
                   for item in store.list_records("annotation", run_id=spec.run_id)]
    return _Source(s1=s1, start=start, prefix=prefix, first_online_s=first_stall_annotation(annotations))


def _prefix_checkpoint(store: ArtifactStoreProtocol, source: _Source, component: str, declared: str, *,
                       bound: Checkpoint | None = None, restorable: bool = True) -> Checkpoint:
    """Resolve a declared checkpoint to a legal same-scope prefix checkpoint or raise a clear error.

    ``bound`` is the checkpoint the declaration must not exceed (the decision
    checkpoint by default; the history donor for an observation cutoff).
    Restorable components must be branchable; a T source or O cutoff need only
    be a real saved prefix checkpoint.
    """
    s1 = source.s1
    limit = bound or s1
    if declared == DECISION:
        return s1
    try:
        checkpoint = store.get_checkpoint(declared)
    except KeyError as exc:
        raise ManualExperimentError(f"{component} {declared} missing") from exc
    if checkpoint.run_id != s1.run_id:
        raise ManualExperimentError(f"{component} {declared} is from another run or branch")
    if checkpoint.ordinal > limit.ordinal or checkpoint.event_sequence > limit.event_sequence:
        raise ManualExperimentError(f"{component} {declared} is after {limit.checkpoint_id} (future leak)")
    if restorable and not checkpoint.branchable:
        raise ManualExperimentError(f"{component} {declared} is not branchable")
    if checkpoint.scope.target_id != s1.scope.target_id or checkpoint.scope.stage_id != s1.scope.stage_id:
        raise ManualExperimentError(f"{component} {declared} is outside the fixed target/stage")
    if (checkpoint.scope.task_id, checkpoint.scope.requirement_version, checkpoint.scope.evaluator_version) != (
            s1.scope.task_id, s1.scope.requirement_version, s1.scope.evaluator_version):
        raise ManualExperimentError(f"{component} {declared} has incompatible task/requirement/evaluator bindings")
    return checkpoint


def _donor(store: ArtifactStoreProtocol, source: _Source, component: str, declared: str) -> Checkpoint:
    return _prefix_checkpoint(store, source, f"{component} donor", declared)


def _target(store: ArtifactStoreProtocol, source: _Source) -> RecoveryTarget:
    stages = (source.start.get("task") or {}).get("stages") or ()
    stage = next((s for s in stages if s.get("stage_id") == source.s1.scope.stage_id), None)
    if stage is None:
        raise ManualExperimentError("recorded source task has no stage matching the decision checkpoint")
    if (stage.get("target_id"), stage.get("requirement_version"), stage.get("evaluator_version")) != (
            source.s1.scope.target_id, source.s1.scope.requirement_version, source.s1.scope.evaluator_version):
        raise ManualExperimentError("recorded stage bindings differ from the decision checkpoint scope")
    if type(stage.get("instructions")) is not str:
        raise ManualExperimentError("recorded stage has no instruction text")
    return RecoveryTarget(target_id=stage["target_id"], stage_id=stage["stage_id"],
                          requirement_version=stage["requirement_version"], evaluator_version=stage["evaluator_version"],
                          prerequisites=tuple(stage.get("prerequisites") or ()),
                          instructions_ref=store.put_bytes(stage["instructions"].encode("utf-8"), media_type="text/plain"))


def _recovery_settings(spec: ManualExperimentSpec, source: _Source) -> dict[str, Any]:
    recorded = worker_operational_settings(source.start.get("config") or {})
    if spec.recovery["settings"] == "source":
        settings = dict(recorded)
        if not recovery_settings_known(settings):
            raise ManualExperimentError("recorded source operational settings are incomplete; declare recovery.settings explicitly")
        return settings
    settings = dict(spec.recovery["settings"])
    if not recovery_settings_known(settings):
        raise ManualExperimentError("recovery.settings must declare call cap, format guard, request policy, tool timeout and detector settings")
    if set(settings) - set(WORKER_OPERATIONAL_FIELDS):
        raise ManualExperimentError("recovery.settings has fields outside the recorded operational vocabulary")
    differing = [key for key in FIXED_TO_SOURCE if key in recorded and recorded[key] != settings.get(key)]
    if differing:
        raise ManualExperimentError(f"recovery.settings differ from the natural source on runner-fixed fields: {differing}")
    return settings


def _runtime(spec: ManualExperimentSpec, source: _Source, store: ArtifactStoreProtocol,
             w_donor: Checkpoint) -> tuple[str, Checkpoint | None]:
    policy = spec.runtime.get("policy", "follow_workspace")
    declared = spec.runtime.get("donor", WORKSPACE)
    s1_class = source.s1.state.runtime_class
    if policy == "follow_workspace":
        policy = {RuntimeClass.CAPTURED: "restore_captured", RuntimeClass.IRRELEVANT: "irrelevant"}.get(s1_class)
        if policy is None:
            raise ManualExperimentError("decision checkpoint runtime is R_UNSUPPORTED; no faithful runtime policy exists")
    if policy == "irrelevant":
        if s1_class != RuntimeClass.IRRELEVANT:
            raise ManualExperimentError("runtime 'irrelevant' declared but the decision checkpoint runtime is not R_IRRELEVANT")
        if "donor" in spec.runtime:
            raise ManualExperimentError("an R_IRRELEVANT runtime has no donor")
        return policy, None
    if policy == "discard_captured":
        if s1_class != RuntimeClass.CAPTURED or "donor" in spec.runtime:
            raise ManualExperimentError("runtime discard_captured needs an R_CAPTURED decision checkpoint and no donor")
        if "runtime_discarded" not in spec.intended_mismatches:
            raise ManualExperimentError("declare intended mismatch 'runtime_discarded' to discard the captured runtime")
        return policy, None
    if s1_class != RuntimeClass.CAPTURED:
        raise ManualExperimentError("restore_captured requires an R_CAPTURED decision checkpoint")
    donor = w_donor if declared == WORKSPACE else _donor(store, source, "runtime", declared)
    if donor.state.runtime_class != RuntimeClass.CAPTURED or donor.state.runtime is None:
        raise ManualExperimentError(f"runtime donor {donor.checkpoint_id} lacks a captured runtime artifact")
    return policy, donor


def _history(spec: ManualExperimentSpec, source: _Source, store: ArtifactStoreProtocol) -> Checkpoint | None:
    if spec.history["policy"] == "fresh":
        return None
    donor = _donor(store, source, "history", spec.history.get("donor", DECISION))
    if donor.scope.session_id != source.s1.scope.session_id:
        if "history_from_other_session" not in spec.declared_secondary or spec.selection_kind != "secondary":
            raise ManualExperimentError("history donor is from another session; declare history_from_other_session "
                                        "with selection_kind 'secondary' to run it as an explicit secondary comparison")
    if donor.checkpoint_id != source.s1.checkpoint_id and "history_donor_differs_from_decision" not in spec.declared_secondary:
        raise ManualExperimentError("history donor differs from the decision checkpoint; declare "
                                    "history_donor_differs_from_decision to acknowledge the intentional mismatch")
    return donor


# --------------------------------------------------------------------- T packets
@dataclass(frozen=True)
class _PacketChoice:
    refs: tuple[ArtifactRef, ...]
    meta: dict[str, Any]
    source_checkpoint: Checkpoint | None
    scope: Mapping[str, Any]          # source-visibility scope of the underlying T/summary
    packets: FrozenPackets | None     # frozen set, when the choice came from one


_NO_PACKET = _PacketChoice((), {"kind": "none"}, None,
                           {"session_scope": "full_trajectory", "include_returned_reasoning": False}, None)


def _frozen_packet(store: ArtifactStoreProtocol, spec: ManualExperimentSpec, source: _Source,
                   branch_inputs: dict[str, Any]) -> _PacketChoice:
    try:
        record = store.get_record("frozen_packets_v2", spec.packet["freeze_id"])
    except KeyError as exc:
        raise ManualExperimentError(f"frozen packets {spec.packet['freeze_id']} missing") from exc
    packets = from_jsonable(FrozenPackets, record["result"])
    if packets.freeze_id != spec.packet["freeze_id"]:
        raise ManualExperimentError("frozen packet record identity differs from its declared freeze_id")
    if packets.run_id != source.s1.run_id:
        raise ManualExperimentError("frozen packets are bound to a different run than the decision checkpoint")
    cp = _prefix_checkpoint(store, source, "packet source checkpoint", packets.checkpoint_id, restorable=False)
    # Reuse the shared packet resolution against a spec whose S1 is the packet's own checkpoint.
    shadow = BranchSpec(**{**branch_inputs, "s1": cp, "donor": select_donor(store, spec.run_id, cp), "packets": packets})
    try:
        refs, meta = _packet(shadow, spec.packet["kind"])
    except TreatmentError as exc:
        raise ManualExperimentError(str(exc)) from exc
    scope = packets.binding.get("source_visibility", {}).get("scope") or _NO_PACKET.scope
    meta = {**meta, "source_checkpoint_id": cp.checkpoint_id, "source_event_cut": cp.event_sequence,
            "provenance": "frozen_packets_v2", "freeze_id": packets.freeze_id}
    return _PacketChoice(refs, meta, cp, scope, packets)


def _selected_packet(store: ArtifactStoreProtocol, spec: ManualExperimentSpec, source: _Source) -> _PacketChoice:
    p = spec.packet
    cp = _prefix_checkpoint(store, source, "packet source checkpoint", p["source_checkpoint_id"], restorable=False)
    # The retained primary pointer is read exactly where declared: the bare
    # chain, or an explicitly named namespace (combined/scoped observers key
    # pointers as <namespace>:<run>:<checkpoint>). Nothing is inferred by scanning.
    namespace = p.get("chain_namespace")
    chain_id = f"{spec.run_id}:{cp.checkpoint_id}" if namespace is None else f"{namespace}:{spec.run_id}:{cp.checkpoint_id}"
    try:
        pointer = store.get_record("t_chain", chain_id)
    except KeyError as exc:
        raise ManualExperimentError(f"no primary T chain entry {chain_id!r}; declare the exact chain_namespace of the "
                                    "retained primary pointer") from exc
    version_id = p.get("version_id", pointer["version_id"])
    if version_id != pointer["version_id"]:
        raise ManualExperimentError(f"version {version_id} is not the primary T version bound at {chain_id!r}")
    try:
        version: TVersion = artifacts.load_version(store, version_id)
    except (KeyError, ValueError) as exc:
        raise ManualExperimentError(f"T version {version_id} unavailable: {exc}") from exc
    if (version.run_id, version.target_checkpoint_id, version.role) != (spec.run_id, cp.checkpoint_id, "t_retention"):
        raise ManualExperimentError("T version binding differs from the declared source checkpoint or is not primary retention T")
    if version.status not in VALID_T_STATUSES:
        raise ManualExperimentError(f"T version {version_id} is {version.status}; failed or missing T cannot be injected")
    if version.through_seq > cp.event_sequence:
        raise ManualExperimentError("T version draws on events after its own checkpoint cut")
    types = set(p.get("types", ENTRY_TYPES))
    ids = p.get("entry_ids")
    known = {e.entry_id for e in version.entries}
    if ids is not None and set(ids) - known:
        raise ManualExperimentError(f"entry_ids not in T version {version_id}: {sorted(set(ids) - known)}")
    kept = [e for e in version.entries if e.type in types and (ids is None or e.entry_id in set(ids))]
    if not kept:
        raise ManualExperimentError("t_selected selects no entries; use packet kind 'none' instead of an empty packet")
    omitted = [e.entry_id for e in version.entries if e.entry_id not in {e.entry_id for e in kept}]
    selection = {"revision": SELECTION_FILTER_REVISION, "types": sorted(types) if "types" in p else None,
                 "entry_ids": list(ids) if ids is not None else None, "render": p["render"]}
    filter_id = f"{SELECTION_FILTER_REVISION}:{content_id(selection)}"
    render = render_prose if p["render"] == "prose" else render_typed
    try:
        packet = render(version, kept, filter_id=filter_id, omitted=omitted)
    except ValueError as exc:
        raise ManualExperimentError(f"selection could not be rendered from the frozen version: {exc}") from exc
    packet_id, ref = artifacts.save_packet(store, spec.run_id, packet)
    meta = {"kind": SELECTED_PACKET_KIND, "packet_id": packet_id, "source_version_id": version_id,
            "chain_namespace": namespace, "chain_pointer_id": chain_id,
            "source_checkpoint_id": cp.checkpoint_id, "source_event_cut": cp.event_sequence,
            "parent_version_id": version.parent_version_id, "version_content_hash": version.content_hash(),
            "update_index": version.update_index, "version_status": version.status,
            "selection": selection, "filter_id": filter_id, "renderer_revision": packet.renderer_revision,
            "entry_ids": list(packet.entry_ids), "omitted_entry_ids": list(packet.omitted_entry_ids),
            "provenance": "deterministic_view_of_frozen_primary_version; no regeneration or refill",
            "approx_token_count": approx_tokens(packet.text), "token_count": None, "token_measurement": None,
            "token_fidelity": "unmeasured_manual_view", "freeze_id": None, "matching_report": {}}
    scope = (version.source_visibility or {}).get("scope") or _NO_PACKET.scope
    return _PacketChoice((ref,), meta, cp, scope, None)


def _research_packet(store: ArtifactStoreProtocol, spec: ManualExperimentSpec, source: _Source) -> _PacketChoice:
    p = spec.packet
    try:
        record, ref, _ = load_research_packet(store, p["packet_id"])
    except ResearchPacketError as exc:
        raise ManualExperimentError(str(exc)) from exc
    if record["label"] != p["label"]:
        raise ManualExperimentError(f"research packet {p['packet_id']} is labelled {record['label']!r}, not {p['label']!r}")
    if record["run_id"] != spec.run_id:
        raise ManualExperimentError("research packet is bound to a different run than the decision checkpoint")
    cp = _prefix_checkpoint(store, source, "research packet source checkpoint", record["source_checkpoint_id"],
                            restorable=False)
    if record["source_event_cut"] != cp.event_sequence or record["source_visibility"]["through_seq"] > cp.event_sequence:
        raise ManualExperimentError("research packet draws on events after its own source checkpoint cut")
    scope = dict(record["source_visibility"]["scope"])
    if scope["session_scope"] == "current_session" and record["source_visibility"].get("session_id") != cp.scope.session_id:
        raise ManualExperimentError("current-session research packet is not bound to its source checkpoint's native session")
    construction = record["construction"]
    meta = {"kind": RESEARCH_PACKET_KIND, "label": record["label"], "packet_id": record["packet_id"],
            "source_checkpoint_id": cp.checkpoint_id, "source_event_cut": cp.event_sequence,
            "token_count": record["token_count"], "token_measurement": record["token_measurement"],
            "approx_token_count": record["token_count"], "source_ids": record["source_ids"],
            "selected_unit_id": construction.get("selected_unit_id"),
            "withheld_item_ids": construction.get("withheld_item_ids"),
            "random_seed": construction.get("random_seed"),
            "provenance": "registered external research packet; exact bytes, no regeneration or refill",
            "freeze_id": None}
    return _PacketChoice((ref,), meta, cp, scope, None)


def _packet_choice(store: ArtifactStoreProtocol, spec: ManualExperimentSpec, source: _Source,
                   branch_inputs: dict[str, Any]) -> _PacketChoice:
    kind = spec.packet["kind"]
    if kind == "none":
        return _NO_PACKET
    choice = (_selected_packet(store, spec, source) if kind == SELECTED_PACKET_KIND
              else _research_packet(store, spec, source) if kind == RESEARCH_PACKET_KIND
              else _frozen_packet(store, spec, source, branch_inputs))
    assert choice.source_checkpoint is not None
    if (choice.source_checkpoint.checkpoint_id != source.s1.checkpoint_id
            and "packet_from_earlier_checkpoint" not in spec.declared_secondary):
        raise ManualExperimentError("packet source checkpoint differs from the decision checkpoint; declare "
                                    "packet_from_earlier_checkpoint to inject T frozen at an earlier prefix checkpoint")
    return choice


def _cross_session(spec: ManualExperimentSpec, source: _Source, choice: _PacketChoice) -> bool:
    if choice.source_checkpoint is None:
        return False
    through = choice.source_checkpoint
    earlier = [c for c in source.prefix if c.ordinal <= through.ordinal]
    multi = len({c.scope.session_id for c in earlier}) > 1 or through.scope.session_id != source.s1.scope.session_id
    if choice.scope.get("session_scope", "full_trajectory") == "full_trajectory" and multi:
        if "cross_session_information" not in spec.declared_secondary:
            raise ManualExperimentError("the packet draws on earlier sessions; declare cross_session_information "
                                        "to run this as an explicit cross-session comparison")
        return True
    return False


# --------------------------------------------------------------------- build
def build_manual_manifest(store: ArtifactStoreProtocol, spec: Mapping[str, Any] | ManualExperimentSpec) -> TreatmentManifest:
    """Deterministically compile a manual declaration into the shared manifest vocabulary."""
    spec = parse_manual_spec(spec)
    source = _load_source(store, spec)
    s1 = source.s1
    if spec.selection_kind == "primary_s1" and (source.first_online_s is None
                                                or source.first_online_s.get("checkpoint_id") != s1.checkpoint_id):
        raise ManualExperimentError("selection_kind primary_s1 requires the decision checkpoint to be the recorded first online S")
    target = _target(store, source)
    settings = _recovery_settings(spec, source)
    config = source.start.get("config") or {}
    worker_model, model_settings = config.get("model"), config.get("model_settings")
    if type(worker_model) is not str or not isinstance(model_settings, Mapping):
        raise ManualExperimentError("source run_started lacks the worker model binding")
    w_donor = _donor(store, source, "workspace", spec.workspace_donor)
    runtime_policy, r_donor = _runtime(spec, source, store, w_donor)
    h_donor = _history(spec, source, store)
    history_policy = spec.history["policy"]
    composition = None
    if spec.workspace_edit is not None:
        restore = _donor(store, source, "workspace", spec.workspace_edit["restore_from"])
        if restore.ordinal >= w_donor.ordinal:
            raise ManualExperimentError("selective revert restore_from must precede the decision checkpoint")
        composition = _compose_selective_revert(store, w_donor, restore, list(spec.workspace_edit["paths"]))
        composition["provenance"] = spec.workspace_edit.get("provenance")
    reference = select_donor(store, spec.run_id, s1)
    branch_inputs = dict(run_id=spec.run_id, s1=s1, donor=reference, target=target, worker_model=worker_model,
                         model_settings=model_settings, generated_token_limit=spec.recovery["generated_token_limit"],
                         selection_kind=spec.selection_kind, runtime_policy=runtime_policy, recovery_settings=settings)
    choice = _packet_choice(store, spec, source, branch_inputs)
    packet_refs, packet_meta = choice.refs, choice.meta
    if packet_refs and history_policy != "fresh" and "packet_on_native_history" not in spec.declared_secondary:
        raise ManualExperimentError("packets enter fresh-scaffold branches by default; native Continue never receives T. "
                                    "Declare packet_on_native_history to run T injection after restored native history "
                                    "as an explicit manual intervention")
    cross_session = _cross_session(spec, source, choice)
    donors: dict[str, str] = {"workspace": w_donor.checkpoint_id,
                              "history": "fresh" if h_donor is None else h_donor.checkpoint_id}
    if r_donor is not None:
        donors["runtime"] = r_donor.checkpoint_id
    mismatches = list(spec.intended_mismatches)
    execution_support: dict[str, str] = {}
    if w_donor.checkpoint_id != s1.checkpoint_id:
        mismatches.append("workspace_from_earlier_checkpoint")
        if h_donor is not None and h_donor.checkpoint_id == s1.checkpoint_id:
            mismatches.append("history_mentions_absent_implementation")
        if r_donor is not None:
            mismatches.append("W+R_restored_from_donor")
    if r_donor is not None and r_donor.checkpoint_id != w_donor.checkpoint_id:
        if "runtime_donor_differs_from_workspace" not in spec.declared_secondary:
            raise ManualExperimentError("runtime donor differs from workspace donor; declare runtime_donor_differs_from_workspace")
        mismatches.append("runtime_and_workspace_from_different_checkpoints")
    if h_donor is not None and h_donor.checkpoint_id != s1.checkpoint_id:
        mismatches.append("history_from_earlier_checkpoint")
        execution_support["explicit_history_donor"] = DECLARED
    if packet_refs and history_policy != "fresh":
        mismatches.append("t_packet_injected_after_native_history")
        execution_support["packet_on_native_history"] = DECLARED
    if choice.source_checkpoint is not None and choice.source_checkpoint.checkpoint_id != s1.checkpoint_id:
        mismatches.append("t_information_stops_before_decision_checkpoint")
        execution_support["packet_from_earlier_checkpoint"] = DECLARED
    if cross_session:
        mismatches.append("packet_restores_prior_session_information")
    metadata: dict[str, Any] = {
        "arm_id": f"manual:{spec.label}", "study_cells": [f"manual:{spec.label}"], "packet": packet_meta,
        "orientation_text": ORIENTATION_TEXT if history_policy == "fresh" else None,
        "recovery_target": target.as_record(), "recovery_target_manifest_id": target.manifest_id(),
        "donor_selection": reference.as_record(), "model_settings": dict(model_settings),
        "recovery_settings": settings,
        "packet_position": (PACKET_AFTER_NATIVE_HISTORY if packet_refs and history_policy != "fresh"
                            else PACKET_AFTER_FRESH_SCAFFOLD),
        "isolation": "fresh_isolated_execution; no access to research store, other branches, grades or T",
        "manual_experiment": {
            "version": MANUAL_VERSION, "spec": spec.as_record(), "spec_digest": spec.digest(),
            "resolved": {"workspace_donor": w_donor.checkpoint_id, "runtime_donor": None if r_donor is None else r_donor.checkpoint_id,
                         "history_donor": None if h_donor is None else h_donor.checkpoint_id,
                         "packet_source_checkpoint": None if choice.source_checkpoint is None else choice.source_checkpoint.checkpoint_id,
                         "reference_best_before_donor": reference.donor_checkpoint_id,
                         "first_online_s_checkpoint_id": None if source.first_online_s is None else source.first_online_s.get("checkpoint_id"),
                         "run_started_task_id": (source.start.get("task") or {}).get("task_id")},
            "declared_secondary": list(spec.declared_secondary), "cross_session_information": cross_session,
            "execution_support": execution_support,
            "continue_equivalent": (w_donor.checkpoint_id == s1.checkpoint_id and history_policy == "native"
                                    and h_donor is not None and h_donor.checkpoint_id == s1.checkpoint_id
                                    and (r_donor is None or r_donor.checkpoint_id == s1.checkpoint_id) and not packet_refs),
        },
    }
    if composition is not None:
        metadata["workspace_composition"] = composition
        metadata["manual_experiment"]["execution_support"]["selective_revert"] = DECLARED
    if history_policy == "edited":
        from .context_edit import EDITS_VERSION, OUTPUTS_RULE, validate_edits
        assert h_donor is not None and spec.context_edits is not None
        edits = [dict(e) for e in spec.context_edits]
        problems = validate_edits(store.get_json(h_donor.state.history)["messages"], edits)
        if problems:
            raise ManualExperimentError("context edits do not apply to the history donor: " + "; ".join(problems))
        metadata["context_edits"] = {"version": EDITS_VERSION, "edits": edits, "outputs_rule": OUTPUTS_RULE}
        mismatches.append("context_edited")
        metadata["manual_experiment"]["execution_support"]["edited_history"] = DECLARED
    if history_policy == "o_masked":
        assert h_donor is not None and spec.observation_mask is not None
        policy = spec.observation_mask["policy"]
        if policy == MANUAL_WINDOW_POLICY:
            if "manual_observation_cutoff" not in spec.declared_secondary or spec.selection_kind != "secondary":
                raise ManualExperimentError("a manual observation window is an explicit secondary comparison: declare "
                                            "manual_observation_cutoff with selection_kind 'secondary'")
            after = _prefix_checkpoint(store, source, "observation window start", spec.observation_mask["after_checkpoint_id"],
                                       bound=h_donor, restorable=False)
            through = None
            if "through_checkpoint_id" in spec.observation_mask:
                through = _prefix_checkpoint(store, source, "observation window end", spec.observation_mask["through_checkpoint_id"],
                                             bound=h_donor, restorable=False)
                if through.event_sequence < after.event_sequence or through.ordinal < after.ordinal:
                    raise ManualExperimentError("observation window end precedes its start")
            for cp in (after, through):
                if cp is not None and cp.scope != h_donor.scope:
                    raise ManualExperimentError("observation window checkpoints must share the history donor's session/target scope")
            try:
                mask = build_observation_mask(store, h_donor, selection_kind=spec.selection_kind, policy=policy,
                                              after_checkpoint_id=after.checkpoint_id,
                                              through_checkpoint_id=None if through is None else through.checkpoint_id,
                                              decision_checkpoint_id=s1.checkpoint_id)
            except (TypeError, MaskError) as exc:
                raise ManualExperimentError("root observation-mask helper does not provide the manual checkpoint window "
                                            f"policy yet: {type(exc).__name__}: {exc}") from exc
            mismatches.append("observations_retained_through_chosen_checkpoint_not_detector_anchor")
            execution_support["manual_observation_window"] = DECLARED
        else:
            if set(spec.observation_mask) != {"policy"}:
                raise ManualExperimentError("window fields are only accepted with the manual checkpoint window policy")
            mask = build_observation_mask(store, h_donor, selection_kind=spec.selection_kind, policy=policy)
        if mask.get("status") != "available":
            raise ManualExperimentError("observation mask unavailable for the selected history donor: " + str(mask.get("reason")))
        if mask.get("placeholder") != MASK_PLACEHOLDER:
            raise ManualExperimentError("observation mask does not carry the ratified placeholder")
        metadata["mask"] = mask
        metadata["mask_comparison_role"] = mask.get("comparison_role")
        metadata["manual_experiment"]["execution_support"] = execution_support
    packet_source = s1 if choice.source_checkpoint is None else choice.source_checkpoint
    if choice.source_checkpoint is not None:
        metadata["packet_source_checkpoint_id"] = packet_source.checkpoint_id
    # The declared information scope stops at the actual T/summary source checkpoint.
    source_scope = {**choice.scope, "through_checkpoint_id": packet_source.checkpoint_id}
    treatment_id = content_id({"revision": TREATMENT_IDENTITY_REVISION, "run": spec.run_id, "s1": s1.checkpoint_id,
        "spec_digest": spec.digest(), "target": target.manifest_id(), "packets": [r.sha256 for r in packet_refs],
        "donors": donors, "runtime_policy": runtime_policy, "history_policy": history_policy,
        "model": worker_model, "model_settings": dict(model_settings), "allowance": spec.recovery["generated_token_limit"],
        "repeat": spec.recovery["repeat_id"], "selection_kind": spec.selection_kind, "metadata": metadata})
    return TreatmentManifest(
        treatment_id=treatment_id, source_checkpoint_id=s1.checkpoint_id, target_id=target.target_id,
        stage_id=target.stage_id, selection_kind=spec.selection_kind, component_donors=donors,
        history_policy=history_policy, runtime_policy=runtime_policy, worker_model=worker_model,
        generated_token_limit=spec.recovery["generated_token_limit"], repeat_id=spec.recovery["repeat_id"],
        packet_refs=packet_refs, source_scope=source_scope, intended_mismatches=tuple(mismatches), metadata=metadata)


# ------------------------------------------------------------------ validate
def manual_structural_problems(manifest: TreatmentManifest, store: ArtifactStoreProtocol) -> list[str]:
    """Manual-level checks: dispatch marker, deterministic reproduction from the bound spec, donor legality."""
    manual = manifest.metadata.get("manual_experiment")
    if not isinstance(manual, Mapping) or manual.get("version") != MANUAL_VERSION:
        return ["manifest is not a manual-experiment.v1 manifest"]
    try:
        rebuilt = build_manual_manifest(store, manual["spec"])
    except ManualExperimentError as exc:
        return [f"manifest does not rebuild from its bound spec: {exc}"]
    except (KeyError, TypeError, ValueError) as exc:
        return [f"manifest spec is malformed: {type(exc).__name__}: {exc}"]
    problems = []
    if manifest_digest(rebuilt) != manifest_digest(manifest):
        problems.append("manifest differs from its deterministic rebuild against current immutable evidence")
    if manual.get("spec_digest") != content_id(to_jsonable(manual["spec"])):
        problems.append("spec digest does not match the bound spec")
    return problems


def validate_manual_manifest(manifest: TreatmentManifest, store: ArtifactStoreProtocol) -> list[str]:
    """Deterministic structural rebuild, then the common validator's checks verbatim.

    ``treatments.validate_manifest`` dispatches manual manifests here, so the
    common helper is called directly to avoid recursion. Nothing is suppressed:
    the common helper already understands explicit history donors, an earlier
    packet source checkpoint and native-history packet injection for manual
    manifests while keeping every provenance check.
    """
    problems = manual_structural_problems(manifest, store)
    if problems:
        return problems
    run_id = store.get_checkpoint(manifest.source_checkpoint_id).run_id
    return list(_validate_manifest_common(manifest, store, run_id))
