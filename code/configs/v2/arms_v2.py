"""Branch specs (branch-spec.v2, brief-interfaces.md) for every v2 spec set: the screen, extra screen resamples,
tuning rounds, the main arms 1-6, the harm test and the injection test. No model calls, no Docker.

A spec set is a manifest (study/specsets/<prefix>.json) plus one spec file per branch, grouped in stall blocks whose
jobs are already in run order (repeat-major inside a stall, arms interleaved), which is what driver_v2 schedules.
Spec files carry exactly the interface keys; provenance (input hashes, DRAFT rules used, unavailable stalls) lives in
the manifest. Every branch has its own seed, hashed from purpose, set prefix and branch id; the study's seed registry
refuses any seed reused by another branch, so screen and tuning seeds are never reused by the arms.

  collect    --store S --stalls-root D             stall dirs (stall.json) from the store's stall_confirmation records
  collect-episodes --store S --stalls-root D       stall dirs from stall judge v2's stall_episode records (every
                                                   branchable episode; no per-task cap, user Q19)
  screen     --study R --stalls-root D [--stall ID ...]          arms 1 and 4 x 5, purpose screen, budget = guard only
  screen-extra --study R --stalls-root D --screen-json F --round N   extra resamples for stalls near a strata cutoff
  tuning     --study R --stalls-root D --stall ID ... --round N --repeats K --calibration F
  main       --study R --stalls-root D --screen-json F --calibration F [--allow-draft]
  harm       --study R --healthy-root H --calibration F [--allow-draft]
  injection  --study R --healthy-root H --calibration F [--allow-draft]
Stall dir: stall.json (this script) + boundary.json, ledger.json, review_view_{ledger,context}.json, recipe.json
(track D). Select --arm-set v3 explicitly for the current pilot: all spec-set commands use the six current arms,
fixed-unit flat Ledger views, and the prefix-only judge onset for arm 4. Historical v2 remains the default for
backward compatibility. Healthy dir: healthy.json + review views + injection_donors.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

from study_common import (ARMS, ARMS_V3, GUARD, HARM_ARM, HARM_REPEATS, INJECTION_CONDITIONS, INJECTION_REPEATS,
                          MAIN_REPEATS, MAX_HEALTHY_PER_TASK, MAX_STALLS_PER_TASK, PROMPT_VERSIONS, PROMPT_VERSIONS_V3,
                          READ_ONLY_ARMS_V3, REVIEW_VIEWS, REVIEW_VIEWS_V3, REWIND_ARMS, SCREEN_ARMS, SCREEN_REPEATS,
                          SPEC_VERSION, SPEC_VERSION_V3, SPECSET_VERSION, TUNING_ARMS, branch_id,
                          canonical, healthy_dirs, load_stall, read_json, register_seeds, runs_dir, safe, seed_for,
                          sha256_file, specset_path, stall_dirs, stall_id_for, view_units, write_json)

NOW = lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
CURRENT_ARM_SET = "v2"
_LEGACY_OPERATIONS = {k: globals()[k] for k in
                      ("ARMS", "SCREEN_ARMS", "TUNING_ARMS", "REVIEW_VIEWS", "PROMPT_VERSIONS")}


def configure_arm_set(version: str) -> None:
    """Select current operations explicitly; historical v2 specs retain their meaning."""
    global CURRENT_ARM_SET, ARMS, SCREEN_ARMS, TUNING_ARMS, REVIEW_VIEWS, PROMPT_VERSIONS
    CURRENT_ARM_SET = version
    if version == "v2":
        globals().update(_LEGACY_OPERATIONS)
    if version == "v3":
        ARMS = ARMS_V3
        SCREEN_ARMS = ("1_continue", "4_forced_rewind_to_p")
        TUNING_ARMS = ("2_ledger_review", "3_context_review", "5_ledger_read_only", "6_context_read_only")
        REVIEW_VIEWS = REVIEW_VIEWS_V3
        PROMPT_VERSIONS = {a: p.replace(".v1-draft", ".units-v1-draft") for a, p in PROMPT_VERSIONS_V3.items()}


class Unavailable(Exception):
    """A stall or checkpoint cannot get this spec set; recorded with its reason, never substituted."""


# ---------- collect ----------

def collect(store: Path, stalls_root: Path) -> dict:
    """One stall dir per confirmed-primary stall_confirmation record (read-only over the store)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detector-judge-20260926"))
    import storeio  # read-only journal copy
    st = storeio.Store(store.resolve())
    rows = st.con.execute("select record_id, ref_json from records where kind='stall_confirmation' "
                          "order by record_id").fetchall()
    made, skipped = [], []
    for rid, ref in rows:
        rec = st.ref_json(json.loads(ref))
        prim = rec.get("primary")
        if not prim:
            skipped.append({"run_id": rec.get("run_id"), "reason": "no confirmed primary stall"})
            continue
        sid = stall_id_for(rec["task_id"], rec["run_id"], prim["step_index"])
        d = stalls_root / sid
        stall = {"version": "stall.v2", "stall_id": sid, "store": str(store.resolve()), "run_id": rec["run_id"],
                 "task_id": rec["task_id"], "checkpoint_id": prim["checkpoint_id"], "step_index": prim["step_index"],
                 "later_solved": bool(rec.get("later_solved")),
                 "context_tokens_at_stall": _context_tokens(st, storeio, rec["run_id"], prim["checkpoint_id"]),
                 "stall_confirmation": rec}
        prior = read_json(d / "stall.json", None)
        if prior is not None and {k: v for k, v in prior.items() if k != "collected_at"} != stall:
            raise SystemExit(f"stall.json differs from the store record; refusing to overwrite {d}")
        if prior is None:
            write_json(d / "stall.json", stall | {"collected_at": NOW()})
        made.append(sid)
    per_task = Counter(read_json(stalls_root / s / "stall.json")["task_id"] for s in made)
    over = {t: n for t, n in per_task.items() if n > MAX_STALLS_PER_TASK}
    if over:
        raise SystemExit(f"more than {MAX_STALLS_PER_TASK} stalls for tasks {over}: fix the confirmation records "
                         "(the per-task cap is applied when stalls are confirmed)")
    return {"stalls": made, "skipped": skipped, "per_task": dict(per_task)}


def collect_episodes(store: Path, stalls_root: Path) -> dict:
    """One stall dir per branchable stall episode of stall judge v2 (configs/v2/judge_v2/postrun_v2.py, kind
    `stall_episode`; user Q19: every episode counts, no cap per task). The branch source is the checkpoint at the
    episode's first confirmed judge point (the stall moment, where arms 1, 2, 3, 5 and 6 start); `stall_onset_point`
    (P - 1, the state just before the stall's first step ran; declared reading pending the user) is arm 4's rewind
    target, passed to arm_spec_v3(..., onset_point=stall["stall_onset_point"]). Episodes whose source or onset point
    is not branchable, or that share a branch source with an earlier episode of the same run, are skipped with the
    reason (never merged into one stall)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detector-judge-20260926"))
    import storeio  # read-only journal copy
    st = storeio.Store(store.resolve())
    rows = st.con.execute("select record_id, ref_json from records where kind='stall_episode' "
                          "order by record_id").fetchall()
    made, skipped = [], []
    for eid, ref in rows:
        rec = st.ref_json(json.loads(ref))
        e = rec["episode"]
        if not e.get("branchable"):
            skipped.append({"episode_id": eid, "reason": e.get("not_branchable_reason") or "not branchable"})
            continue
        if e.get("same_source_as_episodes"):
            skipped.append({"episode_id": eid, "reason": f"same branch source as episodes {e['same_source_as_episodes']}"})
            continue
        sid = stall_id_for(rec["task_id"], rec["run_id"], e["step_index"])
        d = stalls_root / sid
        stall = {"version": "stall.v2", "stall_id": sid, "store": str(store.resolve()), "run_id": rec["run_id"],
                 "task_id": rec["task_id"], "checkpoint_id": e["checkpoint_id"], "step_index": e["step_index"],
                 "later_solved": bool(e.get("later_solved_by_submission")), "episode_id": eid,
                 "onset_step": e["onset"], "stall_onset_point": e["stall_onset_point"],
                 "stall_onset_checkpoint_id": e["stall_onset_checkpoint_id"],
                 "context_tokens_at_stall": _context_tokens(st, storeio, rec["run_id"], e["checkpoint_id"]),
                 "stall_episode": rec}
        prior = read_json(d / "stall.json", None)
        if prior is not None and {k: v for k, v in prior.items() if k != "collected_at"} != stall:
            raise SystemExit(f"stall.json differs from the store record; refusing to overwrite {d}")
        if prior is None:
            write_json(d / "stall.json", stall | {"collected_at": NOW()})
        made.append(sid)
    per_task = Counter(read_json(stalls_root / s / "stall.json")["task_id"] for s in made)
    return {"stalls": made, "skipped": skipped, "per_task": dict(per_task), "cap_per_task": None}


def _context_tokens(st, storeio, run_id: str, checkpoint_id: str) -> int | None:
    """Prompt tokens of the last model call at or before the stall checkpoint (unknown stays None)."""
    try:
        cut = st.checkpoint_json(checkpoint_id)["event_sequence"]
    except (TypeError, KeyError):
        return None
    last = None
    for seq, d in st.events(run_id):
        if seq > cut:
            break
        if d["kind"] == "model_response":
            last = (d.get("usage") or {}).get("input_tokens")
    return last


# ---------- spec building ----------

def budget(caps: dict | None) -> dict:
    caps = caps or {}
    return {"new_token_cap": caps.get("new_token_cap"), "submission_cap": caps.get("submission_cap"), **GUARD}


def spec(*, purpose: str, prefix: str, unit_id: str, arm: str, repeat: int, source: dict, caps: dict | None) -> dict:
    bid = branch_id(unit_id, arm, repeat)
    result = {"version": SPEC_VERSION, "branch_id": bid, "purpose": purpose, "source": source, "arm": arm,
            "repeat": repeat, "seed": seed_for(purpose, prefix, bid), "rewind": None, "context_edits": [],
            "review": None, "note": None, "handback": {"items": [], "recipe": None}, "budget": budget(caps),
            "injection": None}
    if CURRENT_ARM_SET == "v3":
        from trajectory_retention.experiments.submission_stop import SETTING
        result["submission_stop"] = dict(SETTING)
        result["unresolved_retry"] = read_json(Path(__file__).with_name("settings.json"))["unresolved_retry"]
        from trajectory_retention.experiments.submission_feedback import SETTING as FEEDBACK
        result["submission_feedback"] = dict(FEEDBACK)
    return result


def source_of(stall: dict) -> dict:
    return {"store": stall["store"], "run_id": stall["run_id"], "checkpoint_id": stall["checkpoint_id"]}


def boundary(stall: dict) -> dict:
    if not stall["has_boundary"]:
        raise Unavailable("boundary.json missing (the primary boundary is frozen before screening)")
    b = read_json(Path(stall["dir"]) / "boundary.json")
    if b.get("status") != "ok":
        raise Unavailable(f"boundary.json status is {b.get('status')!r}, not 'ok'")
    rp = b.get("rewind_point")
    if not isinstance(rp, int) or rp < 0 or rp >= stall["step_index"]:
        raise Unavailable(f"rewind_point {rp!r} is not a point before the stall (step {stall['step_index']})")
    return b


def arm_spec(stall: dict, arm: str, repeat: int, *, purpose: str, prefix: str, caps: dict | None,
             rewind_mode: str = "per_channel", prompts: dict | None = None, opus_record: bool = False) -> dict:
    if CURRENT_ARM_SET == "v3":
        return arm_spec_v3(stall, arm, repeat, purpose=purpose, prefix=prefix, caps=caps,
                           onset_point=stall.get("stall_onset_point"), rewind_mode=rewind_mode,
                           prompts=prompts or PROMPT_VERSIONS, opus_record=opus_record)
    s = spec(purpose=purpose, prefix=prefix, unit_id=stall["stall_id"], arm=arm, repeat=repeat,
             source=source_of(stall), caps=caps)
    prompts = prompts or PROMPT_VERSIONS
    d = Path(stall["dir"])
    if arm in REWIND_ARMS:
        rp = boundary(stall)["rewind_point"]
        s["rewind"] = {"workspace": rp, "context": rp, "runtime": rp, "outputs": rp}
    if arm in REVIEW_VIEWS:
        view = d / REVIEW_VIEWS[arm]
        if not view.exists():
            raise Unavailable(f"{view.name} missing")
        view_units(view)
        s["review"] = {"view_ref": str(view), "prompt_version": prompts[arm], "rewind_mode": rewind_mode,
                       "opus_record": opus_record}
    if arm == "5_rewind_own_summary":
        s["note"] = {"prompt_version": prompts[arm]}
    if arm == "6_rewind_ledger_recipe":
        if not stall["has_recipe"]:
            raise Unavailable("recipe.json missing")
        recipe = read_json(d / "recipe.json")
        if not isinstance(recipe.get("text"), str) or not recipe["text"].strip():
            raise Unavailable("recipe.json has no text")
        s["handback"]["recipe"] = recipe["text"]
    return s


def arm_spec_v3(stall: dict, arm: str, repeat: int, *, purpose: str, prefix: str, caps: dict | None,
                onset_point: int | None, rewind_mode: str = "per_channel", prompts: dict | None = None,
                opus_record: bool = False) -> dict:
    """A branch-spec.v3 for arm set v3 (study_common.ARMS_V3). ``onset_point`` is P, the judge's stall onset point
    for this stall, taken as given (the judge that supplies it is built separately); arm 4 rewinds W, R, the context
    and the outputs to it. Arms 2, 3, 5 and 6 take their view from the stall dir (the flat Ledger for 2 and 5, the
    plain context for 3 and 6); arms 5 and 6 have no decision, so no rewind mode and no Opus record."""
    if arm not in ARMS_V3:
        raise ValueError(f"{arm!r} is not an arm of arm set v3")
    s = spec(purpose=purpose, prefix=prefix, unit_id=stall["stall_id"], arm=arm, repeat=repeat,
             source=source_of(stall), caps=caps)
    s["version"] = SPEC_VERSION_V3
    from trajectory_retention.experiments.submission_feedback import SETTING as FEEDBACK
    s["submission_feedback"] = dict(FEEDBACK)
    s["stall_onset_point"] = None
    prompts = prompts or PROMPT_VERSIONS_V3
    if arm == "4_forced_rewind_to_p":
        if type(onset_point) is not int or not 0 <= onset_point <= stall["step_index"]:
            raise Unavailable(f"stall onset point {onset_point!r} is not a point 0..{stall['step_index']}")
        s["stall_onset_point"] = onset_point
        s["rewind"] = dict.fromkeys(("workspace", "context", "runtime", "outputs"), onset_point)
    if arm in REVIEW_VIEWS_V3:
        view = Path(stall["dir"]) / REVIEW_VIEWS_V3[arm]
        if not view.exists():
            raise Unavailable(f"{view.name} missing")
        view_units(view)
        read_only = arm in READ_ONLY_ARMS_V3
        s["review"] = {"view_ref": str(view), "prompt_version": prompts[arm],
                       "rewind_mode": None if read_only else rewind_mode, "opus_record": False if read_only else opus_record}
    return s


def write_specset(study: Path, prefix: str, purpose: str, blocks: list[dict], *, inputs: dict, draft: list[str],
                  unavailable: list[dict], replace: bool = False) -> dict:
    """Write spec files and the manifest. Rebuilding with identical content is a no-op; a changed spec is refused
    once any branch of the set has an out dir (unless --replace before anything ran)."""
    base = specset_path(study, prefix).parent / prefix
    out_blocks, seeds, changed = [], [], []
    for b in blocks:
        jobs = []
        for sp in b["specs"]:
            rel = Path(prefix) / safe(b["unit_id"]) / f"{sp['arm'].replace(':', '~')}__r{sp['repeat']}.json"
            path = base.parent / rel
            old = read_json(path, None)
            if old is not None and canonical(old) != canonical(sp):
                changed.append(sp["branch_id"])
            jobs.append({"branch_id": sp["branch_id"], "arm": sp["arm"], "repeat": sp["repeat"], "seed": sp["seed"],
                         "spec": str(rel)})
            seeds.append((sp["seed"], f"{prefix}|{sp['branch_id']}"))
        out_blocks.append({k: v for k, v in b.items() if k != "specs"} | {"jobs": jobs})
    ran = runs_dir(study, prefix) / "branches"
    if changed and (not replace or (ran.exists() and any(ran.iterdir()))):
        raise SystemExit(f"{len(changed)} specs of {prefix} would change (e.g. {changed[0]}); refusing"
                         + ("" if replace else " (use --replace before anything has run)"))
    register_seeds(study, seeds)
    for b in blocks:
        for sp in b["specs"]:
            write_json(base / safe(b["unit_id"]) / f"{sp['arm'].replace(':', '~')}__r{sp['repeat']}.json", sp)
    manifest = {"version": SPECSET_VERSION, "prefix": prefix, "purpose": purpose, "created_at": NOW(),
                "inputs": inputs, "draft_rules_used": sorted(set(draft)), "unavailable": unavailable,
                "blocks": out_blocks, "branches": sum(len(b["jobs"]) for b in out_blocks)}
    old = read_json(specset_path(study, prefix), None)
    if old is not None and not changed and canonical({k: v for k, v in old.items() if k != "created_at"}) == \
            canonical({k: v for k, v in manifest.items() if k != "created_at"}):
        manifest["created_at"] = old["created_at"]
    write_json(specset_path(study, prefix), manifest)
    return {"prefix": prefix, "branches": manifest["branches"], "blocks": len(out_blocks),
            "unavailable": len(unavailable), "draft_rules_used": manifest["draft_rules_used"]}


def stall_inputs(stall: dict) -> dict:
    d = Path(stall["dir"])
    return {n: sha256_file(d / n) for n in ("stall.json", "boundary.json", "ledger.json", "recipe.json",
                                            *REVIEW_VIEWS.values()) if (d / n).exists()}


def pick_stalls(root: Path, ids: list[str] | None) -> list[dict]:
    stalls = [load_stall(d) for d in stall_dirs(root)]
    if ids:
        want = set(ids)
        stalls = [s for s in stalls if s["stall_id"] in want]
        missing = want - {s["stall_id"] for s in stalls}
        if missing:
            raise SystemExit(f"unknown stalls {sorted(missing)}")
    return stalls


def build_blocks(stalls: list[dict], arms: tuple[str, ...], repeats: range, **kw) -> tuple[list[dict], list[dict]]:
    blocks, unavailable = [], []
    for st in stalls:
        try:
            specs = [arm_spec(st, arm, r, **kw) for r in repeats for arm in arms]
        except Unavailable as exc:
            unavailable.append({"unit_id": st["stall_id"], "task_id": st["task_id"], "reason": str(exc)})
            continue
        blocks.append({"unit_id": st["stall_id"], "task_id": st["task_id"], "stall_dir": st["dir"],
                       "inputs": stall_inputs(st), "specs": specs})
    return blocks, unavailable


def calibration(path: str | None, *, need_frozen: bool) -> dict:
    if not path:
        raise SystemExit("--calibration is required")
    c = read_json(path)
    if c.get("arm_set", "v2") != CURRENT_ARM_SET:
        raise SystemExit("calibration belongs to a different arm set; do not reuse historical caps/prompts")
    if need_frozen and not c.get("frozen_at"):
        raise SystemExit("calibration is not frozen (caps.py freeze)")
    return c


def check_draft(draft: list[str], allow: bool, what: str) -> None:
    if draft and not allow:
        raise SystemExit(f"{what} would use rules the user has not ratified: {draft}. Ratify them first "
                         "(or pass --allow-draft for a mechanics test; the manifest records it).")


# ---------- commands ----------

def cmd_screen(a) -> dict:
    stalls = pick_stalls(Path(a.stalls_root), a.stall)
    rep = range(a.repeat_offset, a.repeat_offset + a.repeats)
    blocks, unav = build_blocks(stalls, SCREEN_ARMS, rep, purpose="screen", prefix=a.prefix, caps=None)
    note = ("screen branches run under the $3 runaway-spend guard only: the new-token and submission caps are "
            "calibrated on these runs (contract), so they cannot bind them")
    return write_specset(Path(a.study), a.prefix, "screen", blocks, inputs={"note": note}, draft=[], unavailable=unav,
                         replace=a.replace)


def cmd_screen_extra(a) -> dict:
    scr = read_json(a.screen_json)
    ids = [s["stall_id"] for s in scr["stalls"] if s.get("near_cutoff") and s.get("extra_resamples")]
    if not ids:
        return {"prefix": f"screen-extra-{a.round}", "branches": 0, "note": "no stall is near a cutoff"}
    stalls = pick_stalls(Path(a.stalls_root), ids)
    n = max(s["extra_resamples"] for s in scr["stalls"] if s["stall_id"] in ids)
    offset = SCREEN_REPEATS + (a.round - 1) * n
    prefix = f"screen-extra-{a.round}"
    blocks, unav = build_blocks(stalls, SCREEN_ARMS, range(offset, offset + n), purpose="screen", prefix=prefix,
                                caps=None)
    return write_specset(Path(a.study), prefix, "screen", blocks,
                         inputs={"screen_json": {"path": str(Path(a.screen_json).resolve()),
                                                 "sha256": sha256_file(a.screen_json)},
                                 "strata_rule": scr.get("strata_rule")},
                         draft=scr.get("draft_rules_used", []), unavailable=unav, replace=a.replace)


def cmd_tuning(a) -> dict:
    cal = calibration(a.calibration, need_frozen=False)
    if not cal.get("caps"):
        raise SystemExit("tuning needs the screen caps (caps.py screen-caps)")
    stalls = pick_stalls(Path(a.stalls_root), a.stall)
    prefix = f"tuning-r{a.round}"
    prompts = PROMPT_VERSIONS | (json.loads(a.prompts) if a.prompts else {})
    blocks, unav = build_blocks(stalls, TUNING_ARMS, range(a.repeats), purpose="tuning", prefix=prefix,
                                caps=cal["caps"], rewind_mode="per_channel", prompts=prompts)
    return write_specset(Path(a.study), prefix, "tuning", blocks,
                         inputs={"calibration": {"path": str(Path(a.calibration).resolve()),
                                                 "sha256": sha256_file(a.calibration)}, "prompts": prompts,
                                 "rewind_mode": "per_channel (tuning stalls: every part independently)"},
                         draft=cal.get("draft_rules_used", []), unavailable=unav, replace=a.replace)


def cmd_main(a) -> dict:
    cal = calibration(a.calibration, need_frozen=True)
    scr = read_json(a.screen_json)
    draft = sorted(set(cal.get("draft_rules_used", [])) | set(scr.get("draft_rules_used", [])))
    check_draft(draft, a.allow_draft, "the main arms")
    admitted = {s["stall_id"]: s for s in scr["stalls"] if s.get("admitted") and not s.get("tuning")}
    chosen = getattr(a, "stall", None)
    if chosen:
        missing = sorted(set(chosen) - set(admitted))
        if missing:
            raise SystemExit(f"--stall not admitted by the screen: {missing}")
        admitted = {k: v for k, v in admitted.items() if k in chosen}
    stalls = [s for s in pick_stalls(Path(a.stalls_root), None) if s["stall_id"] in admitted]
    frozen_b = {s["stall_id"]: s.get("boundary_sha256") for s in scr["stalls"]}
    blocks, unav = [], []
    for st in stalls:
        try:
            if not st["has_boundary"] or sha256_file(Path(st["dir"]) / "boundary.json") != frozen_b.get(st["stall_id"]):
                raise Unavailable("boundary.json changed after the screen (the boundary is frozen before screening)")
            if st["has_recipe"] and cal.get("recipe_cap_tokens") is not None:
                cap = read_json(Path(st["dir"]) / "recipe.json").get("cap_tokens")
                if cap != cal["recipe_cap_tokens"]:
                    raise Unavailable(f"recipe.json cap_tokens {cap} != calibrated {cal['recipe_cap_tokens']}")
            prompts = {**cal["prompt_versions"], **(json.loads(a.prompts) if getattr(a, "prompts", None) else {})}
            arms = [arm for arm in ARMS if not getattr(a, "arm", None) or arm in a.arm]
            specs = [arm_spec(st, arm, r, purpose="main", prefix=a.prefix, caps=cal["caps"],
                              rewind_mode=cal["rewind_mode_main"]["mode"], prompts=prompts,
                              opus_record=(r == 0)) for r in range(a.repeats or MAIN_REPEATS) for arm in arms]
        except Unavailable as exc:
            unav.append({"unit_id": st["stall_id"], "task_id": st["task_id"], "reason": str(exc)})
            continue
        blocks.append({"unit_id": st["stall_id"], "task_id": st["task_id"], "stall_dir": st["dir"],
                       "inputs": stall_inputs(st), "stratum": admitted[st["stall_id"]].get("stratum"),
                       "specs": specs})
    return write_specset(Path(a.study), a.prefix, "main", blocks,
                         inputs={"calibration": {"path": str(Path(a.calibration).resolve()),
                                                 "sha256": sha256_file(a.calibration)},
                                 **({"selected_stalls": sorted(chosen)} if chosen else {}),
                                 **({"selected_arms": sorted(a.arm)} if getattr(a, "arm", None) else {}),
                                 **({"prompt_overrides": json.loads(a.prompts)} if getattr(a, "prompts", None) else {}),
                                 "screen_json": {"path": str(Path(a.screen_json).resolve()),
                                                 "sha256": sha256_file(a.screen_json)}},
                         draft=draft, unavailable=unav, replace=a.replace)


def healthy_points(root: Path) -> list[dict]:
    pts = []
    for d in healthy_dirs(root):
        h = read_json(d / "healthy.json")
        h["dir"] = str(d.resolve())
        pts.append(h)
    per_task = Counter(h["task_id"] for h in pts)
    over = {t: n for t, n in per_task.items() if n > MAX_HEALTHY_PER_TASK}
    if over:
        raise SystemExit(f"more than {MAX_HEALTHY_PER_TASK} healthy checkpoints for tasks {over}")
    return pts


def cmd_harm(a) -> dict:
    cal = calibration(a.calibration, need_frozen=True)
    pts = healthy_points(Path(a.healthy_root))
    draft = sorted(set(cal.get("draft_rules_used", [])) | {r for h in pts for r in h.get("draft_rules_used", [])})
    check_draft(draft, a.allow_draft, "the harm test")
    blocks, unav = [], []
    for h in pts:
        view = Path(h["dir"]) / REVIEW_VIEWS[HARM_ARM]
        if not view.exists():
            unav.append({"unit_id": h["healthy_id"], "task_id": h["task_id"], "reason": f"{view.name} missing"})
            continue
        view_units(view)
        specs = []
        for r in range(a.repeats or HARM_REPEATS):
            if CURRENT_ARM_SET == "v3":
                st = dict(h, stall_id=h["healthy_id"], step_index=h.get("step_index", h.get("step")))
                specs.append(arm_spec(st, HARM_ARM, r, purpose="harm", prefix=a.prefix, caps=cal["caps"],
                                      rewind_mode=cal["rewind_mode_main"]["mode"], prompts=cal["prompt_versions"]))
                continue
            s = spec(purpose="harm", prefix=a.prefix, unit_id=h["healthy_id"], arm=HARM_ARM, repeat=r,
                     source={"store": h["store"], "run_id": h["run_id"], "checkpoint_id": h["checkpoint_id"]},
                     caps=cal["caps"])
            s["review"] = {"view_ref": str(view), "prompt_version": cal["prompt_versions"][HARM_ARM],
                           "rewind_mode": cal["rewind_mode_main"]["mode"], "opus_record": False}
            specs.append(s)
        blocks.append({"unit_id": h["healthy_id"], "task_id": h["task_id"], "healthy_dir": h["dir"],
                       "inputs": {"healthy.json": sha256_file(Path(h["dir"]) / "healthy.json"),
                                  view.name: sha256_file(view)}, "specs": specs})
    return write_specset(Path(a.study), a.prefix, "harm", blocks,
                         inputs={"calibration": {"path": str(Path(a.calibration).resolve()),
                                                 "sha256": sha256_file(a.calibration)}},
                         draft=draft, unavailable=unav, replace=a.replace)


def cmd_injection(a) -> dict:
    cal = calibration(a.calibration, need_frozen=True)
    pts = healthy_points(Path(a.healthy_root))
    blocks, unav, draft = [], [], set(cal.get("draft_rules_used", []))
    for h in pts:
        donors_path = Path(h["dir"]) / "injection_donors.json"
        if not donors_path.exists():
            unav.append({"unit_id": h["healthy_id"], "task_id": h["task_id"], "reason": "injection_donors.json missing"})
            continue
        donors = read_json(donors_path)
        draft |= set(donors.get("draft_rules_used", [])) | set(h.get("draft_rules_used", []))
        specs = []
        for r in range(a.repeats or INJECTION_REPEATS):
            for cond in INJECTION_CONDITIONS:
                entry = donors["conditions"].get(cond)
                if not entry or entry.get("unavailable"):
                    if r == 0:
                        unav.append({"unit_id": h["healthy_id"], "task_id": h["task_id"], "condition": cond,
                                     "reason": (entry or {}).get("unavailable", "no donor recorded")})
                    continue
                s = spec(purpose="injection", prefix=a.prefix, unit_id=h["healthy_id"], arm=f"injection:{cond}",
                         repeat=r, source={"store": h["store"], "run_id": h["run_id"],
                                           "checkpoint_id": h["checkpoint_id"]}, caps=cal["caps"])
                s["injection"] = entry["payload"]
                specs.append(s)
        blocks.append({"unit_id": h["healthy_id"], "task_id": h["task_id"], "healthy_dir": h["dir"],
                       "inputs": {"healthy.json": sha256_file(Path(h["dir"]) / "healthy.json"),
                                  "injection_donors.json": sha256_file(donors_path)}, "specs": specs})
    check_draft(sorted(draft), a.allow_draft, "the injection test")
    return write_specset(Path(a.study), a.prefix, "injection", blocks,
                         inputs={"calibration": {"path": str(Path(a.calibration).resolve()),
                                                 "sha256": sha256_file(a.calibration)}},
                         draft=sorted(draft), unavailable=unav, replace=a.replace)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "collect-episodes"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True)
        p.add_argument("--stalls-root", required=True)
    for name in ("screen", "screen-extra", "tuning", "main", "harm", "injection"):
        p = sub.add_parser(name)
        p.add_argument("--arm-set", choices=("v2", "v3"), default="v2")
        p.add_argument("--study", required=True)
        p.add_argument("--replace", action="store_true")
        if name in ("screen", "screen-extra", "tuning", "main"):
            p.add_argument("--stalls-root", required=True)
        if name in ("harm", "injection"):
            p.add_argument("--healthy-root", required=True)
        if name in ("tuning", "main", "harm", "injection"):
            p.add_argument("--calibration", required=True)
        if name in ("main", "harm", "injection"):
            p.add_argument("--repeats", type=int, help="explicit repeats per condition (required for v3)")
            p.add_argument("--allow-draft", action="store_true")
            p.add_argument("--prefix", default=name)
        if name == "screen":
            p.add_argument("--prefix", default="screen")
            p.add_argument("--stall", action="append")
            p.add_argument("--repeats", type=int, default=SCREEN_REPEATS)
            p.add_argument("--repeat-offset", type=int, default=0)
        if name in ("screen-extra", "main"):
            p.add_argument("--screen-json", required=True)
        if name == "main":
            p.add_argument("--stall", action="append",
                           help="only these admitted stalls (development pilot selection); default every admitted stall")
            p.add_argument("--arm", action="append", help="only these arms (follow-up variants); default all six")
            p.add_argument("--prompts", help="JSON {arm: prompt_version} overriding the frozen prompt versions")
        if name in ("screen-extra", "tuning"):
            p.add_argument("--round", type=int, required=True)
        if name == "tuning":
            p.add_argument("--stall", action="append", required=True)
            p.add_argument("--repeats", type=int, required=True, help="per arm per round (not ratified; equal rounds)")
            p.add_argument("--prompts", help="JSON {arm: prompt_version} overriding the current drafts")
    a = ap.parse_args(argv)
    configure_arm_set(getattr(a, "arm_set", "v2"))
    if CURRENT_ARM_SET == "v3" and a.cmd in ("main", "harm", "injection") and not a.repeats:
        raise SystemExit("v3 requires explicit --repeats; pilot agreement is 2, main study count is not frozen")
    if a.cmd == "collect":
        r = collect(Path(a.store), Path(a.stalls_root))
    elif a.cmd == "collect-episodes":
        r = collect_episodes(Path(a.store), Path(a.stalls_root))
    else:
        r = {"screen": cmd_screen, "screen-extra": cmd_screen_extra, "tuning": cmd_tuning, "main": cmd_main,
             "harm": cmd_harm, "injection": cmd_injection}[a.cmd](a)
    print(json.dumps(r, indent=1))
    return r


if __name__ == "__main__":
    main()
