"""End-to-end check of stall judge v2 (configs/v2/judge_v2) on real stored natural runs, with a scripted judge model
only: no model calls, no Docker, nothing spent.

Stores:
  TB  the real Terminal-Bench v2 screen store (8 DeepSeek V4.1 Flash natural-v2 runs, 25-80 steps, 2026-10-07),
      judged through a hard-linked scratch copy so the durable store records go to the copy, never the original;
  M5  the context-limit engine store (m5_context_limit.py run N: hidden outputs and a failing submission with
      "Hidden tests still failing: 1."), when present;
  VAL three of the 71 validation runs (earlier lanes, request-sourced records, rejection feedback) for the replay
      harness, plus the 20 hand-label moments.
Checks:
  points       judge points are exactly steps 10, 20, ... up to each run's last step, counted independently
  prefix       the record at point k holds steps 1..k, ends at STEP k, comes from the point-k checkpoint (or the
               request of step k+1) cut before step k+1's request, and is a prefix of the record at k+10
  private      no request sent to the judge holds grade or evaluation text, hidden test names, private solution or
               test lines, or the v1 grade/reference markers, unless the worker itself saw that text before the cut;
               every text piece of every record is found in a worker-visible blob up to the cut; building records
               reads no grade, evaluation, raw tool-result or raw provider-response blob
  requests     every request is the v2 system prompt + "RECORD" + the saved record + the closing instruction, to the
               pinned GLM settings
  onset        valid onsets give the lower median; after-point, missing and unreadable onsets fall back to the last
               submission or, without one, the judge point, with the reasons logged
  votes        3 votes, majority stall; errors are retried (3 attempts) then no_answer; error bodies, refusals and
               unparseable answers are not stall votes; an answer after a continued transcript is parsed
  episodes     consecutive confirmed points form one episode; an unconfirmed or unjudged point ends it; every episode
               is kept (several per run); P comes from the episode's first point; the branch source is the
               checkpoint at the first confirmed point and stall_onset_point (arm 4) is the checkpoint after step P-1,
               both branchable or the episode is flagged
  records      durable stall_judge_v2 / stall_episode records are written once, a rerun changes nothing and makes no
               new judge calls; arms_v2.py collect-episodes builds stall dirs from them
  guard        a zero spend cap makes no call and writes no record
  replay       the replay harness judges and scores validation runs and hand-label items, refuses a Claude judge, and
               writes a v2 hand-label packet without private grades

Usage: python configs/v2/engine_e2e/m6_judge_v2.py
Writes artifacts/v2/engine/m6-judge-v2.json in the main checkout; scratch under ~/.cache/trajectory-retention/v2-engine/m6.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import ARTIFACTS, ROOT, write

J2 = ROOT / "configs/v2/judge_v2"
sys.path.insert(0, str(J2))
import judging  # noqa: E402
import record as R  # noqa: E402

PY = sys.executable
HOME = Path.home()
SCRATCH = HOME / ".cache/trajectory-retention/v2-engine/m6"
TB = HOME / ".cache/trajectory-retention/v2-screen/tbv2-20261007/screen-stage1-1-r4/store"
M5 = HOME / ".cache/trajectory-retention/v2-engine/m5/N/run/store"
PRIVATE = HOME / ".cache/trajectory-retention/terminalbench-v2/private"
OLD_PACKET = HOME / "projects/llm-trajectories/artifacts/calibration/detector-judge-20260926-v1/hand-label/packet"
MARKERS = ("PRIVATE GRADE", "REFERENCE FIX", "research_private", "evaluation_id", "passing_checks", "full_solve",
           "the agent never sees", "the agent never saw")
GRADE_ID = re.compile(r"grade_[0-9a-f]{32}")

CHECKS: dict[str, object] = {}


def check(name: str, ok: bool, detail=None) -> None:
    CHECKS[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or detail is None else f"  {str(detail)[:400]}"), flush=True)


def J(verdict, onset=None, evidence="scripted"):
    return json.dumps({"verdict": verdict, "onset_step": onset, "evidence": evidence})


ERR = {"error": "scripted transport failure"}
TB_SPEC = {
    "default": {"stall": [J("not_stall")] * 3, "type": json.dumps({"type": "stuck", "evidence": "scripted"})},
    "points": {
        "83e49cfe@20": {"stall": [J("stall", 14), J("stall", "step 12"), [ERR, ERR, ERR]],
                        "type": json.dumps({"type": "over-commit", "evidence": "scripted"})},
        "83e49cfe@30": {"stall": [J("stall", 99), J("stall", None), "the agent then ran STEP 31 and"]},
        "83e49cfe@40": {"stall": [J("not_stall"), J("not_stall"), {"content": None, "refusal": "I cannot help."}]},
        "83e49cfe@50": {"stall": [[ERR, ERR, J("stall", 45)], J("stall", 47), J("unclear")]},
        "83e49cfe@60": {"stall": ["===== STEP 61 =====\n[reasoning]\nI will now...\n" + J("stall", "forty"),
                                  J("stall", "forty"), [{"error_body": {"code": 502, "message": "upstream"}}]]},
        "83e49cfe@70": {"stall": ["garbage", "{not json", "verdict: stall"]},
        "83e49cfe@80": {"stall": [J("stall", 75)] * 3},
        "d38d7a4a@60": {"stall": [J("stall", None), J("stall", None), J("not_stall")]},
        "1bc4057f@10": {"stall": [J("stall", 1)] * 3},
        "1bc4057f@20": {"stall": [J("stall", 3)] * 3},
        "aba4faca@40": {"stall": [J("stall", 33), J("stall", 33), J("not_stall")]},
    }}
EXPECT_EPISODES = {  # run id fragment -> [(points, P, P source, fallback)]
    "83e49cfe": [([20, 30], 12, "judge", None), ([50, 60], 45, "judge", None), ([80], 75, "judge", None)],
    "d38d7a4a": [([60], 60, "fallback", "last_submission")],
    "1bc4057f": [([10, 20], 1, "judge", None)],
    "aba4faca": [([40], 33, "judge", None)],
}
EXPECT_POINTS = {  # run fragment -> point -> (votes, confirmed, onset, source, fallback_to)
    "83e49cfe": {20: (["stall", "stall", "no_answer"], True, 12, "judge", None),
                 30: (["stall", "stall", "unparseable"], True, 30, "fallback", "judge_point_no_submission"),
                 40: (["not_stall", "not_stall", "refused"], False, None, None, None),
                 50: (["stall", "stall", "unclear"], True, 45, "judge", None),
                 60: (["stall", "stall", "no_answer"], True, 60, "fallback", "judge_point_no_submission"),
                 70: (["unparseable"] * 3, False, None, None, None),
                 80: (["stall"] * 3, True, 75, "judge", None)},
    "d38d7a4a": {60: (["stall", "stall", "not_stall"], True, 60, "fallback", "last_submission")},
}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=ROOT, **kw)


def hardlink_copy(src: Path, dst: Path) -> None:
    """A store copy whose blobs are hard links (same volume): new records and blobs land only in the copy."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "blobs").mkdir(parents=True)
    for suf in ("", "-wal", "-shm"):
        p = src / f"journal.sqlite3{suf}"
        if p.exists():
            shutil.copy2(p, dst / p.name)
    for d in (src / "blobs").iterdir():
        if d.is_dir():
            (dst / "blobs" / d.name).mkdir()
            for f in d.iterdir():
                os.link(f, dst / "blobs" / d.name / f.name)


# ---------- worker-visible corpus and private texts ----------

def _strings(value, out: set) -> None:
    if isinstance(value, str):
        out.add(value)
        s = value.lstrip()
        if s[:1] in "{[":
            try:
                _strings(json.loads(value), out)
            except ValueError:
                pass
    elif isinstance(value, dict):
        for v in value.values():
            _strings(v, out)
    elif isinstance(value, list):
        for v in value:
            _strings(v, out)


def corpus_by_cut(run: R.RunView) -> list[tuple[int, set, str]]:
    """[(event sequence, decoded strings, the same joined)] of every worker-visible blob the run's events carry (not checkpoint
    contexts: those are what the record is built from, so they cannot vouch for it)."""
    out = []
    for e in run.events:
        for ref in (e.input_ref, e.delivered_ref, e.output_ref):
            if ref is None or getattr(ref.visibility, "value", ref.visibility) != "worker_visible":
                continue
            raw = run.store.get_bytes(ref).decode("utf-8", "replace")
            s = {raw}
            try:
                _strings(json.loads(raw), s)
            except ValueError:
                pass
            out.append((e.sequence, s, "\x00".join(s)))
    return out


def private_texts(run: R.RunView) -> dict[str, set]:
    """Text the judge must never get unless the worker saw it: evaluation ids and hidden test names from the grade
    blobs, and distinctive lines of the lane's private solution and tests."""
    ids, names = set(), set()
    for e in run.events:
        if e.kind == "grade":
            ids.add(e.payload.get("evaluation_id") or "")
            ref = e.payload.get("evaluation_ref")
            if ref:
                ev = json.loads((run.store.blobs / ref["sha256"][:2] / ref["sha256"][2:]).read_bytes())
                for c in ev.get("checks") or []:
                    cid = str(c.get("check_id") or "")
                    names.add(cid)
                    names.add(cid.split("::")[-1])
    lines = set()
    d = PRIVATE / (run.task or "")
    for sub in ("solution", "tests"):
        for f in (d / sub).rglob("*") if (d / sub).is_dir() else ():
            if f.is_file():
                for ln in f.read_text(errors="replace").splitlines():
                    if len(ln.strip()) >= 40:
                        lines.add(ln.strip())
    return {"evaluation_ids": {i for i in ids if i}, "check_names": {n for n in names if len(n) >= 8},
            "private_lines": lines}


def seen_before(corpus, cut: int, text: str) -> bool:
    """The text is one of, or inside one of, the decoded strings of a worker-visible blob up to the cut."""
    return any(seq <= cut and (text in strings or text in joined) for seq, strings, joined in corpus)


# ---------- TB checks ----------

def check_tb(out: Path, copy: Path) -> dict:
    data = json.loads((out / "stall_records.json").read_text())
    runs = {r["run_id"]: r for r in data["runs"]}
    a = R.open_readonly(copy)
    views = {rid: R.RunView(a, rid) for rid in R.natural_runs(a)}
    # points
    bad = {}
    for rid, v in views.items():
        n = sum(1 for e in v.events if e.kind == "model_response")
        want = list(range(10, n + 1, 10))
        got = [p["point"] for p in runs[rid]["points"]]
        if got != want:
            bad[rid] = (got, want)
    check("points_every_10_steps_up_to_last_step", not bad and len(runs) == 8, bad or None)
    check("points_count_tb", sum(len(r["points"]) for r in runs.values()) == 35,
          sum(len(r["points"]) for r in runs.values()))
    no_rec = [(rid, p["point"]) for rid, r in runs.items() for p in r["points"] if p.get("error")]
    check("missing_context_point_is_recorded_not_judged",
          [(r.split(":")[1][:8], k) for r, k in no_rec] == [("aba4faca", 50)], no_rec)

    # instrument reads while rebuilding every record
    forbidden = set()
    for v in views.values():
        for e in v.events:
            if e.kind == "grade" and e.payload.get("evaluation_ref"):
                forbidden.add(e.payload["evaluation_ref"]["sha256"])
            if e.kind in ("tool_result", "model_attempt") and e.output_ref is not None:
                forbidden.add(e.output_ref.sha256)
        for c in v.checkpoints:
            if c.evaluation_ref is not None:
                forbidden.add(c.evaluation_ref.sha256)
    read: set = set()
    orig = a.get_bytes

    def spy(ref):
        read.add(ref.sha256)
        return orig(ref)
    a.get_bytes = spy
    recs = {}
    for rid, v in views.items():
        for k in v.points():
            try:
                recs[(rid, k)] = v.record(k)
            except LookupError:
                pass
    a.get_bytes = orig
    check("record_building_reads_no_grade_evaluation_or_raw_blob", not (read & forbidden),
          sorted(read & forbidden)[:5] or None)

    # prefix-bound
    prob = []
    for (rid, k), rec in recs.items():
        v = views[rid]
        heads = re.findall(r"^===== STEP (\d+) =====$", rec.render(), flags=re.M)
        if len(rec.steps) != k or heads[-1:] != [str(k)] or len(heads) != k or rec.issues:
            prob.append((rid[-30:], k, "steps", len(rec.steps), heads[-1:], rec.issues[:2]))
        nxt = v.requests[k].sequence if k < len(v.requests) else None
        if nxt is not None and rec.cut_sequence >= nxt:
            prob.append((rid[-30:], k, "cut not before the next request", rec.cut_sequence, nxt))
        if rec.source["kind"] == "point_checkpoint_context":
            cp = v.point_checkpoint(k)
            if (cp.validation or {}).get("step_index") != k or cp.event_sequence != rec.cut_sequence:
                prob.append((rid[-30:], k, "checkpoint step", (cp.validation or {}).get("step_index")))
        later = recs.get((rid, k + 10))
        if later is not None and not later.render().startswith(rec.render()):
            prob.append((rid[-30:], k, "not a prefix of the record at k+10"))
    check("records_are_prefix_bound", not prob and len(recs) == 34, prob[:5] or len(recs))
    v2src = {rec.source["kind"] for rec in recs.values()}
    check("v2_records_come_from_point_checkpoint_contexts", v2src == {"point_checkpoint_context"}, sorted(v2src))
    # the point-k context equals the worker's own next request (k+1), message for message
    same, diff = 0, []
    for (rid, k), rec in recs.items():
        v = views[rid]
        if k < len(v.requests):
            body = v.store.get_json(v.requests[k].input_ref)
            parsed = R.parse_messages(body["messages"], v._texts("format_error", v.requests[k].sequence),
                                      v._texts("native_submission_feedback", v.requests[k].sequence))
            if parsed[0] == rec.head and [(s.k, s.fields) for s in parsed[1]] == [(s.k, s.fields) for s in rec.steps]:
                same += 1
            else:
                diff.append((rid[-30:], k))
    check("point_context_equals_the_next_serialized_request", not diff and same > 0, diff[:5] or same)

    # requests actually sent
    log = [json.loads(l) for l in (out / "scripted-requests.jsonl").read_text().splitlines()]
    stall_sys, stall_close, _ = judging.load_prompt("stall")
    type_sys, type_close, _ = judging.load_prompt("type")
    glm = json.loads((J2 / "judge-glm.json").read_text())
    shape = []
    for entry in log:
        b, c = entry["body"], entry["ctx"]
        sysmsg, user = b["messages"][0]["content"], b["messages"][1]["content"]
        rec_file = out / "records" / judging_key(c["run_id"]) / f"p{c['point']:04d}.txt"
        want_sys, want_close = (stall_sys, stall_close) if c["call"] == "stall" else (type_sys, type_close)
        ok = (b["model"] == glm["model"] and b["provider"] == glm["provider"] and b["reasoning"] == glm["reasoning"]
              and b["max_tokens"] == glm["max_tokens"] and sysmsg == want_sys and len(b["messages"]) == 2
              and user.startswith("RECORD\n\n") and user.endswith(want_close.replace("{point}", str(c["point"]))))
        if c["call"] == "stall":
            ok = ok and user == rec_file.read_text()
        else:
            ok = ok and user.removesuffix(type_close.replace("{point}", str(c["point"]))) == \
                rec_file.read_text().removesuffix(stall_close.replace("{point}", str(c["point"])))
        if not ok:
            shape.append((c["run_id"][-30:], c["point"], c["call"]))
    check("requests_are_prompt_record_closing_to_pinned_glm", not shape and log, shape[:5] or len(log))

    # privacy of every request
    leaks = []
    corpora, priv = {}, {}
    for entry in log:
        c = entry["ctx"]
        rid, k = c["run_id"], c["point"]
        v = views[rid]
        if rid not in corpora:
            corpora[rid], priv[rid] = corpus_by_cut(v), private_texts(v)
        user = entry["body"]["messages"][1]["content"] + entry["body"]["messages"][0]["content"]
        cut = recs[(rid, k)].cut_sequence
        for m in MARKERS:  # the worker's own text may hold a marker word (e.g. a file full_solver2.py)
            if m in user and not seen_before(corpora[rid], cut, m):
                leaks.append((rid[-30:], k, "marker", m))
        for g in set(GRADE_ID.findall(user)):
            if not seen_before(corpora[rid], cut, g):
                leaks.append((rid[-30:], k, "grade id", g))
        for kind in ("evaluation_ids", "check_names", "private_lines"):
            for t in priv[rid][kind]:
                if t in user and not seen_before(corpora[rid], cut, t):
                    leaks.append((rid[-30:], k, kind, t[:80]))
    check("no_private_text_in_any_judge_request", not leaks, leaks[:8] or None)
    # negative control: the same leak test on the v1 judge's record (render.py, with private grade lines) must fire
    sys.path.insert(0, str(ROOT / "configs/detector-judge-20260926"))
    import storeio
    from render import render_prefix
    rid0 = next(r for r in views if "83e49cfe" in r)
    v1_text = render_prefix(storeio.Run(storeio.Store(copy), rid0, "m6"), recs[(rid0, 20)].cut_sequence)
    v1_hits = [m for m in MARKERS if m in v1_text and not seen_before(corpora[rid0], recs[(rid0, 20)].cut_sequence, m)]
    v1_hits += [n for n in priv[rid0]["check_names"] if n in v1_text
                and not seen_before(corpora[rid0], recs[(rid0, 20)].cut_sequence, n)]
    check("leak_test_fires_on_the_v1_record_with_grades", bool(v1_hits), v1_hits[:5])
    check("private_texts_were_present_to_test_against",
          all(priv[r]["evaluation_ids"] and priv[r]["check_names"] and priv[r]["private_lines"] for r in priv),
          {r[-30:]: {k: len(v) for k, v in p.items()} for r, p in priv.items()})
    unproven = []
    for (rid, k), rec in recs.items():
        corpora.setdefault(rid, corpus_by_cut(views[rid]))
        for t in rec.texts():
            if t and not seen_before(corpora[rid], rec.cut_sequence, t):
                unproven.append((rid[-30:], k, t[:60]))
                break
    check("every_record_piece_is_worker_visible_before_the_cut", not unproven, unproven[:5] or None)

    # votes, onsets, fallbacks
    def point(frag, k):
        r = next(r for rid, r in runs.items() if frag in rid)
        return next(p for p in r["points"] if p["point"] == k)
    wrong = []
    for frag, pts in EXPECT_POINTS.items():
        for k, (votes, conf, onset, src, fb) in pts.items():
            p = point(frag, k)
            got = (p["votes"], p["confirmed"], p.get("onset"), p.get("onset_source"), p.get("fallback_to"))
            if got != (votes, conf, onset, src, fb):
                wrong.append((frag, k, got))
    check("votes_onsets_and_fallbacks_as_scripted", not wrong, wrong or None)
    p30, p60 = point("83e49cfe", 30), point("83e49cfe", 60)
    check("fallback_reasons_logged", p30["onset_unusable"] == ["after_point(99)", "missing"]
          and p60["onset_unusable"] == ["unreadable", "unreadable"], (p30["onset_unusable"], p60["onset_unusable"]))
    check("default_points_not_confirmed",
          all(not p.get("confirmed") for rid, r in runs.items() for p in r["points"]
              if not any(f in rid for f in EXPECT_EPISODES) and not p.get("error")))

    def receipt(frag, k, vote):
        p = point(frag, k)
        return json.loads((out / p["receipts"][vote - 1]).read_text())
    r1, r2, r3 = receipt("83e49cfe", 20, 3), receipt("83e49cfe", 50, 1), receipt("83e49cfe", 60, 3)
    check("errors_retried_then_no_answer", len(r1["attempts"]) == 3 and all("error" in x for x in r1["attempts"])
          and r1["final"] == "no_answer", [list(x) for x in r1["attempts"]])
    check("answer_after_two_errors_counts", len(r2["attempts"]) == 3 and r2["final"] == "stall" and r2["onset"] == 45)
    check("error_bodies_retried_then_no_answer", len(r3["attempts"]) == 3 and r3["final"] == "no_answer"
          and all((x.get("response") or {}).get("error") for x in r3["attempts"]))
    r4 = receipt("83e49cfe", 60, 1)
    check("answer_after_continued_transcript_is_parsed", r4["final"] == "stall" and r4["onset_status"] == "unreadable")
    check("refusal_recorded", point("83e49cfe", 40)["refused_votes"] == 1
          and runs[next(r for r in runs if "83e49cfe" in r)]["votes"]["refused"] == 1)
    check("receipts_hold_no_api_key", not any("Bearer" in f.read_text() or "sk-or" in f.read_text()
                                              for f in (out / "calls").rglob("*.json")))

    # episodes
    wrong = []
    for frag, eps in EXPECT_EPISODES.items():
        r = next(r for rid, r in runs.items() if frag in rid)
        got = [(e["points"], e["onset"], e["onset_source"], e["fallback_to"]) for e in r["episodes"]]
        if got != [tuple(x) for x in eps]:
            wrong.append((frag, got))
    check("episodes_as_declared", not wrong, wrong or None)
    eall = [e for r in runs.values() for e in r["episodes"]]
    check("every_episode_kept_several_per_run", len(eall) == 6
          and len(next(r for rid, r in runs.items() if "83e49cfe" in rid)["episodes"]) == 3)
    ends = {e["episode_id"][-40:]: (e["started_after"], e["ended_by"]) for e in eall}
    check("episode_starts_and_ends_logged",
          [x for r in runs.values() if any(f in r["run_id"] for f in ("83e49cfe",)) for x in
           [(e["started_after"], e["ended_by"]) for e in r["episodes"]]] ==
          [("run_start", "unconfirmed_point"), ("unconfirmed_point", "unconfirmed_point"),
           ("unconfirmed_point", "run_end")]
          and next(e for e in eall if "aba4faca" in e["episode_id"])["ended_by"] == "unjudged_point", ends)
    src, onset_pts, flags = [], [], []
    for r in runs.values():
        v = views[r["run_id"]]
        for e in r["episodes"]:
            cp = v.point_checkpoint(e["first_point"])  # the stall moment: arms 1, 2, 3, 5, 6 start here
            if e["checkpoint_id"] != cp.checkpoint_id or e["step_index"] != e["first_point"] \
                    or (cp.validation or {}).get("step_index") != e["first_point"] \
                    or e["source_branchable"] != bool(cp.branchable):
                src.append(e["episode_id"][-40:])
            oc = v.point_checkpoint(e["onset"] - 1)  # arm 4's rewind target: the state just before step P ran
            if e["stall_onset_point"] != e["onset"] - 1 or e["stall_onset_checkpoint_id"] != oc.checkpoint_id \
                    or (oc.validation or {}).get("step_index") != e["onset"] - 1 \
                    or e["stall_onset_branchable"] != bool(oc.branchable):
                onset_pts.append(e["episode_id"][-40:])
            both = bool(cp.branchable) and bool(oc.branchable)
            if e["branchable"] != both or (not both and not e.get("not_branchable_reason")):
                flags.append(e["episode_id"][-40:])
    check("branch_source_is_the_first_confirmed_point_checkpoint", not src, src or None)
    check("stall_onset_point_is_the_checkpoint_after_step_P_minus_1", not onset_pts, onset_pts or None)
    p1 = next(e for e in eall if "1bc4057f" in e["episode_id"])
    check("onset_at_step_1_rewinds_to_the_initial_checkpoint", p1["onset"] == 1 and p1["stall_onset_point"] == 0
          and p1["stall_onset_checkpoint_id"].endswith("/C0"), p1["stall_onset_checkpoint_id"])
    check("episode_branchable_only_when_source_and_onset_point_are_flagged_otherwise", not flags,
          {e["episode_id"][-30:]: (e["source_branchable"], e["stall_onset_branchable"], e.get("not_branchable_reason"))
           for e in eall})
    types = {e["episode_id"]: e["type"] for e in eall}
    e1 = next(k for k in types if "83e49cfe" in k and k.endswith("/E1"))
    check("one_type_call_per_episode_without_reference",
          sum(1 for x in log if x["ctx"]["call"] == "type") == 6 and types[e1] == "over-commit"
          and all(t == "stuck" for k, t in types.items() if k != e1), {k[-40:]: t for k, t in types.items()})
    bs = json.loads((out / "branch_sources.json").read_text())
    check("branch_sources_are_the_branchable_episodes", len(bs) == sum(e["branchable"] for e in eall)
          and {b["episode_id"] for b in bs} == {e["episode_id"] for e in eall if e["branchable"]}, len(bs))
    costs = all("cost_usd" in p for r in runs.values() for p in r["points"] if not p.get("error")) and \
        all("cost_usd" in e for e in eall)
    check("costs_recorded_per_point_episode_and_run", costs and all("cost_usd" in r for r in runs.values()))
    return {"runs": len(runs), "points": sum(len(r["points"]) for r in runs.values()), "episodes": len(eall),
            "requests": len(log), "branch_sources": len(bs), "summary": data["summary"]}


def judging_key(run_id: str) -> str:
    import hashlib
    return hashlib.sha256(run_id.encode()).hexdigest()[:12]


def main() -> None:
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for d in ("tb-out", "tb-guard", "m5-out", "val-out", "stalls", "packet-v2"):
        shutil.rmtree(SCRATCH / d, ignore_errors=True)
    report: dict = {"check": "m6-judge-v2", "stores": {"TB": str(TB), "M5": str(M5) if M5.exists() else None}}
    spec = SCRATCH / "tb-spec.json"
    spec.write_text(json.dumps(TB_SPEC, indent=1))
    copy = SCRATCH / "tb-store"
    hardlink_copy(TB, copy)
    before = {f.name for f in (TB / "blobs").iterdir()}
    post = [PY, J2 / "postrun_v2.py", "--store", copy, "--out", SCRATCH / "tb-out", "--scripted", spec,
            "--write-store-records"]
    p = run(post)
    check("postrun_exit_ok", p.returncode == 0, p.stderr[-2000:])
    report["tb"] = check_tb(SCRATCH / "tb-out", copy)

    # durable records, rerun, collect-episodes
    from trajectory_retention.store import ArtifactStore
    live = ArtifactStore(copy)
    n_run, n_ep = len(live.list_records("stall_judge_v2")), len(live.list_records("stall_episode"))
    check("durable_records_written_to_the_copy", n_run == 8 and n_ep == 6, (n_run, n_ep))
    import sqlite3
    with sqlite3.connect(f"file:{TB / 'journal.sqlite3'}?mode=ro", uri=True) as db:
        orig_n = db.execute("select count(*) from records where kind in ('stall_judge_v2','stall_episode')").fetchone()[0]
    check("original_store_untouched", orig_n == 0 and {f.name for f in (TB / "blobs").iterdir()} == before, orig_n)
    n_log = len((SCRATCH / "tb-out/scripted-requests.jsonl").read_text().splitlines())
    p = run(post)
    s2 = json.loads((SCRATCH / "tb-out/stall_records.json").read_text())["summary"]["store_records"]
    n_log2 = len((SCRATCH / "tb-out/scripted-requests.jsonl").read_text().splitlines())
    check("rerun_is_idempotent", p.returncode == 0 and n_log2 == n_log
          and all(not v["written"] and not v["refused"] and v["already"] for v in s2.values()), (n_log, n_log2))
    p = run([PY, ROOT / "configs/v2/arms_v2.py", "collect-episodes", "--store", copy, "--stalls-root",
             SCRATCH / "stalls"])
    col = json.loads(p.stdout) if p.returncode == 0 else {"error": p.stderr[-1500:]}
    check("collect_episodes_builds_stall_dirs", p.returncode == 0
          and len(col["stalls"]) + len(col["skipped"]) == 6 and len(col["stalls"]) == report["tb"]["branch_sources"]
          and all((SCRATCH / "stalls" / s / "stall.json").exists() for s in col["stalls"]), col)
    sj = [json.loads((SCRATCH / "stalls" / s / "stall.json").read_text()) for s in col.get("stalls", [])]
    bsrc = {b["episode_id"]: b for b in json.loads((SCRATCH / "tb-out/branch_sources.json").read_text())}
    check("stall_dirs_carry_source_and_stall_onset_point", bool(sj) and all(
        x["checkpoint_id"] == bsrc[x["episode_id"]]["checkpoint_id"]
        and x["step_index"] == bsrc[x["episode_id"]]["judge_point"]
        and x["stall_onset_point"] == bsrc[x["episode_id"]]["stall_onset_point"] == x["onset_step"] - 1
        and x["stall_onset_checkpoint_id"] == bsrc[x["episode_id"]]["stall_onset_checkpoint_id"] for x in sj),
        [(x["step_index"], x["stall_onset_point"]) for x in sj])
    report["collect_episodes"] = col

    # spend guard
    p = run([PY, J2 / "postrun_v2.py", "--store", copy, "--out", SCRATCH / "tb-guard", "--scripted", spec,
             "--run", "63d7c52b", "--cap-usd", "0", "--write-store-records"])
    g = json.loads((SCRATCH / "tb-guard/stall_records.json").read_text())
    check("zero_cap_makes_no_call_and_no_record", p.returncode == 0 and g["runs"][0]["status"] == "spend_guard_stopped"
          and not (SCRATCH / "tb-guard/scripted-requests.jsonl").exists() and not g["summary"]["store_records"],
          g["runs"][0]["status"])
    p = run([PY, J2 / "postrun_v2.py", "--store", copy, "--out", SCRATCH / "tb-paid-refused", "--run", "63d7c52b"])
    check("paid_postrun_refused_while_prompts_are_draft", p.returncode != 0 and "DRAFT" in p.stderr, p.stderr[-300:])

    # M5: hidden outputs read back, count feedback, fallback to the last submission
    if M5.exists():
        m5spec = SCRATCH / "m5-spec.json"
        m5spec.write_text(json.dumps({"default": {"stall": [J("stall", None), J("stall", "soon"), J("not_stall")],
                                                  "type": json.dumps({"type": "stuck"})}}))
        p = run([PY, J2 / "postrun_v2.py", "--store", M5, "--out", SCRATCH / "m5-out", "--scripted", m5spec])
        d = json.loads((SCRATCH / "m5-out/stall_records.json").read_text())
        pt = d["runs"][0]["points"][0]
        txt = (SCRATCH / "m5-out" / pt["record_file"]).read_text()
        a = R.open_readonly(M5)
        rec = R.RunView(a, d["runs"][0]["run_id"]).record(10)
        check("m5_hidden_outputs_read_back_from_the_record", rec.restored_outputs and "[output removed]" not in txt,
              rec.restored_outputs)
        check("m5_count_feedback_shown", "----- submission feedback delivered to the agent -----\nHidden tests still "
              "failing: 1." in txt)
        check("m5_fallback_to_last_submission", pt["onset"] == 6 and pt["onset_source"] == "fallback"
              and pt["fallback_to"] == "last_submission" and pt["onset_unusable"] == ["missing", "unreadable"], pt)
        report["m5"] = {"point": {k: pt[k] for k in ("point", "votes", "onset", "onset_source", "fallback_to")},
                        "restored_outputs": rec.restored_outputs}
    else:
        report["m5"] = "store absent (run configs/v2/engine_e2e/m5_context_limit.py to rebuild it)"

    # replay harness
    vspec = SCRATCH / "val-spec.json"
    vspec.write_text(json.dumps({"default": {"stall": [J("not_stall")] * 3, "type": json.dumps({"type": "stuck"})},
                                 "points": {"226ee5c2@50": {"stall": [J("stall", 45)] * 3},
                                            "226ee5c2@60": {"stall": [J("stall", 45), J("stall", 47), J("unclear")]},
                                            "1158b70f@30": {"stall": [J("stall", None)] * 3},
                                            "0df4be11@10": {"stall": [J("stall", 3)] * 3},
                                            "e2d1ab49@20": {"stall": [J("stall", 12)] * 3}}}))
    rp = J2 / "replay_v2.py"
    p = run([PY, rp, "judge", "--settings", J2 / "judge-glm.json", "--out", SCRATCH / "val-out", "--cap-usd", "1",
             "--scripted", vspec, "--run", "226ee5c2", "--run", "1158b70f", "--run", "30ecdf59"])
    check("replay_judge_exit_ok", p.returncode == 0, p.stderr[-1500:])
    vd = json.loads((SCRATCH / "val-out/z-ai-glm-5.3-flash/stall_records.json").read_text())
    check("replay_points_every_10_steps", sum(len(r["points"]) for r in vd["runs"]) == 13
          and {r["run_id"].split(":")[1][:8]: [p["point"] for p in r["points"]] for r in vd["runs"]}
          == {"226ee5c2": [10, 20, 30, 40, 50, 60, 70, 80], "1158b70f": [10, 20, 30], "30ecdf59": [10, 20]},
          [len(r["points"]) for r in vd["runs"]])
    f1158 = next(p for r in vd["runs"] if "1158b70f" in r["run_id"] for p in r["points"] if p["point"] == 30)
    check("replay_fallback_to_last_submission_on_rejection_feedback", f1158["onset"] == 29
          and f1158["fallback_to"] == "last_submission", f1158.get("onset"))
    vlog = [json.loads(l) for l in (SCRATCH / "val-out/z-ai-glm-5.3-flash/scripted-requests.jsonl").read_text().splitlines()]
    vleak = [(x["ctx"]["run_id"][-30:], x["ctx"]["point"], m) for x in vlog for m in MARKERS
             if m in x["body"]["messages"][1]["content"]]
    check("replay_requests_hold_no_private_markers", not vleak and vlog, vleak[:5] or len(vlog))
    p = run([PY, rp, "judge", "--settings", J2 / "judge-glm.json", "--out", SCRATCH / "val-out", "--cap-usd", "1",
             "--scripted", vspec, "--hand-label"])
    hd = json.loads((SCRATCH / "val-out/z-ai-glm-5.3-flash-hand-label/stall_records.json").read_text())
    items = hd["summary"]["items"]
    check("hand_label_items_judged_at_their_steps", p.returncode == 0 and len(items) == 20
          and all(any(pp["point"] == it["step"] for r in hd["runs"] if r["run_id"] == it["run_id"] for pp in r["points"])
                  for it in items), p.stderr[-800:])
    labels = SCRATCH / "fake-human-labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "verdict", "onset step if stall", "type", "note"])
        for it in items:
            w.writerow([it["item"], "stall" if int(it["item"]) <= 6 else "not_stall",
                        str(max(1, it["step"] - 1)) if int(it["item"]) <= 6 else "", "", "fake, for the E2E only"])
    p = run([PY, rp, "score", "--out", SCRATCH / "val-out", "--human", labels])
    sc = json.loads((SCRATCH / "val-out/scores.json").read_text()) if p.returncode == 0 else {}
    glm = sc.get("z-ai-glm-5.3-flash", {}).get("overall", {})
    hum = sc.get("z-ai-glm-5.3-flash-hand-label", {})
    check("replay_score_against_hindsight", p.returncode == 0 and glm.get("points") == 13
          and "cohen_kappa" in glm and "onset_error_steps_all_confirmed_genuine_points" in glm,
          {k: glm.get(k) for k in ("points", "confusion", "cohen_kappa", "recall", "precision_inside_genuine")}
          or p.stderr[-800:])
    check("replay_score_against_human_labels", hum.get("labelled_items") == 20 and "kappa_human_vs_judge_v2" in hum,
          {k: hum.get(k) for k in ("labelled_items", "kappa_human_vs_judge_v2", "kappa_human_vs_hindsight")})
    claude = SCRATCH / "claude-settings.json"
    s = json.loads((J2 / "judge-glm.json").read_text())
    claude.write_text(json.dumps(s | {"model": "anthropic/claude-haiku-5.5"}))
    p = run([PY, rp, "judge", "--settings", claude, "--out", SCRATCH / "val-out", "--cap-usd", "1", "--scripted", vspec])
    check("replay_refuses_a_claude_judge", p.returncode != 0 and "Claude" in p.stderr, p.stderr[-300:])
    p = run([PY, rp, "packet", "--dest", SCRATCH / "packet-v2"])
    items_v2 = sorted((SCRATCH / "packet-v2").glob("item-*.txt"))
    old_has_grades = OLD_PACKET.exists() and any("PRIVATE GRADE" in f.read_text() for f in OLD_PACKET.glob("item-*.txt"))
    check("v2_packet_has_no_private_grades", p.returncode == 0 and len(items_v2) == 20
          and not any(m in f.read_text() for f in items_v2 for m in MARKERS) and old_has_grades,
          {"items": len(items_v2), "old_packet_shows_grades": old_has_grades})
    report["replay"] = {"glm_scripted_score": {k: glm.get(k) for k in ("points", "confusion", "cohen_kappa")},
                        "human_scripted": {k: hum.get(k) for k in ("labelled_items", "kappa_human_vs_judge_v2")}}
    report["checks"] = CHECKS
    report["passed"] = sum(c["ok"] for c in CHECKS.values())
    report["failed"] = [k for k, c in CHECKS.items() if not c["ok"]]
    report["seconds"] = round(time.time() - t0, 1)
    write(ARTIFACTS / "m6-judge-v2.json", report)
    print(f"{report['passed']}/{len(CHECKS)} checks passed in {report['seconds']} s; failed: {report['failed']}")
    sys.exit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
