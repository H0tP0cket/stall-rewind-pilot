"""Stall judge v2: judge one run at every judge point, combine votes and onsets, form episodes.

Rules (user rulings; declared readings are marked DECLARED and listed in RULES for the record):
- Points: steps 10, 20, 30, ... up to the run's last step (Q12). Each point sees only the record after its step.
- Votes: 3 per point; confirmed when a majority (2 or 3) say "stall"; unclear, unparseable, refused and no_answer are
  not stall votes (contract; Q8).
- Onset P (Q11/Q20): each stall vote may name onset_step. A named onset is valid when 1 <= P <= point.
  DECLARED: the point's P is the median of the valid onsets of its stall votes, the lower one when there are two.
  If no stall vote gives a valid onset (missing, unreadable, after the point or before step 1), P falls back to the
  last submission at or before the point (a step whose command submits or that has submission feedback in the
  record) and the fallback and its reasons are logged. DECLARED: with no submission in the record, P falls back to
  the judge point itself (logged as such).
- Episodes (Q19; DECLARED): confirmed points at consecutive judge points (k and k+10) belong to one episode; an
  unconfirmed point (or an unjudged one) ends it. It also ends when a submission between two confirmed points shows
  progress the worker could see: a failing count lower than every earlier failing count of the run. Every episode
  counts; there is no cap per run or per task. An episode's onset P comes from its first confirmed point (the
  earliest moment the stall was confirmed); its later points are kept in the record.
- Branch source (coordinator 2026-10-21): the checkpoint at the episode's first confirmed judge point, the stall
  moment where arms 1, 2, 3, 5 and 6 start. DECLARED, pending the user: arm 4 rewinds to stall_onset_point = P - 1,
  the state just before the stall's first step ran (P = 1 -> point 0, the initial checkpoint).
- Type: one type call per episode, at its first point, with the v2 type prompt (no reference). Description only.
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import client

HERE = Path(__file__).resolve().parent
PROMPTS = {"stall": HERE / "prompts/judge-stall.v2-draft.md", "type": HERE / "prompts/judge-type.v2-draft.md"}
NEUTRAL_PROMPTS = {k: HERE / f"prompts/judge-{k}.v3-feedback-draft.md" for k in ("stall", "type")}
PROMPT_STATUS = "DRAFT v2 (2026-10-21): not yet read by the user"
VOTES = 3
STALL_VALUES = {"stall", "not_stall", "unclear"}
TYPE_VALUES = {"stuck", "over-commit", "under-commit", "unclear"}
JUDGE_VERSION = "stall-judge.v2"
RULES = {
    "points": "steps 10, 20, 30, ... up to the run's last step; each sees only the record after its step (user Q12)",
    "votes": "3 votes; confirmed = at least 2 stall votes; unclear/unparseable/refused/no_answer are not stall votes",
    "errors": "an erroring call is retried (3 attempts in all), then recorded as no_answer (user Q8)",
    "onset": "DECLARED: lower median of the valid onsets (1 <= P <= point) of the point's stall votes",
    "fallback": "no valid onset -> last submission at or before the point (user Q11/Q20); DECLARED: no submission -> "
                "the judge point itself; both logged with the reasons",
    "episodes": "DECLARED: consecutive confirmed points form one episode; an unconfirmed or unjudged point ends it, as "
                "does a submission between two confirmed points whose failing count is lower than every earlier one; "
                "every episode counts, no cap (user Q19)",
    "episode_source": "an episode's P comes from its first confirmed point; its branch source is the checkpoint at that "
                      "first confirmed judge point (the stall moment: arms 1, 2, 3, 5, 6 start there; coordinator "
                      "2026-10-21); DECLARED, pending the user: arm 4's rewind target stall_onset_point is the state "
                      "just before the stall's first step ran, the checkpoint after step P-1 (P = 1 -> the initial "
                      "checkpoint); both must be branchable or the episode is flagged and not selected",
    "type": "one type call per episode at its first point, v2 prompt without reference; description only",
}


def load_prompt(kind: str, *, neutral: bool = False) -> tuple[str, str, str]:
    """(system text, closing template, sha256 of the whole file)."""
    path = (NEUTRAL_PROMPTS if neutral else PROMPTS)[kind]
    raw = path.read_text()
    body = re.sub(r"<!--.*?-->\s*", "", raw, flags=re.S)
    system, sep, closing = body.partition("---- CLOSING ----")
    if not sep:
        raise ValueError(f"{path} has no CLOSING section")
    return system.strip(), closing.strip(), hashlib.sha256(raw.encode()).hexdigest()


def user_message(record_text: str, closing: str, point: int) -> str:
    return f"RECORD\n\n{record_text}\n\n" + closing.replace("{point}", str(point))


def combine_onsets(votes: list[dict], point: int) -> tuple[int | None, list[str]]:
    """(P from the stall votes' valid onsets, or None; the reasons the stall votes' onsets were not usable)."""
    valid, reasons = [], []
    for v in votes:
        if v.get("final") != "stall":
            continue
        o, status = v.get("onset"), v.get("onset_status")
        if status != "given":
            reasons.append(status or "missing")
        elif o > point:
            reasons.append(f"after_point({o})")
        elif o < 1:
            reasons.append(f"before_first_step({o})")
        else:
            valid.append(o)
    if not valid:
        return None, reasons
    valid.sort()
    return valid[(len(valid) - 1) // 2], reasons


def fallback_onset(submissions: list[dict], point: int) -> tuple[int, str]:
    before = [s["step"] for s in submissions if s["step"] <= point]
    if before:
        return max(before), "last_submission"
    return point, "judge_point_no_submission"


def judge_point(settings, transport, budget, calls: Path, run_key: str, rec, prompts: dict, pool: ThreadPoolExecutor,
                *, retry_no_answer: bool = False) -> dict:
    """3 votes at one point; returns the point row (votes, confirmed, onset P, P source, costs)."""
    system, closing, sha = prompts["stall"]
    user = user_message(rec.render(), closing, rec.point)
    h = hashlib.sha256((sha + system + user).encode()).hexdigest()[:16]
    d = calls / settings.slug / run_key / f"p{rec.point:04d}-{h}"
    ctx = {"run_id": rec.run_id, "point": rec.point, "call": "stall"}
    jobs = [(d / f"stall-{i}.json", dict(ctx, vote=i)) for i in range(1, VOTES + 1)]
    res = list(pool.map(lambda j: client.ask(settings, transport, budget, j[0], system, user, "verdict",
                                             STALL_VALUES, j[1], retry_no_answer=retry_no_answer), jobs))
    finals = [r.get("final") for r in res]
    confirmed = sum(f == "stall" for f in finals) * 2 > VOTES
    row = {"point": rec.point, "cut_event_sequence": rec.cut_sequence, "record_source": rec.source,
           "record_issues": rec.issues, "record_chars": len(user) + len(system), "votes": finals,
           "onsets": [r.get("onset_raw") for r in res], "evidence": [r.get("evidence") for r in res],
           "confirmed": confirmed, "receipts": [str(p.relative_to(calls.parent)) for p, _ in jobs],
           "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in res), 6),
           "refused_votes": finals.count("refused"), "no_answer_votes": finals.count("no_answer"),
           "unparseable_votes": finals.count("unparseable"),
           "skipped_spend_guard": "skipped_spend_guard" in finals,
           "submissions_in_record": rec.submissions()}
    if confirmed:
        p, reasons = combine_onsets(res, rec.point)
        if p is not None:
            row.update(onset=p, onset_source="judge", onset_unusable=reasons)
        else:
            p, why = fallback_onset(rec.submissions(), rec.point)
            row.update(onset=p, onset_source="fallback", fallback_to=why, onset_unusable=reasons)
    return row


def judge_type(settings, transport, budget, calls: Path, run_key: str, rec, prompts: dict) -> dict:
    system, closing, sha = prompts["type"]
    user = user_message(rec.render(), closing, rec.point)
    h = hashlib.sha256((sha + system + user).encode()).hexdigest()[:16]
    path = calls / settings.slug / run_key / f"p{rec.point:04d}-type-{h}" / "type-1.json"
    r = client.ask(settings, transport, budget, path, system, user, "type", TYPE_VALUES,
                   {"run_id": rec.run_id, "point": rec.point, "call": "type"})
    return {"type": r.get("final"), "type_evidence": r.get("evidence"),
            "type_receipt": str(path.relative_to(calls.parent)), "type_cost_usd": float(r.get("cost_usd") or 0.0)}


def progress_between(submissions: list[dict], a: int, b: int) -> bool:
    """A submission at a step in (a, b] whose failing count is lower than every earlier failing count of the run."""
    best = None
    for s in sorted(submissions, key=lambda s: s["step"]):
        n = s.get("failing")
        if n is None:
            continue
        if a < s["step"] <= b and best is not None and n < best:
            return True
        if s["step"] > b:
            break
        best = n if best is None else min(best, n)
    return False


def episodes(points: list[dict], submissions: list[dict]) -> list[dict]:
    """Episodes from judged point rows in point order (see the module docstring)."""
    out, cur, prev = [], None, None
    for p in points:
        judged = not p.get("skipped_spend_guard") and not p.get("error")
        if not (judged and p.get("confirmed")):
            cur = None
            prev = p
            continue
        broke = None
        if cur is not None and prev is not None and prev["point"] == cur["last_point"] \
                and progress_between(submissions, cur["last_point"], p["point"]):
            broke = "progress_submission"
            cur["ended_by"] = broke
            cur = None
        if cur is None:
            cur = {"episode": len(out) + 1, "first_point": p["point"], "last_point": p["point"],
                   "points": [p["point"]], "onset": p["onset"], "onset_source": p["onset_source"],
                   "fallback_to": p.get("fallback_to"), "onset_unusable": p.get("onset_unusable", []),
                   "first_point_votes": p["votes"], "started_after": broke or ("run_start" if not out else
                                                                              "unconfirmed_point"),
                   "ended_by": None}
            out.append(cur)
        else:
            cur["points"].append(p["point"])
            cur["last_point"] = p["point"]
        prev = p
    for i, e in enumerate(out):
        if e["ended_by"] is None:
            nxt = next((p for p in points if p["point"] > e["last_point"]), None)
            e["ended_by"] = "run_end" if nxt is None else ("unjudged_point" if nxt.get("skipped_spend_guard")
                                                           or nxt.get("error") else "unconfirmed_point")
    return out
