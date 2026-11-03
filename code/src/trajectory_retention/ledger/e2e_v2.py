"""End-to-end check of labeller protocol v2 (user, 2026-10-22) and the flat item pass, with no paid call.

  PYTHONPATH=src .venv/bin/python -m trajectory_retention.ledger.e2e_v2 [--out artifacts/v2/ledger-v2] [--source-root DIR]

Failure modes looked for, written before the code:
  1. the new answer check accepts a reply that is not one JSON object: agent prose with a JSON fragment (the 1017 pilot
     case the frozen check accepted as an empty answer), tool-call markup, JSON with prose after it, two objects, an
     array, a fence that is not ```json, two fences;
  2. it rejects a reply that is one JSON object, bare or in one ```json fence (with surrounding whitespace);
  3. a call gets more or fewer than two tries, or a failed call is filled in;
  4. the v2 request is not system message + one user message, the log is not inside the <recorded_log> fence exactly
     once, or instructions / answer format come before the log;
  5. the v2 tags prompt still asks about rewinds, or a flat item carries a survives-rewind value;
  6. selecting v1 changes the frozen prompts (hashes) or the frozen call path;
  7. a flat item's quote is not an exact span of the recorded context; a rebuild from receipts differs or calls out;
  8. (v2.1) a step's open/close lines are missing, out of order or nested, a part header lacks its own step number, or
     the v2 render shows different text from the v1 render once the step markup is taken off;
  9. a response cut off mid-body (``http.client.IncompleteRead``, ``RemoteDisconnected``) escapes the client's retry
     loop instead of being retried like a timeout, or a call that is cut off on every attempt leaves no receipt.
On the three real stalls of ``ledger.e2e`` it runs ``build_items`` with the scripted labeller (FAKE labels) under
protocols v2 and v2.1. Cut-off responses are simulated by replacing the HTTP call (no network, no key). If the
1017 linking-pilot receipts are on disk, every recorded pilot answer is also re-checked with both answer checks.
Writes OUT/report.json; exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..observers.t2 import glm as G
from ..observers.t2.glm import GlmAnswer
from ..observers.t2.jev import _ref
from ..observers.t2.store_ro import ReadOnlyStore
from ..store import ArtifactStore
from . import checks as C
from .e2e import STALLS, _source_root
from . import passes as P
from .labeller import ScriptedLabeller, study_labeller
from .passes import _Calls, build_items, parse_json, parse_json_object_only
from .prompts import LABEL_SYSTEM, PROTOCOLS, prompt_record
from .render import labeller_text, labeller_text_v2
from .steps import load_context

REPO = Path(__file__).resolve().parents[3]
COMMAND = "PYTHONPATH=src .venv/bin/python -m trajectory_retention.ledger.e2e_v2"
FROZEN_V1_SHA256 = {  # the hashes recorded by every v1 labeller run (e.g. the linking pilot's manifest-r2.json)
    "decisions": "26b0496392253b271623c853b3d42178e09ad3943c38ce6187eb2ce48b0fca87",
    "grouping": "a3e34748a8683878d0c79e75991549aecf9eab4ab85980ad5178ff2d5c9d2784",
    "direct_boundary": "0ed091a6e8b44d5f3f0412f05951f3fbb225fa13ec1a2dcbc634a881c115d37d",
    "tags": "f1979c0baac54be24786795dd78ee13927e406400f182d306d96cc2af93146d7",
    "approach_status": "03533bd261099792aebd4114bbb1f82ddb624cd9752abe845958c408d919c15b",
}
FROZEN_V2_SHA256 = {  # protocol v2 as used for the first linking-pilot r2b run (r2b/ledger, commit 7b332b90)
    "decisions": "d19c01b4f3a217a7a68436db5652d01875eadc0c2c1aa325931e2900a0666aca",
    "tags": "e163d7511cc5f656f178f2b7c18cb78cedf872e2e384e35b3b23cba4120578b9",
}
PILOT_RECEIPTS = "calibration/linkpilot/ledger-r2/labeller"

# Answers modelled on the 1017 pilot's failed tries (shortened), and the accepted shapes.
OBJ = '{"decisions": [{"step": 3, "quote": "I\'ll rewrite the parser to stream tokens", "refers_to": []}]}'
CASES = [
    # (name, reply, accept under v2)
    ("prose_with_json_fragment",
     "Before attempting recovery, let me inspect the local benchmark harness directory.\ncommand 1:\nls -la /opt/x\n"
     "output of command 1 (exit code 0):\ntotal 8\n...\n{\"decisions\": []}", False),
    ("tool_call_markup_dsml",
     "The follow-up says exact bytes must be kept, so let me inspect it.\n\n<｜DSML｜ calls>\n<｜DSML｜ "
     "invoke name=\"bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">find /opt -maxdepth 4"
     "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>", False),
    ("tool_call_markup_json",
     '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>', False),
    ("json_then_prose", OBJ + "\nNow let me run the tests.", False),
    ("prose_then_fenced_json", "Here are the labels:\n```json\n" + OBJ + "\n```", False),
    ("two_objects", OBJ + "\n" + OBJ, False),
    ("array", "[" + OBJ + "]", False),
    ("plain_fence_not_json", "```\n" + OBJ + "\n```", False),
    ("two_fences", "```json\n```json\n" + OBJ + "\n```\n```", False),
    ("empty", "", False),
    ("pure_object", OBJ, True),
    ("pure_object_whitespace", "\n\n  " + OBJ + "  \n", True),
    ("fenced_object", "```json\n" + OBJ + "\n```", True),
    ("fenced_object_whitespace", "  \n```json\n" + OBJ + "\n```\n", True),
]


class _Canned:
    """A client that returns one fixed reply on every try and counts the tries."""

    def __init__(self, text: str):
        self.text, self.tries = text, 0

    def complete(self, messages, *, purpose, repeat=0, max_tokens=None):
        self.tries += 1
        return GlmAnswer(self.text, None, "canned", "canned", "stop", {}, 0.0, False, f"canned:r{repeat}")


class _Recording:
    """Passes calls through to a labeller client and keeps every request's messages."""

    def __init__(self, inner):
        self.inner, self.requests = inner, []

    def bind_context(self, ctx):
        self.inner.bind_context(ctx)

    def describe(self):
        return self.inner.describe()

    def complete(self, messages, **kw):
        self.requests.append([dict(m) for m in messages])
        return self.inner.complete(messages, **kw)


def answer_check_cases() -> dict[str, Any]:
    rows, bad = [], []
    for name, text, accept in CASES:
        client = _Canned(text)
        calls = _Calls(client, protocol="v2")
        got = calls.run("decisions", "LOG", valid=lambda p: isinstance(p.get("decisions"), list))
        statuses = [r["status"] for r in calls.records]
        v1 = parse_json(text)
        row = {"case": name, "v2_accepts": got is not None, "want": accept, "tries": client.tries,
               "statuses": statuses, "frozen_v1_check_accepts": v1 is not None
               and isinstance(v1, dict) and isinstance(v1.get("decisions"), list)}
        ok = (got is not None) == accept and client.tries == (1 if accept else 2) and \
            (got == json.loads(OBJ) if accept else got is None)
        if not ok:
            bad.append(name)
        rows.append({**row, "pass": ok})
    return {"pass": not bad, "failures": bad, "cases": rows}


def pilot_answers(root: Path) -> dict[str, Any]:
    """Every answer the 1017 pilot recorded, re-checked: the v2 check must reject every answer that is not one JSON
    object. Reported: how many the frozen check accepted and the v2 check rejects."""
    path = root / PILOT_RECEIPTS
    if not path.is_dir():
        return {"pass": True, "skipped": f"no pilot receipts at {path}"}
    store = ReadOnlyStore(path)
    try:
        rows = []
        for row in store.list_records("glm_call"):
            rec = store.get_json(_ref(row["ref"]))
            if not str(rec.get("purpose", "")).startswith("ledger:") or not rec.get("response_ref"):
                continue
            body = store.get_json(_ref(rec["response_ref"]))
            text = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            key = "decisions" if rec["purpose"] == "ledger:decisions" else "items"
            v1, v2 = parse_json(text), parse_json_object_only(text)
            rows.append({"purpose": rec["purpose"], "chars": len(text),
                         "v1_accepts": isinstance(v1, dict) and isinstance(v1.get(key), list),
                         "v2_accepts": isinstance(v2, dict) and isinstance(v2.get(key), list),
                         "stripped_starts_with": text.strip()[:12]})
    finally:
        store.close()
    wrong = [r for r in rows if r["v2_accepts"] and not (r["stripped_starts_with"].startswith("{")
                                                         or r["stripped_starts_with"].startswith("```json"))]
    return {"pass": not wrong and bool(rows), "answers": len(rows),
            "v1_accepts": sum(r["v1_accepts"] for r in rows), "v2_accepts": sum(r["v2_accepts"] for r in rows),
            "v1_accepts_v2_rejects": sum(r["v1_accepts"] and not r["v2_accepts"] for r in rows),
            "v2_accepts_not_starting_with_object_or_fence": len(wrong),
            "examples_v1_only": [{k: r[k] for k in ("purpose", "chars", "stripped_starts_with")}
                                 for r in rows if r["v1_accepts"] and not r["v2_accepts"]][:5]}


_PART = re.compile(r"^\[step (\d+)\] (reasoning:|message:|command \d+:|output of command |submission feedback:|"
                   r"format error:|user message:)")


def step_markup(requests: list[list[dict[str, str]]]) -> dict[str, Any]:
    """v2.1: inside the fenced log, steps open and close in order, never nest, and every part header line carries
    the number of the step it sits in."""
    bad, parts = [], 0
    for n, msgs in enumerate(requests):
        user = msgs[-1]["content"]
        log = user[user.find("\n<recorded_log>\n"):user.find("\n</recorded_log>\n")]
        current, last, problems = None, 0, []
        for line in log.split("\n"):
            mo, mc, mp = re.fullmatch(r"<step (\d+)>", line), re.fullmatch(r"</step (\d+)>", line), _PART.match(line)
            if mo:
                k = int(mo.group(1))
                if current is not None or k <= last:
                    problems.append(f"open {k} inside {current} or after {last}")
                current, last = k, k
            elif mc:
                if current != int(mc.group(1)):
                    problems.append(f"close {mc.group(1)} while in {current}")
                current = None
            elif mp:
                parts += 1
                if current != int(mp.group(1)):
                    problems.append(f"part of step {mp.group(1)} while in {current}")
        if current is not None or last == 0:
            problems.append("unclosed or no steps")
        if problems:
            bad.append({"request": n, "problems": problems[:5]})
    return {"pass": not bad and bool(requests), "requests": len(requests), "part_headers": parts, "failures": bad[:10]}


def render_equivalence(src, run_id: str, checkpoint_id: str, max_chars: int) -> dict[str, Any]:
    """The v2 render, with its step markup taken off, is the v1 render exactly, for every chunk the passes use."""
    ctx = load_context(src, run_id, checkpoint_id)
    steps = [x.as_record() for x in ctx.steps]
    head = [f.as_record() for f in ctx.head]
    bad, n = [], 0
    for lo, hi in P._chunks(steps, head, max_chars):
        for later in (False, True):
            for elide in (False, True):
                kw = dict(target=(lo, hi), earlier=True, later=later, elide_later_outputs=elide)
                v1, v2 = labeller_text(steps, head, **kw), labeller_text_v2(steps, head, **kw)
                back = re.sub(r"^<step (\d+)>$", r"=== Step \1 ===", v2, flags=re.M)
                back = re.sub(r"^</step \d+>\n", "", back, flags=re.M)
                back = re.sub(r"^\[step \d+\] ", "", back, flags=re.M).rstrip() + "\n"  # v1 ends at its last step
                n += 1
                if back != v1:
                    bad.append({"chunk": [lo, hi], "later": later, "elide": elide})
    return {"pass": not bad and n > 0, "renders_compared": n, "failures": bad[:5]}


class _FakeResponse:
    def __init__(self, status, body=None, cut=None):
        self.status, self.body, self.cut, self.headers = status, body, cut, {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        if self.cut is not None:
            raise self.cut
        return self.body


def cut_off_responses() -> dict[str, Any]:
    """The study client with its HTTP call replaced: replies cut off mid-body must be retried like a timeout."""
    import tempfile
    body = json.dumps({"id": "gen-test", "model": "deepseek/deepseek-v4.1-flash-1017", "provider": "Parasail",
                       "choices": [{"message": {"content": '{"decisions": []}'}, "finish_reason": "stop"}],
                       "usage": {"cost": 0.0, "prompt_tokens": 10, "completion_tokens": 5}}).encode()
    cut = lambda: http.client.IncompleteRead(b'{"choices": [', 400)  # noqa: E731
    gone = lambda: http.client.RemoteDisconnected("Remote end closed connection without response")  # noqa: E731
    scenarios = {
        "incomplete_read_then_ok": ([cut], True, ["IncompleteRead"]),
        "remote_disconnected_then_ok": ([gone], True, ["RemoteDisconnected"]),
        "cut_off_every_attempt": ([cut] * G.MAX_ATTEMPTS, False, ["IncompleteRead"] * G.MAX_ATTEMPTS),
    }
    real_urlopen, real_sleep = G.urllib.request.urlopen, G.time.sleep
    rows, bad = [], []
    try:
        G.time.sleep = lambda s: None
        for name, (faults, ok_expected, errors) in scenarios.items():
            queue = list(faults)

            def fake(req, timeout=None):
                if queue:
                    f = queue.pop(0)()
                    if isinstance(f, http.client.IncompleteRead):
                        return _FakeResponse(200, cut=f)
                    raise f
                return _FakeResponse(200, body)

            G.urllib.request.urlopen = fake
            with tempfile.TemporaryDirectory() as d:
                store = ArtifactStore(Path(d) / "labeller")
                client = study_labeller(store, budget_usd=1.0, replay=True)
                client.key = "test-only-not-a-key"  # never sent anywhere: the HTTP call is replaced
                calls = _Calls(client, protocol="v2.1")
                got = calls.run("decisions", "LOG", valid=lambda p: isinstance(p.get("decisions"), list))
                kind = "glm_call" if ok_expected else "glm_call_failed"
                recs = [store.get_json(_ref(r["ref"])) for r in store.list_records(kind)]
                attempts = recs[0]["attempts"] if recs else []
                seen = [a.get("transport_error") for a in attempts if a.get("http_status") == 0]
                ok = (got == {"decisions": []}) == ok_expected and len(recs) == 1 and seen == errors and                     (attempts[-1]["http_status"] == 200 if ok_expected else len(attempts) == G.MAX_ATTEMPTS) and                     calls.records[-1]["status"] == ("ok" if ok_expected else "error")
                rows.append({"scenario": name, "pass": ok, "answered": got is not None,
                             "attempts": [(a["http_status"], a.get("transport_error")) for a in attempts],
                             "call_status": calls.records[-1]["status"], "receipt_kind": kind})
                if not ok:
                    bad.append(name)
    finally:
        G.urllib.request.urlopen, G.time.sleep = real_urlopen, real_sleep
    return {"pass": not bad, "failures": bad, "scenarios": rows}


def request_shape(requests: list[list[dict[str, str]]]) -> dict[str, Any]:
    bad = []
    for n, msgs in enumerate(requests):
        roles = [m["role"] for m in msgs]
        user = msgs[-1]["content"]
        open_, close = user.find("\n<recorded_log>\n"), user.find("\n</recorded_log>\n")
        answer = user.find("Your whole reply must be exactly one JSON object")
        problems = []
        if roles != ["system", "user"] or msgs[0]["content"] != LABEL_SYSTEM:
            problems.append(f"roles {roles}")
        if user.count("\n<recorded_log>\n") != 1 or user.count("\n</recorded_log>\n") != 1 or not 0 < open_ < close \
                or "recorded_log>" in user[open_ + 15:close]:  # fence lines once each; the log never holds a marker
            problems.append("fence")
        if not close < answer:
            problems.append("answer format before the log ends")
        if "Find every DECISION" in user[:close] or "In the target steps, find" in user[:close]:
            problems.append("instructions before the log")
        if "TASK\n" not in user[open_:close]:
            problems.append("log missing from fence")
        prompt_text = (user[:open_] + user[close:]).lower()  # the prompt without the log
        if "rewind" in prompt_text or "survives" in prompt_text:
            problems.append("prompt asks about rewind")
        if problems:
            bad.append({"request": n, "problems": problems})
    return {"pass": not bad and bool(requests), "requests": len(requests), "failures": bad[:10]}


def flat_stall(src: ReadOnlyStore, name: str, run_id: str, cp: str, max_chars: int, out: Path,
               protocol: str) -> dict[str, Any]:
    sdir = out / "stalls" / protocol / name
    checkpoint_id = f"{run_id}/{cp}"
    rec = _Recording(ScriptedLabeller(ArtifactStore(sdir / "labeller")))
    doc = build_items(src, run_id, checkpoint_id, rec, protocol=protocol, out_dir=sdir, max_chars=max_chars)
    first = {p.name: p.read_bytes() for p in sdir.glob("*.json")}
    rebuild = ScriptedLabeller(ArtifactStore(sdir / "labeller"))
    build_items(src, run_id, checkpoint_id, rebuild, protocol=protocol, out_dir=sdir, max_chars=max_chars)
    again = {p.name: p.read_bytes() for p in sdir.glob("*.json")}

    def no_cache_flag(b):
        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items() if k != "cached"}
            return [strip(v) for v in x] if isinstance(x, list) else x
        return strip(json.loads(b))

    differ = sorted(n for n in first if no_cache_flag(first[n]) != no_cache_flag(again.get(n, b"null")))
    passes = doc["labeller"]["passes"]
    checks = {
        "status_ok": {"pass": doc["status"] == "ok", "status": doc["status"]},
        "request_shape": request_shape(rec.requests),
        "two_tries_at_most": {"pass": all(r["repeat"] in (0, 1) for r in passes)
                              and all(r["protocol"] == PROTOCOLS[protocol]["id"]
                                      and r["answer_check"] == "whole_json_object" for r in passes)},
        "no_survives_rewind": {"pass": all(i.get("survives_rewind") is None for i in doc["items"])
                               and doc["rewind_point"] == doc["stall_point"]},
        "no_approach_layer": {"pass": doc["approaches"] == [] and doc["passes_run"] == ["decisions", "tags"]
                              and {r["prompt"] for r in passes} == {"decisions", "tags"}},
        "quotes_resolve": C.check_quotes(src, doc),
        "rebuild_from_receipts": {"pass": not differ and rebuild.calls == 0, "files_differing": differ,
                                  "new_calls": rebuild.calls, "cache_hits": rebuild.cache_hits},
    }
    if PROTOCOLS[protocol]["render"] == "v2":
        checks["step_markup"] = step_markup(rec.requests)
        checks["render_same_text_as_v1"] = render_equivalence(src, run_id, checkpoint_id, max_chars)
    (sdir / "checks.json").write_text(json.dumps(checks, indent=1, default=str) + "\n")
    return {"name": name, "protocol": protocol, "steps": doc["stall_point"], "status": doc["status"],
            "calls": len(passes),
            "items": {t: sum(1 for i in doc["items"] if i["type"] == t) for t in ("decision", "fact", "claim", "idea")},
            "dropped": len(doc["dropped"]), "checks": {k: v["pass"] for k, v in checks.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Labeller protocol v2 end-to-end check (fake labels, no paid calls)")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "v2" / "ledger-v2"))
    ap.add_argument("--source-root", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    root = Path(a.source_root).resolve() if a.source_root else _source_root()
    frozen = {n: r["sha256"] for n, r in prompt_record("v1").items()}
    v2, v21 = prompt_record("v2"), prompt_record("v2.1")
    prompts = {"pass": frozen == FROZEN_V1_SHA256 and PROTOCOLS["v1"]["answer_check"] == "json_anywhere"
               and {n: r["sha256"] for n, r in v2.items()} == FROZEN_V2_SHA256
               and not set(r["sha256"] for r in v21.values()) & (set(frozen.values()) | set(FROZEN_V2_SHA256.values()))
               and all(r["version"].endswith("/v2-draft") for r in v2.values())
               and all(r["version"].endswith("/v2.1-draft") for r in v21.values())
               and PROTOCOLS["v1"]["render"] == PROTOCOLS["v2"]["render"] == "v1" and PROTOCOLS["v2.1"]["render"] == "v2",
               "v1": frozen, "v2": v2, "v2.1": v21}
    cases = answer_check_cases()
    cutoff = cut_off_responses()
    pilot = pilot_answers(root)
    stalls, opened = [], {}
    for name, rel, run_id, cp, max_chars in STALLS:
        src = opened.setdefault(rel, ReadOnlyStore(root / rel))
        for protocol in ("v2", "v2.1"):
            stalls.append(flat_stall(src, name, run_id, cp, max_chars, out, protocol))
            print(f"{name} {protocol}: steps={stalls[-1]['steps']} items={stalls[-1]['items']} "
                  f"checks={stalls[-1]['checks']}", flush=True)
    for src in opened.values():
        src.close()
    all_pass = prompts["pass"] and cases["pass"] and cutoff["pass"] and pilot["pass"] and \
        all(all(s["checks"].values()) for s in stalls)
    report = {"what": "labeller protocol v2 and flat item pass, end to end; scripted labeller (FAKE labels) on real "
                      "stalls, canned answers for the answer check, pilot receipts re-checked",
              "fake_labels": True, "paid_calls": 0, "command": COMMAND, "source_root": str(root), "all_pass": all_pass,
              "prompts": prompts, "answer_check_cases": cases, "cut_off_responses": cutoff, "pilot_answers": pilot,
              "stalls": stalls}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"all_pass": all_pass, "prompts": prompts["pass"], "answer_check_cases": cases["pass"],
                      "cut_off_responses": cutoff["pass"],
                      "pilot_answers": {k: pilot.get(k) for k in ("pass", "answers", "v1_accepts", "v2_accepts",
                                                                   "v1_accepts_v2_rejects", "skipped")},
                      "report": str(out / "report.json")}, indent=1))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
