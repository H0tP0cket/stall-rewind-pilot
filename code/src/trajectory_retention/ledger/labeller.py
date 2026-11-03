"""Labeller clients for the Ledger passes. All expose ``complete(messages, *, purpose, repeat=0, max_tokens=None)``
returning a ``GlmAnswer``, and write a content-addressed receipt (request sha256 + repeat index) for every call.

* ``study_labeller``: ``GlmClient`` pinned to DeepSeek V4.1 Flash on the study's provider pin, with the study's worker
  settings (temperature, reasoning effort, max tokens) read from ``configs/v2/settings.json`` (written by the
  coordinator after the provider probe). Until that file exists the pin and settings are parameters. Receipts:
  ``glm_call`` records in the given ArtifactStore. A response from another provider is refused (``ProviderMismatch``).
* ``study_labeller(..., replay=True)``: the same client with no credential, so it can only serve calls whose receipts
  already exist (a missing receipt raises instead of spending).
* ``ScriptedLabeller``: deterministic fake labels for end-to-end checks with no paid call. Its receipts are a separate
  record kind (``scripted_call``) so a real client can never reuse them, and every output it feeds is marked
  ``fake_labels``. It deliberately emits a few malformed answers (a paraphrased quote, a backwards reference, evidence
  from an earlier step) so the drop-and-log paths run on real data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import canonical_json, utc_now
from ..observers.t2.glm import GlmAnswer, GlmClient
from ..observers.t2.jev import _jsonable_ref, _ref
from .text import prose_spans21

LABELLER_MODEL = "deepseek/deepseek-v4.1-flash"
SETTINGS_PATH = Path(__file__).resolve().parents[3] / "configs" / "v2" / "settings.json"
# Reservation prices for the budget guard only (charged cost is the provider-reported usage.cost); deliberately high.
RESERVE_USD_PER_TOKEN = (0.5 / 1_000_000, 2.0 / 1_000_000)


def load_study_settings(path: str | Path | None = None) -> dict[str, Any] | None:
    """``{"model", "settings"}`` from configs/v2/settings.json, or None while the coordinator has not written it.

    Accepted shapes: ``{"model": ..., "settings": {..., "provider": PIN}}`` or the settings object itself."""
    p = Path(path) if path else SETTINGS_PATH
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    settings = data.get("settings", data)
    return {"model": data.get("model", LABELLER_MODEL), "settings": settings}


class StudyLabeller(GlmClient):
    fake = False

    def __init__(self, store, *, pin: Mapping[str, Any], settings: Mapping[str, Any], budget_usd: float,
                 model: str = LABELLER_MODEL, max_tokens: int | None = None, timeout: float = 900.0,
                 replay: bool = False):
        only = list(pin.get("only") or [])
        if len(only) != 1:
            raise ValueError("the labeller pin must name exactly one provider in 'only'")
        super().__init__(store, budget_usd=budget_usd, max_tokens=int(max_tokens or settings.get("max_tokens") or 32768),
                         timeout=timeout, reasoning=settings.get("reasoning") or {"effort": "max"}, model=model,
                         provider_slug=only[0], expected_provider=only[0], usd_per_token=RESERVE_USD_PER_TOKEN)
        self.pin = {**dict(pin), "allow_fallbacks": False}
        self.extra = {k: settings[k] for k in ("temperature",) if k in settings}
        self.settings = dict(settings)
        if replay:
            self.key = None

    def request_body(self, messages, *, max_tokens=None):
        body = super().request_body(messages, max_tokens=max_tokens)
        body.update(self.extra)
        return body

    def describe(self) -> dict[str, Any]:
        return {"model": self.model, "provider": dict(self.pin), "settings": {**self.settings, "provider": None},
                "fake_labels": False, "replay_only": self.key is None}


def study_labeller(store, *, budget_usd: float, settings_path: str | Path | None = None,
                   pin: Mapping[str, Any] | None = None, settings: Mapping[str, Any] | None = None,
                   replay: bool = False, max_tokens: int | None = None) -> StudyLabeller:
    """The study labeller: pin and settings from configs/v2/settings.json, else from the parameters (never a default
    provider)."""
    loaded = load_study_settings(settings_path)
    model = LABELLER_MODEL
    if loaded is not None:
        model = loaded["model"]
        settings = {**loaded["settings"], **(settings or {})}
        pin = pin or loaded["settings"].get("provider")
    if pin is None or settings is None:
        raise ValueError("no study provider pin: configs/v2/settings.json is missing, so pass pin= and settings=")
    return StudyLabeller(store, pin=pin, settings=settings, budget_usd=budget_usd, model=model, replay=replay,
                         max_tokens=max_tokens)


class CodexPilotLabeller:
    """Terra subscription labels, kept separate from DeepSeek and scripted receipts."""
    fake = False

    def __init__(self, store, sessions: Path):
        from ..observers.codex_subscription import CodexSubscription, MODEL, PROVIDER, VERSION
        self.store, self.model, self.provider, self.version = store, MODEL, PROVIDER, VERSION
        self.transport = CodexSubscription(sessions)
        self.calls = self.cache_hits = 0
        self.spent = 0.0  # external API dollars, not subscription cost

    def describe(self):
        return {"model": self.model, "provider": {"only": ["codex-subscription"], "allow_fallbacks": False},
                "settings": {"transport": self.version, "reasoning": {"effort": "max"},
                             "billing": "Codex allowance; external API $0; subscription dollar cost unknown",
                             "output_limit": "Codex model default"},
                "fake_labels": False, "replay_only": False}

    def complete(self, messages, *, purpose: str, repeat: int = 0, max_tokens=None):
        body = {"model": self.model, "messages": [dict(m) for m in messages],
                "reasoning": {"effort": "max"}, "max_tokens": 128000}
        raw = canonical_json({"transport": self.version, "body": body, "purpose": purpose})
        digest = hashlib.sha256(raw).hexdigest()
        rid = f"{digest}:r{repeat}"
        try:
            saved = self.store.get_record("codex_subscription_call", rid)
        except KeyError:
            saved = None
        if saved is not None:
            self.cache_hits += 1
            text = self.store.get_bytes(_ref(saved["response_ref"])).decode()
            return GlmAnswer(text, None, self.provider, self.model, "stop", saved["usage"], 0.0, True, rid)
        resp = self.transport.post(body, {"purpose": purpose, "repeat": repeat})
        text = resp["choices"][0]["message"]["content"]
        saved = {"purpose": purpose, "repeat": repeat, "request_sha256": digest,
                 "request_ref": _jsonable_ref(self.store.put_bytes(raw, media_type="application/json")),
                 "response_ref": _jsonable_ref(self.store.put_bytes(text.encode(), media_type="text/plain")),
                 "status": "ok", "model": self.model, "served_provider": self.provider,
                 "provider_matches_pin": True, "fake_labels": False, "usage": resp["usage"],
                 "cost_usd": 0.0, "cost_basis": "external API only; subscription cost unknown",
                 "codex": resp["codex"], "recorded_at": utc_now()}
        self.store.put_record("codex_subscription_call", rid, saved)
        self.calls += 1
        return GlmAnswer(text, None, self.provider, self.model, "stop", resp["usage"], 0.0, False, rid)


# ---- scripted (fake) labeller ----

FAKE_MODEL = "scripted/ledger-fake-labels-v1"
_DECIDE = re.compile(r"\b(I'll|I will|I'm going to|Let's|The fix|Now I'll|I need to (?:fix|change|rewrite|replace|add|"
                     r"implement|update|write|make)|Let me (?:fix|rewrite|change|replace|add|implement|switch|update|"
                     r"create|write|patch|remove|use|build|publish|apply))")
_NEW = re.compile(r"\b(instead|different|another|alternative|switch|rewrite|new approach|try)\b", re.I)
_BACK = re.compile(r"\b(that|this|instead|again|didn't|still|already)\b", re.I)
_FACT_LINE = re.compile(r"(Error|error:|FAIL|passed|failed|not found|No such|expected|Traceback|warning|mismatch)")
_CLAIM = re.compile(r"\b(works|passes|pass now|fixed|is correct|green|succeeded|matches|now correct|complete|"
                    r"verified|clean)\b", re.I)
_IDEA = re.compile(r"\b(maybe|perhaps|could|might|alternatively|option)\b", re.I)


def _sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in prose_spans21(text)[0]]


def _first_line(step, rx) -> str | None:
    for f in step.outputs():
        for line in f.text.splitlines():
            line = line.strip()
            if 12 <= len(line) <= 200 and rx.search(line):
                return line
    return None


class ScriptedLabeller:
    """Deterministic fake labels from simple text cues; for end-to-end checks only (see module docstring)."""

    fake = True
    model = FAKE_MODEL

    def __init__(self, store):
        self.store = store
        self.context = None
        self.calls = 0
        self.cache_hits = 0
        self.spent = 0.0

    def bind_context(self, context) -> None:
        self.context = context

    def describe(self) -> dict[str, Any]:
        return {"model": self.model, "provider": {"only": ["scripted"]}, "settings": None, "fake_labels": True,
                "replay_only": False}

    def complete(self, messages: Sequence[Mapping[str, str]], *, purpose: str, repeat: int = 0,
                 max_tokens: int | None = None) -> GlmAnswer:
        body = {"model": self.model, "messages": [dict(m) for m in messages], "purpose": purpose,
                "context": self.context.fingerprint() if self.context else None}
        raw = canonical_json(body)
        digest = hashlib.sha256(raw).hexdigest()
        record_id = f"{digest}:r{repeat}"
        try:
            saved = self.store.get_record("scripted_call", record_id)
        except KeyError:
            saved = None
        if saved is not None:
            self.cache_hits += 1
            text = self.store.get_bytes(_ref(saved["response_ref"])).decode("utf-8")
            return GlmAnswer(text, None, "scripted", self.model, "stop", {}, 0.0, True, record_id)
        text = json.dumps(self._respond(purpose.split(":", 1)[-1], messages[-1]["content"], repeat))
        record = {"purpose": purpose, "repeat": repeat, "request_sha256": digest,
                  "request_ref": _jsonable_ref(self.store.put_bytes(raw, media_type="application/json")),
                  "response_ref": _jsonable_ref(self.store.put_bytes(text.encode(), media_type="application/json")),
                  "status": "ok", "model": self.model, "served_provider": "scripted", "provider_matches_pin": True,
                  "cost_usd": 0.0, "cost_basis": "scripted (no model call)", "fake_labels": True,
                  "recorded_at": utc_now()}
        self.store.put_record("scripted_call", record_id, record)
        self.calls += 1
        return GlmAnswer(text, None, "scripted", self.model, "stop", {}, 0.0, False, record_id)

    # each responder reads only the steps the prompt names (prefix-only by construction of the prompt)
    def _respond(self, kind: str, prompt: str, repeat: int) -> dict[str, Any]:
        ctx = self.context
        target = [int(k) for k in re.findall(r"^(?:=== Step |<step )(\d+)(?: ===|>)$", prompt.split("LATER STEPS")[0], re.M)]
        if "TARGET STEPS" in prompt:
            target = [int(k) for k in re.findall(r"^(?:=== Step |<step )(\d+)(?: ===|>)$",
                                                 prompt.split("TARGET STEPS")[1].split("LATER STEPS")[0], re.M)]
        if kind == "decisions":
            return {"decisions": self._decisions(target)}
        if kind == "grouping":
            wins = re.findall(r"^(W\d+) steps \d+", prompt, re.M)
            if len(wins) >= 3:
                return {"approaches": [wins[i:i + 2] for i in range(0, len(wins), 2)]}
            return {"approaches": [[w] for w in wins]}
        if kind == "direct_boundary":
            ds = [d for d in self._decisions(range(1, len(ctx.steps) + 1), bad=False)]
            pick = [d for d in ds if _NEW.search(d["quote"])] or ds
            return {"first_step": pick[-1]["step"], "quote": pick[-1]["quote"]} if pick else \
                {"first_step": 1, "quote": None}
        if kind == "tags":
            m = re.search(r"at point (\d+) \(before step", prompt)
            return {"items": self._tags(target, int(m.group(1)) if m else 0)}
        if kind == "approach_status":
            rows = [(a, int(f), int(l)) for a, f, l in re.findall(r"^(A\d+): steps (\d+)–(\d+)", prompt, re.M)]
            out = []
            for n, (aid, first, last) in enumerate(rows):
                hit = next(((s.k, _first_line(s, _FACT_LINE)) for s in ctx.steps[first - 1:]
                            if _first_line(s, _FACT_LINE)), None)
                status = ("failed" if hit else "open") if n == len(rows) - 1 else ("failed" if hit else "abandoned")
                out.append({"id": aid, "status": status, "failure_step": hit[0] if hit else None,
                            "failure_quote": hit[1] if hit else None})
            return {"approaches": out}
        raise ValueError(f"scripted labeller has no answer for {kind!r}")

    def _decisions(self, target, *, bad: bool = True) -> list[dict[str, Any]]:
        ctx = self.context
        out, earlier = [], []
        for st in ctx.steps:
            acting = st.workspace_change or st.submission or any(k is None for k in st.detail.get("command_kinds", []))
            sentence = None
            if acting:
                for f in st.fields:
                    if f.kind in ("reasoning", "message"):
                        sentence = next((s for s in _sentences(f.text) if _DECIDE.search(s)), None)
                        if sentence:
                            break
            if sentence and st.k in target:
                refs = [earlier[-1]] if earlier and _BACK.search(sentence) else []
                out.append({"step": st.k, "quote": sentence, "refers_to": refs})
            if sentence:
                earlier.append(st.k)
        if bad and out:
            out.append({"step": out[0]["step"], "quote": out[0]["quote"].upper() + " (paraphrased)", "refers_to": []})
            out.append({"step": out[0]["step"], "quote": out[0]["quote"], "refers_to": [out[0]["step"] + 1]})
        return out

    def _tags(self, target, rewind_point: int) -> list[dict[str, Any]]:
        ctx = self.context
        out = []
        for k in target:
            st = ctx.step(k)
            later = ctx.steps[k:]
            fact = _first_line(st, _FACT_LINE)
            if fact:
                surv = None if k <= rewind_point else bool(st.flags.get("noise") or not st.workspace_change)
                out.append({"step": k, "type": "fact", "quote": fact, "status": None, "evidence_step": None,
                            "evidence_quote": None, "survives_rewind": surv})
            prose = [s for f in st.fields if f.kind in ("reasoning", "message") for s in _sentences(f.text)]
            claim = next((s for s in prose if _CLAIM.search(s)), None)
            if claim:
                bad_ev = next((s for s in later if s.flags.get("outcome") in ("tests_fail", "error")
                               and _first_line(s, _FACT_LINE)), None)
                good_ev = next((s for s in later if s.flags.get("outcome") == "tests_pass"
                                and _first_line(s, re.compile("passed"))), None)
                if bad_ev:
                    item = {"status": "contradicted_later", "evidence_step": bad_ev.k,
                            "evidence_quote": _first_line(bad_ev, _FACT_LINE)}
                elif good_ev:
                    item = {"status": "confirmed_later", "evidence_step": good_ev.k,
                            "evidence_quote": _first_line(good_ev, re.compile("passed"))}
                else:
                    item = {"status": "unchecked", "evidence_step": None, "evidence_quote": None}
                out.append({"step": k, "type": "claim", "quote": claim, "survives_rewind": None, **item})
            idea = next((s for s in prose if _IDEA.search(s) and s != claim), None)
            if idea:
                words = {w.lower() for w in re.findall(r"[A-Za-z_]{6,}", idea)}
                tried = next((s for s in later if any(w in c["command"].lower() for c in s.commands for w in words)), None)
                out.append({"step": k, "type": "idea", "quote": idea, "status": "tried_later" if tried else "never_tried",
                            "evidence_step": tried.k if tried else None, "evidence_quote": None,
                            "survives_rewind": None})
        claims = [x for x in out if x["type"] == "claim"]
        if claims:
            c = claims[0]
            out.append({**c, "quote": c["quote"] + " (sic)"})
            out.append({**c, "type": "claim", "status": "confirmed_later", "evidence_step": max(1, c["step"] - 1),
                        "evidence_quote": c["quote"], "quote": c["quote"]})
        return out

