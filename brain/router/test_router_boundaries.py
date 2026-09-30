"""
brain/router/test_router_boundaries.py
Focused tests for the architectural boundaries added/patched in the hybrid
router stabilization pass. Everything here uses fakes — no torch,
tensorflow, or a running Ollama server, per the stabilization-pass brief.

Covers the 14 required cases:
  1. high-confidence classifier -> ready IR                (test_classifier_high_confidence_ready)
  2. low-confidence classifier -> fallback                  (test_classifier_low_confidence_falls_through)
  3. low-margin classifier -> fallback                       (test_classifier_low_margin_falls_through)
  4. semantic result -> IR                                   (test_semantic_confident_becomes_ready)
  5. LLM result -> validated IR                               (test_llm_valid_becomes_ready)
  6. malformed LLM result -> unknown/rejected                 (test_llm_malformed_is_rejected,
                                                                test_llm_domain_unknown_is_unknown)
  7. missing entity -> clarification                          (test_missing_required_entity_clarifies)
  8. unknown request -> unknown                               (test_pure_oos_is_unknown)
  9. IR -> legacy intent compatibility                        (test_to_legacy_intent_ready,
                                                                test_to_legacy_intent_clarify)
  10. context rewrite                                          (test_context_pending_reply_rewrites,
                                                                test_context_correction_rewrites)
  11. entity extraction                                        (test_extract_duration, test_extract_message,
                                                                test_extract_location)
  12. SkillResult creation                                     (test_skill_result_basic)
  13. legacy skill compatibility                               (test_legacy_skill_adapter_success,
                                                                test_legacy_skill_adapter_carries_meta,
                                                                test_legacy_skill_adapter_exception)
  14. confirmation flag propagation                            (test_requires_confirmation_propagates)

Plus the additional architectural holes fixed during review:
  - LLM validated-but-low-confidence command is NOT trusted            (test_llm_valid_but_low_confidence_unknown)
  - LLM needs_clarification with nothing actually missing -> unknown   (test_llm_clarification_with_nothing_missing)
  - pending clarification does not swallow an unrelated later turn     (test_context_pending_does_not_swallow_unrelated_turn)
  - a bare number with no correction marker does not edit a timer      (test_context_correction_requires_marker)
  - CommandSpec.requires_confirmation defaults to False                (test_command_spec_requires_confirmation_default)
"""
from __future__ import annotations

import asyncio
import re

import pytest

from brain.router import validate
from brain.router.confidence import ConfidenceThresholds
from brain.router.context import ContextResolver, ConversationContext
from brain.router.entities import extract_duration, EXTRACTORS
from brain.router.ir import CommandIR, Status, to_legacy_intent
from brain.router.schemas import Command, CommandCandidate, CommandSpec, RetrievalResult
from brain.router.understand import CommandUnderstander
from skills.base import LegacySkillAdapter, SkillResult, SkillRegistry

run = asyncio.run


# ── Shared fixtures: a tiny registry + a fake legacy classifier ─────────────

TIMER_SPEC = CommandSpec(
    domain="timer", operation="create", legacy_intent="set_timer",
    entities={"duration": {"required": True}, "message": {"required": False}},
    target_mode="raw",
)
SHUTDOWN_SPEC = CommandSpec(
    domain="system", operation="shutdown", legacy_intent="shutdown",
    entities={}, target_mode="raw", requires_confirmation=True,
)
WEATHER_SPEC = CommandSpec(
    domain="weather", operation="current", legacy_intent="get_weather",
    entities={"location": {"required": False}}, target_mode="raw",
)


class FakeRegistry:
    def __init__(self, specs):
        self.specs = specs
        self._by_key = {s.key: s for s in specs}
        self._by_legacy = {s.legacy_intent: s for s in specs}

    def get(self, domain, operation):
        return self._by_key.get(f"{domain}.{operation}")

    def domains(self):
        return sorted({s.domain for s in self.specs})

    def operations_for(self, domain):
        return sorted(s.operation for s in self.specs if s.domain == domain)


REG = FakeRegistry([TIMER_SPEC, SHUTDOWN_SPEC, WEATHER_SPEC])


class FakeLegacy:
    """Mimics the real IntentEngine.classify() return shape (see
    brain/intent_engine.py's PATCHED classify(), which now includes
    second_intent/second_confidence/margin) plus the guard attributes
    brain/router/guards.py reaches into."""

    _response_modes = {"set_timer": "skill", "shutdown": "skill", "get_weather": "skill",
                        "smalltalk": "llm", "greet": "skill", "unknown": "llm"}
    _PRESENCE_RE = re.compile(r"(?!x)x")          # never matches by default
    _ACTION_WORD_RE = re.compile(r"(?!x)x")
    _ACTION_QUESTION_RE = re.compile(r"(?!x)x")

    def __init__(self, table: dict):
        # needle -> (intent, confidence, margin, model)
        self.table = table

    def _is_dismissal(self, t):
        return False

    def _keyword_fallback(self, text):
        return "unknown", 0.0, "keyword"

    def classify(self, text):
        low = text.lower()
        for needle, (intent, conf, margin, model) in self.table.items():
            if needle in low:
                return {
                    "intent": intent, "target": "", "confidence": conf, "raw": text,
                    "model": model, "response_mode": self._response_modes.get(intent, "llm"),
                    "second_intent": None, "second_confidence": None, "margin": margin,
                }
        return {
            "intent": "unknown", "target": "", "confidence": 0.1, "raw": text,
            "model": "low_confidence_fallback", "response_mode": "llm",
            "second_intent": None, "second_confidence": None, "margin": None,
        }


def _guard_check(engine, text):
    """Minimal reimplementation of brain/router/guards.py's check(), scoped
    to what these tests exercise, so the test file doesn't need to import
    the real guards module's IntentEngine type-hint import chain."""
    t = text.lower().strip()
    if engine._is_dismissal(t):
        return "dismissal", 1.0, "guard:dismissal"
    if engine._PRESENCE_RE.search(t):
        return "smalltalk", 1.0, "guard:presence"
    if engine._ACTION_WORD_RE.search(t) and not engine._ACTION_QUESTION_RE.search(t):
        return "perform_action", 1.0, "guard:action"
    kw_intent, kw_conf, _ = engine._keyword_fallback(text)
    if kw_conf > 0 and kw_intent in ("greet", "farewell", "thanks", "help"):
        return kw_intent, kw_conf, "guard:canned_keyword"
    return None


def make_understander(table, **kwargs):
    return CommandUnderstander(FakeLegacy(table), REG, guard_fn=_guard_check, **kwargs)


# ── 1/2/3. classifier confidence + margin gate ──────────────────────────────

def test_classifier_high_confidence_ready():
    u = make_understander({"timer": ("set_timer", 0.97, 0.5, "pytorch+tensorflow")})
    ir = run(u.understand("set a timer for ten minutes"))
    assert ir.status is Status.READY and ir.key == "timer.create"
    assert ir.entities["duration"] == 600


def test_classifier_low_confidence_falls_through():
    # Below MIN_CONF, no semantic/llm wired -> ends UNKNOWN, never forced READY.
    u = make_understander({"timer": ("set_timer", 0.55, 0.5, "pytorch+tensorflow")})
    ir = run(u.understand("timer thing"))
    assert ir.status is Status.UNKNOWN
    assert ir.reason == "low_confidence"


def test_classifier_low_margin_falls_through():
    # High confidence but the runner-up is nearly as strong -> not trusted either.
    u = make_understander({"timer": ("set_timer", 0.90, 0.02, "pytorch+tensorflow")})
    ir = run(u.understand("timer thing"))
    assert ir.status is Status.UNKNOWN
    assert ir.reason == "low_confidence"


def test_margin_none_uses_stricter_bar():
    # margin=None from an ensemble-sourced result (not a keyword/guard
    # source, which would already be trusted outright via the "keyword"/
    # "guard" substring check) must be judged by MIN_CONF_NO_MARGIN
    # (0.93), not the margin-aware bar — 0.90 clears the normal bar but
    # not the stricter one.
    u = make_understander({"timer": ("set_timer", 0.90, None, "pytorch+tensorflow")})
    ir = run(u.understand("timer thing"))
    assert ir.status is Status.UNKNOWN


# ── 4. semantic escalation ───────────────────────────────────────────────────

def _retrieval(spec, sim, second_sim=0.1):
    return RetrievalResult([
        CommandCandidate(spec, sim, "seed"),
        CommandCandidate(WEATHER_SPEC if spec is not WEATHER_SPEC else TIMER_SPEC, second_sim, "seed2"),
    ])


def test_semantic_confident_becomes_ready():
    async def semantic(text):
        return _retrieval(TIMER_SPEC, 0.95, 0.10)
    u = make_understander({}, semantic=semantic)
    ir = run(u.understand("set a timer for ten minutes"))
    assert ir.status is Status.READY and ir.key == "timer.create" and ir.source == "semantic"


def test_semantic_ambiguous_does_not_execute():
    async def semantic(text):
        return _retrieval(TIMER_SPEC, 0.81, 0.79)   # confident sim but thin margin
    u = make_understander({}, semantic=semantic)
    ir = run(u.understand("do the thing"))
    assert ir.status is not Status.READY


def test_semantic_thresholds_default_when_unset():
    # semantic_thresholds=None must not crash evaluate() inside _escalate.
    async def semantic(text):
        return _retrieval(TIMER_SPEC, 0.95, 0.10)
    u = CommandUnderstander(FakeLegacy({}), REG, guard_fn=_guard_check,
                            semantic=semantic, semantic_thresholds=None)
    ir = run(u.understand("set a timer for ten minutes"))
    assert ir.status is Status.READY


# ── 5/6. LLM fallback ────────────────────────────────────────────────────────

def test_llm_valid_becomes_ready():
    async def llm_route(text, domains, ops, hints):
        return {"domain": "weather", "operation": "current", "entities": {},
                "confidence": 0.9, "needs_clarification": False}
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("weather please"))
    assert ir.status is Status.READY and ir.key == "weather.current" and ir.source == "llm"
    assert u.stats["llm"] == 1


def test_llm_malformed_is_rejected():
    async def llm_route(text, domains, ops, hints):
        return {"not": "a valid shape"}   # missing domain/operation -> structurally invalid
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("do something weird"))
    assert ir.status is Status.REJECTED and ir.reason == "malformed_llm_output"


def test_llm_domain_unknown_is_unknown():
    async def llm_route(text, domains, ops, hints):
        return {"domain": "unknown", "operation": "unknown", "entities": {},
                "confidence": 0.0, "needs_clarification": False}
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("blah blah"))
    assert ir.status is Status.UNKNOWN and ir.reason == "conversational"


def test_llm_unavailable_degrades_to_unknown():
    async def llm_route(text, domains, ops, hints):
        return None
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("blah"))
    assert ir.status is Status.UNKNOWN
    assert u.stats["llm"] == 1 and u.stats["llm_errors"] == 0


def test_llm_exception_counts_as_error_not_crash():
    async def llm_route(text, domains, ops, hints):
        raise RuntimeError("connection reset")
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("blah"))
    assert ir.status is Status.UNKNOWN
    assert u.stats["llm_errors"] == 1


def test_llm_valid_but_low_confidence_unknown():
    # PATCH: a structurally valid, registry-matching command must still be
    # gated by its own confidence — not dispatched just because it parsed.
    async def llm_route(text, domains, ops, hints):
        return {"domain": "weather", "operation": "current", "entities": {},
                "confidence": 0.1, "needs_clarification": False}
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("weather"))
    assert ir.status is Status.UNKNOWN and ir.reason == "llm_low_confidence"


def test_llm_clarification_with_nothing_missing():
    # PATCH: needs_clarification=True but every required entity is already
    # present/derivable -> UNKNOWN, not a forced READY at low confidence.
    async def llm_route(text, domains, ops, hints):
        return {"domain": "timer", "operation": "create", "entities": {"duration": "600"},
                "confidence": 0.4, "needs_clarification": True}
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("set a timer for ten minutes"))
    assert ir.status is Status.UNKNOWN
    assert ir.reason == "llm_clarification_but_nothing_missing"


def test_llm_needs_clarification_with_missing_entity():
    async def llm_route(text, domains, ops, hints):
        return {"domain": "timer", "operation": "create", "entities": {},
                "confidence": 0.3, "needs_clarification": True}
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("set a timer"))
    assert ir.status is Status.NEEDS_CLARIFICATION
    assert ir.missing_entities == ("duration",)


# ── 7/8. missing entity + pure OOS ──────────────────────────────────────────

def test_missing_required_entity_clarifies():
    u = make_understander({"timer": ("set_timer", 0.97, 0.5, "pytorch+tensorflow")})
    ir = run(u.understand("set a timer"))
    assert ir.status is Status.NEEDS_CLARIFICATION
    assert ir.missing_entities == ("duration",)
    assert ir.prompt.endswith("?")


def test_pure_oos_is_unknown():
    u = make_understander({})
    ir = run(u.understand("the sky looks orange today"))
    assert ir.status is Status.UNKNOWN


# ── 9. IR -> legacy intent compatibility ────────────────────────────────────

def test_to_legacy_intent_ready():
    ir = CommandIR(Status.READY, legacy_intent="set_timer", target="ten minutes",
                   confidence=0.9, raw_text="set a timer for ten minutes",
                   effective_text="set a timer for ten minutes")
    d = to_legacy_intent(ir)
    assert d["intent"] == "set_timer" and d["response_mode"] == "skill"
    assert d["target"] == "ten minutes"


def test_to_legacy_intent_clarify():
    # PATCH: this must be "clarify" AND brain/router/dispatch.py must have a
    # route for it (see that module's Router._routes) — verified by source
    # inspection; dispatch.py is not imported here (it pulls in kokoro/
    # sounddevice/httpx transitively, which this test file avoids per the
    # stabilization-pass brief).
    ir = CommandIR(Status.NEEDS_CLARIFICATION, legacy_intent="set_timer",
                   prompt="How long?", raw_text="set a timer", effective_text="set a timer")
    d = to_legacy_intent(ir)
    assert d["intent"] == "clarify" and d["response_mode"] == "skill"


def test_to_legacy_intent_unknown_and_rejected_go_to_llm():
    for status in (Status.UNKNOWN, Status.REJECTED):
        ir = CommandIR(status, legacy_intent="", raw_text="x", effective_text="x")
        d = to_legacy_intent(ir)
        assert d["response_mode"] == "llm"


# ── 10. context rewrite + contamination guards ──────────────────────────────

def test_context_pending_reply_rewrites():
    import time as _time
    from brain.router.context import Pending
    ctx = ConversationContext()
    resolver = ContextResolver()
    ctx.pending = Pending(text="set a timer", missing=("duration",), ts=_time.monotonic())
    resolved = resolver.resolve("ten minutes", ctx)
    assert resolved.text == "set a timer ten minutes" and resolved.source == "context"


def test_context_pending_does_not_swallow_unrelated_turn():
    # PATCH: a long unrelated sentence that happens to contain a duration
    # must NOT be rewritten into the pending timer command.
    import time as _time
    from brain.router.context import Pending
    ctx = ConversationContext()
    ctx.pending = Pending(text="set a timer", missing=("duration",), ts=_time.monotonic())
    resolver = ContextResolver()
    resolved = resolver.resolve("I have 5 minutes to kill before my meeting", ctx)
    assert resolved.text == "I have 5 minutes to kill before my meeting"
    assert resolved.source == ""


def test_context_correction_rewrites():
    import time as _time
    from brain.router.context import ActionRecord
    ctx = ConversationContext()
    ctx.last_action = ActionRecord(key="timer.create", entities={"duration": 1200},
                                   text="set a timer for 20 minutes", ts=_time.monotonic(),
                                   duration_unit=60)
    resolver = ContextResolver()
    resolved = resolver.resolve("actually make it 30", ctx)
    assert resolved.text == "set a timer for 30 minutes"
    assert resolved.source == "context_correction"


def test_context_correction_requires_marker():
    # PATCH: a bare number with no correction marker must not edit the
    # active timer — "no 5 apples please" is not "actually make it 5".
    import time as _time
    from brain.router.context import ActionRecord
    ctx = ConversationContext()
    ctx.last_action = ActionRecord(key="timer.create", entities={"duration": 1200},
                                   text="set a timer for 20 minutes", ts=_time.monotonic(),
                                   duration_unit=60)
    resolver = ContextResolver()
    resolved = resolver.resolve("I need about 5 apples please", ctx)
    assert resolved.text == "I need about 5 apples please"
    assert resolved.source == ""


def test_context_end_to_end_pending_then_correction():
    u = make_understander({"timer": ("set_timer", 0.97, 0.5, "pytorch+tensorflow")})
    ir1 = run(u.understand("set a timer"))
    assert ir1.status is Status.NEEDS_CLARIFICATION
    ir2 = run(u.understand("ten minutes"))
    assert ir2.status is Status.READY and ir2.entities["duration"] == 600
    ir3 = run(u.understand("actually make it 30"))
    assert ir3.status is Status.READY and ir3.entities["duration"] == 1800


# ── 11. entity extraction ───────────────────────────────────────────────────

def test_extract_duration():
    assert extract_duration("for twenty five minutes") == 1500
    assert extract_duration("10 minutes") == 600
    assert extract_duration("no time here") is None
    # PATCH: bare word-forms the real timer skill can't parse are no
    # longer accepted here either — see entities.py's module docstring.
    assert extract_duration("an hour") is None
    assert extract_duration("half an hour") is None


def test_extract_message():
    fn = EXTRACTORS["message"]
    msg = fn("remind me to call the dentist in twenty minutes", "")
    assert msg == "call the dentist"


def test_extract_location():
    fn = EXTRACTORS["location"]
    assert fn("what's the weather in london", "") == "london"
    assert fn("weather for today", "") is None   # strips to nothing -> auto-locate
    assert fn("no weather mention here", "") is None


# ── 12/13. SkillResult + legacy adapter ─────────────────────────────────────

def test_skill_result_basic():
    r = SkillResult(data={"temp": 20})
    assert r.ok and "temp" in r.as_system_note()
    err = SkillResult(ok=False, error="boom")
    assert "boom" in err.as_system_note()


def test_legacy_skill_adapter_success():
    async def execute(intent, text):
        return "[happy] done"
    ir = CommandIR(Status.READY, legacy_intent="get_time", raw_text="what time is it",
                   effective_text="what time is it")
    result = run(LegacySkillAdapter(execute).run(ir))
    assert result.ok and result.text == "[happy] done"


def test_legacy_skill_adapter_carries_meta():
    # PATCH: intent["action"] / intent["_no_history"] set by a legacy skill
    # (e.g. perform_action.py) must survive through to SkillResult.meta —
    # the original adapter discarded them.
    async def execute(intent, text):
        intent["action"] = "nod"
        intent["_no_history"] = True
        return "[relaxed] Mm-hm"
    ir = CommandIR(Status.READY, legacy_intent="perform_action", raw_text="nod please",
                   effective_text="nod please")
    result = run(LegacySkillAdapter(execute).run(ir))
    assert result.meta == {"action": "nod", "_no_history": True}


def test_legacy_skill_adapter_exception():
    async def execute(intent, text):
        raise ValueError("boom")
    ir = CommandIR(Status.READY, legacy_intent="x", raw_text="x", effective_text="x")
    result = run(LegacySkillAdapter(execute).run(ir))
    assert not result.ok and "boom" in result.error


def test_skill_registry_bind_legacy_routes_does_not_override_native():
    reg = SkillRegistry()

    class NativeSkill:
        async def run(self, ir, ctx=None):
            return SkillResult(text="native")

    reg.register("get_time")(NativeSkill)

    async def legacy_get_time(intent, text):
        return "legacy"

    reg.bind_legacy_routes({"get_time": legacy_get_time, "get_date": legacy_get_time})
    ir = CommandIR(Status.READY, legacy_intent="get_time", raw_text="x", effective_text="x")
    result = run(reg.resolve(ir).run(ir))
    assert result.text == "native"   # native registration wins
    assert isinstance(reg.get("get_date"), LegacySkillAdapter)   # legacy still bound


# ── 14. confirmation flag propagation ───────────────────────────────────────

def test_requires_confirmation_propagates():
    u = make_understander({"shut": ("shutdown", 0.97, 0.6, "pytorch+tensorflow")})
    ir = run(u.understand("shut it down"))
    assert ir.status is Status.READY and ir.requires_confirmation is True


def test_command_spec_requires_confirmation_default():
    spec = CommandSpec(domain="x", operation="y", legacy_intent="z", entities={}, target_mode="raw")
    assert spec.requires_confirmation is False

def test_llm_command_missing_required_entity_clarifies_not_rejected():
    async def llm_route(*a):
        return {"domain": "timer", "operation": "create", "entities": {},
                "confidence": 0.9, "needs_clarification": False}
    ir = run(make_understander({}, llm_route=llm_route).understand("set a timer"))
    assert ir.status is Status.NEEDS_CLARIFICATION and ir.missing_entities == ("duration",)

def test_llm_hallucinated_domain_is_rejected():
    async def bad(*a):
        return {"domain": "rocket", "operation": "launch", "entities": {}, "confidence": 0.9}
    assert run(make_understander({}, llm_route=bad).understand("launch the rocket")).status is Status.REJECTED

def test_escalated_target_uses_selected_spec_intent():
    seen = []
    class L(FakeLegacy):
        def _extract_target(self, text, intent):
            seen.append(intent); return "spotify"
    web = CommandSpec("app_or_web", "open", "open_target",
                      {"target": {"required": True}}, "entity:target")
    async def semantic(text):
        return RetrievalResult([CommandCandidate(web, 0.95, "s"), CommandCandidate(TIMER_SPEC, 0.1, "s2")])
    u = CommandUnderstander(L({}), FakeRegistry([web]), guard_fn=_guard_check, semantic=semantic)
    ir = run(u.understand("could you pull up spotify"))
    assert ir.status is Status.READY and ir.entities["target"] == "spotify" and "open_target" in seen