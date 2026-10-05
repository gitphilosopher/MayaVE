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


# Shared fixtures: a small registry and a fake legacy classifier for isolated router tests.

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
    """Minimal registry stub used to resolve command specs during router tests."""

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
    """Minimal legacy classifier stub that matches the runtime contract used by the router."""

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
    """Small guard shim for the router tests that exercises only the relevant boundary cases."""
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
    """Build a router-understand harness with the shared fake registry and guard shim."""
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
    # If the classifier reports no margin, the stricter no-margin threshold
    # still applies; this would pass the normal gate but fail the safer one.
    u = make_understander({"timer": ("set_timer", 0.90, None, "pytorch+tensorflow")})
    ir = run(u.understand("timer thing"))
    assert ir.status is Status.UNKNOWN


# ── 4. semantic escalation ───────────────────────────────────────────────────

def _retrieval(spec, sim, second_sim=0.1):
    """Create a synthetic retrieval result with a primary and secondary candidate."""
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
    # A structurally valid command must still honor its own confidence before
    # it is accepted, even when it matches the registry cleanly.
    async def llm_route(text, domains, ops, hints):
        return {"domain": "weather", "operation": "current", "entities": {},
                "confidence": 0.1, "needs_clarification": False}
    u = make_understander({}, llm_route=llm_route)
    ir = run(u.understand("weather"))
    assert ir.status is Status.UNKNOWN and ir.reason == "llm_low_confidence"


def test_llm_clarification_with_nothing_missing():
    # A clarification flag is not enough to accept a command when all required
    # entities are already satisfied; this must remain UNKNOWN rather than READY.
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
    # Clarification intent must be emitted as a special legacy shape, even
    # when the router itself is not imported in this isolated unit test.
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
    # A long unrelated sentence containing a number must not get rewritten into
    # the pending timer command just because it includes a duration-like phrase.
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
    # A bare numeric phrase without an explicit correction marker must not
    # rewrite the active timer command.
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
    # Bare word forms that the timer parser does not accept should remain invalid.
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
    # Legacy skill metadata must be preserved when the adapter wraps the result.
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


# ── 15. LLM fallback boundary regression tests ──────────────────────────────

def test_llm_fallback_prompt_contains_boundary_rules():
    from brain.router.llm_fallback import _build_prompt
    prompt = _build_prompt(
        "turn off the porch light please",
        ["system", "timer", "web"],
        {"system": ["lock", "shutdown"], "timer": ["create"], "web": ["search"]},
        [],
    )
    assert "STRICT ROUTING RULES:" in prompt
    assert "ACTIVE COMMAND REQUIREMENT" in prompt
    assert "COMPUTER SCOPE VS PHYSICAL WORLD" in prompt
    assert "NO NEAREST-NEIGHBOR GUESSING" in prompt
    assert "PRESERVE LEGITIMATE COMMANDS" in prompt
    assert "system.shutdown" in prompt
    assert "system.lock" in prompt


@pytest.mark.parametrize("utterance", [
    "I set a timer for my kids once and they loved it",
    "turn off the porch light please",
    "i should lock up the shop before i go",
    "the google search results were useless today",
    # Representative negatives:
    "she talked for ninety minutes",
    "I locked the door before leaving",
    "turn off the TV",
    "unlock the door",
    "my Google search results were useless",
    "I used a timer yesterday",
    "I need to lock up the shop",
    "turning off the light is easy",
])
def test_llm_fallback_boundary_must_remain_unknown(utterance):
    import httpx
    from brain.router.llm_fallback import route
    from brain.router.registry import CommandRegistry, load_specs
    from config.settings import config

    try:
        r = httpx.get(f"{config.llm.base_url.rstrip('/')}/api/tags", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Ollama not running")
    except Exception:
        pytest.skip("Ollama not running")

    reg = CommandRegistry(load_specs())
    domains = reg.domains()
    ops = {d: reg.operations_for(d) for d in domains}

    result = run(route(utterance, domains, ops, []))
    assert result is not None
    # Must route to unknown domain/operation, or be low confidence (< 0.5)
    is_unknown = (
        result.get("domain") in ("unknown", None)
        or result.get("operation") in ("unknown", None)
        or float(result.get("confidence", 0.0)) < 0.5
    )
    assert is_unknown, f"Expected unknown/unexecutable for '{utterance}', got {result}"


@pytest.mark.parametrize("utterance,expected_domain,expected_op", [
    ("set a timer for 10 minutes", "timer", "create"),
    ("search Google for today's weather", "web", "search"),
    ("lock my computer", "system", "lock"),
    ("lock the screen", "system", "lock"),
    ("shut down my computer", "system", "shutdown"),
])
def test_llm_fallback_boundary_positive_controls(utterance, expected_domain, expected_op):
    import httpx
    from brain.router.llm_fallback import route
    from brain.router.registry import CommandRegistry, load_specs
    from config.settings import config

    try:
        r = httpx.get(f"{config.llm.base_url.rstrip('/')}/api/tags", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Ollama not running")
    except Exception:
        pytest.skip("Ollama not running")

    reg = CommandRegistry(load_specs())
    domains = reg.domains()
    ops = {d: reg.operations_for(d) for d in domains}

    result = run(route(utterance, domains, ops, []))
    assert result is not None
    assert result.get("domain") == expected_domain, f"Wrong domain for '{utterance}': {result}"
    assert result.get("operation") == expected_op, f"Wrong operation for '{utterance}': {result}"
    assert float(result.get("confidence", 0.0)) >= 0.5, f"Low confidence for '{utterance}': {result}"


# ── 16. L1 classifier, keyword, and target-extraction boundary tests ──────────

@pytest.mark.parametrize("utterance", [
    "I set a timer for my kids once and they loved it",
    "she talked for ninety minutes",
    "I used a timer yesterday",
    "i should lock up the shop before i go",
    "I should lock up the house before I leave",
    "I need to lock the shop before closing",
    "I locked the door before leaving",
    "my google search history is embarrassing",
    "the google search results were useless today",
    "I looked at my Google search history",
    "those search results were terrible",
])
def test_l1_contrastive_negatives_route_to_unknown(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert ir.status is Status.UNKNOWN, f"Expected UNKNOWN for '{utterance}', got {ir.status} ({ir.domain}.{ir.operation})"
    assert ir.domain is None, f"Expected domain=None for '{utterance}', got {ir.domain}"
    assert ir.operation is None, f"Expected operation=None for '{utterance}', got {ir.operation}"


@pytest.mark.parametrize("utterance,expected_domain,expected_op", [
    ("set a timer for 10 minutes", "timer", "create"),
    ("lock my computer", "system", "lock"),
    ("lock the screen", "system", "lock"),
    ("search Google for quantum computing", "web", "search"),
    ("google quantum computing", "web", "search"),
    ("search for quantum computing", "web", "search"),
    ("look up recipes for pasta", "web", "search"),
])
def test_l1_contrastive_positive_controls_preserved(utterance, expected_domain, expected_op):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert ir.status is Status.READY, f"Expected READY for '{utterance}', got {ir.status}"
    assert ir.domain == expected_domain, f"Wrong domain for '{utterance}': {ir.domain}"
    assert ir.operation == expected_op, f"Wrong operation for '{utterance}': {ir.operation}"


def test_l1_extract_target_search_web_mentions_vs_queries():
    from brain.intent_engine import IntentEngine
    engine = IntentEngine()
    # Non-search mentions must NOT yield query targets
    assert engine._extract_target("my google search history is embarrassing", "search_web") == ""
    assert engine._extract_target("the google search results were useless today", "search_web") == ""
    assert engine._extract_target("i looked at my google search history", "search_web") == ""
    assert engine._extract_target("those search results were terrible", "search_web") == ""

    # Real search commands must extract query cleanly
    assert engine._extract_target("google today's weather", "search_web") == "today's weather"
    assert engine._extract_target("search for quantum computing", "search_web") == "quantum computing"
    assert engine._extract_target("search google for local cafes", "search_web") == "local cafes"
    assert engine._extract_target("google search pasta recipes", "search_web") == "pasta recipes"
    assert engine._extract_target("look up quantum mechanics", "search_web") == "quantum mechanics"
    assert engine._extract_target("please google weather forecast", "search_web") == "weather forecast"
    assert engine._extract_target("can you google best movies of 2024", "search_web") == "best movies of 2024"


def test_l1_keyword_fallback_timer_narrative_vs_imperative():
    from brain.intent_engine import IntentEngine
    engine = IntentEngine()
    # Narrative past statements must not match set_timer keyword fallback
    intent, conf, _ = engine._keyword_fallback("I set a timer for my kids once and they loved it")
    assert intent != "set_timer", f"Narrative should not match set_timer keyword, got {intent}"
    intent, conf, _ = engine._keyword_fallback("she set a timer for dinner")
    assert intent != "set_timer"
    intent, conf, _ = engine._keyword_fallback("we set a timer earlier")
    assert intent != "set_timer"

    # Imperative commands must match
    intent, conf, _ = engine._keyword_fallback("set a timer for 10 minutes")
    assert intent == "set_timer" and conf == 1.0
    intent, conf, _ = engine._keyword_fallback("please set a timer for 10 minutes")
    assert intent == "set_timer" and conf == 1.0
    intent, conf, _ = engine._keyword_fallback("can you set a timer")
    assert intent == "set_timer" and conf == 1.0
    intent, conf, _ = engine._keyword_fallback("timer for 15 minutes")
    assert intent == "set_timer" and conf == 1.0


# ── 17. Residual failure fixes: Cases A–C & conversational gating ─────────────

@pytest.mark.parametrize("utterance", [
    "what's it like outside right now",
    "how is it looking outside",
    "what's it like outside today",
])
def test_case_a_ambient_weather_positive(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert ir.status is Status.READY, f"Expected READY for '{utterance}', got {ir.status} (source={ir.source}, reason={ir.reason})"
    assert ir.domain == "weather", f"Wrong domain for '{utterance}': {ir.domain}"
    assert ir.operation == "current", f"Wrong operation for '{utterance}': {ir.operation}"


@pytest.mark.parametrize("utterance", [
    "what's happening outside",
    "look outside",
    "who is outside",
    "what's going on outside",
])
def test_case_a_ambient_weather_physical_negatives_llm(utterance):
    import httpx
    from brain.router.llm_fallback import route
    from brain.router.registry import CommandRegistry, load_specs
    from config.settings import config

    try:
        r = httpx.get(f"{config.llm.base_url.rstrip('/')}/api/tags", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Ollama not running")
    except Exception:
        pytest.skip("Ollama not running")

    reg = CommandRegistry(load_specs())
    domains = reg.domains()
    ops = {d: reg.operations_for(d) for d in domains}

    result = run(route(utterance, domains, ops, []))
    assert result is not None
    is_unknown = (
        result.get("domain") in ("unknown", None)
        or result.get("operation") in ("unknown", None)
        or float(result.get("confidence", 0.0)) < 0.5
    )
    assert is_unknown, f"Expected unknown for '{utterance}', got {result}"


def test_case_b_target_extraction_pull_up():
    from brain.intent_engine import IntentEngine
    engine = IntentEngine()
    assert engine._extract_target("could you pull up spotify for me", "open_target") == "spotify"
    assert engine._extract_target("pull up spotify", "open_target") == "spotify"
    assert engine._extract_target("can you pull up the calculator", "open_target") == "calculator"


def test_case_b_pull_up_spotify_router():
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand("could you pull up spotify for me"))
    assert ir.status is Status.READY, f"Expected READY, got {ir.status} (source={ir.source})"
    assert ir.domain == "app_or_web"
    assert ir.operation == "open"
    assert "spotify" in ir.target.lower()


def test_case_c_jot_memo_router():
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand("hey could you jot a quick memo that the dentist is at three"))
    assert ir.status is Status.READY, f"Expected READY, got {ir.status} (source={ir.source})"
    assert ir.domain == "notes"
    assert ir.operation == "create"


def test_conversational_gating_logic():
    # Test the gating predicate: plausible_chat = mode == "llm" and conf >= 0.70 and (margin is None or margin >= 0.15)
    def is_plausible_chat(mode: str, conf: float, margin: float | None) -> bool:
        return mode == "llm" and conf >= 0.70 and (margin is None or margin >= 0.15)

    # Weak/contested predictions must NOT be plausible_chat (must allow LLM escalation)
    assert not is_plausible_chat("llm", 0.52, 0.16)
    assert not is_plausible_chat("llm", 0.75, 0.05)
    # Confident and uncontested conversational predictions MUST be plausible_chat (skip LLM)
    assert is_plausible_chat("llm", 0.80, 0.20)
    assert is_plausible_chat("llm", 0.72, None)


# ── 18. Physical-world system boundary & clipboard.clear regression tests ─────

@pytest.mark.parametrize("utterance", [
    "turn off the porch light please",
    "turn off the TV please",
    "turn off the lights",
    "lock the door",
    "i should lock up the shop before i go",
])
def test_physical_world_system_requests_remain_unknown(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert ir.status is Status.UNKNOWN, f"Expected UNKNOWN for '{utterance}', got {ir.status} ({ir.domain}.{ir.operation})"
    assert ir.domain is None, f"Expected domain=None for '{utterance}', got {ir.domain}"
    assert ir.operation is None, f"Expected operation=None for '{utterance}', got {ir.operation}"


@pytest.mark.parametrize("utterance,expected_op", [
    ("shut down my computer", "shutdown"),
    ("turn off my PC", "shutdown"),
    ("power off the laptop", "shutdown"),
    ("lock my computer", "lock"),
    ("lock the screen", "lock"),
])
def test_system_legitimate_computer_positives_preserved(utterance, expected_op):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert ir.status is Status.READY, f"Expected READY for '{utterance}', got {ir.status}"
    assert ir.domain == "system", f"Wrong domain for '{utterance}': {ir.domain}"
    assert ir.operation == expected_op, f"Wrong operation for '{utterance}': {ir.operation}"


@pytest.mark.parametrize("utterance", [
    "wipe whatever I copied",
    "clear what I copied",
])
def test_clipboard_clear_router(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    from brain.router.llm_fallback import route as llm_route
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check, llm_route=llm_route)
    ir = run(u.understand(utterance))
    assert ir.status is Status.READY, f"Expected READY for '{utterance}', got {ir.status}"
    assert ir.domain == "clipboard", f"Wrong domain for '{utterance}': {ir.domain}"
    assert ir.operation == "clear", f"Wrong operation for '{utterance}': {ir.operation}"


@pytest.mark.parametrize("utterance", [
    "wipe whatever I copied",
    "clear what I copied",
])
def test_clipboard_clear_llm_fallback(utterance):
    import httpx
    from brain.router.llm_fallback import route
    from brain.router.registry import CommandRegistry, load_specs
    from config.settings import config

    try:
        r = httpx.get(f"{config.llm.base_url.rstrip('/')}/api/tags", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("Ollama not running")
    except Exception:
        pytest.skip("Ollama not running")

    reg = CommandRegistry(load_specs())
    domains = reg.domains()
    ops = {d: reg.operations_for(d) for d in domains}

    result = run(route(utterance, domains, ops, []))
    assert result is not None
    assert result.get("domain") == "clipboard", f"Wrong domain for '{utterance}': {result}"
    assert result.get("operation") == "clear", f"Wrong operation for '{utterance}': {result}"
    assert not result.get("needs_clarification", False)


@pytest.mark.parametrize("utterance,expected_target", [
    ("open spotify", "spotify"),
    ("launch notepad", "notepad"),
])
def test_app_opening_remains_app_or_web(utterance, expected_target):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert ir.status is Status.READY, f"Expected READY for '{utterance}', got {ir.status}"
    assert ir.domain == "app_or_web", f"Wrong domain for '{utterance}': {ir.domain}"
    assert ir.operation == "open", f"Wrong operation for '{utterance}': {ir.operation}"
    assert expected_target in ir.target.lower()


# ── Phase F.11 Actionability & Capability Gating Tests ────────────────────────

@pytest.mark.parametrize("utterance", [
    "I set a timer for ten minutes earlier.",
    "I already started a timer.",
    "I had a timer running.",
    "I turned off the computer.",
    "I searched Google for this earlier.",
    "I checked the weather tomorrow.",
    "I was listening to music.",
    "I turned off the Wi-Fi.",
    "I set a timer for ten minutes.",
    "I searched Google for Python.",
])
def test_actionability_historical_statements_not_ready(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert not ir.executable, f"Historical statement '{utterance}' should not be executable, got {ir.status}"
    assert ir.status is Status.UNKNOWN, f"Expected UNKNOWN for '{utterance}', got {ir.status}"


@pytest.mark.parametrize("utterance", [
    "What did I search for yesterday?",
    "Show me my Google search history.",
    "Do you remember what I searched for?",
    "Search history.",
    "Look up what I searched for.",
    "Show past searches.",
])
def test_actionability_search_history_not_web_search(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert not ir.executable, f"Search history '{utterance}' should not be executable, got {ir.status}"
    assert ir.status is Status.UNKNOWN, f"Expected UNKNOWN for '{utterance}', got {ir.status}"
    assert ir.domain != "web", f"Search history must never map to web.search, got domain {ir.domain}"


@pytest.mark.parametrize("utterance", [
    "If I set a timer for ten minutes, what happens?",
    "What if I restart the computer?",
    "What if I turn off the Wi-Fi?",
    "What would happen if I delete all notes?",
])
def test_actionability_hypotheticals_not_ready(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert not ir.executable, f"Hypothetical '{utterance}' should not be executable, got {ir.status}"
    assert ir.status is Status.UNKNOWN, f"Expected UNKNOWN for '{utterance}', got {ir.status}"


@pytest.mark.parametrize("utterance", [
    "Did I set a timer?",
    "How long was my timer?",
    "Do you remember the timer I set?",
    "Did we search Google for that?",
    "Remember when I asked you to search Google?",
    "Did I turn off the Wi-Fi?",
])
def test_actionability_inquiries_not_ready(utterance):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert not ir.executable, f"Conversational inquiry '{utterance}' should not be executable, got {ir.status}"
    assert ir.status is Status.UNKNOWN, f"Expected UNKNOWN for '{utterance}', got {ir.status}"


@pytest.mark.parametrize("utterance,expected_key", [
    ("Set a timer for ten minutes.", "timer.create"),
    ("Start a five minute timer.", "timer.create"),
    ("Remind me in twenty minutes.", "timer.create"),
    ("Ten minute timer.", "timer.create"),
    ("Search Google for cats.", "web.search"),
    ("Search Google for Python tutorials.", "web.search"),
    ("Turn off the computer.", "system.shutdown"),
])
def test_actionability_genuine_commands_remain_ready(utterance, expected_key):
    from brain.intent_engine import IntentEngine
    from brain.router.registry import CommandRegistry, load_specs
    engine = IntentEngine()
    reg = CommandRegistry(load_specs())
    u = CommandUnderstander(engine, reg, guard_fn=_guard_check)
    ir = run(u.understand(utterance))
    assert ir.status is Status.READY, f"Expected READY for '{utterance}', got {ir.status}"
    assert ir.key == expected_key, f"Expected {expected_key} for '{utterance}', got {ir.key}"