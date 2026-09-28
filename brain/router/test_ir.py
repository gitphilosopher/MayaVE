"""Tests for the Command IR pipeline. Needs the small schemas.py/registry.py patch (requires_confirmation)."""
import asyncio

from brain.router.entities import extract_duration
from brain.router.ir import Status, to_legacy_intent
from brain.router.registry import CommandRegistry
from brain.router.schemas import CommandSpec
from brain.router.understand import CommandUnderstander

SPECS = [
    CommandSpec("timer", "create", "set_timer",
                {"duration": {"required": True, "prompt": "How long should I set it for"},
                 "message": {"required": False}}, "raw"),
    CommandSpec("system", "shutdown", "shutdown", {}, "raw", requires_confirmation=True),
]


class FakeLegacy:
    _response_modes = {"set_timer": "skill", "shutdown": "skill", "smalltalk": "llm", "greet": "skill"}

    def __init__(self, table):
        self.table = table

    def classify(self, text):
        for needle, (intent, conf, margin, model) in self.table.items():
            if needle in text.lower():
                return {"intent": intent, "confidence": conf, "margin": margin, "model": model,
                        "target": "", "response_mode": self._response_modes.get(intent, "llm")}
        return {"intent": "unknown", "confidence": 0.1, "margin": 0.0, "model": "low_confidence_fallback",
                "target": "", "response_mode": "llm"}


def make(table, llm=None):
    return CommandUnderstander(FakeLegacy(table), CommandRegistry(SPECS), llm_route=llm)


T = {"timer": ("set_timer", 0.97, 0.5, "pytorch+tensorflow"), "shut": ("shutdown", 0.97, 0.5, "pytorch+tensorflow")}
run = asyncio.run


def test_duration_words():
    assert extract_duration("for twenty five minutes") == 1500
    assert extract_duration("in an hour") == 3600
    assert extract_duration("no time here") is None


def test_ready_with_entities():
    ir = run(make(T).understand("set a timer for ten minutes"))
    assert ir.status is Status.READY and ir.key == "timer.create" and ir.entities["duration"] == 600


def test_missing_entity_needs_clarification_not_guess():
    ir = run(make(T).understand("set a timer"))
    assert ir.status is Status.NEEDS_CLARIFICATION and ir.missing_entities == ("duration",)
    assert to_legacy_intent(ir)["intent"] == "clarify" and ir.prompt.endswith("?")


def test_pending_clarification_resolves():
    u = make(T)
    run(u.understand("set a timer"))
    ir = run(u.understand("ten minutes"))
    assert ir.status is Status.READY and ir.entities["duration"] == 600 and ir.source.startswith("context")


def test_correction_uses_last_unit():
    u = make(T)
    run(u.understand("set a timer for 20 minutes"))
    ir = run(u.understand("actually make it 30"))
    assert ir.status is Status.READY and ir.entities["duration"] == 1800


def test_low_confidence_command_is_never_executed():
    ir = run(make({"timer": ("set_timer", 0.55, 0.02, "pytorch+tensorflow")}).understand("timer maybe"))
    assert ir.status is Status.UNKNOWN and ir.reason == "low_confidence"
    assert to_legacy_intent(ir)["intent"] == "unknown"


def test_invalid_llm_output_is_rejected():
    async def bad(*a, **k):
        return {"domain": "rocket", "operation": "launch", "entities": {}}
    ir = run(make({}, llm=bad).understand("launch the rocket"))
    assert ir.status is Status.REJECTED


def test_llm_unavailable_degrades_to_unknown():
    async def down(*a, **k):
        return None
    assert run(make({}, llm=down).understand("blah")).status is Status.UNKNOWN


def test_confirmation_flag_carried():
    assert run(make(T).understand("shut it down")).requires_confirmation
