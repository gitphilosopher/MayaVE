"""
services/llm/test_conversational_response.py
Regression tests for conversational response generation and post-processing.
"""

import pytest
import re
from services.llm.llm_service import _parse_expression, _resolve_action, _ASTERISK_RE


def test_action_resolution():
    """Verify that only defined action vocabulary items resolve to animations."""
    assert _resolve_action("nod") == "nod"
    assert _resolve_action("giggle") == "giggle"
    assert _resolve_action("giggling") == "giggle"
    assert _resolve_action("sigh") == "sigh"
    assert _resolve_action("shrug") == "shrug"
    assert _resolve_action("wink") == "wink"
    # Normal descriptive words or arbitrary text must NOT resolve to actions
    assert _resolve_action("down") is None
    assert _resolve_action("empty") is None
    assert _resolve_action("blue") is None
    assert _resolve_action("little") is None


def test_parse_expression_preserves_non_action_asterisks():
    """
    Ensure markdown emphasis or asterisks around descriptive words
    (e.g. *empty*, *down*, **blue**) are NOT wiped out, preventing truncated
    sentences like 'I am a little today, senpai.'
    """
    raw = "*sigh* [sad] I'm a little *empty* today, senpai."
    clean, exp, act, _, _ = _parse_expression(raw)
    assert clean == "I'm a little empty today, senpai."
    assert exp == "sad"
    assert act == ["sigh"]

    raw2 = "*sigh* [sad] I am feeling **really** *down* today, senpai."
    clean2, exp2, act2, _, _ = _parse_expression(raw2)
    assert clean2 == "I am feeling really down today, senpai."
    assert exp2 == "sad"
    assert act2 == ["sigh"]


def test_parse_expression_normalizes_hybrid_action_brackets():
    """Ensure '*[sigh]' or '[*sigh*]' are cleanly extracted as animations."""
    raw = "*[sigh] [sad] I'm just a little *shrug* off, senpai."
    clean, exp, act, _, _ = _parse_expression(raw)
    assert clean == "I'm just a little off, senpai."
    assert exp == "sad"
    assert act == ["sigh", "shrug"]


@pytest.mark.parametrize("sentence,expected_clean,expected_exp,expected_actions", [
    ("*giggle* [happy] Oh senpai, that's hilarious!", "Oh senpai, that's hilarious!", "happy", ["giggle"]),
    ("*wink* [excited] Look what I found, senpai!", "Look what I found, senpai!", "excited", ["wink"]),
    ("*shrug* [neutral] I'm functioning within optimal parameters, senpai.", "I'm functioning within optimal parameters, senpai.", "neutral", ["shrug"]),
    ("[happy] Everything looks healthy, senpai!", "Everything looks healthy, senpai!", "happy", []),
    ("*sigh* [sad] I really miss you, senpai...", "I really miss you, senpai...", "sad", ["sigh"]),
])
def test_parse_expression_standard_patterns(sentence, expected_clean, expected_exp, expected_actions):
    clean, exp, act, _, _ = _parse_expression(sentence)
    assert clean == expected_clean
    assert exp == expected_exp
    assert act == expected_actions


@pytest.mark.parametrize("utterance", [
    "how are you",
    "how r u",
    "how are you doing",
    "how is your day going",
])
def test_conversational_route_dispatches_to_llm(utterance):
    """
    Verify that conversational utterances route to conversational IR
    and are not misclassified as executable skills.
    """
    from brain.router.ir import Status, to_legacy_intent
    from brain.router.registry import CommandRegistry, load_specs
    from brain.router.understand import CommandUnderstander
    from brain.intent_engine import IntentEngine
    import asyncio

    engine = IntentEngine()
    registry = CommandRegistry(load_specs())
    understander = CommandUnderstander(engine, registry)
    
    ir = asyncio.run(understander.understand(utterance))
    assert ir.status == Status.UNKNOWN
    assert ir.reason == "conversational"
    assert not ir.executable

    legacy = to_legacy_intent(ir)
    assert legacy["response_mode"] == "llm"
    assert legacy["intent"] in {"smalltalk", "general_query", "unknown"}


def test_get_ollama_client_persistence_and_cleanup():
    """Verify that get_ollama_client returns a persistent instance and cleans up safely."""
    import asyncio
    from services.llm.llm_service import get_ollama_client, close_ollama_client

    async def _test():
        client1 = get_ollama_client()
        client2 = get_ollama_client()
        assert client1 is client2, "Persistent client should be reused within the same loop"
        assert not client1.is_closed

        await close_ollama_client()
        assert client1.is_closed

        client3 = get_ollama_client()
        assert client3 is not client1, "New client should be created after close"
        assert not client3.is_closed
        await close_ollama_client()

    asyncio.run(_test())


def test_get_ollama_client_loop_affinity():
    """Verify that get_ollama_client recreates clients safely across distinct event loops."""
    import asyncio
    from services.llm.llm_service import get_ollama_client, close_ollama_client

    async def _on_loop_1():
        return get_ollama_client()

    async def _on_loop_2():
        return get_ollama_client()

    c1 = asyncio.run(_on_loop_1())
    assert c1 is not None

    c2 = asyncio.run(_on_loop_2())
    assert c2 is not None
    assert c2 is not c1, "Different loops must receive separate client instances"
    asyncio.run(close_ollama_client())

