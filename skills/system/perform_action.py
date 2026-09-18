"""
skills/system/perform_action.py
Direct, on-demand playback of one of Maya's action animations —
nod / giggle / sigh / shrug / wink — the same fixed vocabulary Ollama
uses via *action* tags (see services/llm/llm_service.py's ACTION TAGS
prompt section and _ACTION_VOCABULARY), but triggered deterministically
here from an explicit request like "Maya can you giggle a bit?" instead
of depending on the LLM choosing to include one.

Routing: brain/intent_engine.py has a dedicated guard (_ACTION_WORD_RE)
that classifies any utterance containing one of these action words as
the "perform_action" intent BEFORE the ML models even run, so this isn't
at the mercy of ML confidence on a phrasing it's never been trained on.

This skill only figures out WHICH action was requested and stores it on
the intent dict (intent["action"]) — core/processor.py is what actually
fires ws_server.broadcast_animation(), timed to when the confirmation
line's audio starts (same pattern as the "greet" -> wave animation).
"""

import logging

from config.settings import config

logger = logging.getLogger(__name__)
_U = config.user_name

# Must stay in sync with services/llm/llm_service.py's _ACTION_VOCABULARY —
# this is the same closed set of animations Maya actually has.
_ACTION_WORDS: dict[str, str] = {
    "giggl": "giggle",   # giggle, giggles, giggling
    "nod":   "nod",      # nod, nods, nodding
    "sigh":  "sigh",     # sigh, sighs, sighing
    "shrug": "shrug",    # shrug, shrugs, shrugging
    "wink":  "wink",     # wink, winks, winking
    "wynk":  "wink",     # Google STT sometimes mishears "wink" as "wynk" —
                          # same word, same action, just a common ASR typo.
}

_RESPONSES: dict[str, str] = {
    "giggle": "[excited] Hehe, okay senpai!",
    "nod":    "[relaxed] Mm-hm, there you go senpai.",
    "sigh":   "[relaxed] Alright senpai, if you insist.",
    "shrug":  "[relaxed] Sure, why not senpai.",
    "wink":   "[happy] Wink wink, senpai!",
}


def _resolve(text: str) -> str | None:
    t = text.lower()
    for word, action in _ACTION_WORDS.items():
        if word in t:
            return action
    return None


async def execute(intent: dict, text: str) -> str:
    action = _resolve(text)

    if not action:
        return (
            f"[surprised] Which one did you want, {_U}? "
            "I can nod, giggle, sigh, shrug, or wink."
        )

    # Stash the resolved action on the intent dict so processor.py can
    # sync ws_server.broadcast_animation() to exactly when this response's
    # audio starts playing, instead of firing it here (before synthesis
    # even happens).
    intent["action"] = action

    return _RESPONSES.get(action, f"[happy] There you go, {_U}!")