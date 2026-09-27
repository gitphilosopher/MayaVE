"""
skills/system/perform_action.py
Trigger a specific avatar animation from a user request.

This skill handles direct action requests such as "giggle", "nod", "sigh",
"shrug", or "wink". It is intentionally narrow and deterministic: the intent
router classifies action words before any ML model runs, so the request is not
left to generative ambiguity or a low-confidence classification.

The skill resolves the requested action from the user text, stores it on the
`intent` dict as `intent["action"]`, and returns the spoken confirmation. The
actual animation dispatch happens later in the processor pipeline, where the
browser-side animation is synchronized to the relevant playback point.
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
    """Find the first supported animation keyword in the user text and map it to Maya's action name."""
    t = text.lower()
    for word, action in _ACTION_WORDS.items():
        if word in t:
            return action
    return None


async def execute(intent: dict, text: str) -> str:
    """Resolve the action request, attach it to the intent, and return the spoken confirmation."""
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