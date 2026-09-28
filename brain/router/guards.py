"""
brain/router/guards.py
Deterministic fast paths that must NEVER go through semantic retrieval
or the LLM fallback (see the migration spec's "keep these out of
semantic retrieval" section).

Per the spec, existing guard regexes in brain/intent_engine.py must not
be duplicated. IntentEngine's dismissal/presence/action-word guards are
compiled per-instance from datasets/intents.json at __init__ time (they
depend on config.name/config.user_name and the configured action-word
table), so this module takes a live IntentEngine instance and calls its
existing detection logic directly rather than recompiling anything.

This does reach into IntentEngine's nominally-private attributes
(_is_dismissal, _PRESENCE_RE, _ACTION_WORD_RE, _ACTION_QUESTION_RE).
That is an intentional, documented trade-off for this migration only:
the alternative (copy-pasting those regexes here) is explicitly
forbidden by the spec and would let the two copies drift. If
intent_engine.py ever renames these, this module breaks loudly (an
AttributeError at import/call time) rather than silently — acceptable
for an internal same-package coupling.

canned_fast_paths: greet/farewell/thanks/help are not "guards" in
intent_engine.py's sense (they're ordinary keyword-matched intents),
but the spec explicitly wants them kept on their existing deterministic
path rather than re-routed through embeddings. They're included here
via IntentEngine's own keyword fallback, restricted to just this set —
a keyword hit for e.g. 'search_web' is NOT covered here; that is
legitimate command territory for the semantic router.
"""

from __future__ import annotations

import logging

from brain.intent_engine import IntentEngine

logger = logging.getLogger(__name__)

_CANNED_INTENTS = frozenset({"greet", "farewell", "thanks", "help"})


def check(engine: IntentEngine, text: str) -> tuple[str, float, str] | None:
    """
    Returns (intent, confidence, source) if a deterministic guard or a
    canned fast path fires, else None. Never raises — a broken guard
    must not block routing; it just means this utterance falls through
    to semantic/legacy handling instead.
    """
    try:
        t = text.lower().strip()

        if engine._is_dismissal(t):
            return "dismissal", 1.0, "guard:dismissal"

        if engine._PRESENCE_RE.search(t):
            return "smalltalk", 1.0, "guard:presence"

        if engine._ACTION_WORD_RE.search(t) and not engine._ACTION_QUESTION_RE.search(t):
            return "perform_action", 1.0, "guard:action"

        kw_intent, kw_conf, _ = engine._keyword_fallback(text)
        if kw_conf > 0 and kw_intent in _CANNED_INTENTS:
            return kw_intent, kw_conf, "guard:canned_keyword"

        return None
    except Exception as e:
        logger.warning(f"Router guard check failed (non-fatal, falling through): {e}")
        return None