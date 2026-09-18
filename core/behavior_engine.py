"""
core/behavior_engine.py
Behavioral Engine — decides HOW Maya expresses a reaction, given WHAT she
wants to say. It does NOT pick the primary emotion or generate dialogue —
that's still Ollama's [expression] tags, skill-response tags, and
mood_manager (core/mood.py), unchanged. This module composes that single
tag plus current mood into a richer communicative-intent packet (primary/
secondary emotion, intensity, attitude, gaze, personality bias) so the
frontend Expression Composer (frontend/js/expression-composer.js) can
render blended VRM weights instead of the old flat tag -> single
blendshape map.

Semantic layer: Ollama may optionally supply [attitude:word] and
[intensity:word] tags alongside its existing emotion tag (parsed in
services/llm/llm_service.py's _parse_expression). When present, compose()
uses them directly; when absent (every pre-existing call site: skills,
timer alerts, filler, idle resets), attitude/intensity fall back to the
original mood/personality-derived math — fully backward compatible.

compose() also resolves a fine-grained "recipe" — a dict of verified
VRoid Fcl_BRW_*/Fcl_EYE_*/Fcl_MTH_* morph weights — via
core/expression_library.py (cached lookup or a generated + persisted
default). The frontend only applies recipe keys it has verified exist on
the loaded VRM (see expression-composer.js); everything else keeps
working exactly as before if the model lacks these customs.

Every call site that used to do `ws_server.broadcast_expression(tag)`
now does `ws_server.broadcast_behavior(behavior_engine.compose(tag, ...))`
at the exact same point in the pipeline — timing relative to audio
playback (critical for lip-sync/expression sync) is unchanged.

Deliberately thin: no cooldowns, no invented emotions/animations, no
duplicate mood/animation systems. Recipe lookups are cached in-process
(see expression_library.py) so this stays cheap on the per-sentence path.
"""

import logging
import random

from core.mood import mood_manager
from core.expression_library import compose_default, get_recipe, save_recipe

logger = logging.getLogger(__name__)

_VALID_EXPRESSIONS = {"happy", "sad", "angry", "surprised", "relaxed", "neutral", "excited"}
_VALID_ATTITUDES   = {"sincere", "playful", "teasing", "mock"}

# Personality bias — static tuning knobs for Maya's expressive character.
# See project Handoff for rationale; adjust here to shift her overall
# expressive style without touching composition logic.
PERSONALITY = {
    "dramaticity":    0.65,
    "playfulness":    0.7,
    "wit":            0.6,
    "expressiveness": 0.75,
    "chaos":          0.4,
    "subtlety":       0.3,
}

# Natural secondary-emotion bias per primary tag — reflects personality,
# not a literal "correct" emotion pairing. None means no bias by default.
_SECONDARY_BIAS = {
    "happy":     "excited",
    "excited":   "happy",
    "surprised": "happy",
    "angry":     None,   # only gains a secondary for mock-outrage (see compose())
    "sad":       "relaxed",
    "relaxed":   None,
    "neutral":   None,
}

_GAZE_BIAS = {
    "happy":     "direct",
    "excited":   "direct",
    "surprised": "direct",
    "angry":     "direct",
    "sad":       "away",
    "relaxed":   "soft",
    "neutral":   "soft",
}


def _intensity_word(value: float) -> str:
    """Buckets a numeric 0..1 intensity into the word band expressions.json
    is indexed by."""
    if value < 0.4:
        return "low"
    if value < 0.7:
        return "medium"
    return "high"


class BehaviorEngine:
    """Composes a single expression tag (+ optional attitude/intensity) and
    mood_manager's current state into a communicative-intent packet.
    Stateless aside from reading mood_manager — never mutates mood/
    conversation state itself."""

    def compose(self, expression: str, actions: list[str] | None = None,
                source: str = "dialogue", attitude: str | None = None,
                intensity: str | None = None) -> dict:
        primary = expression if expression in _VALID_EXPRESSIONS else "neutral"
        actions = actions or []

        mood, mood_intensity = mood_manager.current()
        teasing = mood_manager.is_teasing()

        # Ollama's own [attitude:word] tag wins when present/valid;
        # otherwise fall back to the existing mood-teasing inference.
        if attitude in _VALID_ATTITUDES:
            attitude_final = attitude
        else:
            attitude_final = "teasing" if teasing else "sincere"

        secondary = _SECONDARY_BIAS.get(primary)
        if primary == "angry" and (teasing or attitude_final == "mock"):
            secondary = "happy"  # mock outrage — angry delivery, playful undertone
        if mood in ("angry", "sad") and mood_intensity > 0.3 and mood != primary:
            secondary = mood  # active persistent mood bleeding into a different tag

        jitter = random.uniform(-0.05, 0.05) * PERSONALITY["chaos"]
        base_intensity = 0.45 + mood_intensity * 0.25 + PERSONALITY["dramaticity"] * 0.25 + jitter

        # Ollama's own [intensity:word] dominates when present; mood/
        # personality still contributes so an active mood keeps colouring
        # delivery strength even on a semantically-tagged sentence.
        if intensity in ("low", "medium", "high"):
            word_value = {"low": 0.3, "medium": 0.6, "high": 0.9}[intensity]
            intensity_value = word_value * 0.7 + base_intensity * 0.3
        else:
            intensity_value = base_intensity
        intensity_value = max(0.15, min(1.0, intensity_value))
        intensity_word = _intensity_word(intensity_value)

        gaze = _GAZE_BIAS.get(primary, "direct")
        if teasing:
            gaze = "direct"

        recipe, recipe_source = self._resolve_recipe(primary, attitude_final, intensity_word)

        intent = {
            "primary":   primary,
            "secondary": secondary,
            "intensity": round(intensity_value, 3),
            "attitude":  attitude_final,
            "gaze":      gaze,
            "actions":   actions,
            "recipe":    recipe,
        }

        active = {k: v for k, v in recipe.items() if v > 0.02}
        logger.info(
            f"Behavior[{source}]: semantic={primary}|{attitude_final}|{intensity_word} "
            f"recipe_source={recipe_source} morphs={active}"
        )
        return intent

    def _resolve_recipe(self, emotion: str, attitude: str, intensity_word: str) -> tuple[dict, str]:
        """Reuse a calibrated recipe if one exists; otherwise generate and
        persist a deterministic default so the semantic key is stable on
        future lookups. Bounded jitter is applied here, never saved."""
        try:
            cached = get_recipe(emotion, attitude, intensity_word)
            if cached is not None:
                base, source = cached, "cached"
            else:
                base = compose_default(emotion, attitude, intensity_word)
                save_recipe(emotion, attitude, intensity_word, base)
                source = "generated"
        except Exception as e:
            logger.debug(f"Expression library unavailable, using bare default: {e}")
            base = compose_default(emotion, attitude, intensity_word)
            source = "fallback"

        chaos = PERSONALITY["chaos"]
        jittered = {}
        for key, weight in base.items():
            bump = random.uniform(-0.04, 0.04) * chaos
            jittered[key] = round(max(0.0, min(1.0, weight + bump)), 3)
        return jittered, source


# Singleton — shared by every call site that used to call
# ws_server.broadcast_expression() directly.
behavior_engine = BehaviorEngine()
