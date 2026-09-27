"""
core/behavior_engine.py
Semantic expression composer for Maya's avatar behavior.

This module is the bridge between the text-level expression choice and the
frontend's actual VRM morph animation. It does not decide what Maya should say;
it decides how a response should be expressed by combining a primary emotion,
optional attitude/intensity hints, and the current mood state into a rich
`communicative-intent` packet for the UI layer.

The engine accepts an expression tag such as `happy` or `angry`, then derives:
- a primary and secondary emotion
- a final attitude (`sincere`, `teasing`, `playful`, `mock`)
- an intensity value and intensity word band
- a gaze direction
- an action list and a resolved morph recipe

The recipe is resolved through `core/expression_library.py`, which provides a
cached lookup or a generated deterministic default. The frontend may then apply
only the morph keys it knows exist on the current VRM model; the rest of the
system continues to function normally even when a custom recipe includes keys the
loaded avatar does not support.

Important behavior:
- explicit `[attitude:word]` and `[intensity:word]` tags override the fallback
  inference when they are present and valid
- mood state still influences the final behavior so an active angry or sad mood
  continues to color otherwise neutral replies
- teasing and mock-outrage are treated as a short-lived expressive state rather
  than as a full persistent emotional lock
- this module is intentionally thin and stateless beyond reading the shared
  `mood_manager`; it does not mutate conversation or mood state itself
"""

import logging
import random

from core.mood import mood_manager
from core.expression_library import compose_default, get_recipe, save_recipe

logger = logging.getLogger(__name__)

_VALID_EXPRESSIONS = {"happy", "sad", "angry", "surprised", "relaxed", "neutral", "excited"}
_VALID_ATTITUDES   = {"sincere", "playful", "teasing", "mock"}

# Static tuning knobs for Maya's expressive character. These bias the overall
# composition without changing the higher-level intent logic.
PERSONALITY = {
    "dramaticity":    0.65,
    "playfulness":    0.7,
    "wit":            0.6,
    "expressiveness": 0.75,
    "chaos":          0.4,
    "subtlety":       0.3,
}

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
    """Convert a numeric intensity to the low/medium/high band used by the recipe library."""
    if value < 0.4:
        return "low"
    if value < 0.7:
        return "medium"
    return "high"


class BehaviorEngine:
    """Compose a semantic expression into the structured behavior packet sent to the frontend."""

    def compose(self, expression: str, actions: list[str] | None = None,
                source: str = "dialogue", attitude: str | None = None,
                intensity: str | None = None) -> dict:
        """Convert a primary expression into the final behavior packet sent to the avatar layer."""
        primary = expression if expression in _VALID_EXPRESSIONS else "neutral"
        actions = actions or []

        mood, mood_intensity = mood_manager.current()
        teasing = mood_manager.is_teasing()

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
        """Resolve a recipe from cache or default generation, then apply bounded jitter for this render only."""
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


behavior_engine = BehaviorEngine()
