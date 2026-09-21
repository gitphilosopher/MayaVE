"""
core/expression_library.py
Persistent calibrated expression-recipe library backing the semantic
Behavioral Engine (core/behavior_engine.py).

Semantic key: "emotion|attitude|intensity_word" (e.g. "surprised|playful|high").
A recipe is {morph_name: weight} built only from verified VRoid Fcl_BRW_*/
Fcl_EYE_*/Fcl_MTH_* primitives. Fcl_MTH_A/I/U/E/O (vowel visemes) are never
composed here — those belong to lip-sync.

Recipes on disk are the deterministic base (no random jitter) so a given
semantic key is stable across runs; behavior_engine.py applies bounded
jitter on top of the looked-up/generated base, never persists it.

File location: frontend/assets/expressions.json (same folder as the VRM
asset, alongside frontend/js/expression-lab.js's Export output — the
canonical shared file both the Lab and the backend read/write against).

Event-loop rule:
  save_recipe() is called from BehaviorEngine.compose(), which runs on the
  asyncio loop on the per-phrase path. The in-memory cache is updated
  immediately (so lookups see the new recipe at once) and the JSON is
  serialised on the caller, but the .tmp write + atomic replace happens on
  a dedicated single-thread writer. One worker means writes are applied in
  submission order — the last save always wins — and the non-daemon thread
  lets queued writes finish at interpreter exit. The first _load() (a small
  file read) is still synchronous.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LIB_DIR  = _PROJECT_ROOT / "frontend" / "assets"
_LIB_FILE = _LIB_DIR / "expressions.json"

_INTENSITY_BANDS  = {"low": 0.3, "medium": 0.6, "high": 0.9}
_VALID_ATTITUDES  = {"sincere", "playful", "teasing", "mock"}

# Never composed here — vowel/viseme mouth shapes belong to lip-sync.
_MOUTH_VISEME_KEYS = {"Fcl_MTH_A", "Fcl_MTH_I", "Fcl_MTH_U", "Fcl_MTH_E", "Fcl_MTH_O"}

# Compositional base per emotion — verified VRoid standard primitives,
# weight at intensity=1.0. Not a lookup table of full expressions; these
# are the finer-grained knobs combined per the composition principles.
_EMOTION_BASE: dict[str, dict[str, float]] = {
    "happy":     {"Fcl_BRW_Joy": 0.6,  "Fcl_EYE_Joy": 0.55, "Fcl_MTH_Joy": 0.7},
    "excited":   {"Fcl_BRW_Joy": 0.7,  "Fcl_EYE_Surprised": 0.35, "Fcl_EYE_Joy": 0.5,
                  "Fcl_MTH_Joy": 0.85, "Fcl_MTH_Large": 0.25},
    "surprised": {"Fcl_BRW_Surprised": 0.75, "Fcl_EYE_Surprised": 0.75, "Fcl_MTH_Surprised": 0.6},
    "angry":     {"Fcl_BRW_Angry": 0.75, "Fcl_EYE_Angry": 0.65, "Fcl_MTH_Angry": 0.55},
    "sad":       {"Fcl_BRW_Sorrow": 0.6, "Fcl_EYE_Sorrow": 0.6, "Fcl_MTH_Sorrow": 0.55,
                  "Fcl_EYE_Close_L": 0.15, "Fcl_EYE_Close_R": 0.15},
    "scared":    {"Fcl_BRW_Surprised": 0.9, "Fcl_EYE_Surprised": 0.85, "Fcl_EYE_Spread": 0.7,
                  "Fcl_MTH_Surprised": 0.6, "Fcl_MTH_Down": 0.35},
    "relaxed":   {"Fcl_EYE_Natural": 0.4, "Fcl_MTH_Neutral": 0.3},
    "neutral":   {"Fcl_EYE_Natural": 0.25, "Fcl_MTH_Neutral": 0.15},
}

# Attitude modifiers blended on top of the emotion base at reduced weight
# (playful surprise, mock anger, teasing, ...).
_ATTITUDE_MODIFIERS: dict[str, dict[str, float]] = {
    "playful": {"Fcl_BRW_Fun": 0.3,  "Fcl_EYE_Fun": 0.25,   "Fcl_MTH_Fun": 0.3},
    "teasing":  {"Fcl_BRW_Fun": 0.25, "Fcl_EYE_Joy_L": 0.2,  "Fcl_MTH_Fun": 0.2},
    "mock":     {"Fcl_MTH_Fun": 0.25},
    "sincere":  {},
}

# Per-channel-family nonlinear intensity exponent (lower = grows faster
# at low intensity) — mirrors the frontend's nonlinear composition curves.
_INTENSITY_EXPONENT = {
    "Fcl_BRW": 0.85,
    "Fcl_EYE": 0.9,
    "Fcl_MTH": 0.75,
}

_cache: dict | None = None
_load_failed = False   # file existed but was unreadable — never overwrite it

# Single worker → writes land in submission order; see module docstring.
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expr-lib-writer")


def _exponent_for(key: str) -> float:
    for prefix, exp in _INTENSITY_EXPONENT.items():
        if key.startswith(prefix):
            return exp
    return 0.85


def semantic_key(emotion: str, attitude: str, intensity_word: str) -> str:
    return f"{emotion}|{attitude}|{intensity_word}"


def _load() -> dict:
    global _cache, _load_failed
    if _cache is not None:
        return _cache
    try:
        data = json.loads(_LIB_FILE.read_text(encoding="utf-8")) if _LIB_FILE.exists() else {}
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        _cache = data
    except Exception as e:
        logger.warning(f"expressions.json load failed — it will not be overwritten this session: {e}")
        _cache = {}
        _load_failed = True
    return _cache


def _write_payload(payload: str) -> None:
    """Runs on the writer thread: .tmp write + atomic replace (a crash can't
    leave a half-written file). Never raises."""
    tmp = _LIB_FILE.with_name(_LIB_FILE.name + ".tmp")
    try:
        _LIB_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(_LIB_FILE)
    except Exception as e:
        logger.warning(f"expressions.json save failed (non-fatal): {e}")


def _save_all(data: dict) -> None:
    global _cache
    _cache = data
    if _load_failed:
        logger.warning("expressions.json was unreadable at load — skipping write to protect it.")
        return
    try:
        # Serialise here (microseconds for a few dozen small recipes) so the
        # writer thread receives an immutable string, not a live dict.
        payload = json.dumps(data, indent=2, sort_keys=True)
        _writer.submit(_write_payload, payload)
    except Exception as e:
        logger.warning(f"expressions.json save could not be scheduled (non-fatal): {e}")


def get_recipe(emotion: str, attitude: str, intensity_word: str) -> dict | None:
    """Look up a previously calibrated/saved recipe. None if not present."""
    return _load().get(semantic_key(emotion, attitude, intensity_word))


def save_recipe(emotion: str, attitude: str, intensity_word: str, recipe: dict) -> None:
    """Persist (or overwrite) a calibrated recipe for this semantic key.
    Cache updates immediately; the disk write is asynchronous."""
    data = dict(_load())
    data[semantic_key(emotion, attitude, intensity_word)] = recipe
    _save_all(data)


def compose_default(emotion: str, attitude: str, intensity_word: str) -> dict:
    """
    Deterministic default composition from verified VRoid primitives — the
    "reasonable initial composition" generated when no calibrated recipe
    exists yet. No random jitter (see module docstring), so the same
    semantic key always yields the same base until the Expression Lab
    calibrates and saves a replacement.
    """
    base = _EMOTION_BASE.get(emotion, _EMOTION_BASE["neutral"])
    modifier = _ATTITUDE_MODIFIERS.get(attitude if attitude in _VALID_ATTITUDES else "sincere", {})
    intensity = _INTENSITY_BANDS.get(intensity_word, _INTENSITY_BANDS["medium"])

    recipe: dict[str, float] = {}
    for key, weight in base.items():
        if key in _MOUTH_VISEME_KEYS:
            continue
        recipe[key] = round((intensity ** _exponent_for(key)) * weight, 3)

    for key, weight in modifier.items():
        if key in _MOUTH_VISEME_KEYS:
            continue
        add = (intensity ** _exponent_for(key)) * weight
        recipe[key] = round(recipe.get(key, 0.0) + add, 3)

    return recipe