"""
brain/router/entities.py
Level-2 deterministic entity extraction. Extractors are registered by entity
NAME (as declared in config/command_domains.json), so a new skill with a new
entity adds an extractor here or declares none (entity then stays optional or
is filled by the LLM fallback) — router core is untouched.

Extractors must accept exactly what the downstream skill can itself parse:

- extract_duration mirrors skills/utilities/timer.py::_parse_duration: the
  FIRST `<number> (hours?|hrs?|minutes?|mins?|seconds?|secs?)` match per unit,
  summed; number words (one..nineteen, tens up to sixty) as in its
  _words_to_digits. BATCH 1 FIX: it previously summed EVERY match
  (finditer) where the skill uses the first per unit, and accepted "zero"
  which the skill does not convert. A zero total is now "no duration".
  Bare "an hour"/"half an hour" remain unsupported (skill cannot parse them).
- _message mirrors timer.py::_extract_reminder.
- _location mirrors weather.py::_extract_location.
- "target"/"query" come only from the classifier's own extracted target.
"""
from __future__ import annotations

import re
from typing import Callable

# timer.py: _NUM_WORDS = one..nineteen; _TENS_WORDS = twenty..sixty; tens+one..nine.
_WORDS = {w: i for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split(), start=1)}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
_WORD_RE = re.compile(
    rf"\b(?:({'|'.join(_TENS)})(?:[\s-]({'|'.join(list(_WORDS)[:9])}))?|({'|'.join(_WORDS)}))\b"
    r"(?=\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b)", re.I)

# Same three patterns, same order, same single re.search each, as timer._parse_duration.
_DUR_PATTERNS = (
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b'), 3600),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b'), 60),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b'), 1),
)


def words_to_digits(text: str) -> str:
    def repl(m):
        if m.group(1):
            return str(_TENS[m.group(1).lower()] + (_WORDS[m.group(2).lower()] if m.group(2) else 0))
        return str(_WORDS[m.group(3).lower()])
    return _WORD_RE.sub(repl, text)


def extract_duration(text: str) -> int | None:
    """Seconds, or None (also None for a zero total)."""
    t = words_to_digits(text.lower())
    total, found = 0.0, False
    for pattern, mult in _DUR_PATTERNS:
        m = pattern.search(t)
        if m:
            total += float(m.group(1)) * mult
            found = True
    if not found or int(total) <= 0:
        return None
    return int(total)


# Mirrors skills/utilities/timer.py::_REMINDER_RE / _DURATION_RE / _TRAILING_RE
# / _extract_reminder exactly.
_REMINDER_RE = re.compile(r"\bremind(?:er)?\b.*?\b(?:to|about|that)\s+(.+)", re.I)
_MESSAGE_DURATION_RE = re.compile(
    r"\s*\b(?:(?:in|after|for)\s+)?\d+(?:\.\d+)?\s*"
    r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.I,
)
_TRAILING_RE = re.compile(r"(?:\s+(?:and|please|thanks))+\s*$", re.I)
_MESSAGE_MAX = 120


def _message(text: str, target: str):
    m = _REMINDER_RE.search(words_to_digits(text))
    if not m:
        return None
    msg = re.sub(r"\s{2,}", " ", _MESSAGE_DURATION_RE.sub("", m.group(1))).strip(" .,!?")
    msg = _TRAILING_RE.sub("", msg).strip(" .,!?")
    return msg[:_MESSAGE_MAX] or None


# Mirrors skills/web/weather.py::_extract_location / _TRAILING_TIME_RE exactly.
_LOC_RE = re.compile(r'\b(?:in|for|at)\s+([A-Za-z\s]+?)\s*[?.!]*$', re.I)
_TRAILING_TIME_RE = re.compile(
    r"(?:(?:^|\s+)(?:right\s+now|now|today|tonight|tomorrow|"
    r"this\s+(?:morning|afternoon|evening|weekend|week)|please|thanks|thank\s+you))+\s*$",
    re.I,
)


def _location(text: str, target: str):
    m = _LOC_RE.search(text)
    if not m:
        return None
    loc = _TRAILING_TIME_RE.sub("", m.group(1)).strip()
    if not loc or loc.lower() in ("the", "a"):
        return None   # "weather for today" -> auto-locate, same as the skill
    return loc


Extractor = Callable[[str, str], object]
EXTRACTORS: dict[str, Extractor] = {
    "duration": lambda text, target: extract_duration(text),   # int seconds
    "message":  _message,
    "location": _location,
    "target":   lambda text, target: target or None,           # classifier-derived
    "query":    lambda text, target: target or None,
}


def register_extractor(name: str, fn: Extractor) -> None:
    EXTRACTORS[name] = fn


def extract_entities(spec, text: str, target: str = "", llm_entities: dict | None = None):
    """-> (entities, missing_required). Deterministic values win over LLM strings."""
    ents, missing = {}, []
    for name, meta in spec.entities.items():
        fn = EXTRACTORS.get(name)
        val = fn(text, target) if fn else None
        if val in (None, ""):
            val = (llm_entities or {}).get(name)
            if name == "duration" and isinstance(val, str):
                val = extract_duration(val)
        if val not in (None, ""):
            ents[name] = val
        elif meta.get("required"):
            missing.append(name)
    return ents, tuple(missing)