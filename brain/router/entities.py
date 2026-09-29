"""
brain/router/entities.py
Level-2 deterministic entity extraction. Extractors are registered by entity
NAME (as declared in config/command_domains.json), so a new skill with a new
entity adds an extractor here or declares none (entity then stays optional or
is filled by the LLM fallback) — router core is untouched.

PATCH (stabilization pass) — these extractors must accept exactly what the
downstream skill can itself parse, no more:

- extract_duration: dropped "an hour"/"half an hour" bare-word support.
  skills/utilities/timer.py::_parse_duration only matches
  `<number> (hours?|hrs?|minutes?|mins?|seconds?|secs?)` — it has no bare
  "an hour"/"half an hour" case. Accepting those here would let the IR
  mark a request READY with duration=3600 while the skill itself, given
  the same rewritten text, could re-derive a different (or no) duration —
  a silent mismatch between what the router promises and what the skill
  actually does. Word-number conversion ("twenty five minutes") IS kept,
  because timer.py's own `_words_to_digits` does the same conversion
  before parsing.
- _message: now mirrors timer.py::_extract_reminder exactly — duration
  phrases and trailing "and/please/thanks" are stripped the same way, so
  the router's entity and the skill's own re-derived message never
  disagree after a context rewrite hands the skill different text.
- _location: now mirrors weather.py::_extract_location — the same
  trailing-time-word strip and the same "the"/"a"-only rejection, so
  "weather for today" resolves to auto-locate (None) here exactly as it
  does inside the skill, instead of the router extracting the literal
  word "today" as a location.

No speculative extractors were added beyond what config/command_domains.json
declares (duration, message, location, target, query).
"""
from __future__ import annotations

import re
from typing import Callable

_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
_WORD_RE = re.compile(
    rf"\b(?:({'|'.join(_TENS)})(?:[\s-]({'|'.join(list(_WORDS)[1:10])}))?|({'|'.join(_WORDS)}))\b"
    r"(?=\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b)", re.I)
_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b", re.I)
_UNIT = {"h": 3600, "m": 60, "s": 1}


def words_to_digits(text: str) -> str:
    def repl(m):
        if m.group(1):
            return str(_TENS[m.group(1).lower()] + (_WORDS[m.group(2).lower()] if m.group(2) else 0))
        return str(_WORDS[m.group(3).lower()])
    return _WORD_RE.sub(repl, text)


def extract_duration(text: str) -> int | None:
    """Seconds, or None. Word-number conversion only ("twenty five minutes")
    — see module docstring for why bare "an hour"/"half an hour" support
    was deliberately removed; skills/utilities/timer.py can't parse those
    either, so a router-side extraction here would promise something the
    skill can't independently reproduce."""
    t = words_to_digits(text.lower())
    total, found = 0.0, False
    for m in _DUR_RE.finditer(t):
        total += float(m.group(1)) * _UNIT[m.group(2)[0].lower()]
        found = True
    if not found:
        return None
    return int(total)


# Mirrors skills/utilities/timer.py::_REMINDER_RE / _DURATION_RE / _TRAILING_RE
# / _extract_reminder exactly, so the router's "message" entity and the
# skill's own re-derived message never disagree.
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
        return None   # e.g. "weather for today" -> auto-locate, same as the skill
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
