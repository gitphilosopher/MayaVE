"""
brain/router/entities.py
Level-2 deterministic entity extraction. Extractors are registered by entity
NAME (as declared in config/command_domains.json), so a new skill with a new
entity adds an extractor here or declares none (entity then stays optional or
is filled by the LLM fallback) — router core is untouched.
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
    """Seconds, or None. Handles '10 minutes', 'an hour and a half'-lite, number words."""
    t = words_to_digits(text.lower())
    total, found = 0.0, False
    for m in _DUR_RE.finditer(t):
        total += float(m.group(1)) * _UNIT[m.group(2)[0].lower()]
        found = True
    if not found:
        if re.search(r"\bhalf an hour\b", t):
            return 1800
        if re.search(r"\b(?:an|one) hour\b", t):
            return 3600
        return None
    return int(total)


_REMINDER_RE = re.compile(r"\bremind(?:er)?\b.*?\b(?:to|about|that)\s+(.+)", re.I)
_LOC_RE = re.compile(r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z\s]*?)\s*[?.!]*$", re.I)
_TIME_WORDS = re.compile(r"(?:\s+(?:today|tonight|tomorrow|now|right now|please))+$", re.I)


def _message(text: str, target: str):
    m = _REMINDER_RE.search(text)
    if not m:
        return None
    msg = _DUR_RE.sub("", m.group(1))
    msg = re.sub(r"\b(?:in|after|for)\s*$", "", re.sub(r"\s{2,}", " ", msg).strip(" .,!?")).strip(" .,!?")
    return msg[:120] or None


def _location(text: str, target: str):
    m = _LOC_RE.search(text)
    loc = _TIME_WORDS.sub("", m.group(1)).strip() if m else ""
    return loc if loc and loc.lower() not in ("the", "a") else None


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
