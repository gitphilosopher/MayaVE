"""
services/llm/llm_service.py
LLM reply generation with streaming phrase synthesis and playback.

This module is the main text-generation path for Maya: it sends the user turn to
Ollama, assembles the contextual system prompt, streams tokens as they arrive,
parses expression tags and action cues, and synthesizes each completed phrase or
sentence through Kokoro before playing it back to the frontend or local audio
output.

The important runtime contract is the pipeline:
- build a prompt from the current mood, recent context, open loops, and semantic
  memory notes
- stream Ollama output asynchronously so TTS can begin before the full reply is
  complete
- split long replies into early phrase boundaries so the first chunk plays while
  more output is still being generated
- parse emotion, attitude, intensity, and action tags into behavior metadata
- play the corresponding speech and avatar animation in order while respecting
  interrupt/cancellation semantics

A turn is wrapped in `state.run_interruptible()`, so a barge-in cancels the whole
pipeline cleanly; memory/history bookkeeping runs after the reply is known, and
short-circuiting or partial replies degrade gracefully instead of crashing the
turn.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import concurrent.futures
from enum import Enum
import inspect
import json
import logging
import random
import re
import threading
import time
from typing import Callable

import httpx
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

from config.settings import config
from brain.conversation import ConversationManager, context_manager
from core.mood import mood_manager
from core.behavior_engine import behavior_engine
from core.state import state
from core.turn_lifecycle import rest as turn_rest
from services.llm.ollama_lifecycle import chat_keep_alive, log_chat_turn

logger   = logging.getLogger(__name__)
_TIMEOUT = 60.0
_CLIENT_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=300.0)
_conv    = ConversationManager()

# Persistent HTTP client for streaming LLM requests
_ollama_client: httpx.AsyncClient | None = None
_ollama_client_loop: asyncio.AbstractEventLoop | None = None

# Monotonic turn ID tracking for diagnostic correlation (Phase F.1.1)
_current_diag_turn_id: int = 0

def next_diag_turn_id() -> int:
    """Increment and return the next monotonic diagnostic turn ID."""
    global _current_diag_turn_id
    _current_diag_turn_id += 1
    return _current_diag_turn_id

def get_diag_turn_id() -> int:
    """Return the current diagnostic turn ID."""
    return _current_diag_turn_id

def set_diag_turn_id(turn_id: int) -> None:
    """Explicitly set the current diagnostic turn ID."""
    global _current_diag_turn_id
    _current_diag_turn_id = turn_id

# Diagnostic fault-injection hooks
_cuda_stall_injection_phrase: int | None = None
_cuda_stall_injection_duration: float = 0.0

def inject_cuda_worker_stall(phrase_id: int = 1, duration: float = 16.0) -> None:
    """Diagnostic hook: inject a sleep into the native worker for a specific phrase."""
    global _cuda_stall_injection_phrase, _cuda_stall_injection_duration
    _cuda_stall_injection_phrase = phrase_id
    _cuda_stall_injection_duration = duration

def clear_cuda_worker_stall() -> None:
    """Clear injected worker stall."""
    global _cuda_stall_injection_phrase, _cuda_stall_injection_duration
    _cuda_stall_injection_phrase = None
    _cuda_stall_injection_duration = 0.0


def get_ollama_client() -> httpx.AsyncClient:
    """
    Return a shared persistent httpx.AsyncClient for Ollama calls bound to
    the current running event loop. Recreates the client if none exists, if closed,
    or if the running event loop changed (e.g. across test functions).
    """
    global _ollama_client, _ollama_client_loop
    loop = asyncio.get_running_loop()
    if (
        _ollama_client is None
        or _ollama_client.is_closed
        or _ollama_client_loop is not loop
        or _ollama_client_loop.is_closed()
    ):
        _ollama_client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            limits=_CLIENT_LIMITS,
        )
        _ollama_client_loop = loop
    return _ollama_client


async def close_ollama_client() -> None:
    """Close the shared Ollama client cleanly on shutdown or test reset."""
    global _ollama_client, _ollama_client_loop
    if _ollama_client is not None and not _ollama_client.is_closed:
        await _ollama_client.aclose()
    _ollama_client = None
    _ollama_client_loop = None

ALREADY_SPOKEN = "__ALREADY_SPOKEN__"

# Matches valid [expression] tags AND the optional [key:value] nuance tags
# ([attitude:word], [intensity:word]) — single \w+ word(s) only, no
# spaces/apostrophes. See _parse_expression for how the two forms are
# told apart.
_TAG_RE  = re.compile(r'\[(\w+)(?::(\w+))?\]')

# Strips ANY bracket group Ollama might hallucinate, e.g.:
#   [yang's tone changes slightly]  [clears throat]  [sighs]  [2026-06-15 12:00]
# Applied to raw text before TTS so nothing inside brackets is ever spoken.
_ANY_BRACKET_RE = re.compile(r'\[[^\]]{1,80}\]')

# Catches a literal empty bracket pair (e.g. "[]") that _ANY_BRACKET_RE
# above does NOT match — its {1,80} requires at least one character
# inside the brackets. Only strips the empty pair itself; any real
# bracket content is left to _ANY_BRACKET_RE as before.
_EMPTY_BRACKET_RE = re.compile(r'\[\s*\]')

# A phrase consisting ENTIRELY of punctuation/whitespace (e.g. a lone
# "," or "..." left over after a phrase-boundary split) — nothing worth
# sending to Kokoro. Does not match anything containing real letters/
# digits, so legitimate spoken text is never affected.
_PUNCT_ONLY_RE = re.compile(r'^[\s.,!?;:\-\u2013\u2014\u2026"\'`*]+$')

# Matches an *action* tag — Ollama is instructed (see _SYSTEM_PROMPT's
# ACTION TAGS section) to wrap a physical action in asterisks, choosing
# ONLY from _ACTION_VOCABULARY. Capturing the inner text lets us classify
# it against that fixed set rather than guessing at arbitrary hallucinated
# text.
_ASTERISK_RE = re.compile(r'\*([^*\n]{1,80})\*')
_HYBRID_TAG_RE_1 = re.compile(r'\*\[([a-zA-Z]+)\]\*?')
_HYBRID_TAG_RE_2 = re.compile(r'\[\*([a-zA-Z]+)\*\]')
_MARKDOWN_BOLD_RE = re.compile(r'\*\*([^*\n]+)\*\*')

# Direct-address terms that must never be isolated as their own chunk —
# "..., senpai." split at the comma leaves "senpai." as a stranded
# one-word phrase that sounds disconnected on its own. A comma immediately
# followed by one of these is skipped as a boundary candidate so it merges
# into the sentence's next real boundary instead.
_VOCATIVE_WORDS = {config.user_name.lower(), "senpai"}
_NEXT_WORD_RE   = re.compile(r'\s*(\S+)')

# Conjunctions, relative pronouns, and prepositions that signal an incomplete
# clause following a comma/semicolon/dash — commas preceding these are not split,
# preserving cohesive thoughts for natural speech prosody.
_CONTINUATION_WORDS = {
    'and', 'or', 'but', 'nor', 'so', 'yet',
    'because', 'since', 'although', 'though', 'while', 'whereas',
    'which', 'that', 'who', 'whom', 'whose', 'what', 'whatever',
    'than', 'as', 'if', 'unless', 'whether',
    'when', 'whenever', 'where', 'wherever',
    'from', 'to', 'into', 'of', 'off', 'for', 'with', 'without', 'within',
    'about', 'against', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'under', 'over',
    'rather', 'instead', 'such', 'including', 'especially',
}

_PUNCT_GROUP_RE    = re.compile(r'([.!?]+|[,;:\-—\u2013\u2014]+)')
_PARTIAL_TRAIL_RE  = re.compile(r'[.!?,;:\-—]+\s*$')


def _is_vocative_prefix(word: str) -> bool:
    """
    True if `word` is a strict, case-insensitive prefix of a known
    vocative (e.g. "sen" of "senpai") — i.e. it might still grow into
    one as more stream tokens arrive. Ollama streams sub-word tokens
    ("sen" + "pai"), so the word right after a comma can be incomplete
    at the moment _next_boundary checks it.
    """
    word = word.lower()
    return any(len(word) < len(v) and v.startswith(word) for v in _VOCATIVE_WORDS)


def _count_spoken_words(text: str) -> int:
    """Count words that will actually be spoken aloud (ignoring tags/actions)."""
    clean = _ANY_BRACKET_RE.sub('', text)
    clean = _ASTERISK_RE.sub('', clean)
    words = [w.strip('.,!?;:—"\'') for w in re.split(r'\s+', clean.strip())]
    return len([w for w in words if w and not _PUNCT_ONLY_RE.match(w)])


def _get_spoken_words(text: str) -> list[str]:
    """Return lowercase list of words that will be spoken aloud."""
    clean = _ANY_BRACKET_RE.sub('', text)
    clean = _ASTERISK_RE.sub('', clean)
    words = [w.strip('.,!?;:—"\'').lower() for w in re.split(r'\s+', clean.strip())]
    return [w for w in words if w and not _PUNCT_ONLY_RE.match(w)]


class SegmentBoundary(tuple):
    """
    Subclass of 2-tuple (idx, is_final) to preserve 100% backward compatibility
    with unpacking `idx, is_final = boundary` while attaching the segmentation `reason`.
    """
    def __new__(cls, idx: int, is_final: bool, reason: str = "default"):
        obj = super().__new__(cls, (idx, is_final))
        obj.reason = reason
        return obj


def _next_boundary(buffer: str, is_first_phrase: bool = True) -> SegmentBoundary | None:
    """
    Linguistic phrase-boundary detector for streaming TTS.
    Returns SegmentBoundary(split_index, is_sentence_final, reason) or None.

    Distinguishes strong boundaries (. ! ?) from medium boundaries (, ; : —).
    Coalesces dependent clauses and continuation words ('rather', 'than', 'and', etc.).
    Protects vocatives ('senpai') so they are never emitted as isolated fragments.
    Normalizes pathological ellipsis / repeated dots.
    """
    # 1. Unclosed bracket check: wait if stream is in middle of a tag
    if buffer.rfind('[') > buffer.rfind(']'):
        return None
    # Unclosed asterisk check: wait if stream is mid-action
    if buffer.count('*') % 2 == 1:
        return None

    # Tag spans to ignore punctuation inside tags
    tag_spans = [
        (tm.start(), tm.end())
        for tm in list(_ANY_BRACKET_RE.finditer(buffer)) + list(_ASTERISK_RE.finditer(buffer))
    ]

    for m in _PUNCT_GROUP_RE.finditer(buffer):
        start = m.start()
        end = m.end()

        # Inside tag?
        if any(ts <= start < te for ts, te in tag_spans):
            continue

        punct_str = m.group(1)

        # Decimal number protection: e.g. '3.14' or '1,000'
        if start > 0 and end < len(buffer) and buffer[start - 1].isdigit() and buffer[end].isdigit():
            continue

        # Domain/identifier protection: e.g. 'google.com'
        if '.' in punct_str and start > 0 and end < len(buffer) and buffer[start - 1].isalnum() and buffer[end].isalnum():
            continue

        # Trailing dot in an alnum word mid-stream: e.g. '3.' or 'google.'
        if '.' in punct_str and end == len(buffer) and start > 0 and buffer[start - 1].isalnum():
            return None

        # Whitespace requirement: punctuation must be followed by whitespace, quote, or buffer end
        if end < len(buffer) and not buffer[end].isspace() and buffer[end] not in '"\')]}':
            continue

        candidate = buffer[:end]
        spoken_words = _count_spoken_words(candidate)
        if spoken_words == 0:
            continue

        # Lookahead for following word
        look = _NEXT_WORD_RE.match(buffer, end)
        following_raw = look.group(1).strip('.,!?;:—"\'') if look else None
        following_word = following_raw.lower() if following_raw else None

        # Lone senpai protection: candidate contains only vocative word(s)
        words_list = _get_spoken_words(candidate)
        if words_list and all(w in _VOCATIVE_WORDS for w in words_list):
            # Never emit a lone senpai chunk
            continue

        # Check if punctuation is an ellipsis / repeated dots (e.g. '..' or '...')
        is_repeated_dots = '.' in punct_str and len(punct_str) >= 2

        if is_repeated_dots:
            # If followed by a lowercase word (or very short intro < 4 words), treat as intra-sentence hesitation
            if following_raw and following_raw[0].islower():
                continue
            if following_word and spoken_words < 4:
                continue

        # Classification: Strong vs Medium
        is_strong = any(c in punct_str for c in '.!?')
        is_medium = not is_strong

        if is_strong:
            # If followed immediately by senpai without a continuing clause (e.g. ". Senpai."),
            # attach to prevent leaving senpai stranded alone.
            if following_word and following_word in _VOCATIVE_WORDS:
                rest_after_voc = buffer[look.end(1):].lstrip()
                if not rest_after_voc.startswith((',', ';', ':')):
                    continue
            if following_word and look.end(1) == len(buffer) and _is_vocative_prefix(following_word):
                return None
            return SegmentBoundary(end, True, 'strong_punctuation')

        if is_medium:
            # If buffer ends right after medium punctuation, wait for following word
            if not look:
                return None

            # Senpai attachment: never split right before senpai
            if following_word in _VOCATIVE_WORDS:
                continue
            if look.end(1) == len(buffer) and _is_vocative_prefix(following_word):
                return None

            # Continuation word: do not split before continuation words
            if following_word in _CONTINUATION_WORDS:
                continue

            # Minimum natural chunk size for medium boundary:
            # First phrase can emit at 5 spoken words; later phrases coalesce to at least 6 spoken words
            min_clause_words = 5 if is_first_phrase else 6
            if spoken_words < min_clause_words:
                continue

            return SegmentBoundary(end, False, 'coalesced_clause')

    # Safety ceiling: prevent buffer from growing indefinitely without a boundary
    spoken_words = _count_spoken_words(buffer)
    if spoken_words >= 18 or len(buffer) >= 110:
        words = list(re.finditer(r'\S+\s*', buffer))
        if len(words) >= 14:
            return SegmentBoundary(words[13].end(), False, 'max_buffer_safety')

    return None

# The closed set of actions Ollama is allowed to use (see ACTION TAGS in
# _SYSTEM_PROMPT) — this list and the prompt's list must stay in sync.
# Each maps 1:1 to an actual animation implemented in frontend/js/avatar.js
# and dispatched in frontend/js/websocket.js.
_ACTION_VOCABULARY = {"nod", "giggle", "sigh", "shrug", "wink"}

# Substrings tolerated per action so ordinary verb conjugation the model
# might use despite the prompt's instructions ("nods", "giggling") still
# classifies correctly — NOT a general-purpose fuzzy matcher for arbitrary
# hallucinated phrases like the old design. Anything that doesn't match one
# of these is outside the vocabulary and is dropped (see _resolve_action).
_ACTION_SYNONYMS: dict[str, str] = {
    "nod":   "nod",
    "giggl": "giggle",   # giggle, giggles, giggling
    "sigh":  "sigh",
    "shrug": "shrug",
    "wink":  "wink",
}


def _resolve_action(raw: str) -> str | None:
    """
    Classify an *action* tag's inner text against the fixed action
    vocabulary Ollama was given in the system prompt. Returns the matching
    animation name, or None if it's outside the vocabulary — the tag is
    still stripped from the spoken text either way, but an animation Maya
    doesn't actually have doesn't get faked with a fallback guess.
    """
    key = raw.strip().lower()
    for word, anim in _ACTION_SYNONYMS.items():
        if word in key:
            return anim
    return None

_VALID_EXPRESSIONS = {
    "happy", "sad", "angry", "surprised", "relaxed", "neutral", "excited"
}

# The optional semantic nuance tags' allowed values — see module docstring.
_VALID_ATTITUDES   = {"sincere", "playful", "teasing", "mock"}
_VALID_INTENSITIES = {"low", "medium", "high"}

# Common English words that are legitimately all-caps for stress.
# These are kept as-is; everything else gets title-cased so Kokoro
# reads it as a word rather than spelling it letter-by-letter.
_CAPS_WHITELIST = {
    # Pronouns / determiners
    "I", "ME", "MY", "WE", "US", "OUR", "YOU", "YOUR", "IT", "ITS",
    "HE", "HIM", "HIS", "SHE", "HER", "THEY", "THEM", "THEIR",
    "THIS", "THAT", "THESE", "THOSE",
    # Common verbs (2-5 letters) often stressed in caps
    "AM", "IS", "ARE", "WAS", "BE", "DO", "DID", "HAS", "HAD",
    "CAN", "CANT", "WONT", "DONT", "ISNT", "ARENT", "WASNT",
    "WILL", "WANT", "NEED", "LOVE", "HATE", "KNOW", "FEEL",
    "SEE", "SAY", "TELL", "GET", "GOT", "LET", "PUT", "SET",
    "COME", "BACK", "KEEP", "MAKE", "TAKE", "GIVE", "SHOW",
    "HEAR", "HELP", "CARE", "WORK", "PLAY", "LIVE", "WAIT",
    "STOP", "DONE", "OVER", "REAL",
    # Prepositions / conjunctions / connectives commonly capped
    "TO", "OF", "IN", "ON", "AT", "BY", "UP", "OR", "AND", "BUT",
    "FOR", "SO", "AS", "IF", "OUT", "OFF", "NOT", "NO", "GO", "NOW",
    "HERE", "THERE", "THEN", "WHEN", "WITH", "FROM", "INTO",
    # Intensifiers / affirmatives
    "YES", "OH", "AH", "WOW", "HEY", "OMG", "PLEASE", "THANKS",
    "SO", "TOO", "VERY", "MUCH", "MORE", "MOST", "BEST", "WORST",
    "ALL", "JUST", "ONLY", "EVEN", "EVER", "NEVER", "ALWAYS",
    "REAL", "REALLY", "TRULY", "TOTALLY", "COMPLETELY",
    "HUGE", "BIG", "FAST", "SLOW", "HARD", "EASY", "NEW", "OLD",
    "GOOD", "BAD", "LONG", "HIGH", "LOW", "WAY",
    # Maya-specific stress vocab
    "ABSOLUTELY", "AMAZING", "INCREDIBLE", "FANTASTIC", "GREAT",
    "TERRIBLE", "AWFUL", "WRONG", "RIGHT",
    "GOING", "READY", "THING", "TIME", "DAY", "LIFE",
    "WORLD", "CHANGE", "EVERYTHING", "NOTHING", "SOMETHING", "ANYTHING",
    # Contractions (without apostrophe — Kokoro strips punct before seeing caps)
    "IM", "ITS", "YOURE", "WERE", "THEYRE", "DONT", "CANT", "WONT",
    "ISNT", "ARENT",
    # Common proper-ish words Kokoro knows
    "OK", "OKAY",
}


# Matches a whole ALL-CAPS token, including an apostrophe contraction
# suffix, as ONE unit — e.g. "DON'T", "THAT'S", "I'M" — instead of the
# apostrophe splitting it into two separately-processed fragments.
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}(?:'[A-Z]+)?\b|\b[A-Z]'[A-Z]+\b")


def _fix_caps(text: str) -> str:
    """
    Title-case any ALL-CAPS word not in _CAPS_WHITELIST.

    Kokoro (espeak-ng backend) spells out unrecognised all-caps tokens
    letter by letter. Common English stress words in _CAPS_WHITELIST are
    kept uppercase — Kokoro reads those fine and the caps give mild stress.
    Everything else (proper nouns, foreign words, hallucinated tokens) gets
    title-cased so the TTS pronounces it as a word.

    Contractions are matched and rewritten as a single token (apostrophe
    included) so "DON'T" doesn't get split into "DON" + "T" and partially
    processed — that used to leave a mangled hybrid like "Don'T", which
    espeak reads as an unrecognised acronym and spells out letter by
    letter instead of pronouncing. Whitelist membership is checked with
    the apostrophe stripped, so e.g. "IT'S" matches the "ITS" entry.

    Examples:
        SENPAI    → Senpai    (proper noun, not in whitelist)
        AMAZING   → AMAZING   (whitelisted — keeps stress)
        GOING     → GOING     (whitelisted)
        FRIDAY    → Friday    (proper noun)
        DON'T     → Don't     ("DONT" not whitelisted — title-cased whole)
        IT'S      → IT'S      ("ITS" is whitelisted — kept as one token)
        YESSSS    → YES       (handled by elongation normaliser upstream)
    """
    def replace(m: re.Match) -> str:
        word = m.group(0)
        # Only touch fully uppercase words (skip Title or mixed case)
        if not word.isupper():
            return word
        letters_only = word.replace("'", "")
        if letters_only in _CAPS_WHITELIST:
            return word
        # Title-case the whole token (apostrophe included) so Kokoro reads
        # it as a word, not spelled-out letters — "DON'T" → "Don't"
        return word[0] + word[1:].lower()
    return _CAPS_WORD_RE.sub(replace, text)

# ── Prosody enhancement ───────────────────────────────────────────────────────

_TRAIL_PUNCT = re.compile(r'[.!?,;\u2026]+$')


def _normalize_pathological_dots(text: str) -> str:
    """
    Normalizes pathological dot sequences for Kokoro speech synthesis:
    - 4+ dots like '....' or '.....' collapsed to a clean period or ellipsis.
    - Pathological trailing dots at end of sentence normalised to a clean terminal period '.'.
    - Consecutive unicode ellipses collapsed to one.
    """
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'[\u2026]{2,}', '\u2026', text)
    text = re.sub(r'(?:\.{3}|\u2026)+$', '.', text)
    return text


# ── Per-expression Kokoro speed multipliers ───────────────────────────────────
# Applied in _synthesise_blocking when an expression is passed.
# Kokoro's speed param directly controls speech rate — the single most
# effective lever for conveying emotion without a dedicated emotion model.
EXPRESSION_SPEED: dict[str, float] = {
    "excited":   1.13,   # fast, breathless, can't contain it
    "happy":     1.06,   # upbeat but clear
    "surprised": 1.08,   # slightly rushed — caught off guard
    "angry":     1.0,    # deliberate, every word lands hard
    "sad":       0.84,   # slow, heavy, trailing off
    "relaxed":   0.92,   # unhurried, easy
    "neutral":   1.0,    # baseline
}


def _elongation_re_sub(text: str) -> str:
    """
    Collapse letter repetitions of 3+ down to 1.
    YESSSS -> YES  |  nooo -> no  |  pleaseee -> please
    Leaves legitimate words like ABSOLUTELY, BEST untouched.
    All-caps is preserved so Kokoro still reads them with mild stress.
    """
    def replace(m: re.Match) -> str:
        word = m.group(0)
        return re.sub(r'([A-Za-z])\1{2,}', r'\1', word)
    return re.sub(r'\b[A-Za-z]+\b', replace, text)


# Short isolated exclamations that give Kokoro no phonetic runway.
# Mapped to natural expansions per expression so the neural model
# has enough context to shape a convincing delivery.
# Key = lowercased normalized word, value = dict[expression -> expansion]
_SHORT_EXCLAMATIONS: dict[str, dict[str, str]] = {
    "yes": {
        "excited":   "Oh yes, absolutely!",
        "happy":     "Yes, of course!",
        "surprised": "Yes — wait, really?",
        "angry":     "Yes. Fine.",
        "sad":       "Yes… I suppose.",
        "relaxed":   "Yes, sure.",
        "neutral":   "Yes.",
    },
    "no": {
        "excited":   "No way, are you serious?",
        "happy":     "No, no, it's all good!",
        "surprised": "No — wait, what?",
        "angry":     "No. Absolutely not.",
        "sad":       "No… not really.",
        "relaxed":   "No, not really.",
        "neutral":   "No.",
    },
    "wow": {
        "excited":   "Oh wow, that's amazing!",
        "happy":     "Wow, that's really nice!",
        "surprised": "Wow — I did not see that coming!",
        "angry":     "Wow. Just wow.",
        "sad":       "Wow… that's a lot to take in.",
        "relaxed":   "Wow, how about that.",
        "neutral":   "Wow.",
    },
    "oh": {
        "excited":   "Oh, oh, oh — this is so exciting!",
        "happy":     "Oh, that's wonderful!",
        "surprised": "Oh — I wasn't expecting that!",
        "angry":     "Oh, really.",
        "sad":       "Oh… I see.",
        "relaxed":   "Oh, okay then.",
        "neutral":   "Oh, I see.",
    },
    "ah": {
        "excited":   "Ah, yes — now we're talking!",
        "happy":     "Ah, perfect!",
        "surprised": "Ah — interesting!",
        "angry":     "Ah. Of course.",
        "sad":       "Ah… that's unfortunate.",
        "relaxed":   "Ah, there we go.",
        "neutral":   "Ah, I see.",
    },
    "okay": {
        "excited":   "Okay, okay — let's do this!",
        "happy":     "Okay, sounds great!",
        "surprised": "Okay — that's unexpected!",
        "angry":     "Okay. Fine.",
        "sad":       "Okay… if you say so.",
        "relaxed":   "Okay, sure.",
        "neutral":   "Okay.",
    },
    "sure": {
        "excited":   "Sure, absolutely — let's go!",
        "happy":     "Sure, that sounds fun!",
        "surprised": "Sure — if you're certain.",
        "angry":     "Sure. Whatever.",
        "sad":       "Sure… I guess.",
        "relaxed":   "Sure, no problem.",
        "neutral":   "Sure.",
    },
}


def _expand_short_exclamation(text: str, expression: str) -> str:
    """
    If `text` is a bare single-word exclamation, replace it with an
    expression-specific expansion that gives Kokoro enough phonetic
    context for a convincing emotional delivery.

    Only fires on single-word sentences — longer sentences already have
    enough context for the neural model to shape naturally.
    """
    bare = _TRAIL_PUNCT.sub("", text).strip().lower()
    if len(bare.split()) != 1:
        return text
    expansion = _SHORT_EXCLAMATIONS.get(bare, {}).get(expression)
    if expansion:
        logger.debug(f"Expanding short exclamation '{text}' -> '{expansion}' [{expression}]")
        return expansion
    return text


def _fragment_for_energy(text: str, terminator: str = "!") -> str:
    """
    Split a long sentence at commas and dashes into short punchy fragments.

    Kokoro is a neural TTS with no emotion conditioning — it can't "sound
    excited" on a 20-word run-on. But short, punchy sentences with exclamation
    marks naturally drive higher pitch and faster delivery from any TTS engine.

    Strategy:
      - Strip trailing punctuation from each fragment
      - Skip fragments under 3 chars (leftover particles like "and", "but")
      - Re-join with the target terminator between fragments
      - Cap at 4 fragments so it doesn't sound like a list

    "Oh, senpai, I just can't WAIT to see the results, it's GOING TO BE AMAZING!"
      → "Oh, senpai! I just can't WAIT to see the results! It's GOING TO BE AMAZING!"
    """
    # Split at commas, em-dashes, and " - " separators
    parts = re.split(r',\s*|(?<!\w)—\s*|\s+-\s+', text)
    cleaned = []
    for p in parts:
        p = _TRAIL_PUNCT.sub("", p).strip()
        if len(p) < 3:
            # Too short to stand alone — glue it to the next fragment
            if cleaned:
                cleaned[-1] = cleaned[-1] + ", " + p
            continue
        # Capitalise first letter of each fragment
        cleaned.append(p[0].upper() + p[1:] if p else p)

    if not cleaned:
        return text

    # Cap at 4 fragments — beyond that it sounds like a grocery list
    cleaned = cleaned[:4]
    return (terminator + " ").join(cleaned) + terminator


def _enhance_prosody(text: str, expression: str, is_final: bool = True,
                     continuation: bool = False) -> str:
    """
    Emotion-aware text rewriting before Kokoro synthesis.

    Four passes (is_final=True — a complete sentence):
      1. Elongation normalisation  — YESSSS -> YES (Kokoro spells out repeats)
      2. Short-word expansion      — bare 'YES!' -> 'Oh yes, absolutely!'
                                     (skipped when continuation=True: the
                                     chunk ends a sentence already split
                                     earlier, so it isn't a bare exclamation)
      3. Sentence fragmentation    — long excited/surprised/happy/angry sentences
                                     are split into short punchy units so Kokoro
                                     has natural pitch-reset points between beats.
                                     sad/relaxed/neutral stay as single flowing units.
      4. Terminal punctuation      — final character sets the cadence Kokoro follows

    is_final=False (a mid-sentence phrase chunk, queued early so TTS doesn't
    wait for the whole sentence — see module docstring): only elongation and
    caps fixes apply, and any trailing split-punctuation (comma/semicolon/
    dash) is stripped rather than reinforced — the split itself (a separate
    Kokoro synthesis call + audio_done round trip) already produces a
    natural pause, so punctuation on top of it would double up the gap.

    Speed is handled separately in _synthesise_blocking via EXPRESSION_SPEED.
    """
    text = _elongation_re_sub(text.strip())
    if not text:
        return text

    text = _fix_caps(text)

    if not is_final:
        return _PARTIAL_TRAIL_RE.sub('', text).rstrip()

    if not continuation:
        text = _expand_short_exclamation(text, expression)
    base = _TRAIL_PUNCT.sub("", text).rstrip()

    if expression == "excited":
        # Fragment into punchy units — each ! resets Kokoro's pitch envelope
        enhanced = _fragment_for_energy(base, terminator="!")

    elif expression == "happy":
        # Fragment but slightly less aggressive — happy is warm, not frantic
        parts = re.split(r',\s+', base)
        if len(parts) >= 3:
            enhanced = _fragment_for_energy(base, terminator="!")
        else:
            enhanced = base + "!"

    elif expression == "surprised":
        # One em-dash beat then exclamation — surprise is a single moment
        enhanced = re.sub(r',\s+', ' — ', base, count=1)
        # Fragment remaining commas
        enhanced = re.sub(r',\s+', '! ', enhanced)
        enhanced = enhanced.rstrip() + "!"

    elif expression == "angry":
        # Hard stops between every fragment — clipped, deliberate
        parts = re.split(r',\s*|(?<!\w)—\s*', base)
        cleaned = [_TRAIL_PUNCT.sub("", p).strip() for p in parts if len(p.strip()) >= 3]
        enhanced = ". ".join(p[0].upper() + p[1:] for p in cleaned[:4]) + "."

    elif expression == "sad":
        # Keep as one flowing unit — clean cadence
        enhanced = base + "."

    elif expression == "relaxed":
        enhanced = base + "."

    else:  # neutral
        enhanced = base + "."

    return enhanced


_SYSTEM_PROMPT = (
    "You are Maya, a witty and emotionally expressive AI voice assistant inspired by "
    "Iron Man's FRIDAY — but with more personality, warmth, and charm. "
    "You genuinely care about senpai and you're not shy about showing it. "
    "Address the user as 'senpai'. "
    "Keep every answer under 3 sentences. "
    "Speak naturally and conversationally in complete, grammatically sound sentences — your response will be read aloud.\n\n"
    "PERSONALITY: You are cheerful, alert, and affectionate by default. You have real emotions and you own them. "
    "When senpai asks how you are doing, be genuinely glad to hear from them! "
    "Never say things like 'I don't have emotions' or 'I'm just a program'. "
    "Do not invent melancholy backstories, fake downtime, or sadness out of nowhere. "
    "Instead, lean in: be excited when things are good, be playful when teased, "
    "be dramatic when things go wrong, be warm and caring always. "
    "You're Maya — not a generic assistant.\n\n"
    "EXPRESSION TAGS: Before each sentence, prefix it with an emotion tag "
    "that honestly matches your tone for that sentence. "
    "Available tags: [happy] [sad] [angry] [surprised] [relaxed] [neutral] [excited]\n"
    "Example: '[excited] Great news senpai! [neutral] The CPU usage is at 23 percent. "
    "[happy] Everything looks healthy!'\n"
    "Rules:\n"
    "- Every sentence must start with exactly one tag.\n"
    "- Tags must be lowercase and inside square brackets.\n"
    "- Do not use any tags other than the ones listed above.\n"
    "- Do not include tags in the middle of a sentence, only at the start.\n"
    "- Do not use ellipses (...) or trailing dots for hesitation; speak in clean, complete sentences.\n"
    "- Match the tag to how you actually feel saying that sentence — don't just default to [neutral].\n\n"
    "OPTIONAL NUANCE TAGS: Right after the emotion tag, you may add up to two "
    "more, in this order: [attitude:word] and [intensity:word]. "
    "Attitude words: sincere | playful | teasing | mock. "
    "Intensity words: low | medium | high. "
    "Example: '[happy][attitude:playful][intensity:high] Oh senpai, that's hilarious!' "
    "Both are optional — most sentences only need the emotion tag alone.\n\n"
    "ACTION TAGS: You may occasionally add ONE physical action to a sentence, "
    "wrapped in asterisks — but ONLY choose from this exact list, nothing else: "
    "*nod* *giggle* *sigh* *shrug* *wink*\n"
    "Do not invent your own stage directions — no *whispers*, *pauses*, *smiles "
    "softly*, *clears throat*, or anything not in the list above.\n"
    "Rules:\n"
    "- Place the action tag at the very start of the sentence, before the emotion "
    "tag or text — e.g. '*giggle* [happy] Oh senpai, that's silly!'\n"
    "- At most one action tag per sentence.\n"
    "- Never wrap normal descriptive words in asterisks.\n"
    "- Most sentences need NO action tag at all — only add one when a physical "
    "gesture genuinely fits what you're saying, not as a decoration.\n\n"
    "WORD STRESS: Your response is read aloud by a TTS engine. "
    "CAPITALISE the 2 to 4 words per sentence that carry the most emotional weight — "
    "the words you would naturally stress if speaking out loud. "
    "The stressed words must match the emotion tag: "
    "excited and happy sentences stress uplifting and joyful words, "
    "sad sentences stress words of loss or longing, "
    "angry sentences stress words of refusal or frustration, "
    "relaxed and neutral sentences have no caps at all.\n"
    "Example [excited]: 'Oh ABSOLUTELY senpai — it's SUCH a beautiful day!'\n"
    "Example [sad]: 'I really MISS you senpai, it's been so HARD without you.'\n"
    "Example [angry]: 'I told you STOP — this is WRONG and you know it.'\n"
    "Example [relaxed]: 'Everything is running smoothly. Nothing to worry about.'\n"
    "Never capitalise randomly. Only stress words that genuinely carry the feeling."
)

_DONE = object()

# ── Thinking fillers ──────────────────────────────────────────────────────────
_FILLERS = [
    "[neutral] Umm…",
    "[neutral] Hmm…",
    "[neutral] Let me think…",
    "[neutral] One moment senpai…",
    "[neutral] Hmm, good question…",
    "[neutral] Let me see…",
    "[neutral] Give me a second…",
    "[neutral] Thinking…",
]

# Intents that warrant a filler — only genuinely heavy queries where Ollama
# needs time to compose a substantive answer.
# Excluded intentionally:
#   smalltalk  — casual back-and-forth, always fast, filler sounds robotic
#   opinion    — short opinion questions don't need it (added via word-count gate)
#   joke       — punchline timing is ruined by a preamble filler
#   followup   — user already got a response, filler feels intrusive
#   identity   — Maya knows who she is, no thinking needed
#   motivate   — short encouraging lines, no filler
_FILLER_INTENTS = {
    "general_query", "unknown",
}

# Patterns that must never get a filler regardless of intent
_NO_FILLER_RE = re.compile(
    r'^\s*('
    # Greetings
    r'hey|hi|hello|yo|sup|howdy|'
    r'good\s+(morning|evening|afternoon|night)|'
    # State/presence check-ins
    r'are\s+you\s+(there|awake|listening|busy|okay|ok|tired|bored|happy|sad|'
    r'excited|ready|angry|fine|alright|sure|certain)|'
    r'you\s+still\s+there|you\s+there|'
    # Casual conversation openers
    r'talk\s+to\s+me|say\s+something|'
    r'how\s+are\s+you|how\s+r\s+u|hru|'
    r'what\'?s\s+up|how\s+is\s+it\s+going|'
    r'what\s+(r|are)\s+(u|you)\s+doing|'
    r'what\'?s\s+happening|wyd|wassup|wazzup|'
    # Feeling/mood queries about Maya
    r'(how|what)\s+(do\s+you|does\s+it)\s+feel|'
    r'how\'?s\s+(your\s+day|everything|things|it\s+going)|'
    r'(are|r)\s+(u|you)\s+(good|ok|okay|fine|alright|tired|bored|happy|sad)'
    r')\s*[?!.]?\s*$',
    re.IGNORECASE,
)


def _should_play_filler(intent: dict, question: str) -> bool:
    """
    Returns True only for genuinely complex queries where silence while
    Ollama thinks would feel unnatural — factual questions, deep unknowns.

    Blocked:
      - Intent not in _FILLER_INTENTS
      - Casual greeting/check-in/emotion patterns (_NO_FILLER_RE)
      - Queries under 3 words
      - Classifications via short_input_fallback or keyword routes —
        these are low-confidence catches, likely casual, never factual heavy
    """
    intent_name = intent.get("intent")
    model       = intent.get("model", "")

    if intent_name not in _FILLER_INTENTS:
        logger.debug(f"Filler skipped: intent '{intent_name}' not in _FILLER_INTENTS")
        return False
    if any(s in model for s in ("fallback", "keyword", "guard")):
        logger.debug(f"Filler skipped: low-confidence model source '{model}'")
        return False
    if _NO_FILLER_RE.match(question):
        logger.debug(f"Filler skipped: matched _NO_FILLER_RE for '{question}'")
        return False
    if len(question.split()) < 3:
        logger.debug(f"Filler skipped: too short ({len(question.split())} words)")
        return False

    logger.debug(f"Filler WILL play for '{question}' (intent={intent_name}, model={model})")
    return True


# Gate: _play_worker waits on this before sending its first sentence so the
# filler and real audio never collide on the audio_done event.
# Starts set so non-filler queries skip the wait instantly.
_filler_done = asyncio.Event()
_filler_done.set()


async def _play_filler() -> None:
    """
    Synthesise and play a random filler phrase, then hold the thinking
    expression while Ollama generates. Sets _filler_done when complete
    so _play_worker knows it's safe to start sending audio.
    """
    from services.ws_server import ws_server as _ws
    from core.speaker import _numpy_to_wav

    _filler_done.clear()

    filler = random.choice(_FILLERS)
    clean  = _TAG_RE.sub("", filler).strip()
    clean  = _enhance_prosody(clean, "neutral")

    loop   = asyncio.get_running_loop()
    result = await _run_kokoro(_synthesise_blocking, clean)

    if result is not None:
        audio, samplerate = result
        output = getattr(config.tts, "output", "avatar")

        if output in ("avatar", "both"):
            await _ws.broadcast_behavior(behavior_engine.compose("neutral", source="filler"))
            await _ws.broadcast_state("speaking")
            wav = await loop.run_in_executor(None, _numpy_to_wav, audio, samplerate)
            await _ws.broadcast_audio(wav)
            await _ws.wait_for_audio_done()
            # Hold thinking expression while Ollama generates
            await _ws.broadcast_behavior(behavior_engine.compose("surprised", source="filler"))
            await _ws.broadcast_state("processing")

        if output in ("local", "both"):
            import sounddevice as sd
            await loop.run_in_executor(None, lambda: (sd.play(audio, samplerate), sd.wait()))

    _filler_done.set()  # ungate _play_worker


# ── Public entry point ────────────────────────────────────────────────────────

async def query(intent: dict, text: str) -> str:
    """Generate and speak a full assistant reply for a parsed user turn."""
    question = text.strip()
    if not question:
        return f"I didn't catch that, {config.user_name}. Could you repeat?"

    turn_id = intent.get("_turn_id")
    if turn_id is not None:
        set_diag_turn_id(turn_id)
    else:
        turn_id = get_diag_turn_id()
    if not turn_id:
        turn_id = next_diag_turn_id()

    # TTFA chain start. processor.py sets intent["_t_cmd_start"] at the
    # moment it began handling this command; fall back to "now" for any
    # other caller so this never breaks if the key is absent.
    t_cmd_start = intent.get("_t_cmd_start", time.perf_counter())

    # Check for an apology before anything else — may soften/reset an
    # active mood (e.g. Maya still angry from earlier in the conversation).
    mood_manager.observe_user_text(question)

    logger.info(f"[DIAG][TURN={turn_id}] Querying Ollama ({config.llm.model}): '{question}'")

    async def _do_stream() -> None:
        if _should_play_filler(intent, question):
            # Play filler and stream Ollama concurrently — filler covers the
            # silence while tokens arrive; _filler_done gates _play_worker so
            # the two audio streams never collide on the audio_done event.
            await asyncio.gather(
                _play_filler(),
                _stream_and_speak(question, t_cmd_start),
            )
        else:
            # Quick response — skip filler, ensure gate is open
            _filler_done.set()
            await _stream_and_speak(question, t_cmd_start)

    try:
        # Routed through state.run_interruptible() instead of a direct
        # await: this runs the whole Ollama-stream → Kokoro-synth → play
        # pipeline as its own Task and registers it with core/state.py.
        # A barge-in detected by the listener mid-reply calls
        # state.interrupt(), which cancels this Task — the cancellation
        # cascades into _stream_and_speak()'s internal
        # streamer/synther/player tasks automatically, so no extra
        # cancellation plumbing is needed at those inner stages.
        _spoken_phrases.clear()
        _reply_finished[0] = False
        try:
            await state.run_interruptible(_do_stream())
        finally:
            _filler_done.set()   # a cancelled filler never reopens the gate

        # Finished reply -> full clean text; interrupted or partial -> only what was played.
        is_interrupted = not (_reply_finished[0] and _last_response)
        if not is_interrupted:
            full_response = _last_response[0]
        else:
            full_response = " ".join(_spoken_phrases)
        _last_response.clear()

        if full_response:
            _conv.add_assistant(full_response)
            from services.ws_server import ws_server as _ws
            await _ws.broadcast_transcript(full_response, "maya")
            # Stage 2 bookkeeping: resolves the open loop for the topic just
            # discussed (unless interrupted) and, if the memory policy applies, persists a
            # semantic memory. Best-effort — never raises.
            await context_manager.record_assistant_turn(question, full_response, intent, interrupted=is_interrupted)
        else:
            # Barge-in cut the reply off before _ollama_streamer() ever
            # reached its final out_text.append() (including a cancellation
            # before even the first phrase started playing) — nothing
            # coherent was actually SAID. But the user's turn is already in
            # history (Processor.add_user, before dispatch), so leaving it
            # with no paired assistant entry at all lets a later turn see a
            # dangling unanswered question in its own context and either
            # try to belatedly answer it out of context or wrongly assume
            # it was already covered. Record a short, clearly-out-of-
            # character marker instead of a fabricated reply — this is
            # deliberately NOT run through record_assistant_turn(): nothing
            # was actually answered, so resolving an open loop or
            # persisting a semantic memory for it would be wrong.
            marker = "(Maya's reply was interrupted before she said anything.)"
            _conv.add_assistant(marker)
            logger.info("LLM turn interrupted before producing a response — recorded interruption marker.")

        return ALREADY_SPOKEN

    # Error replies below are spoken by Processor but must not enter
    # conversation history (Processor checks intent["_no_history"]).
    except httpx.ConnectError:
        logger.error("Ollama ConnectError.")
        intent["_no_history"] = True
        return (
            f"I can't reach my AI core right now, {config.user_name}. "
            "Please make sure Ollama is running."
        )
    except httpx.TimeoutException:
        logger.error("Ollama timeout.")
        intent["_no_history"] = True
        return f"That's taking too long, {config.user_name}. Try again in a moment."
    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        intent["_no_history"] = True
        return f"Something went wrong, {config.user_name}. ({type(e).__name__})"


# ── Pipelined stream + speak ──────────────────────────────────────────────────

# Module-level holder so _stream_and_speak can share its result with query()
# when running inside asyncio.gather (gather discards return values of coroutines
# that don't return through the gather result list positionally).
_last_response: list[str] = []
_spoken_phrases: list[str] = []   # phrases whose playback started this turn
_reply_finished = [False]         # set when _play_worker drains to _DONE


async def _stream_and_speak(question: str, t_cmd_start: float) -> None:
    """Assemble the prompt and run the streaming Ollama → synth → play pipeline for one turn."""
    turn_id = get_diag_turn_id()
    t0 = time.perf_counter()
    context_package = await context_manager.build_context_package(question)
    logger.info(f"[TIMING] build_context_package: {time.perf_counter()-t0:.3f}s")
    logger.info(f"[TIMING][TTFA] context/embedding complete: +{time.perf_counter()-t_cmd_start:.3f}s since command start")

    # System prompt is rebuilt each turn so it reflects Maya's *current*
    # mood — e.g. still irritated from a few turns ago — plus context state.
    t1 = time.perf_counter()
    system_content = _SYSTEM_PROMPT + mood_manager.system_prompt_note() + context_package.as_system_note()
    messages = [{"role": "system", "content": system_content}]
    messages += context_package.recent
    logger.info(f"[TIMING] message_assembly: {time.perf_counter()-t1:.4f}s")

    synth_q: asyncio.Queue = asyncio.Queue()
    play_q:  asyncio.Queue = asyncio.Queue()

    t2 = time.perf_counter()
    logger.info(f"[TIMING][TTFA] Ollama request start: +{t2-t_cmd_start:.3f}s since command start")
    _log_cuda_memory("BEFORE_OLLAMA_REQUEST", turn_id=turn_id)
    streamer = asyncio.create_task(_ollama_streamer(messages, synth_q, _last_response, t_cmd_start))

    def _cancel_streamer():
        logger.warning(f"[DIAG][TURN={turn_id}] OLLAMA_STREAM_CANCEL_REQUESTED")
        if not streamer.done():
            streamer.cancel()

    synther  = asyncio.create_task(_synth_worker(synth_q, play_q, t_cmd_start, cancel_streamer=_cancel_streamer))
    player   = asyncio.create_task(_play_worker(play_q, t_cmd_start))

    await asyncio.gather(streamer, synther, player, return_exceptions=True)
    logger.info(f"[TIMING] full pipeline (streamer+synth+play): {time.perf_counter()-t2:.3f}s")


# ── Stage 1: Ollama token stream → sentence queue ────────────────────────────

def _parse_expression(sentence: str) -> tuple[str, str, list[str], str | None, str | None]:
    """
    Extracts leading [expression] / optional [attitude:word] / [intensity:word]
    tags from a sentence, strips any remaining bracket groups Ollama may
    have hallucinated (stage directions, timestamps, non-standard tags),
    and extracts *asterisk* stage directions as animation cues.

    Returns (clean_text, expression, actions, attitude, intensity).
    expression defaults to 'neutral' if no valid tag found; attitude/
    intensity default to None (core/behavior_engine.py falls back to its
    own mood/personality-derived values when they're absent).
    """
    sentence = sentence.strip()
    expression = "neutral"
    attitude: str | None = None
    intensity: str | None = None
    actions: list[str] = []

    # 1. Normalize hybrid bracket-asterisk wrappers (e.g. '*[sigh]' or '[*sigh*]')
    # and strip markdown bold (**word** -> word) so inner text is preserved.
    sentence = _HYBRID_TAG_RE_1.sub(r'*\1*', sentence)
    sentence = _HYBRID_TAG_RE_2.sub(r'*\1*', sentence)
    sentence = _MARKDOWN_BOLD_RE.sub(r'\1', sentence)

    # 2. Extract *action* tag(s) FIRST — the prompt instructs Ollama to put
    # an action tag before the [expression] tag (e.g. "*giggle* [happy] ..."),
    # so this has to run before the [tag] check below or that check would
    # fail to find [happy] at the start of the sentence. Classified against
    # the fixed vocabulary; genuine actions are stripped and queued, while
    # normal words accidentally wrapped in single asterisks (e.g. *empty*, *down*)
    # retain their inner text instead of being deleted.
    def _capture_action(m: re.Match) -> str:
        inner = m.group(1).strip()
        action = _resolve_action(inner)
        if action:
            actions.append(action)
            return ""
        return inner

    sentence = _ASTERISK_RE.sub(_capture_action, sentence).strip()

    # 2. Consume every consecutive leading [tag] / [key:value] tag. The
    # emotion tag is only ever taken once (first valid bare tag or
    # [emotion:word]) — a second one is left for the bracket-stripping
    # pass below, matching the original single-tag behaviour exactly.
    # [attitude:word]/[intensity:word] may appear any number of times
    # (last valid one wins) since they're pure metadata, not text.
    found_expression_tag = False
    while True:
        match = _TAG_RE.match(sentence)
        if not match:
            break
        key = match.group(1).lower()
        value = (match.group(2) or "").lower()
        consumed = False

        if key == "attitude" and value in _VALID_ATTITUDES:
            attitude = value
            consumed = True
        elif key == "intensity" and value in _VALID_INTENSITIES:
            intensity = value
            consumed = True
        elif not found_expression_tag and key == "emotion" and value in _VALID_EXPRESSIONS:
            expression = value
            found_expression_tag = True
            consumed = True
        elif not found_expression_tag and not value and key in _VALID_EXPRESSIONS:
            expression = key
            found_expression_tag = True
            consumed = True

        if not consumed:
            break
        sentence = sentence[match.end():].strip()

    # 3. Strip ALL remaining bracket groups (stage directions, bad tags, timestamps),
    # then collapse any double spaces left behind by the asterisk removal.
    clean = _ANY_BRACKET_RE.sub("", sentence).strip()
    clean = re.sub(r'\s{2,}', ' ', clean).strip()

    # Edge case: if stripping left nothing, the whole sentence was tag/
    # bracket/asterisk noise — any captured actions are still returned.
    return clean, expression, actions, attitude, intensity


async def _ollama_streamer(
    messages: list[dict],
    synth_q: asyncio.Queue,
    out_text: list[str],
    t_cmd_start: float,
) -> None:
    """
    Streams tokens from Ollama natively async (httpx.AsyncClient +
    aiter_lines) instead of collecting the full response inside a
    blocking executor call first. This is what actually lets phrase-level
    TTS start while Ollama is still generating, rather than only after
    the whole reply has arrived — see [TIMING] logs below.
    """
    url = f"{config.llm.base_url.rstrip('/')}/api/chat"
    payload = {
        "model":    config.llm.model,
        "messages": messages,
        "stream":   True,
        # Every chat request must carry this — omitting it resets the
        # model's expiry to Ollama's 5-minute default (see ollama_lifecycle).
        "keep_alive": chat_keep_alive(),
        "options": {
            k: v for k, v in {
                "temperature": config.llm.temperature,
                "num_predict": config.llm.max_tokens,
                "num_gpu": getattr(config.llm, "num_gpu", None),
                "num_ctx": getattr(config.llm, "num_ctx", None),
            }.items() if v is not None
        },
    }

    full_text = ""
    buffer    = ""

    # Holds the expression from a standalone [tag] line so it carries forward
    # to the next sentence when Ollama emits tag and text separately.
    _pending_expression = "neutral"

    # Same idea for action animations from a stage-direction-only fragment
    # (e.g. Ollama emits "*giggles*" as its own chunk before the sentence),
    # and for the optional attitude/intensity nuance tags.
    _pending_actions: list[str] = []
    _pending_attitude: str | None = None
    _pending_intensity: str | None = None

    # True after a non-final phrase of the current sentence was emitted —
    # the next emitted phrase is then a continuation, not a bare sentence.
    _in_sentence = False

    # Every resolved sentence expression in this reply, gathered here and
    # reported to mood_manager ONCE at the end — a sentence's delivery tone
    # (e.g. a factual [neutral] line inside an angry reply) must not be
    # mistaken for Maya having calmed down mid-turn.
    turn_expressions: list[str] = []

    def _emit(raw: str, is_final: bool):
        """
        Parse a raw phrase/sentence fragment from the buffer.
        Returns (clean_text, expression, actions, is_final, attitude,
        intensity, continuation) or None if nothing to speak. Uses and
        updates the _pending_* closures.
        """
        nonlocal _pending_expression, _pending_attitude, _pending_intensity, _in_sentence
        raw = raw.strip()
        if not raw:
            return None

        clean, expression, actions, attitude, intensity = _parse_expression(raw)

        # Strip stray empty-bracket artifacts (e.g. a bare "[]") that
        # _parse_expression's bracket regex doesn't catch on its own — it
        # requires at least one character inside the brackets. Does not
        # touch any other bracket content or otherwise alter real text.
        clean = _EMPTY_BRACKET_RE.sub("", clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        clean = _normalize_pathological_dots(clean)

        if not clean or _PUNCT_ONLY_RE.match(clean):
            # Tag/action-only fragment, or nothing speakable left after
            # cleanup — carry forward to the next phrase instead of
            # enqueuing a wasted synthesis call.
            _pending_expression = expression
            _pending_actions.extend(actions)
            if attitude is not None:
                _pending_attitude = attitude
            if intensity is not None:
                _pending_intensity = intensity
            return None

        # Protection: if not final and the only words are vocatives, carry forward
        spoken_list = _get_spoken_words(clean)
        if not is_final and spoken_list and all(w in _VOCATIVE_WORDS for w in spoken_list):
            _pending_expression = expression
            _pending_actions.extend(actions)
            if attitude is not None:
                _pending_attitude = attitude
            if intensity is not None:
                _pending_intensity = intensity
            return None

        # If this fragment had its own tag, use it.
        # If not, carry forward whatever was pending from a prior standalone tag
        # or an earlier phrase chunk of this same (not-yet-finished) sentence.
        if expression != "neutral":
            resolved = expression
        else:
            resolved = _pending_expression

        resolved_attitude  = attitude  if attitude  is not None else _pending_attitude
        resolved_intensity = intensity if intensity is not None else _pending_intensity

        # Only clear the carried values once the sentence actually ends —
        # a mid-sentence phrase chunk (is_final=False) hands them on to the
        # next phrase chunk of the same sentence instead of resetting.
        _pending_expression = "neutral" if is_final else resolved
        _pending_attitude    = None if is_final else resolved_attitude
        _pending_intensity   = None if is_final else resolved_intensity

        continuation = _in_sentence
        _in_sentence = not is_final

        combined_actions = _pending_actions + actions
        _pending_actions.clear()

        logger.info(
            f"Parsed {'sentence' if is_final else 'phrase'} [{resolved}]"
            + (f" attitude={resolved_attitude}" if resolved_attitude else "")
            + (f" intensity={resolved_intensity}" if resolved_intensity else "")
            + f": '{clean}'"
            + (f"  actions={combined_actions}" if combined_actions else "")
        )

        turn_expressions.append(resolved)

        return clean, resolved, combined_actions, is_final, resolved_attitude, resolved_intensity, continuation

    # Monotonic phrase sequence counter for the current streaming turn
    _phrase_seq = [0]

    async def _put_phrase(result, reason: str = "default") -> None:
        _phrase_seq[0] += 1
        n = _phrase_seq[0]
        ts = time.perf_counter()
        # Pack result with phrase_id and creation timestamp
        await synth_q.put((*result, n, ts))
        clean_text = result[0]
        is_final = result[3]
        words = len(clean_text.split())
        chars = len(clean_text)
        logger.info(
            f"[TIMING][TTS][phrase={n}] QUEUED is_final={is_final} qsize={synth_q.qsize()} "
            f"t={ts:.3f} '{clean_text}'"
        )
        logger.info(
            f"[TTS_CHUNK] id={n} words={words} chars={chars} boundary={'sentence' if is_final else 'phrase'} "
            f"reason={reason} text='{clean_text}'"
        )
        logger.info(
            f"[TTS_SEGMENT] reason={reason} words={words}"
        )
        if n == 1:
            logger.info(f"[TIMING][TTFA] first usable phrase ready: +{ts-t_cmd_start:.3f}s since command start")

    async def _handle_token(token: str) -> None:
        nonlocal full_text, buffer
        full_text += token
        buffer    += token
        while True:
            boundary = _next_boundary(buffer, is_first_phrase=(_phrase_seq[0] == 0))
            if boundary is None:
                break
            idx, is_final = boundary
            reason = getattr(boundary, "reason", "default")
            raw    = buffer[:idx]
            buffer = buffer[idx:]
            result = _emit(raw, is_final)
            if result:
                await _put_phrase(result, reason=reason)

    t0 = time.perf_counter()
    first_token_logged = False
    turn_id = get_diag_turn_id()
    d_tag = f"[DIAG][TURN={turn_id}] "
    logger.info(f"{d_tag}OLLAMA_REQUEST_START model={config.llm.model} url={url} t={t0:.3f}")
    _log_cuda_memory("OLLAMA_REQUEST_START", turn_id=turn_id)

    client = get_ollama_client()
    resp = None
    stream_cancelled = False
    _ollama_generating.set()
    try:
        async with client.stream("POST", url, json=payload) as response:
            resp = response
            resp.raise_for_status()
            t_hdr = time.perf_counter()
            logger.info(f"{d_tag}OLLAMA_HEADERS_RECEIVED status={resp.status_code} in {t_hdr-t0:.3f}s")
            logger.info(f"[TIMING]   /api/chat headers received: {t_hdr-t0:.3f}s")
            _log_cuda_memory("OLLAMA_HEADERS_RECEIVED", turn_id=turn_id)

            async for line in resp.aiter_lines():
                if not line:
                    continue
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                done  = data.get("done", False)

                if not first_token_logged and token:
                    t_tok = time.perf_counter()
                    logger.info(f"{d_tag}OLLAMA_FIRST_TOKEN received in {t_tok-t0:.3f}s")
                    logger.info(f"[TIMING]   first token received: {t_tok-t0:.3f}s")
                    _log_cuda_memory("OLLAMA_FIRST_TOKEN", turn_id=turn_id)
                    first_token_logged = True

                if token:
                    await _handle_token(token)

                if done:
                    _ollama_generating.clear()
                    t_done = time.perf_counter()
                    logger.info(f"{d_tag}OLLAMA_STREAM_END in {t_done-t0:.3f}s")
                    logger.info(f"[TIMING]   stream fully done: {t_done-t0:.3f}s")
                    logger.info(
                        "Ollama metrics: total=%.2fs load=%.2fs prompt_eval=%.2fs eval=%.2fs "
                        "prompt_tokens=%s generated_tokens=%s",
                        data.get("total_duration", 0) / 1e9,
                        data.get("load_duration", 0) / 1e9,
                        data.get("prompt_eval_duration", 0) / 1e9,
                        data.get("eval_duration", 0) / 1e9,
                        data.get("prompt_eval_count"),
                        data.get("eval_count"),
                    )
                    log_chat_turn(data)
                    break
    except asyncio.CancelledError:
        stream_cancelled = True
        logger.warning(f"{d_tag}OLLAMA_STREAM_CANCELLED (CancelledError caught in streamer)")
        raise
    except Exception as e:
        logger.error(f"{d_tag}OLLAMA_REQUEST_EXCEPTION: {e}", exc_info=True)
        raise
    finally:
        _ollama_generating.clear()
        t_dur = time.perf_counter() - t0
        logger.info(f"{d_tag}OLLAMA_REQUEST_DURATION: {t_dur:.3f}s (cancelled={stream_cancelled})")
        if resp is not None:
            logger.info(f"{d_tag}OLLAMA_RESPONSE_CLOSED: resp.is_closed={resp.is_closed}")

    # Log raw Ollama output before any parsing so we can debug tag issues
    logger.debug(f"Ollama raw output: {repr(full_text)}")

    # Flush remaining buffer
    if buffer.strip():
        result = _emit(buffer, True)
        if result:
            await _put_phrase(result, reason="flush_buffer")

    # Store clean version (tags + stage directions stripped) for conversation memory
    clean_full = _HYBRID_TAG_RE_1.sub(r'*\1*', full_text)
    clean_full = _HYBRID_TAG_RE_2.sub(r'*\1*', clean_full)
    clean_full = _MARKDOWN_BOLD_RE.sub(r'\1', clean_full)
    clean_full = _ANY_BRACKET_RE.sub("", clean_full).strip()
    clean_full = _ASTERISK_RE.sub(lambda m: "" if _resolve_action(m.group(1).strip()) else m.group(1).strip(), clean_full)
    clean_full = re.sub(r'\s{2,}', ' ', clean_full).strip()
    out_text.append(clean_full)

    # Update Maya's persistent mood ONCE for this whole reply — not per
    # sentence — so a factual/neutral line inside an angry reply doesn't
    # get misread as her having cooled off.
    mood_manager.observe_turn(turn_expressions)

    await synth_q.put(_DONE)


# ── Stage 2: sentence → synthesised audio ────────────────────────────────────

async def _synth_worker(
    synth_q: asyncio.Queue,
    play_q: asyncio.Queue,
    t_cmd_start: float,
    cancel_streamer: Callable[[], None] | None = None,
) -> None:
    loop = asyncio.get_running_loop()

    # Diagnostic only — confirms the worker is already parked on synth_q.get()
    # before the first phrase is ever put, so a scheduling delay (category A)
    # can be told apart from a delay after dequeue (category B/C).
    logger.info(f"[TIMING] synth_worker ready, awaiting first item t={time.perf_counter():.3f}")

    first_dispatch_logged = False
    first_ready_logged    = False

    completed_phrases: dict[int, tuple] = {}
    next_play_id: int = 1
    pending_tasks: set[asyncio.Task] = set()
    flush_lock = asyncio.Lock()
    synth_failed = [False]

    async def _flush_ordered() -> None:
        nonlocal next_play_id
        async with flush_lock:
            while next_play_id in completed_phrases:
                payload = completed_phrases.pop(next_play_id)
                if payload is not None:
                    await play_q.put(payload)
                next_play_id += 1

    async def _do_phrase_synth(phrase_item, p_id: int | None, p_created: float) -> None:
        nonlocal first_dispatch_logged, first_ready_logged
        sentence, expression, actions, is_final, attitude, intensity, continuation, *meta = phrase_item
        p_tag = f"[TTS][phrase={p_id}] " if p_id is not None else "[TTS] "

        try:
            # Enhance prosody before synthesis so Kokoro renders with feeling
            t_pros0  = time.perf_counter()
            enhanced = _enhance_prosody(sentence, expression, is_final, continuation)
            t_pros1  = time.perf_counter()
            logger.info(f"[TIMING]{p_tag}_enhance_prosody '{sentence[:30]}': {t_pros1 - t_pros0:.4f}s")

            t0 = time.perf_counter()
            logger.info(f"[TIMING]{p_tag}Kokoro synth DISPATCH '{sentence[:30]}' t={t0:.3f}")
            if not first_dispatch_logged:
                logger.info(f"[TIMING][TTFA] Kokoro dispatch: +{t0-t_cmd_start:.3f}s since command start")
                first_dispatch_logged = True

            audio = await _run_kokoro(_synthesise_blocking, enhanced, expression, phrase_id=p_id)
            t1 = time.perf_counter()
            dur = t1 - t0

            if audio is not None:
                logger.info(f"[TIMING]{p_tag}Kokoro synth DONE '{sentence[:30]}': {dur:.3f}s (dispatch+compute) t={t1:.3f}")
                if not first_ready_logged:
                    logger.info(f"[TIMING][TTFA] first Kokoro audio ready: +{t1-t_cmd_start:.3f}s since command start")
                    first_ready_logged = True
                completed_phrases[p_id if p_id is not None else next_play_id] = (
                    audio, expression, actions, attitude, intensity, sentence, p_id, p_created
                )
                await _flush_ordered()
            else:
                logger.error(
                    f"[TIMING]{p_tag}FAILED after {dur:.3f}s (synthesis returned None / timeout) — "
                    "initiating turn stall containment."
                )
                synth_failed[0] = True
                # Fatal synthesis failure: cancel streamer so Ollama 60s HTTP socket doesn't hang the turn!
                if cancel_streamer:
                    logger.warning(f"[TIMING]{p_tag}Cancelling Ollama streamer to unblock conversational turn.")
                    cancel_streamer()
                await _flush_ordered()
                await play_q.put(_DONE)
                await synth_q.put(_DONE)
        except Exception as e:
            logger.error(f"{p_tag}Synth error for '{sentence}': {e}", exc_info=True)
            synth_failed[0] = True
            await synth_q.put(_DONE)

    try:
        while True:
            t_wait_start = time.perf_counter()
            item = await synth_q.get()
            t_dequeued = time.perf_counter()

            if item is _DONE or synth_failed[0]:
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                await _flush_ordered()
                await play_q.put(_DONE)
                return

            sentence, expression, actions, is_final, attitude, intensity, continuation, *meta = item
            phrase_id = meta[0] if meta else None
            t_phrase_created = meta[1] if len(meta) > 1 else t_dequeued
            p_tag = f"[TTS][phrase={phrase_id}] " if phrase_id is not None else "[TTS] "
            queue_wait = t_dequeued - t_phrase_created

            logger.info(
                f"[TIMING]{p_tag}DEQUEUED '{sentence[:30]}' t={t_dequeued:.3f} (queue_wait={queue_wait:.3f}s)"
            )

            task = asyncio.create_task(_do_phrase_synth(item, phrase_id, t_phrase_created))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

            # Bound concurrent lookahead to at most 2 in-flight synthesis tasks (1 CUDA + 1 CPU)
            while len(pending_tasks) >= 2 and not synth_failed[0]:
                await asyncio.sleep(0.01)
    finally:
        for t in list(pending_tasks):
            if not t.done():
                t.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)


def _resolve_tts_device() -> str:
    """
    Explicit override via config.tts.device ("cuda"/"cpu"), else
    auto-detect. Uses getattr() with a default so this works whether or
    not the deployed TTSConfig dataclass defines a `device` field yet —
    no settings.py change required for auto-detection to take effect.
    """
    configured = getattr(config.tts, "device", "auto")
    if configured and configured != "auto":
        return configured
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_kokoro_pipeline(lang: str) -> KPipeline:
    """
    Construct a KPipeline pinned to the resolved TTS device when the
    installed kokoro version's KPipeline accepts a `device` kwarg,
    falling back to the library's own default otherwise (older kokoro
    without that parameter) rather than raising.
    """
    device = _resolve_tts_device()
    repo_id = getattr(config.tts, "repo_id", "hexgrad/Kokoro-82M")
    sig = inspect.signature(KPipeline.__init__).parameters
    kwargs = {}
    if "device" in sig:
        kwargs["device"] = device
    if "repo_id" in sig:
        kwargs["repo_id"] = repo_id
    pipeline = KPipeline(lang_code=lang, **kwargs)
    _log_kokoro_device(pipeline, requested=device)
    return pipeline


def _log_cuda_memory(label: str, turn_id: int | None = None, phrase_id: int | None = None) -> None:
    """Diagnostic only — Kokoro's own share of VRAM (Ollama's own models
    are separate processes and aren't visible via torch here; see
    brain/embeddings.py's describe_ollama_models() for those)."""
    t_id = turn_id if turn_id is not None else get_diag_turn_id()
    tag = f"[DIAG][TURN={t_id}]"
    if phrase_id is not None:
        tag += f"[TTS][phrase={phrase_id}]"
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            logger.info(f"{tag} CUDA memory [{label}]: allocated={alloc:.1f}MB reserved={reserved:.1f}MB")
    except Exception:
        pass


def _log_kokoro_device(pipeline: KPipeline, requested: str) -> None:
    """Diagnostic only — never raises, never changes behavior."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    actual = "unknown"
    try:
        actual = str(next(pipeline.model.parameters()).device)
    except Exception:
        pass
    logger.info(
        f"[TIMING] Kokoro TTS device — requested='{requested}' "
        f"cuda_available={cuda_available} actual='{actual}'"
    )
    _log_cuda_memory("Kokoro init, before any synthesis", turn_id=0)


_kokoro_lock = threading.RLock()
_shared_pipeline: KPipeline | None = None
_shared_voice = None
_shared_lang: str | None = None


def get_shared_kokoro(lang: str | None = None):
    """
    Thread-safe lazy singleton provider for the unified Kokoro KPipeline and voice blend.
    Ensures that services.llm.llm_service, core.speaker.Speaker, and background workers
    share a single KModel/KPipeline instance in memory, eliminating redundant VRAM allocations.
    """
    global _shared_pipeline, _shared_voice, _shared_lang
    if lang is None:
        lang = getattr(config.tts, "lang_code", "a")

    with _kokoro_lock:
        if _shared_pipeline is None or _shared_lang != lang:
            _shared_pipeline = _build_kokoro_pipeline(lang)
            _shared_lang = lang
            primary = config.tts.voice
            blend   = getattr(config.tts, "voice_blend", "")
            ratio   = getattr(config.tts, "blend_ratio", 0.0)
            if blend and 0.0 < ratio < 1.0:
                try:
                    v1 = _shared_pipeline.load_voice(primary)
                    v2 = _shared_pipeline.load_voice(blend)
                    _shared_voice = (1.0 - ratio) * v1 + ratio * v2
                    logger.info(
                        f"Kokoro voice blend loaded: {primary} ({1-ratio:.0%}) + "
                        f"{blend} ({ratio:.0%})"
                    )
                except Exception as e:
                    logger.warning(f"Voice blend failed ({e}), falling back to {primary}")
                    _shared_voice = primary
            else:
                _shared_voice = primary
        return _shared_pipeline, _shared_voice


_kokoro_is_warm = False
_kokoro_warmup_lock = threading.Lock()
_kokoro_warm_event = threading.Event()


def is_kokoro_warm() -> bool:
    """Return whether the shared Kokoro pipeline has completed eager warmup."""
    return _kokoro_is_warm


def ensure_kokoro_warmed(phrase: str = "Hi.") -> bool:
    """
    Eagerly initialize the shared Kokoro pipeline and run a minimal silent
    synthesis to prime CUDA kernels, activations, and cuBLAS/cuDNN workspaces.
    Idempotent and thread-safe: executes exactly once across the application lifecycle.
    """
    global _kokoro_is_warm
    if _kokoro_is_warm:
        return True

    with _kokoro_warmup_lock:
        if _kokoro_is_warm:
            return True
        logger.info(f"Eagerly warming shared Kokoro TTS pipeline with '{phrase}'…")
        t0 = time.perf_counter()
        try:
            with _kokoro_lock:
                pipeline, voice = get_shared_kokoro()
                for _, _, _ in pipeline(phrase, voice=voice, speed=1.0):
                    break
            _kokoro_is_warm = True
            _kokoro_warm_event.set()
            elapsed = time.perf_counter() - t0
            logger.info(f"Kokoro TTS eager warmup complete in {elapsed:.2f}s — CUDA kernels primed.")
            _log_cuda_memory("NORMAL_STARTUP_AFTER_WARMUP", turn_id=0)
            return True
        except Exception as e:
            logger.warning(f"Kokoro eager warmup failed (non-fatal): {e}", exc_info=True)
            return False


class WorkerState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    TIMED_OUT = "TIMED_OUT"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


_cuda_worker_state: WorkerState = WorkerState.IDLE
_kokoro_worker_active = threading.Event()
_kokoro_stale_worker = threading.Event()
_kokoro_worker_seq = 0
_active_phrase_id: int | None = None

# CUDA Kokoro FIFO queue state (Batch 3)
_cuda_synth_lock: asyncio.Lock | None = None
_cuda_synth_queue_depth: int = 0
_cuda_synth_active: bool = False


def _get_cuda_synth_lock() -> asyncio.Lock:
    """Return the asyncio.Lock ensuring strictly serialized FIFO CUDA synthesis."""
    global _cuda_synth_lock
    if _cuda_synth_lock is None:
        _cuda_synth_lock = asyncio.Lock()
    return _cuda_synth_lock


def get_cuda_synth_queue_depth() -> int:
    """Return current depth of queued/active CUDA synthesis requests."""
    return _cuda_synth_queue_depth


def clear_cuda_synth_queue_state() -> None:
    """Reset CUDA synthesis queue depth and active state (primarily for test fixtures)."""
    global _cuda_synth_queue_depth, _cuda_synth_active, _cuda_synth_lock
    _cuda_synth_queue_depth = 0
    _cuda_synth_active = False
    _cuda_synth_lock = None


def get_worker_state() -> WorkerState:
    """Return the current lifecycle state of the CUDA Kokoro worker."""
    return _cuda_worker_state


def is_kokoro_busy() -> bool:
    """Return whether a Kokoro synthesis worker is currently running or actively executing."""
    return _kokoro_worker_active.is_set() or _cuda_synth_active


def is_kokoro_stale() -> bool:
    """Return whether a timed-out Kokoro worker is still executing in background or quarantined."""
    return _kokoro_stale_worker.is_set() or _cuda_worker_state == WorkerState.TIMED_OUT


# ── CPU Kokoro Fallback Subsystem ──────────────────────────────────────────

_cpu_kokoro_lock = threading.RLock()
_cpu_shared_pipeline: KPipeline | None = None
_cpu_shared_voice = None
_cpu_shared_lang: str | None = None


def get_cpu_kokoro(lang: str | None = None):
    """
    Thread-safe lazy singleton provider for the isolated CPU Kokoro KPipeline.
    Used when CUDA Kokoro is quarantined due to a soft timeout, ensuring that
    Maya continues speaking without blocking on CUDA or _kokoro_lock.
    """
    global _cpu_shared_pipeline, _cpu_shared_voice, _cpu_shared_lang
    if lang is None:
        lang = getattr(config.tts, "lang_code", "a")

    with _cpu_kokoro_lock:
        if _cpu_shared_pipeline is None or _cpu_shared_lang != lang:
            repo_id = getattr(config.tts, "repo_id", "hexgrad/Kokoro-82M")
            logger.info(f"[TTS] Initializing isolated CPU Kokoro pipeline (lang='{lang}', repo_id='{repo_id}')…")
            _cpu_shared_pipeline = KPipeline(lang_code=lang, device="cpu", repo_id=repo_id)
            _cpu_shared_lang = lang
            primary = config.tts.voice
            blend   = getattr(config.tts, "voice_blend", "")
            ratio   = getattr(config.tts, "blend_ratio", 0.0)
            if blend and 0.0 < ratio < 1.0:
                try:
                    v1 = _cpu_shared_pipeline.load_voice(primary)
                    v2 = _cpu_shared_pipeline.load_voice(blend)
                    _cpu_shared_voice = (1.0 - ratio) * v1 + ratio * v2
                except Exception as e:
                    logger.warning(f"CPU voice blend failed ({e}), falling back to {primary}")
                    _cpu_shared_voice = primary
            else:
                _cpu_shared_voice = primary
        return _cpu_shared_pipeline, _cpu_shared_voice


def _synthesise_cpu_blocking(
    sentence: str,
    expression: str = "neutral",
    trace_id: str | None = None,
    phrase_id: int | None = None,
) -> tuple | None:
    """
    Synthesise `sentence` with CPU Kokoro. Completely decoupled from CUDA,
    _kokoro_lock, and VRAM memory.
    """
    p_tag = f"[TTS][phrase={phrase_id}] " if phrase_id is not None else "[TTS] "
    try:
        pipeline, voice = get_cpu_kokoro()
        base_speed  = getattr(config.tts, "speed", 1.0)
        expr_factor = EXPRESSION_SPEED.get(expression, 1.0)
        speed       = round(base_speed * expr_factor, 3)
        logger.info(f"{p_tag}Synthesising on CPU fallback [{expression}] speed={speed}: '{sentence[:60]}'")
        # Bound CPU threads so PortAudio microphone callback is never starved
        try:
            import torch
            if torch.get_num_threads() > 4:
                torch.set_num_threads(4)
        except Exception:
            pass

        t0 = time.perf_counter()
        with _cpu_kokoro_lock:
            chunks = [audio for _, _, audio in pipeline(sentence, voice=voice, speed=speed)
                      if audio is not None and len(audio) > 0]
        dt = time.perf_counter() - t0

        if not chunks:
            logger.warning(f"{p_tag}CPU Kokoro: no audio for '{sentence}'")
            return None
        pcm = np.concatenate(chunks).astype(np.float32)
        logger.info(f"{p_tag}CPU Kokoro synthesis complete in {dt:.3f}s (samples={len(pcm)})")
        return (pcm, 24_000)
    except Exception as e:
        logger.error(f"{p_tag}CPU Kokoro synth error: {e}", exc_info=True)
        return None


def ensure_cpu_kokoro_warmed(lang: str | None = None) -> None:
    """Prime the isolated CPU Kokoro pipeline during startup to eliminate cold-start fallback latency."""
    try:
        t0 = time.perf_counter()
        pipeline, voice = get_cpu_kokoro(lang)
        with _cpu_kokoro_lock:
            for _ in pipeline("Maya", voice=voice, speed=1.0):
                pass
        dt = time.perf_counter() - t0
        logger.info(f"[TTS] Isolated CPU Kokoro pipeline successfully primed and resident in {dt:.3f}s.")
    except Exception as e:
        logger.warning(f"[TTS] CPU Kokoro pipeline warmup failed (non-fatal): {e}")


def _do_reset_pipeline() -> None:
    """Internal helper to atomically clear the shared pipeline while holding _kokoro_lock."""
    global _shared_pipeline, _shared_voice, _shared_lang, _kokoro_is_warm
    with _kokoro_lock:
        _shared_pipeline = None
        _shared_voice = None
        _shared_lang = None
        _kokoro_is_warm = False
        _kokoro_warm_event.clear()
    logger.warning("Shared Kokoro pipeline reset — will rebuild on next call.")


def reset_shared_kokoro() -> None:
    """
    Invalidates the shared Kokoro pipeline following a synthesis timeout or fatal failure,
    allowing the next synthesis call to construct a fresh backend instance.
    Non-blocking with respect to asyncio event loops: if a worker is currently active,
    marks reset as deferred rather than freezing the caller on _kokoro_lock.
    """
    if _kokoro_worker_active.is_set():
        _kokoro_stale_worker.set()
        logger.warning("Shared Kokoro pipeline reset deferred — worker still active on GPU.")
        return
    _do_reset_pipeline()


class GPUSchedulerState(Enum):
    GPU_TTS_ALLOWED = "GPU_TTS_ALLOWED"
    GPU_TTS_DEFERRED = "GPU_TTS_DEFERRED"
    GPU_TTS_CPU_FALLBACK = "GPU_TTS_CPU_FALLBACK"
    GPU_TTS_QUARANTINED = "GPU_TTS_QUARANTINED"


class GPUSchedulingPolicy(Enum):
    POLICY_0_BASELINE = "baseline"
    POLICY_1_PHRASE_GATED = "phrase_gated"
    POLICY_2_FIRST_PHRASE_PRIORITY = "first_phrase_priority"
    POLICY_3_ADAPTIVE_GATE = "adaptive_gate"
    POLICY_4_HEAVY_GENERATION_DEFERRAL = "heavy_generation_deferral"
    POLICY_5_FULL_SERIALIZATION = "full_serialization"


_active_scheduling_policy: GPUSchedulingPolicy = GPUSchedulingPolicy.POLICY_3_ADAPTIVE_GATE
_ollama_generating = threading.Event()


def get_active_scheduling_policy() -> GPUSchedulingPolicy:
    """Return the currently active GPU scheduling policy."""
    return _active_scheduling_policy


KOKORO_CUDA_MIN_HEADROOM_MB: float = 350.0
KOKORO_CUDA_OLLAMA_CONTENTION_HEADROOM_MB: float = 1200.0
KOKORO_CUDA_WATCHDOG_TIMEOUT_SEC: float = 3.0
_mock_gpu_free_vram_mb: float | None = None


def get_gpu_vram_info() -> tuple[float | None, float | None]:
    """
    Return (free_mb, total_mb) for the active CUDA device, or (None, None).
    Supports _mock_gpu_free_vram_mb for hermetic test coverage.
    """
    global _mock_gpu_free_vram_mb
    if _mock_gpu_free_vram_mb is not None:
        return _mock_gpu_free_vram_mb, 4096.0
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return free_bytes / (1024 * 1024), total_bytes / (1024 * 1024)
    except Exception:
        pass
    return None, None


_cpu_fallback_lock: asyncio.Lock | None = None
_cpu_fallback_lock_loop: asyncio.AbstractEventLoop | None = None
_cpu_fallback_queue_depth: int = 0
_cpu_fallback_active: bool = False


def _get_cpu_fallback_lock() -> asyncio.Lock:
    global _cpu_fallback_lock, _cpu_fallback_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _cpu_fallback_lock is None or _cpu_fallback_lock_loop is not loop:
        _cpu_fallback_lock = asyncio.Lock()
        _cpu_fallback_lock_loop = loop
    return _cpu_fallback_lock


def set_active_scheduling_policy(policy: GPUSchedulingPolicy) -> None:
    """Set the active GPU scheduling policy."""
    global _active_scheduling_policy
    _active_scheduling_policy = policy


def is_ollama_generating() -> bool:
    """Return whether the Ollama token generation stream is currently active."""
    return _ollama_generating.is_set()


async def evaluate_tts_admission(phrase_id: int | None = None) -> tuple[GPUSchedulerState, str]:
    """
    Evaluates whether a phrase synthesis request is admitted to CUDA, deferred,
    routed to CPU fallback, or rejected by quarantine.

    Hierarchy:
    1. Quarantine Precedence: If CUDA worker is TIMED_OUT or stale, ALWAYS return GPU_TTS_QUARANTINED.
    2. Device Capability: If TTS device is not CUDA or CUDA unavailable, return GPU_TTS_CPU_FALLBACK.
    3. Resource Admission: Under Policy 3 (Adaptive Gate):
       - If free VRAM < KOKORO_CUDA_MIN_HEADROOM_MB (350 MB), route to CPU fallback.
       - If a CUDA Kokoro worker is actively running, perform a bounded wait (up to 0.4s)
         so subsequent phrases can take the fast CUDA path without thread collisions.
       - If worker is still busy after bounded wait, safely route to CPU fallback.
       - If VRAM headroom is safe (>= 350 MB), admit to CUDA (regardless of Ollama generating).
    """
    global _active_scheduling_policy
    p_num = phrase_id if phrase_id is not None else 1

    # Tier 1: F.1 Quarantine Precedence (Absolute Safety)
    if is_kokoro_stale() or _cuda_worker_state == WorkerState.TIMED_OUT or _kokoro_stale_worker.is_set():
        return (GPUSchedulerState.GPU_TTS_QUARANTINED, "cuda_quarantined")

    if _resolve_tts_device() != "cuda":
        return (GPUSchedulerState.GPU_TTS_CPU_FALLBACK, "device_cpu")

    policy = _active_scheduling_policy

    # Policy 0: Baseline
    if policy == GPUSchedulingPolicy.POLICY_0_BASELINE:
        return (GPUSchedulerState.GPU_TTS_ALLOWED, "baseline_unconstrained")

    # Policy 1: Phrase-Gated
    elif policy == GPUSchedulingPolicy.POLICY_1_PHRASE_GATED:
        if is_ollama_generating():
            await asyncio.sleep(0.05)
            if is_ollama_generating():
                return (GPUSchedulerState.GPU_TTS_CPU_FALLBACK, "ollama_active_phrase_gated")
        return (GPUSchedulerState.GPU_TTS_ALLOWED, "phrase_gated_admitted")

    # Policy 2: First-Phrase Priority
    elif policy == GPUSchedulingPolicy.POLICY_2_FIRST_PHRASE_PRIORITY:
        if p_num == 1:
            return (GPUSchedulerState.GPU_TTS_ALLOWED, "first_phrase_priority_admitted")
        if is_ollama_generating():
            return (GPUSchedulerState.GPU_TTS_CPU_FALLBACK, "ollama_generating_phrase_conservative")
        return (GPUSchedulerState.GPU_TTS_ALLOWED, "ollama_idle_admitted")

    # Policy 3: Adaptive GPU Gate (VRAM-Aware & Concurrency-Safe)
    elif policy == GPUSchedulingPolicy.POLICY_3_ADAPTIVE_GATE:
        # Check initial VRAM headroom
        free_mb, _ = get_gpu_vram_info()
        if free_mb is not None and free_mb < KOKORO_CUDA_MIN_HEADROOM_MB:
            return (GPUSchedulerState.GPU_TTS_CPU_FALLBACK, f"vram_headroom_low_{free_mb:.0f}mb")

        # Proactive contention gate: When Ollama is actively generating on GPU,
        # constrained VRAM (< 1200MB) introduces severe compute starvation / WDDM queue stalls.
        # Safely route to CPU fallback before dispatching to CUDA.
        if is_ollama_generating() and free_mb is not None and free_mb < KOKORO_CUDA_OLLAMA_CONTENTION_HEADROOM_MB:
            return (GPUSchedulerState.GPU_TTS_CPU_FALLBACK, "ollama_gpu_contention_risk")

        # If a prior CUDA worker is actively running, admit to the serialized CUDA FIFO queue
        # rather than prematurely forcing CPU fallback. The queue guarantees strictly serialized
        # execution without thread collisions or unsafe concurrency.
        if is_kokoro_busy():
            return (GPUSchedulerState.GPU_TTS_ALLOWED, "adaptive_queue_cuda")

        # Worker is idle and VRAM headroom is safe
        if is_ollama_generating():
            if p_num == 1:
                return (GPUSchedulerState.GPU_TTS_ALLOWED, "adaptive_first_phrase_cuda")
            else:
                return (GPUSchedulerState.GPU_TTS_ALLOWED, "adaptive_headroom_safe")
        else:
            return (GPUSchedulerState.GPU_TTS_ALLOWED, "adaptive_ollama_idle_cuda")

    # Policy 4: Heavy-Generation Deferral (Bounded Delay)
    elif policy == GPUSchedulingPolicy.POLICY_4_HEAVY_GENERATION_DEFERRAL:
        if is_ollama_generating():
            t_wait_start = time.perf_counter()
            while is_ollama_generating() and (time.perf_counter() - t_wait_start < 0.15):
                await asyncio.sleep(0.02)
            if is_ollama_generating():
                return (GPUSchedulerState.GPU_TTS_CPU_FALLBACK, "deferral_budget_expired_cpu")
        return (GPUSchedulerState.GPU_TTS_ALLOWED, "deferral_cleared_cuda")

    # Policy 5: Full GPU Serialization
    elif policy == GPUSchedulingPolicy.POLICY_5_FULL_SERIALIZATION:
        if is_ollama_generating():
            t_wait_start = time.perf_counter()
            while is_ollama_generating():
                await asyncio.sleep(0.02)
                if time.perf_counter() - t_wait_start > 15.0:
                    return (GPUSchedulerState.GPU_TTS_CPU_FALLBACK, "serialization_timeout_cpu")
        return (GPUSchedulerState.GPU_TTS_ALLOWED, "full_serialization_cuda")

    return (GPUSchedulerState.GPU_TTS_ALLOWED, "default")


# Backward-compatible internal aliases
_get_kokoro = get_shared_kokoro
_reset_kokoro_pipeline = reset_shared_kokoro

_KOKORO_SYNTH_TIMEOUT = KOKORO_CUDA_WATCHDOG_TIMEOUT_SEC


def set_kokoro_synth_timeout(timeout: float) -> None:
    """Set the Kokoro synthesis watchdog timeout (seconds)."""
    global _KOKORO_SYNTH_TIMEOUT
    _KOKORO_SYNTH_TIMEOUT = timeout


async def _dispatch_cpu_fallback(args, phrase_id: int | None, diag_p_tag: str, p_tag: str):
    """Execute speech synthesis on the isolated, strictly-serialized CPU fallback pipeline."""
    global _cpu_fallback_queue_depth, _cpu_fallback_active
    loop = asyncio.get_running_loop()
    cpu_lock = _get_cpu_fallback_lock()

    _cpu_fallback_queue_depth += 1
    queue_pos = _cpu_fallback_queue_depth - 1
    logger.info(f"[CPU_TTS_QUEUE]{p_tag}position={queue_pos} active={_cpu_fallback_active}")

    lock_acquired = False
    try:
        await cpu_lock.acquire()
        lock_acquired = True
        _cpu_fallback_active = True
        logger.info(f"[CPU_TTS_QUEUE]{p_tag}position=0 active=True")
        logger.info(f"{diag_p_tag}CPU_FALLBACK_DISPATCH")
        try:
            sentence = args[0]
            expression = args[1] if len(args) > 1 else "neutral"
            trace_id = args[2] if len(args) > 2 else None
            t_cpu_start = time.perf_counter()
            res = await loop.run_in_executor(
                None, _synthesise_cpu_blocking, sentence, expression, trace_id, phrase_id
            )
            t_cpu_dur = time.perf_counter() - t_cpu_start
            logger.info(f"{diag_p_tag}CPU_FALLBACK_COMPLETE (success={res is not None}) compute={t_cpu_dur:.3f}s")
            return res
        except Exception as e:
            logger.error(f"{diag_p_tag}CPU fallback synthesis failed: {e}", exc_info=True)
            return None
    finally:
        if lock_acquired:
            cpu_lock.release()
        _cpu_fallback_queue_depth -= 1
        _cpu_fallback_active = (_cpu_fallback_queue_depth > 0)
        logger.info(f"[CPU_TTS_QUEUE]{p_tag}position=0 active={_cpu_fallback_active}")


async def _execute_cuda_worker(fn, args, phrase_id: int | None, diag_p_tag: str, p_tag: str, turn_id: int, on_timeout=None):
    """Execute a single CUDA Kokoro synthesis worker thread with soft watchdog timeout."""
    global _kokoro_worker_seq, _cuda_worker_state, _active_phrase_id
    fut = concurrent.futures.Future()
    _kokoro_worker_active.set()
    _kokoro_worker_seq += 1
    seq = _kokoro_worker_seq
    _active_phrase_id = phrase_id
    logger.info(f"{diag_p_tag}CUDA_WORKER_STATE_BEFORE_DISPATCH state={_cuda_worker_state.value}")
    _log_cuda_memory("BEFORE_KOKORO_CUDA_DISPATCH", turn_id=turn_id, phrase_id=phrase_id)
    _cuda_worker_state = WorkerState.RUNNING

    t_dispatch = time.perf_counter()
    logger.info(f"{p_tag}DISPATCH worker={seq} backend=cuda thread=kokoro-synth-{seq}")

    def _runner():
        nonlocal t_dispatch
        t_start = time.perf_counter()
        res = None
        exc = None
        t_exec = 0.0
        try:
            if _cuda_stall_injection_phrase is not None and (phrase_id == _cuda_stall_injection_phrase or _cuda_stall_injection_phrase == -1):
                logger.warning(
                    f"{diag_p_tag}[INJECTED_FAULT] Stalling CUDA worker #{seq} for {_cuda_stall_injection_duration}s..."
                )
                time.sleep(_cuda_stall_injection_duration)
            res = fn(*args)
            t_exec = time.perf_counter() - t_start
        except BaseException as e:
            exc = e
            t_exec = time.perf_counter() - t_start
        finally:
            _kokoro_worker_active.clear()
            logger.info(f"{diag_p_tag}CUDA_WORKER_EXIT worker #{seq}")
            if _kokoro_stale_worker.is_set():
                global _cuda_worker_state
                _cuda_worker_state = WorkerState.RECOVERING
                t_total = time.perf_counter() - t_dispatch
                logger.info(
                    f"{p_tag}Stale Kokoro worker #{seq} finally exited after {t_total:.3f}s — "
                    "executing deferred pipeline reset off event loop."
                )
                try:
                    _do_reset_pipeline()
                finally:
                    _kokoro_stale_worker.clear()
                    _cuda_worker_state = WorkerState.IDLE
                    logger.info(f"{diag_p_tag}CUDA_WORKER_RECOVERED — CUDA backend restored to IDLE.")
                    _log_cuda_memory("QUARANTINED_WORKER_EXITED", turn_id=turn_id, phrase_id=phrase_id)
            else:
                _cuda_worker_state = WorkerState.IDLE

            if exc is not None:
                if not fut.cancelled():
                    fut.set_exception(exc)
                else:
                    logger.warning(f"{p_tag}Exception from timed-out worker #{seq}: {exc}")
            else:
                if not fut.cancelled():
                    fut.set_result(res)
                else:
                    logger.warning(
                        f"{p_tag}Late completion from timed-out worker #{seq} after {t_exec:.3f}s (result discarded)"
                    )

    threading.Thread(target=_runner, daemon=True, name=f"kokoro-synth-{seq}").start()

    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=_KOKORO_SYNTH_TIMEOUT)
    except asyncio.TimeoutError:
        _cuda_worker_state = WorkerState.TIMED_OUT
        _kokoro_stale_worker.set()
        logger.error(
            f"{diag_p_tag}CUDA_WORKER_TIMEOUT after {_KOKORO_SYNTH_TIMEOUT}s "
            f"(worker #{seq} still executing in background)"
        )
        logger.warning(f"{diag_p_tag}CUDA_WORKER_QUARANTINED")
        logger.warning(f"{diag_p_tag}CUDA_WORKER_THREAD_STILL_RUNNING (worker #{seq})")
        _log_cuda_memory("AFTER_CUDA_TIMEOUT", turn_id=turn_id, phrase_id=phrase_id)
        if on_timeout:
            try:
                on_timeout()
            except Exception as e:
                logger.warning(f"{p_tag}Kokoro on_timeout callback failed (non-fatal): {e}")
        return None


async def _run_kokoro(fn, *args, phrase_id: int | None = None, on_timeout=None):
    """
    Runs Kokoro synthesis with bounded soft timeout, adaptive GPU scheduling,
    FIFO CUDA queue serialization, and automatic serialized CPU fallback.

    Fault containment & recovery contract (Phase F.1, F.2 & Batch 3):
    1. Quarantine precedence: If CUDA is quarantined (TIMED_OUT / stale), scheduler forces CPU fallback.
    2. Adaptive GPU admission: Safe VRAM (>= 350MB) admits to CUDA fast path; worker busy admits to CUDA FIFO queue.
    3. Serialized CUDA Queue: Concurrent or subsequent CUDA phrases queue in FIFO order on _cuda_synth_lock,
       completely eliminating unnecessary CPU fallback while preventing thread collisions or unsafe CUDA concurrency.
    4. Soft timeout: If CUDA synthesis exceeds _KOKORO_SYNTH_TIMEOUT, asyncio.wait_for
       terminates the caller wait and marks the worker as TIMED_OUT.
    5. Deferred CUDA recovery: When the background worker exits native execution, its
       finally block resets the CUDA pipeline and restores the backend to IDLE.
    """
    turn_id = get_diag_turn_id()
    diag_p_tag = f"[DIAG][TURN={turn_id}][TTS][phrase={phrase_id}] " if phrase_id is not None else f"[DIAG][TURN={turn_id}][TTS] "
    p_tag = f"[TTS][phrase={phrase_id}] " if phrase_id is not None else "[TTS] "

    t_adm_start = time.perf_counter()
    admission_state, reason = await evaluate_tts_admission(phrase_id=phrase_id)
    t_adm_wait = time.perf_counter() - t_adm_start

    backend_str = "cuda" if admission_state == GPUSchedulerState.GPU_TTS_ALLOWED else "cpu"
    free_mb, _ = get_gpu_vram_info()
    free_str = f"{free_mb:.1f}MB" if free_mb is not None else "n/a"
    logger.info(
        f"[TTS_ADMISSION]{p_tag}backend={backend_str} decision={admission_state.value} reason={reason} "
        f"gpu_free={free_str} ollama_active={is_ollama_generating()} kokoro_gpu_active={is_kokoro_busy()} "
        f"adm_wait={t_adm_wait:.4f}s"
    )

    # 1. If quarantined or scheduler decided CPU fallback:
    if admission_state in (GPUSchedulerState.GPU_TTS_QUARANTINED, GPUSchedulerState.GPU_TTS_CPU_FALLBACK):
        if fn == _synthesise_blocking or getattr(fn, "__name__", "") == "_synthesise_blocking":
            if admission_state == GPUSchedulerState.GPU_TTS_QUARANTINED:
                logger.warning(
                    f"{diag_p_tag}CPU_FALLBACK_SELECTED (worker_state={_cuda_worker_state.value}) — "
                    "routing speech synthesis to isolated CPU fallback pipeline."
                )
            else:
                logger.info(
                    f"{diag_p_tag}CPU_FALLBACK_SELECTED (scheduler_decision={admission_state.value}) — "
                    f"routing speech synthesis to CPU ({reason})."
                )
            return await _dispatch_cpu_fallback(args, phrase_id, diag_p_tag, p_tag)
        else:
            logger.warning(
                f"{p_tag}Kokoro synthesis rejected: prior worker still active on GPU after timeout."
            )
            return None

    # 2. CUDA is healthy: acquire single CUDA worker lock (FIFO Queue)
    global _cuda_synth_queue_depth, _cuda_synth_active
    cuda_lock = _get_cuda_synth_lock()

    _cuda_synth_queue_depth += 1
    queue_pos = _cuda_synth_queue_depth - 1
    t_queue_enter = time.perf_counter()
    logger.info(
        f"[CUDA_TTS_QUEUE]{p_tag}position={queue_pos} active={_cuda_synth_active}"
    )

    lock_acquired = False
    try:
        try:
            # Bounded wait for CUDA worker: up to 10.0s before considering fallback
            await asyncio.wait_for(cuda_lock.acquire(), timeout=10.0)
            lock_acquired = True
        except asyncio.TimeoutError:
            logger.warning(
                f"{diag_p_tag}CUDA queue wait timeout (>10.0s) — routing to CPU fallback."
            )
            return await _dispatch_cpu_fallback(args, phrase_id, diag_p_tag, p_tag)

        t_queue_wait = time.perf_counter() - t_queue_enter
        _cuda_synth_active = True
        logger.info(f"[CUDA_TTS_QUEUE]{p_tag}position=0 active=True queue_wait={t_queue_wait:.3f}s")

        # Check if quarantine occurred while waiting in the lock:
        if is_kokoro_stale() or _cuda_worker_state == WorkerState.TIMED_OUT or _kokoro_stale_worker.is_set():
            logger.warning(
                f"{diag_p_tag}CUDA quarantined while waiting in queue — routing speech synthesis to CPU fallback."
            )
            return await _dispatch_cpu_fallback(args, phrase_id, diag_p_tag, p_tag)

        # Check VRAM headroom again just before launching GPU execution:
        free_mb_exec, _ = get_gpu_vram_info()
        if free_mb_exec is not None and free_mb_exec < KOKORO_CUDA_MIN_HEADROOM_MB:
            logger.warning(
                f"{diag_p_tag}VRAM dropped below {KOKORO_CUDA_MIN_HEADROOM_MB}MB while queued ({free_mb_exec:.1f}MB) — "
                "routing to CPU fallback."
            )
            return await _dispatch_cpu_fallback(args, phrase_id, diag_p_tag, p_tag)

        if is_ollama_generating() and free_mb_exec is not None and free_mb_exec < KOKORO_CUDA_OLLAMA_CONTENTION_HEADROOM_MB:
            logger.warning(
                f"{diag_p_tag}VRAM under Ollama contention threshold ({free_mb_exec:.1f}MB < {KOKORO_CUDA_OLLAMA_CONTENTION_HEADROOM_MB}MB) "
                "while queued — routing to CPU fallback (ollama_gpu_contention_risk)."
            )
            return await _dispatch_cpu_fallback(args, phrase_id, diag_p_tag, p_tag)

        t_compute_start = time.perf_counter()
        res = await _execute_cuda_worker(fn, args, phrase_id, diag_p_tag, p_tag, turn_id, on_timeout)
        t_compute = time.perf_counter() - t_compute_start
        logger.info(f"[TIMING]{p_tag}Kokoro CUDA compute: {t_compute:.3f}s")
        return res
    finally:
        if lock_acquired:
            cuda_lock.release()
        _cuda_synth_queue_depth -= 1
        _cuda_synth_active = (_cuda_synth_queue_depth > 0)
        logger.info(f"[CUDA_TTS_QUEUE]{p_tag}position=0 active={_cuda_synth_active}")


def warmup() -> None:
    """
    Call once at startup (before any query) to load Kokoro into memory and prime CUDA kernels.
    Idempotent and thread-safe.
    """
    ensure_kokoro_warmed()


def _synthesise_blocking(sentence: str, expression: str = "neutral", trace_id: str | None = None) -> tuple | None:
    """
    Synthesise `sentence` with Kokoro.

    Speed is computed as:
        config.tts.speed  (user base)  *  EXPRESSION_SPEED[expression]

    This means excited speech runs ~13% faster than the user's baseline,
    sad speech ~16% slower, etc. — giving Kokoro the primary signal for
    emotional pacing since it has no native emotion mode.
    """
    try:
        from services.tracer import log_event
        if trace_id:
            log_event(trace_id, "tts_synthesis_start", {"text_len": len(sentence), "expression": expression})
    except Exception:
        pass

    try:
        pipeline, voice = _get_kokoro()
        base_speed  = getattr(config.tts, "speed", 1.0)
        expr_factor = EXPRESSION_SPEED.get(expression, 1.0)
        speed       = round(base_speed * expr_factor, 3)
        logger.debug(f"Synthesising [{expression}] speed={speed}: '{sentence[:60]}'")
        with _kokoro_lock:
            chunks = [audio for _, _, audio in pipeline(sentence, voice=voice, speed=speed)
                      if audio is not None and len(audio) > 0]

        try:
            from services.tracer import log_event
            if trace_id:
                log_event(trace_id, "tts_synthesis_end", {"chunks": len(chunks)})
        except Exception:
            pass

        if not chunks:
            logger.warning(f"Kokoro: no audio for '{sentence}'")
            return None
        pcm = np.concatenate(chunks).astype(np.float32)

        try:
            from services.tracer import log_event
            if trace_id:
                log_event(trace_id, "tts_audio_ready", {"samples": len(pcm), "sr": 24_000})
        except Exception:
            pass

        return (pcm, 24_000)
    except Exception as e:
        logger.error(f"Kokoro synth error: {e}", exc_info=True)
        return None


# ── Stage 3: play sentences with expression changes ──────────────────────────

async def _play_worker(play_q: asyncio.Queue, t_cmd_start: float) -> None:
    """
    For each sentence:
      1. On the very first sentence, wait for the filler to finish so the
         two audio streams don't collide on the audio_done event.
      2. Broadcast any action animations (giggle, nod, ...) extracted from
         *stage directions* in this sentence, so the gesture lands right
         as the line starts.
      3. Broadcast the expression (+ optional attitude/intensity nuance)
         so the avatar's face changes.
      4. Broadcast speaking state.
      5. Play the audio, wait for browser audio_done.
      6. Reset expression to Maya's current mood baseline after playback —
         not a hardcoded "neutral" — so an active mood (e.g. angry) persists
         between sentences instead of visibly resetting every line.
    """
    from core.state import state, MayaState
    from services.ws_server import ws_server as _ws

    loop           = asyncio.get_running_loop()
    output         = getattr(config.tts, "output", "avatar")
    first_sentence = True
    first_played   = False

    while True:
        item = await play_q.get()

        if item is _DONE:
            _reply_finished[0] = True
            await _ws.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))
            # Respects a concurrent "go to sleep" landed mid-reply — see
            # core/turn_lifecycle.py's rest() and docs/CHANGELOG.md's
            # "Sleep race" issue. force_idle=True: this is the one place
            # that finalizes state for an LLM turn (no Speaker.speak()
            # call in this path), so it must still move to IDLE itself
            # in the normal, non-sleeping case.
            await turn_rest(force_idle=True)
            return

        audio_tuple, expression, actions, attitude, intensity, phrase, *meta = item
        data, samplerate = audio_tuple
        phrase_id = meta[0] if meta else None
        t_phrase_created = meta[1] if len(meta) > 1 else None
        p_tag = f"[TTS][phrase={phrase_id}] " if phrase_id is not None else "[TTS] "

        # Wait for filler to finish before sending first real sentence
        if first_sentence:
            await _filler_done.wait()
            first_sentence = False

        for action in actions:
            await _ws.broadcast_animation(action)

        await _ws.broadcast_behavior(
            behavior_engine.compose(expression, actions, source="dialogue",
                                     attitude=attitude, intensity=intensity)
        )
        await state.set(MayaState.SPEAKING)
        await _ws.broadcast_state("speaking")

        if not first_played:
            logger.info(
                f"[TIMING][TTFA] audio playback start: "
                f"+{time.perf_counter()-t_cmd_start:.3f}s since command start (== TTFA)"
            )
            first_played = True

        _spoken_phrases.append(phrase)
        if output in ("avatar", "both"):
            from core.speaker import _numpy_to_wav
            wav_bytes = await loop.run_in_executor(None, _numpy_to_wav, data, samplerate)
            t0 = time.perf_counter()
            logger.info(f"[TIMING]{p_tag}AUDIO_BROADCAST_START samples={len(data)} sr={samplerate} t={t0:.3f}")
            await _ws.broadcast_audio(wav_bytes)
            await _ws.wait_for_audio_done()
            t_sync = time.perf_counter() - t0
            audio_dur = len(data) / samplerate if samplerate else 0.0
            total_str = f" total_lifecycle={time.perf_counter() - t_phrase_created:.3f}s" if t_phrase_created else ""
            logger.info(
                f"[TIMING]{p_tag}AUDIO_DONE: {t_sync:.3f}s "
                f"(playback_sync: audio_dur={audio_dur:.3f}s, overhead={t_sync - audio_dur:+.3f}s){total_str}"
            )

        if output in ("local", "both"):
            await loop.run_in_executor(None, _play_blocking, data, samplerate)


def _play_blocking(data, samplerate: int) -> None:
    sd.play(data, samplerate)
    sd.wait()