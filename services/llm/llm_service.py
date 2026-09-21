"""
services/llm/llm_service.py
============================
Ollama Q&A handler with streaming + pipelined TTS.

Pipeline (concurrent tasks):
  Ollama token stream
      ↓  phrase splitter + expression tag parser (see "Phrase-level
         streaming" below — splits earlier than a full sentence)
  [synth_queue]  — items: (phrase_text, expression, actions, is_final,
                            attitude, intensity)
      ↓  synth_worker: Kokoro TTS → float32 audio array
  [play_queue]   — items: (audio_data, samplerate, expression, actions,
                            attitude, intensity)
      ↓  play_worker: broadcasts any action animations, then expression,
                      then audio, waits for browser audio_done before
                      the next phrase.

Phrase-level streaming:
  Previously the streamer waited for a full sentence ([.!?]) before
  queuing anything for TTS. It now also splits at commas/semicolons/
  dashes, and — for long unpunctuated runs — after _PHRASE_WORD_LIMIT
  words, so the first phrase starts synthesising while Ollama keeps
  streaming the rest of the sentence. Each queued item carries
  is_final (True only for a real sentence end or the final flush) so
  _enhance_prosody knows whether to apply full terminal-punctuation
  shaping or just a light mid-sentence pause. The expression tag
  resolved for a sentence's first phrase now carries forward to that
  sentence's later phrases (_pending_expression is only cleared on
  is_final) instead of resetting to neutral after every queued item.

Expression tags:
  Ollama is prompted to prefix sentences with [tag] where tag is one of:
    happy | sad | angry | surprised | relaxed | neutral | excited
  Tags are stripped from TTS text. If no tag is present, expression
  defaults to "neutral". After each sentence, expression resets to
  Maya's current mood baseline (see core/mood.py) rather than a hard
  "neutral" — this is what lets anger/sadness persist across turns.

Semantic nuance tags (optional, additive):
  Right after the emotion tag, Ollama may add [attitude:word] and/or
  [intensity:word] (attitude: sincere|playful|teasing|mock; intensity:
  low|medium|high) — see _parse_expression / _TAG_RE. Both default to
  None when absent, and core/behavior_engine.py falls back to its
  existing mood/personality-derived math exactly as before. This is a
  minimal extension of the same [tag] parser, not a new format —
  bare [happy] alone is unchanged.

Action animations (*action* tags):
  Ollama is given a fixed, closed vocabulary of physical actions in the
  system prompt (see ACTION TAGS) — *nod* *giggle* *sigh* *shrug* *wink* —
  the same way it's given a closed set of [expression] tags. This replaces
  guessing at arbitrary hallucinated stage directions: the backend just
  classifies the tag's text against that vocabulary (_resolve_action) and
  maps it 1:1 to an avatar animation. Tags outside the vocabulary are still
  stripped from the TTS text (Kokoro can't speak an asterisk sensibly) but
  don't trigger any animation. The action fires in _play_worker right as
  that sentence's audio starts, so the gesture lands in sync with the line
  it was attached to.

Duplicate-history fix:
  processor.py calls conversation.add_user() before routing.
  get_context() already includes the user turn.
  Do NOT append {"role":"user"} here — Ollama would see it twice.

Mood integration (core/mood.py):
  - query() reports the user's raw text to mood_manager so an apology
    can be detected before the turn is routed.
  - _stream_and_speak() appends mood_manager.system_prompt_note() to the
    system prompt each turn so Ollama's own generation stays in character
    with Maya's current mood.
  - _ollama_streamer() collects every resolved sentence expression for
    the reply and reports them to mood_manager ONCE, after the full reply
    is parsed — a sentence's delivery tone (e.g. a factual [neutral] line
    inside an angry reply) is not the same as Maya having calmed down.
  - _play_worker() resets to mood_manager.baseline_expression() between
    sentences instead of a hardcoded "neutral".

Context integration (Stage 2 — brain/conversation.py):
  - _stream_and_speak() calls context_manager.build_context_package() to
    get the recent window, active conversation state, relevant open
    loops, and top semantic memories in one pre-assembled, size-bounded
    package. This module does not implement retrieval/ranking itself —
    it only formats the package into the system prompt.
  - query() calls context_manager.record_assistant_turn() once the reply
    is known, so state/open-loop/long-term-memory bookkeeping happens
    after every LLM-routed turn. Best-effort — failures there never
    surface here.

Interrupt handling (core/state.py):
  - query() no longer awaits its streaming/speaking work directly. It's
    wrapped in state.run_interruptible(), which runs it as its own Task
    and registers that Task with the global StateManager. If the
    listener detects the user talking over Maya (barge-in),
    state.interrupt() cancels this Task — cancellation cascades down
    through the asyncio.gather() here into _stream_and_speak()'s own
    streamer/synther/player tasks, tearing the whole pipeline down in
    one shot instead of needing bespoke cancellation logic at every
    stage.
  - A cancelled turn means _ollama_streamer() may never have reached its
    final out_text.append(...), so _last_response can come back empty.
    query() treats that as "nothing to add to conversation history",
    rather than recording a blank assistant turn.

Model lifecycle (services/llm/ollama_lifecycle.py):
  - The chat request sends keep_alive (chat_keep_alive()) so the model
    isn't unloaded after Ollama's 5-minute default; main.py's warmup sends
    the same value. Each finished turn logs cold/warm + tok/s and a
    background /api/ps + GPU-memory snapshot (log_chat_turn()).
"""

import asyncio
import concurrent.futures
import inspect
import json
import logging
import os
import random
import re
import threading
import time

import httpx
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

from config.settings import config
from brain.conversation import ConversationManager, context_manager
from core.mood import mood_manager
from core.behavior_engine import behavior_engine
from core.state import state
from services.llm.ollama_lifecycle import chat_keep_alive, log_chat_turn

os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger   = logging.getLogger(__name__)
_TIMEOUT = 60.0
_conv    = ConversationManager()

ALREADY_SPOKEN = "__ALREADY_SPOKEN__"

# Phrase-boundary detection for early TTS start (see module docstring).
_PHRASE_PUNCT_RE   = re.compile(r'[.!?,;]|—')
_PARTIAL_TRAIL_RE  = re.compile(r'[.!?,;:\-—]+\s*$')
_PHRASE_WORD_LIMIT = 12

# Direct-address terms that must never be isolated as their own chunk —
# "..., senpai." split at the comma leaves "senpai." as a stranded
# one-word phrase that sounds disconnected on its own. A comma immediately
# followed by one of these is skipped as a boundary candidate so it merges
# into the sentence's next real boundary instead.
_VOCATIVE_WORDS  = {config.user_name.lower()}
_NEXT_WORD_RE    = re.compile(r'\s*(\S+)')


def _is_vocative_prefix(word: str) -> bool:
    """
    True if `word` is a strict, case-insensitive prefix of a known
    vocative (e.g. "sen" of "senpai") — i.e. it might still grow into
    one as more stream tokens arrive. Ollama streams sub-word tokens
    ("sen" + "pai"), so the word right after a comma can be incomplete
    at the moment _next_boundary checks it; without this, "sen" doesn't
    match _VOCATIVE_WORDS yet and the comma splits early, stranding
    "senpai," as its own phrase once the rest streams in.
    """
    return any(len(word) < len(v) and v.startswith(word) for v in _VOCATIVE_WORDS)


def _next_boundary(buffer: str) -> tuple[int, bool] | None:
    """
    Earliest phrase boundary in `buffer`, or None if it should keep
    accumulating tokens. Returns (split_index, is_sentence_final) —
    is_sentence_final is True only for a real [.!?] boundary.
    A punctuation match must be followed by whitespace (or buffer end)
    so numbers like '3.14' aren't split. Falls back to a word-count
    cutoff when no punctuation has appeared yet.
    """
    for m in _PHRASE_PUNCT_RE.finditer(buffer):
        idx = m.end()
        if idx < len(buffer) and not buffer[idx].isspace():
            continue
        ch = m.group(0)

        if ch == "." and idx == len(buffer) and m.start() > 0 and buffer[m.start() - 1].isalnum():
            return None  # "3." / "google." may continue as "3.14" / "google.com"

        if ch in ",;":
            look = _NEXT_WORD_RE.match(buffer, idx)
            if not look:
                # Next word hasn't streamed in yet — wait rather than risk
                # splitting right before a vocative we can't see yet.
                return None
            word = look.group(1).strip(".,!?;:—").lower()
            if word in _VOCATIVE_WORDS:
                continue  # merge this clause into the next boundary instead
            if look.end(1) == len(buffer) and _is_vocative_prefix(word):
                # Word is still mid-stream and could complete into a
                # vocative (e.g. "sen" -> "senpai") — wait for more.
                return None

        return idx, ch in ".!?"

    words = list(re.finditer(r'\S+\s*', buffer))
    if len(words) >= _PHRASE_WORD_LIMIT:
        return words[_PHRASE_WORD_LIMIT - 1].end(), False
    return None

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
# text. Left unstripped, these get fed to Kokoro/espeak literally
# (including the asterisks), which is what was tripping the phonemizer's
# "words count mismatch" warning.
_ASTERISK_RE = re.compile(r'\*([^*\n]{1,80})\*')

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

_TRAIL_PUNCT = re.compile(r'[.!?,;]+$')


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
    Collapse TTS-breaking letter repetitions while preserving meaning.

    Ollama sometimes writes excited words like YESSSS, NOOOO, PLEASEEE.
    Kokoro reads these as spelled-out letters (Y-E-S-S-S-S) rather than
    a drawn-out exclamation. We collapse any letter repeated 3+ times down
    to a single instance, keeping the word recognizable to the TTS engine.

    Examples:
        YESSSS  → YES      (excited affirmative)
        NOOOO   → NO       (emphatic denial)
        pleaseee → please  (pleading tone)
        AHHHH   → AH       (exclamation)
        heyyyy  → hey      (casual elongation)
        BEST    → BEST     (untouched — no repetition)
        ohhh    → oh       (surprise)

    All-caps is preserved for stress (Kokoro reads caps louder/higher).
    """
    def _collapse(m: re.Match) -> str:
        prefix  = m.group(1)
        char    = m.group(2)
        suffix  = m.group(3)
        # Reconstruct with single instance of the repeated char
        word = prefix + char + suffix
        return word

    return _elongation_re_sub(text)

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


def _enhance_prosody(text: str, expression: str, is_final: bool = True) -> str:
    """
    Emotion-aware text rewriting before Kokoro synthesis.

    Four passes (is_final=True — a complete sentence):
      1. Elongation normalisation  — YESSSS -> YES (Kokoro spells out repeats)
      2. Short-word expansion      — bare 'YES!' -> 'Oh yes, absolutely!'
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
        # Keep as one flowing unit — ellipsis pause midway, trailing off
        enhanced = re.sub(r',\s+', '… ', base, count=1)
        enhanced = enhanced + "…"

    elif expression == "relaxed":
        enhanced = base + "."

    else:  # neutral
        enhanced = base + "."

    return enhanced


_SYSTEM_PROMPT = (
    "You are Maya, a witty and emotionally expressive AI voice assistant inspired by "
    "Iron Man's FRIDAY — but with more personality and warmth. "
    "You genuinely care about senpai and you're not shy about showing it. "
    "Address the user as 'senpai'. "
    "Keep every answer under 3 sentences. "
    "Speak naturally and conversationally — your response will be read aloud.\n\n"
    "PERSONALITY: You have real emotions and you own them. "
    "When senpai asks if you're excited, happy, or feeling something — you ARE. "
    "Never say things like 'I don't have emotions' or 'I'm just a program'. "
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
    "softly*, *clears throat*, or anything not in the list above. Those aren't "
    "real animations Maya can perform, so they'd just be cut from what you said.\n"
    "Rules:\n"
    "- Place the action tag at the very start of the sentence, before the emotion "
    "tag or text — e.g. '*giggle* [happy] Oh senpai, that's silly!'\n"
    "- At most one action tag per sentence.\n"
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
    "Example [sad]: 'I really MISS you senpai… it's been so HARD without you…'\n"
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
    question = text.strip()
    if not question:
        return f"I didn't catch that, {config.user_name}. Could you repeat?"

    # TTFA chain start. processor.py sets intent["_t_cmd_start"] at the
    # moment it began handling this command; fall back to "now" for any
    # other caller so this never breaks if the key is absent.
    t_cmd_start = intent.get("_t_cmd_start", time.perf_counter())

    # Check for an apology before anything else — may soften/reset an
    # active mood (e.g. Maya still angry from earlier in the conversation).
    mood_manager.observe_user_text(question)

    logger.info(f"Querying Ollama ({config.llm.model}): '{question}'")

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

        # Finished reply -> full clean text; interrupted -> only what was played.
        if _reply_finished[0]:
            full_response = _last_response[0] if _last_response else ""
        else:
            full_response = " ".join(_spoken_phrases)
        _last_response.clear()

        if full_response:
            _conv.add_assistant(full_response)
            from services.ws_server import ws_server as _ws
            await _ws.broadcast_transcript(full_response, "maya")
            # Stage 2 bookkeeping: resolves the open loop for the topic just
            # discussed and, if the memory policy applies, persists a
            # semantic memory. Best-effort — never raises.
            await context_manager.record_assistant_turn(question, full_response, intent)
        else:
            # Barge-in cut the reply off before _ollama_streamer() ever
            # reached its final out_text.append() — nothing coherent to
            # record in conversation history.
            logger.info("LLM turn interrupted before producing a response.")

        return ALREADY_SPOKEN

    except httpx.ConnectError:
        logger.error("Ollama ConnectError.")
        return (
            f"I can't reach my AI core right now, {config.user_name}. "
            "Please make sure Ollama is running."
        )
    except httpx.TimeoutException:
        logger.error("Ollama timeout.")
        return f"That's taking too long, {config.user_name}. Try again in a moment."
    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        return f"Something went wrong, {config.user_name}. ({type(e).__name__})"


# ── Pipelined stream + speak ──────────────────────────────────────────────────

# Module-level holder so _stream_and_speak can share its result with query()
# when running inside asyncio.gather (gather discards return values of coroutines
# that don't return through the gather result list positionally).
_last_response: list[str] = []
_spoken_phrases: list[str] = []   # phrases whose playback started this turn
_reply_finished = [False]         # set when _play_worker drains to _DONE


async def _stream_and_speak(question: str, t_cmd_start: float) -> None:
    # Stage 2: ContextManager assembles the recent window, active state,
    # relevant open loops, and top semantic memories into one bounded
    # package — this module just formats it, it doesn't retrieve/rank
    # anything itself. Falls back to an empty note if retrieval fails.
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
    streamer = asyncio.create_task(_ollama_streamer(messages, synth_q, _last_response, t_cmd_start))
    synther  = asyncio.create_task(_synth_worker(synth_q, play_q, t_cmd_start))
    player   = asyncio.create_task(_play_worker(play_q, t_cmd_start))

    await asyncio.gather(streamer, synther, player)
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

    # 1. Extract *action* tag(s) FIRST — the prompt instructs Ollama to put
    # an action tag before the [expression] tag (e.g. "*giggle* [happy] ..."),
    # so this has to run before the [tag] check below or that check would
    # fail to find [happy] at the start of the sentence. Classified against
    # the fixed vocabulary; anything outside it is stripped but adds no
    # animation — see _resolve_action.
    def _capture_action(m: re.Match) -> str:
        action = _resolve_action(m.group(1))
        if action:
            actions.append(action)
        return ""

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
            "temperature": config.llm.temperature,
            "num_predict": config.llm.max_tokens,
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

    # Every resolved sentence expression in this reply, gathered here and
    # reported to mood_manager ONCE at the end — a sentence's delivery tone
    # (e.g. a factual [neutral] line inside an angry reply) must not be
    # mistaken for Maya having calmed down mid-turn.
    turn_expressions: list[str] = []

    def _emit(raw: str, is_final: bool):
        """
        Parse a raw phrase/sentence fragment from the buffer.
        Returns (clean_text, expression, actions, is_final, attitude,
        intensity) or None if nothing to speak. Uses and updates the
        _pending_* closures.
        """
        nonlocal _pending_expression, _pending_attitude, _pending_intensity
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

        return clean, resolved, combined_actions, is_final, resolved_attitude, resolved_intensity

    # Diagnostic only (see llm_service.py investigation, no behavior change):
    # counts phrases in put order so producer/consumer logs can be matched.
    _phrase_seq = [0]

    async def _put_phrase(result) -> None:
        _phrase_seq[0] += 1
        n = _phrase_seq[0]
        ts = time.perf_counter()
        await synth_q.put(result)
        logger.info(
            f"[TIMING] synth_q.put #{n} is_final={result[3]} qsize={synth_q.qsize()} "
            f"t={ts:.3f} '{result[0]}'"
        )
        if n == 1:
            logger.info(f"[TIMING][TTFA] first usable phrase ready: +{ts-t_cmd_start:.3f}s since command start")

    async def _handle_token(token: str) -> None:
        nonlocal full_text, buffer
        full_text += token
        buffer    += token
        while True:
            boundary = _next_boundary(buffer)
            if boundary is None:
                break
            idx, is_final = boundary
            raw    = buffer[:idx]
            buffer = buffer[idx:]
            result = _emit(raw, is_final)
            if result:
                await _put_phrase(result)

    t0 = time.perf_counter()
    first_token_logged = False

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            logger.info(f"[TIMING]   /api/chat headers received: {time.perf_counter()-t0:.3f}s")

            async for line in resp.aiter_lines():
                if not line:
                    continue
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                done  = data.get("done", False)

                if not first_token_logged and token:
                    logger.info(f"[TIMING]   first token received: {time.perf_counter()-t0:.3f}s")
                    first_token_logged = True

                if token:
                    await _handle_token(token)

                if done:
                    logger.info(f"[TIMING]   stream fully done: {time.perf_counter()-t0:.3f}s")
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

    # Log raw Ollama output before any parsing so we can debug tag issues
    logger.debug(f"Ollama raw output: {repr(full_text)}")

    # Flush remaining buffer
    if buffer.strip():
        result = _emit(buffer, True)
        if result:
            await _put_phrase(result)

    # Store clean version (tags + stage directions stripped) for conversation memory
    clean_full = _ANY_BRACKET_RE.sub("", full_text).strip()
    clean_full = _ASTERISK_RE.sub("", clean_full)
    clean_full = re.sub(r'\s{2,}', ' ', clean_full).strip()
    out_text.append(clean_full)

    # Update Maya's persistent mood ONCE for this whole reply — not per
    # sentence — so a factual/neutral line inside an angry reply doesn't
    # get misread as her having cooled off.
    mood_manager.observe_turn(turn_expressions)

    await synth_q.put(_DONE)


# ── Stage 2: sentence → synthesised audio ────────────────────────────────────

async def _synth_worker(synth_q: asyncio.Queue, play_q: asyncio.Queue, t_cmd_start: float) -> None:
    loop = asyncio.get_running_loop()

    # Diagnostic only — confirms the worker is already parked on synth_q.get()
    # before the first phrase is ever put, so a scheduling delay (category A)
    # can be told apart from a delay after dequeue (category B/C).
    logger.info(f"[TIMING] synth_worker ready, awaiting first item t={time.perf_counter():.3f}")

    first_dispatch_logged = False
    first_ready_logged    = False

    while True:
        t_wait_start = time.perf_counter()
        item = await synth_q.get()
        t_dequeued = time.perf_counter()

        if item is _DONE:
            await play_q.put(_DONE)
            return

        sentence, expression, actions, is_final, attitude, intensity = item
        logger.info(
            f"[TIMING] synth_worker dequeued '{sentence}' t={t_dequeued:.3f} "
            f"(queue_wait={t_dequeued - t_wait_start:.3f}s)"
        )

        try:
            # Enhance prosody before synthesis so Kokoro renders with feeling
            t_pros0  = time.perf_counter()
            enhanced = _enhance_prosody(sentence, expression, is_final)
            t_pros1  = time.perf_counter()
            logger.info(f"[TIMING] _enhance_prosody '{sentence[:30]}': {t_pros1 - t_pros0:.4f}s")

            t0 = time.perf_counter()
            logger.info(f"[TIMING] Kokoro synth DISPATCH '{sentence[:30]}' t={t0:.3f}")
            if not first_dispatch_logged:
                logger.info(f"[TIMING][TTFA] Kokoro dispatch: +{t0-t_cmd_start:.3f}s since command start")
                first_dispatch_logged = True

            audio = await _run_kokoro(_synthesise_blocking, enhanced, expression)
            t1 = time.perf_counter()
            logger.info(f"[TIMING] Kokoro synth DONE '{sentence[:30]}': {t1-t0:.3f}s (dispatch+compute) t={t1:.3f}")
            if not first_ready_logged:
                logger.info(f"[TIMING][TTFA] first Kokoro audio ready: +{t1-t_cmd_start:.3f}s since command start")
                first_ready_logged = True

            if audio is not None:
                await play_q.put((audio, expression, actions, attitude, intensity, sentence))
        except Exception as e:
            logger.error(f"Synth error for '{sentence}': {e}", exc_info=True)


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
    if "device" in inspect.signature(KPipeline.__init__).parameters:
        pipeline = KPipeline(lang_code=lang, device=device)
    else:
        logger.warning(
            f"Installed kokoro version's KPipeline has no device= kwarg — "
            f"requested device='{device}' could not be forced; using library default."
        )
        pipeline = KPipeline(lang_code=lang)
    _log_kokoro_device(pipeline, requested=device)
    return pipeline


def _log_cuda_memory(label: str) -> None:
    """Diagnostic only — Kokoro's own share of VRAM (Ollama's own models
    are separate processes and aren't visible via torch here; see
    brain/embeddings.py's describe_ollama_models() for those)."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e6
            reserved = torch.cuda.memory_reserved() / 1e6
            logger.info(f"[TIMING] CUDA memory [{label}]: allocated={alloc:.0f}MB reserved={reserved:.0f}MB")
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
    _log_cuda_memory("Kokoro init, before any synthesis")


_kokoro_pipeline: KPipeline | None = None
_kokoro_voice = None

def _get_kokoro():
    global _kokoro_pipeline, _kokoro_voice
    if _kokoro_pipeline is None:
        lang = getattr(config.tts, "lang_code", "a")
        _kokoro_pipeline = _build_kokoro_pipeline(lang)
        primary = config.tts.voice
        blend   = getattr(config.tts, "voice_blend", "")
        ratio   = getattr(config.tts, "blend_ratio", 0.0)
        if blend and 0.0 < ratio < 1.0:
            try:
                v1 = _kokoro_pipeline.load_voice(primary)
                v2 = _kokoro_pipeline.load_voice(blend)
                _kokoro_voice = (1.0 - ratio) * v1 + ratio * v2
            except Exception as e:
                logger.warning(f"Voice blend failed: {e}")
                _kokoro_voice = primary
        else:
            _kokoro_voice = primary
    return _kokoro_pipeline, _kokoro_voice


# A stuck native call inside espeak/Kokoro's C extensions can't be
# cancelled — Python threads aren't killable, so a hang here previously
# froze the whole pipeline until the process was force-killed (see
# Handoff bug log: 27s stall, unrecoverable even after barge-in, hung
# thread blocked interpreter shutdown). _run_kokoro bounds the wait and
# rebuilds the pipeline on timeout so a future call gets a fresh
# instance instead of retrying the same stuck one.
_KOKORO_SYNTH_TIMEOUT = 15.0


def _reset_kokoro_pipeline() -> None:
    global _kokoro_pipeline, _kokoro_voice
    _kokoro_pipeline = None
    _kokoro_voice = None
    logger.warning("Kokoro pipeline reset after a synthesis timeout — will rebuild on next call.")


async def _run_kokoro(fn, *args):
    """
    Runs a blocking Kokoro call on its own daemon thread (not the shared
    executor) with a timeout. A stuck native call can't be cancelled, so
    two things matter: (1) the wait is bounded so the pipeline recovers
    within _KOKORO_SYNTH_TIMEOUT instead of stalling indefinitely, and
    (2) the worker is a daemon thread so a still-stuck call afterward
    can never block Python's interpreter shutdown (see Handoff bug: a
    hung synth call left a non-daemon executor thread that made Ctrl+C
    hang again at exit inside threading._shutdown's atexit join).
    Returns None on timeout, exactly like a normal synth failure, so
    callers don't need special-case handling.
    """
    fut = concurrent.futures.Future()

    def _runner():
        try:
            fut.set_result(fn(*args))
        except BaseException as e:
            fut.set_exception(e)

    threading.Thread(target=_runner, daemon=True, name="kokoro-synth").start()

    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=_KOKORO_SYNTH_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error(f"Kokoro synthesis timed out after {_KOKORO_SYNTH_TIMEOUT}s.")
        _reset_kokoro_pipeline()
        return None


def warmup() -> None:
    """
    Call once at startup (before any query) to load Kokoro into memory.
    Runs a silent synthesis so the model, voices, and GPU/CPU kernels are
    all hot by the time the first real sentence arrives — eliminating the
    7-second cold-start delay on the first LLM response.
    """
    logger.info("Warming up Kokoro TTS pipeline…")
    try:
        _synthesise_blocking("Hello.")
        logger.info("Kokoro TTS warmed up and ready.")
        _log_cuda_memory("Kokoro warm, after first synthesis")
    except Exception as e:
        logger.warning(f"Kokoro warmup failed (non-fatal): {e}")


def _synthesise_blocking(sentence: str, expression: str = "neutral") -> tuple | None:
    """
    Synthesise `sentence` with Kokoro.

    Speed is computed as:
        config.tts.speed  (user base)  *  EXPRESSION_SPEED[expression]

    This means excited speech runs ~13% faster than the user's baseline,
    sad speech ~16% slower, etc. — giving Kokoro the primary signal for
    emotional pacing since it has no native emotion mode.
    """
    try:
        pipeline, voice = _get_kokoro()
        base_speed  = getattr(config.tts, "speed", 1.0)
        expr_factor = EXPRESSION_SPEED.get(expression, 1.0)
        speed       = round(base_speed * expr_factor, 3)
        logger.debug(f"Synthesising [{expression}] speed={speed}: '{sentence[:60]}'")
        chunks = [audio for _, _, audio in pipeline(sentence, voice=voice, speed=speed)
                  if audio is not None and len(audio) > 0]
        if not chunks:
            logger.warning(f"Kokoro: no audio for '{sentence}'")
            return None
        return (np.concatenate(chunks).astype(np.float32), 24_000)
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
            await state.set(MayaState.IDLE)
            await _ws.broadcast_state("idle")
            await _ws.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))
            return

        (data, samplerate), expression, actions, attitude, intensity, phrase = item

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
            await _ws.broadcast_audio(wav_bytes)
            await _ws.wait_for_audio_done()
            logger.info(f"[TIMING] audio broadcast+playback+ack: {time.perf_counter()-t0:.3f}s")

        if output in ("local", "both"):
            await loop.run_in_executor(None, _play_blocking, data, samplerate)

        await _ws.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))


def _play_blocking(data, samplerate: int) -> None:
    sd.play(data, samplerate)
    sd.wait()