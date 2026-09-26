"""
skills/utilities/timer.py
Countdown timers with expression tags.

Expired timers are announced through the Speaker injected by Router
(set_speaker), run under state.run_interruptible so a barge-in can cut
the alert. The alert is queued on the command worker (queue_manager.put_job)
so it never overlaps a turn and commands spoken during it wait behind it.

"remind me to X" with no duration asks for one and keeps X pending; Router
calls resolve_pending() first so the next utterance supplies the duration.

Intent merge: datasets/intents.json now declares a single 'set_timer' intent
covering both a bare countdown and a timer carrying a reminder message
(the former separate 'set_reminder' no longer exists). execute() used to
tell them apart by intent id to decide whether to skip the cancel/status
word checks below; it now checks the wording itself (_REMINDER_RE) instead
— see the comment at that check.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict

from config.settings import config
from core.queue_manager import queue_manager
from core.state import state

logger = logging.getLogger(__name__)
_U = config.user_name

_MESSAGE_MAX = 120   # cap on spoken reminder text
_PENDING_TTL_S = 30  # how long a reminder waits for its duration

_REMINDER_RE = re.compile(r"\bremind(?:er)?\b.*?\b(?:to|about|that)\s+(.+)", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\s*\b(?:(?:in|after|for)\s+)?\d+(?:\.\d+)?\s*"
    r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.IGNORECASE,
)
_TRAILING_RE = re.compile(r"(?:\s+(?:and|please|thanks))+\s*$", re.IGNORECASE)

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
_NUM_WORD_RE = re.compile(
    rf"\b(?:({'|'.join(_TENS_WORDS)})(?:[\s-]({'|'.join(list(_NUM_WORDS)[:9])}))?"
    rf"|({'|'.join(_NUM_WORDS)}))"
    r"(?=\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b)",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"\b(?:cancel(?:l?ed|l?ing|s)?|stop(?:s|ped|ping)?|clear(?:s|ed|ing)?|delete(?:s|d)?)\b")
_STATUS_RE = re.compile(r"\b(?:status|how long|remaining|left|lists?)\b")
_ALL_RE    = re.compile(r"\ball\b")

@dataclass
class Timer:
    name:    str
    seconds: int
    message: str = ""
    task:    asyncio.Task = field(default=None, repr=False)


_timers: Dict[str, Timer] = {}
_timer_counter = 0
_speaker = None
_pending_reminder: tuple[str, float] | None = None   # (message, expiry on time.monotonic())


def set_speaker(speaker) -> None:
    """Called once by Router so alerts can speak through the shared Speaker."""
    global _speaker
    _speaker = speaker


def _words_to_digits(text: str) -> str:
    """'one hour' -> '1 hour' — only number words directly before a time unit."""
    def repl(m: re.Match) -> str:
        if m.group(1):
            units = _NUM_WORDS[m.group(2).lower()] if m.group(2) else 0
            return str(_TENS_WORDS[m.group(1).lower()] + units)
        return str(_NUM_WORDS[m.group(3).lower()])
    return _NUM_WORD_RE.sub(repl, text)


async def execute(intent: dict, text: str) -> str:
    global _pending_reminder
    text = _words_to_digits(text)
    t = text.lower()

    # A reminder's own wording ("remind me to stop by...") must not hit
    # the cancel/status word checks — e.g. "remind me to cancel the
    # subscription in 2 days" is a new reminder, not a cancel request.
    # Previously gated on the intent id (set_reminder vs. set_timer); both
    # now share the single 'set_timer' intent (see datasets/intents.json),
    # so this checks the reminder wording itself instead.
    if not _REMINDER_RE.search(t):
        if _CANCEL_RE.search(t):
            return _cancel(t)
        if _STATUS_RE.search(t):
            return _status()

    seconds, label = _parse_duration(text)
    if seconds is None:
        reminder = _extract_reminder(text)
        if reminder:
            _pending_reminder = (reminder, time.monotonic() + _PENDING_TTL_S)
            return f"[surprised] How long from now should I remind you, {_U}?"
        return f"[surprised] How long should I set the timer for, {_U}?"

    return await _set(seconds, label, _extract_reminder(text))


async def resolve_pending(text: str) -> str | None:
    """
    Called by Router.dispatch before intent routing. One-shot: the next
    utterance always consumes a pending reminder. Returns the spoken reply
    if it supplied a duration; None if nothing was pending, it expired, or
    the utterance had no duration (then it routes normally, reminder dropped).
    """
    global _pending_reminder
    if _pending_reminder is None:
        return None
    message, expires = _pending_reminder
    _pending_reminder = None
    if time.monotonic() > expires:
        return None
    seconds, _ = _parse_duration(_words_to_digits(text))
    if seconds is None:
        return None
    return await _set(seconds, "", message)


async def _set(seconds: int, label: str, message: str = "") -> str:
    global _timer_counter
    _timer_counter += 1
    name = label or f"timer {_timer_counter}"

    if name in _timers and not _timers[name].task.done():
        _timers[name].task.cancel()

    t = Timer(name=name, seconds=seconds, message=message)
    t.task = asyncio.create_task(_countdown(t))
    _timers[name] = t

    duration_str = _format_duration(seconds)
    logger.info(f"Timer set: '{name}' for {duration_str}" + (f" — reminder: '{message}'" if message else ""))
    return f"[happy] Timer set for {duration_str}, {_U}."


def _cancel(text: str) -> str:
    if not _timers:
        return f"[neutral] No active timers, {_U}."

    if _ALL_RE.search(text):
        for t in _timers.values():
            if not t.task.done():
                t.task.cancel()
        _timers.clear()
        return f"[relaxed] All timers cancelled, {_U}."

    name = list(_timers.keys())[-1]
    t = _timers.pop(name)
    if not t.task.done():
        t.task.cancel()
    return f"[relaxed] Timer '{name}' cancelled, {_U}."


def _status() -> str:
    active = {n: t for n, t in _timers.items() if not t.task.done()}
    if not active:
        return f"[neutral] No active timers, {_U}."
    parts = [f"'{n}'" for n in active]
    return f"[relaxed] Active timers: {', '.join(parts)}, {_U}."


def _forget(timer: Timer) -> None:
    """Remove this timer from the registry — never a newer one with the same name."""
    if _timers.get(timer.name) is timer:
        del _timers[timer.name]


async def _countdown(timer: Timer) -> None:
    try:
        await asyncio.sleep(timer.seconds)
        logger.info(f"Timer '{timer.name}' expired.")
        _forget(timer)   # expired — no longer listed/cancellable while the alert is queued
        await _alert(timer)
    except asyncio.CancelledError:
        logger.debug(f"Timer '{timer.name}' cancelled.")
    finally:
        _forget(timer)


async def _alert(timer: Timer) -> None:
    """Queue the alert on the command worker so it never overlaps a turn."""
    if _speaker is None:
        logger.warning("Timer alert skipped — no speaker registered.")
        return

    if timer.message:
        msg = f"[excited] Time's up, {_U}! Reminder: {timer.message}."
    else:
        msg = f"[excited] Time's up, {_U}! Your {timer.name} is done."

    async def _speak() -> None:
        logger.info(f"Timer alert: {msg}")
        await state.run_interruptible(_speaker.speak(msg))

    await queue_manager.put_job(_speak)


def _extract_reminder(text: str) -> str:
    """'remind me to X in 5 minutes' -> 'X'. Empty if there's no reminder wording."""
    m = _REMINDER_RE.search(text)
    if not m:
        return ""
    msg = re.sub(r"\s{2,}", " ", _DURATION_RE.sub("", m.group(1))).strip(" .,!?")
    msg = _TRAILING_RE.sub("", msg).strip(" .,!?")
    return msg[:_MESSAGE_MAX]


def _parse_duration(text: str) -> tuple[int | None, str]:
    text = text.lower()
    total = 0
    found = False

    # One pattern per unit — separate long/short-form patterns used to both
    # match "5 minutes" and double-count it.
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b',      3600),
        (r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b',   60),
        (r'(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b',   1),
    ]

    for pattern, multiplier in patterns:
        m = re.search(pattern, text)
        if m:
            total += float(m.group(1)) * multiplier
            found = True

    if not found:
        return None, ""

    label_match = re.search(
        r'(\b(?!(?:set|a|an|the|for|me|timer|minute|second|hour|min|sec)\b)\w+\b)\s+timer',
        text
    )
    label = label_match.group(1) if label_match else ""
    return int(total), label


def _format_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    parts  = []
    if h: parts.append(f"{h} hour{'s' if h > 1 else ''}")
    if m: parts.append(f"{m} minute{'s' if m > 1 else ''}")
    if s: parts.append(f"{s} second{'s' if s > 1 else ''}")
    return " and ".join(parts) if parts else "0 seconds"