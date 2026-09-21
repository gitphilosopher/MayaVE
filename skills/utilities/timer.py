"""
skills/utilities/timer.py
Countdown timers with expression tags.

Expired timers are announced through the Speaker injected by Router
(set_speaker), run under state.run_interruptible so a barge-in can cut
the alert. The alert waits for any in-flight turn to finish first.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict

from config.settings import config
from core.state import state

logger = logging.getLogger(__name__)
_U = config.user_name

_ALERT_MAX_WAIT_S = 60    # longest an alert waits for Maya to finish a turn
_MESSAGE_MAX      = 120   # cap on spoken reminder text

_REMINDER_RE = re.compile(r"\bremind(?:er)?\b.*?\b(?:to|about|that)\s+(.+)", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\s*\b(?:(?:in|after|for)\s+)?\d+(?:\.\d+)?\s*"
    r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.IGNORECASE,
)
_TRAILING_RE = re.compile(r"(?:\s+(?:and|please|thanks))+\s*$", re.IGNORECASE)


@dataclass
class Timer:
    name:    str
    seconds: int
    message: str = ""
    task:    asyncio.Task = field(default=None, repr=False)


_timers: Dict[str, Timer] = {}
_timer_counter = 0
_speaker = None


def set_speaker(speaker) -> None:
    """Called once by Router so alerts can speak through the shared Speaker."""
    global _speaker
    _speaker = speaker


async def execute(intent: dict, text: str) -> str:
    t = text.lower()

    # A reminder's own wording ("remind me to stop by...") must not hit
    # the cancel/status word checks.
    if intent.get("intent") != "set_reminder":
        if any(w in t for w in ("cancel", "stop", "clear", "delete")):
            return _cancel(t)

        if any(w in t for w in ("status", "how long", "remaining", "left", "list")):
            return _status()

    seconds, label = _parse_duration(text)
    if seconds is None:
        return f"[surprised] How long should I set the timer for, {_U}?"

    return await _set(seconds, label, _extract_reminder(text))


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

    if "all" in text:
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
        _forget(timer)   # expired — no longer listed/cancellable while the alert waits
        await _alert(timer)
    except asyncio.CancelledError:
        logger.debug(f"Timer '{timer.name}' cancelled.")
    finally:
        _forget(timer)


async def _alert(timer: Timer) -> None:
    """Speak the alert via the shared Speaker once Maya isn't mid-turn."""
    if _speaker is None:
        logger.warning("Timer alert skipped — no speaker registered.")
        return

    for _ in range(_ALERT_MAX_WAIT_S * 2):
        if not state.is_busy():
            break
        await asyncio.sleep(0.5)

    if timer.message:
        msg = f"[excited] Time's up, {_U}! Reminder: {timer.message}."
    else:
        msg = f"[excited] Time's up, {_U}! Your {timer.name} is done."
    logger.info(f"Timer alert: {msg}")

    await state.run_interruptible(_speaker.speak(msg))


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
        r'(\b(?!set|a|an|the|for|me|timer|minute|second|hour|min|sec)\w+\b)\s+timer',
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