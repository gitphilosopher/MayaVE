"""
skills/utilities/timer.py
Countdown timers with expression tags.

Bug fix: _alert() no longer instantiates a new Speaker() (which would reload
Kokoro from scratch on every alert). Instead it uses the module-level Kokoro
pipeline from llm_service directly via _synthesise_blocking, same as the LLM
pipeline does — fast, no cold-start.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict

from config.settings import config

logger = logging.getLogger(__name__)
_U = config.user_name


@dataclass
class Timer:
    name:    str
    seconds: int
    task:    asyncio.Task = field(default=None, repr=False)


_timers: Dict[str, Timer] = {}
_timer_counter = 0


async def execute(intent: dict, text: str) -> str:
    t = text.lower()

    if any(w in t for w in ("cancel", "stop", "clear", "delete")):
        return _cancel(t)

    if any(w in t for w in ("status", "how long", "remaining", "left", "list")):
        return _status()

    seconds, label = _parse_duration(text)
    if seconds is None:
        return f"[surprised] How long should I set the timer for, {_U}?"

    return await _set(seconds, label)


async def _set(seconds: int, label: str) -> str:
    global _timer_counter
    _timer_counter += 1
    name = label or f"timer {_timer_counter}"

    if name in _timers and not _timers[name].task.done():
        _timers[name].task.cancel()

    t = Timer(name=name, seconds=seconds)
    t.task = asyncio.create_task(_countdown(t))
    _timers[name] = t

    duration_str = _format_duration(seconds)
    logger.info(f"Timer set: '{name}' for {duration_str}")
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


async def _countdown(timer: Timer) -> None:
    try:
        await asyncio.sleep(timer.seconds)
        logger.info(f"Timer '{timer.name}' expired.")
        await _alert(timer.name)
    except asyncio.CancelledError:
        logger.debug(f"Timer '{timer.name}' cancelled.")
    finally:
        _timers.pop(timer.name, None)


async def _alert(name: str) -> None:
    """
    Fire the timer alert using the existing Kokoro pipeline from llm_service
    rather than spinning up a new Speaker() instance (which would reload the
    entire Kokoro model from disk — slow and wasteful).
    """
    from services.llm.llm_service import _synthesise_blocking
    from core.speaker import _numpy_to_wav
    from services.ws_server import ws_server
    from core.behavior_engine import behavior_engine
    from config.settings import config as cfg

    msg_clean = f"Time's up, {_U}! Your {name} is done."
    logger.info(f"Timer alert: {msg_clean}")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _synthesise_blocking, msg_clean)

    if result is not None:
        audio, samplerate = result
        output = getattr(cfg.tts, "output", "avatar")

        if output in ("avatar", "both"):
            await ws_server.broadcast_behavior(behavior_engine.compose("excited", source="alert"))
            await ws_server.broadcast_state("speaking")
            wav = await loop.run_in_executor(None, _numpy_to_wav, audio, samplerate)
            await ws_server.broadcast_audio(wav)
            await ws_server.wait_for_audio_done()
            await ws_server.broadcast_behavior(behavior_engine.compose("neutral", source="alert"))

        if output in ("local", "both"):
            import sounddevice as sd
            await loop.run_in_executor(None, lambda: (sd.play(audio, samplerate), sd.wait()))


def _parse_duration(text: str) -> tuple[int | None, str]:
    text = text.lower()
    total = 0
    found = False

    patterns = [
        (r'(\d+(?:\.\d+)?)\s*hour',   3600),
        (r'(\d+(?:\.\d+)?)\s*hr',     3600),
        (r'(\d+(?:\.\d+)?)\s*minute', 60),
        (r'(\d+(?:\.\d+)?)\s*min',    60),
        (r'(\d+(?:\.\d+)?)\s*second', 1),
        (r'(\d+(?:\.\d+)?)\s*sec',    1),
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