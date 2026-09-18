"""
core/wake_word.py
Wake-word detector — runs on every audio frame while Maya is SLEEPING.

Strategy: lightweight keyword spotting using Google Speech Recognition
on short audio windows (no extra ML model required).

How it works:
  1. While state == SLEEPING, accumulate audio frames into a rolling
     2-second window.
  2. Every time the window fills, transcribe it with Google STT.
  3. If the transcription contains the wake word → wake Maya up.
  4. While state != SLEEPING, the detector is a no-op (zero overhead).

Wake word is set in config/settings.py:
  config.wake_word = "hey maya"   ← default

To go back to sleep, say "go to sleep" / "sleep" / "goodbye".

Shared wake-phrase matching:
  compute_wake_triggers() / contains_wake_word() below are also used by
  main.py to gate barge-in (see core/state.py's interrupt()) — talking
  over Maya only cuts her off if you actually call her by name ("hey
  maya", "hello maya", "maya", ...), not just any speech captured while
  she's mid-sentence. Keeping one shared trigger set means changing
  config.wake_word moves both behaviours together instead of drifting
  out of sync.
"""

import asyncio
import logging
from collections import deque
from typing import Callable, Awaitable, Optional

import numpy as np
import speech_recognition as sr

from config.settings import config
from core.state import state, MayaState

logger = logging.getLogger(__name__)

# How many frames to accumulate before attempting a wake-word transcription.
# At 512 samples / 16 000 Hz = 32 ms per frame → 63 frames ≈ 2 seconds.
_SAMPLE_RATE    = config.audio.sample_rate
_FRAME_SAMPLES  = max(512, int(_SAMPLE_RATE * config.audio.chunk_ms / 1000))
_WINDOW_SECONDS = 2
_WINDOW_FRAMES  = int(_WINDOW_SECONDS * _SAMPLE_RATE / _FRAME_SAMPLES)

WakeCallback = Callable[[], Awaitable[None]]

# Common greeting prefixes people naturally say before a wake word —
# "hey maya", "hello maya", "hi maya" all mean the same thing. Matched
# generically rather than assuming the exact phrase in config.wake_word
# is the only way someone will address her.
_GREETING_PREFIXES = ("hey", "hello", "hi", "yo")


def compute_wake_triggers(wake_word: Optional[str] = None) -> set[str]:
    """
    Build the full set of phrases that count as "calling Maya by name",
    derived from config.wake_word (or an explicit override).

    For a configured wake_word of "hey maya" this produces roughly:
      {"hey maya", "hello maya", "hi maya", "yo maya", "maya"}

    For a bare single-word wake_word like "maya" it produces:
      {"maya", "hey maya", "hello maya", "hi maya", "yo maya"}
    """
    wake = (wake_word or config.wake_word).lower().strip()
    name = wake.split()[-1] if " " in wake else wake

    triggers = {wake, name}
    for prefix in _GREETING_PREFIXES:
        triggers.add(f"{prefix} {name}")
    return triggers


def contains_wake_word(text: str, wake_word: Optional[str] = None) -> bool:
    """True if `text` contains one of the wake-phrase triggers."""
    t = text.lower()
    return any(trigger in t for trigger in compute_wake_triggers(wake_word))


class WakeWordDetector:
    """
    Plug into the Listener's frame pipeline.
    Call feed_frame(frame) on every VAD frame.
    Fires on_wake() coroutine when the wake word is detected.
    """

    def __init__(self, on_wake: WakeCallback, loop: asyncio.AbstractEventLoop):
        self._on_wake   = on_wake
        self._loop      = loop
        self._recogniser = sr.Recognizer()
        self._window: deque[np.ndarray] = deque(maxlen=_WINDOW_FRAMES)
        self._pending   = False   # True while a transcription is in-flight

        self._triggers = compute_wake_triggers()

        logger.info(
            f"WakeWordDetector ready — triggers: {self._triggers}  "
            f"window: {_WINDOW_SECONDS}s ({_WINDOW_FRAMES} frames)"
        )

    # ── Called from the sounddevice callback thread ───────────────────

    def feed_frame(self, frame: np.ndarray) -> None:
        """
        Receive one audio frame. Only does real work while SLEEPING.
        Zero-overhead no-op at all other times.
        """
        if not state.is_sleeping():
            self._window.clear()   # reset buffer when we wake up
            self._pending = False
            return

        self._window.append(frame)

        # Check once per full window, skip if a check is already running
        if len(self._window) == _WINDOW_FRAMES and not self._pending:
            audio = np.concatenate(list(self._window))
            self._window.clear()
            self._pending = True
            asyncio.run_coroutine_threadsafe(
                self._check(audio), self._loop
            )

    # ── Async transcription & detection ──────────────────────────────

    async def _check(self, audio: np.ndarray) -> None:
        """Transcribe the window and fire on_wake if trigger found."""
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                None, self._transcribe, audio
            )
            if text:
                logger.debug(f"Wake-word window heard: '{text}'")
                if any(trigger in text for trigger in self._triggers):
                    logger.info(f"🔔 Wake word detected: '{text}'")
                    await self._on_wake()
        finally:
            self._pending = False

    def _transcribe(self, audio: np.ndarray) -> str | None:
        """Blocking Google STT call — runs in executor."""
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        audio_data = sr.AudioData(pcm, _SAMPLE_RATE, 2)
        try:
            return self._recogniser.recognize_google(audio_data).lower().strip()
        except (sr.UnknownValueError, sr.RequestError):
            return None