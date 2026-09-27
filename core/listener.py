"""
core/listener.py
Microphone capture and VAD pipeline for Maya.

This module owns the raw audio ingest path. It opens the configured input stream,
feeds each audio frame into the wake-word detector regardless of state, and only
runs the Silero VAD speech detection path while Maya is awake. When a completed
utterance ends, it delivers the captured waveform to the registered `on_speech`
callback for transcription and command handling.

The listener is intentionally a pure capture component. It does not decide whether
speech should interrupt Maya's current response; that policy is left to the
higher-level processing path in `main.py` after the utterance has been transcribed,
because the interrupt decision depends on the actual spoken text and wake-word
match, not only on audio energy.

Important implementation details:
- the wake-word detector is always active, even while sleeping, so the system can
  react immediately when the user says the wake phrase
- VAD processing is skipped while sleeping to avoid capturing and queuing stray
  audio after the agent has intentionally gone to sleep
- `_FRAME_SAMPLES` is clamped to at least 512 samples so the Silero VAD input
  remains valid on Windows and other environments with small chunk sizes
- the listener uses a pre-roll buffer to keep the leading portion of speech when
  VAD starts mid-frame, and it clears any partial utterance if sleep wins while
  the mic remains active
"""

import asyncio
import logging
import numpy as np
from collections import deque
from typing import Callable, Awaitable

import sounddevice as sd
import torch

from config.settings import config
from core.state import state, MayaState
from core.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)

AudioCallback = Callable[[np.ndarray], Awaitable[None]]

_SAMPLE_RATE = config.audio.sample_rate
_FRAME_SAMPLES = max(512, int(_SAMPLE_RATE * config.audio.chunk_ms / 1000))
_FRAME_MS = _FRAME_SAMPLES / _SAMPLE_RATE * 1000
_SILENCE_FRAMES = max(1, int(config.audio.silence_ms / _FRAME_MS))
_PRE_ROLL_FRAMES = max(1, int(config.audio.pre_roll_ms / _FRAME_MS))


class Listener:
    """Open the microphone, detect wake words, and emit completed utterance audio."""

    def __init__(self, on_speech: AudioCallback, on_wake: Callable[[], Awaitable[None]]):
        """Initialize the mic pipeline and VAD model for the provided callbacks."""
        self._on_speech = on_speech
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_wake_cb = on_wake

        # Silero VAD
        logger.info("Loading Silero VAD model…")
        self._vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._vad_model.eval()
        (self._get_speech_ts, *_) = utils
        logger.info(
            f"Silero VAD ready — frame={_FRAME_SAMPLES} samples "
            f"({_FRAME_MS:.1f} ms), silence={_SILENCE_FRAMES} frames"
        )

        # VAD state
        self._in_speech     = False
        self._silence_count = 0
        self._speech_buffer: list[np.ndarray] = []
        self._pre_roll: deque = deque(maxlen=_PRE_ROLL_FRAMES)
        self._overflow: np.ndarray = np.empty(0, dtype=np.float32)

        # Wake-word detector (initialised after event loop is known)
        self._wake_detector: WakeWordDetector | None = None

    async def start(self) -> None:
        """Open the mic stream and block until the listener is cancelled."""
        self._loop = asyncio.get_running_loop()

        # Wire up the wake-word detector now that we have the loop
        self._wake_detector = WakeWordDetector(
            on_wake=self._on_wake_cb,
            loop=self._loop,
        )

        logger.info(
            f"Opening mic — {_SAMPLE_RATE} Hz, "
            f"{_FRAME_SAMPLES} samples/frame, "
            f"device={config.audio.device_index or 'default'}"
        )
        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=config.audio.channels,
            dtype="float32",
            blocksize=_FRAME_SAMPLES,
            device=config.audio.device_index,
            callback=self._sd_callback,
        ):
            logger.info("🎙️  Mic open.")
            if state.is_sleeping():
                logger.info(f"💤 Sleeping — say '{config.wake_word}' to wake Maya.")
            else:
                logger.info(f"✅ Awake and listening for commands.")
            await asyncio.Event().wait()

    def _sd_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning(f"sounddevice: {status}")

        incoming = indata[:, 0].copy()
        buf = np.concatenate([self._overflow, incoming])

        offset = 0
        while offset + _FRAME_SAMPLES <= len(buf):
            frame = buf[offset : offset + _FRAME_SAMPLES]
            self._process_frame(frame)
            offset += _FRAME_SAMPLES

        self._overflow = buf[offset:]

    def _process_frame(self, frame: np.ndarray) -> None:
        # The wake-word detector is always active; it filters itself when Maya is awake.
        if self._wake_detector:
            self._wake_detector.feed_frame(frame)

        if state.is_sleeping():
            # Discard any half-captured utterance so it cannot leak through after wake.
            if self._in_speech:
                self._in_speech = False
                self._silence_count = 0
                self._speech_buffer = []
                self._pre_roll.clear()
            return

        self._pre_roll.append(frame)
        is_speech = self._vad_frame(frame)

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_buffer = list(self._pre_roll)
                logger.debug("VAD: speech start")
            self._speech_buffer.append(frame)
            self._silence_count = 0

        elif self._in_speech:
            self._speech_buffer.append(frame)
            self._silence_count += 1

            if self._silence_count >= _SILENCE_FRAMES:
                self._in_speech     = False
                self._silence_count = 0
                audio = np.concatenate(self._speech_buffer)
                self._speech_buffer = []
                logger.debug(f"VAD: utterance end ({len(audio)/_SAMPLE_RATE:.2f}s)")
                asyncio.run_coroutine_threadsafe(
                    self._on_speech(audio), self._loop
                )

    def _vad_frame(self, frame: np.ndarray) -> bool:
        tensor = torch.from_numpy(frame).unsqueeze(0)
        with torch.no_grad():
            prob = self._vad_model(tensor, _SAMPLE_RATE).item()
        return prob > 0.5