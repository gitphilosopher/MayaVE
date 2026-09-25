"""
core/listener.py
Phase 1 + 2: Continuous microphone capture with Silero VAD + wake-word gate.

Pipeline:
  sounddevice stream (raw PCM, 16 kHz mono)
      ↓
  every frame → WakeWordDetector.feed_frame()   [always, even while sleeping]
      ↓
  if SLEEPING → detector only; VAD speech pipeline is skipped
      ↓
  if AWAKE → Silero VAD speech detection
      ↓
  on utterance end → on_speech(audio) callback

Barge-in lives in main.py, not here:
  The VAD pipeline below runs continuously whenever Maya isn't
  SLEEPING — including while she's SPEAKING — exactly as it always
  has. Speech captured while she's talking is still delivered via the
  normal on_speech() callback once the user stops talking.

  Deciding whether that speech should actually CUT HER OFF requires
  knowing what was said — specifically, whether she was called by name
  ("hey maya", "hello maya", "maya", ...) — and that requires the full
  utterance to be transcribed first. Since transcription is a Google
  STT round trip that only main.py's on_speech() has access to (via
  core/transcriber.py), the barge-in decision itself is made there
  (see main.py's on_speech(), which checks
  core.wake_word.contains_wake_word() against the transcribed text and
  calls state.interrupt() only when it matches). Listener.py stays a
  pure audio-capture component with no opinion on interrupt policy.

Frame-size fix (Windows):
  Silero VAD requires sample_rate / frame_samples > 31.25.
  At 16 000 Hz → minimum 512 samples (32 ms).
  _FRAME_SAMPLES is clamped to 512 regardless of config.audio.chunk_ms.

Startup log fix (Batch 4):
  Maya starts AWAKE (main.py sets IDLE before the startup greeting), so
  the mic-open log line now reflects the FSM's actual state instead of
  unconditionally claiming she's sleeping — this used to be printed
  regardless of state, a stale leftover from before "start awake" was
  introduced (see docs/CHANGELOG.md's "Sleep not persisting" fix).
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

# ── Frame sizing ──────────────────────────────────────────────────────────────
_SAMPLE_RATE     = config.audio.sample_rate
_FRAME_SAMPLES   = max(512, int(_SAMPLE_RATE * config.audio.chunk_ms / 1000))
_FRAME_MS        = _FRAME_SAMPLES / _SAMPLE_RATE * 1000
_SILENCE_FRAMES  = max(1, int(config.audio.silence_ms  / _FRAME_MS))
_PRE_ROLL_FRAMES = max(1, int(config.audio.pre_roll_ms / _FRAME_MS))


class Listener:
    def __init__(self, on_speech: AudioCallback, on_wake: Callable[[], Awaitable[None]]):
        """
        Args:
            on_speech : async callback → receives float32 numpy utterance array
            on_wake   : async callback → fired when wake word is detected
        """
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

    # ── Public ────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Open mic stream and run until cancelled."""
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

    # ── Private ───────────────────────────────────────────────────────────────

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
        # Always feed wake-word detector (it self-gates when not sleeping)
        if self._wake_detector:
            self._wake_detector.feed_frame(frame)

        # Skip VAD speech pipeline while sleeping
        if state.is_sleeping():
            # Drop a half-captured utterance so it isn't delivered after wake.
            if self._in_speech:
                self._in_speech     = False
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