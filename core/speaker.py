"""
core/speaker.py
Text → Speech using Kokoro TTS (fully offline, Apache 2.0).

Dual-output mode:
  - LOCAL  (config.tts.output = "local")  → sounddevice plays on the machine
  - AVATAR (config.tts.output = "avatar") → WAV bytes broadcast over WebSocket
                                             to the browser avatar
  - BOTH   (config.tts.output = "both")   → local + avatar simultaneously

Expression tags:
  Responses can contain [expression] tags e.g. "[happy] Done senpai!"
  speaker.py strips tags from TTS text, broadcasts the expression to the
  avatar before playing, then resets to Maya's current mood baseline
  (core/mood.py) after playback — not a hardcoded "neutral" — so an
  active mood (e.g. still angry from earlier) persists between skill
  responses instead of visibly resetting every line.

Behavioral Engine integration (core/behavior_engine.py):
  The raw tag is no longer broadcast directly — it's composed via
  behavior_engine.compose() into a communicative-intent packet (blended
  primary/secondary emotion + gaze) and sent via broadcast_behavior().
  Same call sites, same timing relative to audio — only what's sent over
  the wire changed.

Bug fixes:
  - avatar mode now waits for audio_done before returning, preventing overlap
    when processor.py calls speak() for consecutive skill responses
  - broadcast_state("speaking") moved here so processor.py doesn't double-fire it
  - speak() now ALSO broadcasts "idle" (+ baseline expression) back over the
    WebSocket once it's done, instead of only flipping the internal
    core/state.py FSM to IDLE. Every OTHER caller of speak() — processor.py
    already re-broadcast idle itself after a skill/LLM turn, but main.py's
    startup greeting, its wake-up ("I'm here, how can I help?") and
    go-to-sleep lines, and timer.py's alert all call speak() directly and
    had no such follow-up. Without this, the frontend's `_currentBackendState`
    (see frontend/js/avatar.js's idle-fidget gate) would get stuck on
    whatever it last saw — "speaking", or even null if speak() hadn't
    broadcast anything yet — and idle fidgets could never fire until the
    first real voice command flowed through processor.handle(). Doing it
    once here means every caller gets it for free instead of each call site
    needing to remember.
  - speak() restores SLEEPING (instead of forcing IDLE) when it was
    SLEEPING on entry, so the go-to-sleep goodbye line no longer wakes
    Maya back up.
"""

import asyncio
import io
import logging
import re
import wave

import numpy as np
import sounddevice as sd
from kokoro import KPipeline

from config.settings import config
from core.state import state, MayaState
from core.mood import mood_manager
from core.behavior_engine import behavior_engine
from services.llm.llm_service import (
    _enhance_prosody, _build_kokoro_pipeline, _log_cuda_memory, _run_kokoro,
)

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24_000   # Kokoro always outputs 24 kHz

_TAG_RE = re.compile(r'\[(\w+)\]')
_VALID_EXPRESSIONS = {
    "happy", "sad", "angry", "surprised", "relaxed", "neutral", "excited"
}


def _strip_tags(text: str) -> tuple[str, str]:
    """
    Extract the first valid [expression] tag and return (clean_text, expression).
    All tags are removed from clean_text so Kokoro doesn't speak them.
    Only tags whose names are in _VALID_EXPRESSIONS are treated as expression tags —
    timestamp brackets like [2026-06-15 12:00] are left untouched by the expression
    picker but still stripped from TTS text (Kokoro shouldn't speak brackets).
    """
    text = text.strip()
    expression = "neutral"
    for match in _TAG_RE.finditer(text):
        tag = match.group(1).lower()
        if tag in _VALID_EXPRESSIONS:
            expression = tag
            break  # use first valid expression tag found
    clean = _TAG_RE.sub("", text).strip()
    return clean, expression


def _build_voice(pipeline: KPipeline):
    primary = config.tts.voice
    blend   = getattr(config.tts, "voice_blend", "")
    ratio   = getattr(config.tts, "blend_ratio", 0.0)

    if blend and 0.0 < ratio < 1.0:
        try:
            v1 = pipeline.load_voice(primary)
            v2 = pipeline.load_voice(blend)
            mixed = (1.0 - ratio) * v1 + ratio * v2
            logger.info(
                f"Kokoro voice blend: {primary} ({1-ratio:.0%}) + "
                f"{blend} ({ratio:.0%})"
            )
            return mixed
        except Exception as e:
            logger.warning(f"Voice blend failed ({e}), falling back to {primary}")
    return primary


def _numpy_to_wav(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Convert float32 numpy array → WAV bytes (in-memory, no temp file)."""
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class Speaker:
    def __init__(self):
        lang = getattr(config.tts, "lang_code", "a")
        self._output = getattr(config.tts, "output", "both")
        logger.info(
            f"Loading Kokoro TTS — voice='{config.tts.voice}'  "
            f"blend='{getattr(config.tts, 'voice_blend', '')}'  "
            f"output='{self._output}'"
        )
        self._pipeline = _build_kokoro_pipeline(lang)
        self._voice    = _build_voice(self._pipeline)
        self._speed    = getattr(config.tts, "speed", 1.1)
        logger.info("Kokoro TTS ready — fully offline.")

    def warmup(self) -> None:
        """
        Blocking dummy synthesis — call once via an executor at startup
        (see main.py) so first-inference cost (CUDA kernel warm-up, etc.)
        isn't paid on the first real line spoken (e.g. the greeting).
        """
        try:
            self._synthesise("Hello.")
            logger.info("Speaker Kokoro pipeline warmed up.")
            _log_cuda_memory("Speaker pipeline warm, after first synthesis")
        except Exception as e:
            logger.warning(f"Speaker warmup failed (non-fatal): {e}")

    # ── Public API ────────────────────────────────────────────────────

    async def speak(self, text: str, on_audio_start=None) -> None:
        """
        Synthesise and play text.
        on_audio_start: optional async callable fired right before audio is
                        broadcast — use this to trigger animations that should
                        be simultaneous with speech, not before synthesis.
        """
        if not text or not text.strip():
            return

        clean, expression = _strip_tags(text)
        if not clean:
            return

        # Feed this response's expression into the persistent mood layer
        # as a single-tag turn (a skill response only ever resolves to
        # one expression — see _strip_tags above).
        mood_manager.observe_turn([expression])

        clean = _enhance_prosody(clean, expression)

        # Snapshot before SPEAKING overwrites it — the go-to-sleep line is
        # spoken while SLEEPING and must not wake Maya back up.
        was_sleeping = state.is_sleeping()

        await state.set(MayaState.SPEAKING)
        logger.info(f"Speaking [{expression}]: '{clean}'")

        from services.ws_server import ws_server

        try:
            loop = asyncio.get_running_loop()
            audio = await self._synthesise_guarded(clean)
            if audio is None:
                return

            if self._output in ("avatar", "both"):
                await ws_server.broadcast_behavior(behavior_engine.compose(expression, source="skill"))
                await ws_server.broadcast_state("speaking")
                wav_bytes = await loop.run_in_executor(None, _numpy_to_wav, audio)

                # Fire the callback (e.g. wave animation) right as audio lands
                if on_audio_start:
                    await on_audio_start()

                await ws_server.broadcast_audio(wav_bytes)
                await ws_server.wait_for_audio_done()
                await ws_server.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))

            if self._output in ("local", "both"):
                if on_audio_start:
                    await on_audio_start()
                await loop.run_in_executor(None, self._play_blocking, audio)

        except Exception as e:
            logger.error(f"Speaker error: {e}", exc_info=True)
        finally:
            await state.set(MayaState.SLEEPING if was_sleeping else MayaState.IDLE)
            # Tell the browser too — not just the internal FSM. Every
            # caller of speak() (main.py's greeting/wake/sleep lines,
            # timer.py's alert, processor.py's skill responses) gets this
            # for free now, so the frontend's idle-fidget gate
            # (avatar.js's _currentBackendState) always eventually sees
            # "idle" instead of getting stuck on "speaking" — or on
            # whatever it initialised to, if speak() hadn't broadcast
            # anything yet at all.
            await ws_server.broadcast_state("idle")
            await ws_server.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))

    # ── Private ───────────────────────────────────────────────────────

    async def _synthesise_guarded(self, text: str) -> np.ndarray | None:
        """
        Runs _synthesise via llm_service._run_kokoro — a timeout-bounded
        daemon-thread call, not the shared executor, so a stuck native
        Kokoro/espeak call (a) can't stall this pipeline indefinitely and
        (b) can't block interpreter shutdown either. On timeout this
        Speaker's own pipeline instance is rebuilt too (separate from
        llm_service's module-level one) so the next call gets a fresh
        Kokoro/espeak backend instead of retrying the same stuck one.
        """
        audio = await _run_kokoro(self._synthesise, text)
        if audio is None:
            try:
                lang = getattr(config.tts, "lang_code", "a")
                self._pipeline = _build_kokoro_pipeline(lang)
                self._voice    = _build_voice(self._pipeline)
            except Exception as e:
                logger.error(f"Speaker pipeline rebuild failed: {e}")
        return audio

    def _synthesise(self, text: str) -> np.ndarray | None:
        """Blocking Kokoro synthesis — called in executor."""
        chunks = [
            audio for _, _, audio in
            self._pipeline(text, voice=self._voice, speed=self._speed)
            if audio is not None and len(audio) > 0
        ]
        if not chunks:
            logger.warning("Kokoro returned no audio chunks.")
            return None
        return np.concatenate(chunks).astype(np.float32)

    @staticmethod
    def _play_blocking(audio: np.ndarray) -> None:
        sd.play(audio, samplerate=_SAMPLE_RATE)
        sd.wait()