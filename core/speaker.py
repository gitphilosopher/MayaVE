"""
core/speaker.py
Text-to-speech facade for Maya.

This module owns the offline Kokoro TTS pipeline and the output routing used by
Maya's responses. It accepts text, strips expression markers such as [happy],
converts the active expression into the avatar behavior packet, and routes the
result to the configured output mode:
  - local: sounddevice playback on this machine
  - avatar: WebSocket audio plus avatar state/behavior updates
  - both: local playback and avatar output together

The speaker is integrated with the shared state machine and mood system:
- `state.set(MayaState.SPEAKING)` and the final state transition keep the rest of
  the agent informed about whether Maya is currently speaking or has returned to
  idle/sleep.
- expression tags are fed into the persistent mood layer and turned into a
  behavior packet via `behavior_engine.compose()` before the speech output is
  sent.
- the end of a turn broadcasts the matching idle/sleeping frontend state so the
  avatar can resume its normal idle motion without getting stuck in a speaking
  state.

The synthesis path is guarded against stuck native Kokoro/espeak calls by
running the blocking work in a timeout-aware helper and rebuilding the local
pipeline only after a real timeout. The public API is intentionally small:
`Speaker.warmup()` for startup preparation and `Speaker.speak()` for playback and
state sync.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import io
import logging
import re
import wave

import numpy as np
import sounddevice as sd

from config.settings import config
from core.state import state, MayaState
from core.mood import mood_manager
from core.behavior_engine import behavior_engine
from services.llm.llm_service import (
    _enhance_prosody, _log_cuda_memory, _run_kokoro,
    get_shared_kokoro, reset_shared_kokoro, _kokoro_lock,
)

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24_000   # Kokoro always outputs 24 kHz

_TAG_RE = re.compile(r'\[(\w+)\]')
_VALID_EXPRESSIONS = {
    "happy", "sad", "angry", "surprised", "relaxed", "neutral", "excited"
}

# Marks "synthesis ran but produced no audio" (vs None = _run_kokoro timeout).
_NO_AUDIO = object()


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
    """Coordinate Kokoro synthesis, output routing, and state updates for speech."""

    def __init__(self):
        lang = getattr(config.tts, "lang_code", "a")
        self._output = getattr(config.tts, "output", "both")
        logger.info(
            f"Loading Kokoro TTS — voice='{config.tts.voice}'  "
            f"blend='{getattr(config.tts, 'voice_blend', '')}'  "
            f"output='{self._output}'"
        )
        self._pipeline, self._voice = get_shared_kokoro(lang)
        self._speed    = getattr(config.tts, "speed", 1.1)
        logger.info("Kokoro TTS ready — fully offline.")

    def warmup(self) -> None:
        """Prime the Kokoro pipeline once during startup to avoid a cold first reply."""
        try:
            self._synthesise("Hello.")
            logger.info("Speaker Kokoro pipeline warmed up.")
            _log_cuda_memory("Speaker pipeline warm, after first synthesis")
        except Exception as e:
            logger.warning(f"Speaker warmup failed (non-fatal): {e}")

    async def speak(self, text: str, on_audio_start=None) -> None:
        """Synthesize a response, emit the matching avatar state, and return to idle/sleep when done."""
        if not text or not text.strip():
            return

        clean, expression = _strip_tags(text)
        if not clean:
            return

        # The active expression becomes a single-turn mood observation for the
        # persistent emotional baseline; a response can only carry one tag here.
        mood_manager.observe_turn([expression])

        clean = _enhance_prosody(clean, expression)

        # Keep the sleep-start snapshot before the state flips to SPEAKING; the
        # sleep goodbye path must end back in SLEEPING even though the state is
        # already speaking for the full duration of the turn.
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
            # Sleep wins over a stale wake snapshot: either this turn started as a
            # goodbye line, or some other concurrent turn put Maya to sleep while
            # playback was still active. In both cases, the final state must remain
            # asleep rather than returning to idle.
            sleeping_now = was_sleeping or state.is_sleeping()
            await state.set(MayaState.SLEEPING if sleeping_now else MayaState.IDLE)
            if sleeping_now:
                await ws_server.broadcast_state("sleeping")
            else:
                await ws_server.broadcast_state("idle")
                await ws_server.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))

    async def _synthesise_guarded(self, text: str) -> np.ndarray | None:
        """
        Runs _synthesise via llm_service._run_kokoro — a timeout-bounded
        daemon-thread call, not the shared executor, so a stuck native
        Kokoro/espeak call (a) can't stall this pipeline indefinitely and
        (b) can't block interpreter shutdown either. Only on a real
        timeout is this Speaker's own pipeline instance rebuilt (off the
        event loop; separate from llm_service's module-level one, which
        is left alone) so the next call gets a fresh Kokoro/espeak
        backend instead of retrying the same stuck one. A normal "no
        audio produced" result does not trigger a rebuild.
        """
        def _job():
            audio = self._synthesise(text)
            return _NO_AUDIO if audio is None else audio

        # on_timeout no-op: this Speaker's pipeline is rebuilt below, and
        # llm_service's own pipeline wasn't the one that got stuck.
        result = await _run_kokoro(_job, on_timeout=lambda: None)
        if result is None:   # _run_kokoro returns None only on timeout
            try:
                lang = getattr(config.tts, "lang_code", "a")

                def _rebuild():
                    reset_shared_kokoro()
                    return get_shared_kokoro(lang)

                self._pipeline, self._voice = await asyncio.get_running_loop().run_in_executor(None, _rebuild)
            except Exception as e:
                logger.error(f"Speaker pipeline rebuild failed: {e}")
            return None
        return None if result is _NO_AUDIO else result

    def _synthesise(self, text: str, trace_id: str | None = None) -> np.ndarray | None:
        """Blocking Kokoro synthesis — called in executor."""
        try:
            from services.tracer import log_event
            if trace_id:
                log_event(trace_id, "tts_synthesis_start", {"text_len": len(text)})
        except Exception:
            pass

        with _kokoro_lock:
            pipeline, voice = get_shared_kokoro()
            self._pipeline, self._voice = pipeline, voice
            chunks = [
                audio for _, _, audio in
                pipeline(text, voice=voice, speed=self._speed)
                if audio is not None and len(audio) > 0
            ]

        try:
            from services.tracer import log_event
            if trace_id:
                log_event(trace_id, "tts_synthesis_end", {"chunks": len(chunks)})
        except Exception:
            pass

        if not chunks:
            logger.warning("Kokoro returned no audio chunks.")
            return None
        res = np.concatenate(chunks).astype(np.float32)
        try:
            from services.tracer import log_event
            if trace_id:
                log_event(trace_id, "tts_audio_ready", {"samples": len(res), "sr": _SAMPLE_RATE})
        except Exception:
            pass
        return res

    @staticmethod
    def _play_blocking(audio: np.ndarray) -> None:
        sd.play(audio, samplerate=_SAMPLE_RATE)
        sd.wait()