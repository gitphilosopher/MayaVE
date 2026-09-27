"""
core/transcriber.py
Speech-to-text adapter for Maya's audio pipeline.

This module converts a captured microphone frame into a transcription string for
further intent parsing. The transcriber sits downstream of the listener/VAD
pipeline: it receives a float32 numpy array at the configured sample rate,
converts it to the 16-bit PCM format expected by `speech_recognition`, and
passes it to Google Speech Recognition via `recognize_google()`.

The implementation intentionally runs the blocking network request in an executor
so the async turn loop remains responsive while a microphone capture is being
processed. This is a thin compatibility layer around the online Google STT API:
network availability is required, but the rest of Maya does not need to know the
low-level audio conversion details.
"""

import asyncio
import logging

import numpy as np
import speech_recognition as sr

from config.settings import config

logger = logging.getLogger(__name__)


class Transcriber:
    """Convert raw microphone frames into text using the online Google STT backend."""

    def __init__(self):
        """Create the recognizer and log the configured backend for operator visibility."""
        self._recogniser = sr.Recognizer()
        logger.info("Transcriber ready — Google Speech Recognition (online, free).")

    async def transcribe(self, audio: np.ndarray) -> str | None:
        """Transcribe a float32 mono frame while keeping the async event loop responsive."""
        if audio is None or len(audio) == 0:
            return None

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._recognise, audio)
        return text

    def _recognise(self, audio: np.ndarray) -> str | None:
        """Perform the blocking Google recognition work in a worker thread."""
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

        audio_data = sr.AudioData(
            pcm,
            sample_rate=config.audio.sample_rate,
            sample_width=2,
        )

        try:
            text = self._recogniser.recognize_google(
                audio_data,
                language=config.stt.language,
            )
            text = text.strip()
            logger.info(f"Transcribed: '{text}'")
            return text

        except sr.UnknownValueError:
            logger.debug("Google STT: could not understand audio.")
            return None

        except sr.RequestError as e:
            logger.error(f"Google STT request failed: {e}")
            return None