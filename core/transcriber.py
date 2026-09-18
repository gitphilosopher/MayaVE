"""
core/transcriber.py
Phase 3: Speech → Text using Google Speech Recognition (online, free).

Pipeline:
  numpy float32 audio (16 kHz mono)  ← from Silero VAD in listener.py
      ↓
  convert to 16-bit PCM bytes
      ↓
  wrap in speech_recognition.AudioData
      ↓
  recognize_google()  (sends to Google STT API, no key required)
      ↓
  cleaned text string

Note: requires an active internet connection.
"""

import asyncio
import logging

import numpy as np
import speech_recognition as sr

from config.settings import config

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self):
        self._recogniser = sr.Recognizer()
        logger.info("Transcriber ready — Google Speech Recognition (online, free).")

    async def transcribe(self, audio: np.ndarray) -> str | None:
        """
        Transcribe a float32 numpy array captured at 16 kHz mono.
        Runs the blocking Google API call in an executor thread so the
        asyncio event loop is never blocked.
        Returns stripped lowercase text, or None on failure.
        """
        if audio is None or len(audio) == 0:
            return None

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._recognise, audio)
        return text

    def _recognise(self, audio: np.ndarray) -> str | None:
        """Blocking recognition — called inside an executor thread."""

        # Convert float32 [-1, 1] → int16 PCM bytes (what SpeechRecognition expects)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

        audio_data = sr.AudioData(
            pcm,
            sample_rate=config.audio.sample_rate,
            sample_width=2,   # 16-bit = 2 bytes
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