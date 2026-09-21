"""
core/processor.py
Orchestrates a single command: text → intent → skill → speak.

WebSocket broadcasts:
  - state changes (listening / processing / idle) → avatar expression
  - transcripts (user + maya turns) → transcript overlay

Note: broadcast_state("speaking") and broadcast_expression() are handled
inside speaker.speak() — processor must NOT call them separately or they
double-fire and race with the audio.

Mood integration (core/mood.py):
  Whenever we go idle, we also broadcast Maya's current mood baseline
  expression so the avatar rests on "angry"/"sad" instead of snapping
  back to neutral/relaxed while a mood is still active.

Context integration (Stage 2 — brain/conversation.py):
  After intent classification, context_manager.observe_user_turn() updates
  topic/state/open-loop tracking for every command (not just LLM-routed
  ones). This is state tracking only — conversation.add_user() above it
  remains the single place user turns are written to history.

Event-loop rule:
  IntentEngine.classify() runs a PyTorch + TensorFlow ensemble forward
  pass — tens of milliseconds of CPU that would otherwise stall the loop
  (WS pings, audio_done handling, timers). It runs in the default
  executor. Commands are serialized by queue_manager, so classify() is
  never called concurrently.
"""

import asyncio
import logging
import time

from core.speaker import Speaker, _strip_tags
from core.state import state, MayaState
from core.mood import mood_manager
from core.behavior_engine import behavior_engine
from brain.intent_engine import IntentEngine
from brain.conversation import ConversationManager, context_manager
from brain.router import Router
from services.llm.llm_service import ALREADY_SPOKEN
from services.ws_server import ws_server

logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, speaker: Speaker):
        self._speaker       = speaker
        self._intent_engine = IntentEngine()
        self._conversation  = ConversationManager()
        self._router        = Router(speaker)

    async def handle(self, item: dict) -> None:
        """Entry point called by QueueManager for each command."""
        t0 = time.perf_counter()
        text: str = item["text"]
        if not text:
            return

        logger.info(f"[TIMING][TTFA] user command start t={t0:.3f} text='{text[:40]}'")

        await state.set(MayaState.PROCESSING)
        await ws_server.broadcast_state("processing")
        await ws_server.broadcast_transcript(text, "user")

        logger.info(f"Processing: '{text}'")
        self._conversation.add_user(text)

        try:
            loop = asyncio.get_running_loop()
            intent = await loop.run_in_executor(None, self._intent_engine.classify, text)
            intent["_t_cmd_start"] = t0
            logger.info(f"[TIMING] intent_classify: {time.perf_counter()-t0:.3f}s")
            logger.info(
                f"Intent: {intent['intent']} "
                f"({intent['confidence']:.2f} via {intent['model']})"
            )
            # _extract_target falls back to the whole utterance when no trigger
            # strips — that's not a topic/entity, so hide it from context state.
            ctx_intent = intent
            if intent.get("target") == text.strip().lower():
                ctx_intent = {**intent, "target": ""}
            context_manager.observe_user_turn(text, ctx_intent)

            t_dispatch = time.perf_counter()
            response = await self._router.dispatch(intent, text)
            logger.info(f"[TIMING] router.dispatch total: {time.perf_counter()-t_dispatch:.3f}s")

            if response and response != ALREADY_SPOKEN:
                # History/transcript get tag-free text; replies flagged
                # _no_history (llm_service error strings) stay out of history.
                clean, _ = _strip_tags(response)
                if clean:
                    if not intent.get("_no_history"):
                        self._conversation.add_assistant(clean)
                    await ws_server.broadcast_transcript(clean, "maya")

                # Set by skills/system/perform_action.py on the intent dict.
                action = intent.get("action")

                if intent.get("intent") == "greet":
                    # Fire wave exactly when audio starts playing, not before synthesis
                    await self._speaker.speak(
                        response,
                        on_audio_start=lambda: ws_server.broadcast_animation("wave"),
                    )
                elif action:
                    await self._speaker.speak(
                        response,
                        on_audio_start=lambda: ws_server.broadcast_animation(action),
                    )
                else:
                    await self._speaker.speak(response)

            await ws_server.broadcast_state("idle")
            await ws_server.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))
            logger.info(f"[TIMING] handle() total: {time.perf_counter()-t0:.3f}s")

        except Exception as e:
            logger.error(f"Processor error: {e}", exc_info=True)
            await ws_server.broadcast_state("idle")
            await ws_server.broadcast_behavior(behavior_engine.compose(mood_manager.baseline_expression(), source="idle"))
            await state.set(MayaState.IDLE)