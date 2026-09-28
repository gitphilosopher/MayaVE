"""
core/processor.py
Single-turn command pipeline for Maya.

This module is the main orchestration layer for one queued user utterance. It
accepts a command item from the queue, exits early if Maya is already asleep,
marks the agent as PROCESSING, classifies intent, updates contextual tracking,
routes the command to the right skill or LLM response, speaks the reply, and then
hands the turn back to the shared lifecycle cleanup path.

The processor is responsible for coordinating the actual turn flow rather than
owning speech synthesis or core state transitions directly. It sends frontend
transcripts and state updates, records user/assistant turns in the conversation
history, and invokes the shared rest-of-turn cleanup so idle/sleep state and
mood behavior remain consistent even when a sleep command lands mid-response.

Important behavior:
- `speaker.speak()` owns the speaking broadcasts; the processor must not emit
  duplicate speaking/behavior messages while the audio is already in flight.
- intent classification runs in the default executor because it is CPU-heavy and
  should not stall the event loop while WS traffic and queued work continue.
- all commands are serialized by the queue, so intent classification is never
  concurrent within a process.
- the success and error tails both go through `turn_lifecycle.rest()`, which re-
  checks the current state and avoids waking Maya back up if a concurrent sleep
  request has already won.
- on a successful turn (never on exception/interruption), the resolved intent
  name is queued to MayaNode as a `mayave.turn_completed` event via
  `services/node/events.py::record_event()`. This never blocks or raises: it
  just drops the event into the local outbox (see that module's docstring),
  which is delivered whenever `NodeSyncManager` next syncs — or sits there
  harmlessly forever if MayaNode integration is disabled/unreachable. No
  conversation text, intent target, or confidence is sent — only the intent
  id itself, matching the narrow `mayave.turn_completed` schema MayaNode and
  MayaVE both register (see services/node/protocol.py).
"""

import asyncio
import logging
import time

from core.speaker import Speaker, _strip_tags
from core.state import state, MayaState
from core.turn_lifecycle import rest as turn_rest
from brain.intent_engine import IntentEngine
from brain.conversation import ConversationManager, context_manager
from brain.router import Router
from services.llm.llm_service import ALREADY_SPOKEN
from services.ws_server import ws_server
from services.node.events import record_event

logger = logging.getLogger(__name__)


class Processor:
    """Handle one queued command from transcription through response and cleanup."""

    def __init__(self, speaker: Speaker):
        self._speaker       = speaker
        self._intent_engine = IntentEngine()
        self._conversation  = ConversationManager()
        self._router        = Router(speaker)

    async def handle(self, item: dict) -> None:
        """Process a queued command item and speak a reply when one is produced."""
        text: str = item["text"]
        if not text:
            return

        if state.is_sleeping():
            # A queued command may still be in flight after a concurrent sleep
            # command has already won the race. Drop it instead of answering while
            # Maya is meant to remain asleep.
            logger.info(f"Dropping queued command — already asleep: '{text[:40]}'")
            return

        t0 = time.perf_counter()
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
            # A whole-utterance fallback is not a topic/entity, so hide it from
            # context tracking to avoid polluting the topic state.
            ctx_intent = intent
            if intent.get("target") == text.strip().lower():
                ctx_intent = {**intent, "target": ""}
            context_manager.observe_user_turn(text, ctx_intent)

            t_dispatch = time.perf_counter()
            response = await self._router.dispatch(intent, text)
            logger.info(f"[TIMING] router.dispatch total: {time.perf_counter()-t_dispatch:.3f}s")

            if response and response != ALREADY_SPOKEN:
                # Strip expression tags for transcript/history while preserving the
                # raw response for speech and behavior output.
                clean, _ = _strip_tags(response)
                if clean:
                    if not intent.get("_no_history"):
                        self._conversation.add_assistant(clean)
                    await ws_server.broadcast_transcript(clean, "maya")

                action = intent.get("action")

                # The speech task is wrapped so a barge-in can cancel the reply
                # itself, not only the underlying audio playback.
                if intent.get("intent") == "greet":
                    # Trigger the wave animation at the start of the audio, not before.
                    await state.run_interruptible(self._speaker.speak(
                        response,
                        on_audio_start=lambda: ws_server.broadcast_animation("wave"),
                    ))
                elif action:
                    await state.run_interruptible(self._speaker.speak(
                        response,
                        on_audio_start=lambda: ws_server.broadcast_animation(action),
                    ))
                else:
                    await state.run_interruptible(self._speaker.speak(response))

            # Re-check the live state so a concurrent sleep command wins over a
            # normal idle return.
            await turn_rest()

            # Best-effort MayaNode sync hand-off — queues into the local outbox
            # and returns immediately regardless of whether MayaNode integration
            # is enabled or reachable (see services/node/events.py). Only the
            # resolved intent id is sent; never raw text, target, or confidence.
            record_event("mayave.turn_completed", {"intent": intent.get("intent", "unknown")})

            logger.info(f"[TIMING] handle() total: {time.perf_counter()-t0:.3f}s")

        except Exception as e:
            logger.error(f"Processor error: {e}", exc_info=True)
            await turn_rest(force_idle=True)