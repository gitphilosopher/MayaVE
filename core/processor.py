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
from brain.router.dispatch import Router
from brain.router.ir import to_legacy_intent
from config.settings import config
from services.llm.llm_service import ALREADY_SPOKEN, next_diag_turn_id
from services.ws_server import ws_server
from services.node.events import record_event

logger = logging.getLogger(__name__)


class Processor:
    """Handle one queued command from transcription through response and cleanup."""

    def __init__(self, speaker: Speaker):
        self._speaker       = speaker
        self._conversation  = ConversationManager()
        self._router        = Router(speaker)   # also injects the speaker into timer.py
        self._intent_engine = IntentEngine()
        self._understander  = self._build_understander(self._intent_engine)
        logger.info("Routing: %s", "Hybrid Router" if self._understander else "legacy IntentEngine")

    async def _classify(self, text: str) -> dict:
        """Legacy-shaped intent dict; carries intent["_ir"] on the hybrid path."""
        if self._understander is not None:
            try:
                return to_legacy_intent(await self._understander.understand(text))
            except Exception:
                logger.error("Understander failed — falling back to legacy classifier.", exc_info=True)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._intent_engine.classify, text)

    @staticmethod
    def _build_understander(engine):
        """CommandUnderstander when config.router.backend == "hybrid", else None (legacy path)."""
        if config.router.backend != "hybrid":
            return None
        try:
            from brain.embeddings import get_embedding_provider
            from brain.router import guards
            from brain.router.command_vector_store import CommandVectorStore
            from brain.router.confidence import ConfidenceThresholds
            from brain.router.llm_fallback import route as llm_route
            from brain.router.registry import CommandRegistry, load_specs
            from brain.router.semantic_router import SemanticRouter
            from brain.router.understand import CommandUnderstander

            registry = CommandRegistry(load_specs())
            provider_type = getattr(config.router, "embedding_provider", "local")
            store = CommandVectorStore(config.router.command_vector_db_path, provider=provider_type)
            embedder = get_embedding_provider(provider_type)
            model_id = getattr(embedder, "model_id", "")

            if store.is_stale(registry.specs, model_id):
                if provider_type == "local" and hasattr(embedder, "embed_sync"):
                    logger.info("Command vector corpus stale or unseeded — auto-seeding local BGE index...")
                    store.reseed(registry.specs, embedder.embed_sync, model_id)
                else:
                    logger.warning("Command vector corpus stale/unseeded — semantic tier inert until "
                                "`python -m brain.router.eval_router --seed` is run.")
            thresholds = ConfidenceThresholds(
                min_similarity=config.router.min_similarity,
                min_margin=config.router.min_margin,
                low_similarity_floor=config.router.low_similarity_floor,
            )
            semantic = SemanticRouter(embedder, store, registry).retrieve
            return CommandUnderstander(engine, registry, guard_fn=guards.check, semantic=semantic,
                                    semantic_thresholds=thresholds, llm_route=llm_route)
        except Exception:
            logger.error("Hybrid router unavailable — using the legacy classifier.", exc_info=True)
            return None

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

        turn_id = next_diag_turn_id()
        t0 = time.perf_counter()
        logger.info(f"[DIAG][TURN={turn_id}] user command start t={t0:.3f} text='{text[:40]}'")
        logger.info(f"[TIMING][TTFA] user command start t={t0:.3f} text='{text[:40]}'")

        await state.set(MayaState.PROCESSING)
        await ws_server.broadcast_state("processing")
        await ws_server.broadcast_transcript(text, "user")

        logger.info(f"Processing: '{text}'")
        self._conversation.add_user(text)
        t_pre = time.perf_counter()

        try:
            t_classify_start = time.perf_counter()
            intent = await self._classify(text)
            intent["_t_cmd_start"] = t0
            intent["_turn_id"] = turn_id
            t_classify = time.perf_counter() - t_classify_start
            logger.info(f"[TIMING] intent_classify: {t_classify:.3f}s")
            logger.info(
                f"Intent: {intent['intent']} "
                f"({intent['confidence']:.2f} via {intent['model']})"
            )
            ir = intent.get("_ir")
            if ir is not None:
                logger.info(f"IR: status={ir.status.value} key={ir.key} source={ir.source} reason={ir.reason!r}")
            # A whole-utterance fallback is not a topic/entity, so hide it from
            # context tracking to avoid polluting the topic state.
            # The IR's target is the un-lowercased (possibly context-rewritten) utterance for
            # raw-target ops, so compare case-insensitively against both raw and effective text.
            ctx_intent = intent
            whole = {text.strip().lower(), str(intent.get("raw") or "").strip().lower()}
            if str(intent.get("target") or "").strip().lower() in whole:
                ctx_intent = {**intent, "target": ""}
            t_ctx_start = time.perf_counter()
            context_manager.observe_user_turn(text, ctx_intent)
            t_ctx = time.perf_counter() - t_ctx_start

            t_dispatch_start = time.perf_counter()
            response = await self._router.dispatch(intent, text)
            t_dispatch = time.perf_counter() - t_dispatch_start
            logger.info(f"[TIMING] router.dispatch total: {t_dispatch:.3f}s")

            t_speak_start = time.perf_counter()
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
            t_speak = time.perf_counter() - t_speak_start

            # Re-check the live state so a concurrent sleep command wins over a
            # normal idle return.
            t_rest_start = time.perf_counter()
            await turn_rest()
            t_rest = time.perf_counter() - t_rest_start

            # Best-effort MayaNode sync hand-off — queues into the local outbox
            # and returns immediately regardless of whether MayaNode integration
            # is enabled or reachable (see services/node/events.py). Only the
            # resolved intent id is sent; never raw text, target, or confidence.
            t_outbox_start = time.perf_counter()
            record_event("mayave.turn_completed", {"intent": intent.get("intent", "unknown")})
            t_outbox = time.perf_counter() - t_outbox_start

            t_total = time.perf_counter() - t0
            t_pipeline = t_total - t_speak
            logger.info(
                f"[TIMING] handle() turn complete: {t_total:.3f}s "
                f"(speech_and_playback={t_speak:.3f}s, pipeline={t_pipeline:.3f}s)"
            )
            logger.info(f"[TIMING] handle() total: {t_total:.3f}s")

        except Exception as e:
            logger.error(f"Processor error: {e}", exc_info=True)
            await turn_rest(force_idle=True)