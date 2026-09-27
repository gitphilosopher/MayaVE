"""
main.py
MayaVE - Maya Virtual Entity runtime entry point.

This module assembles and starts the assistant's long-lived voice runtime. It
configures logging and offline model behavior, creates the listener,
transcriber, speaker, and processor, starts the WebSocket and optional
MayaNode synchronization services, warms the local Ollama and Kokoro models,
and then runs the listener and command queue concurrently.

The state machine distinguishes sleeping, idle, listening, processing, and
speaking phases. While sleeping, only the wake-word path can activate Maya.
When awake, commands flow from microphone input through transcription and the
queue processor. A request containing Maya's wake word can interrupt speech;
ordinary speech heard over a response is queued instead. Sleep commands stop
active work before the goodbye response, and all direct speech uses the
interruptible state path so cancellation stops both the task and its audio.

The listening watchdog repairs a rare stranded LISTENING state when no command
can advance it. Its state observer may be called synchronously from an audio
callback thread, so it only schedules work on the asyncio event loop. The
MayaNode sync task is isolated from the voice pipeline and is safe to start
even when node integration is disabled or the node is unavailable.

Run this module directly to start Maya. Configuration is supplied through
``config.settings.config`` and the collaborating runtime components.
"""

# Must run before ANY import that can pull in kokoro / huggingface_hub
# (core.speaker and services.llm.llm_service both do, directly or not):
# huggingface_hub reads HF_HUB_OFFLINE once, at import time. Setting it in
# llm_service.py after `from kokoro import KPipeline` was too late.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import datetime
import logging
import sys
from logging.handlers import RotatingFileHandler

import httpx
import sounddevice as sd

from config.settings import config
from core.state import state, MayaState
from core.queue_manager import queue_manager
from core.transcriber import Transcriber
from core.speaker import Speaker
from core.processor import Processor
from core.listener import Listener
from core.wake_word import contains_wake_word, is_sleep_command
from core.behavior_engine import behavior_engine
from core.mood import mood_manager
from services.ws_server import ws_server
from services.llm.llm_service import warmup as llm_warmup
from services.llm.ollama_lifecycle import chat_keep_alive
from services.node.sync_manager import node_sync_manager
from brain.embeddings import _resolve_embedding_device, _embedding_gpu_options, describe_ollama_models

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(config.log_dir, exist_ok=True)   # FileHandler fails if logs/ is missing
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            f"{config.log_dir}/maya.log",
            maxBytes=5_000_000, backupCount=3, encoding="utf-8",
        ),
    ],
)
# Third-party libs log every HTTP request / WS connection at INFO — warnings only.
for _noisy in ("httpx", "httpcore", "websockets", "urllib3", "tensorflow"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("main")

_U = config.user_name

# LISTENING should transition to PROCESSING for a queued command or back to
# IDLE for empty input. The watchdog handles rare cases where neither occurs,
# such as an interrupt without follow-up speech; on_speech() handles a full
# queue immediately.
_LISTENING_WATCHDOG_S = 15.0
_listening_watchdog_task: asyncio.Task | None = None


def _on_state_change(old: MayaState, new: MayaState) -> None:
    """
    core/state.py observer — fired synchronously on every transition, and
    set_sync() can call it from the sounddevice callback thread, so all
    real work is handed to the event loop via call_soon_threadsafe rather
    than touched here directly (safe to call even when already on the
    loop thread).
    """
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.call_soon_threadsafe(_handle_state_change, new)
    except RuntimeError:
        pass   # loop not running (e.g. during shutdown) — nothing to schedule


def _handle_state_change(new: MayaState) -> None:
    """Restart the watchdog whenever the state enters or leaves LISTENING."""
    global _listening_watchdog_task
    if _listening_watchdog_task is not None:
        _listening_watchdog_task.cancel()
        _listening_watchdog_task = None
    if new == MayaState.LISTENING:
        _listening_watchdog_task = asyncio.create_task(
            _listening_watchdog(), name="listening-watchdog"
        )


async def _listening_watchdog() -> None:
    """Return a stranded LISTENING state to IDLE after the recovery timeout."""
    try:
        await asyncio.sleep(_LISTENING_WATCHDOG_S)
    except asyncio.CancelledError:
        return
    if state.current == MayaState.LISTENING:
        logger.warning(
            f"Listening watchdog: stuck in LISTENING for {_LISTENING_WATCHDOG_S:.0f}s "
            "with no follow-up (a dropped command, or an interrupt with nothing queued "
            "after it) — resetting to idle."
        )
        await _end_listening()


# ── Ollama warm-up ────────────────────────────────────────────────────────────

async def _warmup_ollama() -> None:
    """
    Fire a silent 1-token request so Ollama loads the model into RAM
    during Maya's startup. First real question will then respond instantly.
    Sends the same keep_alive as real chat requests — without it the model
    would expire on Ollama's 5-minute default if the first command is late.
    """
    url = f"{config.llm.base_url.rstrip('/')}/api/chat"
    payload = {
        "model":    config.llm.model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream":   False,
        "keep_alive": chat_keep_alive(),
        "options":  {"num_predict": 1},   # generate just 1 token — fast
    }
    try:
        logger.info(f"Pre-warming Ollama model '{config.llm.model}' (keep_alive={payload['keep_alive']})…")
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(url, json=payload)
        logger.info("✅ Ollama model is warm and ready.")
    except Exception as e:
        logger.warning(f"Ollama warm-up skipped ({type(e).__name__}): {e}")


async def _warmup_embeddings() -> None:
    """
    Fire a throwaway embedding request so Ollama loads the embedding
    model into RAM at startup, same as _warmup_ollama() does for the
    chat model. Without this, ContextManager's first semantic-memory
    lookup (brain/conversation.py -> brain/embeddings.py) pays a
    multi-second cold-load cost mid-conversation instead of at startup.
    """
    url = f"{config.llm.base_url.rstrip('/')}/api/embeddings"
    model = config.context.embedding_model
    payload = {"model": model, "prompt": "hello", "keep_alive": "30m"}
    gpu_opts = _embedding_gpu_options()
    if gpu_opts:
        payload["options"] = gpu_opts
    try:
        logger.info(f"Pre-warming Ollama embedding model '{model}' (device={_resolve_embedding_device()})…")
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(url, json=payload)
        logger.info("✅ Embedding model is warm and ready.")
    except Exception as e:
        logger.warning(f"Embedding warm-up skipped ({type(e).__name__}): {e}")


# ── Interrupt / barge-in ──────────────────────────────────────────────────────

async def _hard_stop_audio() -> None:
    """
    The stop callback registered with core/state.py — called from
    state.interrupt() BEFORE the current speech Task is cancelled, so
    blocking waits (sd.wait() for local playback, ws_server's
    wait_for_audio_done() for the avatar) unblock immediately instead of
    the cancellation racing against hardware/network latency.

    Halts both possible audio paths regardless of config.tts.output,
    since it's cheap to call and avoids missing a case if output mode
    changes later.
    """
    try:
        sd.stop()
    except Exception as e:
        logger.debug(f"sd.stop() during interrupt: {e}")
    await ws_server.broadcast_stop_audio()


async def _end_listening() -> None:
    """Empty STT result — back to IDLE, unless something already moved the FSM on."""
    if state.current != MayaState.LISTENING:
        return
    await state.set(MayaState.IDLE)
    await ws_server.broadcast_state("idle")
    await ws_server.broadcast_behavior(
        behavior_engine.compose(mood_manager.baseline_expression(), source="idle")
    )


def _on_ws_done(task: asyncio.Task) -> None:
    """Surface a WebSocket server failure (e.g. port already in use)."""
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"WebSocket server stopped: {task.exception()!r}")


def _on_node_task_done(task: asyncio.Task) -> None:
    """
    Surface an unexpected MayaNode sync-manager crash. run() is designed
    to never raise — every failure (Node not found, connection refused,
    a malformed response) is caught, logged, and backed off from inside
    the manager itself — so reaching here means a genuine bug in that
    module, not a normal "Node is offline" condition.
    """
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"MayaNode sync manager stopped unexpectedly: {task.exception()!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Initialize Maya's services and run the listener and command worker."""
    # Register before starting runtime tasks; the observer only schedules work
    # on the event loop and is harmless before user-facing states are possible.
    state.add_observer(_on_state_change)

    ws_task = asyncio.create_task(ws_server.serve(), name="ws-server")   # kept so it can't be GC'd
    ws_task.add_done_callback(_on_ws_done)

    # The sync manager isolates discovery and connection failures from voice I/O.
    node_task = asyncio.create_task(node_sync_manager.run(), name="node-sync")   # kept so it can't be GC'd
    node_task.add_done_callback(_on_node_task_done)

    logger.info(f"Starting {config.name}…")

    speaker     = Speaker()
    transcriber = Transcriber()
    processor   = Processor(speaker)

    # Wire the interrupt stop callback before anything can possibly speak.
    state.register_stop_callback(_hard_stop_audio)

    # Pre-warm Ollama in background while greeting plays
    warmup_task = asyncio.create_task(_warmup_ollama())
    embed_warmup_task = asyncio.create_task(_warmup_embeddings())

    # Pre-warm Kokoro TTS — loads model, voices, and JIT kernels into memory.
    # run_in_executor returns a Future directly — no create_task needed.
    loop = asyncio.get_running_loop()
    kokoro_warmup_task = loop.run_in_executor(None, llm_warmup)

    # Speaker owns its own separate Kokoro pipeline instance (used for
    # greetings/skills/alerts, distinct from llm_service.py's module-level
    # one warmed above) — warm it too so the startup greeting isn't the
    # first real inference call.
    speaker_warmup_task = loop.run_in_executor(None, speaker.warmup)

    # ── Wake callback ──────────────────────────────────────────────────
    async def on_wake() -> None:
        """Wake Maya and deliver an interruptible greeting."""
        if not state.is_sleeping():
            return
        await state.set(MayaState.IDLE)
        logger.info("✅ Maya is awake.")
        print(f"\n✅  {config.name} is awake — listening for commands {_U}.")
        # Cancellation must stop the greeting task as well as its audio.
        await state.run_interruptible(speaker.speak(f"I'm here {_U}. How can I help?"))

    # ── Interrupt callback (barge-in) ───────────────────────────────────
    async def on_interrupt() -> None:
        """
        Cancels Maya's current speech task. Called from three places:
          1. on_speech() below, only when the just-transcribed utterance
             contains a wake phrase ("hey maya", "maya", ...) AND she
             was speaking when it started — i.e. she was deliberately
             called by name mid-sentence.
          2. on_speech()'s sleep-trigger branch, unconditionally when
             something is currently speaking/processing — a "go to
             sleep" always takes precedence, wake word or not (see
             "Sleep race").
          3. ws_server's interrupt handler, for a future browser-side
             manual "stop talking" button — that's an explicit user
             action already, so it doesn't need a wake-word check.
        """
        interrupted = await state.interrupt()
        if interrupted:
            logger.info("🛑 Barge-in — Maya stopped talking.")
            print(f"\n🛑  {config.name} was interrupted — listening {_U}.")

    # Let the browser's own interrupt message (future manual "stop
    # talking" button) drive the exact same path as a mic barge-in.
    ws_server.set_interrupt_handler(on_interrupt)

    # ── on_speech ─────────────────────────────────────────────────────
    async def on_speech(audio) -> None:
        """Transcribe one audio segment and route it through state and the command queue."""
        if state.is_sleeping():
            return

        # Snapshot BEFORE transcribing — speak()'s own finally block can
        # legitimately flip SPEAKING -> IDLE while the STT round-trip
        # below is in flight, so we need to know what was true when
        # this utterance STARTED, not whatever happens to be true by
        # the time we're done deciding whether it's a barge-in.
        was_speaking = state.can_interrupt()

        # LISTENING is only taken from a quiet state, never over an
        # in-flight PROCESSING/SPEAKING turn.
        began_listening = not state.is_busy()
        if began_listening:
            await state.set(MayaState.LISTENING)
            await ws_server.broadcast_state("listening")

        text = await transcriber.transcribe(audio)

        if not text:
            if began_listening:
                await _end_listening()
            return

        print(f"\n👂 {_U}: {text}")
        text_lower = text.lower().strip()

        if was_speaking:
            # Barge-in only fires when Maya is explicitly called by
            # name — talking over her about something else no longer
            # cuts her off. This still gets transcribed and queued
            # below exactly like any other command; it just doesn't
            # interrupt her current sentence to do it.
            if contains_wake_word(text_lower):
                await on_interrupt()
            else:
                logger.debug(
                    f"Speech during SPEAKING didn't include the wake "
                    f"word — not a barge-in, queuing normally: '{text}'"
                )

        if is_sleep_command(text_lower):
            # Sleep takes precedence over active work so the goodbye line
            # cannot race another response on the shared audio pipeline.
            if state.can_interrupt():
                await state.interrupt()
            await state.set(MayaState.SLEEPING)
            logger.info("💤 Maya going to sleep.")
            print(f"💤  {config.name} is sleeping — say '{config.wake_word}' to wake.")
            await state.run_interruptible(speaker.speak(
                f"Going to sleep {_U}. Say {config.wake_word} when you need me."
            ))
            return

        queued = await queue_manager.put(text)
        if not queued:
            # A full queue leaves no worker transition for this utterance.
            await _end_listening()
        # Processor.handle() moves a queued utterance to PROCESSING.

    queue_manager.set_handler(processor.handle)

    # Wait for both warm-ups to finish before going live
    await asyncio.gather(warmup_task, kokoro_warmup_task, embed_warmup_task, speaker_warmup_task)
    print("🧠  Ollama ready.")
    print("🎙️  Kokoro TTS ready.\n")

    # Record model residency after warm-up so configured device placement is
    # visible before the first user turn.
    await describe_ollama_models()

    # ── Startup banner + greeting ──────────────────────────────────────
    hour = datetime.datetime.now().hour
    tod  = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    print(f"\n{'─'*55}")
    print(f"  {config.name} – Voice Assistant")
    print(f"  Wake word : '{config.wake_word}'")
    print(f"  Sleep     : say 'goodbye' or 'go to sleep'")
    print(f"  Interrupt : say her name while she's talking, e.g. 'hey maya'")
    print(f"  Stop      : Ctrl+C")
    print(f"{'─'*55}\n")

    # Start awake — speak() now restores the prior state, so this must be set first.
    await state.set(MayaState.IDLE)
    await ws_server.broadcast_animation("wave")
    # Keep startup speech on the same cancellation path as later responses.
    await state.run_interruptible(speaker.speak(
        f"GOOD {tod} {_U}! I'm {config.name}. "
    ))

    # ── Launch concurrent tasks ────────────────────────────────────────
    listener = Listener(on_speech=on_speech, on_wake=on_wake)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(listener.start(),    name="listener")
        tg.create_task(queue_manager.run(), name="queue-worker")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print(f"\n👋  {config.name} stopped.")