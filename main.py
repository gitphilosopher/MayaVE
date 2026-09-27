"""Run MayaVE's voice assistant and its supporting background services.

Run with ``python main.py`` from the project environment. Say "Maya" or
"hey Maya" to wake the assistant, and "go to sleep", "sleep", or
"goodbye" to put it to sleep. Calling Maya by name while she is speaking
interrupts her; other speech is queued. Press Ctrl+C to stop the process.

Startup also launches the WebSocket server and optional MayaNode sync,
warms the configured models, and starts the listener and command queue.
"""

# huggingface_hub reads this setting at import time.
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

os.makedirs(config.log_dir, exist_ok=True)
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

# Keep request-level logs from dependencies out of the normal output.
for _noisy in ("httpx", "httpcore", "websockets", "urllib3", "tensorflow"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("main")

_U = config.user_name

# Recover LISTENING if a dropped command or an interrupt without follow-up
# leaves no event to move the state to PROCESSING or IDLE.
_LISTENING_WATCHDOG_S = 15.0
_listening_watchdog_task: asyncio.Task | None = None


def _on_state_change(old: MayaState, new: MayaState) -> None:
    """Schedule state-change handling safely from any callback thread."""
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.call_soon_threadsafe(_handle_state_change, new)
    except RuntimeError:
        pass  # The loop may already be closed during shutdown.


def _handle_state_change(new: MayaState) -> None:
    """Restart the watchdog whenever the assistant enters a new state."""
    global _listening_watchdog_task
    if _listening_watchdog_task is not None:
        _listening_watchdog_task.cancel()
        _listening_watchdog_task = None
    if new == MayaState.LISTENING:
        _listening_watchdog_task = asyncio.create_task(
            _listening_watchdog(), name="listening-watchdog"
        )


async def _listening_watchdog() -> None:
    """Return to IDLE if LISTENING receives no follow-up before the timeout."""
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


async def _warmup_ollama() -> None:
    """Load the configured chat model into Ollama during startup."""
    url = f"{config.llm.base_url.rstrip('/')}/api/chat"
    payload = {
        "model":    config.llm.model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream":   False,
        "keep_alive": chat_keep_alive(),
        "options":  {"num_predict": 1},
    }
    try:
        logger.info(f"Pre-warming Ollama model '{config.llm.model}' (keep_alive={payload['keep_alive']})…")
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(url, json=payload)
        logger.info("✅ Ollama model is warm and ready.")
    except Exception as e:
        logger.warning(f"Ollama warm-up skipped ({type(e).__name__}): {e}")


async def _warmup_embeddings() -> None:
    """Load the configured embedding model into Ollama during startup."""
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


async def _hard_stop_audio() -> None:
    """Stop local and browser audio before the active speech task is cancelled."""
    try:
        sd.stop()
    except Exception as e:
        logger.debug(f"sd.stop() during interrupt: {e}")
    await ws_server.broadcast_stop_audio()


async def _end_listening() -> None:
    """Return to IDLE after empty or unprocessable input, if still LISTENING."""
    if state.current != MayaState.LISTENING:
        return
    await state.set(MayaState.IDLE)
    await ws_server.broadcast_state("idle")
    await ws_server.broadcast_behavior(
        behavior_engine.compose(mood_manager.baseline_expression(), source="idle")
    )


def _on_ws_done(task: asyncio.Task) -> None:
    """Log an unexpected WebSocket server task failure."""
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"WebSocket server stopped: {task.exception()!r}")


def _on_node_task_done(task: asyncio.Task) -> None:
    """Log an unexpected MayaNode sync task failure."""
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"MayaNode sync manager stopped unexpectedly: {task.exception()!r}")


async def main() -> None:
    """Initialize services, warm models, greet the user, and run the assistant."""
    state.add_observer(_on_state_change)

    ws_task = asyncio.create_task(ws_server.serve(), name="ws-server")
    ws_task.add_done_callback(_on_ws_done)

    node_task = asyncio.create_task(node_sync_manager.run(), name="node-sync")
    node_task.add_done_callback(_on_node_task_done)

    logger.info(f"Starting {config.name}…")

    speaker     = Speaker()
    transcriber = Transcriber()
    processor   = Processor(speaker)

    state.register_stop_callback(_hard_stop_audio)

    warmup_task = asyncio.create_task(_warmup_ollama())
    embed_warmup_task = asyncio.create_task(_warmup_embeddings())

    loop = asyncio.get_running_loop()
    kokoro_warmup_task = loop.run_in_executor(None, llm_warmup)

    # Speaker has its own Kokoro pipeline, separate from the LLM pipeline.
    speaker_warmup_task = loop.run_in_executor(None, speaker.warmup)

    async def on_wake() -> None:
        """Wake Maya and deliver the greeting."""
        if not state.is_sleeping():
            return
        await state.set(MayaState.IDLE)
        logger.info("✅ Maya is awake.")
        print(f"\n✅  {config.name} is awake — listening for commands {_U}.")
        await state.run_interruptible(speaker.speak(f"I'm here {_U}. How can I help?"))

    async def on_interrupt() -> None:
        """Cancel the active speech or processing task and stop its audio."""
        interrupted = await state.interrupt()
        if interrupted:
            logger.info("🛑 Barge-in — Maya stopped talking.")
            print(f"\n🛑  {config.name} was interrupted — listening {_U}.")

    ws_server.set_interrupt_handler(on_interrupt)

    async def on_speech(audio) -> None:
        """Transcribe input, handle sleep/barge-in commands, and queue speech."""
        if state.is_sleeping():
            return

        # Capture this before transcription; speech may finish while STT runs.
        was_speaking = state.can_interrupt()

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
            if contains_wake_word(text_lower):
                await on_interrupt()
            else:
                logger.debug(
                    f"Speech during SPEAKING didn't include the wake "
                    f"word — not a barge-in, queuing normally: '{text}'"
                )

        if is_sleep_command(text_lower):
            # Stop the current turn before sharing the audio pipeline with goodbye.
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
            await _end_listening()

    queue_manager.set_handler(processor.handle)

    await asyncio.gather(warmup_task, kokoro_warmup_task, embed_warmup_task, speaker_warmup_task)
    print("🧠  Ollama ready.")
    print("🎙️  Kokoro TTS ready.\n")

    await describe_ollama_models()

    hour = datetime.datetime.now().hour
    tod  = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    print(f"\n{'─'*55}")
    print(f"  {config.name} – Voice Assistant")
    print(f"  Wake word : '{config.wake_word}'")
    print(f"  Sleep     : say 'goodbye' or 'go to sleep'")
    print(f"  Interrupt : say her name while she's talking, e.g. 'hey maya'")
    print(f"  Stop      : Ctrl+C")
    print(f"{'─'*55}\n")

    # Set IDLE before speaking so the speaker restores the correct state.
    await state.set(MayaState.IDLE)
    await ws_server.broadcast_animation("wave")
    await state.run_interruptible(speaker.speak(
        f"GOOD {tod} {_U}! I'm {config.name}. "
    ))

    listener = Listener(on_speech=on_speech, on_wake=on_wake)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(listener.start(),    name="listener")
        tg.create_task(queue_manager.run(), name="queue-worker")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print(f"\n👋  {config.name} stopped.")