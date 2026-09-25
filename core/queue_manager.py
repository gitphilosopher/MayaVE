"""
core/queue_manager.py
The sequential backbone of Maya.

All voice commands enter ONE asyncio.Queue.
The worker loop drains it one item at a time — preventing overlapping
responses no matter how fast the user speaks.

Queue item schema:
  {
    "text":      str,          # transcribed command ("" for jobs)
    "timestamp": float,        # time.time() at capture
    "priority":  int,          # 0 = normal, 1 = interrupt (future)
    "job":       callable,     # optional — see put_job()
  }

put() return value (Batch 4 — "Listening state edge cases"):
  put() now returns True/False depending on whether the command was
  actually enqueued. main.py's on_speech() already sets the FSM to
  LISTENING before calling put(); Processor.handle() is what eventually
  moves it to PROCESSING. If put() silently drops the item (queue full),
  nothing would ever make that move, and the FSM — and the frontend's
  idle-fidget gate along with it — would be stuck on LISTENING forever.
  on_speech() uses the return value to reset immediately instead of
  relying solely on main.py's LISTENING watchdog as a slower backstop.
"""

import asyncio
import logging
import time
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

# Type alias for the async handler the worker calls
CommandHandler = Callable[[dict], Awaitable[None]]
Job = Callable[[], Awaitable[None]]


class QueueManager:
    def __init__(self, maxsize: int = 10):
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=maxsize)
        self._handler: CommandHandler | None = None
        self._running = False

    # ── Producer side ─────────────────────────────────────────────────

    async def put(self, text: str, priority: int = 0) -> bool:
        """
        Enqueue a transcribed command. Returns True if it was queued,
        False if the queue was full and it was dropped — callers whose
        own FSM tracking depends on this actually being picked up should
        check the return value (see main.py's on_speech).
        """
        item = {
            "text":      text.strip(),
            "timestamp": time.time(),
            "priority":  priority,
        }
        try:
            self._queue.put_nowait(item)
            logger.info(f"Queued command: '{text}'  (depth={self._queue.qsize()})")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full — dropping command: '{text}'")
            return False

    async def put_job(self, job: Job) -> None:
        """
        Enqueue an async callable to run on the worker, in order with
        commands, so it never overlaps a turn (e.g. timer alerts).
        Waits for room instead of dropping.
        """
        await self._queue.put({
            "text":      "",
            "timestamp": time.time(),
            "priority":  0,
            "job":       job,
        })
        logger.info(f"Queued job  (depth={self._queue.qsize()})")

    # ── Consumer side ─────────────────────────────────────────────────

    def set_handler(self, handler: CommandHandler) -> None:
        """Register the async function that processes each command."""
        self._handler = handler

    async def run(self) -> None:
        """
        Blocking worker loop — run as an asyncio Task.
        Processes commands one at a time, in order.
        """
        if self._handler is None:
            raise RuntimeError("No handler registered. Call set_handler() first.")

        self._running = True
        logger.info("QueueManager worker started.")

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue   # nothing in queue — keep looping

            try:
                job = item.get("job")
                if job is not None:
                    await job()
                else:
                    await self._handler(item)
            except Exception as e:
                logger.error(f"Handler error for '{item['text'] or 'job'}': {e}", exc_info=True)
            finally:
                self._queue.task_done()

    async def stop(self) -> None:
        self._running = False
        logger.info("QueueManager stopped.")


# Singleton
queue_manager = QueueManager()