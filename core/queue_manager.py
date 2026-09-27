"""
core/queue_manager.py
Single-threaded command queue for Maya.

This module serializes all inbound work through a single `asyncio.Queue`, ensuring
that voice commands and scheduled jobs never overlap while the agent is handling
one turn. The worker loop is the gatekeeper: it drains items in order, calls the
registered handler for commands, and executes queued jobs in the same sequence.

Queue item schema:
  {
    "text":      str,          # transcribed command ("" for jobs)
    "timestamp": float,        # time.time() at capture
    "priority":  int,          # currently informational; normal/interrupt order is not yet used
    "job":       callable,     # optional async callable for queued jobs
  }

The queue is intentionally a backpressure point for the voice pipeline. `put()`
returns whether the command was accepted, which lets callers detect a dropped
command before the FSM remains stuck in LISTENING without any follow-up work.
The `run()` loop keeps processing until `stop()` flips `_running` off, and each
item is removed from the queue in a `finally` block so the worker can continue
without deadlocking on exceptions.
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
    """Serialize command and job execution behind a single worker queue."""

    def __init__(self, maxsize: int = 10):
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=maxsize)
        self._handler: CommandHandler | None = None
        self._running = False

    async def put(self, text: str, priority: int = 0) -> bool:
        """Enqueue a transcribed command and report whether it was accepted."""
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

    def set_handler(self, handler: CommandHandler) -> None:
        """Register the async function that processes each queued command."""
        self._handler = handler

    async def run(self) -> None:
        """Drain the queue sequentially until stopped, executing commands and jobs in order."""
        if self._handler is None:
            raise RuntimeError("No handler registered. Call set_handler() first.")

        self._running = True
        logger.info("QueueManager worker started.")

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

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


queue_manager = QueueManager()