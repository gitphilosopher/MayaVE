"""
brain/memory.py
In-memory conversation buffer for Maya's recent-turn context.

This module provides the lightweight, in-process memory window used to keep a
bounded history of user and assistant turns. It is intentionally simple: the
store is ephemeral and resets on restart, but it preserves the public API that
higher layers expect so it can later be swapped for SQLite, vector-backed, or
persisted storage without changing callers.

The key behavior is eviction: once the window exceeds `max_entries`, the oldest
entry is removed. If an `on_evict` callback was supplied, it is invoked just
before that entry is dropped. This allows higher-level context logic to fold the
expired turn into semantic memory or another summary path without losing the
material entirely. The callback is best-effort; exceptions are suppressed so a
failed summary hook never destabilizes normal conversation flow.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single stored conversation turn, keeping the role, text, and timestamp."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


class Memory:
    """Bounded in-memory history for recent conversation turns."""

    def __init__(self, max_entries: int = 50, on_evict: Optional[Callable[[MemoryEntry], None]] = None):
        self._entries: list[MemoryEntry] = []
        self._max = max_entries
        self._on_evict = on_evict

    def set_evict_callback(self, fn: Optional[Callable[[MemoryEntry], None]]) -> None:
        """Register or replace the callback fired just before an entry is evicted."""
        self._on_evict = fn

    def add(self, role: str, content: str) -> None:
        """Append a message and evict the oldest entry if the window has reached capacity."""
        entry = MemoryEntry(role=role, content=content)
        self._entries.append(entry)
        if len(self._entries) > self._max:
            evicted = self._entries.pop(0)
            if self._on_evict:
                try:
                    self._on_evict(evicted)
                except Exception:
                    logger.debug("Memory on_evict callback failed (non-fatal)", exc_info=True)
        logger.debug(f"Memory +{role}: '{content[:60]}…'")

    def get_history(self, last_n: int = 10) -> list[dict]:
        """Return the most recent `last_n` turns as OpenAI-style message objects."""
        return [
            {"role": e.role, "content": e.content}
            for e in self._entries[-last_n:]
        ]

    def clear(self) -> None:
        """Remove all stored turns and reset the in-memory window."""
        self._entries.clear()
        logger.info("Memory cleared.")

    def __len__(self) -> int:
        return len(self._entries)


memory = Memory()