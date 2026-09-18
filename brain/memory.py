"""
brain/memory.py
Phase 7 (stub): Conversation memory store.

Current: simple in-process list (resets on restart).
Designed for future upgrade to SQLite or vector memory
without changing the public API.

Stage 2 addition: an optional on_evict callback fires right before the
oldest entry is dropped from the window, so ContextManager (see
brain/conversation.py) can fold it into a semantic-memory summary
instead of losing it outright. Callback is best-effort — a failure here
must never break normal conversation flow.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    role:      str    # "user" | "assistant"
    content:   str
    timestamp: float  = field(default_factory=time.time)


class Memory:
    def __init__(self, max_entries: int = 50, on_evict: Optional[Callable[[MemoryEntry], None]] = None):
        self._entries: list[MemoryEntry] = []
        self._max = max_entries
        self._on_evict = on_evict

    def set_evict_callback(self, fn: Optional[Callable[[MemoryEntry], None]]) -> None:
        """Register/replace the callback fired just before an entry is dropped."""
        self._on_evict = fn

    def add(self, role: str, content: str) -> None:
        entry = MemoryEntry(role=role, content=content)
        self._entries.append(entry)
        if len(self._entries) > self._max:
            evicted = self._entries.pop(0)   # drop oldest
            if self._on_evict:
                try:
                    self._on_evict(evicted)
                except Exception:
                    logger.debug("Memory on_evict callback failed (non-fatal)", exc_info=True)
        logger.debug(f"Memory +{role}: '{content[:60]}…'")

    def get_history(self, last_n: int = 10) -> list[dict]:
        """Return last N turns as OpenAI-style message dicts."""
        return [
            {"role": e.role, "content": e.content}
            for e in self._entries[-last_n:]
        ]

    def clear(self) -> None:
        self._entries.clear()
        logger.info("Memory cleared.")

    def __len__(self) -> int:
        return len(self._entries)


# Singleton
memory = Memory()