"""
services/node/sync_state.py
Persists the /sync cursor (and last-sync timestamp) across restarts so a
restart doesn't silently replay or skip MayaNode's change feed. JSON file,
atomic write (.tmp + replace) — same pattern as core/expression_library.py.

Cursor semantics mirror MayaNode's own (see services/sync_service.py):
monotonically increasing, only ever moved forward. Step 1 never applies
anything pulled from MayaNode to MayaVE's own state (see sync_manager.py's
docstring — the sync body is always empty), so this cursor just tracks how
far the empty-payload loop has progressed; a restart resumes from there
instead of re-requesting the whole feed from zero every time.
"""
import asyncio
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class SyncStateStore:
    def __init__(self, state_dir: str | None = None):
        self._dir = Path(state_dir) if state_dir else (Path.home() / "Maya" / "Node")
        self._path = self._dir / "sync_state.json"
        self._lock = asyncio.Lock()
        self._cursor = 0
        self._last_synced_at: float | None = None

    def load(self) -> None:
        """
        Synchronous, best-effort load — call once at startup, before
        anything else touches this store (matches the project's existing
        pattern of small, one-time synchronous startup reads, e.g.
        core/expression_library.py's first _load()). A missing or corrupt
        file starts fresh at cursor=0 rather than failing startup —
        MayaNode's cursor semantics tolerate a client re-requesting from
        0 (an oversized first batch), never corruption.
        """
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == _SCHEMA_VERSION:
                    self._cursor = max(0, int(data.get("cursor", 0)))
                    self._last_synced_at = data.get("last_synced_at")
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"Node sync state unreadable ({e}) — starting from cursor 0.")
        logger.info(f"Node sync state loaded — cursor={self._cursor}")

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def last_synced_at(self) -> float | None:
        return self._last_synced_at

    async def advance(self, new_cursor: int) -> None:
        """
        Advance the cursor and persist it. Never moves backwards — a
        stale or out-of-order response must not undo progress a later
        sync already recorded (defensive; today's loop is strictly
        sequential, but this keeps the invariant explicit rather than
        assumed).
        """
        if new_cursor <= self._cursor:
            return
        async with self._lock:
            if new_cursor <= self._cursor:   # re-check inside the lock
                return
            self._cursor = new_cursor
            self._last_synced_at = time.time()
            await asyncio.get_running_loop().run_in_executor(None, self._write)

    def _write(self) -> None:
        payload = json.dumps({
            "version": _SCHEMA_VERSION,
            "cursor": self._cursor,
            "last_synced_at": self._last_synced_at,
        })
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._path)
        except OSError as e:
            logger.warning(f"Failed to persist node sync state (non-fatal): {e}")