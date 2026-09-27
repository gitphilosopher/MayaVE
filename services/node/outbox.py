"""
services/node/outbox.py
Local persistent pending queue ("outbox") for MayaVE events and memory
writes awaiting MayaNode sync (Step 3).

This is the durability layer between the application-facing producers
(events.py, memory_producer.py) and NodeSyncManager: anything recorded
here survives a MayaVE process restart and does not require MayaNode to
be reachable at the moment it's recorded. Same on-disk pattern as
services/node/sync_state.py and core/expression_library.py: an
in-memory cache updated immediately (synchronous, non-blocking for
callers) with the JSON snapshot written on a single background thread
so writes are applied in submission order without racing each other.

Idempotency:
  - Events are keyed by event_uid — re-adding the same uid is a no-op,
    so a caller (or a resumed sync round) can safely retry.
  - Memory writes are keyed by `key`, last-write-wins locally too (only
    the latest queued value per key is ever kept), matching the
    server's own last-write-wins semantics for (device_id, key).

Removal is deliberately narrow: NodeSyncManager removes only the exact
items a sync round's response actually settled (see remove_events /
remove_memory), never the whole outbox — anything left unsent (over a
batch limit) or superseded by a newer local write made after the round
started stays pending for the next round.
"""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

# Single-worker, matches core/expression_library.py's pattern — writes
# apply in submission order without callers ever blocking on disk IO.
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="node-outbox-writer")


class OutboxStore:
    """One process's view of the pending-events/pending-memory outbox
    for a given state directory. Construct directly (like SyncStateStore)
    or fetch the shared instance for a directory via outbox_store()."""

    def __init__(self, state_dir: str | None = None):
        self._dir = Path(state_dir) if state_dir else (Path.home() / "Maya" / "Node")
        self._path = self._dir / "outbox.json"
        self._lock = threading.Lock()
        self._events: dict[str, dict] = {}   # event_uid -> envelope dict, insertion order
        self._memory: dict[str, dict] = {}   # key -> {"value":..., "updated_at":...}
        self._loaded = False

    # ── Load ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Best-effort, idempotent load — safe to call from every public
        method so callers never have to remember to load() first. A
        missing or corrupt file starts empty rather than failing."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                if self._path.exists():
                    data = json.loads(self._path.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("version") == _SCHEMA_VERSION:
                        for e in data.get("events", []):
                            if isinstance(e, dict) and e.get("event_uid"):
                                self._events[e["event_uid"]] = e
                        for k, v in (data.get("memory") or {}).items():
                            if isinstance(v, dict) and "value" in v and "updated_at" in v:
                                self._memory[k] = v
            except (OSError, ValueError, TypeError) as e:
                logger.warning(f"Node outbox unreadable ({e}) — starting empty.")
        logger.info(
            f"Node outbox loaded ({self._path}) — "
            f"{len(self._events)} pending event(s), {len(self._memory)} pending memory key(s)"
        )

    # ── Mutators ──────────────────────────────────────────────────────

    def add_event(self, envelope: dict) -> bool:
        """Idempotent by event_uid. Returns False only if the envelope
        has no event_uid at all (caller error)."""
        self.load()
        uid = envelope.get("event_uid")
        if not uid:
            return False
        with self._lock:
            if uid not in self._events:
                self._events[uid] = envelope
                self._persist_locked()
        return True

    def add_memory(self, key: str, value, updated_at: str) -> None:
        """Last-write-wins locally — only the latest pending value per
        key is ever queued. A write with an updated_at older than what's
        already queued is dropped (out-of-order caller), never regressing
        a fresher pending write."""
        self.load()
        with self._lock:
            current = self._memory.get(key)
            if current is not None and current.get("updated_at", "") > updated_at:
                return
            self._memory[key] = {"value": value, "updated_at": updated_at}
            self._persist_locked()

    def remove_events(self, event_uids) -> None:
        if not event_uids:
            return
        self.load()
        with self._lock:
            changed = False
            for uid in event_uids:
                if uid in self._events:
                    del self._events[uid]
                    changed = True
            if changed:
                self._persist_locked()

    def remove_memory(self, settled: list[tuple[str, str]]) -> None:
        """Remove a pending (key, updated_at) write only if it is still
        exactly the one that was sent — a newer local write queued after
        the sync request went out survives and gets retried next round."""
        if not settled:
            return
        self.load()
        with self._lock:
            changed = False
            for key, updated_at in settled:
                current = self._memory.get(key)
                if current is not None and current.get("updated_at") == updated_at:
                    del self._memory[key]
                    changed = True
            if changed:
                self._persist_locked()

    # ── Readers ───────────────────────────────────────────────────────

    def pending_events(self, limit: int | None = None) -> list[dict]:
        self.load()
        with self._lock:
            items = list(self._events.values())
        return items[:limit] if limit else items

    def pending_memory(self, limit: int | None = None) -> list[dict]:
        """[{"key":..., "value":..., "updated_at":...}, ...]."""
        self.load()
        with self._lock:
            items = [
                {"key": k, "value": v["value"], "updated_at": v["updated_at"]}
                for k, v in self._memory.items()
            ]
        return items[:limit] if limit else items

    def has_pending(self) -> bool:
        self.load()
        with self._lock:
            return bool(self._events or self._memory)

    def counts(self) -> tuple[int, int]:
        self.load()
        with self._lock:
            return len(self._events), len(self._memory)

    # ── Persistence ───────────────────────────────────────────────────

    def _persist_locked(self) -> None:
        """Must be called with self._lock held. Snapshots current state
        and schedules the write on the background thread so callers
        (including record_event/record_memory from arbitrary — possibly
        sync — call sites) never block on disk IO."""
        snapshot = {
            "version": _SCHEMA_VERSION,
            "events": list(self._events.values()),
            "memory": dict(self._memory),
        }
        try:
            payload = json.dumps(snapshot, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"Node outbox snapshot not JSON-serialisable (non-fatal, not persisted): {e}")
            return
        try:
            _writer.submit(self._write_payload, payload)
        except RuntimeError:
            pass  # executor shut down (interpreter exit) — best effort only

    def _write_payload(self, payload: str) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._path)
        except OSError as e:
            logger.warning(f"Node outbox save failed (non-fatal): {e}")

    def flush(self) -> None:
        """Block until every write scheduled so far on this store has
        completed. Test/diagnostic use only — production code never
        needs this; the outbox is eventually-persistent by design."""
        fut = _writer.submit(lambda: None)
        fut.result(timeout=5)


# ── Shared, per-directory singleton ──────────────────────────────────────
# Cached by resolved directory so application code (events.py /
# memory_producer.py) and NodeSyncManager, both resolving the same
# config.node.state_dir, always share the same in-memory store — writes
# from one are immediately visible to the other in-process, with disk
# persistence as the cross-restart durability layer.

_stores: dict[str, "OutboxStore"] = {}
_stores_lock = threading.Lock()


def outbox_store(state_dir: str | None = None) -> OutboxStore:
    from config.settings import config

    resolved = state_dir if state_dir is not None else config.node.state_dir
    key = str(Path(resolved) if resolved else (Path.home() / "Maya" / "Node"))
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = OutboxStore(resolved)
            _stores[key] = store
        return store


def reset_all_for_tests() -> None:
    """Test-only: drop every cached OutboxStore so the next outbox_store()
    call for the same directory constructs a fresh instance that reloads
    from disk — simulates a process restart without restarting the
    interpreter."""
    with _stores_lock:
        _stores.clear()