"""
services/node/sync_manager.py
MayaVE <-> MayaNode integration (see docs/CONTRIBUTING.md):
discovery, connection, and a periodic /sync loop that safely
persists/advances a cursor.

Step 1 established: a stable device_id, a working discover -> connect
-> sync -> reconnect loop, an auth-ready request shape (see client.py),
and a durable cursor.

Step 2 added: the versioned protocol layer (services/node/protocol.py)
— envelope validation, protocol_version negotiation.

Step 3 connected real MayaVE data to that pipeline: each sync round
pulls pending events/memory writes from the local outbox
(services/node/outbox.py) — populated by application code through
services/node/events.py's record_event() and
services/node/memory_producer.py's record_memory(), never by a direct
/sync call anywhere else — pushes them, and reconciles the outbox
against the response: only the items the server actually settled this
round (accepted/duplicate/rejected for events; accepted/noop/conflict
for memory) are removed, so a batch-size cap or a mid-round crash never
loses anything.

Step 4 (this file) closes the loop on the READ side: `changes` pulled
back from MayaNode are now handed to `brain.conversation.context_manager`
so genuinely durable memory (facts/preferences another MayaVE instance
pushed) gets merged into this device's own semantic-memory store through
the existing system, rather than sitting unused. See
`_apply_incoming_changes()` below and
`ContextManager.apply_remote_memory_changes()`'s docstring for exactly
what is and isn't applied. `changes["events"]` (e.g. other devices'
`mayave.turn_completed` records) are intentionally NOT replayed into any
local state — there is no MayaVE-side state a cross-device turn-completed
log should mutate; they exist purely as MayaNode-side history.

Never raises out of run(): every failure (Node not found, connection
refused, timeout, malformed response) is caught, logged, and backed off
from with jitter-free exponential backoff (capped at
config.node.max_backoff_s), so this can run as a background task for the
whole process lifetime — alongside main.py's listener/queue-worker —
without a Node outage ever affecting Maya's voice pipeline. Recording
events/memory (events.py / memory_producer.py) never depends on this
manager running at all — see their own docstrings. Applying incoming
changes is equally best-effort: a failure there is logged and swallowed,
never allowed to affect the sync loop's own cursor/outbox bookkeeping,
which has already completed successfully by the time changes are applied.
"""
import asyncio
import logging

from config.settings import config
from services.node.client import NodeClient
from services.node.discovery import NodeDiscovery
from services.node.identity import resolve_device_id
from services.node.outbox import OutboxStore, outbox_store
from services.node.sync_state import SyncStateStore

logger = logging.getLogger(__name__)

_DEFAULT_MAX_EVENTS = 200
_DEFAULT_MAX_MEMORY = 200


class NodeSyncManager:
    def __init__(self, cfg=None):
        self._cfg = cfg or config.node
        self._discovery = NodeDiscovery(self._cfg)
        self._state = SyncStateStore(self._cfg.state_dir)
        self._outbox: OutboxStore = outbox_store(self._cfg.state_dir)
        self._device_id: str | None = None
        self._client: NodeClient | None = None
        self._connected = False
        self._running = False

    # ── Status (read-only) ────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def cursor(self) -> int:
        return self._state.cursor

    @property
    def device_id(self) -> str | None:
        return self._device_id

    @property
    def pending_counts(self) -> tuple[int, int]:
        """(pending_event_count, pending_memory_key_count) — for
        diagnostics/health checks; never required for normal operation."""
        return self._outbox.counts()

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def run(self) -> None:
        if not self._cfg.enabled:
            logger.info("MayaNode integration disabled (config.node.enabled=False).")
            return

        self._state.load()
        self._outbox.load()
        self._device_id = resolve_device_id(self._cfg.state_dir)
        logger.info(f"MayaNode sync manager starting — device_id={self._device_id}")

        self._running = True
        backoff = self._cfg.initial_backoff_s

        while self._running:
            if self._client is None:
                base_url = await self._safe_discover()
                if base_url is None:
                    self._connected = False
                    await asyncio.sleep(backoff)
                    backoff = min(self._cfg.max_backoff_s, backoff * 2)
                    continue
                self._client = NodeClient(base_url, self._cfg)

            ok = await self._safe_connect_and_sync()
            if ok:
                self._connected = True
                backoff = self._cfg.initial_backoff_s
                await asyncio.sleep(self._cfg.sync_interval_s)
            else:
                self._connected = False
                # Force rediscovery next pass — the base_url that worked
                # before may no longer be right (Node restarted on a
                # different port, moved host, etc.), and re-probing
                # /status is cheap compared to repeatedly failing /sync
                # against a dead address.
                self._client = None
                await asyncio.sleep(backoff)
                backoff = min(self._cfg.max_backoff_s, backoff * 2)

    def stop(self) -> None:
        """Signals run()'s loop to exit after its current sleep/attempt.
        Not currently called anywhere (this manager runs for the process
        lifetime) — provided for a future graceful-shutdown path."""
        self._running = False

    # ── Guarded wrappers — belt-and-braces around already-guarded calls ─

    async def _safe_discover(self) -> str | None:
        try:
            return await self._discovery.discover()
        except Exception as e:   # NodeDiscovery.discover() shouldn't raise, but never trust that alone
            logger.error(f"Unexpected error during MayaNode discovery (non-fatal): {e}", exc_info=True)
            return None

    async def _safe_connect_and_sync(self) -> bool:
        try:
            return await self._connect_and_sync()
        except Exception as e:
            logger.error(f"Unexpected error in MayaNode sync pass (non-fatal): {e}", exc_info=True)
            return False

    # ── One connect + sync pass ─────────────────────────────────────────

    async def _connect_and_sync(self) -> bool:
        """
        "Connect" is proven by a successful heartbeat; only then is a
        sync attempted. Keeping them as two explicit steps (rather than
        just letting sync() double as the connectivity check) means a
        future step can add real per-heartbeat bookkeeping without
        touching the sync path.
        """
        if not await self._client.heartbeat(self._device_id):
            return False
        return await self._sync_once()

    async def _sync_once(self) -> bool:
        """
        Pulls up to config.node.max_events_per_sync / max_memory_per_sync
        pending items from the local outbox, pushes them alongside the
        cursor request, advances the cursor per the Step 2 contract (the
        highest seq the server delivered, never moved backwards — see
        SyncStateStore.advance()), and reconciles the outbox against the
        response. A connection/version/shape failure (result is None)
        touches neither the cursor nor the outbox — everything queued
        stays queued for the next attempt.

        Once the cursor/outbox bookkeeping has succeeded, whatever
        MayaNode delivered in `changes` is handed off for best-effort
        local application (Step 4) — this happens strictly AFTER the
        sync round's own state is committed, so a failure while applying
        changes can never roll back or duplicate the cursor advance.
        """
        max_events = getattr(self._cfg, "max_events_per_sync", _DEFAULT_MAX_EVENTS)
        max_memory = getattr(self._cfg, "max_memory_per_sync", _DEFAULT_MAX_MEMORY)

        sent_events = self._outbox.pending_events(limit=max_events)
        sent_memory = self._outbox.pending_memory(limit=max_memory)
        memory_payload = [
            {
                "key": m["key"], "value": m["value"], "updated_at": m["updated_at"],
                "device_id": self._device_id,
            }
            for m in sent_memory
        ]

        result = await self._client.sync(
            self._device_id, self._state.cursor,
            events=sent_events, memory=memory_payload,
        )
        if result is None:
            return False

        await self._state.advance(result.cursor)
        self._reconcile_outbox(sent_events, sent_memory, result)

        await self._apply_incoming_changes(result.changes)

        logger.debug(
            f"Node sync ok — cursor now {result.cursor} (has_more={result.has_more}) "
            f"pushed events={len(sent_events)} memory={len(sent_memory)}"
        )
        return True

    def _reconcile_outbox(self, sent_events: list[dict], sent_memory: list[dict], result) -> None:
        """
        Removes from the outbox exactly the items this round's response
        settled — never the whole outbox. `result` is a
        services.node.client.SyncResult. An event's status
        (accepted/duplicate/rejected) is always terminal per
        docs/PROTOCOL_CONTRACT.md, so any event_uid present in
        accepted_events is done with, successfully or not (a rejected
        event is permanently invalid; resending it can't help). A memory
        key's status (accepted/noop/conflict) is likewise always
        terminal, but is only cleared for the exact (key, updated_at)
        pair that was actually sent — a newer local write queued after
        this round started is left untouched and retried next round.
        Never raises: a malformed response here just means the next
        round resends the same items, which is always safe (idempotent
        event_uids, last-write-wins memory keys).
        """
        try:
            acked_uids = {
                item.get("event_uid") for item in result.accepted_events
                if isinstance(item, dict) and item.get("event_uid")
            }
            if acked_uids:
                self._outbox.remove_events(acked_uids)

            settled_keys = {
                item.get("key") for item in result.memory_results
                if isinstance(item, dict) and item.get("key")
            }
            if settled_keys:
                pairs = [
                    (m["key"], m["updated_at"]) for m in sent_memory
                    if m["key"] in settled_keys
                ]
                self._outbox.remove_memory(pairs)
        except Exception as e:
            logger.warning(f"Node sync outbox reconciliation failed (non-fatal): {e}")

    async def _apply_incoming_changes(self, changes: dict) -> None:
        """
        Step 4 — best-effort hand-off of MayaNode-originated changes into
        MayaVE's own runtime state, through the EXISTING system that owns
        that state rather than a new parallel path:

          - changes["memory"]: handed to
            brain.conversation.context_manager.apply_remote_memory_changes(),
            which recognizes only its own `semantic_memory:*` key shape
            (see brain/conversation.py's ContextManager) and merges
            matching entries into the local SQLiteVectorStore — the same
            store Maya's own memory policy writes to. Anything else is
            ignored there, not here, so the recognition logic has exactly
            one home.

          - changes["events"]: deliberately NOT applied anywhere. A
            `mayave.turn_completed` record from another device has no
            MayaVE-side state to mutate — it's cross-device history that
            lives on MayaNode, not application state this process needs
            to react to. If a future event type needs local application,
            it gets its own explicit handler here rather than a blanket
            replay.

        `brain.conversation` is imported lazily (not at module load) so
        this module — whose own docstring promises isolation from voice
        I/O and a lightweight import footprint — never pays the cost of
        importing the full context/embedding/vector-store stack unless a
        sync round actually has memory changes to apply. Never raises:
        any failure here is logged and swallowed, well after this round's
        own cursor/outbox bookkeeping has already succeeded.
        """
        memory_changes = (changes or {}).get("memory") or []
        if not memory_changes:
            return
        try:
            from brain.conversation import context_manager
            await context_manager.apply_remote_memory_changes(memory_changes)
        except Exception as e:
            logger.debug(f"Applying incoming MayaNode changes failed (non-fatal): {e}")


# Singleton — matches the existing project pattern (mood_manager,
# queue_manager, ws_server, context_manager).
node_sync_manager = NodeSyncManager()