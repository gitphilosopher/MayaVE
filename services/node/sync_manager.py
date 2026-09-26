"""
services/node/sync_manager.py
Step 1 of MayaVE <-> MayaNode integration (see docs/CONTRIBUTING.md):
discovery, connection, and a periodic empty-payload /sync loop that
safely persists/advances a cursor.

Deliberately inert: no MayaVE event, memory, mood, or intent data is
pushed or applied yet — every sync() call carries empty events/memory
lists, and nothing pulled back from MayaNode's `changes` is applied to
any MayaVE state. What this DOES establish for a later integration step:
a stable device_id, a working discover -> connect -> sync -> reconnect
loop, an auth-ready request shape (see client.py), and a durable cursor —
so pushing/pulling real payloads later is a payload change, not a
plumbing change.

Never raises out of run(): every failure (Node not found, connection
refused, timeout, malformed response) is caught, logged, and backed off
from with jitter-free exponential backoff (capped at
config.node.max_backoff_s), so this can run as a background task for the
whole process lifetime — alongside main.py's listener/queue-worker —
without a Node outage ever affecting Maya's voice pipeline.
"""
import asyncio
import logging

from config.settings import config
from services.node.client import NodeClient
from services.node.discovery import NodeDiscovery
from services.node.identity import resolve_device_id
from services.node.sync_state import SyncStateStore

logger = logging.getLogger(__name__)


class NodeSyncManager:
    def __init__(self, cfg=None):
        self._cfg = cfg or config.node
        self._discovery = NodeDiscovery(self._cfg)
        self._state = SyncStateStore(self._cfg.state_dir)
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

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def run(self) -> None:
        if not self._cfg.enabled:
            logger.info("MayaNode integration disabled (config.node.enabled=False).")
            return

        self._state.load()
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
        result = await self._client.sync(self._device_id, self._state.cursor)
        if result is None:
            return False
        try:
            new_cursor = int(result["cursor"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Node sync response missing/invalid cursor — ignoring: {result!r}")
            return False
        await self._state.advance(new_cursor)
        logger.debug(
            f"Node sync ok — cursor now {new_cursor} (has_more={result.get('has_more')})"
        )
        return True


# Singleton — matches the existing project pattern (mood_manager,
# queue_manager, ws_server, context_manager).
node_sync_manager = NodeSyncManager()