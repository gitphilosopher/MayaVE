"""
services/node/client.py
Thin async HTTP client for MayaNode's /heartbeat and /sync endpoints.
Every public method is guarded — connection errors, timeouts, and non-2xx
responses are caught and reported to the caller as a plain failure result
(False / None), never raised — so a Node outage can never propagate into
MayaVE's own event loop or crash NodeSyncManager's loop.

Auth-ready: _headers() attaches `Authorization: Bearer <token>` whenever
config.node.auth_token is set. MayaNode does not check it yet (see
docs/PROTOCOL_CONTRACT.md's 'Authentication' section) — this hook exists
so turning on real auth later, on both sides, is a config change rather
than a protocol redesign here.

Step 2 (see services/node/protocol.py — MayaVE's own, self-contained
implementation of docs/PROTOCOL_CONTRACT.md; there is no shared code
with MayaNode's node/protocol.py): sync() builds its request body
through protocol.build_sync_request() (adds the protocol_version field
to the wire shape) and validates any outgoing events locally first — an
invalid event is dropped with a logged warning rather than either
reaching the server malformed or blocking the rest of the batch. The
response's own protocol_version is checked with
protocol.check_response_version(); a peer speaking a version this build
doesn't understand is treated as a sync failure (logs a warning, returns
None), exactly like a connection failure — callers already handle that
uniformly. Step 1 callers still always pass empty events/memory (see
sync_manager.py) — this path is exercised by tests today, not by
production traffic yet.
"""
import logging

import httpx

from config.settings import config
from services.node import protocol
from services.node.protocol import ProtocolError

logger = logging.getLogger(__name__)


class NodeClient:
    def __init__(self, base_url: str, cfg=None, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self._cfg = cfg or config.node
        # transport: injection point for tests (httpx.MockTransport).
        self._transport = transport

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._cfg.auth_token:
            headers["Authorization"] = f"Bearer {self._cfg.auth_token}"
        return headers

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._cfg.request_timeout, transport=self._transport)

    async def heartbeat(self, device_id: str) -> bool:
        """POST /heartbeat. True on a 2xx reply, False on any failure
        (connection refused, timeout, non-2xx) — never raises."""
        url = f"{self.base_url}/heartbeat"
        try:
            async with self._make_client() as client:
                resp = await client.post(url, json={"device_id": device_id}, headers=self._headers())
                resp.raise_for_status()
                return True
        except httpx.HTTPError as e:
            logger.debug(f"Node heartbeat failed: {e}")
            return False

    def _filter_valid_events(self, events: list[dict] | None, device_id: str) -> list[dict]:
        """Drops any event that fails local validation before it ever
        leaves this process, logging why. Best-effort convenience — the
        server validates independently regardless (see
        docs/PROTOCOL_CONTRACT.md), so this only saves a wasted round
        trip and gives the caller earlier, local feedback via logs."""
        if not events:
            return []
        valid = []
        for item in events:
            result = protocol.validate_envelope_safe(item, default_device_id=device_id)
            if result.ok:
                valid.append(item)
            else:
                logger.warning(
                    f"Node sync: dropping invalid event before send "
                    f"(event_uid={result.event_uid!r} error_code={result.error_code}): {result.error}"
                )
        return valid

    async def sync(self, device_id: str, cursor: int, events: list[dict] | None = None,
                    memory: list[dict] | None = None, limit: int = 500) -> dict | None:
        """
        POST /sync with an (optionally empty) events/memory batch. Returns
        the parsed response dict on success, or None on any failure — the
        caller (NodeSyncManager) decides what "no response" means for its
        own retry/backoff; this method never raises.
        """
        valid_events = self._filter_valid_events(events, device_id)
        try:
            payload = protocol.build_sync_request(
                device_id, cursor, events=valid_events, memory=memory, limit=limit,
            )
        except ProtocolError as e:
            logger.warning(f"Node sync: refusing to build a malformed request: {e}")
            return None

        url = f"{self.base_url}/sync"
        try:
            async with self._make_client() as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"Node sync failed: {e}")
            return None
        except ValueError as e:   # malformed JSON in an otherwise-2xx response
            logger.warning(f"Node sync returned unparseable JSON: {e}")
            return None

        try:
            protocol.check_response_version(data)
        except ProtocolError as e:
            logger.warning(f"Node sync response rejected: {e}")
            return None

        return data