"""
services/node/client.py
Thin async HTTP client for MayaNode's /heartbeat and /sync endpoints (see
MayaNode's api/heartbeat.py, api/sync.py, services/sync_service.py). Every
public method is guarded — connection errors, timeouts, and non-2xx
responses are caught and reported to the caller as a plain failure result
(False / None), never raised — so a Node outage can never propagate into
MayaVE's own event loop or crash NodeSyncManager's loop.

Auth-ready: _headers() attaches `Authorization: Bearer <token>` whenever
config.node.auth_token is set. MayaNode does not check it yet (see its
api/sync.py docstring: "No session, no auth yet — device_id is trusted as
given") — this hook exists so turning on real auth later, on both sides,
is a config change rather than a protocol redesign here.
"""
import logging

import httpx

from config.settings import config

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

    async def sync(self, device_id: str, cursor: int, events: list[dict] | None = None,
                    memory: list[dict] | None = None, limit: int = 500) -> dict | None:
        """
        POST /sync with an (optionally empty) events/memory batch. Returns
        the parsed response dict on success, or None on any failure — the
        caller (NodeSyncManager) decides what "no response" means for its
        own retry/backoff; this method never raises.

        Step 1 callers always pass empty events/memory (see
        sync_manager.py) — the parameters exist so a later integration
        step can start pushing real payloads without changing this
        method's signature.
        """
        url = f"{self.base_url}/sync"
        payload = {
            "device_id": device_id,
            "cursor": cursor,
            "limit": limit,
            "events": events or [],
            "memory": memory or [],
        }
        try:
            async with self._make_client() as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"Node sync failed: {e}")
            return None
        except ValueError as e:   # malformed JSON in an otherwise-2xx response
            logger.warning(f"Node sync returned unparseable JSON: {e}")
            return None