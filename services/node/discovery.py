"""
services/node/discovery.py
Finds a reachable MayaNode instance. Step 1 discovery is deliberately
simple — no mDNS/broadcast, no service registry — just a configured
base_url (if set) tried first, then an ordered list of local candidates,
each probed with a short-timeout GET /status: the cheapest endpoint
MayaNode exposes (no DB write, unlike /health's sqlite check — see
MayaNode's api/status.py), so probing it is safe to do frequently and
doesn't compete with real health checks.
"""
import logging

import httpx

from config.settings import config

logger = logging.getLogger(__name__)


class NodeDiscovery:
    def __init__(self, cfg=None, transport: httpx.BaseTransport | None = None):
        self._cfg = cfg or config.node
        # transport: injection point for tests (httpx.MockTransport) —
        # never set in production, where httpx picks its own transport.
        self._transport = transport

    def _candidates(self) -> list[str]:
        ordered: list[str] = []
        if self._cfg.base_url:
            ordered.append(self._cfg.base_url)
        for url in self._cfg.discovery_candidates:
            if url not in ordered:
                ordered.append(url)
        return ordered

    async def discover(self) -> str | None:
        """
        Returns the first candidate base_url that answers /status, or
        None if none do. Never raises — every candidate probe is
        independently guarded so one bad/unreachable URL can't stop the
        rest from being tried.
        """
        for base_url in self._candidates():
            if await self._probe(base_url):
                logger.info(f"MayaNode discovered at {base_url}")
                return base_url
        logger.info("MayaNode not found among discovery candidates.")
        return None

    async def _probe(self, base_url: str) -> bool:
        url = f"{base_url.rstrip('/')}/status"
        try:
            async with httpx.AsyncClient(
                timeout=self._cfg.discovery_timeout, transport=self._transport
            ) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except httpx.HTTPError as e:
            logger.debug(f"MayaNode probe failed for {base_url}: {e}")
            return False