"""
services/node/client.py
Thin async HTTP client for MayaNode's /health, /heartbeat, and /sync
endpoints.

Every public method is guarded — connection errors, timeouts, and non-2xx
responses are caught and reported to the caller as a plain failure result
(False / None / a HealthResult with healthy=False), never raised — so a
Node outage can never propagate into MayaVE's own event loop or crash
NodeSyncManager's loop.

Auth-ready: _headers() attaches `Authorization: Bearer <token>` whenever
config.node.auth_token is set. MayaNode does not check it yet (see
docs/PROTOCOL_CONTRACT.md's 'Authentication' section) — this hook exists
so turning on real auth later, on both sides, is a config change rather
than a protocol redesign here.

health(): calls MayaNode's GET /health (see docs/PROTOCOL_CONTRACT.md's
README — /health runs MayaNode's registered checks and returns 503 the
moment any one of them fails, 200 otherwise). This is distinct from
heartbeat() (POST /heartbeat, "I'm alive, register me") and from
NodeDiscovery's GET /status probe (cheapest liveness check, no DB
touch) — health() is the one endpoint that actually proves the
database/application is usable, not merely that the process is up.

sync(): builds its request body through protocol.build_sync_request()
(adds the protocol_version field to the wire shape) and validates any
outgoing events locally first — an invalid event is dropped with a
logged warning rather than either reaching the server malformed or
blocking the rest of the batch. The response's own protocol_version and
overall shape are validated by SyncResult.from_response(); a peer
speaking a version this build doesn't understand, or a response missing
required fields, is treated as a sync failure (logs a warning, returns
None), exactly like a connection failure — callers already handle that
uniformly.
"""
import logging
from dataclasses import dataclass, field

import httpx

from config.settings import config
from services.node import protocol
from services.node.protocol import ProtocolError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthResult:
    """Outcome of one GET /health call. Never raised — always returned.

    healthy:     True only for HTTP 200 with a parseable body.
    status_code: the HTTP status actually received, or None if the
                 request never got a response at all (connection
                 refused, timeout, or another transport-level failure).
    detail:      the parsed JSON body, when the response had one and it
                 was a JSON object; None otherwise (including when the
                 body was present but not valid/object JSON — a
                 malformed body must not crash the caller).
    error:       a short machine-checkable reason when healthy=False:
                 "connection_refused" | "timeout" | "unavailable"
                 (503) | "unexpected_status_<code>" | the raw
                 httpx error text for anything else. None when healthy.
    """
    healthy: bool
    status_code: int | None = None
    detail: dict | None = None
    error: str | None = None


@dataclass(frozen=True)
class SyncResult:
    """Typed, validated view of one POST /sync response (see
    docs/PROTOCOL_CONTRACT.md's 'Sync response' shape). Constructed only
    via from_response() so a caller never has to re-check basic shape
    (missing/wrong-typed cursor, unsupported protocol_version, ...)
    itself — SyncResult.from_response() raises ProtocolError/ValueError
    for that, and NodeClient.sync() turns either into a returned None.
    """
    device_id: str
    status: str
    accepted_events: list
    memory_results: list
    changes: dict
    cursor: int
    has_more: bool
    protocol_version: int

    @classmethod
    def from_response(cls, data: dict) -> "SyncResult":
        """Validate and adapt a raw /sync JSON body into a SyncResult.
        Raises ProtocolError (via protocol.check_response_version) for
        an unsupported protocol_version, or ValueError for any other
        malformed/missing-required-field response. Never returns a
        partially-valid result — either every field below round-trips
        cleanly or nothing is returned at all."""
        if not isinstance(data, dict):
            raise ValueError(f"sync response must be an object, got {type(data).__name__}")

        # Raises UnsupportedVersionError (a ProtocolError) on a peer
        # speaking a version this client build doesn't understand.
        protocol.check_response_version(data)

        cursor = data.get("cursor")
        if not isinstance(cursor, int):
            raise ValueError("sync response missing/invalid integer 'cursor'")

        changes = data.get("changes")
        if not isinstance(changes, dict):
            changes = {"events": [], "memory": []}

        accepted_events = data.get("accepted_events")
        if not isinstance(accepted_events, list):
            accepted_events = []

        memory_results = data.get("memory_results")
        if not isinstance(memory_results, list):
            memory_results = []

        return cls(
            device_id=data.get("device_id") or "",
            status=data.get("status") or "",
            accepted_events=accepted_events,
            memory_results=memory_results,
            changes={
                "events": changes.get("events") or [],
                "memory": changes.get("memory") or [],
            },
            cursor=cursor,
            has_more=bool(data.get("has_more", False)),
            protocol_version=data.get("protocol_version", 1),
        )


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

    async def health(self) -> HealthResult:
        """
        GET /health. Always returns a HealthResult — never raises, and
        safe to call whether or not MayaNode (or even the network path
        to it) is reachable at all:
          - connection refused / DNS failure  -> healthy=False, error="connection_refused"
          - timeout                            -> healthy=False, error="timeout"
          - HTTP 503 (a registered check failed) -> healthy=False, status_code=503, error="unavailable"
          - any other non-200                  -> healthy=False, error="unexpected_status_<code>"
          - HTTP 200 with a non-JSON/non-object body -> healthy=True, detail=None
          - HTTP 200 with a JSON object body    -> healthy=True, detail=<that body>
        """
        url = f"{self.base_url}/health"
        try:
            async with self._make_client() as client:
                resp = await client.get(url, headers=self._headers())
        except httpx.ConnectError as e:
            logger.debug(f"Node health check: connection refused: {e}")
            return HealthResult(healthy=False, error="connection_refused")
        except httpx.TimeoutException as e:
            logger.debug(f"Node health check: timed out: {e}")
            return HealthResult(healthy=False, error="timeout")
        except httpx.HTTPError as e:
            logger.debug(f"Node health check failed: {e}")
            return HealthResult(healthy=False, error=str(e))

        try:
            body = resp.json()
            detail = body if isinstance(body, dict) else None
        except ValueError:
            # Malformed/non-JSON body — never let this crash the caller.
            detail = None

        if resp.status_code == 200:
            return HealthResult(healthy=True, status_code=200, detail=detail)
        if resp.status_code == 503:
            return HealthResult(healthy=False, status_code=503, detail=detail, error="unavailable")
        return HealthResult(
            healthy=False, status_code=resp.status_code, detail=detail,
            error=f"unexpected_status_{resp.status_code}",
        )

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
                    memory: list[dict] | None = None, limit: int = 500) -> SyncResult | None:
        """
        POST /sync with an (optionally empty) events/memory batch. Returns
        a SyncResult on success, or None on any failure (connection
        error, timeout, non-2xx, unparsable JSON, or a response that
        fails SyncResult.from_response()'s validation) — the caller
        (NodeSyncManager) decides what "no response" means for its own
        retry/backoff; this method never raises.
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
            return SyncResult.from_response(data)
        except (ProtocolError, ValueError) as e:
            logger.warning(f"Node sync response rejected: {e}")
            return None