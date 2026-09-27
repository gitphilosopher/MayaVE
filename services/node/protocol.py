"""
services/node/protocol.py
==========================
MayaVE's own, self-contained implementation of the versioned MayaVE <->
MayaNode wire contract. See docs/PROTOCOL_CONTRACT.md for the
authoritative JSON/HTTP shape this module implements.

Repository boundary: MayaVE and MayaNode are SEPARATE repositories. This
module must never import from MayaNode's side (node/, api/, database/,
services/event_service.py, services/memory_service.py, ...) — none of
that is reachable from a real MayaVE deployment anyway. MayaNode has its
own independent implementation of the same contract at node/protocol.py
(a different repo). Compatibility between the two is enforced by both
sides conforming to docs/PROTOCOL_CONTRACT.md and by contract tests that
exercise both implementations against the same fixture payloads — not by
shared code.

This module owns:
  - protocol version negotiation (PROTOCOL_VERSION, is_version_supported)
  - the event envelope shape + a local event-type registry (only
    'protocol.ping' today — no real MayaVE event type exists yet)
  - building an outgoing /sync request body
  - checking an incoming /sync response's protocol_version
  - a defensive ordering check for incoming 'changes' arrays

Deliberately does not import node/clock.py (that's MayaNode-repo code) —
ISO-8601 parsing/formatting is reimplemented locally, matching the same
fixed-width-milliseconds convention documented in the contract so the
two sides produce byte-identical timestamp strings without sharing code.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Timestamp helpers (self-contained — mirrors node/clock.py's contract
#    documented in docs/PROTOCOL_CONTRACT.md, not its code) ────────────

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def utc_now_iso() -> str:
    return to_iso(datetime.now(timezone.utc))


# ── Versioning ──────────────────────────────────────────────────────────

PROTOCOL_VERSION = 1
SUPPORTED_PROTOCOL_VERSIONS = frozenset({1})


def is_version_supported(version) -> bool:
    """True if `version` (an int, or anything int()-able) is one this
    client build can speak. Never raises."""
    try:
        return int(version) in SUPPORTED_PROTOCOL_VERSIONS
    except (TypeError, ValueError):
        return False


# ── Errors ────────────────────────────────────────────────────────────────

class ProtocolError(ValueError):
    """Base class for every protocol-layer validation failure."""


class SchemaValidationError(ProtocolError):
    """An envelope or sync payload doesn't match the required shape."""


class UnknownEventTypeError(ProtocolError):
    """event_type isn't in this client's registry."""


class UnsupportedVersionError(ProtocolError):
    """A peer's declared protocol_version, or an event's schema_version,
    isn't one this client build can speak."""


_ERROR_CODES: dict[type, str] = {
    UnknownEventTypeError: "unknown_event_type",
    UnsupportedVersionError: "unsupported_schema_version",
    SchemaValidationError: "malformed",
}

# ── Envelope shape (must match docs/PROTOCOL_CONTRACT.md exactly) ────────

_EVENT_UID_MAX_LEN  = 128
_EVENT_TYPE_MAX_LEN = 64
_DEVICE_ID_MAX_LEN  = 64
_EVENT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class EventTypeSpec:
    name: str
    schema_version: int = 1
    required_fields: dict = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class EventEnvelope:
    event_uid: str
    event_type: str
    occurred_at: str
    payload: dict
    schema_version: int
    device_id: str | None = None

    def dedup_key(self) -> tuple[str, str]:
        return (self.device_id or "", self.event_uid)

    def to_dict(self) -> dict:
        return {
            "event_uid": self.event_uid,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "payload": self.payload,
            "schema_version": self.schema_version,
            "device_id": self.device_id,
        }


@dataclass(frozen=True)
class EventValidationResult:
    ok: bool
    envelope: EventEnvelope | None = None
    event_uid: str | None = None
    error_code: str | None = None
    error: str | None = None


# name -> EventTypeSpec. A future real MayaVE event type is registered
# here (client side) the same change it's registered in MayaNode's own
# node/protocol.py (server side) — see module docstring.
_REGISTRY: dict[str, EventTypeSpec] = {}


def register_event_type(name: str, *, schema_version: int = 1,
                         required_fields: dict | None = None,
                         description: str = "") -> EventTypeSpec:
    spec = EventTypeSpec(
        name=name, schema_version=schema_version,
        required_fields=dict(required_fields or {}), description=description,
    )
    existing = _REGISTRY.get(name)
    if existing is not None and existing != spec:
        raise ProtocolError(
            f"event type '{name}' is already registered with a different spec "
            f"({existing!r} != {spec!r})"
        )
    _REGISTRY[name] = spec
    return spec


def registered_event_types() -> dict[str, EventTypeSpec]:
    return dict(_REGISTRY)


register_event_type(
    "protocol.ping", schema_version=1, required_fields={},
    description="No-op connectivity/idempotency check event; carries no application data.",
)

# First real MayaVE-originated event type (Step 3) — records that a
# conversation turn was resolved to a given intent, for cross-device
# history/analytics. Kept intentionally minimal; more event types are
# added here (and mirrored in node/protocol.py, per module docstring)
# only as concrete need arises — see docs/PROTOCOL_CONTRACT.md.
register_event_type(
    "mayave.turn_completed", schema_version=1, required_fields={"intent": str},
    description="A MayaVE conversation turn was resolved to the given intent.",
)

def _check_json_serializable(payload: Any) -> None:
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as e:
        raise SchemaValidationError(f"payload is not JSON-serialisable: {e}") from e


def validate_envelope(data: dict, *, default_device_id: str | None = None) -> EventEnvelope:
    """Validate one raw event dict this client is about to send, against
    the same shape MayaNode's node/protocol.py enforces on receipt (see
    docs/PROTOCOL_CONTRACT.md) — validating locally first means a bad
    event is caught before a wasted round trip, not instead of server
    validation."""
    if not isinstance(data, dict):
        raise SchemaValidationError(f"event must be an object, got {type(data).__name__}")

    event_uid = data.get("event_uid")
    if not isinstance(event_uid, str) or not event_uid.strip():
        raise SchemaValidationError("event_uid is required and must be a non-empty string")
    if len(event_uid) > _EVENT_UID_MAX_LEN:
        raise SchemaValidationError(f"event_uid exceeds {_EVENT_UID_MAX_LEN} characters")

    event_type = data.get("event_type")
    if not isinstance(event_type, str) or not _EVENT_TYPE_RE.match(event_type):
        raise SchemaValidationError(
            "event_type is required and must look like '<namespace>.<name>' "
            "(lowercase letters/digits/underscore)"
        )
    if len(event_type) > _EVENT_TYPE_MAX_LEN:
        raise SchemaValidationError(f"event_type exceeds {_EVENT_TYPE_MAX_LEN} characters")

    occurred_at = data.get("occurred_at")
    if not isinstance(occurred_at, str) or not occurred_at:
        raise SchemaValidationError("occurred_at is required and must be a string")
    try:
        dt = parse_iso(occurred_at)
    except ValueError as e:
        raise SchemaValidationError(f"occurred_at is not a valid ISO-8601 datetime: {e}") from e
    if dt.tzinfo is None:
        raise SchemaValidationError("occurred_at must include a UTC offset")
    occurred_at = to_iso(dt)

    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"payload must be an object, got {type(payload).__name__}")
    _check_json_serializable(payload)

    schema_version = data.get("schema_version")
    if schema_version is not None and (not isinstance(schema_version, int) or schema_version < 1):
        raise SchemaValidationError("schema_version must be a positive integer when present")

    device_id = data.get("device_id") or default_device_id
    if device_id is not None:
        if not isinstance(device_id, str) or not device_id.strip():
            raise SchemaValidationError("device_id must be a non-empty string when present")
        if len(device_id) > _DEVICE_ID_MAX_LEN:
            raise SchemaValidationError(f"device_id exceeds {_DEVICE_ID_MAX_LEN} characters")

    spec = _REGISTRY.get(event_type)
    if spec is None:
        raise UnknownEventTypeError(f"event_type '{event_type}' is not a registered event type")

    effective_version = schema_version if schema_version is not None else spec.schema_version
    if effective_version != spec.schema_version:
        raise UnsupportedVersionError(
            f"event_type '{event_type}' schema_version {effective_version} is not supported "
            f"by this client (expects {spec.schema_version})"
        )

    for field_name, expected_type in spec.required_fields.items():
        if field_name not in payload:
            raise SchemaValidationError(
                f"event_type '{event_type}' payload is missing required field '{field_name}'"
            )
        if not isinstance(payload[field_name], expected_type):
            raise SchemaValidationError(
                f"event_type '{event_type}' payload field '{field_name}' has the wrong type"
            )

    return EventEnvelope(
        event_uid=event_uid, event_type=event_type, occurred_at=occurred_at,
        payload=payload, schema_version=spec.schema_version, device_id=device_id,
    )


def validate_envelope_safe(data, *, default_device_id: str | None = None) -> EventValidationResult:
    """Never raises — see node/protocol.py's identical contract on the
    server side. Used to filter outgoing events before they're sent."""
    event_uid = data.get("event_uid") if isinstance(data, dict) else None
    try:
        envelope = validate_envelope(data, default_device_id=default_device_id)
        return EventValidationResult(ok=True, envelope=envelope, event_uid=envelope.event_uid)
    except ProtocolError as e:
        code = next((c for t, c in _ERROR_CODES.items() if isinstance(e, t)), "malformed")
        return EventValidationResult(ok=False, event_uid=event_uid, error_code=code, error=str(e))
    except Exception as e:
        logger.error(f"Unexpected error validating event envelope: {e}", exc_info=True)
        return EventValidationResult(ok=False, event_uid=event_uid, error_code="internal_error", error=str(e))


def validate_event_batch(items: list, *, default_device_id: str | None = None) -> list[EventValidationResult]:
    return [validate_envelope_safe(item, default_device_id=default_device_id) for item in items]


# ── Sync request/response shape ──────────────────────────────────────────

def build_sync_request(device_id: str, cursor: int, events=None, memory=None,
                        limit: int = 500, protocol_version: int = PROTOCOL_VERSION) -> dict:
    """The canonical POST /sync body this client sends (see
    docs/PROTOCOL_CONTRACT.md's 'Sync request'). `events`/`memory` are
    passed through as plain dicts — callers should have already run
    them through validate_event_batch() if they want bad items dropped
    before this point (see services/node/client.py)."""
    if not isinstance(device_id, str) or not device_id.strip():
        raise SchemaValidationError("device_id is required and must be a non-empty string")
    if not isinstance(cursor, int) or cursor < 0:
        raise SchemaValidationError("cursor must be a non-negative integer")
    if not isinstance(limit, int) or limit < 1:
        raise SchemaValidationError("limit must be a positive integer")
    return {
        "protocol_version": protocol_version,
        "device_id": device_id,
        "cursor": cursor,
        "limit": limit,
        "events": list(events or []),
        "memory": list(memory or []),
    }


def check_response_version(response: dict) -> None:
    """Raise UnsupportedVersionError if the peer's declared
    protocol_version isn't one this client can speak. A response with no
    protocol_version field at all is tolerated as version 1 (compat with
    a server running before this field existed)."""
    if not isinstance(response, dict):
        raise SchemaValidationError(f"sync response must be an object, got {type(response).__name__}")
    version = response.get("protocol_version", 1)
    if not is_version_supported(version):
        raise UnsupportedVersionError(f"peer protocol_version {version!r} is not supported by this client")


def validate_change_ordering(changes: list) -> None:
    """Defensive check that a 'changes' array received FROM the server is
    strictly increasing by 'seq' — the ordering guarantee the whole
    cursor model depends on (see docs/PROTOCOL_CONTRACT.md). Raises
    SchemaValidationError on any violation."""
    last = None
    for item in changes:
        if not isinstance(item, dict) or not isinstance(item.get("seq"), int):
            raise SchemaValidationError(f"change item is missing an integer 'seq': {item!r}")
        seq = item["seq"]
        if last is not None and seq <= last:
            raise SchemaValidationError(f"changes are not strictly increasing by seq: {last} -> {seq}")
        last = seq