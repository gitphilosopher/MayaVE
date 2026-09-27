"""
services/node/events.py
Application-facing event producer for MayaVE's side of the MayaVE <->
MayaNode sync pipeline (Step 3).

This is the ONLY place application code should touch to record an event
bound for MayaNode — callers never talk to services/node/client.py or
the /sync endpoint directly, and no /sync calls should be scattered
elsewhere in MayaVE. record_event() validates the event locally against
services/node/protocol.py's registry (the same validation MayaNode
itself performs on receipt — see docs/PROTOCOL_CONTRACT.md), stamps a
UTC timestamp and a unique event_uid, and hands the resulting envelope
to the local outbox (services/node/outbox.py) for eventual delivery by
NodeSyncManager. Nothing here talks to the network, blocks the caller,
or requires MayaNode (or even config.node.enabled) to be on — an event
recorded while sync is disabled just waits harmlessly in the outbox.
"""

import logging
import uuid

from services.node import protocol
from services.node.outbox import OutboxStore, outbox_store

logger = logging.getLogger(__name__)


def _new_event_uid(event_type: str) -> str:
    return f"mayave-{event_type}-{uuid.uuid4()}"


def record_event(
    event_type: str,
    payload: dict | None = None,
    *,
    occurred_at: str | None = None,
    device_id: str | None = None,
    event_uid: str | None = None,
    store: OutboxStore | None = None,
    state_dir: str | None = None,
) -> bool:
    """
    Record one application event for eventual sync to MayaNode.

    `event_type` must already be registered (see
    services/node/protocol.py's register_event_type calls) — an
    unregistered type, or a payload missing that type's required
    fields, is logged and dropped rather than raised, so a bad call
    site never crashes the caller.

    `event_uid`, if given, lets a caller make its OWN retries idempotent
    (recording "the same" occurrence twice with the same event_uid is a
    no-op the second time) — omit it to get a fresh random id, the
    right default for a genuinely new occurrence.

    `store`/`state_dir` are test/advanced-use hooks; application code
    should normally pass neither and let this resolve to the shared
    outbox at config.node.state_dir.

    Returns True if the event was queued, False if it was dropped.
    """
    occurred = occurred_at or protocol.utc_now_iso()
    raw: dict = {
        "event_uid": event_uid or _new_event_uid(event_type),
        "event_type": event_type,
        "occurred_at": occurred,
        "payload": payload or {},
    }
    if device_id:
        raw["device_id"] = device_id

    result = protocol.validate_envelope_safe(raw, default_device_id=device_id)
    if not result.ok:
        logger.warning(
            f"Dropping invalid event '{event_type}' (uid={raw['event_uid']}): {result.error}"
        )
        return False

    target = store or outbox_store(state_dir)
    return target.add_event(result.envelope.to_dict())