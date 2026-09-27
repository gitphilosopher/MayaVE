"""
services/node/memory_producer.py
Application-facing memory-write producer for MayaVE's side of the
MayaVE <-> MayaNode sync pipeline (Step 3). Mirrors events.py's shape
for key/value memory state: callers never talk to
services/node/client.py or the /sync endpoint directly, and don't need
MayaNode (or even config.node.enabled) to be on to record a write — it
just waits in the local outbox until a sync round picks it up.

Preserves the existing memory-write semantics used on the MayaNode side
(services/memory_service.py / api/memory.py): one (device_id, key) pair
holds exactly one current value, last-write-wins on `updated_at`. This
module enforces that same rule LOCALLY too — only the latest queued
value per key is kept in the outbox, since an intermediate superseded
value could never win the server's own conflict check anyway (see
services/memory_service.py's sync_set_value).
"""

import json
import logging

from services.node import protocol
from services.node.outbox import OutboxStore, outbox_store

logger = logging.getLogger(__name__)

_KEY_MAX_LEN = 128


def record_memory(
    key: str,
    value,
    *,
    updated_at: str | None = None,
    store: OutboxStore | None = None,
    state_dir: str | None = None,
) -> bool:
    """
    Queue a (key, value) memory write for eventual sync to MayaNode.

    `updated_at`, if given, must be a moment the caller can vouch for
    (e.g. an earlier offline edit's real timestamp) — omit it to stamp
    "now". `store`/`state_dir` are test/advanced-use hooks.

    Returns True if the write was queued, False if it was dropped
    (empty/oversized key, or a value that can't round-trip through
    JSON — the server would reject it anyway, see
    docs/PROTOCOL_CONTRACT.md's Memory write shape).
    """
    if not isinstance(key, str) or not key.strip():
        logger.warning("Dropping memory write with an empty key.")
        return False
    if len(key) > _KEY_MAX_LEN:
        logger.warning(f"Dropping memory write — key exceeds {_KEY_MAX_LEN} characters: '{key[:40]}...'")
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        logger.warning(f"Dropping memory write '{key}' — value is not JSON-serialisable: {e}")
        return False

    occurred = updated_at or protocol.utc_now_iso()
    target = store or outbox_store(state_dir)
    target.add_memory(key, value, occurred)
    return True