# MayaVE ⇄ MayaNode Wire Contract (v1)

This document is the single source of truth for the JSON/HTTP contract
between MayaVE (client) and MayaNode (server). **There is no shared code
between the two repositories.** MayaNode implements its half in
`node/protocol.py`; MayaVE implements its half in
`services/node/protocol.py`. Both must conform to this document
independently. If the two ever disagree, this document wins — fix
whichever implementation drifted.

## Transport

One endpoint, reused from Step 1: `POST /sync`, JSON body/response.
`GET /status` (discovery) and `POST /heartbeat` (connection) are
unchanged from Step 1 and not part of this contract.

## Protocol versioning

Every `/sync` request and response carries an integer `protocol_version`
field.

- Current version: **1**.
- A request/response with no `protocol_version` field is treated as
  version 1 (tolerates a peer built before this field existed).
- A peer receiving a `protocol_version` it does not support MUST reject
  the request (server: HTTP 422) or discard the response (client: treat
  as a sync failure, same as a connection error) rather than guessing at
  compatibility.
- Bumping this number is a breaking wire-shape change; both repos must
  update their own `SUPPORTED_PROTOCOL_VERSIONS` together as part of a
  coordinated rollout.

## Sync request

```json
{
  "protocol_version": 1,
  "device_id": "mayave-<uuid>",
  "cursor": 0,
  "limit": 500,
  "events": [ <EventEnvelope>, ... ],
  "memory": [ <MemoryWrite>, ... ]
}
```

- `device_id`: required, non-empty, ≤64 chars.
- `cursor`: required, integer ≥ 0. The last `seq` this client has fully
  applied; the server returns everything newer.
- `limit`: 1–1000, default 500.
- `events` / `memory`: optional, default `[]`.

## Event envelope

Every item in `events` is an **EventEnvelope**:

```json
{
  "event_uid": "mayave-2026-01-01T00:00:00Z-abc123",
  "event_type": "protocol.ping",
  "schema_version": 1,
  "occurred_at": "2026-01-01T00:00:00.000+00:00",
  "device_id": "mayave-<uuid>",
  "payload": { }
}
```

| Field | Required | Rules |
|---|---|---|
| `event_uid` | yes | non-empty string, ≤128 chars. The idempotency key — a retried push of the same `event_uid` MUST be a no-op, not a duplicate row. |
| `event_type` | yes | `"<namespace>.<name>"`, lowercase letters/digits/underscore only, ≤64 chars. Must be a type the SERVER recognizes; an unrecognized type is a rejection, not silent storage. |
| `schema_version` | no | positive integer. Defaults to the event type's current schema version. A mismatch is a rejection (`unsupported_schema_version`) — no cross-version migration exists yet. |
| `occurred_at` | yes | UTC ISO-8601 with an explicit offset. |
| `device_id` | no | defaults to the request's own `device_id` when absent. |
| `payload` | no | JSON object, defaults to `{}`. Its required sub-fields are defined per `event_type` by whichever side registers that type. |

Only one event type is defined by this contract today:
**`protocol.ping`** (`schema_version: 1`, no required payload fields) —
a connectivity/idempotency dial tone, not real application data. No
MayaVE event types are defined yet; adding one means extending this
document plus both implementations' registries in the same change.

## Memory write (unchanged from Step 1)

```json
{ "key": "some.key", "value": <any JSON>, "updated_at": "<UTC ISO-8601>", "device_id": "..." }
```

## Sync response

```json
{
  "protocol_version": 1,
  "device_id": "mayave-<uuid>",
  "status": "ok",
  "accepted_events": [
    {"event_uid": "...", "status": "accepted", "id": 1, "seq": 10},
    {"event_uid": "...", "status": "duplicate", "id": 1, "seq": 10},
    {"event_uid": "...", "status": "rejected", "error_code": "malformed", "error": "..."}
  ],
  "memory_results": [ ... ],
  "changes": {
    "events": [ {"seq": 11, ...}, {"seq": 12, ...} ],
    "memory": [ {"seq": 13, ...} ]
  },
  "cursor": 12,
  "has_more": false
}
```

- `accepted_events[i].status` ∈ `accepted | duplicate | rejected`.
  `rejected` carries `error_code` ∈ `malformed | unknown_event_type |
  unsupported_schema_version` plus a human-readable `error`. **One
  rejected event never fails the rest of the batch.**
- `changes.events` / `changes.memory`: each item carries an integer
  `seq`, strictly increasing within its own array — this is what makes
  replay from `cursor` deterministic and gap-free.
- `cursor`: the highest `seq` delivered in this response (or the
  request's own cursor, unchanged, if nothing new was delivered).

## Authentication (not yet enforced)

Every request MAY carry `Authorization: Bearer <token>`. The server does
not validate this yet — the header is defined now so turning on real
auth later is a config change on both sides, not a protocol change.

## Safety requirements for both implementations

- Malformed JSON, missing required fields, wrong types, an unknown
  `event_type`, or an unsupported `schema_version`/`protocol_version`
  MUST be rejected without raising an unhandled exception — a bad event
  becomes a `rejected` item; a bad request-level field (e.g.
  `protocol_version`) becomes a clean 4xx.
- Neither side may crash or hang the whole batch because one item in it
  was malformed.