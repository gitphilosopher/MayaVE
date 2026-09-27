"""
test_mayave_protocol.py
Focused tests for MayaVE's own protocol implementation
(services/node/protocol.py) and its wiring into services/node/client.py:
schema validation, versioning, malformed payloads, and client behavior
(request shape, event filtering, response-version rejection). Stdlib
unittest + unittest.mock only, matching test_node_integration.py's style.

Per docs/PROTOCOL_CONTRACT.md, MayaNode and MayaVE are separate
repositories with independent implementations of the same wire shape —
this file tests ONLY the MayaVE (client) side. See test_node_protocol.py
for the MayaNode (server) side, and test_protocol_compatibility.py for a
contract test that exercises both.
"""
import tempfile
import unittest
from datetime import datetime, timezone

import httpx

from config.settings import NodeConfig
from services.node.client import NodeClient
from services.node.protocol import (
    EventEnvelope, ProtocolError, SchemaValidationError, UnknownEventTypeError,
    UnsupportedVersionError, PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
    build_sync_request, check_response_version, is_version_supported,
    register_event_type, registered_event_types, validate_change_ordering,
    validate_envelope, validate_envelope_safe, validate_event_batch,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_cfg(tmp_dir: str, **overrides) -> NodeConfig:
    base = dict(
        enabled=True, base_url=None,
        discovery_candidates=["http://127.0.0.1:8000"],
        discovery_timeout=0.5, request_timeout=0.5,
        sync_interval_s=0.02, initial_backoff_s=0.01, max_backoff_s=0.02,
        auth_token=None, state_dir=tmp_dir,
    )
    base.update(overrides)
    return NodeConfig(**base)


register_event_type("test.sample", schema_version=1, required_fields={"note": str})


class EnvelopeValidationTests(unittest.TestCase):
    def test_valid_envelope_round_trips(self):
        raw = {
            "event_uid": "evt-1", "event_type": "test.sample",
            "occurred_at": _now_iso(), "payload": {"note": "hello"},
        }
        envelope = validate_envelope(raw)
        self.assertIsInstance(envelope, EventEnvelope)
        self.assertEqual(envelope.schema_version, 1)

    def test_missing_event_uid_rejected(self):
        raw = {"event_type": "test.sample", "occurred_at": _now_iso(), "payload": {"note": "x"}}
        with self.assertRaises(SchemaValidationError):
            validate_envelope(raw)

    def test_unknown_event_type_rejected(self):
        raw = {
            "event_uid": "evt-2", "event_type": "nope.unregistered",
            "occurred_at": _now_iso(), "payload": {},
        }
        with self.assertRaises(UnknownEventTypeError):
            validate_envelope(raw)

    def test_unsupported_schema_version_rejected(self):
        raw = {
            "event_uid": "evt-3", "event_type": "test.sample", "schema_version": 99,
            "occurred_at": _now_iso(), "payload": {"note": "x"},
        }
        with self.assertRaises(UnsupportedVersionError):
            validate_envelope(raw)

    def test_missing_utc_offset_rejected(self):
        raw = {
            "event_uid": "evt-4", "event_type": "test.sample",
            "occurred_at": "2024-01-01T00:00:00", "payload": {"note": "x"},
        }
        with self.assertRaises(SchemaValidationError):
            validate_envelope(raw)


class SafeValidationTests(unittest.TestCase):
    def test_safe_validation_never_raises_on_garbage(self):
        for item in [{}, "not a dict", None, 42, {"event_uid": 123}]:
            with self.subTest(item=item):
                result = validate_envelope_safe(item)
                self.assertFalse(result.ok)
                self.assertIsNotNone(result.error_code)

    def test_batch_reports_one_rejection_without_losing_the_rest(self):
        good = {
            "event_uid": "evt-ok", "event_type": "test.sample",
            "occurred_at": _now_iso(), "payload": {"note": "fine"},
        }
        bad = {
            "event_uid": "evt-bad", "event_type": "test.sample",
            "occurred_at": _now_iso(), "payload": {},
        }
        results = validate_event_batch([good, bad])
        self.assertTrue(results[0].ok)
        self.assertFalse(results[1].ok)


class VersionTests(unittest.TestCase):
    def test_current_version_is_supported(self):
        self.assertTrue(is_version_supported(PROTOCOL_VERSION))
        self.assertIn(PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS)

    def test_check_response_version_tolerates_missing_field(self):
        check_response_version({"cursor": 1})   # must not raise

    def test_check_response_version_rejects_unsupported(self):
        with self.assertRaises(UnsupportedVersionError):
            check_response_version({"protocol_version": 999, "cursor": 1})


class SyncRequestBuilderTests(unittest.TestCase):
    def test_build_sync_request_shape(self):
        payload = build_sync_request("device-1", 5, events=[{"a": 1}], memory=[{"b": 2}], limit=10)
        self.assertEqual(payload["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(payload["device_id"], "device-1")
        self.assertEqual(payload["cursor"], 5)
        self.assertEqual(payload["events"], [{"a": 1}])

    def test_build_sync_request_rejects_negative_cursor(self):
        with self.assertRaises(SchemaValidationError):
            build_sync_request("device-1", -1)


class OrderingTests(unittest.TestCase):
    def test_validate_change_ordering_accepts_strictly_increasing(self):
        validate_change_ordering([{"seq": 1}, {"seq": 2}])

    def test_validate_change_ordering_rejects_non_increasing(self):
        with self.assertRaises(SchemaValidationError):
            validate_change_ordering([{"seq": 2}, {"seq": 2}])


class NodeClientProtocolTests(unittest.IsolatedAsyncioTestCase):
    """NodeClient behavior added in Step 2: outgoing events are validated
    and bad ones dropped, the request carries protocol_version, and an
    unsupported response protocol_version is treated as a sync failure."""

    async def test_sync_request_includes_protocol_version(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"cursor": 1, "protocol_version": PROTOCOL_VERSION})

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            await client.sync("device-1", cursor=0)
            self.assertEqual(captured["body"]["protocol_version"], PROTOCOL_VERSION)

    async def test_invalid_outgoing_event_is_dropped_not_sent(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"cursor": 1})

        good = {
            "event_uid": "evt-ok", "event_type": "test.sample",
            "occurred_at": _now_iso(), "payload": {"note": "fine"},
        }
        bad = {
            "event_uid": "evt-bad", "event_type": "test.sample",
            "occurred_at": _now_iso(), "payload": {},   # missing 'note'
        }

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            await client.sync("device-1", cursor=0, events=[good, bad])
            sent_uids = [e["event_uid"] for e in captured["body"]["events"]]
            self.assertIn("evt-ok", sent_uids)
            self.assertNotIn("evt-bad", sent_uids)

    async def test_unsupported_response_version_treated_as_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"cursor": 1, "protocol_version": 999})

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            result = await client.sync("device-1", cursor=0)
            self.assertIsNone(result)

    async def test_response_missing_version_field_is_tolerated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"cursor": 1})   # no protocol_version at all

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            result = await client.sync("device-1", cursor=0)
            self.assertEqual(result["cursor"], 1)


if __name__ == "__main__":
    unittest.main()