"""
tests/test_node_integration.py
Focused tests for Step 1 of the MayaVE <-> MayaNode integration
(services/node/*): discovery, connection (heartbeat), sync + cursor
persistence, and offline recovery. Stdlib-only (unittest +
unittest.mock) — no new test-framework dependency added.

Network is never actually touched: httpx.MockTransport (part of httpx,
already a project dependency) simulates MayaNode's responses, including
simulated connection failures, without a real server or extra sockets.

Timing note: NodeSyncManager.run() is a real, continuously-looping
coroutine — under a fast test config (small sync_interval_s/backoff) it
can iterate many more times than a test cares about before mgr.stop()
takes effect. Tests here never rely on an exact number of loop
iterations: mocks that model "the interesting sequence of events" use
_sequence_then_repeat() (returns the final value forever once the
sequence is exhausted, instead of raising StopIteration/StopAsyncIteration
on an unanticipated extra call), and manager-level assertions poll for the
actual condition (_wait_until()) rather than sleeping a guessed duration.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from config.settings import NodeConfig
from services.node.client import NodeClient
from services.node.discovery import NodeDiscovery
from services.node.identity import resolve_device_id
from services.node.sync_manager import NodeSyncManager
from services.node.sync_state import SyncStateStore


def _make_cfg(tmp_dir: str, **overrides) -> NodeConfig:
    base = dict(
        enabled=True,
        base_url=None,
        discovery_candidates=["http://127.0.0.1:8000", "http://localhost:8000"],
        discovery_timeout=0.5,
        request_timeout=0.5,
        sync_interval_s=0.02,
        initial_backoff_s=0.01,
        max_backoff_s=0.02,
        auth_token=None,
        state_dir=tmp_dir,
    )
    base.update(overrides)
    return NodeConfig(**base)


def _sequence_then_repeat(*values):
    """
    An AsyncMock side_effect that yields `values` in order, then keeps
    returning the LAST value forever on every subsequent call — instead
    of a plain list side_effect, which raises StopIteration/
    StopAsyncIteration once exhausted. Needed because NodeSyncManager's
    loop may call a mocked method more times than the test's "interesting"
    sequence covers before mgr.stop() takes effect.
    """
    remaining = list(values)

    async def _effect(*args, **kwargs):
        if len(remaining) > 1:
            return remaining.pop(0)
        result = remaining[0]
        if isinstance(result, BaseException):
            raise result
        return result

    return _effect


async def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> None:
    """Polls `predicate()` until it's truthy or `timeout` elapses (then
    raises AssertionError) — avoids guessing a fixed sleep duration
    against a variable-speed background loop."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"Condition not met within {timeout}s")


# ══════════════════════════════════════════════════════════════════════
# Discovery
# ══════════════════════════════════════════════════════════════════════

class NodeDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_discover_falls_back_to_second_candidate(self):
        """First candidate is unreachable; discovery must still find the
        second one rather than giving up after the first failure."""
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "127.0.0.1":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            discovery = NodeDiscovery(cfg, transport=httpx.MockTransport(handler))
            result = await discovery.discover()
            self.assertEqual(result, "http://localhost:8000")

    async def test_discover_returns_none_when_nothing_answers(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            discovery = NodeDiscovery(cfg, transport=httpx.MockTransport(handler))
            result = await discovery.discover()
            self.assertIsNone(result)

    async def test_configured_base_url_is_tried_first(self):
        seen_hosts = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_hosts.append(request.url.host)
            return httpx.Response(200)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, base_url="http://mynode.local:9000")
            discovery = NodeDiscovery(cfg, transport=httpx.MockTransport(handler))
            result = await discovery.discover()
            self.assertEqual(result, "http://mynode.local:9000")
            self.assertEqual(seen_hosts[0], "mynode.local")


# ══════════════════════════════════════════════════════════════════════
# Connection (heartbeat) + sync
# ══════════════════════════════════════════════════════════════════════

class NodeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_success(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=transport)
            self.assertTrue(await client.heartbeat("device-1"))

    async def test_heartbeat_failure_never_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            self.assertFalse(await client.heartbeat("device-1"))

    async def test_sync_success_returns_parsed_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"cursor": 42, "has_more": False, "changes": {"events": [], "memory": []}})

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            result = await client.sync("device-1", cursor=0)
            self.assertEqual(result["cursor"], 42)

    async def test_sync_sends_empty_payload_by_default(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"cursor": 1})

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            await client.sync("device-1", cursor=0)
            self.assertEqual(captured["body"]["events"], [])
            self.assertEqual(captured["body"]["memory"], [])
            self.assertEqual(captured["body"]["device_id"], "device-1")

    async def test_sync_attaches_bearer_token_when_configured(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"cursor": 1})

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, auth_token="secret-token")
            client = NodeClient("http://127.0.0.1:8000", cfg, transport=httpx.MockTransport(handler))
            await client.sync("device-1", cursor=0)
            self.assertEqual(captured["auth"], "Bearer secret-token")

    async def test_sync_connection_failure_returns_none(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with tempfile.TemporaryDirectory() as tmp:
            client = NodeClient("http://127.0.0.1:8000", _make_cfg(tmp), transport=httpx.MockTransport(handler))
            result = await client.sync("device-1", cursor=0)
            self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════
# Cursor persistence
# ══════════════════════════════════════════════════════════════════════

class SyncStateStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_cursor_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_a = SyncStateStore(tmp)
            store_a.load()
            self.assertEqual(store_a.cursor, 0)
            await store_a.advance(17)

            store_b = SyncStateStore(tmp)   # simulates a restart
            store_b.load()
            self.assertEqual(store_b.cursor, 17)

    async def test_cursor_never_moves_backwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SyncStateStore(tmp)
            store.load()
            await store.advance(10)
            await store.advance(3)     # stale/out-of-order — must be ignored
            self.assertEqual(store.cursor, 10)
            await store.advance(10)    # equal — also a no-op
            self.assertEqual(store.cursor, 10)

    async def test_corrupt_state_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sync_state.json"
            path.write_text("not valid json {{{", encoding="utf-8")
            store = SyncStateStore(tmp)
            store.load()   # must not raise
            self.assertEqual(store.cursor, 0)

    async def test_write_failure_is_non_fatal(self):
        """If the state directory can't be written to (e.g. blocked by a
        file), advance() must still update the in-memory cursor rather
        than raising."""
        with tempfile.TemporaryDirectory() as tmp:
            blocker_path = Path(tmp) / "blocked"
            blocker_path.write_text("i am a file, not a directory", encoding="utf-8")
            store = SyncStateStore(str(blocker_path))   # state_dir points at a file
            store.load()
            await store.advance(5)   # must not raise
            self.assertEqual(store.cursor, 5)


# ══════════════════════════════════════════════════════════════════════
# Device identity
# ══════════════════════════════════════════════════════════════════════

class DeviceIdentityTests(unittest.TestCase):
    def test_device_id_persists_across_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = resolve_device_id(tmp)
            second = resolve_device_id(tmp)
            self.assertEqual(first, second)
            self.assertTrue(first.startswith("mayave-"))

    def test_unwritable_directory_falls_back_to_ephemeral_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker_path = Path(tmp) / "blocked"
            blocker_path.write_text("i am a file, not a directory", encoding="utf-8")
            device_id = resolve_device_id(str(blocker_path))   # can't mkdir here
            self.assertTrue(device_id.startswith("mayave-ephemeral-"))


# ══════════════════════════════════════════════════════════════════════
# NodeSyncManager — end-to-end offline recovery
# ══════════════════════════════════════════════════════════════════════

class NodeSyncManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_manager_returns_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = NodeSyncManager(_make_cfg(tmp, enabled=False))
            await asyncio.wait_for(mgr.run(), timeout=1.0)   # must return, not hang
            self.assertFalse(mgr.connected)

    async def test_recovers_from_initial_discovery_failure_and_advances_cursor(self):
        """Discovery fails once (Node not up yet), then succeeds; sync
        then runs at least twice, advancing the persisted cursor each
        time, and settles on the final cursor value."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = NodeSyncManager(_make_cfg(tmp))
            mgr._discovery.discover = AsyncMock(
                side_effect=_sequence_then_repeat(None, "http://test-node:8000")
            )

            with patch.object(NodeClient, "heartbeat", new=AsyncMock(return_value=True)), \
                 patch.object(NodeClient, "sync", new=AsyncMock(
                     side_effect=_sequence_then_repeat(
                         {"cursor": 5, "has_more": False},
                         {"cursor": 9, "has_more": False},
                     )
                 )):
                task = asyncio.create_task(mgr.run())
                try:
                    await _wait_until(lambda: mgr.cursor == 9, timeout=2.0)
                    # Give one more successful pass a chance to land so
                    # `connected` reflects the settled state, not a
                    # mid-flight moment right after the cursor updated.
                    await _wait_until(lambda: mgr.connected, timeout=2.0)
                finally:
                    mgr.stop()
                    await asyncio.wait_for(task, timeout=2.0)

            self.assertTrue(mgr.connected)
            self.assertEqual(mgr.cursor, 9)
            # Cursor must have survived to disk too, not just in-memory.
            reloaded = SyncStateStore(tmp)
            reloaded.load()
            self.assertEqual(reloaded.cursor, 9)

    async def test_sync_failure_does_not_crash_and_forces_rediscovery(self):
        """A heartbeat failure after a successful connection must not
        raise, and must cause the next pass to rediscover rather than
        keep hammering a possibly-moved/dead node."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = NodeSyncManager(_make_cfg(tmp))
            discover_mock = AsyncMock(
                side_effect=_sequence_then_repeat("http://test-node:8000")
            )
            mgr._discovery.discover = discover_mock

            with patch.object(NodeClient, "heartbeat", new=AsyncMock(
                     side_effect=_sequence_then_repeat(True, False, True)
                 )), \
                 patch.object(NodeClient, "sync", new=AsyncMock(
                     return_value={"cursor": 3, "has_more": False}
                 )):
                task = asyncio.create_task(mgr.run())
                try:
                    # Rediscovery only happens after a failed pass sets
                    # self._client = None — wait for that to have
                    # actually occurred (call_count > 1) rather than a
                    # fixed sleep.
                    await _wait_until(lambda: discover_mock.call_count >= 2, timeout=2.0)
                    await _wait_until(lambda: mgr.cursor == 3, timeout=2.0)
                finally:
                    mgr.stop()
                    await asyncio.wait_for(task, timeout=2.0)

            self.assertGreaterEqual(discover_mock.call_count, 2)
            self.assertEqual(mgr.cursor, 3)

    async def test_never_raises_out_of_run_on_unexpected_exception(self):
        """Even a bug inside discovery/client code must not propagate out
        of run() — it should be caught, logged, and backed off from."""
        with tempfile.TemporaryDirectory() as tmp:
            # Larger backoff than the other tests here so a short-lived
            # test doesn't spend its whole run spinning on the same
            # immediate failure — the point of this test is just that
            # run() survives at least one such failure, not how many.
            cfg = _make_cfg(tmp, initial_backoff_s=0.05, max_backoff_s=0.1)
            mgr = NodeSyncManager(cfg)
            call_count = 0

            async def _boom(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("boom")

            mgr._discovery.discover = AsyncMock(side_effect=_boom)
            task = asyncio.create_task(mgr.run())
            try:
                await _wait_until(lambda: call_count >= 1, timeout=2.0)
            finally:
                mgr.stop()
                # stop() only takes effect after the loop's current sleep
                # finishes — give it up to max_backoff_s plus headroom.
                await asyncio.wait_for(task, timeout=2.0)

            # Must complete cleanly — no exception raised out of the task.
            self.assertIsNone(task.exception())


if __name__ == "__main__":
    unittest.main()