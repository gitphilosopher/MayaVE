"""
services/test_ws_disconnect.py
Regression tests for Opt20 last-client disconnect recovery and multi-client safety.
"""

import asyncio
import json
import time
import pytest
from services.ws_server import MayaWebSocketServer


class MockWebSocket:
    def __init__(self, remote_address=("127.0.0.1", 12345)):
        self.remote_address = remote_address
        self.sent_messages = []
        self._inbox = asyncio.Queue()
        self.closed = False

    async def send(self, data: str):
        if self.closed:
            raise Exception("Connection closed")
        self.sent_messages.append(data)

    async def recv(self):
        msg = await self._inbox.get()
        if msg is None:
            raise StopAsyncIteration
        return msg

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.recv()
        except StopAsyncIteration:
            raise StopAsyncIteration

    def close(self):
        self.closed = True
        self._inbox.put_nowait(None)


def test_zero_clients_immediate_exit():
    """When no clients are connected, wait_for_audio_done returns immediately."""
    server = MayaWebSocketServer(host="127.0.0.1", port=0)

    async def run():
        t0 = time.perf_counter()
        res = await server.wait_for_audio_done(timeout=5.0)
        dur = time.perf_counter() - t0
        assert res is False
        assert dur < 0.05  # sub-50ms immediate return

    asyncio.run(run())


def test_healthy_client_ack():
    """When a client connects and sends audio_done, wait_for_audio_done returns True cleanly."""
    server = MayaWebSocketServer(host="127.0.0.1", port=0)
    ws = MockWebSocket()

    async def run():
        handler_task = asyncio.create_task(server._handler(ws))
        await asyncio.sleep(0.01)
        assert ws in server._clients

        async def send_ack_later():
            await asyncio.sleep(0.05)
            await ws._inbox.put(json.dumps({"type": "audio_done"}))

        ack_task = asyncio.create_task(send_ack_later())

        t0 = time.perf_counter()
        res = await server.wait_for_audio_done(timeout=5.0)
        dur = time.perf_counter() - t0

        assert res is True
        assert 0.04 < dur < 0.5

        ws.close()
        await handler_task
        await ack_task

    asyncio.run(run())


def test_last_client_disconnect_unblocks_wait():
    """Opt20: When the last client disconnects mid-wait, wait_for_audio_done unblocks immediately."""
    server = MayaWebSocketServer(host="127.0.0.1", port=0)
    ws = MockWebSocket()

    async def run():
        handler_task = asyncio.create_task(server._handler(ws))
        await asyncio.sleep(0.01)
        assert ws in server._clients

        async def disconnect_later():
            await asyncio.sleep(0.05)
            ws.close()

        disc_task = asyncio.create_task(disconnect_later())

        t0 = time.perf_counter()
        # Even with large timeout (e.g. 10.0s), disconnect releases wait in <0.2s
        res = await server.wait_for_audio_done(timeout=10.0)
        dur = time.perf_counter() - t0

        assert dur < 0.2
        assert len(server._clients) == 0

        await handler_task
        await disc_task

    asyncio.run(run())


def test_multi_client_safety():
    """When 2 clients are connected and 1 disconnects, the wait is NOT released until client 2 acks."""
    server = MayaWebSocketServer(host="127.0.0.1", port=0)
    ws1 = MockWebSocket(remote_address=("127.0.0.1", 10001))
    ws2 = MockWebSocket(remote_address=("127.0.0.1", 10002))

    async def run():
        h1 = asyncio.create_task(server._handler(ws1))
        h2 = asyncio.create_task(server._handler(ws2))
        await asyncio.sleep(0.01)
        assert len(server._clients) == 2

        # Client 1 disconnects at 0.05s, Client 2 sends ack at 0.12s
        async def client_lifecycle():
            await asyncio.sleep(0.05)
            ws1.close()  # ws1 disconnects, but ws2 is still active!
            await asyncio.sleep(0.07)
            await ws2._inbox.put(json.dumps({"type": "audio_done"}))

        life_task = asyncio.create_task(client_lifecycle())

        t0 = time.perf_counter()
        res = await server.wait_for_audio_done(timeout=5.0)
        dur = time.perf_counter() - t0

        # Wait must NOT release at 0.05s when ws1 disconnects; must release at ~0.12s when ws2 acks
        assert res is True
        assert dur >= 0.10

        ws2.close()
        await h1
        await h2
        await life_task

    asyncio.run(run())

