"""
services/ws_server.py
=====================
WebSocket server — bridges Maya's audio pipeline to the browser avatar.

Fix: pass origins=None to websockets.serve() so connections from
     http://localhost:5173 (Vite dev server) are accepted.
     Without this, websockets >=12 rejects cross-origin connections silently
     and the browser never gets past the opening handshake.

Interrupt handling (barge-in):
  Two directions now exist:
    client → server : browser sends {"type": "interrupt"} (e.g. a future
                       manual "stop talking" button). Routed to an
                       injectable handler registered via
                       set_interrupt_handler() — main.py wires this to
                       core.state.interrupt().
    server → client : broadcast_stop_audio() tells the browser to halt
                       whatever it's currently playing immediately. This
                       is the registered core.state stop callback's
                       browser-side half — see main.py's
                       _hard_stop_audio().

Behavioral Engine integration (core/behavior_engine.py):
  broadcast_behavior() sends a composed communicative-intent packet
  ({"type": "behavior", ...}) instead of the old flat
  {"type": "expression", "name": ...}. broadcast_expression() is kept
  for backward compatibility but is no longer called anywhere in this
  codebase — every prior call site now goes through
  behavior_engine.compose() + broadcast_behavior() instead, at the exact
  same point in the pipeline.
"""

import asyncio
import base64
import json
import logging
from typing import Awaitable, Callable, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from config.settings import config

logger = logging.getLogger(__name__)

InterruptHandler = Callable[[], Awaitable[None]]


class MayaWebSocketServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self._host    = host
        self._port    = port
        self._clients: Set[WebSocketServerProtocol] = set()
        self._audio_done_event: asyncio.Event = asyncio.Event()
        self._interrupt_handler: Optional[InterruptHandler] = None

    # ── Server lifecycle ──────────────────────────────────────────────

    async def serve(self) -> None:
        logger.info(f"WebSocket server starting on ws://{self._host}:{self._port}")
        async with websockets.serve(
            self._handler,
            self._host,
            self._port,
            origins=None,
            ping_interval=20,
            ping_timeout=60,
        ):
            logger.info(f"WebSocket server ready — accepting connections from any origin")
            await asyncio.Future()

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        addr = ws.remote_address
        logger.info(f"Avatar connected: {addr}  (clients={len(self._clients)})")
        try:
            async for message in ws:
                await self._on_message(ws, message)
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as e:
            logger.debug(f"WS connection closed with error: {e}")
        except Exception as e:
            logger.warning(f"WS client error: {e}")
        finally:
            self._clients.discard(ws)
            logger.info(f"Avatar disconnected: {addr}  (clients={len(self._clients)})")

    async def _on_message(self, ws: WebSocketServerProtocol, message: str) -> None:
        try:
            data = json.loads(message)
            if data.get("type") == "interrupt":
                logger.info("Avatar sent interrupt signal.")
                if self._interrupt_handler:
                    asyncio.create_task(self._interrupt_handler())
                else:
                    logger.debug("No interrupt handler registered — ignoring.")
            elif data.get("type") == "audio_done":
                logger.debug("Browser audio_done received.")
                self._audio_done_event.set()
        except Exception:
            pass

    # ── Interrupt wiring ──────────────────────────────────────────────

    def set_interrupt_handler(self, handler: InterruptHandler) -> None:
        """
        Register the async callback fired when the browser sends
        {"type": "interrupt"}. Wired in main.py to core.state.interrupt()
        so a client-initiated interrupt behaves exactly like a
        listener-detected barge-in.
        """
        self._interrupt_handler = handler

    # ── Broadcast helpers ─────────────────────────────────────────────

    async def broadcast_audio(self, wav_bytes: bytes) -> None:
        if not self._clients:
            return
        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        await self._broadcast(json.dumps({"type": "audio", "data": b64}))

    async def broadcast_stop_audio(self) -> None:
        """
        Tell the browser to immediately stop any audio currently playing
        (and its lip-sync) — the client-side half of a barge-in. Safe to
        call even if nothing is playing.
        """
        if not self._clients:
            return
        logger.debug("Broadcasting stop_audio (barge-in)")
        await self._broadcast(json.dumps({"type": "stop_audio"}))

    async def broadcast_state(self, state_value: str) -> None:
        if not self._clients:
            return
        await self._broadcast(json.dumps({"type": "state", "value": state_value}))

    async def broadcast_expression(self, expression: str) -> None:
        """Set a specific VRM expression on the avatar, independent of state.
        Kept for backward compatibility — no longer called anywhere in this
        codebase; see broadcast_behavior() below."""
        if not self._clients:
            return
        logger.debug(f"Broadcasting expression: '{expression}'")
        await self._broadcast(json.dumps({"type": "expression", "name": expression}))

    async def broadcast_behavior(self, intent: dict) -> None:
        """Broadcast a composed communicative-intent packet (see
        core/behavior_engine.py's BehaviorEngine.compose()) for the
        frontend Expression Composer to render as blended VRM weights +
        gaze. Replaces the old flat broadcast_expression() call sites."""
        if not self._clients:
            return
        logger.debug(f"Broadcasting behavior: {intent}")
        await self._broadcast(json.dumps({"type": "behavior", **intent}))

    async def broadcast_transcript(self, text: str, role: str) -> None:
        if not self._clients:
            return
        await self._broadcast(json.dumps({"type": "transcript", "text": text, "role": role}))

    async def broadcast_animation(self, animation: str) -> None:
        if not self._clients:
            return
        logger.info(f"Broadcasting animation: '{animation}'")
        await self._broadcast(json.dumps({"type": "animation", "name": animation}))

    async def wait_for_audio_done(self, timeout: float = 30.0) -> bool:
        self._audio_done_event.clear()
        try:
            await asyncio.wait_for(self._audio_done_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("wait_for_audio_done timed out — continuing anyway.")
            return False

    async def _broadcast(self, msg: str) -> None:
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
            except websockets.exceptions.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead


# Singleton
ws_server = MayaWebSocketServer(
    host=getattr(config, "ws_host", "localhost"),
    port=getattr(config, "ws_port", 8765),
)