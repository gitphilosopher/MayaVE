"""
services/ws_server.py
WebSocket bridge between Maya's runtime and the browser avatar.

This module starts the server used by the frontend to receive avatar state,
transcript updates, behavior packets, animation commands, and audio chunks, while
also handling the browser-to-runtime interrupt path. The singleton `ws_server`
keeps a set of live client connections and remembers the most recent state so a
newly connected frontend immediately receives the current avatar status.

Key responsibilities:
- validate browser origins against `config.ws_allowed_origins` during the
  websocket handshake
- broadcast audio, transcript, state, behavior, and animation events to the
  connected browser client
- accept client interrupt requests (`{"type": "interrupt"}`) and invoke an
  injectable async handler, usually `core.state.interrupt()`
- grant barge-in safety by sending `stop_audio` and unblocking any pending
  `wait_for_audio_done()` call when playback is cut off early
- persist the last state even when no clients are connected so a reconnect does
  not leave the frontend in an uninitialized state

The server stays transport-focused: it does not decide speech logic itself. It
carries the runtime events and interruption signals between Maya's core system and
its browser frontend.
"""

import asyncio
import base64
import json
import logging
from typing import Any, Awaitable, Callable, Optional, Sequence, Set

import websockets

from config.settings import config

logger = logging.getLogger(__name__)

InterruptHandler = Callable[[], Awaitable[None]]

# A server-side connection. Its concrete class depends on the installed
# websockets version, and it is only used for annotations here.
WSConnection = Any


class MayaWebSocketServer:
    """Small stateful websocket server for avatar audio and UI events."""

    def __init__(self, host: str = "localhost", port: int = 8765,
                 origins: Optional[Sequence[Optional[str]]] = None):
        self._host    = host
        self._port    = port
        # None = accept any origin; otherwise the exact Origin values allowed
        # (a None entry allows connections that send no Origin header).
        self._origins: Optional[list] = list(origins) if origins is not None else None
        self._clients: Set[WSConnection] = set()
        self._audio_done_event: asyncio.Event = asyncio.Event()
        self._interrupt_handler: Optional[InterruptHandler] = None
        self._last_state: str = "idle"
        self._stop_gen: int = 0         # bumped by every stop_audio broadcast
        self._audio_sent_gen: int = 0   # _stop_gen when the latest audio was sent

    # ── Server lifecycle ──────────────────────────────────────────────

    async def serve(self) -> None:
        logger.info(f"WebSocket server starting on ws://{self._host}:{self._port}")
        async with websockets.serve(
            self._handler,
            self._host,
            self._port,
            origins=self._origins,
            ping_interval=20,
            ping_timeout=60,
        ):
            if self._origins is None:
                logger.warning("WebSocket server ready — origin check DISABLED, accepting any origin")
            else:
                allowed = [o if o is not None else "<no Origin header>" for o in self._origins]
                logger.info(f"WebSocket server ready — allowed origins: {allowed}")
            await asyncio.Future()

    async def _handler(self, ws: WSConnection) -> None:
        """Serve one websocket client and forward its incoming messages to the server logic."""
        self._clients.add(ws)
        addr = ws.remote_address
        logger.info(f"Avatar connected: {addr}  (clients={len(self._clients)})")
        try:
            await ws.send(json.dumps({"type": "state", "value": self._last_state}))
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

    async def _on_message(self, ws: WSConnection, message: str) -> None:
        """Handle browser events such as interrupts and audio completion notifications."""
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
        """Register the async callback invoked by browser-side interrupt events."""
        self._interrupt_handler = handler

    # ── Broadcast helpers ─────────────────────────────────────────────

    async def broadcast_audio(self, wav_bytes: bytes) -> None:
        """Send one WAV payload to the browser; tracks generation order for stop handling."""
        self._audio_sent_gen = self._stop_gen   # snapshot before any await
        if not self._clients:
            return
        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        await self._broadcast(json.dumps({"type": "audio", "data": b64}))

    async def broadcast_stop_audio(self) -> None:
        """Tell the browser to stop current playback and release any pending audio wait."""
        self._stop_gen += 1
        if not self._clients:
            return
        logger.debug("Broadcasting stop_audio (barge-in)")
        await self._broadcast(json.dumps({"type": "stop_audio"}))
        self._audio_done_event.set()

    async def broadcast_state(self, state_value: str) -> None:
        """Persist and emit a new avatar state value to connected clients."""
        self._last_state = state_value
        if not self._clients:
            return
        await self._broadcast(json.dumps({"type": "state", "value": state_value}))

    async def broadcast_expression(self, expression: str) -> None:
        """Broadcast a legacy flat expression payload for compatibility."""
        if not self._clients:
            return
        logger.debug(f"Broadcasting expression: '{expression}'")
        await self._broadcast(json.dumps({"type": "expression", "name": expression}))

    async def broadcast_behavior(self, intent: dict) -> None:
        """Send a composed behavior packet for the frontend expression composer."""
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
        """Wait for the browser to report that the current audio clip finished, unless it was stopped."""
        if not self._clients:
            return False
        self._audio_done_event.clear()
        if self._stop_gen != self._audio_sent_gen:
            return False
        try:
            await asyncio.wait_for(self._audio_done_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("wait_for_audio_done timed out — continuing anyway.")
            return False

    async def _broadcast(self, msg: str) -> None:
        dead = set()
        for ws in list(self._clients):   # copy — clients may connect/leave during send
            try:
                await ws.send(msg)
            except websockets.exceptions.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead


# Singleton
ws_server = MayaWebSocketServer(
    host=getattr(config, "ws_host", "localhost"),
    port=getattr(config, "ws_port", 8765),
    origins=config.ws_allowed_origins,
)