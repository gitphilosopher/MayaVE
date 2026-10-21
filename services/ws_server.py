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
from dataclasses import dataclass
from enum import Enum
import json
import logging
from typing import Any, Awaitable, Callable, Optional, Sequence, Set
import uuid

import websockets
import websockets.exceptions

from config.settings import config

logger = logging.getLogger(__name__)

InterruptHandler = Callable[[], Awaitable[None]]

class PlaybackOutcome(str, Enum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NO_CLIENT = "no_client"


@dataclass
class PlaybackResult:
    outcome: PlaybackOutcome
    playback_id: str
    error: Optional[str] = None

    @property
    def is_completed(self) -> bool:
        return self.outcome == PlaybackOutcome.COMPLETED

    def __bool__(self) -> bool:
        return self.outcome == PlaybackOutcome.COMPLETED

_STATE_STR_MAP = {
    "SLEEPING": "sleeping",
    "IDLE": "idle",
    "LISTENING": "listening",
    "PROCESSING": "processing",
    "SPEAKING": "speaking",
    "INTERRUPTED": "listening",
}

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
        self._bg_tasks: Set[asyncio.Task] = set()
        self._pending_playbacks: dict[str, asyncio.Future[PlaybackResult]] = {}
        self._active_playback_id: Optional[str] = None
        self._compat_playback_id: Optional[str] = None
        self._playback_authority: Optional[WSConnection] = None
        self._playback_lock: Optional[asyncio.Lock] = None
        self._playback_lock_loop: Optional[Any] = None
        self._interrupt_handler: Optional[InterruptHandler] = None
        self._state_lock: Optional[asyncio.Lock] = None
        self._state_lock_loop: Optional[Any] = None
        self._last_state: str = self.get_current_state()
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

    @property
    def _lock(self) -> asyncio.Lock:
        """Return an asyncio.Lock tied to the currently running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._state_lock is None or self._state_lock_loop is not loop:
            self._state_lock = asyncio.Lock()
            self._state_lock_loop = loop
        return self._state_lock

    @property
    def playback_authority(self) -> Optional[WSConnection]:
        """Return the currently designated authoritative renderer connection, if any."""
        return self._playback_authority

    @property
    def playback_lock(self) -> asyncio.Lock:
        """Return an asyncio.Lock tied to the currently running event loop for audio serialization."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._playback_lock is None or self._playback_lock_loop is not loop:
            self._playback_lock = asyncio.Lock()
            self._playback_lock_loop = loop
        return self._playback_lock

    def get_stop_gen(self) -> int:
        """Return the current stop generation counter."""
        return self._stop_gen

    def get_current_state(self) -> str:
        """Resolve the authoritative current state string from core.state."""
        try:
            from core.state import state
            curr = state.current
            name = getattr(curr, "name", str(curr))
            return _STATE_STR_MAP.get(name, getattr(self, "_last_state", "idle"))
        except Exception:
            return getattr(self, "_last_state", "idle")

    async def _handler(self, ws: WSConnection) -> None:
        """Serve one websocket client and forward its incoming messages to the server logic."""
        addr = ws.remote_address
        logger.info(f"Avatar connected: {addr}  (clients={len(self._clients) + 1})")
        try:
            async with self._lock:
                current_state = self.get_current_state()
                self._last_state = current_state
                await ws.send(json.dumps({"type": "state", "value": current_state}))
                self._clients.add(ws)
                if self._playback_authority is None:
                    self._playback_authority = ws
                    logger.info(f"Designated {addr} as authoritative playback renderer.")
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
            if self._playback_authority == ws:
                self._playback_authority = None
                # Settle active playbacks as disconnected
                for pid, fut in list(self._pending_playbacks.items()):
                    if not fut.done():
                        fut.set_result(
                            PlaybackResult(PlaybackOutcome.DISCONNECTED, pid, "Authority disconnected")
                        )
                # Promote another connected client for future playback only
                remaining = list(self._clients)
                if remaining:
                    self._playback_authority = remaining[0]
                    logger.info(f"Promoted {getattr(self._playback_authority, 'remote_address', 'client')} as new authoritative renderer.")

    async def _on_message(self, ws: WSConnection, message: str) -> None:
        """Handle browser events such as interrupts and audio completion notifications."""
        try:
            data = json.loads(message)
            if data.get("type") == "interrupt":
                logger.info("Avatar sent interrupt signal.")
                if self._interrupt_handler:
                    task = asyncio.create_task(self._interrupt_handler())
                    self._bg_tasks.add(task)
                    task.add_done_callback(self._bg_tasks.discard)
                else:
                    logger.debug("No interrupt handler registered — ignoring.")
            elif data.get("type") == "audio_done":
                # Accept audio_done only from the authoritative connection
                if ws != self._playback_authority:
                    logger.warning("Ignoring audio_done from non-authoritative client.")
                    return

                pid = data.get("playback_id")
                status = data.get("status", "completed")
                reason = data.get("reason")

                if not pid:
                    logger.warning("Ignoring audio_done without playback_id.")
                    return

                fut = self._pending_playbacks.get(pid)
                if fut is None:
                    logger.debug(f"Ignoring audio_done for unknown/stale playback_id '{pid}'.")
                    return
                if fut.done():
                    logger.debug(f"Ignoring duplicate audio_done for settled playback_id '{pid}'.")
                    return

                if status == "completed":
                    outcome = PlaybackOutcome.COMPLETED
                elif status == "interrupted":
                    outcome = PlaybackOutcome.INTERRUPTED
                else:
                    outcome = PlaybackOutcome.FAILED

                fut.set_result(PlaybackResult(outcome, pid, reason))
        except Exception:
            pass

    # ── Interrupt wiring ──────────────────────────────────────────────

    def set_interrupt_handler(self, handler: InterruptHandler) -> None:
        """Register the async callback invoked by browser-side interrupt events."""
        self._interrupt_handler = handler

    # ── Broadcast helpers ─────────────────────────────────────────────

    async def play_audio_and_wait(
        self,
        wav_bytes: bytes,
        trace_id: Optional[str] = None,
        timeout: float = 30.0,
        playback_id: Optional[str] = None,
        expected_stop_gen: Optional[int] = None,
    ) -> PlaybackResult:
        """
        Send WAV payload to authoritative renderer and await explicit correlated completion.
        Registers the transaction waiter BEFORE sending the audio message. `timeout`
        bounds the acknowledgement wait after `authority.send()` returns; it does
        not bound the send operation itself.
        """
        if playback_id is None:
            playback_id = uuid.uuid4().hex

        authority = self._playback_authority
        if authority is None or not self._clients:
            return PlaybackResult(PlaybackOutcome.NO_CLIENT, playback_id, "No connected renderer")

        if expected_stop_gen is not None and self._stop_gen != expected_stop_gen:
            return PlaybackResult(PlaybackOutcome.INTERRUPTED, playback_id, "Stopped before send")

        if playback_id in self._pending_playbacks:
            return PlaybackResult(PlaybackOutcome.FAILED, playback_id, "Playback ID is already pending")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PlaybackResult] = loop.create_future()
        self._pending_playbacks[playback_id] = future
        self._active_playback_id = playback_id
        self._compat_playback_id = playback_id
        self._audio_sent_gen = self._stop_gen

        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        msg = {
            "type": "audio",
            "data": b64,
            "playback_id": playback_id,
        }
        if trace_id:
            msg["trace_id"] = trace_id

        try:
            from services.tracer import log_event
            if trace_id:
                log_event(trace_id, "ws_send_start", {"bytes": len(wav_bytes)})
        except Exception:
            pass

        try:
            try:
                await authority.send(json.dumps(msg))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Failed to send audio to authority: {e}")
                return PlaybackResult(PlaybackOutcome.FAILED, playback_id, f"Send failed: {e}")

            try:
                from services.tracer import log_event
                if trace_id:
                    log_event(trace_id, "ws_send_end", {"bytes": len(wav_bytes)})
            except Exception:
                pass

            try:
                return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Playback {playback_id} timed out after {timeout}s.")
                return PlaybackResult(PlaybackOutcome.TIMEOUT, playback_id, f"Timed out after {timeout}s")
            except asyncio.CancelledError:
                logger.info(f"Playback {playback_id} cancelled.")
                raise
        finally:
            owns_transaction = self._pending_playbacks.get(playback_id) is future
            if owns_transaction:
                self._pending_playbacks.pop(playback_id)
            if owns_transaction and self._active_playback_id == playback_id:
                self._active_playback_id = None

    async def broadcast_audio(self, wav_bytes: bytes, trace_id: Optional[str] = None, playback_id: Optional[str] = None) -> str:
        """Send one WAV payload to the authoritative renderer; tracks generation order for stop handling."""
        if playback_id is None:
            playback_id = uuid.uuid4().hex
        if playback_id in self._pending_playbacks:
            logger.warning(f"Playback ID '{playback_id}' is already pending; compatibility send skipped.")
            return playback_id
        self._compat_playback_id = playback_id
        self._audio_sent_gen = self._stop_gen

        authority = self._playback_authority
        if authority is None or not self._clients:
            return playback_id

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PlaybackResult] = loop.create_future()
        self._pending_playbacks[playback_id] = future
        self._active_playback_id = playback_id

        try:
            from services.tracer import log_event
            if trace_id:
                log_event(trace_id, "ws_send_start", {"bytes": len(wav_bytes)})
        except Exception:
            pass

        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        msg = {"type": "audio", "data": b64, "playback_id": playback_id}
        if trace_id:
            msg["trace_id"] = trace_id

        try:
            await authority.send(json.dumps(msg))
            try:
                from services.tracer import log_event
                if trace_id:
                    log_event(trace_id, "ws_send_end", {"bytes": len(wav_bytes)})
            except Exception:
                pass
        except asyncio.CancelledError:
            owns_transaction = self._pending_playbacks.get(playback_id) is future
            if owns_transaction:
                self._pending_playbacks.pop(playback_id, None)
            if owns_transaction and self._active_playback_id == playback_id:
                self._active_playback_id = None
            raise
        except Exception as e:
            logger.warning(f"Failed to send audio to authority: {e}")
            owns_transaction = self._pending_playbacks.get(playback_id) is future
            if owns_transaction:
                self._pending_playbacks.pop(playback_id, None)
            if owns_transaction and self._active_playback_id == playback_id:
                self._active_playback_id = None

        return playback_id

    async def broadcast_stop_audio(self, playback_id: Optional[str] = None) -> None:
        """Tell the browser to stop current playback and release any pending audio wait."""
        self._stop_gen += 1
        target_id = playback_id or self._active_playback_id

        # Settle pending playback(s) as INTERRUPTED
        if target_id and target_id in self._pending_playbacks:
            fut = self._pending_playbacks[target_id]
            if not fut.done():
                fut.set_result(PlaybackResult(PlaybackOutcome.INTERRUPTED, target_id, "Stopped by user/barge-in"))
        elif not playback_id:
            for pid, fut in list(self._pending_playbacks.items()):
                if not fut.done():
                    fut.set_result(PlaybackResult(PlaybackOutcome.INTERRUPTED, pid, "Stopped by user/barge-in"))

        self._active_playback_id = None

        if not self._clients:
            return
        logger.debug(f"Broadcasting stop_audio (playback_id={target_id})")
        msg = {"type": "stop_audio"}
        if target_id:
            msg["playback_id"] = target_id
        await self._broadcast(json.dumps(msg))

    async def broadcast_state(self, state_value: str) -> None:
        """Persist and emit a new avatar state value to connected clients."""
        async with self._lock:
            if state_value == self._last_state and not self._clients:
                return
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

    async def wait_for_audio_done(self, timeout: float = 30.0, playback_id: Optional[str] = None) -> bool:
        """Wait for the browser to report that the current audio clip finished, unless it was stopped."""
        pid = playback_id or self._compat_playback_id
        if not self._clients or self._playback_authority is None:
            return False
        if self._stop_gen != self._audio_sent_gen:
            return False

        fut = self._pending_playbacks.get(pid) if pid else None
        if fut is None:
            return False

        try:
            res = await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
            return res.is_completed
        except asyncio.TimeoutError:
            logger.warning("wait_for_audio_done timed out — continuing anyway.")
            return False
        except asyncio.CancelledError:
            raise
        finally:
            if pid:
                owns_transaction = self._pending_playbacks.get(pid) is fut
                if owns_transaction:
                    self._pending_playbacks.pop(pid, None)
                if owns_transaction and self._active_playback_id == pid:
                    self._active_playback_id = None

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