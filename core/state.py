"""
core/state.py
Global state machine for Maya.

This module is the single source of truth for the agent's lifecycle state. The
system should read or mutate the shared `state` instance rather than maintaining
separate boolean flags or hidden FSMs in other modules.

The state values reflect the conversation lifecycle:
  SLEEPING    — wake-word listener is active and no command flow is running
  IDLE        — awake and waiting for user input
  LISTENING   — microphone/VAD capture is active
  PROCESSING  — transcription, intent routing, or skill execution is running
  SPEAKING    — Maya is generating or streaming speech output
  INTERRUPTED — a barge-in was triggered and the system is returning to LISTENING

`StateManager` centralizes transitions, observer notifications, and interrupt
control. It intentionally stays decoupled from audio backends: the speaker and
browser/websocket layers register their stop callback and active speech task
with this class, while the FSM decides whether an interruption is currently
allowed and what should be cancelled.

Typical usage:
- read `state.current` or predicates like `is_busy()`/`is_speaking()`
- move between states with `await state.set(...)`
- use `state.set_sync(...)` from callback threads
- subscribe to transitions with `add_observer(...)`
- register the active speech task with `set_current_task(...)`
- run speak/LLM output inside `run_interruptible(...)`
- trigger a barge-in with `await state.interrupt()`

The interrupt path is intentionally conservative: a barge-in is allowed only
while speaking or while a live registered speech task is active during
processing. `interrupt()` stops physical playback first, cancels the task, and
then transitions back to LISTENING so the interrupted utterance can be captured
normally. Observers are synchronous and best-effort; exceptions are logged and
ignored so they never break the state machine.
"""

import asyncio
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)


class MayaState(Enum):
    """Finite states for Maya's lifecycle and interaction flow."""

    SLEEPING    = auto()   # only wake-word detector is active
    IDLE        = auto()   # awake, waiting for speech
    LISTENING   = auto()   # VAD triggered, accumulating utterance
    PROCESSING  = auto()   # transcribing + intent + skill
    SPEAKING    = auto()   # TTS playback in progress
    INTERRUPTED = auto()   # user spoke during SPEAKING (barge-in)


StopCallback = Callable[[], Awaitable[None]]
StateObserver = Callable[[MayaState, MayaState], None]   # (old, new) — sync, best-effort


@dataclass
class StateManager:
    """Singleton-style state holder for the agent lifecycle and interrupt logic."""

    _state: MayaState = field(default=MayaState.SLEEPING, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    _current_task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)
    _stop_cb: Optional[StopCallback] = field(default=None, init=False, repr=False)
    _observers: List[StateObserver] = field(default_factory=list, init=False, repr=False)

    @property
    def current(self) -> MayaState:
        return self._state

    def is_sleeping(self) -> bool:
        return self._state == MayaState.SLEEPING

    def is_idle(self) -> bool:
        return self._state == MayaState.IDLE

    def is_speaking(self) -> bool:
        return self._state == MayaState.SPEAKING

    def is_busy(self) -> bool:
        return self._state in (MayaState.PROCESSING, MayaState.SPEAKING)

    def can_interrupt(self) -> bool:
        """True if a barge-in would currently have something to stop."""
        if self._state == MayaState.SPEAKING:
            return True
        task = self._current_task
        return (
            self._state == MayaState.PROCESSING
            and task is not None
            and not task.done()
        )

    async def set(self, new_state: MayaState) -> None:
        async with self._lock:
            old = self._state
            self._state = new_state
            if old != new_state:
                logger.debug(f"State: {old.name} → {new_state.name}")
        self._notify(old, new_state)

    def set_sync(self, new_state: MayaState) -> None:
        """Non-async setter — safe to call from sounddevice callback thread."""
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.debug(f"State: {old.name} → {new_state.name}")
        self._notify(old, new_state)

    def add_observer(self, cb: StateObserver) -> None:
        """
        Register cb(old, new), fired after every genuine transition (no-op
        transitions where old == new are not reported). Observers run
        synchronously and must not block; a raising observer is logged and
        swallowed rather than propagated, since a side-effect hook must
        never be able to break the FSM itself.
        """
        self._observers.append(cb)

    def _notify(self, old: MayaState, new: MayaState) -> None:
        if old == new or not self._observers:
            return
        for cb in self._observers:
            try:
                cb(old, new)
            except Exception:
                logger.debug("State observer failed (non-fatal)", exc_info=True)

    def register_stop_callback(self, cb: StopCallback) -> None:
        """
        Register the async function that hard-stops whatever audio is
        currently playing (local sounddevice + browser avatar audio).
        Wired once in main.py — state.py deliberately has no direct
        sounddevice/ws_server imports so the FSM stays decoupled from
        the audio backends.
        """
        self._stop_cb = cb

    def set_current_task(self, task: Optional[asyncio.Task]) -> None:
        """
        Register the asyncio.Task currently producing Maya's speech so
        interrupt() has something to cancel. Call with None to clear
        once the task finishes normally.
        """
        self._current_task = task

    async def run_interruptible(self, coro: Awaitable) -> None:
        """
        Run `coro` as its own Task and register it as the current
        speech task for the duration. If interrupt() cancels it, the
        CancelledError is swallowed here — callers don't need their
        own try/except for the common barge-in path.
        """
        task = asyncio.ensure_future(coro)
        self._current_task = task
        try:
            await task
        except asyncio.CancelledError:
            logger.info("Speech task cancelled (barge-in).")
        finally:
            if self._current_task is task:
                self._current_task = None

    async def interrupt(self) -> bool:
        """
        Barge-in: called when the listener detects Maya being called by
        name while can_interrupt() is True (SPEAKING, or an in-flight LLM
        turn in PROCESSING), or when something else (e.g. a "go to sleep"
        command landing mid-reply) wants to cleanly stop whatever's
        currently playing before doing its own thing. Hard-stops current
        audio, cancels the registered speech task, and drops the state to
        LISTENING so the utterance that triggered the interrupt gets
        processed normally once the listener finishes capturing it.

        Returns True if there was actually something to interrupt.
        """
        async with self._lock:
            if not self.can_interrupt():
                return False
            old = self._state
            self._state = MayaState.INTERRUPTED
            logger.info(f"State: {old.name} → INTERRUPTED (barge-in)")
        self._notify(old, MayaState.INTERRUPTED)

        # Hard-stop physical audio first so blocking waits (sd.wait(),
        # ws_server.wait_for_audio_done()) unblock immediately instead
        # of the cancellation racing against hardware/network latency.
        if self._stop_cb is not None:
            try:
                await self._stop_cb()
            except Exception as e:
                logger.warning(f"Interrupt stop callback failed: {e}")

        task = self._current_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Speech task raised after cancel: {e}")
        self._current_task = None

        await self.set(MayaState.LISTENING)
        return True


# Singleton
state = StateManager()