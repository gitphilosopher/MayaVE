"""
core/state.py
Global state machine for Maya.
All modules read/write state through this single object — no hidden flags.

States:
  SLEEPING    — mic is hot but only wake-word detector is active
  IDLE        — awake, waiting for a command
  LISTENING   — VAD triggered, capturing speech
  PROCESSING  — transcribing + intent + skill execution
  SPEAKING    — TTS audio playing
  INTERRUPTED — user spoke while Maya was speaking (barge-in)

Interrupt handling (barge-in)
------------------------------
Whatever produces Maya's speech (Speaker.speak() or
llm_service._stream_and_speak()) is expected to run wrapped in
run_interruptible() rather than being awaited directly. That wraps the
coroutine in its own asyncio.Task and registers it here via
set_current_task(), so a barge-in detected mid-sentence has something
concrete to cancel.

interrupt() itself doesn't know how to physically stop already-issued
audio (sounddevice buffers, browser-side WebAudio playback) — that's
speaker/ws_server concerns. Instead it calls an injectable stop
callback (register_stop_callback(), wired up once in main.py) *before*
cancelling the task, so blocking calls like sd.wait() or
ws_server.wait_for_audio_done() unblock immediately instead of waiting
on hardware/network round trips.

can_interrupt() defines when a barge-in is allowed: SPEAKING, or
PROCESSING while a registered speech task is live (an LLM turn,
including its filler and the wait before the first phrase). PROCESSING
with no registered task (skill turns) is not interruptible — see
core/processor.py and skills/system/power.py's shutdown/restart line
for the two current exceptions (their own speak() calls DO register a
task; a skill's own blocking dispatch work before it starts speaking
still doesn't).

Observers (Batch 4 — "Listening state edge cases")
----------------------------------------------------
add_observer() lets other modules react to every genuine transition
without this module importing anything about WHY they'd want to (no
sounddevice/ws_server imports here — deliberately kept decoupled from
the audio backends). main.py uses this to run a watchdog that resets a
LISTENING state stuck with no follow-up (a dropped queued command, or a
client-side "interrupt" with no speech behind it) back to IDLE instead
of leaving the FSM — and the frontend's idle-fidget gate — stuck
forever. Observers are synchronous and best-effort: a raising observer
is logged and skipped, never allowed to break a transition. set_sync()
runs on the sounddevice callback thread, so an observer that needs to
touch asyncio must hop back onto the loop itself (e.g.
loop.call_soon_threadsafe) rather than assume it's already there.
"""

import asyncio
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)


class MayaState(Enum):
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
    _state: MayaState = field(default=MayaState.SLEEPING, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # ── Interrupt plumbing ───────────────────────────────────────────
    _current_task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)
    _stop_cb: Optional[StopCallback] = field(default=None, init=False, repr=False)

    # ── Transition observers ─────────────────────────────────────────
    _observers: List[StateObserver] = field(default_factory=list, init=False, repr=False)

    # ── Read ──────────────────────────────────────────────────────────

    @property
    def current(self) -> MayaState:
        return self._state

    def is_sleeping(self) -> bool:
        return self._state == MayaState.SLEEPING

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

    # ── Write ─────────────────────────────────────────────────────────

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

    # ── Observers ────────────────────────────────────────────────────

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

    # ── Interrupt handling (barge-in) ────────────────────────────────

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