"""
core/turn_lifecycle.py
Turn-completion helper that restores Maya to a resting state safely.

This module centralizes the end-of-turn cleanup used by the conversation and
speech pipeline. When a skill reply, LLM reply, or handled exception finishes,
callers can invoke `rest()` to return the state machine to a neutral resting
state and broadcast the matching frontend update.

The key safety rule is that the helper checks the current FSM state at the
moment the turn completes, rather than assuming the earlier state. That matters
when a "go to sleep" command arrives while a previous turn is still running: the
sleep transition must win, and the helper must not wake Maya back up by forcing
IDLE after a concurrent sleep command has already landed.

This function intentionally stays small and focused; the speaker layer has its
own related sleep-handling logic for its own goodbye line, but the lifecycle
helper is the shared single point for end-of-turn reset logic.
"""

import logging

from core.state import MayaState, state
from core.behavior_engine import behavior_engine
from core.mood import mood_manager

logger = logging.getLogger(__name__)


async def rest(force_idle: bool = False) -> None:
    """Return Maya to a resting state and broadcast the matching frontend update."""
    from services.ws_server import ws_server

    if state.is_sleeping():
        await ws_server.broadcast_state("sleeping")
        return

    # If already idle and not forced, the turn has already rested (e.g. via Speaker.speak).
    if state.is_idle() and not force_idle:
        return

    await state.set(MayaState.IDLE)
    await ws_server.broadcast_state("idle")
    await ws_server.broadcast_behavior(
        behavior_engine.compose(mood_manager.baseline_expression(), source="idle")
    )

