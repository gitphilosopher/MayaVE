"""
core/turn_lifecycle.py
Shared "a turn just finished — go back to resting" helper.

Several call sites (services/llm/llm_service.py's _play_worker at _DONE,
core/processor.py's Processor.handle on both its success and exception
paths) each need to reset the FSM + tell the frontend once a reply/skill/
LLM turn completes. Before this existed, each did so unconditionally
("always go to IDLE, always broadcast idle + baseline behavior") — but if
a "go to sleep" landed while that turn was still in flight, the state was
already SLEEPING by the time the turn finished, and blindly forcing IDLE
here silently woke Maya back up and re-enabled full speech processing
right when the wake-word detector should have taken over instead (see
docs/CHANGELOG.md's "Sleep race" issue). rest() checks the CURRENT state
fresh at the moment the turn ends, not a snapshot taken earlier, so a
concurrent sleep always wins.

core/speaker.py's Speaker.speak() has a related but distinct need (it
must remember whether IT ITSELF was the sleep-command's own goodbye line,
since by the time its own finally runs the FSM has already been SPEAKING
for a while) and keeps its own inline logic rather than using this helper
— see its docstring.
"""

import logging

from core.state import state, MayaState
from core.mood import mood_manager
from core.behavior_engine import behavior_engine

logger = logging.getLogger(__name__)


async def rest(force_idle: bool = False) -> None:
    """
    Call once a turn (skill reply, LLM reply, or a handled exception) has
    finished. If Maya is currently SLEEPING — meaning a "go to sleep"
    landed while this turn was still running — leaves her asleep and
    tells the frontend to sleep visually instead of stomping it back to
    IDLE. Otherwise broadcasts IDLE + the current mood baseline face.

    force_idle: the caller has already decided the FSM itself should move
    to IDLE (e.g. after an unhandled exception, or at the natural end of
    an LLM turn) — still respects a concurrent sleep (skips the state.set
    AND the WS broadcast in that case), but callers that don't need an
    explicit transition (state is already correct via Speaker.speak()'s
    own finally) can leave this False and only get the WS-broadcast half.
    """
    from services.ws_server import ws_server   # local import — see Speaker.speak()'s own precedent

    if state.is_sleeping():
        await ws_server.broadcast_state("sleeping")
        return

    if force_idle:
        await state.set(MayaState.IDLE)
    await ws_server.broadcast_state("idle")
    await ws_server.broadcast_behavior(
        behavior_engine.compose(mood_manager.baseline_expression(), source="idle")
    )
