"""
brain/router/dispatch.py   (moved from brain/router.py — `git mv`)
Intent-to-skill dispatch for Maya's turn pipeline.

Resolves in-flight confirmations first (power, reminder duration, note
delete), then routes to a skill, a built-in reply, or the LLM.

- "clarify" route: a CommandIR that needs clarification is turned into a
  spoken question (intent["_ir"].prompt) instead of falling through to the LLM.
- When the intent carries a CommandIR (`_ir`), skills receive its
  effective_text (after context rewriting). Pending-confirmation resolvers
  and the LLM still receive the user's original words.
- BATCH 1: IR safety net. If the intent carries an IR whose status is not
  READY, dispatch will not run a skill regardless of intent name: a
  NEEDS_CLARIFICATION IR always goes to _clarify, UNKNOWN/REJECTED always go
  to the LLM. (ir.to_legacy_intent already avoids skill names for these;
  this is defense in depth.)
- BATCH 2: READY-with-missing-entities is also non-executable
  (CommandIR.executable). requires_confirmation is DECLARED on the spec/IR and
  ENFORCED by the skill (power.py, notepad.py delete); dispatch fails closed if
  a flagged IR would reach a handler that does not self-confirm.
- Router._routes is the single intent -> handler map (see skills/base.py).
"""

import logging

from config.settings import config
from core.speaker import Speaker

from brain.router.ir import Status
from services.llm.llm_service import query as llm_query
from skills.media.play_music import execute as play_music
from skills.system.clipboard import execute as clipboard
from skills.system.lock_screen import execute as lock_screen
from skills.system.open_target import execute as open_target
from skills.system.perform_action import execute as perform_action
from skills.system.power import execute as power_action, resolve_pending as resolve_power_confirmation
from skills.system.system_info import execute as system_info
from skills.utilities.datetime_skill import execute as get_datetime
from skills.utilities.notepad import (
    execute as notepad,
    resolve_pending as resolve_note_delete_confirmation,
)
from skills.utilities.timer import (
    execute as timer,
    set_speaker as set_timer_speaker,
    resolve_pending as resolve_reminder_duration,
)
from skills.web.google_search import execute as google_search
from skills.web.weather import execute as get_weather

logger = logging.getLogger(__name__)

_U = config.user_name


class Router:
    """Route a parsed intent to the correct skill, built-in response, or LLM fallback."""

    def __init__(self, speaker: Speaker):
        """Register the route map and set the shared speaker used by timer-derived skills."""
        self._speaker = speaker
        set_timer_speaker(speaker)
        # Handlers that enforce their own spoken confirmation (declared via
        # CommandSpec.requires_confirmation, enforced HERE by the skill):
        # power.py asks before shutdown/restart; notepad.py asks before delete.
        self._self_confirming = (power_action, notepad)
        self._routes: dict = {
            "open_target": open_target,
            "system_info": system_info,
            "screenshot": system_info,
            "lock_screen": lock_screen,
            "shutdown": power_action,
            "restart": power_action,
            "search_web": google_search,
            "play_music": play_music,
            "pause_music": play_music,
            "next_track": play_music,
            "prev_track": play_music,
            "volume_up": play_music,
            "volume_down": play_music,
            "mute": play_music,
            "get_time": get_datetime,
            "get_date": get_datetime,
            "get_weather": get_weather,
            "clipboard_read": clipboard,
            "clipboard_write": clipboard,
            "clipboard_clear": clipboard,
            "set_timer": timer,
            "cancel_timer": timer,
            "timer_status": timer,
            "note_write": notepad,
            "note_view": notepad,
            "note_delete": notepad,
            "perform_action": perform_action,
            "greet": self._greet,
            "farewell": self._farewell,
            "thanks": self._thanks,
            "help": self._help,
            "clarify": self._clarify,
            "confirm": llm_query,
            "dismissal": llm_query,
            "smalltalk": llm_query,
            "identity": llm_query,
            "joke": llm_query,
            "motivate": llm_query,
            "opinion": llm_query,
            "followup": llm_query,
            "general_query": llm_query,
            "unknown": llm_query,
        }

    async def dispatch(self, intent: dict, raw_text: str) -> str:
        """Resolve pending confirmations first, then execute the matching handler or fallback."""
        power_reply = await resolve_power_confirmation(raw_text)
        if power_reply is not None:
            return power_reply

        reminder_reply = await resolve_reminder_duration(raw_text)
        if reminder_reply is not None:
            return reminder_reply

        note_reply = await resolve_note_delete_confirmation(raw_text)
        if note_reply is not None:
            return note_reply

        intent_name = intent.get("intent", "unknown")
        handler = self._routes.get(intent_name)

        ir = intent.get("_ir")
        if ir is not None:
            if ir.status is Status.NEEDS_CLARIFICATION:
                handler = self._clarify
            elif not ir.executable and handler is not llm_query:
                # UNKNOWN / REJECTED / READY-with-missing-entities: never a skill.
                logger.warning(
                    f"Non-executable IR ({ir.status.value}) carried intent "
                    f"'{intent_name}' — routing to LLM, not a skill."
                )
                handler = llm_query
            elif ir.executable and ir.requires_confirmation and handler not in self._self_confirming:
                # Declaration without enforcement: fail closed.
                logger.error(
                    f"IR for '{intent_name}' requires confirmation but its handler does "
                    f"not enforce one — refusing to run it."
                )
                intent["_no_history"] = True
                return f"[sad] Sorry {_U}, I can't run that safely without a confirmation step."

        if handler is None:
            logger.warning(f"No route for intent '{intent_name}' — falling back to LLM.")
            return await llm_query(intent, raw_text)

        # Skills re-parse text themselves, so give them the context-rewritten
        # command when a CommandIR produced one; the LLM keeps the user's words.
        text = raw_text
        if ir is not None and handler is not llm_query:
            text = getattr(ir, "effective_text", "") or raw_text

        try:
            return await handler(intent, text)
        except Exception as e:
            logger.error(f"Skill error ({intent_name}): {e}", exc_info=True)
            intent["_no_history"] = True
            return f"[sad] Sorry {_U}, I ran into a problem with that."

    async def _clarify(self, intent: dict, text: str) -> str:
        """Speak the CommandIR's clarification question."""
        ir = intent.get("_ir")
        prompt = (getattr(ir, "prompt", "") or "What did you have in mind").rstrip("?. ")
        return f"[surprised] {prompt}, {_U}?"

    async def _greet(self, intent: dict, text: str) -> str:
        """Return a short built-in greeting tailored to the configured user name."""
        return f"[happy] HELLO {_U}! How can I help you?"

    async def _farewell(self, intent: dict, text: str) -> str:
        """Return a short built-in farewell."""
        return f"[sad] Goodbye {_U}! Come back soon."

    async def _thanks(self, intent: dict, text: str) -> str:
        """Return a short built-in thanks response."""
        return f"[happy] You're welcome {_U}!"

    async def _help(self, intent: dict, text: str) -> str:
        """Return a compact help summary for the capabilities exposed by the router."""
        return (
            f"[happy] Here's what I can do {_U}: "
            "tell you the time and date, set countdown timers, check the weather, "
            "manage your clipboard, take notes, open websites and apps, "
            "[excited] control your music, check system info, tell jokes, and have a conversation. Just ask!"
        )