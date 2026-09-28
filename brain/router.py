"""
brain/router.py
Intent-to-skill dispatch for Maya's turn pipeline.

This module is the central routing layer between a parsed intent dictionary and
its concrete implementation. It sits after the intent classifier and before the
actual skill execution, deciding whether a turn should be handled by a direct
system action, a small built-in conversational response, or a fallback LLM call.

The router is intentionally thin and centralized:
- `Router.__init__()` registers all supported intents to their handlers.
- `Router.dispatch()` resolves any in-flight confirmation before normal routing,
  so single-shot confirmations such as shutdown, reminder duration, or note
  deletion consume the next utterance before ordinary chat is processed.
- Unknown or conversational intents fall back to the LLM rather than failing the
  turn.
- Skill-level exceptions are caught here so the system can return a graceful
  apology without crashing the conversation loop.

Most callers interact with this module through `Router.dispatch(intent, raw_text)`.
"""

import logging

from config.settings import config
from core.speaker import Speaker

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

        if handler is None:
            logger.warning(f"No route for intent '{intent_name}' — falling back to LLM.")
            return await llm_query(intent, raw_text)

        try:
            return await handler(intent, raw_text)
        except Exception as e:
            logger.error(f"Skill error ({intent_name}): {e}", exc_info=True)
            intent["_no_history"] = True
            return f"[sad] Sorry {_U}, I ran into a problem with that."

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