"""
brain/router.py
Routes an intent dict to the correct skill and returns a response string.
"""

import logging
from config.settings import config
from core.speaker import Speaker

# Skills
from skills.system.open_app     import execute as open_app
from skills.system.system_info  import execute as system_info
from skills.system.lock_screen  import execute as lock_screen
from skills.system.power        import execute as power_action, resolve_pending as resolve_power_confirmation
from skills.web.google_search   import execute as google_search
from skills.web.open_website    import execute as open_website
from skills.media.play_music    import execute as play_music
from skills.utilities.datetime_skill import execute as get_datetime
from services.llm.llm_service   import query as llm_query
from skills.web.weather          import execute as get_weather
from skills.system.clipboard     import execute as clipboard
from skills.utilities.timer      import execute as timer, set_speaker as set_timer_speaker
from skills.utilities.notepad    import execute as notepad
from skills.system.perform_action import execute as perform_action

logger = logging.getLogger(__name__)

_U = config.user_name


class Router:
    def __init__(self, speaker: Speaker):
        self._speaker = speaker
        set_timer_speaker(speaker)
        self._routes: dict = {
            # System
            "open_app":      open_app,
            "system_info":   system_info,
            "screenshot":    system_info,
            "lock_screen":   lock_screen,
            "shutdown":      power_action,
            "restart":       power_action,
            # Web
            "search_web":    google_search,
            "open_website":  open_website,
            # Media
            "play_music":    play_music,
            "pause_music":   play_music,
            "next_track":    play_music,
            "prev_track":    play_music,
            "volume_up":     play_music,
            "volume_down":   play_music,
            "mute":          play_music,
            # Utilities
            "get_time":      get_datetime,
            "get_date":      get_datetime,
            "set_reminder":  timer,
            # Weather
            "get_weather":   get_weather,
            # Clipboard
            "clipboard_read":  clipboard,
            "clipboard_write": clipboard,
            "clipboard_clear": clipboard,
            # Timers
            "set_timer":     timer,
            "cancel_timer":  timer,
            "timer_status":  timer,
            # Notepad
            "note_create":   notepad,
            "note_append":   notepad,
            "note_read":     notepad,
            "note_list":     notepad,
            "note_delete":   notepad,
            "note_open":     notepad,
            # Action animations (nod/giggle/sigh/shrug/wink)
            "perform_action": perform_action,
            # Conversational built-ins
            "greet":         self._greet,
            "farewell":      self._farewell,
            "thanks":        self._thanks,
            "help":          self._help,
            # All conversational/unknown → Ollama
            "confirm":       llm_query,
            "dismissal":     llm_query,
            "smalltalk":     llm_query,
            "identity":      llm_query,
            "joke":          llm_query,
            "motivate":      llm_query,
            "opinion":       llm_query,
            "followup":      llm_query,
            "general_query": llm_query,
            "unknown":       llm_query,
        }

    async def dispatch(self, intent: dict, raw_text: str) -> str:
        # A pending shutdown/restart confirmation gets first claim on the
        # next utterance ("yes" would otherwise be routed to the LLM).
        power_reply = await resolve_power_confirmation(raw_text)
        if power_reply is not None:
            return power_reply

        intent_name = intent.get("intent", "unknown")
        handler = self._routes.get(intent_name)

        if handler is None:
            logger.warning(f"No route for intent '{intent_name}' — falling back to LLM.")
            return await llm_query(intent, raw_text)

        try:
            return await handler(intent, raw_text)
        except Exception as e:
            logger.error(f"Skill error ({intent_name}): {e}", exc_info=True)
            return f"[sad] Sorry {_U}, I ran into a problem with that."

    # ── Built-in fast responses ───────────────────────────────────────────────

    async def _greet(self, intent: dict, text: str) -> str:
        return f"[happy] HELLO {_U}! How can I help you?"

    async def _farewell(self, intent: dict, text: str) -> str:
        return f"[sad] Goodbye {_U}! Come back soon."

    async def _thanks(self, intent: dict, text: str) -> str:
        return f"[happy] You're welcome {_U}!"

    async def _help(self, intent: dict, text: str) -> str:
        return (
            f"[happy] Here's what I can do {_U}: "
            "tell you the time and date, set countdown timers, check the weather, "
            "manage your clipboard, take notes, open websites and apps, "
            "[excited] control your music, check system info, tell jokes, and have a conversation. Just ask!"
        )