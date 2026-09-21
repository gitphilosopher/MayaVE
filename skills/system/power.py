"""skills/system/power.py — shutdown / restart with spoken confirmation."""
import asyncio
import logging
import platform
import re
import subprocess
import time

from config.settings import config

logger = logging.getLogger(__name__)
_U = config.user_name

_DELAY_S       = 10   # OS countdown, so the goodbye line can play first
_CONFIRM_TTL_S = 30   # a stale request expires; a late "yes" does nothing

# intent -> (shutdown.exe flag, spoken verb)
_ACTIONS = {
    "shutdown": ("/s", "shut down"),
    "restart":  ("/r", "restart"),
}

_ADDRESS = rf"(?:\s+(?:{re.escape(config.name.lower())}|{re.escape(_U.lower())}))?"
_CONFIRM_RE = re.compile(
    rf"^(?:yes|yeah|yep|yup|confirm|confirmed|affirmative|do it|go ahead|proceed)"
    rf"(?:\s+please)?{_ADDRESS}$"
)
_DENY_RE = re.compile(
    rf"^(?:no|nope|nah|cancel|abort|never ?mind|don'?t)"
    rf"(?:\s+(?:please|thanks|thank you))?{_ADDRESS}$"
)

# (intent name, expiry on time.monotonic()) while awaiting confirmation.
_pending: tuple[str, float] | None = None


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z' ]", "", text.lower()).split())


async def execute(intent: dict, text: str) -> str:
    """Routed for the shutdown/restart intents — asks, never acts."""
    global _pending
    name = intent.get("intent", "")
    _, verb = _ACTIONS[name]
    if platform.system() != "Windows":
        return f"[sad] I can only {verb} the computer on Windows, {_U}."
    _pending = (name, time.monotonic() + _CONFIRM_TTL_S)
    logger.info(f"Power action '{name}' awaiting confirmation.")
    return f"[surprised] Are you sure you want me to {verb} the computer, {_U}? Say yes to confirm."


async def resolve_pending(text: str) -> str | None:
    """
    Called by Router.dispatch before intent routing. One-shot: the next
    utterance always consumes a pending request. Returns the spoken reply
    if it confirmed or declined; None if nothing was pending, it expired,
    or the utterance was unrelated (then it routes normally, request dropped).
    """
    global _pending
    if _pending is None:
        return None
    name, expires = _pending
    _pending = None
    if time.monotonic() > expires:
        return None
    t = _normalize(text)
    if _CONFIRM_RE.match(t):
        return await asyncio.get_running_loop().run_in_executor(None, _run, name)
    if _DENY_RE.match(t):
        logger.info(f"Power action '{name}' declined.")
        return f"[relaxed] Okay {_U}, cancelled."
    logger.info(f"Power action '{name}' dropped — unrelated utterance.")
    return None


def _run(name: str) -> str:
    flag, verb = _ACTIONS[name]
    try:
        subprocess.run(["shutdown", flag, "/t", str(_DELAY_S)], check=True, capture_output=True)
        logger.info(f"Power action '{name}' scheduled in {_DELAY_S}s.")
        return f"[relaxed] Okay {_U}, I'll {verb} the computer in {_DELAY_S} seconds. Goodbye!"
    except Exception as e:
        logger.error(f"Power action '{name}' failed: {e}")
        return f"[sad] I couldn't {verb} the computer, {_U}."
