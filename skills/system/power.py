"""skills/system/power.py — shutdown / restart with spoken confirmation.

Two-step power-control skill for Windows.

This module is responsible for the confirmation-gated shutdown and restart flow.
When the router sees a power intent, `execute()` asks for explicit confirmation
instead of acting immediately. If the next utterance confirms within the TTL, the
request is executed with `shutdown /s /t <delay>` or `shutdown /r /t <delay>`;
if it is denied or unrelated, the pending request is discarded without acting.

The design is intentionally one-shot and short-lived: a stale confirmation request
expires after a fixed timeout, so a late "yes" cannot trigger a power action in
error.
"""
import asyncio
import logging
import platform
import subprocess
import time

from config.settings import config
from core.confirmation import is_confirm, is_deny

logger = logging.getLogger(__name__)
_U = config.user_name

_DELAY_S = 10
_CONFIRM_TTL_S = 30

_ACTIONS = {
    "shutdown": ("/s", "shut down"),
    "restart":  ("/r", "restart"),
}

_pending: tuple[str, float] | None = None


async def execute(intent: dict, text: str) -> str:
    """Ask for confirmation before scheduling a Windows shutdown or restart."""
    global _pending
    name = intent.get("intent", "")
    _, verb = _ACTIONS[name]
    if platform.system() != "Windows":
        return f"[sad] I can only {verb} the computer on Windows, {_U}."
    _pending = (name, time.monotonic() + _CONFIRM_TTL_S)
    logger.info(f"Power action '{name}' awaiting confirmation.")
    return f"[surprised] Are you sure you want me to {verb} the computer, {_U}? Say yes to confirm."


async def resolve_pending(text: str) -> str | None:
    """Consume a pending power request if the next utterance confirms, denies, or expires it."""
    global _pending
    if _pending is None:
        return None
    name, expires = _pending
    _pending = None
    if time.monotonic() > expires:
        return None
    if is_confirm(text):
        return await asyncio.get_running_loop().run_in_executor(None, _run, name)
    if is_deny(text):
        logger.info(f"Power action '{name}' declined.")
        return f"[relaxed] Okay {_U}, cancelled."
    logger.info(f"Power action '{name}' dropped — unrelated utterance.")
    return None


def _run(name: str) -> str:
    """Schedule the OS-level shutdown or restart with a short delay for the goodbye line to finish."""
    flag, verb = _ACTIONS[name]
    try:
        subprocess.run(["shutdown", flag, "/t", str(_DELAY_S)], check=True, capture_output=True)
        logger.info(f"Power action '{name}' scheduled in {_DELAY_S}s.")
        return f"[relaxed] Okay {_U}, I'll {verb} the computer in {_DELAY_S} seconds. Goodbye!"
    except Exception as e:
        logger.error(f"Power action '{name}' failed: {e}")
        return f"[sad] I couldn't {verb} the computer, {_U}."