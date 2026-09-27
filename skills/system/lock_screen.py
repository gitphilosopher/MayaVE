"""skills/system/lock_screen.py — lock the Windows workstation.

Small system skill that locks the current Windows session via the Win32
`LockWorkStation` API.

This module is intentionally minimal: the router calls `execute(intent, text)`,
which checks the OS and returns a user-facing status string without raising to the
caller. The Windows-specific implementation is performed locally via
`ctypes.windll.user32.LockWorkStation()`, and any failure is logged and surfaced
as a spoken error message instead of crashing the runtime.
"""
import logging
import platform

from config.settings import config

logger = logging.getLogger(__name__)
_U = config.user_name


async def execute(intent: dict, text: str) -> str:
    """Attempt to lock the workstation on Windows and return a user-facing status message."""
    if platform.system() != "Windows":
        return f"[sad] I can only lock the screen on Windows, {_U}."
    try:
        import ctypes
        if not ctypes.windll.user32.LockWorkStation():
            raise OSError("LockWorkStation returned 0")
        return f"[relaxed] Locking the screen, {_U}."
    except Exception as e:
        logger.error(f"Lock screen failed: {e}")
        return f"[sad] I couldn't lock the screen, {_U}."
