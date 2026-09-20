"""skills/system/lock_screen.py — lock the Windows workstation."""
import logging
import platform

from config.settings import config

logger = logging.getLogger(__name__)
_U = config.user_name


async def execute(intent: dict, text: str) -> str:
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
