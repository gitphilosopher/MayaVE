"""
skills/system/clipboard.py
Read from and write to the system clipboard.
"""

import logging
import pyperclip

from config.settings import config

logger = logging.getLogger(__name__)
_U = config.user_name


async def execute(intent: dict, text: str) -> str:
    action = intent.get("intent", "")

    if action == "clipboard_read":
        return _read()
    elif action == "clipboard_write":
        return _write(text)
    elif action == "clipboard_clear":
        return _clear()
    else:
        t = text.lower()
        if any(w in t for w in ("read", "what's", "whats", "copy", "get", "show")):
            return _read()
        elif any(w in t for w in ("clear", "empty", "wipe", "delete")):
            return _clear()
        else:
            return _read()


def _read() -> str:
    try:
        content = pyperclip.paste()
        if not content or not content.strip():
            return f"[neutral] Your clipboard is empty, {_U}."
        preview = content.strip()
        if len(preview) > 200:
            preview = preview[:200] + "…"
        return f"[relaxed] Your clipboard has: {preview}"
    except Exception as e:
        logger.error(f"Clipboard read error: {e}")
        return f"[sad] I couldn't read the clipboard, {_U}."


def _write(text: str) -> str:
    import re
    content = re.sub(r'(?i)(copy|write|put|save|add|set)\s+', '', text, count=1).strip()
    content = re.sub(r'(?i)\s*(to|in(to)?|on)\s*(the\s+)?clipboard\s*$', '', content).strip()

    if not content:
        return f"[surprised] What would you like me to copy to the clipboard, {_U}?"
    try:
        pyperclip.copy(content)
        return f"[happy] Copied to clipboard, {_U}."
    except Exception as e:
        logger.error(f"Clipboard write error: {e}")
        return f"[sad] I couldn't write to the clipboard, {_U}."


def _clear() -> str:
    try:
        pyperclip.copy("")
        return f"[relaxed] Clipboard cleared, {_U}."
    except Exception as e:
        logger.error(f"Clipboard clear error: {e}")
        return f"[sad] I couldn't clear the clipboard, {_U}."