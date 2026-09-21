"""
skills/system/open_app.py

Spoken app names are resolved through _APP_TABLE before launching (Windows):
"calculator" -> calc.exe, "file explorer" -> explorer.exe, "vs code" -> code,
"discord" -> discord: (protocol handler) and so on. Each entry lists launch
candidates tried in order, so e.g. Windows Terminal can fall back to cmd.
A name that isn't in the table is passed through unchanged (the previous
behaviour), so any app ShellExecute can resolve by name still works.

The target is normalised first ("the chrome browser please" -> "chrome").
Launch runs in the executor (os.startfile can block on a slow shell), and
replies carry expression tags like the other skills.

Table entries are Windows-only and unverified at runtime — extend it as
real launch failures show up. On macOS/Linux the cleaned name is used as-is.
"""
import asyncio
import logging
import os
import platform
import re
import subprocess

from config.settings import config

logger = logging.getLogger(__name__)
_OS = platform.system()
_U = config.user_name

# (spoken aliases...) -> (display name, launch candidates tried in order).
# A candidate is anything os.startfile accepts: an exe / App Paths name or a
# protocol URI.
_APP_TABLE: dict[tuple[str, ...], tuple[str, tuple[str, ...]]] = {
    ("notepad",):                                    ("Notepad",          ("notepad.exe",)),
    ("calculator", "calc"):                          ("Calculator",       ("calc.exe",)),
    ("file explorer", "explorer", "windows explorer"): ("File Explorer",  ("explorer.exe",)),
    ("task manager",):                               ("Task Manager",     ("taskmgr.exe",)),
    ("terminal", "windows terminal"):                ("Terminal",         ("wt.exe", "cmd.exe")),
    ("command prompt", "cmd"):                       ("Command Prompt",   ("cmd.exe",)),
    ("powershell", "power shell"):                   ("PowerShell",       ("powershell.exe",)),
    ("paint", "ms paint"):                           ("Paint",            ("mspaint.exe",)),
    ("settings", "windows settings"):                ("Settings",         ("ms-settings:",)),
    ("chrome", "google chrome"):                     ("Chrome",           ("chrome.exe",)),
    ("edge", "microsoft edge"):                      ("Edge",             ("msedge.exe",)),
    ("firefox",):                                    ("Firefox",          ("firefox.exe",)),
    ("word", "microsoft word"):                      ("Word",             ("winword.exe",)),
    ("excel", "microsoft excel"):                    ("Excel",            ("excel.exe",)),
    ("powerpoint", "microsoft powerpoint"):          ("PowerPoint",       ("powerpnt.exe",)),
    ("vs code", "vscode", "visual studio code"):     ("Visual Studio Code", ("code.exe", "code")),
    ("discord",):                                    ("Discord",          ("discord:",)),
    ("whatsapp",):                                   ("WhatsApp",         ("whatsapp:",)),
}
_APPS = {alias: entry for aliases, entry in _APP_TABLE.items() for alias in aliases}

_LEADING_RE = re.compile(r"^(?:(?:please|the|my|a|an)\s+)+")
_TRAILING_WORDS = {"app", "application", "program", "browser", "please"}


def _clean(target: str) -> str:
    """'the chrome browser please.' -> 'chrome'. Never strips the last word."""
    t = _LEADING_RE.sub("", target.lower().strip().rstrip(".?!,"))
    words = t.split()
    while len(words) > 1 and words[-1] in _TRAILING_WORDS:
        words.pop()
    return " ".join(words)


def _resolve(name: str) -> tuple[str, tuple[str, ...]]:
    """(display name, launch candidates). Unknown names pass through unchanged."""
    if _OS == "Windows" and name in _APPS:
        return _APPS[name]
    return name, (name,)


def _launch(candidates: tuple[str, ...]) -> None:
    """Blocking — runs in an executor. Raises if nothing could be launched."""
    if _OS == "Windows":
        last: OSError | None = None
        for candidate in candidates:
            try:
                os.startfile(candidate)
                return
            except OSError as e:
                last = e
                logger.debug(f"open_app: '{candidate}' failed: {e}")
        raise last or OSError("no launch candidate")
    if _OS == "Darwin":
        subprocess.Popen(["open", "-a", candidates[0]])
    else:
        subprocess.Popen([candidates[0]])


async def execute(intent: dict, text: str) -> str:
    target = _clean(intent.get("target", ""))
    if not target:
        return f"[surprised] Which app would you like me to open, {_U}?"

    display, candidates = _resolve(target)
    try:
        await asyncio.get_running_loop().run_in_executor(None, _launch, candidates)
        return f"[happy] Opening {display}, {_U}."
    except Exception as e:
        logger.error(f"open_app failed for '{target}' ({candidates}): {e}")
        return f"[sad] I couldn't open {display}, {_U}."