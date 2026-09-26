"""
skills/system/open_target.py

Merged from the former separate skills/web/open_website.py and
skills/system/open_app.py, behind the single 'open_target' intent (see
datasets/intents.json) — the classifier no longer tells a website request
apart from an app-launch request, so this skill resolves it itself.

Resolution order (most to least confident):
  1. A known name from _SITES appearing anywhere in the utterance
     (checked on the raw text, same as the former open_website).
  2. A known name from _APP_TABLE (Windows launch table), after cleaning
     the extracted target ("the chrome browser please" -> "chrome").
  3. A strong website signal in the raw target — explicit http(s) URL,
     a dotted host ("example.org"), or spoken "dot" ("github dot com").
  4. App-launch pass-through: the cleaned target handed to the OS
     unchanged (ShellExecute/os.startfile can resolve many registered
     apps by name even when they're not in _APP_TABLE) — same as the
     former open_app's fallback for an unrecognised name.
  5. Last resort: treat a bare single-word target as a website label
     ("wikipedia" -> https://www.wikipedia.com) — same as the former
     open_website's weakest heuristic, now tried only AFTER an app
     launch has already failed. Guessing a wrong domain silently
     "succeeds" while being wrong, which is worse than the apology an
     app-launch failure already gives, so it's deliberately last.

Table entries are Windows-only and unverified at runtime — extend
_APP_TABLE as real launch failures show up. On macOS/Linux the cleaned
name is used as-is for app launches (see _launch).
"""
import asyncio
import logging
import os
import platform
import re
import subprocess
import webbrowser

from config.settings import config

logger = logging.getLogger(__name__)
_OS = platform.system()
_U = config.user_name

# ── Known websites ───────────────────────────────────────────────────────
_SITES = {
    "youtube":        "https://www.youtube.com",
    "github":         "https://www.github.com",
    "reddit":         "https://www.reddit.com",
    "netflix":        "https://www.netflix.com",
    "spotify":        "https://open.spotify.com",
    "gmail":          "https://mail.google.com",
    "twitter":        "https://www.twitter.com",
    "google":         "https://www.google.com",
    "amazon":         "https://www.amazon.com",
    "linkedin":       "https://www.linkedin.com",
    "stackoverflow":  "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
}
_SITE_RES = {name: re.compile(rf"\b{re.escape(name)}\b") for name in _SITES}

_LABEL_RE  = re.compile(r"^[a-z0-9-]+$")
_HOST_RE   = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?$")
_FILLER_RE = re.compile(r"^(?:the\s+)?(?:website\s+)?")


def _resolve_url_strong(target: str) -> str | None:
    """Explicit URL, dotted host, or spoken 'dot' — a target that already
    unambiguously reads as a website. None if it doesn't."""
    t = target.strip().lower().rstrip(".?!,")
    if not t:
        return None
    if t.startswith(("http://", "https://")):
        return t
    t = re.sub(r"\s+dot\s+", ".", t)          # STT: "github dot com"
    t = _FILLER_RE.sub("", t).strip()
    if not t or re.search(r"\s", t):
        return None
    if _HOST_RE.match(t):
        return f"https://{t}"
    return None


def _resolve_url_label_guess(target: str) -> str | None:
    """Weakest signal: a bare single-word target guessed as <label>.com.
    Only tried after every stronger signal AND an app-launch attempt have
    already failed (see module docstring, step 5)."""
    t = _FILLER_RE.sub("", target.strip().lower().rstrip(".?!,")).strip()
    if t and _LABEL_RE.match(t):
        return f"https://www.{t}.com"
    return None


# ── Known desktop apps (Windows) ─────────────────────────────────────────
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


def _clean_app_name(target: str) -> str:
    """'the chrome browser please.' -> 'chrome'. Never strips the last word."""
    t = _LEADING_RE.sub("", target.lower().strip().rstrip(".?!,"))
    words = t.split()
    while len(words) > 1 and words[-1] in _TRAILING_WORDS:
        words.pop()
    return " ".join(words)


def _lookup_app(name: str) -> tuple[str, tuple[str, ...]] | None:
    """(display name, launch candidates) if `name` is a known app; None if
    it isn't — callers decide what an unknown name means (pass-through vs.
    a website guess), this just reports the table lookup itself."""
    if _OS == "Windows" and name in _APPS:
        return _APPS[name]
    return None


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
                logger.debug(f"open_target: '{candidate}' failed: {e}")
        raise last or OSError("no launch candidate")
    if _OS == "Darwin":
        subprocess.Popen(["open", "-a", candidates[0]])
    else:
        subprocess.Popen([candidates[0]])


async def _launch_app(display: str, candidates: tuple[str, ...]) -> str | None:
    """Returns the spoken success reply, or None on failure so the caller
    can move on to the next fallback instead of apologising immediately."""
    try:
        await asyncio.get_running_loop().run_in_executor(None, _launch, candidates)
        return f"[happy] Opening {display}, {_U}."
    except Exception as e:
        logger.debug(f"open_target: app launch failed for '{display}' ({candidates}): {e}")
        return None


async def execute(intent: dict, text: str) -> str:
    t = text.lower()

    # 1. Known website name anywhere in the utterance.
    for name, url in _SITES.items():
        if _SITE_RES[name].search(t):
            webbrowser.open(url)
            return f"[happy] Opening {name.capitalize()} for you senpai!"

    raw_target = intent.get("target", "").strip()
    cleaned = _clean_app_name(raw_target) if raw_target else ""

    # 2. Known desktop app.
    if cleaned:
        known = _lookup_app(cleaned)
        if known:
            reply = await _launch_app(*known)
            if reply:
                return reply

    # 3. Strong website signal — explicit URL, dotted host, spoken "dot".
    if raw_target:
        url = _resolve_url_strong(raw_target)
        if url:
            webbrowser.open(url)
            return f"[happy] Opening {url} senpai!"

    # 4. App-launch pass-through for an unrecognised name — previous
    # open_app fallback: hand it to the OS unchanged.
    if cleaned:
        reply = await _launch_app(cleaned, (cleaned,))
        if reply:
            return reply

    # 5. Last resort — guess a bare single word as a website label.
    if raw_target:
        url = _resolve_url_label_guess(raw_target)
        if url:
            webbrowser.open(url)
            return f"[happy] Opening {url} senpai!"

    if not raw_target:
        return f"[surprised] What would you like me to open, {_U}?"
    logger.info(f"open_target: '{raw_target}' could not be resolved as a site or an app.")
    return f"[sad] I couldn't open '{raw_target}', {_U}."
