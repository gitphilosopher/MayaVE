"""
skills/system/open_target.py
Resolve a user request to either a known website, a known desktop app, or an
unqualified target that should be passed to the OS.

This skill is the single `open_target` handler for both website-opening and
app-launch requests. The classifier does not separate them cleanly, so the logic
must decide which interpretation is most likely from the spoken text and the
parsed target. The resolution order is intentionally conservative: known website
names are checked first, then known Windows app aliases, then explicit website
signals such as a URL or a dotted host, then pass-through app launch, and only
then a last-resort website guess for a bare word.

The purpose is to keep the router simple while preserving the older behavior of
`open_website` and `open_app`: known destinations are opened directly, stronger
signals win before weaker ones, and failures degrade to a user-facing apology
instead of silently opening the wrong destination.
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
    """Recognize a direct website target such as a URL, dotted host, or spoken 'dot' form."""
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
    """Guess a bare single-word target as a website label only after stronger options fail."""
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
    """Normalize an app-like target by stripping filler words and trailing app labels."""
    t = _LEADING_RE.sub("", target.lower().strip().rstrip(".?!,"))
    words = t.split()
    while len(words) > 1 and words[-1] in _TRAILING_WORDS:
        words.pop()
    return " ".join(words)


def _lookup_app(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return known Windows app metadata for a normalized name if it exists in the lookup table."""
    if _OS == "Windows" and name in _APPS:
        return _APPS[name]
    return None


def _launch(candidates: tuple[str, ...]) -> None:
    """Attempt each launch candidate in order until one opens successfully or all fail."""
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
    """Launch an app and return a success message only if the candidate list succeeds."""
    try:
        await asyncio.get_running_loop().run_in_executor(None, _launch, candidates)
        return f"[happy] Opening {display}, {_U}."
    except Exception as e:
        logger.debug(f"open_target: app launch failed for '{display}' ({candidates}): {e}")
        return None


async def execute(intent: dict, text: str) -> str:
    """Resolve a target to a website or app launch using the fallback priority defined for this skill."""
    t = text.lower()

    for name, url in _SITES.items():
        if _SITE_RES[name].search(t):
            webbrowser.open(url)
            return f"[happy] Opening {name.capitalize()} for you senpai!"

    raw_target = intent.get("target", "").strip()
    cleaned = _clean_app_name(raw_target) if raw_target else ""

    if cleaned:
        known = _lookup_app(cleaned)
        if known:
            reply = await _launch_app(*known)
            if reply:
                return reply

    if raw_target:
        url = _resolve_url_strong(raw_target)
        if url:
            webbrowser.open(url)
            return f"[happy] Opening {url} senpai!"

    if cleaned:
        reply = await _launch_app(cleaned, (cleaned,))
        if reply:
            return reply

    if raw_target:
        url = _resolve_url_label_guess(raw_target)
        if url:
            webbrowser.open(url)
            return f"[happy] Opening {url} senpai!"

    if not raw_target:
        return f"[surprised] What would you like me to open, {_U}?"
    logger.info(f"open_target: '{raw_target}' could not be resolved as a site or an app.")
    return f"[sad] I couldn't open '{raw_target}', {_U}."
