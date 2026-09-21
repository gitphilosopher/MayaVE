"""
skills/web/open_website.py

Resolution order:
  1. A known name from _SITES appearing in the utterance (unchanged).
  2. intent["target"] resolved by _resolve_url():
       - explicit http(s) URL           -> used as-is
       - dotted host ("example.org")    -> https://example.org
       - bare single label ("amazon")   -> https://www.amazon.com
       - spoken dots ("github dot com") -> normalised to "github.com" first
     Anything else (multi-word phrases, junk) is NOT turned into a URL —
     _extract_target() falls back to the whole utterance when no trigger
     strips, and "https://my project board" is not a website.
"""
import re
import webbrowser
import logging
logger = logging.getLogger(__name__)

_SITES = {
    "youtube":       "https://www.youtube.com",
    "github":        "https://www.github.com",
    "reddit":        "https://www.reddit.com",
    "netflix":       "https://www.netflix.com",
    "spotify":       "https://open.spotify.com",
    "gmail":         "https://mail.google.com",
    "twitter":       "https://www.twitter.com",
    "google":        "https://www.google.com",
    "amazon":        "https://www.amazon.com",
    "linkedin":      "https://www.linkedin.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
}

_LABEL_RE = re.compile(r"^[a-z0-9-]+$")
_HOST_RE  = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?$")
_FILLER_RE = re.compile(r"^(?:the\s+)?(?:website\s+)?")


def _resolve_url(target: str) -> str | None:
    """Turn a spoken/extracted target into a valid URL, or None if it isn't one."""
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
    if _LABEL_RE.match(t):
        return f"https://www.{t}.com"
    return None


async def execute(intent: dict, text: str) -> str:
    t = text.lower()
    for name, url in _SITES.items():
        if name in t:
            webbrowser.open(url)
            return f"[happy] Opening {name.capitalize()} for you senpai!"
    target = intent.get("target", "").strip()
    if target:
        url = _resolve_url(target)
        if url:
            webbrowser.open(url)
            return f"[happy] Opening {url} senpai!"
        logger.info(f"open_website: '{target}' is not a resolvable site.")
        return "[surprised] I couldn't tell which website you meant. Could you say the site name?"
    return "[surprised] Which website would you like me to open?"