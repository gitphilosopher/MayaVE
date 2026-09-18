"""skills/web/open_website.py"""
import webbrowser, logging
logger = logging.getLogger(__name__)

_SITES = {
    "youtube": "https://www.youtube.com",
    "github":  "https://www.github.com",
    "reddit":  "https://www.reddit.com",
    "netflix": "https://www.netflix.com",
    "spotify": "https://open.spotify.com",
    "gmail":   "https://mail.google.com",
    "twitter": "https://www.twitter.com",
    "google":  "https://www.google.com",
}

async def execute(intent: dict, text: str) -> str:
    t = text.lower()
    for name, url in _SITES.items():
        if name in t:
            webbrowser.open(url)
            return f"[happy] Opening {name.capitalize()} for you senpai!"
    target = intent.get("target", "").strip()
    if target:
        url = f"https://{target}" if not target.startswith("http") else target
        webbrowser.open(url)
        return f"[happy] Opening {url} senpai!"
    return "[surprised] Which website would you like me to open?"