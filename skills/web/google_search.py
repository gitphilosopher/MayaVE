"""skills/web/google_search.py"""
import webbrowser, urllib.parse, logging
logger = logging.getLogger(__name__)

async def execute(intent: dict, text: str) -> str:
    query = intent.get("target", "").strip()
    if not query:
        return "[surprised] What would you like me to search for?"
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    webbrowser.open(url)
    return f"[happy] Searching Google for {query} right away senpai!"