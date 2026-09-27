"""skills/web/google_search.py

Web-search skill used by the intent router for ``search_web`` requests.

The intent engine provides the requested search phrase in ``intent["target"]``.
This module URL-encodes that phrase, opens a Google results page in the user's
default browser, and returns a short response with an avatar expression tag for
the speech and presentation layers. It does not fetch or parse search results
itself. Requests without a target are rejected with a clarification response so
the browser is not opened with an empty query.
"""
import webbrowser, urllib.parse, logging
logger = logging.getLogger(__name__)

async def execute(intent: dict, text: str) -> str:
    """Open a Google search for the intent target and return a speech-ready reply."""
    query = intent.get("target", "").strip()
    if not query:
        return "[surprised] What would you like me to search for?"
    url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    webbrowser.open(url)
    return f"[happy] Searching Google for {query} right away senpai!"