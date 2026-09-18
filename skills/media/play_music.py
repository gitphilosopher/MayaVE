"""skills/media/play_music.py — media key control"""
import logging
logger = logging.getLogger(__name__)

try:
    import keyboard as _kb
    _KB = True
except ImportError:
    _KB = False

_KEY_MAP = {
    "play_music":  "play/pause media",
    "pause_music": "play/pause media",
    "next_track":  "next track",
    "prev_track":  "previous track",
    "volume_up":   "volume up",
    "volume_down": "volume down",
    "mute":        "volume mute",
}

async def execute(intent: dict, text: str) -> str:
    if not _KB:
        return "[sad] Install the 'keyboard' library for media control."
    key = _KEY_MAP.get(intent.get("intent", ""), "play/pause media")
    repeat = 5 if "volume" in key else 1
    for _ in range(repeat):
        _kb.send(key)
    labels = {
        "play/pause media": "[happy] Toggling play/pause senpai!",
        "next track":       "[excited] Skipping to the next track!",
        "previous track":   "[relaxed] Going back to the previous track.",
        "volume up":        "[happy] Volume increased senpai!",
        "volume down":      "[relaxed] Volume decreased senpai.",
        "volume mute":      "[neutral] Audio muted.",
    }
    return labels.get(key, "[neutral] Done.")