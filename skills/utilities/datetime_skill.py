"""skills/utilities/datetime_skill.py

Lightweight time/date helper for Maya's utility skill set.

This module responds to simple time and date requests by inspecting the local
system clock and returning a short, user-facing sentence. It does not maintain
state or rely on any external service; it only distinguishes whether the request
is about the current time, the current date, the weekday, or a general clock
summary.
"""
import datetime

async def execute(intent: dict, text: str) -> str:
    """Return a friendly time or date response based on the user's request text."""
    now = datetime.datetime.now()
    t = text.lower()
    if "time" in t:
        return f"[happy] Beep boop! My internal clock says it's {now.strftime('%I:%M %p')}."
    if "date" in t or "today" in t:
        return f"[relaxed] Calendar check complete! Today is {now.strftime('%A, %B %d, %Y')}."
    if "day" in t:
        return f"[happy] Yep, definitely {now.strftime('%A')}. I checked twice."
    return f"[relaxed] After consulting the ancient scrolls... {now.strftime('%I:%M %p on %A, %B %d, %Y')}."