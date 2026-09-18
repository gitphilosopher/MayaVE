"""skills/utilities/datetime_skill.py"""
import datetime

async def execute(intent: dict, text: str) -> str:
    now = datetime.datetime.now()
    t = text.lower()
    if "time" in t:
        return f"[happy] Beep boop! My internal clock says it's {now.strftime('%I:%M %p')}."
    if "date" in t or "today" in t:
        return f"[relaxed] Calendar check complete! Today is {now.strftime('%A, %B %d, %Y')}."
    if "day" in t:
        return f"[happy] Yep, definitely {now.strftime('%A')}. I checked twice."
    return f"[relaxed] After consulting the ancient scrolls... {now.strftime('%I:%M %p on %A, %B %d, %Y')}."