"""skills/utilities/reminder.py — background countdown timers"""
import asyncio, re, logging
logger = logging.getLogger(__name__)

def _parse_seconds(text: str) -> int:
    total = 0
    for amt, unit in re.findall(r"(\d+)\s*(second|minute|hour)s?", text):
        amt = int(amt)
        if "second" in unit: total += amt
        elif "minute" in unit: total += amt * 60
        elif "hour" in unit: total += amt * 3600
    return total

async def execute(intent: dict, text: str) -> str:
    secs = _parse_seconds(text)
    if secs <= 0:
        return "[sad] I couldn't figure out the duration. [neutral] Try: 'set a timer for 5 minutes'."

    async def _fire():
        await asyncio.sleep(secs)
        print(f"\n⏰ Timer done! ({text})")

    asyncio.create_task(_fire())
    mins, s = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    parts = []
    if hrs:  parts.append(f"{hrs}h")
    if mins: parts.append(f"{mins}m")
    if s:    parts.append(f"{s}s")
    return f"[happy] Timer set for {' '.join(parts)}."
