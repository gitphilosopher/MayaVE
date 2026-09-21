"""
skills/system/system_info.py

Event-loop rule: psutil.cpu_percent(interval=1) sleeps for a full second
and pyautogui.screenshot() does a synchronous screen grab + PNG write, so
all reply building runs in the default executor. mood_manager is loop-owned
state, so the worker only *describes* a mood event (kwargs dict); execute()
reports it back on the loop after the executor call returns.
"""
import asyncio
import logging
import os
import re

from core.mood import mood_manager

logger = logging.getLogger(__name__)

_BATTERY_RE    = re.compile(r"\bbatter(?:y|ies)\b")
_CPU_RE        = re.compile(r"\bcpus?\b")
_RAM_RE        = re.compile(r"\b(?:ram|memory)\b")
_DISK_RE       = re.compile(r"\bdisks?\b")
_SCREENSHOT_RE = re.compile(r"\bscreen\s?shots?\b")

async def execute(intent: dict, text: str) -> str:
    loop = asyncio.get_running_loop()
    reply, event = await loop.run_in_executor(
        None, _build_reply, text, intent.get("intent", "")
    )
    if event:
        mood_manager.report_event(**event)
    return reply


def _build_reply(text: str, intent_name: str = "") -> tuple[str, dict | None]:
    """Blocking. Returns (reply, mood_event_kwargs_or_None)."""
    try:
        import psutil
    except ImportError:
        return "[sad] Please install psutil: pip install psutil", None

    t = text.lower()

    if _BATTERY_RE.search(t): 
        b = psutil.sensors_battery()
        if not b:
            return "[surprised] No battery detected.", None
        p = b.percent
        if b.power_plugged:
            if p >= 30:
                return f"[happy] Senpai, we're at {p:.0f}% and plugged in, so no worries.", None
            elif p >= 10:
                return f"[relaxed] We're currently at {p:.0f}% battery. We'll be back to full power soon!", None
            else:
                return (
                    f"[angry] Senpai, we're down to {p:.0f}% so don't even try to unplug me.",
                    dict(source="skill", emotion="angry", intensity=0.2,
                         reason=f"battery low ({p:.0f}%) despite being plugged in"),
                )
        else:
            if p >= 50:
                return f"[happy] Current battery level is {p:.0f}%. That's plenty of energy in me.", None
            elif p >= 30:
                return f"[relaxed] Senpai, we're sitting at {p:.0f}% and no worries yet.", None
            elif p >= 10:
                return f"[surprised] Uh-oh, senpai we're down to {p:.0f}%. A charger might be a good idea.", None
            else:
                return (
                    f"[angry] Battery critical: {p:.0f}%. Find me a charger before I suffer a living death.",
                    dict(source="skill", emotion="angry", intensity=0.35,
                         reason=f"battery critical ({p:.0f}%), unplugged"),
                )

    if _CPU_RE.search(t):
        cpu = psutil.cpu_percent(interval=1)
        if cpu < 30:
            return f"[happy] The processor is taking it easy at {cpu}%, senpai. Plenty of headroom.", None
        elif cpu < 70:
            return f"[relaxed] Current processor load: {cpu}%. Everything looks normal senpai.", None
        elif cpu < 90:
            return f"[surprised] CPU usage has reached {cpu}%. Someone's been keeping me busy huh!", None
        else:
            return (
                f"[angry] Senpai! CPU usage is extremely high at {cpu}%. Things are getting intense!",
                dict(source="skill", emotion="angry", intensity=0.25,
                     reason=f"CPU usage critical ({cpu}%)"),
            )

    if _RAM_RE.search(t):
        r = psutil.virtual_memory()
        ram = r.percent
        if ram < 40:
            return f"[happy] Memory usage is currently {ram}%. Looking healthy.", None
        elif ram < 75:
            return f"[relaxed] Senpai currently we're using {ram}% of available RAM.", None
        elif ram < 90:
            return f"[surprised] RAM usage is up to {ram}%. Senpai you've got quite a busy workspace today.", None
        else:
            return (
                f"[angry] Yikes! RAM usage is at {ram}%. We're almost out of breathing room senpai.",
                dict(source="skill", emotion="angry", intensity=0.25,
                     reason=f"RAM usage critical ({ram}%)"),
            )

    if _DISK_RE.search(t):
        d = psutil.disk_usage("/")
        return (
            f"[relaxed] Disk: {d.used // 1024**3} GB used of "
            f"{d.total // 1024**3} GB ({d.percent}%)."
        ), None

    if _SCREENSHOT_RE.search(t) or intent_name == "screenshot":
        try:
            import pyautogui
            from datetime import datetime
            from pathlib import Path
            onedrive = os.environ.get("OneDrive")
            if onedrive:
                folder = Path(onedrive) / "Pictures" / "Screenshots"
            else:
                folder = Path.home() / "Pictures" / "Screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png"
            pyautogui.screenshot(str(path))
            return f"[happy] Screenshot saved to {path}", None
        except ImportError:
            return "[sad] Install pyautogui for screenshots.", None

    return "[neutral] What system information do you need?", None