"""skills/system/system_info.py"""
import logging, os, platform
from core.mood import mood_manager

logger = logging.getLogger(__name__)

async def execute(intent: dict, text: str) -> str:
    try:
        import psutil
    except ImportError:
        return "[sad] Please install psutil: pip install psutil"

    t = text.lower()

    if "battery" in t:
        b = psutil.sensors_battery()
        if not b:
            return "[surprised] No battery detected."
        p = b.percent
        if b.power_plugged:
            if p >= 30:
                return f"[happy] Senpai, we're at {p:.0f}% and plugged in, so no worries."
            elif p >= 10:
                return f"[relaxed] We're currently at {p:.0f}% battery. We'll be back to full power soon!"
            else:
                mood_manager.report_event(
                    source="skill", emotion="angry", intensity=0.2,
                    reason=f"battery low ({p:.0f}%) despite being plugged in",
                )
                return f"[angry] Senpai, we're down to {p:.0f}% so don't even try to unplug me."
        else:
            if p >= 50:
                return f"[happy] Current battery level is {p:.0f}%. That's plenty of energy in me."
            elif p >= 30:
                return f"[relaxed] Senpai, we're sitting at {p:.0f}% and no worries yet."
            elif p >= 10:
                return f"[surprised] Uh-oh, senpai we're down to {p:.0f}%. A charger might be a good idea."
            else:
                mood_manager.report_event(
                    source="skill", emotion="angry", intensity=0.35,
                    reason=f"battery critical ({p:.0f}%), unplugged",
                )
                return f"[angry] Battery critical: {p:.0f}%. Find me a charger before I suffer a living death."

    if "cpu" in t:
        cpu = psutil.cpu_percent(interval=1)
        if cpu < 30:
            return f"[happy] The processor is taking it easy at {cpu}%, senpai. Plenty of headroom."
        elif cpu < 70:
            return f"[relaxed] Current processor load: {cpu}%. Everything looks normal senpai."
        elif cpu < 90:
            return f"[surprised] CPU usage has reached {cpu}%. Someone's been keeping me busy huh!"
        else:
            mood_manager.report_event(
                source="skill", emotion="angry", intensity=0.25,
                reason=f"CPU usage critical ({cpu}%)",
            )
            return f"[angry] Senpai! CPU usage is extremely high at {cpu}%. Things are getting intense!"

    if "ram" in t or "memory" in t:
        r = psutil.virtual_memory()
        ram = r.percent
        if ram < 40:
            return f"[happy] Memory usage is currently {ram}%. Looking healthy."
        elif ram < 75:
            return f"[relaxed] Senpai currently we're using {ram}% of available RAM."
        elif ram < 90:
            return f"[surprised] RAM usage is up to {ram}%. Senpai you've got quite a busy workspace today."
        else:
            mood_manager.report_event(
                source="skill", emotion="angry", intensity=0.25,
                reason=f"RAM usage critical ({ram}%)",
            )
            return f"[angry] Yikes! RAM usage is at {ram}%. We're almost out of breathing room senpai."

    if "disk" in t:
        d = psutil.disk_usage("/")
        return (
            f"[relaxed] Disk: {d.used // 1024**3} GB used of "
            f"{d.total // 1024**3} GB ({d.percent}%)."
        )

    if "screenshot" in t:
        try:
            import pyautogui
            from pathlib import Path
            onedrive = os.environ.get("OneDrive")
            if onedrive:
                path = Path(onedrive) / "Pictures" / "Screenshots"
            else:
                logging.log("OneDrive folder not found")
                path = os.path.expanduser("~/Pictures/Screenshots")
            pyautogui.screenshot(path)
            return f"[happy] Screenshot saved to {path}"
        except ImportError:
            return "[sad] Install pyautogui for screenshots."

    return "[neutral] What system information do you need?"