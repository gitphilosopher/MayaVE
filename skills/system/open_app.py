"""skills/system/open_app.py"""
import os, subprocess, platform, logging
logger = logging.getLogger(__name__)
_OS = platform.system()

async def execute(intent: dict, text: str) -> str:
    target = intent.get("target", "").strip()
    if not target:
        return "Which app would you like me to open?"
    try:
        if _OS == "Windows":
            os.startfile(target)
        elif _OS == "Darwin":
            subprocess.Popen(["open", "-a", target])
        else:
            subprocess.Popen([target])
        return f"Opening {target}."
    except Exception as e:
        logger.error(e)
        return f"I couldn't open {target}."
