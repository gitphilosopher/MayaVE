"""
skills/utilities/notepad.py
Voice-driven notepad with expression tags.

Bug fix: note files contain [2026-06-15 12:00] timestamp lines written by
_create() and _append(). When _read() returns file content prefixed with
"[relaxed] Here's your note: [2026-06-15 12:00]...", the timestamp bracket
would be parsed as an expression tag and stripped silently.

Fix: strip timestamp lines from note content before returning to speaker.

Delete confirmation: _delete() only ASKS ("Say yes to confirm") and stores
the resolved note path; resolve_pending() — called first by Router.dispatch,
same pattern as skills/system/power.py — confirms, declines or drops it on
the next utterance. One-shot with a 30 s TTL, so a stale request can never
delete a note on a late "yes". Yes/no phrases come from core/confirmation.py.
"""

import asyncio
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from config.settings import config
from core.confirmation import is_confirm, is_deny

logger = logging.getLogger(__name__)
_U = config.user_name

_NOTES_DIR = Path(getattr(config, "notes_dir", Path.home() / "Maya" / "Notes"))

_CONFIRM_TTL_S = 30   # a stale delete request expires; a late "yes" does nothing

# Matches timestamp lines written by _create/_append: [2026-06-15 12:00]
_TIMESTAMP_RE = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*', re.MULTILINE)

# Filename suffix added by _generate_filename ("_MMDD_HHMM") — not worth speaking.
_STAMP_SUFFIX_RE = re.compile(r"_\d{4}_\d{4}$")

# (note path, expiry on time.monotonic()) while awaiting delete confirmation.
_pending_delete: tuple[Path, float] | None = None


def _spoken_name(note: Path) -> str:
    """'buy_milk_0615_1200' -> 'buy milk' (for the delete prompt/reply only)."""
    return _STAMP_SUFFIX_RE.sub("", note.stem).replace("_", " ") or note.stem.replace("_", " ")


def _ensure_dir() -> None:
    _NOTES_DIR.mkdir(parents=True, exist_ok=True)


async def execute(intent: dict, text: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _handle, intent, text)


async def resolve_pending(text: str) -> str | None:
    """
    Called by Router.dispatch before intent routing. One-shot: the next
    utterance always consumes a pending delete request. Returns the spoken
    reply if it confirmed or declined; None if nothing was pending, it
    expired, or the utterance was unrelated (then it routes normally and the
    request is dropped).
    """
    global _pending_delete
    if _pending_delete is None:
        return None
    note, expires = _pending_delete
    _pending_delete = None
    if time.monotonic() > expires:
        return None
    if is_confirm(text):
        return await asyncio.get_running_loop().run_in_executor(None, _run_delete, note)
    if is_deny(text):
        logger.info(f"Note delete declined: {note.name}")
        return f"[relaxed] Okay {_U}, I kept the note."
    logger.info(f"Note delete dropped — unrelated utterance: {note.name}")
    return None


def _handle(intent: dict, text: str) -> str:
    _ensure_dir()
    t    = text.lower()
    name = intent.get("intent", "")

    # A specific note_* intent wins over word matching, so a note whose
    # content contains "read"/"show"/etc. is saved instead of misrouted.
    handlers = {
        "note_create": _create,
        "note_append": _append,
        "note_read":   _read,
        "note_list":   lambda _text: _list(),
        "note_delete": _delete,
        "note_open":   _open_in_editor,
    }
    if name in handlers:
        return handlers[name](text)

    if name == "note_read"   or any(w in t for w in ("read", "show", "what's in", "whats in", "open note")):
        return _read(text)
    if name == "note_list"   or any(w in t for w in ("list notes", "my notes", "all notes", "what notes")):
        return _list()
    if name == "note_append" or any(w in t for w in ("add to", "append", "add note")):
        return _append(text)
    if name == "note_delete" or any(w in t for w in ("delete note", "remove note")):
        return _delete(text)
    if name == "note_open"   or any(w in t for w in ("open note", "launch note")):
        return _open_in_editor(text)

    return _create(text)


def _create(text: str) -> str:
    content = _extract_content(text, verbs=["note", "write", "save", "create", "take"])
    if not content:
        return f"[surprised] What would you like me to note down, {_U}?"

    filename = _generate_filename(content)
    path = _NOTES_DIR / filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(f"[{timestamp}]\n{content}\n", encoding="utf-8")
    logger.info(f"Note created: {path}")
    return f"[happy] Got it, {_U}. I've saved a note: '{content[:60]}{'…' if len(content) > 60 else ''}'."


def _append(text: str) -> str:
    notes = _get_notes()
    if not notes:
        return _create(text)

    content = _extract_content(text, verbs=["add", "append", "also", "note"])
    if not content:
        return f"[surprised] What should I add to the note, {_U}?"

    latest = max(notes, key=lambda p: p.stat().st_mtime)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(latest, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n{content}\n")
    logger.info(f"Appended to note: {latest}")
    return f"[happy] Added to your latest note, {_U}."


def _read(text: str) -> str:
    notes = _get_notes()
    if not notes:
        return f"[sad] You have no saved notes, {_U}."

    target = _extract_note_name(text)
    if target:
        matches = [n for n in notes if target in n.stem.lower()]
        if matches:
            note = matches[0]
        else:
            return f"[sad] I couldn't find a note matching '{target}', {_U}."
    else:
        note = max(notes, key=lambda p: p.stat().st_mtime)

    raw = note.read_text(encoding="utf-8").strip()

    # Strip timestamp lines so they aren't spoken and don't collide with
    # expression tag parsing (e.g. [2026-06-15 12:00] looks like a tag)
    content = _TIMESTAMP_RE.sub("", raw).strip()

    if len(content) > 300:
        content = content[:300] + "…"
    return f"[relaxed] Here's your note: {content}"


def _list() -> str:
    notes = _get_notes()
    if not notes:
        return f"[sad] You have no saved notes yet, {_U}."
    names = [n.stem.replace("_", " ") for n in sorted(notes, key=lambda p: p.stat().st_mtime, reverse=True)[:5]]
    return f"[relaxed] Your recent notes: {', '.join(names)}."


def _delete(text: str) -> str:
    """Only asks — the note is removed by resolve_pending() on a spoken yes."""
    global _pending_delete
    notes = _get_notes()
    if not notes:
        return f"[neutral] You have no notes to delete, {_U}."

    target = _extract_note_name(text)
    if target:
        matches = [n for n in notes if target in n.stem.lower()]
        if not matches:
            return f"[sad] I couldn't find a note matching '{target}', {_U}."
        note = matches[0]
    else:
        note = max(notes, key=lambda p: p.stat().st_mtime)

    _pending_delete = (note, time.monotonic() + _CONFIRM_TTL_S)
    logger.info(f"Note delete awaiting confirmation: {note.name}")
    return (
        f"[surprised] Delete the note '{_spoken_name(note)}', {_U}? "
        "Say yes to confirm."
    )


def _run_delete(note: Path) -> str:
    """Blocking — called in an executor after a spoken yes."""
    label = _spoken_name(note)
    try:
        note.unlink()
    except FileNotFoundError:
        return f"[neutral] That note is already gone, {_U}."
    except Exception as e:
        logger.error(f"Note delete failed ({note.name}): {e}")
        return f"[sad] I couldn't delete that note, {_U}."
    logger.info(f"Note deleted: {note.name}")
    return f"[relaxed] Deleted note '{label}', {_U}."


def _open_in_editor(text: str) -> str:
    notes = _get_notes()
    if not notes:
        return f"[sad] You have no notes to open, {_U}."

    target = _extract_note_name(text)
    if target:
        matches = [n for n in notes if target in n.stem.lower()]
        note = matches[0] if matches else max(notes, key=lambda p: p.stat().st_mtime)
    else:
        note = max(notes, key=lambda p: p.stat().st_mtime)

    try:
        os.startfile(str(note))
    except Exception:
        subprocess.Popen(["notepad.exe", str(note)])

    return f"[happy] Opening your note in Notepad, {_U}."


def _get_notes() -> list[Path]:
    return list(_NOTES_DIR.glob("*.txt"))


def _extract_content(text: str, verbs: list[str]) -> str:
    t = text.strip()
    for verb in verbs:
        t = re.sub(rf'(?i)^.*?\b{verb}\b[:\s]*', '', t, count=1).strip()
    t = re.sub(r'(?i)\s*(please|thanks|okay|ok)\.?$', '', t).strip()
    return t


def _extract_note_name(text: str) -> str | None:
    m = re.search(r'(?:called|named|about|titled)\s+["\']?([a-zA-Z0-9 ]+?)["\']?(?:\s|$)', text, re.IGNORECASE)
    return m.group(1).strip().lower() if m else None


def _generate_filename(content: str) -> str:
    slug = re.sub(r'[^\w\s]', '', content[:30]).strip()
    slug = re.sub(r'\s+', '_', slug).lower()
    ts   = datetime.now().strftime("%m%d_%H%M")
    return f"{slug}_{ts}.txt" if slug else f"note_{ts}.txt"