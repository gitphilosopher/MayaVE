"""
skills/utilities/notepad.py
Voice-driven note-taking and retrieval utility.

This skill manages Maya's local note files in a user-facing, speech-driven way.
It supports creating, appending, viewing, listing, opening, and deleting notes,
with the intent layer merging older create/append and read/list/open variants into
`note_write` and `note_view` while keeping `note_delete` as a dedicated flow.

The module intentionally handles the ambiguous sub-operations in the skill itself
instead of depending on the classifier to distinguish them. A plain write request
can become create-vs-append based on wording; a plain view request can become
list-vs-open-vs-read based on wording. The safety behavior is also important:
deletion is confirmation-gated and one-shot, so a stale follow-up confirmation
cannot delete an old note.

Timestamp lines are stored alongside note text but stripped before returning file
content to the speaker, because bracketed timestamps would otherwise look like
expression tags and be interpreted by the speech pipeline as metadata rather than
note content.
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

# ── Sub-action cues for the merged intents (see module docstring) ─────────
# 'note_write': append vs. create. _append() itself falls back to _create()
# when no notes exist yet, so a stray append cue on the very first note
# still works — this only needs to catch the append phrasing itself.
# The windowed "add ... note(s)" form catches "add this to my note" (a
# real training phrasing) without a bare "add" elsewhere in a longer
# sentence ("add butter to my grocery list") false-matching — "note"/
# "notes" has to actually show up within a few words of "add".
_APPEND_CUE_RE = re.compile(
    r"\bappend\b"
    r"|\badd\b(?:\s+\S+){0,4}\s+notes?\b"
    r"|\balso\s+(?:note|add)\b",
    re.IGNORECASE,
)

# 'note_view': list vs. open-in-editor vs. read-aloud (default). Checked in
# this order — list/open cues are specific multi-word phrasings; anything
# else (including a bare "read my notes", which contains "my notes" but
# isn't a list request) falls through to reading the latest note aloud.
# Both are windowed the same way as _APPEND_CUE_RE above so an unrelated
# "list" or "open" elsewhere in the sentence ("read my shopping list",
# "open the door and note that down") doesn't false-match — the cue word
# has to actually be near "note(s)".
_LIST_CUE_RE = re.compile(
    r"\blist\b(?:\s+\S+){0,2}\s+notes?\b"
    r"|\ball (?:my )?notes\b"
    r"|\bwhat notes\b",
    re.IGNORECASE,
)
_OPEN_CUE_RE = re.compile(
    r"\b(?:open|launch)\b(?:\s+\S+){0,2}\s+notes?\b"
    r"|\bin (?:notepad|the editor|an editor)\b",
    re.IGNORECASE,
)


def _spoken_name(note: Path) -> str:
    """Turn a filename back into a human-readable note title for confirmations and replies."""
    return _STAMP_SUFFIX_RE.sub("", note.stem).replace("_", " ") or note.stem.replace("_", " ")


def _ensure_dir() -> None:
    """Create the notes directory if it does not already exist."""
    _NOTES_DIR.mkdir(parents=True, exist_ok=True)


async def execute(intent: dict, text: str) -> str:
    """Dispatch the note operation to the correct blocking helper for the given intent and utterance."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _handle, intent, text)


async def resolve_pending(text: str) -> str | None:
    """Consume the next utterance as confirmation for a pending delete request when applicable."""
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
    """Route a note request to the correct create/view/delete helper based on intent or text cues."""
    _ensure_dir()
    t = text.lower()
    name = intent.get("intent", "")

    handlers = {
        "note_write":  _write,
        "note_view":   _view,
        "note_delete": _delete,
    }
    if name in handlers:
        return handlers[name](text)

    if any(w in t for w in ("delete note", "remove note")):
        return _delete(text)
    if _APPEND_CUE_RE.search(t):
        return _append(text)
    if _LIST_CUE_RE.search(t):
        return _list()
    if _OPEN_CUE_RE.search(t):
        return _open_in_editor(text)
    if any(w in t for w in ("read", "show", "what's in", "whats in")):
        return _read(text)

    return _create(text)


def _write(text: str) -> str:
    """Create a new note or append to the latest note based on the wording of the request."""
    if _APPEND_CUE_RE.search(text.lower()):
        return _append(text)
    return _create(text)


def _view(text: str) -> str:
    """List notes, open a note in an editor, or read the most relevant note aloud based on the wording."""
    t = text.lower()
    if _LIST_CUE_RE.search(t):
        return _list()
    if _OPEN_CUE_RE.search(t):
        return _open_in_editor(text)
    return _read(text)


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
    """Ask for confirmation before deleting the selected note; the actual removal happens on a later yes."""
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
    """Delete the selected note after it has been confirmed by a later utterance."""
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