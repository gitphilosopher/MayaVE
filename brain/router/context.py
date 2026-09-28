"""
brain/router/context.py
Small, separate conversational context for the ROUTER only (last action,
pending clarification). It is not ContextManager (brain/conversation.py) and
holds no dialogue. Resolution works by REWRITING the utterance into a
standalone command, so the classifier/semantic/LLM stages stay stateless.

  "Set a timer" -> clarify(duration) ... "10 minutes"
        => rewritten "set a timer 10 minutes"
  "set a timer for 20 minutes" ... "actually make it 30"
        => rewritten "set a timer for 30 minutes"
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from brain.router.entities import EXTRACTORS, extract_duration, words_to_digits

PENDING_TTL_S = 30.0
CORRECTION_TTL_S = 600.0


@dataclass
class ActionRecord:
    key: str
    entities: dict
    text: str
    ts: float
    duration_unit: int = 60


@dataclass
class Pending:
    text: str
    missing: tuple
    ts: float


@dataclass
class Resolved:
    text: str
    source: str = ""     # "" (unchanged) | "context"


class ConversationContext:
    def __init__(self):
        self.last_action: ActionRecord | None = None
        self.pending: Pending | None = None

    def observe(self, ir) -> None:
        from brain.router.ir import Status
        if ir.status is Status.READY and ir.key:
            unit = 60
            secs = ir.entities.get("duration")
            if isinstance(secs, int) and secs:
                unit = next(u for u in (3600, 60, 1) if secs % u == 0)
            self.last_action = ActionRecord(ir.key, dict(ir.entities), ir.effective_text, time.monotonic(), unit)
        if ir.status is Status.NEEDS_CLARIFICATION:
            self.pending = Pending(ir.effective_text or ir.raw_text, ir.missing_entities, time.monotonic())


_CORRECTION_RE = re.compile(
    r"^(?:actually|no|wait|make it|change (?:it|that) to|set (?:it|that) to|"
    r"(?:let'?s )?(?:say|do|go with))\b[,\s]*(?P<rest>.+)$", re.I)
_FMT = {3600: "hour", 60: "minute", 1: "second"}


def _fmt(seconds: int, unit: int) -> str:
    n = seconds // unit
    return f"{n} {_FMT[unit]}{'s' if n != 1 else ''}"


class ContextResolver:
    def resolve(self, text: str, ctx: ConversationContext) -> Resolved:
        now = time.monotonic()
        pending, ctx.pending = ctx.pending, None          # one-shot, like power/notepad
        if pending and now - pending.ts <= PENDING_TTL_S:
            for name in pending.missing:
                fn = EXTRACTORS.get(name)
                if fn and fn(text, "") not in (None, ""):
                    return Resolved(f"{pending.text} {text}".strip(), "context")

        last = ctx.last_action
        if last and last.key.startswith("timer.") and now - last.ts <= CORRECTION_TTL_S:
            m = _CORRECTION_RE.match(text.strip())
            if m:
                rest = words_to_digits(m.group("rest"))
                secs = extract_duration(rest)
                if secs is None:
                    n = re.match(r"^(\d+(?:\.\d+)?)\b", rest)
                    secs = int(float(n.group(1)) * last.duration_unit) if n else None
                if secs:
                    unit = next(u for u in (3600, 60, 1) if secs % u == 0)
                    return Resolved(f"set a timer for {_fmt(secs, unit)}", "context")
        return Resolved(text)
