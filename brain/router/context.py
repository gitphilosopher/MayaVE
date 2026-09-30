"""
brain/router/context.py
Small, separate conversational context for the ROUTER only (last action,
pending clarification). NOT brain/conversation.py's ContextManager. Resolution
works by REWRITING the utterance into a standalone command, so the
classifier/semantic/LLM stages stay stateless.

  "Set a timer" -> clarify(duration) ... "10 minutes"
        => rewritten "set a timer 10 minutes"
  "set a timer for 20 minutes" ... "actually make it 30"
        => rewritten "set a timer for 30 minutes"

Scoping rules (BATCH 1 tightening):
- Pending reply: only a reply that is NOTHING BUT a duration (optionally
  "in/for/about" before and "please/thanks" after) resolves a pending
  `duration`. Previously any reply of <=4 words containing a duration
  ("I need 5 minutes") was swallowed. Pending `target`/`query` are never
  rewritten (their extractors only echo the classifier target, so an
  arbitrary reply must go through normal classification).
- Correction: needs a marker AND the remainder must be only a duration or a
  bare number (optionally "to"/"please"/"thanks"). Previously "no 5 apples
  please" and "actually I have 5 minutes of homework" both edited the timer.
  Only a last action of `timer.create` is correctable (not cancel/status).
- Both paths are TTL-bound; pending is consumed by the very next turn whether
  or not it resolves.

Timer correction still yields a NEW "set a timer for N ..." command (no
timer.update exists yet); reason/source "context_correction" marks it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from brain.router.entities import extract_duration, words_to_digits

PENDING_TTL_S = 30.0
CORRECTION_TTL_S = 600.0

_DUR = r"\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)"
_DURATION_ONLY_RE = re.compile(
    rf"^(?:(?:in|for|about|around|to)\s+)?{_DUR}(?:\s+(?:and\s+)?{_DUR})*(?:\s+(?:please|thanks))?$", re.I)
_BARE_NUMBER_RE = re.compile(r"^(?:to\s+)?(\d+(?:\.\d+)?)(?:\s+(?:please|thanks))?$", re.I)


def _norm(text: str) -> str:
    return words_to_digits(text.strip().lower()).strip(" .,!?")


def _is_duration_only(text: str) -> bool:
    return bool(_DURATION_ONLY_RE.match(_norm(text)))


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
    source: str = ""     # "" (unchanged) | "context" | "context_correction"


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
        if pending and now - pending.ts <= PENDING_TTL_S and "duration" in pending.missing:
            if _is_duration_only(text):
                return Resolved(f"{pending.text} {text}".strip(), "context")
            # anything else: fall through to normal classification untouched

        last = ctx.last_action
        if last and last.key == "timer.create" and now - last.ts <= CORRECTION_TTL_S:
            m = _CORRECTION_RE.match(text.strip())
            if m:
                rest = m.group("rest")
                # "actually make it 30": strip a second marker ("make it").
                m2 = _CORRECTION_RE.match(rest)
                if m2:
                    rest = m2.group("rest")
                rest = _norm(rest)
                secs = None
                if _is_duration_only(rest):
                    secs = extract_duration(rest)
                else:
                    n = _BARE_NUMBER_RE.match(rest)
                    if n:
                        secs = int(float(n.group(1)) * last.duration_unit)
                if secs:
                    unit = next(u for u in (3600, 60, 1) if secs % u == 0)
                    return Resolved(f"set a timer for {_fmt(secs, unit)}", "context_correction")
        return Resolved(text)