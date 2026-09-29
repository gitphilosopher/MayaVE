"""
brain/router/context.py
Small, separate conversational context for the ROUTER only (last action,
pending clarification). It is NOT brain/conversation.py's ContextManager and
holds no dialogue — see docs/architecture.md §6.1 for that system. Resolution
works by REWRITING the utterance into a standalone command, so the
classifier/semantic/LLM stages stay stateless.

  "Set a timer" -> clarify(duration) ... "10 minutes"
        => rewritten "set a timer 10 minutes"
  "set a timer for 20 minutes" ... "actually make it 30"
        => rewritten "set a timer for 30 minutes"

PATCH (stabilization pass) — both rewrite paths were too permissive and
could contaminate an unrelated later turn:

- Pending-clarification reply: previously ANY text from which the missing
  entity's extractor could pull a value was accepted, so e.g. after "set a
  timer" a genuinely unrelated "I have 5 minutes to kill before my meeting"
  got rewritten into a timer command. The reply must now be (after
  stripping simple filler words like "in"/"about"/"please") ENTIRELY
  consumed by the extractor — i.e. a bare answer to the question, not a
  sentence that happens to contain a duration.
- Correction rewrite: previously any bare number in the text following the
  last timer action triggered a correction, so "no 5 apples please" would
  edit an active timer. A correction now requires an explicit correction
  marker (see _CORRECTION_RE) before a number/duration is accepted.
- Both paths remain one-shot (pending is consumed on the very next
  understand() call regardless of whether it resolves) and TTL-bound, so a
  stale pending/last-action state can't reach across an unrelated topic
  switch that took longer than the TTL.

Timer UPDATE functionality is intentionally not implemented: a correction
still produces a brand-new "set a timer for N minutes" command rather than
editing the original timer in place (skills/utilities/timer.py has no
"replace this timer" operation today). The IR carries reason=
"context_correction" so a future integration can special-case it once
timer.py grows an update path — see Router.dispatch()'s _clarify handling
for the same "wire the signal, don't invent the feature" pattern.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from brain.router.entities import EXTRACTORS, extract_duration, words_to_digits

PENDING_TTL_S = 30.0
CORRECTION_TTL_S = 600.0

# Filler words a bare clarification reply may carry alongside the answer
# itself ("in 10 minutes", "about 10 minutes", "10 minutes please") without
# counting as "an unrelated sentence that happens to mention a duration".
_FILLER_RE = re.compile(
    r"^(?:in|about|for|around|approximately)\s+|(?:\s+please|\s+thanks)$", re.I
)
_MAX_FILLER_WORDS = 2   # generous cap so a genuine sentence never passes


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


def _is_bare_reply(text: str) -> bool:
    """True only when `text`, after stripping simple filler, has few
    enough remaining words that it reads as a direct answer rather than
    an unrelated sentence that happens to contain the entity."""
    stripped = _FILLER_RE.sub("", text.strip()).strip()
    return 0 < len(stripped.split()) <= _MAX_FILLER_WORDS + 2   # entity itself may be 2-3 tokens ("ten minutes")


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
        if pending and now - pending.ts <= PENDING_TTL_S and _is_bare_reply(text):
            for name in pending.missing:
                fn = EXTRACTORS.get(name)
                if fn and fn(text, "") not in (None, ""):
                    return Resolved(f"{pending.text} {text}".strip(), "context")
            # Extractor found nothing usable in a short reply — fall through
            # to normal classification rather than guessing.

        last = ctx.last_action
        if last and last.key.startswith("timer.") and now - last.ts <= CORRECTION_TTL_S:
            m = _CORRECTION_RE.match(text.strip())
            if m:
                rest = m.group("rest")
                # PATCH: a marker can be immediately followed by a second
                # marker rather than the value itself — the canonical
                # example in this module's docstring, "actually make it
                # 30", matches only the "actually" alternative on the
                # first pass (alternation is tried in listed order and
                # the first successful match wins; Python's re does not
                # backtrack into trying a different top-level alternative
                # once one already produced an overall match), leaving
                # rest="make it 30" — which itself starts with the "make
                # it" alternative. Re-applying the same regex once more
                # strips that second marker too, so "actually make it 30"
                # and "make it 30" both resolve identically.
                m2 = _CORRECTION_RE.match(rest)
                if m2:
                    rest = m2.group("rest")
                rest = words_to_digits(rest)
                secs = extract_duration(rest)
                if secs is None:
                    n = re.match(r"^(\d+(?:\.\d+)?)\b", rest)
                    secs = int(float(n.group(1)) * last.duration_unit) if n else None
                if secs:
                    unit = next(u for u in (3600, 60, 1) if secs % u == 0)
                    return Resolved(f"set a timer for {_fmt(secs, unit)}", "context_correction")
        return Resolved(text)
