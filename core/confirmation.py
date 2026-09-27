"""
core/confirmation.py
Shared yes/no phrase matcher for one-shot confirmations.

This module provides a tiny, stateless confirmation parser for skills that want
simple spoken confirmation flows without duplicating phrase logic. The same
matching rules are intentionally shared by features such as shutdown/restart
confirmation and notepad deletion confirmation so callers can rely on a single
consistent vocabulary.

The matcher operates on normalized whole-utterance text and accepts only exact
confirmation or denial patterns, not arbitrary text containing a yes/no word.
For example, `"yes the report is done"` is not accepted as confirmation because
it is not a bare confirmation phrase.

Usage:
- call `normalize()` before custom matching when you want a stable text form
- use `is_confirm()` and `is_deny()` to decide whether a spoken response should
  proceed with or cancel the pending action
"""

import re

from config.settings import config

_ADDRESS = rf"(?:\s+(?:{re.escape(config.name.lower())}|{re.escape(config.user_name.lower())}))?"

_CONFIRM_RE = re.compile(
    rf"^(?:yes|yeah|yep|yup|confirm|confirmed|affirmative|do it|go ahead|proceed)"
    rf"(?:\s+please)?{_ADDRESS}$"
)
_DENY_RE = re.compile(
    rf"^(?:no|nope|nah|cancel|abort|never ?mind|don'?t)"
    rf"(?:\s+(?:please|thanks|thank you))?{_ADDRESS}$"
)


def normalize(text: str) -> str:
    """Lowercase and normalize a spoken phrase into a single canonical matching form."""
    return " ".join(re.sub(r"[^a-z' ]", "", text.lower()).split())


def is_confirm(text: str) -> bool:
    """Return True when the utterance is a bare confirmation phrase."""
    return _CONFIRM_RE.match(normalize(text)) is not None


def is_deny(text: str) -> bool:
    """Return True when the utterance is a bare denial or cancellation phrase."""
    return _DENY_RE.match(normalize(text)) is not None
