"""
core/confirmation.py
Shared spoken yes/no matching for one-shot confirmations — used by
skills/system/power.py (shutdown/restart) and skills/utilities/notepad.py
(note delete) so both accept exactly the same phrases.

Pure text matching, no state: each skill owns its own pending request,
TTL and one-shot handling (see their resolve_pending(), called first by
Router.dispatch). Phrases are matched against the WHOLE normalized
utterance, so "yes the report is done" is not a confirmation.
"""

import re

from config.settings import config

# Optional trailing address: "yes maya", "no senpai".
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
    return " ".join(re.sub(r"[^a-z' ]", "", text.lower()).split())


def is_confirm(text: str) -> bool:
    return _CONFIRM_RE.match(normalize(text)) is not None


def is_deny(text: str) -> bool:
    return _DENY_RE.match(normalize(text)) is not None
