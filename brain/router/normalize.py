"""
brain/router/normalize.py
Light, non-destructive text normalization before embedding/guard checks.

Deliberately conservative: this feeds both the embedding model (which
tolerates casing/punctuation fine) and, later, the LLM fallback prompt
(where the raw utterance should stay recognizable). It must never
rewrite content in a way that would change what a skill later extracts
from intent["target"] or intent["raw"] — callers keep the original
`text` alongside the normalized form for exactly that reason.
"""

import re

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[?.!]+$")


def normalize(text: str) -> str:
    """Collapse whitespace and strip trailing sentence punctuation; case is preserved."""
    t = text.strip()
    t = _WHITESPACE_RE.sub(" ", t)
    t = _TRAILING_PUNCT_RE.sub("", t).strip()
    return t