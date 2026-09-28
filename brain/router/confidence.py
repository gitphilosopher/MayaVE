"""
brain/router/confidence.py
Confidence policy for semantic retrieval results.

Thresholds are NOT hardcoded guesses — see brain/router/eval_router.py,
the offline evaluation script this migration requires before tuning
these in production (see docs/CONTRIBUTING.md's "Measure before
tuning" stage). The defaults below are conservative placeholders
pending that evaluation: they intentionally favor routing to the LLM
fallback over a confident-but-wrong direct dispatch, since a wrong
skill firing is worse than one extra local Ollama round trip.

Decision outcomes mirror the spec exactly:
  CONFIDENT — top-1 similarity and the top1-top2 margin are both
              strong enough to dispatch directly, no LLM call.
  AMBIGUOUS — top-1 is plausible but too close to top-2 to trust alone.
  LOW       — no candidate is similar enough to be worth trusting at all.

AMBIGUOUS and LOW are handled identically by the caller (both go to the
LLM fallback) but are kept distinct in the returned Decision for
observability — a router that is constantly LOW confidence signals a
corpus/seed problem, whereas constant AMBIGUOUS signals a threshold or
margin problem. See docs/CONTRIBUTING.md's `router source` logging.
"""

from dataclasses import dataclass
from enum import Enum

from brain.router.schemas import RetrievalResult


class Decision(Enum):
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    LOW = "low"


@dataclass(frozen=True)
class ConfidenceThresholds:
    """See config.router for the live values; these are the fallback
    defaults used when config doesn't override them."""
    min_similarity: float = 0.80     # top-1 must clear this to ever be CONFIDENT
    min_margin: float = 0.08         # top1 - top2 must clear this to be CONFIDENT
    low_similarity_floor: float = 0.55   # below this, don't even trust it as AMBIGUOUS evidence for the LLM prompt


def evaluate(result: RetrievalResult, thresholds: ConfidenceThresholds) -> Decision:
    if not result.candidates:
        return Decision.LOW
    if result.top1_similarity < thresholds.low_similarity_floor:
        return Decision.LOW
    if result.top1_similarity >= thresholds.min_similarity and result.margin >= thresholds.min_margin:
        return Decision.CONFIDENT
    return Decision.AMBIGUOUS