"""
brain/router/ir.py
Command IR — the stable boundary between understanding and execution.

    natural language -> CommandIR -> validation -> Skill

The router only ever *produces* a CommandIR; it never runs a skill. Every
outcome is explicit: ready / needs_clarification / unknown / rejected.
`to_legacy_intent()` is the compatibility bridge to the existing
Router.dispatch(intent, text) contract, so no skill has to change.

BATCH 1 FIX: a non-READY IR could carry a *skill* legacy_intent (the LLM
fallback set legacy_intent=spec.legacy_intent on its low-confidence /
clarification-with-nothing-missing UNKNOWN results). to_legacy_intent()
copied that into intent["intent"], and Router.dispatch routes by intent
name — so an UNKNOWN IR would have executed the skill anyway. Non-READY
IRs now only keep their legacy_intent when reason == "conversational"
(an LLM-mode intent such as smalltalk/general_query); otherwise "unknown".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNKNOWN = "unknown"      # out of scope, conversational, or too uncertain to act on
    REJECTED = "rejected"    # a candidate (e.g. LLM output) failed validation


@dataclass(frozen=True)
class CommandIR:
    status: Status
    domain: str | None = None
    operation: str | None = None
    entities: dict = field(default_factory=dict)
    confidence: float = 0.0
    margin: float | None = None          # None = the source could not measure one
    source: str = "none"                 # guard | classifier | semantic | llm | context
    missing_entities: tuple = ()
    requires_confirmation: bool = False
    reason: str = ""                     # why unknown/rejected/clarifying (observability)
    prompt: str = ""                     # clarification question, when needed
    raw_text: str = ""
    effective_text: str = ""             # text after context rewriting; what skills re-parse
    legacy_intent: str = ""              # compat: id the existing skills/router know
    target: str = ""                     # compat: legacy intent["target"]

    @property
    def key(self) -> str | None:
        return f"{self.domain}.{self.operation}" if self.domain and self.operation else None

    @property
    def requires_clarification(self) -> bool:
        return self.status is Status.NEEDS_CLARIFICATION

    @property
    def executable(self) -> bool:
        """The single gate for running a skill: READY, nothing missing, and a
        legacy_intent to route by. UNKNOWN / REJECTED / NEEDS_CLARIFICATION and
        a READY IR that still lacks required entities are all NOT executable."""
        return self.status is Status.READY and not self.missing_entities and bool(self.legacy_intent)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value, "domain": self.domain, "operation": self.operation,
            "entities": dict(self.entities), "confidence": round(self.confidence, 3),
            "margin": None if self.margin is None else round(self.margin, 3),
            "source": self.source, "requires_clarification": self.requires_clarification,
            "missing_entities": list(self.missing_entities),
            "requires_confirmation": self.requires_confirmation,
        }


def to_legacy_intent(ir: CommandIR) -> dict:
    """Bridge to the dict shape Router.dispatch and every skill already expect."""
    if ir.status is Status.NEEDS_CLARIFICATION:
        intent, mode = "clarify", "skill"
    elif ir.status is Status.READY:
        intent, mode = ir.legacy_intent, "skill"
    else:
        # UNKNOWN / REJECTED never become an action. Only a conversational
        # (LLM-mode) label may keep its name; anything else collapses to
        # "unknown" so dispatch can never route it to a skill.
        keep = ir.status is Status.UNKNOWN and ir.reason == "conversational" and ir.legacy_intent
        intent, mode = (ir.legacy_intent if keep else "unknown"), "llm"
    return {
        "intent": intent, "target": ir.target, "confidence": round(ir.confidence, 3),
        "raw": ir.effective_text or ir.raw_text, "model": ir.source, "response_mode": mode,
        "_ir": ir,
    }