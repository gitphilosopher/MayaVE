"""
brain/router/understand.py
CommandUnderstander — progressive-cost understanding that produces a CommandIR.
It never executes anything and never imports a skill.

  L0  guards (existing IntentEngine guards, reused via guards.check)
  L1  CNN/BiLSTM ensemble via the existing IntentEngine, gated on
      confidence AND margin; semantic retrieval only when L1 is not trusted
  L2  deterministic entity extraction (entities.py) -> ready / needs_clarification
  L3  local Ollama fallback (llm_fallback.route) -> validated before use

A low-confidence COMMAND is never forced into an intent: it ends as UNKNOWN.
Low-confidence *conversational* labels are harmless and go to the chat LLM.
"""
from __future__ import annotations

import asyncio
import logging

from brain.router import validate
from brain.router.context import ContextResolver, ConversationContext
from brain.router.entities import extract_entities
from brain.router.ir import CommandIR, Status

logger = logging.getLogger(__name__)

# Placeholders pending eval_ir.py measurements on real data (same policy as confidence.py).
MIN_CONF, MIN_MARGIN, MIN_CONF_NO_MARGIN = 0.85, 0.25, 0.93


class CommandUnderstander:
    def __init__(self, legacy, registry, *, ctx: ConversationContext | None = None,
                 guard_fn=None, semantic=None, semantic_thresholds=None, llm_route=None,
                 min_conf=MIN_CONF, min_margin=MIN_MARGIN):
        self._legacy, self._registry = legacy, registry
        self._by_legacy = {s.legacy_intent: s for s in registry.specs}
        self._guard = guard_fn
        self._semantic, self._sem_thr, self._llm = semantic, semantic_thresholds, llm_route
        self._min_conf, self._min_margin = min_conf, min_margin
        self.ctx = ctx or ConversationContext()
        self._resolver = ContextResolver()

    # ── public ────────────────────────────────────────────────────────────
    async def understand(self, text: str) -> CommandIR:
        raw = text.strip()
        resolved = self._resolver.resolve(raw, self.ctx)
        ir = await self._route(resolved.text, raw, resolved.source)
        self.ctx.observe(ir)
        return ir

    # ── routing ───────────────────────────────────────────────────────────
    async def _route(self, text: str, raw: str, tag: str) -> CommandIR:
        base = dict(raw_text=raw, effective_text=text)
        if self._guard:
            hit = self._guard(self._legacy, text)
            if hit:
                intent, conf, src = hit
                return self._from_intent_name(intent, conf, 1.0, src, "", **base)

        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self._legacy.classify, text)
        intent, conf, margin = res["intent"], float(res["confidence"]), res.get("margin")
        src = f"context+{res['model']}" if tag else res["model"]
        mode, target = res.get("response_mode", "llm"), res.get("target", "")
        trusted = self._trusted(res["model"], conf, margin)

        if trusted:
            return self._from_intent_name(intent, conf, margin, src, target, **base)

        # not trusted -> escalate, unless it is a plausible conversational label
        if not (mode == "llm" and conf >= 0.5):
            ir = await self._escalate(text, target, **base)
            if ir is not None:
                return ir
        if mode == "llm":
            return CommandIR(Status.UNKNOWN, confidence=conf, margin=margin, source=src,
                             reason="conversational", legacy_intent=intent, **base)
        return CommandIR(Status.UNKNOWN, confidence=conf, margin=margin, source=src,
                         reason="low_confidence", legacy_intent="unknown", **base)

    def _trusted(self, model: str, conf: float, margin) -> bool:
        if "keyword" in model or "guard" in model:
            return True                                   # deterministic evidence
        if margin is None:
            return conf >= MIN_CONF_NO_MARGIN             # can't measure margin -> be stricter
        return conf >= self._min_conf and margin >= self._min_margin

    def _from_intent_name(self, intent, conf, margin, src, target, **base) -> CommandIR:
        spec = self._by_legacy.get(intent)
        if spec:
            return self._from_spec(spec, conf, margin, src, target, **base)
        mode = getattr(self._legacy, "_response_modes", {}).get(intent, "llm")
        if mode == "skill":   # canned/guarded skills outside the registry (greet, perform_action, ...)
            return CommandIR(Status.READY, domain="legacy", operation=intent, confidence=conf,
                             margin=margin, source=src, legacy_intent=intent, target=target, **base)
        return CommandIR(Status.UNKNOWN, confidence=conf, margin=margin, source=src,
                         reason="conversational", legacy_intent=intent, **base)

    def _from_spec(self, spec, conf, margin, src, target, *, llm_entities=None, **base) -> CommandIR:
        text = base["effective_text"]
        ents, missing = extract_entities(spec, text, target, llm_entities)
        ent_target = ents.get(spec.target_mode.split(":", 1)[1]) if spec.target_mode.startswith("entity:") else None
        common = dict(domain=spec.domain, operation=spec.operation, entities=ents, confidence=conf,
                      margin=margin, source=src, legacy_intent=spec.legacy_intent,
                      requires_confirmation=getattr(spec, "requires_confirmation", False),
                      target=str(ent_target) if ent_target else text, **base)
        if missing:
            q = spec.entities[missing[0]].get("prompt") or f"What {missing[0].replace('_', ' ')} did you have in mind"
            return CommandIR(Status.NEEDS_CLARIFICATION, missing_entities=missing,
                             prompt=q + "?", reason="missing_required_entity", **common)
        return CommandIR(Status.READY, **common)

    async def _escalate(self, text, target, **base) -> CommandIR | None:
        if self._semantic:
            from brain.router.confidence import Decision, evaluate
            r = await self._semantic(text)
            if evaluate(r, self._sem_thr) is Decision.CONFIDENT:
                return self._from_spec(r.top1.spec, r.top1_similarity, r.margin, "semantic", target, **base)
        if self._llm:
            return await self._from_llm(text, target, **base)
        return None

    async def _from_llm(self, text, target, **base) -> CommandIR | None:
        reg = self._registry
        raw = await self._llm(text, reg.domains(), {d: reg.operations_for(d) for d in reg.domains()}, [])
        if raw is None:
            return None                                    # LLM down -> caller ends UNKNOWN
        parsed = validate.validate_raw_llm_output(raw)
        if parsed is None:
            return CommandIR(Status.REJECTED, source="llm", reason="malformed_llm_output", **base)
        v = validate.validate(parsed, reg)
        if v.ok:
            return self._from_spec(v.spec, min(parsed.confidence, 0.7), None, "llm", target,
                                   llm_entities=parsed.entities, **base)
        if v.error == "not_a_command":
            return CommandIR(Status.UNKNOWN, source="llm", reason="conversational", legacy_intent="general_query", **base)
        spec = reg.get(parsed.domain, parsed.operation)
        if v.error == "needs_clarification" and spec:
            return self._from_spec(spec, min(parsed.confidence, 0.3), None, "llm", target, **base)
        return CommandIR(Status.REJECTED, source="llm", reason=v.error or "invalid", **base)
