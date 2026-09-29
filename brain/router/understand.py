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

PATCH (stabilization pass):
- `res.get("margin")` was always None — IntentEngine.classify() never
  returned that key (verified against brain/intent_engine.py's real
  return dict: intent/target/confidence/raw/model/response_mode). This
  silently forced every ML-sourced result through the stricter
  MIN_CONF_NO_MARGIN branch. IntentEngine.classify() has been extended
  (additively — see that module) to also return second_intent/
  second_confidence/margin, computed from the SAME averaged ensemble
  probabilities `_predict` already produces internally; nothing about the
  PyTorch/TensorFlow models themselves changed.
- An LLM result with needs_clarification=True on a spec with NO required
  entities previously still became READY at confidence 0.3 (validate.py's
  ValidationResult(ok=False, error="needs_clarification") was read as
  "ok enough to dispatch at low confidence"). It is now UNKNOWN —
  clarification without any entity to ask for is not actionable.
- A validated LLM Command is no longer trusted regardless of its own
  confidence: LLM_MIN_CONF gates it (placeholder, like every other
  threshold here — see confidence.py's docstring on "measure before
  tuning"). Below the gate the turn falls through to UNKNOWN, matching
  the same "an uncertain result must not silently become an executable
  command" rule applied to the classifier and to semantic retrieval.
- `semantic_thresholds=None` (the default when no `semantic` callable is
  wired) previously crashed `evaluate()` on first use inside `_escalate`.
  It now defaults to ConfidenceThresholds() so a caller that wires
  `semantic=` without also wiring `semantic_thresholds=` doesn't crash.
- `self.stats` counts guard/classifier/semantic/llm hits and semantic/LLM
  exceptions, read by eval_ir.py for the real LLM-fallback-rate metric
  (source-string sniffing undercounts an "LLM was consulted but returned
  None/invalid" turn).
- semantic()/llm() calls are now wrapped so an unexpected exception from
  either degrades to UNKNOWN instead of propagating out of understand().
"""
from __future__ import annotations

import asyncio
import logging

from brain.router import validate
from brain.router.confidence import ConfidenceThresholds
from brain.router.context import ContextResolver, ConversationContext
from brain.router.entities import extract_entities
from brain.router.ir import CommandIR, Status

logger = logging.getLogger(__name__)

# Placeholders pending eval_ir.py measurements on real data (same policy as confidence.py).
MIN_CONF, MIN_MARGIN, MIN_CONF_NO_MARGIN = 0.85, 0.25, 0.93
LLM_MIN_CONF = 0.5   # below this, a validated-but-weak LLM command is not trusted either


class CommandUnderstander:
    def __init__(self, legacy, registry, *, ctx: ConversationContext | None = None,
                 guard_fn=None, semantic=None, semantic_thresholds=None, llm_route=None,
                 min_conf=MIN_CONF, min_margin=MIN_MARGIN):
        self._legacy, self._registry = legacy, registry
        self._by_legacy = {s.legacy_intent: s for s in registry.specs}
        self._guard = guard_fn
        self._semantic, self._llm = semantic, llm_route
        self._sem_thr = semantic_thresholds or ConfidenceThresholds()
        self._min_conf, self._min_margin = min_conf, min_margin
        self.ctx = ctx or ConversationContext()
        self._resolver = ContextResolver()
        # Real counters for eval_ir.py's LLM-fallback-rate metric — a
        # source-string sniff can't distinguish "LLM was asked and said no"
        # from "LLM was never reached".
        self.stats = {"guard": 0, "classifier": 0, "semantic": 0, "llm": 0,
                      "semantic_errors": 0, "llm_errors": 0}

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
                self.stats["guard"] += 1
                intent, conf, src = hit
                return self._from_intent_name(intent, conf, None, src, "", **base)

        self.stats["classifier"] += 1
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, self._legacy.classify, text)
        intent, conf = res["intent"], float(res["confidence"])
        margin = res.get("margin")   # None for keyword/guard-sourced results — see intent_engine.py
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
                      requires_confirmation=spec.requires_confirmation,
                      target=str(ent_target) if ent_target else text, **base)
        if missing:
            q = spec.entities[missing[0]].get("prompt") or f"What {missing[0].replace('_', ' ')} did you have in mind"
            return CommandIR(Status.NEEDS_CLARIFICATION, missing_entities=missing,
                             prompt=q + "?", reason="missing_required_entity", **common)
        return CommandIR(Status.READY, **common)

    async def _escalate(self, text, target, **base) -> CommandIR | None:
        if self._semantic:
            from brain.router.confidence import Decision, evaluate
            try:
                r = await self._semantic(text)
                self.stats["semantic"] += 1
            except Exception as e:
                self.stats["semantic_errors"] += 1
                logger.warning(f"Semantic retrieval failed (non-fatal, falling through): {e}")
                r = None
            if r is not None and evaluate(r, self._sem_thr) is Decision.CONFIDENT:
                return self._from_spec(r.top1.spec, r.top1_similarity, r.margin, "semantic", target, **base)
        if self._llm:
            return await self._from_llm(text, target, **base)
        return None

    async def _from_llm(self, text, target, **base) -> CommandIR | None:
        reg = self._registry
        try:
            raw = await self._llm(text, reg.domains(), {d: reg.operations_for(d) for d in reg.domains()}, [])
            self.stats["llm"] += 1
        except Exception as e:
            self.stats["llm_errors"] += 1
            logger.warning(f"LLM fallback call failed (non-fatal): {e}")
            return None
        if raw is None:
            return None                                    # LLM down -> caller ends UNKNOWN
        parsed = validate.validate_raw_llm_output(raw)
        if parsed is None:
            return CommandIR(Status.REJECTED, source="llm", reason="malformed_llm_output", **base)
        v = validate.validate(parsed, reg)
        if v.ok:
            eff_conf = min(parsed.confidence, 0.7)
            if eff_conf < LLM_MIN_CONF:
                # Validated shape, but the model itself wasn't confident enough
                # to trust as an executable command — never force it.
                return CommandIR(Status.UNKNOWN, source="llm", confidence=eff_conf,
                                 reason="llm_low_confidence", legacy_intent=v.spec.legacy_intent, **base)
            return self._from_spec(v.spec, eff_conf, None, "llm", target,
                                   llm_entities=parsed.entities, **base)
        if v.error == "not_a_command":
            return CommandIR(Status.UNKNOWN, source="llm", reason="conversational", legacy_intent="general_query", **base)
        spec = reg.get(parsed.domain, parsed.operation)
        if v.error == "needs_clarification" and spec:
            ents, missing = extract_entities(spec, base["effective_text"], target, parsed.entities)
            if not missing:
                # The LLM asked for clarification but every required entity
                # is already present/derivable — nothing left to clarify,
                # and forcing READY here would be trusting an uncertain
                # signal as if it were confirmed. Treat as UNKNOWN rather
                # than silently executing OR silently asking a pointless
                # question.
                return CommandIR(Status.UNKNOWN, source="llm", confidence=min(parsed.confidence, 0.3),
                                 reason="llm_clarification_but_nothing_missing",
                                 legacy_intent=spec.legacy_intent, **base)
            q = spec.entities[missing[0]].get("prompt") or f"What {missing[0].replace('_', ' ')} did you have in mind"
            return CommandIR(Status.NEEDS_CLARIFICATION, domain=spec.domain, operation=spec.operation,
                             entities=ents, confidence=min(parsed.confidence, 0.3), source="llm",
                             legacy_intent=spec.legacy_intent, missing_entities=missing,
                             prompt=q + "?", reason="llm_needs_clarification",
                             requires_confirmation=spec.requires_confirmation, **base)
        return CommandIR(Status.REJECTED, source="llm", reason=v.error or "invalid", **base)
