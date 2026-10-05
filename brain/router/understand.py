"""
brain/router/understand.py
CommandUnderstander — progressive-cost understanding that produces a CommandIR.
It never executes anything and never imports a skill.

  Context -> L0 guards -> L1 CNN/BiLSTM (confidence + margin gate)
          -> semantic retrieval (normalized text) -> LLM fallback
          -> entity extraction -> CommandIR -> validation

A low-confidence COMMAND is never forced into an intent: it ends as UNKNOWN.
Low-confidence *conversational* labels are harmless and go to the chat LLM.

Stabilization + BATCH 1 fixes:
- classifier margin now real (IntentEngine.classify returns it).
- LLM: needs_clarification with nothing missing -> UNKNOWN; validated command
  below LLM_MIN_CONF -> UNKNOWN. BATCH 1: both UNKNOWNs now carry
  legacy_intent="unknown" (they carried the SPEC's skill intent, which
  Router.dispatch would have executed by name; ir.to_legacy_intent also
  enforces this).
- BATCH 1: semantic retrieval now receives normalize(text) (previously raw);
  the LLM still receives the user's own words.
- BATCH 1: understand() cannot raise because a stage failed — an unexpected
  exception from the guard/classifier/anything downstream degrades to an
  UNKNOWN IR (reason "internal_error") and is logged with traceback.
- semantic_thresholds=None defaults to ConfidenceThresholds().
- self.stats counts stage hits/errors for eval_ir.py.

Routing-gate fix: an untrusted but plausible conversational classifier label
(response_mode "llm", conf >= 0.5) no longer skips escalation outright. It now
gets one round of semantic retrieval, using the existing semantic thresholds,
so a confident command match can compete with it before the conversational
result is accepted. The LLM fallback is NOT consulted for that case (it is
still used exactly as before for every other escalating path).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from brain.router import validate
from brain.router.confidence import ConfidenceThresholds
from brain.router.context import ContextResolver, ConversationContext
from brain.router.entities import extract_entities
from brain.router.ir import CommandIR, Status
from brain.router.normalize import normalize

logger = logging.getLogger(__name__)

# Placeholders pending eval_ir.py measurements on real data (same policy as confidence.py).
MIN_CONF, MIN_MARGIN, MIN_CONF_NO_MARGIN = 0.85, 0.25, 0.93
LLM_MIN_CONF = 0.5   # below this, a validated-but-weak LLM command is not trusted either

# System operations (shutdown, lock, restart) apply strictly to the host computer / PC environment.
# Physical devices, fixtures, appliances, lights, TVs, doors, and premises are out of scope.
_PHYSICAL_SYSTEM_TARGETS = re.compile(
    r"\b(?:"
    r"lights?|lamps?|porch|living room|bedroom|kitchen|"
    r"tv|tvs|televisions?|"
    r"appliances?|oven|stove|fans?|heaters?|thermostats?|"
    r"doors?|gates?|shops?|house|premises"
    r")\b",
    re.I,
)

# Unsupported capabilities that must never map to an external command (e.g. search history)
_UNSUPPORTED_SEARCH_HISTORY_RE = re.compile(
    r"\b(?:"
    r"(?:search|browsing|browser|google)\s+history|"
    r"history\s+of\s+(?:my\s+)?(?:searches|browsing)|"
    r"(?:what|things?)\s+(?:did\s+)?(?:i|we)\s+search(?:\s+for)?|"
    r"(?:what|things?)\s+i\s+(?:have\s+)?searched(?:\s+for)?|"
    r"(?:past|previous|recent)\s+searches|"
    r"look\s+up\s+what\s+(?:i|we)\s+searched(?:\s+for)?|"
    r"searched\s+(?:for\s+)?(?:yesterday|earlier|before)|"
    r"remember\s+what\s+(?:i|we)\s+searched|"
    r"remember\s+when\s+(?:i|we)\s+(?:asked\s+you\s+to\s+)?search"
    r")\b",
    re.I,
)

# Hypothetical conditionals: asking "what if" or consequence of hypothetical action
_HYPOTHETICAL_RE = re.compile(
    r"^(?:(?:and|so|but|well)\s+)*"
    r"(?:"
    r"what\s+if\s+(?:i|we|you)\b|"
    r"what\s+would\s+happen\s+if\b|"
    r"if\s+(?:i|we|you)\b.+?\b(?:what\s+happens|what\s+will\s+happen|what\s+would\s+happen|will\s+it|would\s+it)\b|"
    r"suppose\s+(?:i|we)\b"
    r")",
    re.I,
)

# Conversational inquiries / memory questions about past actions
_PAST_INQUIRY_RE = re.compile(
    r"^(?:(?:and|so|but|well)\s+)*"
    r"(?:"
    r"did\s+(?:i|we|you)\b|"
    r"how\s+long\s+was\b|"
    r"do\s+you\s+remember\b|"
    r"remember\s+when\b|"
    r"what\s+did\s+(?:i|we)\b|"
    r".*?\b(?:we\s+talked\s+about|thing\s+we\s+talked\s+about)\b"
    r")",
    re.I,
)

# Declarative past / historical actions (user recounting what they or someone already did)
_HISTORICAL_STATEMENT_RE = re.compile(
    r"^(?:(?:and|so|but|well|actually|yeah|yes|no)\s+)*"
    r"(?:"
    r"(?:i|we|he|she|they|someone)\s+(?:have\s+|had\s+)?already\b|"
    r"(?:i|we|he|she|they|someone)\s+(?:already\s+)?(?:had|was|were)\b|"
    r"(?:i|we|he|she|they|someone)\s+(?:"
    r"set|started|created|turned(?:\s+off)?|shut(?:\s+down)?|restarted|locked|searched|checked|"
    r"opened|cleared|copied|deleted|listened|played|paused|asked|looked"
    r")\b.*?\b(?:earlier|yesterday|ago|before|previously|last\s+(?:night|week|month)|already)\b|"
    r"(?:earlier|yesterday|previously)\s+(?:i|we|he|she|they|someone)\b|"
    r"(?:i|we)\s+(?:"
    r"set\s+a\s+timer|"
    r"turned\s+off|"
    r"shut\s+down|"
    r"restarted|"
    r"locked|"
    r"searched(?:\s+google)?\s+for|"
    r"checked(?:\s+the\s+weather)?|"
    r"was\s+listening"
    r")\b"
    r")",
    re.I,
)


def evaluate_actionability(domain: str, operation: str, text: str) -> tuple[bool, str]:
    t = text.lower().strip()
    if domain == "web" or "search" in operation:
        if _UNSUPPORTED_SEARCH_HISTORY_RE.search(t):
            return False, "unsupported_capability"
    if _HYPOTHETICAL_RE.search(t):
        return False, "hypothetical"
    if _PAST_INQUIRY_RE.search(t):
        return False, "conversational_inquiry"
    if _HISTORICAL_STATEMENT_RE.search(t):
        return False, "historical_statement"
    return True, "actionable"


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
        self.stats = {"guard": 0, "classifier": 0, "semantic": 0, "llm": 0,
                      "semantic_errors": 0, "llm_errors": 0, "internal_errors": 0, "semantic_empty": 0}

    # ── public ────────────────────────────────────────────────────────────
    async def understand(self, text: str) -> CommandIR:
        raw = (text or "").strip()
        try:
            resolved = self._resolver.resolve(raw, self.ctx)
            effective, tag = resolved.text, resolved.source
        except Exception:
            logger.warning("Context resolution failed (non-fatal) — using raw text.", exc_info=True)
            effective, tag = raw, ""
        try:
            ir = await self._route(effective, raw, tag)
        except Exception:
            self.stats["internal_errors"] += 1
            logger.error("Router stage failed — degrading to UNKNOWN.", exc_info=True)
            ir = CommandIR(Status.UNKNOWN, reason="internal_error", legacy_intent="unknown",
                           raw_text=raw, effective_text=effective)
        try:
            self.ctx.observe(ir)
        except Exception:
            logger.debug("Context observe failed (non-fatal).", exc_info=True)
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
        margin = res.get("margin")   # None for keyword/guard-sourced results
        src = f"context+{res['model']}" if tag else res["model"]
        mode, target = res.get("response_mode", "llm"), res.get("target", "")

        trusted = self._trusted(res["model"], conf, margin)
        logger.info(f"[router] classifier intent={intent} conf={conf:.2f} margin={margin} "
                    f"model={res['model']} mode={mode} trusted={trusted}")
        if trusted:
            return self._from_intent_name(intent, conf, margin, src, target, **base)

        # Not trusted -> escalate. A plausible conversational label used to
        # skip escalation entirely, so it could block a semantically strong
        # command. It now always gets semantic retrieval as competing
        # evidence; the LLM fallback stays off for that case only, so the
        # existing LLM behaviour and latency are unchanged elsewhere.
        plausible_chat = mode == "llm" and conf >= 0.70 and (margin is None or margin >= 0.15)
        ir = await self._escalate(text, target, allow_llm=not plausible_chat, **base)
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

    def _target_for(self, spec, text, fallback):
        """The classifier's target was extracted for ITS predicted intent; re-derive
        it for the command actually selected (semantic/LLM may disagree)."""
        fn = getattr(self._legacy, "_extract_target", None)
        if callable(fn):
            try:
                return fn(text.lower(), spec.legacy_intent) or ""
            except Exception:
                logger.debug("Target re-extraction failed (non-fatal).", exc_info=True)
        return fallback

    def _from_spec(self, spec, conf, margin, src, target, *, llm_entities=None, **base) -> CommandIR:
        text = base["effective_text"]
        if spec.domain == "system" and _PHYSICAL_SYSTEM_TARGETS.search(text):
            logger.info(f"[router] physical-world target out of scope for {spec.key}: {text!r}")
            return CommandIR(Status.UNKNOWN, confidence=conf, margin=margin, source=src,
                             reason="physical_world_out_of_scope", legacy_intent="unknown", **base)
        act, act_reason = evaluate_actionability(spec.domain, spec.operation, text)
        if not act:
            logger.info(f"[router] non-actionable utterance for {spec.key}: {text!r} ({act_reason})")
            return CommandIR(Status.UNKNOWN, confidence=conf, margin=margin, source=src,
                             reason=act_reason, legacy_intent="unknown", **base)
        target = self._target_for(spec, text, target)
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

    async def _escalate(self, text, target, *, allow_llm: bool = True, **base) -> CommandIR | None:
        if self._semantic:
            from brain.router.confidence import Decision, evaluate
            try:
                r = await self._semantic(normalize(text))   # embeddings get normalized text
                self.stats["semantic"] += 1
                if not getattr(r, "candidates", None):
                    self.stats["semantic_empty"] += 1   # embedder down / corpus empty
            except Exception as e:
                self.stats["semantic_errors"] += 1
                logger.warning(f"Semantic retrieval failed (non-fatal, falling through): {e}")
                r = None
            dec = evaluate(r, self._sem_thr) if r is not None else None
            if r is not None:
                d = r.diagnostics()
                logger.info(f"[router] semantic {dec.value} top1={d['winning_command']}({d['winning_score']:.2f}) "
                            f"second={d['second_command']}({d['second_score']:.2f}) margin={d['margin']:.2f} text={text!r}")
            if dec is Decision.CONFIDENT:
                return self._from_spec(r.top1.spec, r.top1_similarity, r.margin, "semantic", target, **base)
        if self._llm and allow_llm:
            return await self._from_llm(text, target, **base)
        return None

    async def _from_llm(self, text, target, **base) -> CommandIR | None:
        reg = self._registry
        t0 = time.perf_counter()
        try:
            raw = await self._llm(text, reg.domains(), {d: reg.operations_for(d) for d in reg.domains()}, [])
            self.stats["llm"] += 1
        except Exception as e:
            self.stats["llm_errors"] += 1
            logger.warning(f"LLM fallback call failed (non-fatal): {e}")
            return None
        outcome = "none(timeout/unavailable)" if raw is None else f"{raw.get('domain')}.{raw.get('operation')} conf={raw.get('confidence')}"
        logger.info(f"[router] llm fallback -> {outcome} in {time.perf_counter()-t0:.2f}s")
        if raw is None:
            return None                                    # LLM down -> caller ends UNKNOWN
        parsed = validate.validate_raw_llm_output(raw)
        if parsed is None:
            return CommandIR(Status.REJECTED, source="llm", reason="malformed_llm_output", **base)
        # Identity only; a missing required entity is turned into a clarification by _from_spec.
        v = validate.validate(parsed, reg, enforce_required=False)
        if v.ok:
            eff_conf = min(parsed.confidence, 0.7)
            if eff_conf < LLM_MIN_CONF:
                return CommandIR(Status.UNKNOWN, source="llm", confidence=eff_conf,
                                 reason="llm_low_confidence", legacy_intent="unknown", **base)
            return self._from_spec(v.spec, eff_conf, None, "llm", target,
                                   llm_entities=parsed.entities, **base)
        if v.error == "not_a_command":
            return CommandIR(Status.UNKNOWN, source="llm", reason="conversational", legacy_intent="general_query", **base)
        spec = reg.get(parsed.domain, parsed.operation)
        if v.error == "needs_clarification" and spec:
            ents, missing = extract_entities(spec, base["effective_text"], target, parsed.entities)
            if not missing:
                # Nothing left to ask; forcing READY would trust an uncertain
                # signal. UNKNOWN, and NOT under the spec's skill intent.
                return CommandIR(Status.UNKNOWN, source="llm", confidence=min(parsed.confidence, 0.3),
                                 reason="llm_clarification_but_nothing_missing",
                                 legacy_intent="unknown", **base)
            q = spec.entities[missing[0]].get("prompt") or f"What {missing[0].replace('_', ' ')} did you have in mind"
            return CommandIR(Status.NEEDS_CLARIFICATION, domain=spec.domain, operation=spec.operation,
                             entities=ents, confidence=min(parsed.confidence, 0.3), source="llm",
                             legacy_intent=spec.legacy_intent, missing_entities=missing,
                             prompt=q + "?", reason="llm_needs_clarification",
                             requires_confirmation=spec.requires_confirmation, **base)
        return CommandIR(Status.REJECTED, source="llm", reason=v.error or "invalid", **base)