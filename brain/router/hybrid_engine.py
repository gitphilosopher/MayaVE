"""
brain/router/hybrid_engine.py
HybridIntentEngine — legacy-shaped hybrid pipeline (guards -> semantic ->
confidence -> LLM fallback -> validate -> adapt).

Two entry points:
  classify(text) -> dict   [SYNC]  legacy drop-in; always returns the legacy
                                   engine's result (+ optional shadow compare).
  aclassify(text) -> dict  [ASYNC] full hybrid pipeline.

BATCH 2 FIX (semantic path vs required entities): a CONFIDENT semantic hit
selects a command before any entity exists, but it was validated with
entities={} against the full required-entity rules. Any operation with a
required entity (timer.create -> duration, app_or_web.open -> target,
web.search -> query) therefore failed validation and fell through to the LLM
even though retrieval was confident. Now:
  1. the command's identity is validated with enforce_required=False;
  2. entities are extracted AFTER selection with the existing extractors
     (brain/router/entities.py) and the classifier's own target extraction
     (IntentEngine._extract_target — reused, not re-implemented);
  3. missing required entities do not block dispatch here: this legacy-shaped
     path has no clarification status, and adapter.py already falls back to
     the raw utterance so the skill asks its own clarification. (The IR
     pipeline in understand.py is the path that represents NEEDS_CLARIFICATION.)
The LLM branch is unchanged: LLM output is validated with required-entity
enforcement. Nothing here executes a skill.

Allowlisting: only domains in config.router.hybrid_domains are routed through
this pipeline when backend == "hybrid"; the allowlist is applied to the
WINNING command, and an out-of-allowlist runner-up still counts toward margin.
"""

from __future__ import annotations

import asyncio
import logging
import time

from brain.embeddings import OllamaEmbedder
from brain.intent_engine import IntentEngine
from brain.router import guards, validate
from brain.router.adapter import to_legacy_intent
from brain.router.command_vector_store import CommandVectorStore
from brain.router.confidence import ConfidenceThresholds, Decision, evaluate
from brain.router.entities import extract_entities
from brain.router.llm_fallback import route as llm_route
from brain.router.normalize import normalize
from brain.router.registry import CommandRegistry, load_specs
from brain.router.schemas import Command
from brain.router.semantic_router import SemanticRouter
from config.settings import config

logger = logging.getLogger(__name__)


class HybridIntentEngine:
    def __init__(self, legacy_engine: IntentEngine | None = None):
        self._legacy = legacy_engine or IntentEngine()

        self._registry = CommandRegistry(load_specs())
        self._store = CommandVectorStore(getattr(config.router, "command_vector_db_path", None))
        self._embedder = OllamaEmbedder()
        self._semantic = SemanticRouter(self._embedder, self._store, self._registry)
        self._thresholds = ConfidenceThresholds(
            min_similarity=getattr(config.router, "min_similarity", 0.80),
            min_margin=getattr(config.router, "min_margin", 0.08),
            low_similarity_floor=getattr(config.router, "low_similarity_floor", 0.55),
        )

        self._shadow_loop: asyncio.AbstractEventLoop | None = None

        if self._store.is_stale(self._registry.specs):
            logger.warning(
                "Command vector corpus is stale or unseeded — hybrid/shadow routing will "
                "return low-confidence results until brain/router/eval_router.py's seed step "
                "(or an explicit reseed call) has been run for the current command_domains.json."
            )

    # ── Corpus management ────────────────────────────────────────────────

    async def reseed_corpus(self) -> int:
        """Rebuild the command vector index from the current registry. Explicit
        only — never implicit on a stale corpus."""
        return await self._store.areseed(self._registry.specs, self._embedder.embed)

    # ── Sync entry point (drop-in for IntentEngine.classify) ────────────

    def classify(self, text: str) -> dict:
        guard_hit = guards.check(self._legacy, text)
        if guard_hit is not None:
            intent, confidence, source = guard_hit
            result = self._legacy_shaped(intent, text, confidence, source)
        else:
            result = self._legacy.classify(text)

        if getattr(config.router, "shadow_mode", False) and self._shadow_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shadow_compare(text, result), self._shadow_loop)
            except Exception as e:
                logger.debug(f"Shadow-mode scheduling failed (non-fatal): {e}")

        return result

    def set_shadow_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._shadow_loop = loop

    def _legacy_shaped(self, intent: str, text: str, confidence: float, source: str) -> dict:
        return {
            "intent": intent,
            "target": "",
            "confidence": round(float(confidence), 3),
            "raw": text,
            "model": source,
            "response_mode": self._legacy._response_modes.get(intent, "llm"),
        }

    def _with_extracted_entities(self, seed: Command, spec, text: str) -> Command:
        """Extract entities for a semantically selected command. Reuses the
        classifier's target extraction (private, like guards.py's coupling) and
        the shared extractors; never raises."""
        target = ""
        extract_target = getattr(self._legacy, "_extract_target", None)
        if callable(extract_target):
            try:
                target = extract_target(text.lower(), spec.legacy_intent) or ""
            except Exception as e:
                logger.debug(f"Target extraction failed (non-fatal): {e}")
        try:
            ents, missing = extract_entities(spec, text, target)
        except Exception as e:
            logger.debug(f"Entity extraction failed (non-fatal): {e}")
            ents, missing = {}, ()
        if missing:
            logger.info(
                f"[router] semantic hit {spec.key} is missing required {list(missing)} — "
                f"dispatching; the skill will ask for it."
            )
        return Command(domain=seed.domain, operation=seed.operation, entities=ents,
                       confidence=seed.confidence)

    # ── Async entry point (full hybrid pipeline) ────────────────────────

    async def aclassify(self, text: str) -> dict:
        t0 = time.perf_counter()

        guard_hit = guards.check(self._legacy, text)
        if guard_hit is not None:
            intent, confidence, source = guard_hit
            logger.info(f"[router] source=guard intent={intent} latency={time.perf_counter()-t0:.3f}s")
            return self._legacy_shaped(intent, text, confidence, source)

        backend = getattr(config.router, "backend", "legacy")
        if backend == "legacy":
            return self._legacy.classify(text)

        normalized = normalize(text)
        allowed_domains = set(getattr(config.router, "hybrid_domains", []) or [])

        t_embed0 = time.perf_counter()
        retrieval = await self._semantic.retrieve(normalized)
        t_embed1 = time.perf_counter()

        top = retrieval.top1
        if top is None or (allowed_domains and top.spec.domain not in allowed_domains):
            logger.info(
                f"[router] source=legacy_fallback reason=no_in_scope_candidate "
                f"winner={top.spec.key if top else None} latency={time.perf_counter()-t0:.3f}s"
            )
            return self._legacy.classify(text)

        diag = retrieval.diagnostics()
        decision = evaluate(retrieval, self._thresholds)

        if decision == Decision.CONFIDENT:
            seed = Command(domain=top.spec.domain, operation=top.spec.operation,
                           entities={}, confidence=top.similarity)
            # Identity only: required entities are extracted after selection.
            v = validate.validate(seed, self._registry, enforce_required=False)
            if v.ok:
                command = self._with_extracted_entities(seed, v.spec, text)
                logger.info(
                    f"[router] source=semantic winner={diag['winning_command']} "
                    f"score={diag['winning_score']:.3f} second={diag['second_command']} "
                    f"second_score={diag['second_score']:.3f} margin={diag['margin']:.3f} "
                    f"embed_search={t_embed1-t_embed0:.3f}s latency={time.perf_counter()-t0:.3f}s"
                )
                intent = to_legacy_intent(command, v.spec, text,
                                           confidence=top.similarity, model_source="semantic_retrieval")
                intent["_command"]["retrieval"] = diag
                return intent
            logger.warning(f"[router] semantic candidate failed validation ({v.error}) — falling to LLM.")

        # AMBIGUOUS, LOW, or a confident-but-invalid semantic match: ask the LLM.
        top_candidate_keys = [c.spec.key for c in retrieval.candidates[:3]]
        operations_by_domain = {d: self._registry.operations_for(d) for d in self._registry.domains()}
        t_llm0 = time.perf_counter()
        raw = await llm_route(text, self._registry.domains(), operations_by_domain, top_candidate_keys)
        t_llm1 = time.perf_counter()

        if raw is None:
            logger.info(
                f"[router] source=legacy_fallback reason=llm_unavailable decision={decision.value} "
                f"retrieval={diag} latency={time.perf_counter()-t0:.3f}s"
            )
            return self._legacy.classify(text)

        parsed = validate.validate_raw_llm_output(raw)
        if parsed is None:
            logger.warning(f"[router] LLM output structurally invalid: {raw!r} — falling to legacy.")
            return self._legacy.classify(text)

        v = validate.validate(parsed, self._registry)
        if not v.ok:
            logger.info(
                f"[router] source=legacy_fallback reason=llm_validation_failed error={v.error} "
                f"decision={decision.value} llm_latency={t_llm1-t_llm0:.3f}s "
                f"latency={time.perf_counter()-t0:.3f}s"
            )
            return self._legacy.classify(text)

        if allowed_domains and parsed.domain not in allowed_domains:
            logger.info(f"[router] source=legacy_fallback reason=llm_domain_not_allowlisted domain={parsed.domain}")
            return self._legacy.classify(text)

        logger.info(
            f"[router] source=llm_fallback domain={parsed.domain} operation={parsed.operation} "
            f"decision={decision.value} retrieval={diag} llm_latency={t_llm1-t_llm0:.3f}s "
            f"latency={time.perf_counter()-t0:.3f}s"
        )
        intent = to_legacy_intent(parsed, v.spec, text, confidence=0.7, model_source="llm_fallback")
        intent["_command"]["retrieval"] = diag
        return intent

    # ── Shadow mode ──────────────────────────────────────────────────────

    async def _shadow_compare(self, text: str, legacy_result: dict) -> None:
        """Compute the hybrid decision purely for comparison logging; never
        affects what was dispatched. Best-effort."""
        try:
            t0 = time.perf_counter()
            hybrid_result = await self.aclassify(text)
            latency = time.perf_counter() - t0
            agree = hybrid_result.get("intent") == legacy_result.get("intent")
            logger.info(
                f"[router:shadow] agree={agree} legacy_intent={legacy_result.get('intent')} "
                f"hybrid_intent={hybrid_result.get('intent')} "
                f"hybrid_source={hybrid_result.get('model')} latency={latency:.3f}s"
            )
        except Exception as e:
            logger.debug(f"Shadow comparison failed (non-fatal): {e}")