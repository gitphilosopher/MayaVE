"""
brain/router/hybrid_engine.py
============================
Hybrid compatibility engine for legacy-shaped command routing.

This module is the bridge between the legacy ``IntentEngine.classify()`` output
and the newer hybrid route selection strategy. It keeps the legacy API shape for
compatibility while optionally evaluating semantic retrieval and LLM fallback
before returning a legacy-style result dictionary.

The important runtime behavior is:
- ``classify()`` remains the synchronous drop-in entry point and preserves the
  legacy output contract while optionally scheduling a shadow comparison.
- ``aclassify()`` runs the full hybrid pipeline for async callers, including
  guard checks, semantic retrieval, and LLM fallback when needed.
- semantic hits are validated as command identities first with required-entity
  checks relaxed, then entity extraction is performed afterward so required
  properties such as timer duration or search query can be resolved without
  prematurely rejecting a confident command.
- when the command is not trusted or not fully actionable, the engine falls back
  to the legacy classifier or to the LLM path without executing any skill.
- hybrid routing remains allowlist-aware: only configured domains are run
  through the hybrid path when the backend is set to ``"hybrid"``.

The file is intentionally compatibility-oriented rather than a primary execution
layer; it adapts command selection back into the older dict-based intent contract
for callers that still rely on that shape.
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
from brain.router.understand import LLM_MIN_CONF
from config.settings import config

logger = logging.getLogger(__name__)


class HybridIntentEngine:
    """Compatibility wrapper that can route through semantic and LLM fallback while preserving the legacy result shape."""

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
        """Return the legacy-style classification result while optionally running a shadow comparison."""
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
        """Normalize a routed intent into the legacy dict shape expected by older callers."""
        return {
            "intent": intent,
            "target": "",
            "confidence": round(float(confidence), 3),
            "raw": text,
            "model": source,
            "response_mode": self._legacy._response_modes.get(intent, "llm"),
        }

    def _with_extracted_entities(self, seed: Command, spec, text: str) -> tuple[Command, tuple]:
        """Fill a semantic command with extracted entities after selection and before legacy adaptation."""
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
                f"not dispatching from the legacy-shaped path (it has no clarification status)."
            )
        return Command(domain=seed.domain, operation=seed.operation, entities=ents,
                       confidence=seed.confidence), missing

    # ── Async entry point (full hybrid pipeline) ────────────────────────

    async def aclassify(self, text: str, *, force_hybrid: bool = False) -> dict:
        """Run the full hybrid routing pipeline and return the legacy-shaped result."""
        t0 = time.perf_counter()

        guard_hit = guards.check(self._legacy, text)
        if guard_hit is not None:
            intent, confidence, source = guard_hit
            logger.info(f"[router] source=guard intent={intent} latency={time.perf_counter()-t0:.3f}s")
            return self._legacy_shaped(intent, text, confidence, source)

        backend = getattr(config.router, "backend", "legacy")
        if backend == "legacy" and not force_hybrid:
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
                command, missing = self._with_extracted_entities(seed, v.spec, text)
                if missing:
                    return self._legacy.classify(text)
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

        # When semantic evidence is ambiguous or unusable, the LLM becomes the
        # next fallback. This keeps the hybrid engine conservative without
        # changing the legacy output contract.
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

        if parsed.confidence < LLM_MIN_CONF:
            logger.info(f"[router] source=legacy_fallback reason=llm_low_confidence conf={parsed.confidence:.2f}")
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
        """Log the hybrid decision for comparison only; it never changes dispatch behavior."""
        try:
            t0 = time.perf_counter()
            hybrid_result = await self.aclassify(text, force_hybrid=True)
            latency = time.perf_counter() - t0
            agree = hybrid_result.get("intent") == legacy_result.get("intent")
            logger.info(
                f"[router:shadow] agree={agree} legacy_intent={legacy_result.get('intent')} "
                f"hybrid_intent={hybrid_result.get('intent')} "
                f"hybrid_source={hybrid_result.get('model')} latency={latency:.3f}s"
            )
        except Exception as e:
            logger.debug(f"Shadow comparison failed (non-fatal): {e}")