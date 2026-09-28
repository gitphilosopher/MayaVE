"""
brain/router/hybrid_engine.py
HybridIntentEngine — the single orchestration point for the migration.

Two entry points, by design:

  classify(text) -> dict          [SYNCHRONOUS]
      Exactly the shape core/processor.py already calls today via
      `await loop.run_in_executor(None, self._intent_engine.classify, text)`.
      When config.router.backend == "legacy" (the default), this method
      does nothing but delegate to the wrapped real IntentEngine and,
      if config.router.shadow_mode is on and a loop was registered via
      set_shadow_loop(), schedules a non-blocking shadow comparison.
      Guard checks (dismissal/presence/action/canned) are always applied
      first regardless of backend, matching IntentEngine's own guard
      precedence exactly — the guard result IS the legacy result for
      those utterances, so this is not a behavior change.

  aclassify(text) -> dict         [ASYNC — full hybrid pipeline]
      Runs the whole guard -> semantic -> confidence -> (LLM fallback) ->
      validate -> adapt pipeline. This is what a future wiring change in
      core/processor.py would call instead of classify() once hybrid
      routing is enabled for real (see docs/CONTRIBUTING.md's migration
      stages) — NOT part of this migration's acceptance criteria, which
      requires zero behavior change until router.backend is flipped.

Per-domain allowlisting: even with backend == "hybrid", only domains
listed in config.router.hybrid_domains are actually routed through the
new pipeline; every other domain's utterances still fall through to the
wrapped legacy engine. This lets Stage 5 enable e.g. just "datetime" and
"clipboard" while everything else stays on the proven path.
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
        """Rebuild the command vector index from the current registry.
        Call this explicitly (e.g. from a startup task or a maintenance
        script) — never happens implicitly on a stale corpus, since that
        would mean silently blocking on N embedding calls mid-request."""
        specs = self._registry.specs
        written = 0
        # Reimplemented inline (rather than CommandVectorStore.reseed's sync
        # embed_fn) so the embed awaits happen on this loop.
        from brain.router.command_vector_store import corpus_fingerprint
        from brain.vector_store import SQLiteVectorStore, MemoryRecord

        self._store._store.clear()
        self._store._store = SQLiteVectorStore(db_path=self._store._db_path)

        for spec in specs:
            for seed in spec.seeds:
                embedding = await self._embedder.embed(seed)
                if embedding is None:
                    logger.warning(f"Skipping unembeddable seed for {spec.key}: '{seed}'")
                    continue
                record = MemoryRecord(
                    content=seed, mem_type="command", topic=spec.key, importance=1.0,
                    source="router_seed",
                    metadata={"domain": spec.domain, "operation": spec.operation,
                              "legacy_intent": spec.legacy_intent},
                    embedding=embedding,
                )
                if self._store._store.add(record) is not None:
                    written += 1

        self._store._write_fingerprint(corpus_fingerprint(specs))
        logger.info(f"Command vector store reseeded — {written} seed vector(s).")
        return written

    # ── Sync entry point (drop-in for IntentEngine.classify) ────────────

    def classify(self, text: str) -> dict:
        guard_hit = guards.check(self._legacy, text)
        if guard_hit is not None:
            intent, confidence, source = guard_hit
            result = self._legacy_shaped(intent, text, confidence, source)
        else:
            result = self._legacy.classify(text)

        # classify() always dispatches the legacy-shaped result above,
        # regardless of config.router.backend — enabling "hybrid" only
        # takes effect once a caller switches to aclassify() (see this
        # module's docstring). Shadow mode compares against that same
        # legacy result without ever affecting what was returned here.
        if getattr(config.router, "shadow_mode", False) and self._shadow_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shadow_compare(text, result), self._shadow_loop)
            except Exception as e:
                logger.debug(f"Shadow-mode scheduling failed (non-fatal): {e}")

        return result

    def set_shadow_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the running asyncio loop so classify() (called from a
        worker thread via run_in_executor) can schedule shadow-mode
        comparisons without blocking the caller."""
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

        candidates_in_scope = [
            c for c in retrieval.candidates
            if not allowed_domains or c.spec.domain in allowed_domains
        ]
        if not candidates_in_scope:
            logger.info(
                f"[router] source=legacy_fallback reason=no_in_scope_candidate "
                f"latency={time.perf_counter()-t0:.3f}s"
            )
            return self._legacy.classify(text)

        from brain.router.schemas import RetrievalResult
        scoped = RetrievalResult(candidates=candidates_in_scope)
        decision = evaluate(scoped, self._thresholds)

        if decision == Decision.CONFIDENT:
            top = scoped.top1
            command = Command(domain=top.spec.domain, operation=top.spec.operation,
                               entities={}, confidence=top.similarity)
            v = validate.validate(command, self._registry)
            if v.ok:
                logger.info(
                    f"[router] source=semantic domain={top.spec.domain} operation={top.spec.operation} "
                    f"top1={scoped.top1_similarity:.3f} top2={scoped.top2_similarity:.3f} "
                    f"margin={scoped.margin:.3f} embed_search={t_embed1-t_embed0:.3f}s "
                    f"latency={time.perf_counter()-t0:.3f}s"
                )
                return to_legacy_intent(command, v.spec, text,
                                         confidence=top.similarity, model_source="semantic_retrieval")
            logger.warning(f"[router] semantic candidate failed validation ({v.error}) — falling to LLM.")

        # AMBIGUOUS, LOW, or a confident-but-invalid semantic match: ask the LLM.
        top_candidate_keys = [c.spec.key for c in scoped.candidates[:3]]
        operations_by_domain = {d: self._registry.operations_for(d) for d in self._registry.domains()}
        t_llm0 = time.perf_counter()
        raw = await llm_route(text, self._registry.domains(), operations_by_domain, top_candidate_keys)
        t_llm1 = time.perf_counter()

        if raw is None:
            logger.info(
                f"[router] source=legacy_fallback reason=llm_unavailable decision={decision.value} "
                f"latency={time.perf_counter()-t0:.3f}s"
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

        logger.info(
            f"[router] source=llm_fallback domain={parsed.domain} operation={parsed.operation} "
            f"decision={decision.value} llm_latency={t_llm1-t_llm0:.3f}s "
            f"latency={time.perf_counter()-t0:.3f}s"
        )
        return to_legacy_intent(parsed, v.spec, text, confidence=0.7, model_source="llm_fallback")

    # ── Shadow mode ──────────────────────────────────────────────────────

    async def _shadow_compare(self, text: str, legacy_result: dict) -> None:
        """Compute the hybrid decision purely for comparison logging.
        Never affects what was dispatched — legacy_result already went
        to the router by the time this runs. Best-effort; any failure is
        swallowed so shadow diagnostics can never destabilize a turn."""
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
