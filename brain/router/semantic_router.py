"""
brain/router/semantic_router.py
Embedding-based command retrieval. Reuses brain/embeddings.py's
OllamaEmbedder — no second embedding client is introduced.
"""

import logging
import time

from brain.embeddings import OllamaEmbedder
from brain.router.command_vector_store import CommandVectorStore
from brain.router.registry import CommandRegistry
from brain.router.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class SemanticRouter:
    """Stateless-ish facade: holds the embedder + vector store + registry
    needed to turn one utterance into a ranked RetrievalResult."""

    def __init__(self, embedder: OllamaEmbedder, store: CommandVectorStore, registry: CommandRegistry):
        self._embedder = embedder
        self._store = store
        self._registry = registry

    async def retrieve(self, normalized_text: str, top_k: int = 5) -> RetrievalResult:
        t0 = time.perf_counter()
        embedding = await self._embedder.embed(normalized_text)
        if embedding is None:
            logger.debug("Router: embedding unavailable — returning empty retrieval result.")
            return RetrievalResult(candidates=[])
        t1 = time.perf_counter()
        result = self._store.search(embedding, self._registry.by_key, top_k=top_k)
        t2 = time.perf_counter()
        logger.info(
            f"[TIMING][router] embed={t1-t0:.3f}s search={t2-t1:.3f}s "
            f"top1={result.top1_similarity:.3f} top2={result.top2_similarity:.3f} margin={result.margin:.3f}"
        )
        return result