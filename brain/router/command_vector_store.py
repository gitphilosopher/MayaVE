"""
brain/router/command_vector_store.py
Command-embedding index for the hybrid semantic router.

Reuses brain/vector_store.py's SQLiteVectorStore (NumPy brute-force
cosine search) exactly as-is — no new vector database, no changes to
that module. The only new thing here is a SEPARATE physical SQLite file
(~/Maya/Router/command_vectors.sqlite3 by default, overridable via
config.router.command_vector_db_path, mirroring config.context.memory_dir's
pattern) so command vectors can never mix with ContextManager's
conversational semantic memories in brain/conversation.py.

SQLiteVectorStore.add()/search() place no restriction on `mem_type` —
the VALID_MEM_TYPES check lives in brain/conversation.py's own
_persist_memory(), not in the store — so this module is free to use
mem_type="command" without touching vector_store.py at all.

Fingerprinting: a companion `<db>.fingerprint.json` file (same
atomic-write pattern as services/node/sync_state.py) records the hash
of config/command_domains.json that produced the current seed set. A
mismatch means the taxonomy changed since the corpus was built; callers
should reseed rather than search a stale index.
"""

import hashlib
import json
import logging
from pathlib import Path

from brain.router.schemas import CommandSpec, CommandCandidate, RetrievalResult
from brain.vector_store import SQLiteVectorStore, MemoryRecord

logger = logging.getLogger(__name__)

_MEM_TYPE = "command"


def _default_db_path() -> Path:
    return Path.home() / "Maya" / "Router" / "command_vectors.sqlite3"


def corpus_fingerprint(specs: list[CommandSpec]) -> str:
    """Deterministic hash of every seed the corpus would produce — changes
    whenever config/command_domains.json's domains/operations/seeds change."""
    payload = json.dumps(
        [
            {"key": s.key, "legacy_intent": s.legacy_intent,
             "entities": s.entities, "target_mode": s.target_mode,
             "seeds": sorted(s.seeds)}
            for s in sorted(specs, key=lambda s: s.key)
        ],
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CommandVectorStore:
    """Owns the command-vector SQLite file and its fingerprint sidecar."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp_path = self._db_path.with_suffix(self._db_path.suffix + ".fingerprint.json")
        self._store = SQLiteVectorStore(db_path=self._db_path)

    # ── Fingerprint / reseed lifecycle ──────────────────────────────────

    def stored_fingerprint(self) -> str | None:
        try:
            if self._fp_path.exists():
                return json.loads(self._fp_path.read_text(encoding="utf-8")).get("fingerprint")
        except (OSError, ValueError) as e:
            logger.warning(f"Command corpus fingerprint unreadable ({e}) — treating as stale.")
        return None

    def _write_fingerprint(self, fingerprint: str) -> None:
        tmp = self._fp_path.with_name(self._fp_path.name + ".tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
        tmp.replace(self._fp_path)

    def is_stale(self, specs: list[CommandSpec]) -> bool:
        return self.stored_fingerprint() != corpus_fingerprint(specs)

    def reseed(self, specs: list[CommandSpec], embed_fn) -> int:
        """
        Rebuild the command index from scratch. `embed_fn` is a sync
        callable str -> list[float] | None (the caller runs the actual
        async OllamaEmbedder.embed() and adapts it — kept sync here so
        this module has no asyncio dependency of its own).

        Clears the underlying SQLite table first so a stale/renamed domain
        never leaves orphaned rows behind. Returns the number of seed vectors
        written.
        """
        self._store.clear()
        self._store = SQLiteVectorStore(db_path=self._db_path)

        written = 0
        for spec in specs:
            for seed in spec.seeds:
                embedding = embed_fn(seed)
                if embedding is None:
                    logger.warning(f"Skipping unembeddable seed for {spec.key}: '{seed}'")
                    continue
                record = MemoryRecord(
                    content=seed,
                    mem_type=_MEM_TYPE,
                    topic=spec.key,
                    importance=1.0,
                    source="router_seed",
                    metadata={
                        "domain": spec.domain,
                        "operation": spec.operation,
                        "legacy_intent": spec.legacy_intent,
                    },
                    embedding=embedding,
                )
                if self._store.add(record) is not None:
                    written += 1

        self._write_fingerprint(corpus_fingerprint(specs))
        logger.info(f"Command vector store reseeded — {written} seed vector(s) across {len(specs)} operation(s).")
        return written

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, embedding: list[float], specs_by_key: dict[str, CommandSpec],
               top_k: int = 5) -> RetrievalResult:
        """Top-k command matches, deduplicated to the single best-scoring
        seed per domain.operation (multiple seeds for the same operation
        must not crowd out a different operation from the top-k)."""
        raw = self._store.search(embedding, top_k=top_k * 3, mem_type=_MEM_TYPE, min_similarity=0.0)

        best_per_key: dict[str, CommandCandidate] = {}
        for record, sim in raw:
            key = record.topic
            spec = specs_by_key.get(key)
            if spec is None:
                continue   # stale row from a since-removed operation
            existing = best_per_key.get(key)
            if existing is None or sim > existing.similarity:
                best_per_key[key] = CommandCandidate(spec=spec, similarity=sim, seed_text=record.content)

        ranked = sorted(best_per_key.values(), key=lambda c: c.similarity, reverse=True)[:top_k]
        return RetrievalResult(candidates=ranked)