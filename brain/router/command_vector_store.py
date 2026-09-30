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

One vector per prototype: every prototype (seed) of an operation in
config/command_domains.json becomes one row whose metadata carries
`domain`, `operation`, `legacy_intent` and `prototype`, and whose `topic`
is the `domain.operation` key. search() reads every command vector,
aggregates them per logical command (see schemas.aggregate_by_command)
and returns a command-level ranking — the top-k cut happens AFTER
aggregation, never before, so a command with many prototypes can no
longer crowd other commands out of the candidate list.

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

from brain.router.schemas import (
    CommandSpec, RetrievalResult, aggregate_by_command,
)
from brain.vector_store import SQLiteVectorStore, MemoryRecord

logger = logging.getLogger(__name__)

_MEM_TYPE = "command"

# Upper bound on vectors read per search. The store is brute-force, so the
# whole command corpus is scanned regardless; this only keeps the top-k cut
# from truncating the corpus before per-command aggregation.
_SCAN_ALL = 1_000_000


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


def _prototype_record(spec: CommandSpec, prototype: str, embedding: list[float]) -> MemoryRecord:
    """The single place a prototype becomes a stored row, so every vector
    carries the metadata aggregation needs (domain/operation/prototype)."""
    return MemoryRecord(
        content=prototype,
        mem_type=_MEM_TYPE,
        topic=spec.key,
        importance=1.0,
        source="router_seed",
        metadata={
            "domain": spec.domain,
            "operation": spec.operation,
            "legacy_intent": spec.legacy_intent,
            "prototype": prototype,
        },
        embedding=embedding,
    )


class CommandVectorStore:
    """Owns the command-vector SQLite file and its fingerprint sidecar."""

    def __init__(self, db_path: "Path | str | None" = None):
        self._db_path = Path(db_path) if db_path else _default_db_path()
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

    def _reset_index(self) -> None:
        """Empty the table through the store (never by deleting the open
        SQLite file) so the store keeps owning its own lifecycle."""
        self._store.clear()
        self._store = SQLiteVectorStore(db_path=self._db_path)

    def _finish_reseed(self, specs: list[CommandSpec], written: int) -> int:
        self._write_fingerprint(corpus_fingerprint(specs))
        logger.info(
            f"Command vector store reseeded — {written} prototype vector(s) "
            f"across {len(specs)} operation(s)."
        )
        return written

    def reseed(self, specs: list[CommandSpec], embed_fn) -> int:
        """
        Rebuild the command index from scratch. `embed_fn` is a sync
        callable str -> list[float] | None (kept sync here so this module
        has no asyncio dependency of its own; see areseed() for the async
        embedder used at runtime).

        Clears the underlying SQLite table first so a stale/renamed domain
        never leaves orphaned rows behind. Returns the number of prototype
        vectors written.
        """
        self._reset_index()
        written = 0
        for spec in specs:
            for prototype in spec.seeds:
                embedding = embed_fn(prototype)
                if embedding is None:
                    logger.warning(f"Skipping unembeddable prototype for {spec.key}: '{prototype}'")
                    continue
                if self._store.add(_prototype_record(spec, prototype, embedding)) is not None:
                    written += 1
        return self._finish_reseed(specs, written)

    async def areseed(self, specs: list[CommandSpec], embed_fn) -> int:
        """Async twin of reseed() — `embed_fn` is an async callable
        (e.g. OllamaEmbedder.embed). Same ordering, same records."""
        self._reset_index()
        written = 0
        for spec in specs:
            for prototype in spec.seeds:
                embedding = await embed_fn(prototype)
                if embedding is None:
                    logger.warning(f"Skipping unembeddable prototype for {spec.key}: '{prototype}'")
                    continue
                if self._store.add(_prototype_record(spec, prototype, embedding)) is not None:
                    written += 1
        return self._finish_reseed(specs, written)

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, embedding: list[float], specs_by_key: dict[str, CommandSpec],
               top_k: int = 5) -> RetrievalResult:
        """Top-k DISTINCT commands. Every prototype vector is scored, then
        reduced to one score per domain.operation (its best prototype);
        only then is the top-k cut applied. Multiple prototypes of the same
        command therefore never appear as competing candidates."""
        raw = self._store.search(embedding, top_k=_SCAN_ALL, mem_type=_MEM_TYPE, min_similarity=0.0)

        scored = []
        for record, sim in raw:
            meta = record.metadata or {}
            if meta.get("domain") and meta.get("operation"):
                key = f"{meta['domain']}.{meta['operation']}"
            else:
                key = record.topic
            spec = specs_by_key.get(key)
            if spec is None:
                continue   # stale row from a since-removed operation
            scored.append((spec, sim, meta.get("prototype") or record.content))

        return aggregate_by_command(scored, top_k=top_k)