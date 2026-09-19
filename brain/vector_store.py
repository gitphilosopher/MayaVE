"""
brain/vector_store.py
Local persistent vector store for Maya's long-term semantic memory.

Backend: SQLite (stdlib) + brute-force cosine similarity via numpy.
No new dependency — both are already required by Maya. Adequate at
desktop scale (comfortably thousands of memory records).

Isolated behind the VectorStore interface so a real ANN backend
(Chroma, FAISS, sqlite-vec, ...) can replace SQLiteVectorStore later
without any change to ContextManager or callers.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config.settings import config

logger = logging.getLogger(__name__)

_MEMORY_DIR = Path(getattr(config.context, "memory_dir", None) or (Path.home() / "Maya" / "Memory"))
_DB_PATH = _MEMORY_DIR / "semantic_memory.sqlite3"

VALID_MEM_TYPES = {
    "fact", "preference", "goal", "decision",
    "relationship", "project", "conversation_summary",
}


@dataclass
class MemoryRecord:
    content:    str
    mem_type:   str
    topic:      str = ""
    importance: float = 0.5
    source:     str = "conversation"
    timestamp:  float = field(default_factory=time.time)
    metadata:   dict = field(default_factory=dict)
    id:         int | None = None
    embedding:  list[float] | None = None


class VectorStore:
    """Abstract interface — implement this to swap the storage backend."""

    def add(self, record: MemoryRecord) -> int | None:
        raise NotImplementedError

    def update(self, record_id: int, content: str, embedding: list[float], timestamp: float) -> None:
        raise NotImplementedError

    def has_records(self) -> bool:
        """Cheap "is there anything to search?" gate. Default True (never skips)."""
        return True

    def search(self, embedding: list[float], top_k: int = 3,
               mem_type: str | None = None, min_similarity: float = 0.0) -> list[tuple[MemoryRecord, float]]:
        raise NotImplementedError

    def find_similar(self, embedding: list[float], mem_type: str, topic: str,
                      threshold: float) -> MemoryRecord | None:
        raise NotImplementedError


class SQLiteVectorStore(VectorStore):
    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    mem_type TEXT NOT NULL,
                    topic TEXT,
                    importance REAL,
                    source TEXT,
                    timestamp REAL,
                    metadata TEXT
                )
            """)
            conn.commit()
        logger.info(f"Semantic memory store ready — {self._db_path}")

    @staticmethod
    def _to_blob(embedding: list[float]) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def add(self, record: MemoryRecord) -> int | None:
        if record.embedding is None:
            return None
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO memories (content, embedding, mem_type, topic, importance, "
                    "source, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (record.content, self._to_blob(record.embedding), record.mem_type,
                     record.topic, record.importance, record.source, record.timestamp,
                     json.dumps(record.metadata)),
                )
                conn.commit()
                return cur.lastrowid
        except Exception as e:
            logger.error(f"Vector store add failed: {e}", exc_info=True)
            return None

    def update(self, record_id: int, content: str, embedding: list[float], timestamp: float) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE memories SET content=?, embedding=?, timestamp=? WHERE id=?",
                    (content, self._to_blob(embedding), timestamp, record_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Vector store update failed: {e}", exc_info=True)

    def has_records(self) -> bool:
        # Fail open: on any error report True so callers do the full search.
        try:
            with self._connect() as conn:
                return conn.execute("SELECT 1 FROM memories LIMIT 1").fetchone() is not None
        except Exception as e:
            logger.error(f"Vector store has_records failed: {e}", exc_info=True)
            return True

    def _all_rows(self, mem_type: str | None = None):
        query = ("SELECT id, content, embedding, mem_type, topic, importance, "
                  "source, timestamp, metadata FROM memories")
        params: tuple = ()
        if mem_type:
            query += " WHERE mem_type=?"
            params = (mem_type,)
        with self._connect() as conn:
            return conn.execute(query, params).fetchall()

    def search(self, embedding: list[float], top_k: int = 3,
               mem_type: str | None = None, min_similarity: float = 0.0) -> list[tuple[MemoryRecord, float]]:
        try:
            rows = self._all_rows(mem_type)
            if not rows:
                return []
            q = np.asarray(embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return []
            scored = []
            for row in rows:
                vec = self._from_blob(row[2])
                denom = np.linalg.norm(vec) * q_norm
                sim = float(np.dot(q, vec) / denom) if denom else 0.0
                if sim >= min_similarity:
                    scored.append((self._row_to_record(row), sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.error(f"Vector store search failed: {e}", exc_info=True)
            return []

    def find_similar(self, embedding: list[float], mem_type: str, topic: str,
                      threshold: float) -> MemoryRecord | None:
        results = self.search(embedding, top_k=1, mem_type=mem_type, min_similarity=threshold)
        if results and (not topic or results[0][0].topic == topic):
            return results[0][0]
        return None

    @staticmethod
    def _row_to_record(row) -> MemoryRecord:
        return MemoryRecord(
            id=row[0], content=row[1], embedding=None, mem_type=row[3],
            topic=row[4] or "", importance=row[5] if row[5] is not None else 0.5,
            source=row[6] or "conversation", timestamp=row[7] or 0.0,
            metadata=json.loads(row[8]) if row[8] else {},
        )