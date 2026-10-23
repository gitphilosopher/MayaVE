"""
Context and Evidence Caching for MayaVE Maintenance Coordinator.
Computes and caches SHA-256 hashes and file contexts to prevent redundant disk
reads, hashing operations, and token overhead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from .models import TaskDefinition
except ImportError:
    from models import TaskDefinition


@dataclass
class FileCacheEntry:
    path: str
    sha256: str
    mtime: float
    size: int
    content: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "mtime": self.mtime,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileCacheEntry:
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            mtime=float(data["mtime"]),
            size=int(data["size"]),
        )


class ContextCache:
    """
    Durable and in-memory cache for file hashes and context digests.
    Validates mtime and size before re-reading from disk.
    """

    def __init__(self, cache_file: Path | str | None = None) -> None:
        self.cache_file = Path(cache_file).resolve() if cache_file else None
        self._entries: dict[str, FileCacheEntry] = {}
        if self.cache_file and self.cache_file.is_file():
            self._load_cache()

    def _normalize_key(self, path: Path | str) -> str:
        return str(Path(path).resolve())

    def _load_cache(self) -> None:
        if not self.cache_file or not self.cache_file.is_file():
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self._entries[k] = FileCacheEntry.from_dict(v)
        except Exception:
            # Fall back to empty cache on corrupt or unreadable cache file
            self._entries = {}

    def save_cache(self) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self._entries.items()}
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get_file_entry(self, path: Path | str, retain_content: bool = False) -> Optional[FileCacheEntry]:
        """
        Fetch or compute file cache entry.
        Returns None if file does not exist on disk.
        """
        resolved = Path(path).resolve()
        if not resolved.is_file():
            # If deleted or absent, remove stale cache entry
            key = str(resolved)
            self._entries.pop(key, None)
            return None

        stat = resolved.stat()
        key = str(resolved)
        cached = self._entries.get(key)

        # Check cache validity (mtime and size match)
        if cached and cached.mtime == stat.st_mtime and cached.size == stat.st_size:
            if retain_content and cached.content is None:
                try:
                    cached.content = resolved.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            return cached

        # Recompute hash
        hasher = hashlib.sha256()
        content: Optional[str] = None
        try:
            raw_bytes = resolved.read_bytes()
            hasher.update(raw_bytes)
            sha = hasher.hexdigest()
            if retain_content:
                content = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            return None

        entry = FileCacheEntry(
            path=key,
            sha256=sha,
            mtime=stat.st_mtime,
            size=stat.st_size,
            content=content,
        )
        self._entries[key] = entry
        return entry

    def get_file_hash(self, path: Path | str) -> Optional[str]:
        entry = self.get_file_entry(path)
        return entry.sha256 if entry else None

    def get_file_content(self, path: Path | str) -> Optional[str]:
        entry = self.get_file_entry(path, retain_content=True)
        if not entry:
            return None
        if entry.content is None:
            try:
                entry.content = Path(path).resolve().read_text(encoding="utf-8", errors="replace")
            except Exception:
                return None
        return entry.content

    def get_context_digest(self, paths: list[Path | str]) -> str:
        """
        Compute an aggregated SHA-256 digest of multiple context files.
        Order-independent deterministic representation.
        """
        hasher = hashlib.sha256()
        sorted_keys = sorted([self._normalize_key(p) for p in paths])
        for key in sorted_keys:
            entry = self.get_file_entry(key)
            file_hash = entry.sha256 if entry else "FILE_ABSENT"
            hasher.update(f"{key}:{file_hash};".encode("utf-8"))
        return hasher.hexdigest()

    def get_task_context(self, task: TaskDefinition, base_dir: Path) -> dict:
        """
        Gather pre-execution context for a task, returning file paths, contents, and digests.
        """
        context_files: dict[str, dict] = {}
        target_paths: list[Path] = []

        for rel in task.target_files:
            file_path = (base_dir / rel).resolve()
            target_paths.append(file_path)
            entry = self.get_file_entry(file_path, retain_content=True)
            if entry:
                context_files[rel] = {
                    "sha256": entry.sha256,
                    "size": entry.size,
                    "exists": True,
                }
            else:
                context_files[rel] = {
                    "sha256": None,
                    "size": 0,
                    "exists": False,
                }

        digest = self.get_context_digest(target_paths)
        return {
            "task_id": task.task_id,
            "target_files": context_files,
            "context_digest": digest,
        }
