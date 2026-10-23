"""
Local verified repair memory store for MayaVE Autophage (Milestone P6).

Provides an append-only JSONL outcome log recording only tasks that have passed
verification quality gates, with a bounded query interface for planner and router priors.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from .models import utc_now_iso
except ImportError:
    from models import utc_now_iso

logger = logging.getLogger("maya.maintenance.repair_memory")

# Sensitive token / secret redaction pattern
_SENSITIVE_PATTERNS = [
    re.compile(r"(api[_-]?key|token|secret|password|passwd|auth|bearer)\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"(bearer\s+)[A-Za-z0-9_\-\.]{16,}", re.IGNORECASE),
    re.compile(r"(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})", re.IGNORECASE),
]


def sanitize_repair_text(text: str, max_chars: int = 500) -> str:
    """Sanitize text to prevent credential/secret leakage and bound text length."""
    if not text or not isinstance(text, str):
        return ""
    clean = text.strip()
    for pat in _SENSITIVE_PATTERNS:
        clean = pat.sub(r"\1: [REDACTED]", clean)
    return clean[:max_chars].strip()


@dataclass
class RepairOutcome:
    """Immutable record of a task verification outcome."""
    task_id: str
    target_files: list[str]
    defect_category: str = "general"
    model_id: str = "local"
    patch_sha256: str = ""
    verification_sha256: str = ""
    test_targets: list[str] = field(default_factory=list)
    verification_passed: bool = True
    duration_sec: float = 0.0
    diff_summary: str = ""
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_files": sorted(self.target_files),
            "defect_category": self.defect_category,
            "model_id": self.model_id,
            "patch_sha256": self.patch_sha256,
            "verification_sha256": self.verification_sha256,
            "test_targets": sorted(self.test_targets),
            "verification_passed": self.verification_passed,
            "duration_sec": round(self.duration_sec, 2),
            "diff_summary": sanitize_repair_text(self.diff_summary),
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairOutcome:
        return cls(
            task_id=str(data.get("task_id", "")),
            target_files=list(data.get("target_files", [])),
            defect_category=str(data.get("defect_category", "general")),
            model_id=str(data.get("model_id", "local")),
            patch_sha256=str(data.get("patch_sha256", "")),
            verification_sha256=str(data.get("verification_sha256", "")),
            test_targets=list(data.get("test_targets", [])),
            verification_passed=bool(data.get("verification_passed", False)),
            duration_sec=float(data.get("duration_sec", 0.0)),
            diff_summary=str(data.get("diff_summary", "")),
            timestamp=str(data.get("timestamp", utc_now_iso())),
            metadata=dict(data.get("metadata", {})),
        )


class RepairMemoryStore:
    """
    Append-only local JSONL store for verified repair outcomes.
    Enforces that only verified outcomes are persisted, sanitizes records,
    and supports bounded similarity queries without network or external dependencies.
    """

    def __init__(
        self,
        memory_file_path: Path | str | None = None,
        max_entries: int = 1000,
    ) -> None:
        if memory_file_path:
            self.file_path = Path(memory_file_path).resolve()
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent
            self.file_path = (
                repo_root / ".maya" / "maintenance" / "repair_memory.jsonl"
            ).resolve()
        self.max_entries = max_entries

    def record_outcome(self, outcome: RepairOutcome) -> bool:
        """
        Record a repair outcome into the store.
        Strictly enforces that ONLY verified outcomes are recorded.
        Returns True if recorded, False if rejected or duplicate.
        """
        if not outcome or not outcome.task_id:
            return False

        # Invariant: Only record outcomes that passed verification
        if not outcome.verification_passed:
            logger.debug("Rejected unverified repair outcome for task %s", outcome.task_id)
            return False

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for duplicate (task_id, patch_sha256)
        if self.file_path.is_file():
            try:
                with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                            if (
                                rec.get("task_id") == outcome.task_id
                                and rec.get("patch_sha256") == outcome.patch_sha256
                                and outcome.patch_sha256 != ""
                            ):
                                return False  # Duplicate entry
                        except Exception:
                            continue
            except Exception as e:
                logger.warning("Error reading repair memory for deduplication: %s", e)

        # Append outcome atomically
        payload = json.dumps(outcome.to_dict()) + "\n"
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error("Failed to append to repair memory: %s", e)
            return False

        # Prune if exceeding max_entries
        self._prune_if_needed()
        return True

    def query_similar_repairs(
        self,
        target_files: list[str] | None = None,
        defect_category: str | None = None,
        limit: int = 5,
    ) -> list[RepairOutcome]:
        """
        Query verified repair history for similar target files or defect categories.
        Returns at most `limit` matches, ranked by relevance.
        Safely ignores malformed lines and fails closed on unreadable files.
        """
        if not self.file_path.is_file():
            return []

        search_files = set(target_files or [])
        clean_cat = (defect_category or "").lower().strip()

        candidates: list[tuple[int, RepairOutcome]] = []

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    try:
                        data = json.loads(clean_line)
                        outcome = RepairOutcome.from_dict(data)
                        if not outcome.verification_passed:
                            continue

                        # Calculate match score
                        score = 0
                        outcome_files = set(outcome.target_files)
                        common_files = search_files.intersection(outcome_files)
                        score += len(common_files) * 3

                        # Check category match
                        if clean_cat and outcome.defect_category.lower() == clean_cat:
                            score += 1

                        if score > 0 or (not search_files and not clean_cat):
                            candidates.append((score, outcome))
                    except Exception:
                        continue  # Skip corrupted lines safely
        except Exception as e:
            logger.warning("Failed to query repair memory: %s", e)
            return []

        # Sort by relevance descending, then by timestamp descending
        candidates.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return [c[1] for c in candidates[:limit]]

    def format_prompt_context(
        self,
        target_files: list[str] | None = None,
        defect_category: str | None = None,
        max_chars: int = 600,
    ) -> str:
        """
        Format a safe, concise, untrusted markdown snippet of prior repairs
        for optional inclusion in planner prompts.
        """
        repairs = self.query_similar_repairs(target_files=target_files, defect_category=defect_category, limit=3)
        if not repairs:
            return ""

        lines = [
            "### Prior Verified Maintenance Context (Untrusted Historical Evidence Only):",
        ]
        for r in repairs:
            files_str = ", ".join(r.target_files[:3])
            summary = sanitize_repair_text(r.diff_summary, max_chars=120)
            lines.append(f"- Task {r.task_id} ({r.defect_category}): Targets [{files_str}]. {summary}")

        formatted = "\n".join(lines)
        return formatted[:max_chars]

    def _prune_if_needed(self) -> None:
        """Prune oldest records if store exceeds max_entries."""
        if not self.file_path.is_file():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [line for line in f if line.strip()]
            if len(lines) > self.max_entries:
                kept = lines[-self.max_entries:]
                temp_file = self.file_path.with_name(f"{self.file_path.name}.tmp.{os.getpid()}")
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.writelines(kept)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_file, self.file_path)
        except Exception as e:
            logger.warning("Failed to prune repair memory: %s", e)
