"""
skills/maintenance/ledger.py
Durable task ledger for AutoPhage 3.0.
Handles atomic persistence, state transitions, and recovery evidence.
"""
from __future__ import annotations
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from models import TaskRisk


logger = logging.getLogger(__name__)

class LedgerValidationError(Exception):
    """Raised when a ledger entry fails structural or semantic validation."""
    pass

class DependencyCycleError(Exception):
    """Raised when a circular dependency is detected in the task ledger."""
    pass

class DuplicateTaskIdError(Exception):
    """Raised when a task is registered with an ID that already exists."""
    pass

class UnknownDependencyError(Exception):
    """Raised when a task refers to a dependency that does not exist in the ledger."""
    pass

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

@dataclass
class VerificationEvidence:
    command: str
    exit_status: int
    evidence_path: str
    source_fingerprint: str  # SHA-256 of modified files
    timestamp: str

@dataclass
class TaskEntry:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    worker: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    risk: TaskRisk = TaskRisk.LOW
    target_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    verification: Optional[VerificationEvidence] = None
    last_tool_call: Optional[dict[str, Any]] = None
    last_updated: str = ""
    failure_reason: Optional[str] = None
    checkpoint: Optional[CheckpointRecord] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "worker": self.worker,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "risk": self.risk.value if hasattr(self.risk, "value") else str(self.risk),
            "target_files": self.target_files,
            "dependencies": self.dependencies,
            "verification": (
                {
                    "command": self.verification.command,
                    "exit_status": self.verification.exit_status,
                    "evidence_path": self.verification.evidence_path,
                    "source_fingerprint": self.verification.source_fingerprint,
                    "timestamp": self.verification.timestamp,
                }
                if self.verification else None
            ),
            "last_tool_call": self.last_tool_call,
            "last_updated": self.last_updated,
            "failure_reason": self.failure_reason,
            "checkpoint": (
                {
                    "timestamp": self.checkpoint.timestamp,
                    "stage": self.checkpoint.stage,
                    "details": self.checkpoint.details,
                }
                if self.checkpoint else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskEntry:
        verif = data.get("verification")
        risk_val = data.get("risk", "LOW")
        try:
            risk = TaskRisk(risk_val)
        except ValueError:
            risk = TaskRisk.LOW
        cp_data = data.get("checkpoint")
        checkpoint = CheckpointRecord(**cp_data) if cp_data else None
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data.get("status", "PENDING")),
            worker=data.get("worker"),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            risk=risk,
            target_files=data.get("target_files", []),
            dependencies=data.get("dependencies", []),
            verification=VerificationEvidence(**verif) if verif else None,
            last_tool_call=data.get("last_tool_call"),
            last_updated=data.get("last_updated", ""),
            failure_reason=data.get("failure_reason"),
            checkpoint=checkpoint,
        )

class MaintenanceLedger:
    """
    Atomic, crash-safe ledger for AutoPhage 3.0.
    Persists to .maya/maintenance/maintenance_ledger.json.
    """
    def __init__(self, ledger_path: Path | str):
        self.ledger_path = Path(ledger_path).resolve()
        self.backup_path = self.ledger_path.with_suffix(".json.bak")
        self.events: list[dict[str, Any]] = []
        # Initialize cache from disk if it exists
        self._tasks = self.load()
        self._status_snapshot = {tid: t.status for tid, t in self._tasks.items()}


    def __iter__(self):
        """Allow iteration over tasks in the ledger."""
        return iter(self.load().values())

    def items(self):
        """Return an iterator over the loaded tasks."""
        return self.load().items()

    def get(self, task_id: str) -> Optional[TaskEntry]:
        """Retrieve a task by ID."""
        return self.load().get(task_id)

    def get_task(self, task_id: str) -> Optional[TaskEntry]:
        """Alias for get()."""
        return self.get(task_id)

    def update(self, task: TaskEntry) -> None:
        """Update a task in the ledger."""
        tasks = self.load()
        tasks[task.task_id] = task
        self.save(tasks)

    def __getitem__(self, key: str) -> TaskEntry:
        """Allow dict-like access to tasks."""
        res = self.get(key)
        if res is None:
            raise KeyError(key)
        return res










    def load(self) -> dict[str, TaskEntry]:
        if not self.ledger_path.exists():
            return {}
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {tid: TaskEntry.from_dict(v) for tid, v in data.items()}
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Ledger corrupted: %s. Attempting backup restore...", e)
            if self.backup_path.exists():
                try:
                    with open(self.backup_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return {tid: TaskEntry.from_dict(v) for tid, v in data.items()}
                except Exception:
                    pass
            return {}

    def save(self, tasks: dict[str, TaskEntry]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        data = {tid: t.to_dict() for tid, t in tasks.items()}

        # Backup current
        if self.ledger_path.exists():
            shutil.copy2(self.ledger_path, self.backup_path)

        # Atomic write
        tmp_file = self.ledger_path.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.ledger_path)
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    def validate_graph(self) -> None:
        """
        Placeholder for graph integrity validation.
        In a full implementation, this would check for cycles and missing dependencies.
        """
        pass

    def validate_status_integrity(self, *args, **kwargs) -> None:
        """
        Placeholder for status transition validation.
        """
        pass



    def to_dict(self) -> dict[str, Any]:
        """
        Serialize ledger state for persistence.
        """
        return {
            "tasks": {tid: t.to_dict() for tid, t in self.load().items()},
            "schema_version": "3.0",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], ledger_path: Optional[str] = None) -> MaintenanceLedger:
        """
        Create a ledger from a dictionary.
        """
        path = ledger_path or ".maya/maintenance/maintenance_ledger.json"
        ledger = cls(path)
        return ledger

    def quarantine_legacy_unintegrated_verified(self) -> None:
        """
        Placeholder for legacy quarantine logic.
        """
        pass

    def update_task(self, task_id: str, **kwargs) -> None:
        tasks = self.load()
        if task_id not in tasks:
            tasks[task_id] = TaskEntry(task_id=task_id)

        for k, v in kwargs.items():
            setattr(tasks[task_id], k, v)

        tasks[task_id].last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save(tasks)
        # Ensure in-memory cache and snapshot are updated to prevent Orchestrator state errors
        self._tasks[task_id] = tasks[task_id]
        self._status_snapshot[task_id] = tasks[task_id].status


# Alias for backward compatibility with AutoPhage 2.x (adaptive.py)
TaskLedger = MaintenanceLedger
