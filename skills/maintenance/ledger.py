"""
Task Ledger implementation with cycle detection and validation.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Iterator

try:
    from .models import (
        BudgetState,
        CheckpointRecord,
        EvidenceRecord,
        TaskDefinition,
        TaskStatus,
        WorkspaceMetadata,
        utc_now_iso,
    )
except ImportError:
    from models import (
        BudgetState,
        CheckpointRecord,
        EvidenceRecord,
        TaskDefinition,
        TaskStatus,
        WorkspaceMetadata,
        utc_now_iso,
    )


class LedgerValidationError(Exception):
    """Raised when task or graph validation fails."""
    pass


class DuplicateTaskIdError(LedgerValidationError):
    """Raised when a duplicate task ID is encountered."""
    pass


class UnknownDependencyError(LedgerValidationError):
    """Raised when a task depends on a non-existent task ID."""
    pass


class DependencyCycleError(LedgerValidationError):
    """Raised when circular dependencies exist between tasks."""
    pass


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class TaskLedger:
    """In-memory collection of tasks with dependency and graph integrity checks."""

    def __init__(self, tasks: list[TaskDefinition] | None = None, *, record_creation_events: bool = True) -> None:
        self._tasks: dict[str, TaskDefinition] = {}
        self.events: list[dict] = []
        self._status_snapshot: dict[str, TaskStatus] = {}
        self.budget: BudgetState = BudgetState()
        if tasks:
            for task in tasks:
                self.add_task(task)
            self.validate_graph()
            if not record_creation_events:
                self.events = []

    def add_task(self, task: TaskDefinition) -> None:
        """Add a validated task to the ledger."""
        self._validate_task(task)
        if task.task_id in self._tasks:
            raise DuplicateTaskIdError(f"Duplicate task ID '{task.task_id}' already exists in ledger.")
        self._tasks[task.task_id] = task
        self._status_snapshot[task.task_id] = task.status
        if not any(event.get("idempotency_key") == f"task-created:{task.task_id}" for event in self.events):
            self.events.append({
            "event_id": str(uuid.uuid4()),
            "schema_version": 1,
            "task_id": task.task_id,
            "event_type": "TASK_CREATED",
            "previous_status": None,
            "resulting_status": task.status.value,
            "timestamp": task.created_at,
            "actor": "coordinator",
            "baseline_sha256": None,
            "artifact_sha256": None,
            "correlation_id": None,
            "idempotency_key": f"task-created:{task.task_id}",
            "outcome": "accepted",
            "error_code": None,
            "message": None,
            "details": {"title": task.title, "risk": task.risk.value},
            })

    def get_task(self, task_id: str) -> TaskDefinition:
        """Retrieve task by ID."""
        if task_id not in self._tasks:
            raise KeyError(f"Task ID '{task_id}' not found in ledger.")
        return self._tasks[task_id]

    def get(self, task_id: str, default: TaskDefinition | None = None) -> TaskDefinition | None:
        """Retrieve task by ID, returning default if not found."""
        return self._tasks.get(task_id, default)

    def update(self, task: TaskDefinition) -> None:
        """Update an existing task in the ledger."""
        self._validate_task(task)
        expected_status = self._status_snapshot.get(task.task_id)
        if expected_status is not None and task.status != expected_status:
            raise LedgerValidationError(
                f"Task '{task.task_id}' status changed outside CoordinatorAPI "
                f"({expected_status.value} -> {task.status.value})."
            )
        self._tasks[task.task_id] = task

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        failure_reason: str | None = None,
        blocking_reason: str | None = None,
    ) -> None:
        """Reject legacy status mutation; protected changes must use CoordinatorAPI."""
        raise LedgerValidationError(
            "Direct task status changes are disabled. Use CoordinatorAPI.transition_task()."
        )

    def record_checkpoint(
        self,
        task_id: str,
        stage: str,
        details: dict | None = None,
        status: TaskStatus | None = None,
    ) -> CheckpointRecord:
        """Record an explicit execution checkpoint with timestamp and stage."""
        task = self.get_task(task_id)
        cp = CheckpointRecord(
            timestamp=utc_now_iso(),
            stage=stage,
            details=details or {},
        )
        if status is not None and status != task.status:
            raise LedgerValidationError(
                "record_checkpoint cannot change task status; use CoordinatorAPI.transition_task()."
            )
        task.checkpoint = cp
        task.updated_at = utc_now_iso()
        task.history.append(
            EvidenceRecord(
                timestamp=cp.timestamp,
                stage=stage,
                details=json.dumps(details or {}),
                success=(status != TaskStatus.FAILED and status != TaskStatus.RECOVERY_REQUIRED),
            )
        )
        return cp

    def update_workspace(
        self,
        task_id: str,
        workspace: WorkspaceMetadata,
    ) -> None:
        """Attach or update workspace metadata on a task."""
        task = self.get_task(task_id)
        task.workspace = workspace
        task.updated_at = utc_now_iso()

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[TaskDefinition]:
        return iter(self._tasks.values())

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def keys(self) -> list[str]:
        return list(self._tasks.keys())

    def values(self) -> list[TaskDefinition]:
        return list(self._tasks.values())

    def items(self) -> list[tuple[str, TaskDefinition]]:
        return list(self._tasks.items())

    @staticmethod
    def _validate_task(task: TaskDefinition) -> None:
        """Validate single task field rules."""
        if not task.task_id or not _TASK_ID_RE.match(task.task_id):
            raise LedgerValidationError(
                f"Invalid task_id '{task.task_id}'. Must be non-empty alphanumeric (dashes/underscores allowed)."
            )
        if not task.title.strip():
            raise LedgerValidationError(f"Task '{task.task_id}' has empty title.")
        if task.attempts < 0:
            raise LedgerValidationError(f"Task '{task.task_id}' has negative attempt count {task.attempts}.")
        if task.max_attempts <= 0:
            raise LedgerValidationError(f"Task '{task.task_id}' has invalid max_attempts {task.max_attempts}.")

    def validate_graph(self) -> None:
        """Validate dependency graph for unknown nodes and circular dependencies."""
        # 1. Unknown dependencies
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    raise UnknownDependencyError(
                        f"Task '{task.task_id}' depends on unknown task '{dep_id}'."
                    )
                if dep_id == task.task_id:
                    raise DependencyCycleError(
                        f"Task '{task.task_id}' cannot depend on itself."
                    )

        # 2. Cycle detection via DFS
        # 0 = unvisited, 1 = visiting (in current recursion stack), 2 = visited
        visited: dict[str, int] = {t_id: 0 for t_id in self._tasks}

        def _dfs(node: str, path: list[str]) -> None:
            visited[node] = 1
            task = self._tasks[node]
            for dep in task.dependencies:
                if visited[dep] == 1:
                    cycle = " -> ".join(path + [dep])
                    raise DependencyCycleError(f"Dependency cycle detected: {cycle}")
                if visited[dep] == 0:
                    _dfs(dep, path + [dep])
            visited[node] = 2

        for t_id in self._tasks:
            if visited[t_id] == 0:
                _dfs(t_id, [t_id])

    def to_dict(self) -> dict:
        self.validate_status_integrity()
        return {
            "schema_version": 1,
            "tasks": {task_id: task.to_dict() for task_id, task in self._tasks.items()},
            "events": list(self.events[-5000:]),
            "budget": self.budget.to_dict(),
        }

    def validate_status_integrity(self) -> None:
        """Detect mutable TaskDefinition.status edits that bypass the coordinator."""
        for task_id, task in self._tasks.items():
            expected = self._status_snapshot.get(task_id)
            if expected is not None and task.status != expected:
                raise LedgerValidationError(
                    f"Task '{task_id}' status was modified outside CoordinatorAPI "
                    f"({expected.value} -> {task.status.value})."
                )

    def _commit_status(self, task_id: str, status: TaskStatus) -> None:
        """Internal snapshot update used only after the coordinator persists a transition."""
        self._status_snapshot[task_id] = status

    def quarantine_legacy_unintegrated_verified(self) -> list[str]:
        """Fail closed on persisted legacy VERIFIED tasks without integration evidence."""
        quarantined: list[str] = []
        for task in self._tasks.values():
            if task.status != TaskStatus.VERIFIED:
                continue
            stage = task.checkpoint.stage if task.checkpoint else None
            if stage in ("INTEGRATED", "POST_INTEGRATION_VERIFIED"):
                continue
            previous = task.status
            task.status = TaskStatus.RECOVERY_REQUIRED
            task.blocking_reason = "Legacy VERIFIED state has no post-integration verification checkpoint; human recovery review is required."
            self._status_snapshot[task.task_id] = task.status
            self.events.append({
                "event_id": str(uuid.uuid4()),
                "schema_version": 1,
                "task_id": task.task_id,
                "event_type": "LEGACY_VERIFICATION_QUARANTINED",
                "previous_status": previous.value,
                "resulting_status": task.status.value,
                "timestamp": utc_now_iso(),
                "actor": "state-migration",
                "baseline_sha256": None,
                "artifact_sha256": None,
                "correlation_id": None,
                "idempotency_key": f"legacy-verified-quarantine:{task.task_id}",
                "outcome": "accepted",
                "error_code": "MISSING_INTEGRATION_EVIDENCE",
                "message": task.blocking_reason,
                "details": {"legacy_checkpoint_stage": stage},
            })
            quarantined.append(task.task_id)
        self.events = self.events[-5000:]
        return quarantined

    @classmethod
    def from_dict(cls, data: dict) -> TaskLedger:
        if not isinstance(data, dict):
            raise LedgerValidationError("Ledger payload must be a JSON dictionary.")
        version = data.get("schema_version")
        if version != 1:
            raise LedgerValidationError(f"Unsupported schema version '{version}'. Expected version 1.")
        tasks_data = data.get("tasks", {})
        if not isinstance(tasks_data, dict):
            raise LedgerValidationError("'tasks' field must be a dictionary.")

        tasks = []
        for task_id, t_dict in tasks_data.items():
            if not isinstance(t_dict, dict):
                raise LedgerValidationError(f"Task record for '{task_id}' must be a dictionary.")
            if t_dict.get("task_id") != task_id:
                raise LedgerValidationError(
                    f"Key '{task_id}' does not match task_id '{t_dict.get('task_id')}' in record."
                )
            tasks.append(TaskDefinition.from_dict(t_dict))

        ledger = cls(tasks, record_creation_events=False)
        events = data.get("events", [])
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise LedgerValidationError("'events' field must be a list of event dictionaries.")
        ledger.events = list(events[-5000:])
        ledger.budget = BudgetState.from_dict(data.get("budget"))
        return ledger
