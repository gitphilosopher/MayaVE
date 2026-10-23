"""
Data models and type definitions for MayaVE Maintenance Coordinator.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class TaskPriority(str, enum.Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

    @property
    def rank(self) -> int:
        """Lower number = higher priority."""
        return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[self.value]


class TaskRisk(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFYING = "VERIFYING"
    READY_TO_INTEGRATE = "READY_TO_INTEGRATE"
    INTEGRATING = "INTEGRATING"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class PreValidationStatus(str, enum.Enum):
    CONFIRMED_UNPATCHED = "CONFIRMED_UNPATCHED"
    ALREADY_PATCHED = "ALREADY_PATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    TARGET_MISSING = "TARGET_MISSING"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BudgetState:
    """Persistent provider budget and token/cost accounting."""
    max_cloud_tasks: int = 10
    cloud_tasks_consumed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    currency: str = "USD"
    last_updated: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "max_cloud_tasks": self.max_cloud_tasks,
            "cloud_tasks_consumed": self.cloud_tasks_consumed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "currency": self.currency,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> BudgetState:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            max_cloud_tasks=int(data.get("max_cloud_tasks", 10)),
            cloud_tasks_consumed=int(data.get("cloud_tasks_consumed", 0)),
            total_input_tokens=int(data.get("total_input_tokens", 0)),
            total_output_tokens=int(data.get("total_output_tokens", 0)),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            currency=str(data.get("currency", "USD")),
            last_updated=str(data.get("last_updated", utc_now_iso())),
        )


@dataclass
class EvidenceRecord:
    timestamp: str
    stage: str
    details: str
    success: bool

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "details": self.details,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceRecord:
        return cls(
            timestamp=data["timestamp"],
            stage=data["stage"],
            details=data["details"],
            success=data["success"],
        )


@dataclass
class CheckpointRecord:
    timestamp: str
    stage: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointRecord:
        return cls(
            timestamp=data["timestamp"],
            stage=data["stage"],
            details=data.get("details", {}),
        )


@dataclass
class WorkspaceMetadata:
    worktree_path: str
    branch_name: str
    base_commit: str
    created_at: str = field(default_factory=utc_now_iso)
    is_dirty: bool = False
    head_commit: str = ""
    lifecycle_state: str = "PREPARED"  # PREPARED, ACTIVE, INTERRUPTED, VERIFIED, RECOVERY_REQUIRED

    def to_dict(self) -> dict:
        return {
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "base_commit": self.base_commit,
            "created_at": self.created_at,
            "is_dirty": self.is_dirty,
            "head_commit": self.head_commit,
            "lifecycle_state": self.lifecycle_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkspaceMetadata:
        return cls(
            worktree_path=data["worktree_path"],
            branch_name=data["branch_name"],
            base_commit=data.get("base_commit", ""),
            created_at=data.get("created_at", utc_now_iso()),
            is_dirty=data.get("is_dirty", False),
            head_commit=data.get("head_commit", ""),
            lifecycle_state=data.get("lifecycle_state", "PREPARED"),
        )


@dataclass
class PreValidationPattern:
    file_path: str
    pattern: str
    description: str

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "pattern": self.pattern,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PreValidationPattern:
        return cls(
            file_path=data["file_path"],
            pattern=data["pattern"],
            description=data.get("description", ""),
        )


@dataclass
class TaskDefinition:
    task_id: str
    title: str
    audit_source: str
    priority: TaskPriority
    risk: TaskRisk
    description: str
    evidence: str
    target_files: list[str]
    test_targets: list[str]
    read_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    verification_instructions: str = ""
    pre_validation_pattern: PreValidationPattern | None = None
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    history: list[EvidenceRecord] = field(default_factory=list)
    workspace: WorkspaceMetadata | None = None
    checkpoint: CheckpointRecord | None = None
    failure_reason: str | None = None
    blocking_reason: str | None = None
    integration_approval: dict | None = None
    allow_file_deletion: bool = False
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "audit_source": self.audit_source,
            "priority": self.priority.value,
            "risk": self.risk.value,
            "description": self.description,
            "evidence": self.evidence,
            "target_files": list(self.target_files),
            "test_targets": list(self.test_targets),
            "read_files": list(self.read_files),
            "dependencies": list(self.dependencies),
            "acceptance_criteria": list(self.acceptance_criteria),
            "verification_instructions": self.verification_instructions,
            "pre_validation_pattern": self.pre_validation_pattern.to_dict() if self.pre_validation_pattern else None,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "history": [rec.to_dict() for rec in self.history],
            "workspace": self.workspace.to_dict() if self.workspace else None,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "failure_reason": self.failure_reason,
            "blocking_reason": self.blocking_reason,
            "integration_approval": dict(self.integration_approval) if self.integration_approval else None,
            "allow_file_deletion": self.allow_file_deletion,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskDefinition:
        pattern_data = data.get("pre_validation_pattern")
        pattern = PreValidationPattern.from_dict(pattern_data) if pattern_data else None
        history_raw = data.get("history", [])
        history = [EvidenceRecord.from_dict(rec) for rec in history_raw]
        workspace_data = data.get("workspace")
        workspace = WorkspaceMetadata.from_dict(workspace_data) if workspace_data else None
        checkpoint_data = data.get("checkpoint")
        checkpoint = CheckpointRecord.from_dict(checkpoint_data) if checkpoint_data else None

        return cls(
            task_id=data["task_id"],
            title=data["title"],
            audit_source=data.get("audit_source", ""),
            priority=TaskPriority(data["priority"]),
            risk=TaskRisk(data.get("risk", "LOW")),
            description=data.get("description", ""),
            evidence=data.get("evidence", ""),
            target_files=list(data.get("target_files", [])),
            test_targets=list(data.get("test_targets", [])),
            read_files=list(data.get("read_files", [])),
            dependencies=list(data.get("dependencies", [])),
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            verification_instructions=data.get("verification_instructions", ""),
            pre_validation_pattern=pattern,
            status=TaskStatus(data.get("status", "PENDING")),
            attempts=int(data.get("attempts", 0)),
            max_attempts=max(3, int(data.get("max_attempts", 3))),
            history=history,
            workspace=workspace,
            checkpoint=checkpoint,
            failure_reason=data.get("failure_reason"),
            blocking_reason=data.get("blocking_reason"),
            integration_approval=dict(data["integration_approval"]) if data.get("integration_approval") else None,
            allow_file_deletion=bool(data.get("allow_file_deletion", False)),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
        )


@dataclass
class PlanProposal:
    """Structured plan proposal produced by LocalPlanner awaiting coordinator authorization."""
    plan_id: str
    objective: str
    tasks: list[dict]
    status: str = "PROPOSED"  # PROPOSED, APPROVED, REJECTED, REGISTERED
    proposal_hash: str = ""   # SHA-256 of canonical task/objective payload
    created_at: str = field(default_factory=utc_now_iso)
    approved_at: str | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    registered_task_ids: list[str] = field(default_factory=list)
    max_tasks: int = 5

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "tasks": [dict(t) for t in self.tasks],
            "status": self.status,
            "proposal_hash": self.proposal_hash,
            "created_at": self.created_at,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
            "rejection_reason": self.rejection_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "registered_task_ids": list(self.registered_task_ids),
            "max_tasks": self.max_tasks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanProposal:
        return cls(
            plan_id=data["plan_id"],
            objective=data["objective"],
            tasks=[dict(t) for t in data.get("tasks", [])],
            status=data.get("status", "PROPOSED"),
            proposal_hash=data.get("proposal_hash", ""),
            created_at=data.get("created_at", utc_now_iso()),
            approved_at=data.get("approved_at"),
            approved_by=data.get("approved_by"),
            rejection_reason=data.get("rejection_reason"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            registered_task_ids=list(data.get("registered_task_ids", [])),
            max_tasks=int(data.get("max_tasks", 5)),
        )

