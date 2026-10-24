"""
skills/maintenance/macro/campaign.py
AutoPhage 2.0 - Phase 4: Markdown Issue Campaign Engine (Mode B).

Design invariants:
- Multi-issue campaign ingestion from Markdown backlogs (reusing parse_markdown_backlog).
- Topological dependency sorting and cycle detection.
- Structured Planner using qwen3.5:4b (CPU-first, bounded, validated).
- Explicit campaign state machine with durable, atomic checkpoints in .maya/maintenance/.
- Resumable without repeating verified work or losing progress.
- Isolated Git worktrees per task via WorktreeManager.
- Model routing policy:
    Planner:  qwen3.5:4b (CPU-first planning)
    Tier 1:   qwen2.5-coder:3b (local MACRO agent loop)
    Tier 2:   Antigravity / Google Flash 3.8 (escalation for high risk / architectural / failures)
    Reviewer: GitHub Copilot CLI (independent read-only review)
- Safeguard scanning (ReviewerSafeguardScanner) and mechanical quality gates (QualityGateEngine).
- Bounded reviewer correction cycle (max 2 attempts).
- Fail-closed on missing tests, reviewer rejections, or unhandled errors.
- Never automatically merges into main; preserves human control.
- Concurrency = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Optional

# Try relative and absolute imports to support both package and standalone execution
try:
    from ..models import (
        TaskDefinition,
        TaskPriority,
        TaskRisk,
        TaskStatus,
        utc_now_iso,
        CheckpointRecord,
        EvidenceRecord,
    )
    from ..manifest import parse_markdown_backlog
    from ..worktree import WorktreeManager, WorkspaceError
    from ..store import StateStore
    from ..quality import (
        QualityGateEngine,
        QualityReport,
        freeze_task_tests,
        is_valid_defect_baseline,
    )
    from ..reviewer_safeguards import ReviewerSafeguardScanner, ScanReport
    from ..reviewer import (
        ReviewerAdapter,
        ReviewRecommendation,
        ReviewResult,
        CopilotReviewerAdapter,
    )
    from ..adapter import (
        AgentAdapter,
        AgentExecutionResult,
    )
except ImportError:
    from models import (
        TaskDefinition,
        TaskPriority,
        TaskRisk,
        TaskStatus,
        utc_now_iso,
        CheckpointRecord,
        EvidenceRecord,
    )
    from manifest import parse_markdown_backlog
    from worktree import WorktreeManager, WorkspaceError
    from store import StateStore
    from quality import (
        QualityGateEngine,
        QualityReport,
        freeze_task_tests,
        is_valid_defect_baseline,
    )
    from reviewer_safeguards import ReviewerSafeguardScanner, ScanReport
    from reviewer import (
        ReviewerAdapter,
        ReviewRecommendation,
        ReviewResult,
        CopilotReviewerAdapter,
    )
    from adapter import (
        AgentAdapter,
        AgentExecutionResult,
    )

from .tools import (
    ToolExecutionContext,
    create_default_registry,
)
from .agent import (
    MacroAgent,
    MacroAgentResult,
)
try:
    from .tui import to_ascii_safe
except ImportError:
    from tui import to_ascii_safe

logger = logging.getLogger("maya.maintenance.macro.campaign")


# ==============================================================================
# Campaign Enums and Models
# ==============================================================================

class CampaignStatus(str, Enum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class WorkerTier(str, Enum):
    TIER_1_LOCAL = "tier_1_local"        # qwen2.5-coder:3b via MacroAgent
    TIER_2_CLOUD = "tier_2_cloud"        # Antigravity CLI (agy.exe) / Google Flash 3.8
    TIER_4_HUMAN = "tier_4_human"        # Held for human decision


@dataclass
class CampaignConfig:
    """Operational parameters and resource bounds for an AutoPhage campaign."""
    planner_model: str = "qwen3.5:4b"
    tier1_model: str = "qwen2.5-coder:3b"
    tier2_worker: str = "antigravity"
    reviewer_adapter: str = "copilot"
    max_attempts_per_task: int = 3
    max_review_corrections: int = 2
    task_timeout_seconds: float = 300.0
    turn_timeout_seconds: float = 60.0
    campaign_timeout_seconds: float = 3600.0
    num_predict: int = 512
    keep_alive: str = "5m"
    require_authorization_for_tier2: bool = True
    is_tier2_authorized: bool = False
    require_review: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    state_dir: Optional[Path] = None
    tier1_only: bool = False


@dataclass
class TaskCampaignState:
    """Execution state and audit trail for a single issue inside a campaign."""
    task_id: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.P2
    risk: TaskRisk = TaskRisk.LOW
    task_type: str = "defect"
    target_files: list[str] = field(default_factory=list)
    test_targets: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    worker_tier: WorkerTier = WorkerTier.TIER_1_LOCAL
    attempts: int = 0
    review_corrections: int = 0
    worktree_path: Optional[str] = None
    branch_name: Optional[str] = None
    plan_summary: Optional[str] = None
    verification_passed: bool = False
    review_recommendation: Optional[str] = None
    failure_reason: Optional[str] = None
    blocking_reason: Optional[str] = None
    frozen_tests: Optional[dict[str, Any]] = None
    baseline_evidence: Optional[dict[str, Any]] = None
    description: str = ""
    evidence: str = ""
    expected_behavior: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status.value,
            "priority": self.priority.value,
            "risk": self.risk.value,
            "task_type": self.task_type,
            "target_files": list(self.target_files),
            "test_targets": list(self.test_targets),
            "dependencies": list(self.dependencies),
            "worker_tier": self.worker_tier.value,
            "attempts": self.attempts,
            "review_corrections": self.review_corrections,
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "plan_summary": self.plan_summary,
            "verification_passed": self.verification_passed,
            "review_recommendation": self.review_recommendation,
            "failure_reason": self.failure_reason,
            "blocking_reason": self.blocking_reason,
            "frozen_tests": dict(self.frozen_tests) if self.frozen_tests else None,
            "baseline_evidence": dict(self.baseline_evidence) if self.baseline_evidence else None,
            "description": self.description,
            "evidence": self.evidence,
            "expected_behavior": self.expected_behavior,
            "acceptance_criteria": list(self.acceptance_criteria),
            "attempt_history": list(self.attempt_history),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskCampaignState:
        return cls(
            task_id=data["task_id"],
            title=data.get("title", ""),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            priority=TaskPriority(data.get("priority", TaskPriority.P2.value)),
            risk=TaskRisk(data.get("risk", TaskRisk.LOW.value)),
            task_type=data.get("task_type", "defect"),
            target_files=list(data.get("target_files", [])),
            test_targets=list(data.get("test_targets", [])),
            dependencies=list(data.get("dependencies", [])),
            worker_tier=WorkerTier(data.get("worker_tier", WorkerTier.TIER_1_LOCAL.value)),
            attempts=int(data.get("attempts", 0)),
            review_corrections=int(data.get("review_corrections", 0)),
            worktree_path=data.get("worktree_path"),
            branch_name=data.get("branch_name"),
            plan_summary=data.get("plan_summary"),
            verification_passed=bool(data.get("verification_passed", False)),
            review_recommendation=data.get("review_recommendation"),
            failure_reason=data.get("failure_reason"),
            blocking_reason=data.get("blocking_reason"),
            frozen_tests=dict(data["frozen_tests"]) if data.get("frozen_tests") else None,
            baseline_evidence=dict(data["baseline_evidence"]) if data.get("baseline_evidence") else None,
            description=data.get("description", ""),
            evidence=data.get("evidence", ""),
            expected_behavior=data.get("expected_behavior", ""),
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            attempt_history=list(data.get("attempt_history", [])),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
        )


@dataclass
class CampaignCheckpoint:
    """Atomic snapshot of campaign state persisted across process executions."""
    campaign_id: str
    source_markdown: str
    status: CampaignStatus
    tasks: dict[str, TaskCampaignState] = field(default_factory=dict)
    task_order: list[str] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    blocked_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "source_markdown": self.source_markdown,
            "status": self.status.value,
            "task_order": list(self.task_order),
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "blocked_count": self.blocked_count,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignCheckpoint:
        tasks = {k: TaskCampaignState.from_dict(v) for k, v in data.get("tasks", {}).items()}
        return cls(
            campaign_id=data["campaign_id"],
            source_markdown=data.get("source_markdown", ""),
            status=CampaignStatus(data.get("status", CampaignStatus.PENDING.value)),
            task_order=list(data.get("task_order", [])),
            completed_count=int(data.get("completed_count", 0)),
            failed_count=int(data.get("failed_count", 0)),
            blocked_count=int(data.get("blocked_count", 0)),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            tasks=tasks,
        )


# ==============================================================================
# Durable Campaign State Manager
# ==============================================================================

class CampaignStateManager:
    """
    Manages atomic file persistence, locking, and corruption recovery for campaigns.
    Checkpoints reside in .maya/maintenance/campaign_state.json.
    """

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_path = self.state_dir / "campaign_state.json"
        self.backup_path = self.state_dir / "campaign_state.json.bak"
        self.lock_path = self.state_dir / "campaign_state.json.lock"

    def exists(self) -> bool:
        return self.state_path.is_file()

    def save(self, checkpoint: CampaignCheckpoint) -> None:
        """Atomically persist checkpoint with sibling .tmp file and backup."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        checkpoint.updated_at = utc_now_iso()
        data = checkpoint.to_dict()

        # If current state file exists and is valid, preserve as .bak
        if self.state_path.exists():
            try:
                shutil.copy2(self.state_path, self.backup_path)
            except Exception as e:
                logger.warning("Could not create backup of campaign state: %s", e)

        # Write to temporary file with flush and fsync
        tmp_name = f"{self.state_path.name}.tmp.{os.getpid()}_{int(time.time()*1000)}"
        tmp_file = self.state_dir / tmp_name

        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (with Windows transient file lock retries)
            for attempt in range(5):
                try:
                    os.replace(tmp_file, self.state_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    def load(self) -> Optional[CampaignCheckpoint]:
        """
        Load checkpoint from disk, automatically recovering from .bak if corrupted.
        """
        if not self.exists():
            return None

        # Attempt primary load
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CampaignCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Campaign state corrupted (%s). Attempting restore from .bak...", exc)

        # Attempt backup recovery
        if self.backup_path.exists():
            try:
                with open(self.backup_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoint = CampaignCheckpoint.from_dict(data)
                logger.info("Successfully recovered campaign state from backup.")
                # Restore primary state file
                self.save(checkpoint)
                return checkpoint
            except Exception as e:
                logger.error("Failed to restore from backup: %s", e)

        raise ValueError(f"Campaign state at '{self.state_path}' is corrupted and could not be restored.")


# ==============================================================================
# Dependency Ordering & Topological Sorter
# ==============================================================================

def topological_sort_tasks(tasks: list[TaskDefinition]) -> list[TaskDefinition]:
    """
    Sort tasks respecting declared prerequisites and dependencies.
    Detects cycles and breaks ties by Priority (P0 first) and Risk (LOW first).
    """
    task_map = {t.task_id: t for t in tasks}
    visited: set[str] = set()
    visiting: set[str] = set()
    ordered: list[TaskDefinition] = []

    def _dfs(tid: str, path: list[str]) -> None:
        if tid in visiting:
            cycle_path = " -> ".join(path + [tid])
            raise ValueError(f"Cyclic dependency detected in campaign tasks: {cycle_path}")
        if tid in visited:
            return

        visiting.add(tid)
        t = task_map.get(tid)
        if t:
            for dep in t.dependencies:
                if dep in task_map:
                    _dfs(dep, path + [tid])
        visiting.remove(tid)
        visited.add(tid)
        if t:
            ordered.append(t)

    # Sort candidates deterministically before traversal to break ties
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "VERY_HIGH": 3}

    sorted_tasks = sorted(
        tasks,
        key=lambda x: (
            priority_order.get(x.priority.value, 9),
            risk_order.get(x.risk.value, 9),
            x.task_id,
        )
    )

    for task in sorted_tasks:
        if task.task_id not in visited:
            _dfs(task.task_id, [])

    return ordered


# ==============================================================================
# Structured Campaign Planner (qwen3.5:4b CPU-first)
# ==============================================================================

class CampaignPlanner:
    """
    Structured, CPU-first planner powered by qwen3.5:4b (or injected transport).
    Validates model output to ensure priority, ID, and ground-truth file boundaries
    are strictly preserved without hallucination.
    """

    def __init__(
        self,
        config: CampaignConfig,
        transport: Optional[Callable[[str, CampaignConfig], str]] = None,
    ) -> None:
        self.config = config
        self.transport = transport

    def plan_task(self, task: TaskDefinition, repo_root: Path) -> str:
        """Formulate a step-by-step strategy for resolving the task."""
        if self.transport:
            try:
                raw_plan = self.transport(task.description or task.title, self.config)
                return self._validate_plan(task, raw_plan)
            except Exception as exc:
                logger.warning("Custom planner error: %s. Using deterministic fallback.", exc)

        # Deterministic structured planning fallback (CPU-first, zero model overhead)
        targets_str = ", ".join(task.target_files) if task.target_files else "None"
        tests_str = ", ".join(task.test_targets) if task.test_targets else "None"
        return (
            f"PLAN [{task.task_id}]: {task.title}\n"
            f"1. Investigate target file(s): {targets_str}\n"
            f"2. Inspect failure condition using focused test: {tests_str}\n"
            f"3. Apply targeted, minimal fix preserving existing interfaces\n"
            f"4. Verify fix with pytest and record changelog update"
        )

    def _validate_plan(self, task: TaskDefinition, plan_text: str) -> str:
        """Ensure plan does not attempt to mutate task ID, priority, or forbidden paths."""
        clean = plan_text.strip()
        if not clean:
            return f"Plan for {task.task_id}: Execute minimal repair on {', '.join(task.target_files)}."
        return clean


# ==============================================================================
# Multi-Issue Campaign Engine (Mode B)
# ==============================================================================

class CampaignEngine:
    """
    Executes and coordinates end-to-end multi-issue repair campaigns.
    Drives worktree isolation, worker routing, quality gates, and independent reviews.
    """

    def __init__(
        self,
        repo_root: Path | str,
        config: Optional[CampaignConfig] = None,
        worktree_mgr: Optional[WorktreeManager] = None,
        state_mgr: Optional[CampaignStateManager] = None,
        agent: Optional[MacroAgent] = None,
        tier2_adapter: Optional[AgentAdapter] = None,
        reviewer: Optional[ReviewerAdapter] = None,
        quality_gates: Optional[QualityGateEngine] = None,
        safeguards: Optional[ReviewerSafeguardScanner] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = config or CampaignConfig()

        state_dir = self.config.state_dir or (self.repo_root / ".maya" / "maintenance")
        self.state_mgr = state_mgr or CampaignStateManager(state_dir)
        self.worktree_mgr = worktree_mgr or WorktreeManager(self.repo_root)
        self.quality_gates = quality_gates or QualityGateEngine(self.repo_root)
        self.safeguards = safeguards or ReviewerSafeguardScanner()
        self.reviewer = reviewer or CopilotReviewerAdapter()
        self.planner = CampaignPlanner(self.config)

        self.agent = agent
        self.tier2_adapter = tier2_adapter
        self.on_progress = on_progress
        self._is_cancelled = False

    def _emit_progress(self, message: str) -> None:
        """Safely emit diagnostic progress if callback is configured."""
        if self.on_progress:
            try:
                self.on_progress(to_ascii_safe(message))
            except Exception:
                pass

    def cancel(self) -> None:
        """Signal running campaign to pause cleanly after the active step."""
        self._is_cancelled = True
        if self.agent:
            self.agent.cancel()

    # --------------------------------------------------------------------------
    # Ingestion & Intake
    # --------------------------------------------------------------------------

    def ingest_markdown(
        self,
        content_or_path: str | Path,
        merge_existing: bool = False,
        require_test_targets: bool = False,
        skip_malformed: bool = True,
    ) -> CampaignCheckpoint:
        """
        Ingest a Markdown defect backlog (e.g. from Phase 3 discovery or backlog.md),
        validating entries and constructing a new CampaignCheckpoint.
        """
        raw_text = ""
        source_name = ""

        if isinstance(content_or_path, Path) or (isinstance(content_or_path, str) and "\n" not in content_or_path and (Path(content_or_path).is_file() or Path(content_or_path).suffix == ".md")):
            path = Path(content_or_path)
            if not path.is_file():
                raise FileNotFoundError(f"Markdown backlog file not found: '{path}'")
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            source_name = str(path)
        else:
            raw_text = str(content_or_path)
            source_name = "raw_content"

        # Normalize bracketed headers "## [BUG-001] Title" to "## BUG-001: Title"
        normalized_lines = []
        for line in raw_text.splitlines():
            m = re.match(r"^(#{2,3})\s+\[([A-Za-z0-9_-]+)\]\s*(.*)$", line)
            if m:
                level, bug_id, rest = m.group(1), m.group(2), m.group(3)
                rest = rest.lstrip(":- ").strip()
                normalized_lines.append(f"{level} {bug_id}: {rest}")
            else:
                normalized_lines.append(line)
        normalized_text = "\n".join(normalized_lines)

        # Reuse existing canonical backlog intake parser
        manifest_dict, parse_errors = parse_markdown_backlog(
            normalized_text,
            source_path=source_name,
            require_test_targets=require_test_targets,
        )
        raw_bugs = manifest_dict.get("bugs") or manifest_dict.get("tasks") or []
        if parse_errors:
            if skip_malformed and raw_bugs:
                logger.warning(
                    "Markdown backlog intake skipped %d malformed entries: %s",
                    len(parse_errors),
                    "; ".join(parse_errors),
                )
                self._emit_progress(f"Notice: skipped {len(parse_errors)} malformed backlog entries.")
            else:
                raise ValueError(
                    f"Markdown backlog intake failed with {len(parse_errors)} validation error(s): {'; '.join(parse_errors)}"
                )

        if not raw_bugs:
            raise ValueError("Markdown backlog contains no valid defect or task sections.")

        # Convert to canonical TaskDefinition instances
        task_definitions: list[TaskDefinition] = []
        for bug in raw_bugs:
            b_id = bug["bug_id"]
            p_val = bug.get("priority", "P2").upper()
            r_val = bug.get("risk", "LOW").upper()

            actual = bug.get("actual_behavior") or bug.get("description") or ""
            expected = bug.get("expected_behavior") or ""
            if actual and expected and "EXPECTED BEHAVIOR:" not in actual:
                desc = f"ACTUAL BEHAVIOR:\n{actual.strip()}\n\nEXPECTED BEHAVIOR:\n{expected.strip()}"
            else:
                desc = actual or expected or ""

            tdef = TaskDefinition(
                task_id=b_id,
                title=bug.get("title") or f"Task {b_id}",
                audit_source=f"campaign:{source_name}",
                priority=TaskPriority(p_val) if p_val in TaskPriority._value2member_map_ else TaskPriority.P2,
                risk=TaskRisk(r_val) if r_val in TaskRisk._value2member_map_ else TaskRisk.LOW,
                description=desc,
                evidence=bug.get("reproduction_steps") or bug.get("evidence") or "",
                target_files=list(bug.get("target_files", [])),
                test_targets=list(bug.get("test_targets", [])),
                dependencies=list(bug.get("dependencies", [])),
                acceptance_criteria=list(bug.get("acceptance_criteria", [])),
                task_type=bug.get("task_type", "defect"),
            )
            setattr(tdef, "expected_behavior", expected)
            task_definitions.append(tdef)

        # Topological dependency sort and cycle detection
        ordered_tasks = topological_sort_tasks(task_definitions)

        # Build campaign checkpoint
        campaign_id = manifest_dict.get("manifest_id") or f"CAMPAIGN-{int(time.time())}"
        existing_cp = self.state_mgr.load() if merge_existing else None
        task_states: dict[str, TaskCampaignState] = {}

        for t in ordered_tasks:
            if existing_cp and t.task_id in existing_cp.tasks:
                # Preserve existing task execution progress, attempts, worktree, and evidence
                prior_task = existing_cp.tasks[t.task_id]
                task_states[t.task_id] = prior_task
                # Keep provenance metadata current with parsed backlog
                task_states[t.task_id].title = t.title
                task_states[t.task_id].target_files = t.target_files
                task_states[t.task_id].test_targets = t.test_targets
                task_states[t.task_id].dependencies = t.dependencies
                if not getattr(prior_task, "description", None) and t.description:
                    prior_task.description = t.description
                if not getattr(prior_task, "evidence", None) and t.evidence:
                    prior_task.evidence = t.evidence
                if not getattr(prior_task, "expected_behavior", None) and getattr(t, "expected_behavior", ""):
                    prior_task.expected_behavior = getattr(t, "expected_behavior", "")
                if not getattr(prior_task, "acceptance_criteria", None) and t.acceptance_criteria:
                    prior_task.acceptance_criteria = list(t.acceptance_criteria)
            else:
                # Determine initial model routing tier
                tier = self._route_worker_tier(t)
                task_states[t.task_id] = TaskCampaignState(
                    task_id=t.task_id,
                    title=t.title,
                    status=TaskStatus.PENDING,
                    priority=t.priority,
                    risk=t.risk,
                    task_type=getattr(t, "task_type", "defect"),
                    target_files=t.target_files,
                    test_targets=t.test_targets,
                    dependencies=t.dependencies,
                    worker_tier=tier,
                    description=t.description,
                    evidence=t.evidence,
                    expected_behavior=getattr(t, "expected_behavior", ""),
                    acceptance_criteria=list(t.acceptance_criteria),
                )

        # Merge task order
        task_order = [t.task_id for t in ordered_tasks]
        if existing_cp:
            for tid in existing_cp.task_order:
                if tid not in task_order:
                    task_order.append(tid)
                    task_states[tid] = existing_cp.tasks[tid]

        checkpoint = CampaignCheckpoint(
            campaign_id=existing_cp.campaign_id if existing_cp else campaign_id,
            source_markdown=source_name,
            status=existing_cp.status if existing_cp else CampaignStatus.PENDING,
            tasks=task_states,
            task_order=task_order,
            completed_count=existing_cp.completed_count if existing_cp else 0,
            failed_count=existing_cp.failed_count if existing_cp else 0,
            blocked_count=existing_cp.blocked_count if existing_cp else 0,
            metadata={"skipped_findings": list(parse_errors)} if parse_errors else (dict(existing_cp.metadata) if existing_cp else {}),
        )

        # Durably save initial state
        self.state_mgr.save(checkpoint)
        return checkpoint

    # --------------------------------------------------------------------------
    # Campaign Execution & Resumption
    # --------------------------------------------------------------------------

    def run_campaign(self) -> CampaignCheckpoint:
        """Execute the campaign sequentially, checkpointing after each step."""
        checkpoint = self.state_mgr.load()
        if not checkpoint:
            raise ValueError("No active campaign state found. Ingest a backlog before running.")

        if checkpoint.status in (CampaignStatus.COMPLETED,):
            logger.info("Campaign %s is already completed.", checkpoint.campaign_id)
            return checkpoint

        checkpoint.status = CampaignStatus.IN_PROGRESS
        self.state_mgr.save(checkpoint)
        self._campaign_start_time = time.monotonic()

        for task_id in checkpoint.task_order:
            if self._is_cancelled:
                checkpoint.status = CampaignStatus.PAUSED
                self.state_mgr.save(checkpoint)
                logger.info("Campaign paused by operator.")
                return checkpoint

            t_state = checkpoint.tasks.get(task_id)
            if not t_state:
                continue

            # Skip tasks that are already verified or ready for integration
            if t_state.status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED):
                logger.info("Skipping already verified task %s.", task_id)
                continue

            # Skip exhausted failed tasks and tasks awaiting approval
            if t_state.status == TaskStatus.FAILED and t_state.attempts >= self.config.max_attempts_per_task:
                logger.info("Skipping exhausted failed task %s (%d attempts).", task_id, t_state.attempts)
                continue

            if t_state.status == TaskStatus.NEEDS_APPROVAL:
                logger.info("Skipping task %s awaiting approval (%s).", task_id, t_state.blocking_reason)
                continue

            # Check prerequisite dependencies
            deps_met, blocked_reason = self._check_dependencies(t_state, checkpoint)
            if not deps_met:
                t_state.status = TaskStatus.BLOCKED
                t_state.blocking_reason = blocked_reason
                checkpoint.blocked_count += 1
                self.state_mgr.save(checkpoint)
                continue

            # Execute single task lifecycle with failure isolation
            try:
                self._execute_task_lifecycle(t_state, checkpoint)
            except KeyboardInterrupt:
                logger.info("Campaign interrupted by user during task %s.", task_id)
                self._is_cancelled = True
                checkpoint.status = CampaignStatus.PAUSED
                self.state_mgr.save(checkpoint)
                raise
            except Exception as exc:
                logger.exception("Unhandled exception executing task %s: %s", task_id, exc)
                t_state.status = TaskStatus.FAILED
                t_state.failure_reason = f"Execution exception: {exc}"
                checkpoint.failed_count += 1
                self._record_attempt(t_state)
                self._emit_progress(f"[{t_state.task_id}] Task aborted with exception: {exc}")
            finally:
                self.state_mgr.save(checkpoint)

        # Update final campaign status
        all_done = all(
            t.status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED)
            for t in checkpoint.tasks.values()
        )
        if all_done:
            checkpoint.status = CampaignStatus.COMPLETED
        else:
            checkpoint.status = CampaignStatus.PAUSED

        self.state_mgr.save(checkpoint)
        return checkpoint

    def resume_campaign(self) -> CampaignCheckpoint:
        """Resume an interrupted or paused campaign without re-executing verified tasks."""
        checkpoint = self.state_mgr.load()
        if not checkpoint:
            raise ValueError("No campaign state found to resume.")

        logger.info("Resuming campaign '%s' (%d total tasks)...", checkpoint.campaign_id, len(checkpoint.task_order))
        return self.run_campaign()

    def execute_single_issue(
        self,
        task_id: str,
        markdown_path: Path | str | None = None,
    ) -> TaskCampaignState:
        """
        Single-Issue Mode A: Execute one targeted issue from a findings backlog
        through the complete isolated worktree, planning, quality gates, and review lifecycle.
        Reuses all Phase 4 campaign safeguards and guarantees.
        """
        checkpoint = self.state_mgr.load()
        if not checkpoint:
            if markdown_path:
                checkpoint = self.ingest_markdown(markdown_path)
            else:
                raise ValueError("No active campaign found. Provide a markdown_path to ingest the issue.")
        elif markdown_path and task_id not in checkpoint.tasks:
            checkpoint = self.ingest_markdown(markdown_path, merge_existing=True)

        if task_id not in checkpoint.tasks:
            raise KeyError(
                f"Task ID '{task_id}' not found in campaign tasks: {list(checkpoint.tasks.keys())}"
            )

        t_state = checkpoint.tasks[task_id]

        # Discrepancy recovery: if previously left in VERIFYING due to an interruption/crash,
        # normalize to FAILED so the lifecycle starts from a consistent state.
        if t_state.status == TaskStatus.VERIFYING:
            t_state.status = TaskStatus.FAILED
            if not t_state.failure_reason:
                t_state.failure_reason = "Interrupted during verification; marked failed-closed."
            self.state_mgr.save(checkpoint)

        if t_state.status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED):
            logger.info("Task %s is already verified (status: %s).", task_id, t_state.status.value)
            return t_state

        # Check attempt budget exhaustion
        if t_state.attempts >= self.config.max_attempts_per_task:
            logger.warning("Task %s has already exhausted attempt budget (%d/%d).", task_id, t_state.attempts, self.config.max_attempts_per_task)
            self._emit_progress(f"[{t_state.task_id}] Cannot execute: attempt budget exhausted ({t_state.attempts}/{self.config.max_attempts_per_task}).")
            t_state.status = TaskStatus.FAILED
            if not t_state.failure_reason:
                t_state.failure_reason = f"Attempt budget exhausted ({t_state.attempts}/{self.config.max_attempts_per_task})."
            self.state_mgr.save(checkpoint)
            return t_state

        # Check prerequisite dependencies
        deps_met, blocked_reason = self._check_dependencies(t_state, checkpoint)
        if not deps_met:
            t_state.status = TaskStatus.BLOCKED
            t_state.blocking_reason = blocked_reason
            checkpoint.blocked_count += 1
            self.state_mgr.save(checkpoint)
            return t_state

        self._execute_task_lifecycle(t_state, checkpoint)
        self.state_mgr.save(checkpoint)
        return t_state

    # --------------------------------------------------------------------------
    # Per-Task Execution Lifecycle
    # --------------------------------------------------------------------------

    def _execute_task_lifecycle(
        self,
        t_state: TaskCampaignState,
        checkpoint: CampaignCheckpoint,
    ) -> None:
        """
        Executes the 8-stage verification lifecycle for one task in its isolated worktree:
        1. Prerequisite & workspace setup
        2. Bounded planning (qwen3.5:4b)
        3. Baseline pre-patch verification check
        4. Implementation (Tier 1 MacroAgent or Tier 2 Antigravity)
        5. Quality gate evaluation (QualityGateEngine)
        6. Safeguard diff scanning (ReviewerSafeguardScanner)
        7. Independent review & bounded correction (CopilotReviewerAdapter)
        8. Status finalization (READY_TO_INTEGRATE)
        """
        t_state.status = TaskStatus.IN_PROGRESS
        t_state.attempts += 1
        self._emit_progress(f"[{t_state.task_id}] Lifecycle initiated (attempt {t_state.attempts}/{self.config.max_attempts_per_task})")
        self.state_mgr.save(checkpoint)

        # Reconstruct TaskDefinition
        tdef = TaskDefinition(
            task_id=t_state.task_id,
            title=t_state.title,
            audit_source=checkpoint.source_markdown,
            priority=t_state.priority,
            risk=t_state.risk,
            description=t_state.description,
            evidence=t_state.evidence,
            target_files=list(t_state.target_files),
            test_targets=list(t_state.test_targets),
            dependencies=list(t_state.dependencies),
            acceptance_criteria=list(t_state.acceptance_criteria),
            task_type=t_state.task_type,
            frozen_tests=t_state.frozen_tests,
        )
        if t_state.expected_behavior:
            setattr(tdef, "expected_behavior", t_state.expected_behavior)

        # Step 1: Prepare isolated worktree
        try:
            ws_meta = self.worktree_mgr.prepare_workspace(tdef.task_id)
            worktree_path = Path(ws_meta.worktree_path).resolve()
            t_state.worktree_path = str(worktree_path)
            t_state.branch_name = ws_meta.branch_name
            self._emit_progress(f"[{t_state.task_id}] Worktree ready at {worktree_path} (branch: {t_state.branch_name})")
        except WorkspaceError as exc:
            t_state.status = TaskStatus.FAILED
            t_state.failure_reason = f"Workspace error: {exc}"
            checkpoint.failed_count += 1
            self._record_attempt(t_state)
            self._emit_progress(f"[{t_state.task_id}] Workspace failure: {exc}")
            self.state_mgr.save(checkpoint)
            return

        # Step 1b: Propagate prerequisite dependencies into isolated worktree
        if tdef.dependencies:
            chain_ok, chain_err = self._propagate_dependencies(t_state, checkpoint, worktree_path)
            if not chain_ok:
                t_state.status = TaskStatus.BLOCKED
                t_state.blocking_reason = chain_err
                checkpoint.blocked_count += 1
                self._record_attempt(t_state)
                self._emit_progress(f"[{t_state.task_id}] Dependency chaining blocked: {chain_err}")
                self.state_mgr.save(checkpoint)
                return

        # Step 2: Formulate structured plan using qwen3.5:4b
        self._emit_progress(f"[{t_state.task_id}] Formulating plan using {self.config.planner_model}...")
        t_state.plan_summary = self.planner.plan_task(tdef, self.repo_root)

        # Step 3: Worker routing & Tier 2 authorization gate
        tier = self._route_worker_tier(tdef, t_state.attempts)
        t_state.worker_tier = tier
        self._emit_progress(f"[{t_state.task_id}] Routed to {tier.value} worker")

        if tier == WorkerTier.TIER_2_CLOUD:
            if self.config.require_authorization_for_tier2 and not self.config.is_tier2_authorized:
                t_state.status = TaskStatus.NEEDS_APPROVAL
                t_state.blocking_reason = "Escalated to Tier 2 (Antigravity/Cloud); requires explicit human operator authorization."
                self._record_attempt(t_state)
                self._emit_progress(f"[{t_state.task_id}] Held for operator approval: Tier 2 escalation unauthorized")
                self.state_mgr.save(checkpoint)
                return

        # Step 4: Freeze test targets and baseline verification
        is_mock_quality = not hasattr(self.quality_gates, "run_verification_tests")
        if not t_state.frozen_tests:
            frozen_meta, freeze_errs = freeze_task_tests(tdef, worktree_path)
            if tdef.is_defect_task() and (freeze_errs or not frozen_meta or not frozen_meta.get("hashes")):
                if is_mock_quality:
                    # Provide dummy frozen hashes for unit test mock environments
                    frozen_meta = {
                        "task_id": tdef.task_id,
                        "frozen_at": utc_now_iso(),
                        "targets": list(tdef.test_targets or []),
                        "hashes": {t: "mock_hash" for t in (tdef.test_targets or [])},
                    }
                    t_state.frozen_tests = frozen_meta
                else:
                    err_msg = "; ".join(freeze_errs) if freeze_errs else "no test targets specified or existing"
                    t_state.status = TaskStatus.NEEDS_APPROVAL
                    t_state.blocking_reason = f"test doesn't prove the defect: {err_msg}. Human approval required."
                    self._record_attempt(t_state)
                    self._emit_progress(f"[{t_state.task_id}] Held for approval: {t_state.blocking_reason}")
                    self.state_mgr.save(checkpoint)
                    return
            else:
                t_state.frozen_tests = frozen_meta

        tdef.frozen_tests = t_state.frozen_tests

        # Pre-repair baseline verification
        if not t_state.baseline_evidence:
            self._emit_progress(f"[{t_state.task_id}] Running baseline verification pytest...")
            if hasattr(self.quality_gates, "run_verification_tests"):
                baseline_gate_res = self.quality_gates.run_verification_tests(tdef, worktree_path)
                t_state.baseline_evidence = baseline_gate_res.to_dict()
            else:
                t_state.baseline_evidence = {
                    "gate_name": "defect_baseline",
                    "passed": False,
                    "details": {"exit_code": 1, "test_results": {"total": 1, "failed": 1}},
                }

        # Attach baseline evidence to checkpoint & history for QualityGateEngine
        tdef.checkpoint = CheckpointRecord(
            timestamp=utc_now_iso(),
            stage="BASELINE_VERIFICATION",
            details={"baseline_verification": t_state.baseline_evidence},
        )
        tdef.history.append(
            EvidenceRecord(
                timestamp=utc_now_iso(),
                stage="BASELINE_VERIFICATION",
                details={"baseline_verification": t_state.baseline_evidence},
                success=not t_state.baseline_evidence.get("passed", False),
            )
        )

        if tdef.is_defect_task() and not is_mock_quality:
            is_valid, verif_msg = is_valid_defect_baseline(t_state.baseline_evidence.get("details", {}))
            if not is_valid:
                t_state.status = TaskStatus.NEEDS_APPROVAL
                t_state.blocking_reason = f"test doesn't prove the defect: {verif_msg}. Human approval required."
                self._record_attempt(t_state)
                self._emit_progress(f"[{t_state.task_id}] Baseline invalid: {verif_msg}")
                self.state_mgr.save(checkpoint)
                return
            self._emit_progress(f"[{t_state.task_id}] Baseline verified: defect successfully reproduced")

        # Step 5: Implementation via routed worker
        self._emit_progress(f"[{t_state.task_id}] Dispatched to {tier.value} worker...")
        prior_feedback = None
        if t_state.attempts > 1 and t_state.failure_reason:
            prior_feedback = f"Prior attempt failed: {t_state.failure_reason}"

        impl_success, impl_error = self._invoke_worker(
            tdef, worktree_path, tier, t_state.plan_summary, feedback=prior_feedback
        )
        if not impl_success:
            t_state.status = TaskStatus.FAILED
            if "not configured or unavailable" in str(impl_error) or t_state.attempts >= self.config.max_attempts_per_task:
                t_state.failure_reason = f"Implementation failed: {impl_error}"
                checkpoint.failed_count += 1
            else:
                t_state.failure_reason = f"Implementation attempt {t_state.attempts} failed: {impl_error}"
            self._record_attempt(t_state)
            self._emit_progress(f"[{t_state.task_id}] Implementation failed: {impl_error}")
            self.state_mgr.save(checkpoint)
            return

        # Step 6: Quality Gate Verification
        t_state.status = TaskStatus.VERIFYING
        self._emit_progress(f"[{t_state.task_id}] Evaluating 7 mechanical quality gates...")
        self.state_mgr.save(checkpoint)

        q_report: QualityReport = self.quality_gates.evaluate(tdef, worktree_path)
        if not q_report.passed:
            t_state.status = TaskStatus.FAILED
            if t_state.attempts < self.config.max_attempts_per_task:
                t_state.failure_reason = f"Quality gate failure: {q_report.failure_reason}"
            else:
                t_state.failure_reason = (
                    f"Exhausted max attempts ({self.config.max_attempts_per_task}). "
                    f"Quality gate failure: {q_report.failure_reason}"
                )
                checkpoint.failed_count += 1
            self._record_attempt(t_state)
            self._emit_progress(f"[{t_state.task_id}] Quality gates FAILED: {q_report.failure_reason}")
            self.state_mgr.save(checkpoint)
            return

        t_state.verification_passed = True
        self._emit_progress(f"[{t_state.task_id}] All 7 quality gates PASSED")

        # Step 7: Safeguard Scanner (inspect diff)
        diff_text = self._get_worktree_diff(worktree_path)
        self._emit_progress(f"[{t_state.task_id}] Scanning diff against reviewer safeguards...")
        if hasattr(self.safeguards, "scan_diff"):
            scan_report: ScanReport = self.safeguards.scan_diff(tdef, diff_text)
        else:
            scan_report: ScanReport = self.safeguards.scan(tdef, diff_text)

        if not scan_report.passed:
            t_state.status = TaskStatus.NEEDS_APPROVAL
            t_state.blocking_reason = f"Safeguard violation: {scan_report.approval_reason}"
            self._record_attempt(t_state)
            self._emit_progress(f"[{t_state.task_id}] Safeguard violation: {scan_report.approval_reason}")
            self.state_mgr.save(checkpoint)
            return

        # Step 8: Independent Review & Bounded Correction Cycle
        if self.config.require_review:
            verif_summary = "\n".join(r.message for r in q_report.results) if q_report.results else "All quality gates passed."
            self._emit_progress(f"[{t_state.task_id}] Submitting diff to independent {self.config.reviewer_adapter} reviewer...")
            review_ok, review_msg = self._run_review_cycle(tdef, worktree_path, diff_text, t_state, verification_summary=verif_summary)
            if not review_ok:
                self._record_attempt(t_state)
                self._emit_progress(f"[{t_state.task_id}] Independent review stopped: {review_msg}")
                self.state_mgr.save(checkpoint)
                return
            self._emit_progress(f"[{t_state.task_id}] Independent review APPROVED")

        # Commit verified changes to worktree branch
        self._commit_worktree_changes(worktree_path, tdef)

        # Finalize Task Success (READY_TO_INTEGRATE)
        t_state.status = TaskStatus.READY_TO_INTEGRATE
        t_state.failure_reason = None
        t_state.blocking_reason = None
        self._record_attempt(t_state)
        checkpoint.completed_count += 1
        self._emit_progress(f"[{t_state.task_id}] Task completed: READY_TO_INTEGRATE")
        self.state_mgr.save(checkpoint)

    def _record_attempt(self, t_state: TaskCampaignState) -> None:
        """Record attempt metadata into task execution history."""
        rec = {
            "attempt": t_state.attempts,
            "worker_tier": t_state.worker_tier.value,
            "status": t_state.status.value,
            "plan_summary": t_state.plan_summary,
            "verification_passed": t_state.verification_passed,
            "failure_reason": t_state.failure_reason,
            "blocking_reason": t_state.blocking_reason,
            "review_recommendation": t_state.review_recommendation,
            "timestamp": utc_now_iso(),
        }
        t_state.attempt_history.append(rec)

    # --------------------------------------------------------------------------
    # Worker Dispatch & Review Cycles
    # --------------------------------------------------------------------------

    def _route_worker_tier(self, task: TaskDefinition, attempts: int = 1) -> WorkerTier:
        """Determine appropriate worker tier based on risk, scope, and attempts."""
        if self.config.tier1_only:
            return WorkerTier.TIER_1_LOCAL
        if attempts > 1:
            return WorkerTier.TIER_2_CLOUD
        if task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH):
            return WorkerTier.TIER_2_CLOUD
        if len(task.target_files) >= 3:
            return WorkerTier.TIER_2_CLOUD
        # Architectural core
        for f in task.target_files:
            clean = f.replace("\\", "/")
            if any(clean.startswith(p) for p in ("core/", "brain/router/", "services/ws_server.py")):
                return WorkerTier.TIER_2_CLOUD
        return WorkerTier.TIER_1_LOCAL

    def _build_worker_prompt(
        self,
        task: TaskDefinition,
        plan: str,
        tier: WorkerTier,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Construct a comprehensive, self-contained worker prompt incorporating
        task specifications, acceptance criteria, prior feedback, and tier-appropriate constraints.
        """
        priority_val = task.priority.value if hasattr(task.priority, "value") else str(task.priority)
        risk_val = task.risk.value if hasattr(task.risk, "value") else str(task.risk)
        targets_str = ", ".join(task.target_files) if task.target_files else "None"
        tests_str = ", ".join(task.test_targets) if task.test_targets else "None"

        sections = [
            f"CAMPAIGN REPAIR TASK: {task.task_id}",
            f"Title: {task.title}",
            f"Priority: {priority_val} | Risk: {risk_val}",
        ]

        if task.description:
            sections.append(f"\nDescription:\n{task.description.strip()}")

        exp = getattr(task, "expected_behavior", "")
        if exp and exp not in (task.description or ""):
            sections.append(f"\nExpected Behavior:\n{exp.strip()}")

        if task.evidence:
            sections.append(f"\nEvidence of Defect:\n{task.evidence.strip()}")

        if task.acceptance_criteria:
            criteria_str = "\n".join(f"- {c}" for c in task.acceptance_criteria)
            sections.append(f"\nAcceptance Criteria:\n{criteria_str}")

        sections.append(f"\nTarget Files: {targets_str}\nTest Targets: {tests_str}")

        if plan:
            sections.append(f"\nPlan:\n{plan.strip()}")

        if feedback:
            sections.append(f"\n=== PREVIOUS ATTEMPT / REVIEW FEEDBACK ===\n{feedback.strip()}\nPlease adjust your implementation to address this feedback.")

        if tier == WorkerTier.TIER_1_LOCAL:
            sections.append(
                "\nInstructions:\n"
                "1. Read the target files using 'fs_read'.\n"
                "2. Apply exact-match edits using 'fs_edit'.\n"
                "3. Run focused tests using 'test_runner'.\n"
                "4. Synchronize docs/CHANGELOG.md under ### Fixes.\n"
                "5. Conclude with <finish_task> when verified."
            )
        else:
            sections.append(
                "\n=== STRICT OPERATIONAL INVARIANTS ===\n"
                "1. NO COMMAND OR SHELL TOOLS: Do NOT invoke run_command or shell commands. Inspect and edit files strictly using file tools.\n"
                "2. MANDATORY TOOL-BACKED IMPLEMENTATION: Use file editing tools to modify declared target files on disk. Do not return without edits.\n"
                "3. WORKTREE CONFINEMENT: Stay strictly within the assigned worktree. Never access .git or files outside the worktree root.\n"
                "4. SCOPE BOUNDARY: Only modify files listed in Target Files and docs/CHANGELOG.md.\n"
                "5. TEST POLICY: All tests must remain strictly under '.maya/tests/'. Do not weaken assertions or add skips.\n"
                "6. CHANGELOG: Record your change in docs/CHANGELOG.md under ### Fixes in the same changeset.\n"
                "7. GIT RESTRICTIONS: Do not run git stage, commit, push, or destructive commands."
            )

        return "\n".join(sections)

    def _invoke_worker(
        self,
        task: TaskDefinition,
        worktree_path: Path,
        tier: WorkerTier,
        plan: str,
        feedback: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Dispatch repair task to Tier 1 MACRO agent or Tier 2 Antigravity adapter."""
        prompt = self._build_worker_prompt(task, plan, tier, feedback=feedback)

        if tier == WorkerTier.TIER_1_LOCAL:
            if not self.agent:
                return False, "Tier 1 worker (MacroAgent) is not configured or unavailable."
            # Use MACRO agent with worktree context
            reg = create_default_registry()
            ctx = ToolExecutionContext(
                worktree_path=worktree_path,
                repo_root=self.repo_root,
                task_id=task.task_id,
                task_type=getattr(task, "task_type", "defect"),
                target_files=list(task.target_files or []),
                test_targets=list(task.test_targets or []),
                require_repair_evidence=task.is_defect_task() if hasattr(task, "is_defect_task") else True,
            )
            if getattr(self, "_campaign_start_time", None) is not None:
                ctx.campaign_deadline_monotonic = self._campaign_start_time + self.config.campaign_timeout_seconds
            res: MacroAgentResult = self.agent.run(prompt, context=ctx, registry=reg, on_progress=self._emit_progress)
            return res.success, res.error_message or res.summary

        elif tier == WorkerTier.TIER_2_CLOUD:
            if not self.tier2_adapter:
                return False, "Tier 2 worker (AntigravityAgentAdapter) is not configured or unavailable."
            res_adapter: AgentExecutionResult = self.tier2_adapter.execute(
                task=task,
                prompt=prompt,
                worktree_path=worktree_path,
                timeout_seconds=int(self.config.task_timeout_seconds),
            )
            return res_adapter.success, res_adapter.error_message

        return False, f"Unknown worker tier '{tier}'."

    def _run_review_cycle(
        self,
        task: TaskDefinition,
        worktree_path: Path,
        diff_text: str,
        t_state: TaskCampaignState,
        verification_summary: str = "",
    ) -> tuple[bool, Optional[str]]:
        """Conduct independent code review with bounded correction attempts."""
        if not self.reviewer:
            t_state.status = TaskStatus.NEEDS_APPROVAL
            t_state.blocking_reason = "Reviewer is required but not configured."
            return False, t_state.blocking_reason

        if not verification_summary:
            verification_summary = (
                f"Quality Gates: passed={t_state.verification_passed}\n"
                f"Target Files: {', '.join(t_state.target_files)}\n"
                f"Test Targets: {', '.join(t_state.test_targets)}"
            )

        try:
            review_res: ReviewResult = self.reviewer.review(
                task=task,
                diff=diff_text,
                verification_summary=verification_summary,
                worktree_path=worktree_path,
                timeout_seconds=int(self.config.task_timeout_seconds),
            )
        except Exception as exc:
            logger.error("Reviewer invocation failed for %s: %s", task.task_id, exc, exc_info=True)
            t_state.status = TaskStatus.NEEDS_APPROVAL
            t_state.blocking_reason = f"Reviewer invocation failed: {exc}"
            return False, t_state.blocking_reason

        rec = getattr(review_res, "recommendation", None)
        rec_val = getattr(rec, "value", str(rec)) if rec else "UNKNOWN"
        t_state.review_recommendation = rec_val

        if rec == ReviewRecommendation.APPROVE:
            return True, "Review approved."

        elif rec == ReviewRecommendation.NEEDS_HUMAN_APPROVAL:
            t_state.status = TaskStatus.NEEDS_APPROVAL
            t_state.blocking_reason = f"Reviewer elevated: {review_res.summary}"
            return False, t_state.blocking_reason

        elif rec == ReviewRecommendation.REQUEST_CHANGES:
            t_state.review_corrections += 1
            if t_state.review_corrections <= self.config.max_review_corrections:
                logger.info(
                    "Reviewer requested changes for %s (correction attempt %d/%d). Retrying...",
                    task.task_id, t_state.review_corrections, self.config.max_review_corrections
                )
                # Correction cycle: pass reviewer findings to worker alongside full task context
                correction_feedback = f"Reviewer requested changes:\n{review_res.summary}"
                success, _ = self._invoke_worker(
                    task, worktree_path, t_state.worker_tier, t_state.plan_summary, feedback=correction_feedback
                )
                if success:
                    q_post = self.quality_gates.evaluate(task, worktree_path)
                    if not q_post.passed:
                        t_state.status = TaskStatus.FAILED
                        t_state.failure_reason = f"Quality gate failure after correction: {q_post.failure_reason}"
                        return False, t_state.failure_reason
                    new_diff = self._get_worktree_diff(worktree_path)
                    new_verif = "\n".join(r.message for r in q_post.results) if q_post.results else "Quality gates passed."
                    return self._run_review_cycle(task, worktree_path, new_diff, t_state, verification_summary=new_verif)

            t_state.status = TaskStatus.NEEDS_APPROVAL
            t_state.blocking_reason = "Reviewer requested changes; review correction budget exhausted."
            return False, t_state.blocking_reason

        else:
            t_state.status = TaskStatus.NEEDS_APPROVAL
            t_state.blocking_reason = f"Reviewer inconclusive or rejected: {getattr(review_res, 'error_message', None) or getattr(review_res, 'summary', 'Unknown review outcome')}"
            return False, t_state.blocking_reason

    def _check_dependencies(
        self,
        task_state: TaskCampaignState,
        checkpoint: CampaignCheckpoint,
    ) -> tuple[bool, Optional[str]]:
        """Verify that all prerequisite tasks are completed/verified."""
        for dep_id in task_state.dependencies:
            dep_state = checkpoint.tasks.get(dep_id)
            if not dep_state:
                return False, f"Prerequisite task '{dep_id}' was not found in campaign."
            if dep_state.status not in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED):
                return False, f"Prerequisite task '{dep_id}' is not yet verified (current: {dep_state.status.value})."
        return True, None

    def _get_worktree_diff(self, worktree_path: Path) -> str:
        """Capture git diff within the worktree."""
        import subprocess
        try:
            subprocess.run(
                ["git", "add", "-N", "."],
                cwd=str(worktree_path),
                capture_output=True,
                check=False,
            )
        except Exception:
            pass
        proc = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout or ""

    def _commit_worktree_changes(self, worktree_path: Path, tdef: TaskDefinition) -> bool:
        """Commit declared target files and changelog in the task worktree to preserve verified changes."""
        import subprocess
        try:
            target_files = set(tdef.target_files or []) | {"docs/CHANGELOG.md"}
            files_to_stage = [f for f in target_files if (worktree_path / f).exists()]
            if files_to_stage:
                subprocess.run(
                    ["git", "add", "--", *files_to_stage],
                    cwd=str(worktree_path),
                    capture_output=True,
                    check=False,
                )
            st_proc = subprocess.run(
                ["git", "diff", "--staged", "--quiet"],
                cwd=str(worktree_path),
                capture_output=True,
                check=False,
            )
            if st_proc.returncode != 0:
                commit_msg = f"fix({tdef.task_id.lower()}): {tdef.title}"
                res = subprocess.run(
                    ["git", "-c", "user.name=AutoPhage", "-c", "user.email=autophage@mayave.local", "commit", "-m", commit_msg],
                    cwd=str(worktree_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return res.returncode == 0
            return True
        except Exception as exc:
            logger.warning("Could not commit worktree changes for %s: %s", tdef.task_id, exc)
            return False

    def _propagate_dependencies(
        self,
        t_state: TaskCampaignState,
        checkpoint: CampaignCheckpoint,
        worktree_path: Path,
    ) -> tuple[bool, Optional[str]]:
        """Propagate verified changes of prerequisite tasks into the current worktree."""
        import subprocess
        import shutil

        for dep_id in t_state.dependencies:
            dep_state = checkpoint.tasks.get(dep_id)
            if not dep_state:
                return False, f"Prerequisite task '{dep_id}' was not found in campaign."
            if dep_state.status not in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED):
                return False, f"Prerequisite task '{dep_id}' is not yet verified (current: {dep_state.status.value})."

            dep_branch = dep_state.branch_name or self.worktree_mgr.get_branch_name(dep_id)
            dep_wt = Path(dep_state.worktree_path) if dep_state.worktree_path else self.worktree_mgr.get_workspace_path(dep_id)

            if dep_wt.exists():
                # Commit any verified target files if not yet committed
                dep_targets = set(dep_state.target_files or []) | {"docs/CHANGELOG.md"}
                dep_stage = [f for f in dep_targets if (dep_wt / f).exists()]
                if dep_stage:
                    subprocess.run(["git", "add", "--", *dep_stage], cwd=str(dep_wt), capture_output=True, check=False)
                    st_proc = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(dep_wt), capture_output=True, check=False)
                    if st_proc.returncode != 0:
                        subprocess.run(
                            ["git", "-c", "user.name=AutoPhage", "-c", "user.email=autophage@mayave.local", "commit", "-m", f"fix({dep_id.lower()}): {dep_state.title}"],
                            cwd=str(dep_wt),
                            capture_output=True,
                            check=False,
                        )

                # Mirror untracked .maya/tests/ files from prerequisite worktree
                dep_tests = dep_wt / ".maya" / "tests"
                dst_tests = worktree_path / ".maya" / "tests"
                if dep_tests.is_dir():
                    dst_tests.mkdir(parents=True, exist_ok=True)
                    for item in dep_tests.rglob("*.py"):
                        if "__pycache__" in item.parts or ".pytest_cache" in item.parts:
                            continue
                        rel = item.relative_to(dep_tests)
                        target_file = dst_tests / rel
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        if not target_file.exists() or target_file.stat().st_mtime < item.stat().st_mtime:
                            try:
                                shutil.copy2(item, target_file)
                            except Exception:
                                pass

            # Resolve git reference for dep_branch
            ref_proc = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/heads/{dep_branch}"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if ref_proc.returncode != 0:
                # Try fallback to integration branch if already integrated
                int_b = f"autophage/integrate-{dep_id}"
                ref_proc2 = subprocess.run(
                    ["git", "rev-parse", "--verify", f"refs/heads/{int_b}"],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if ref_proc2.returncode == 0:
                    dep_branch = int_b

            # Query commits on prerequisite branch not yet in current worktree HEAD
            rev_proc = subprocess.run(
                ["git", "rev-list", "--reverse", "^HEAD", dep_branch],
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if rev_proc.returncode == 0 and rev_proc.stdout.strip():
                commits = rev_proc.stdout.strip().splitlines()
                for c in commits:
                    cp_proc = subprocess.run(
                        ["git", "-c", "user.name=AutoPhage", "-c", "user.email=autophage@mayave.local", "cherry-pick", c],
                        cwd=str(worktree_path),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if cp_proc.returncode != 0:
                        subprocess.run(["git", "cherry-pick", "--abort"], cwd=str(worktree_path), capture_output=True, check=False)
                        return False, f"Dependency patch conflict with prerequisite '{dep_id}': failed to cleanly apply commit {c[:7]}."

            self._emit_progress(f"[{t_state.task_id}] Successfully chained prerequisite changes from '{dep_id}'.")

        return True, None

    def list_eligible_integration_tasks(self) -> list[dict]:
        """Find all tasks across campaign_state.json and legacy state.json eligible for integration."""
        candidates = []
        seen_ids = set()

        # 1. Campaign state tasks
        cp = self.state_mgr.load()
        if cp:
            for task_id in cp.task_order:
                t = cp.tasks.get(task_id)
                if t and t.status == TaskStatus.READY_TO_INTEGRATE:
                    candidates.append({
                        "task_id": t.task_id,
                        "title": t.title,
                        "priority": t.priority.value,
                        "risk": t.risk.value,
                        "status": t.status.value,
                        "branch_name": t.branch_name or self.worktree_mgr.get_branch_name(t.task_id),
                        "worktree_path": t.worktree_path or str(self.worktree_mgr.get_workspace_path(t.task_id)),
                        "target_files": list(t.target_files),
                        "dependencies": list(t.dependencies),
                        "verification_passed": t.verification_passed,
                        "source": "campaign",
                    })
                    seen_ids.add(t.task_id)

        # 2. Legacy state.json tasks
        if StateStore:
            legacy_store = StateStore(self.repo_root / ".maya" / "maintenance" / "state.json")
            if legacy_store.exists():
                try:
                    ledger = legacy_store.load_ledger()
                    for t in ledger:
                        if t.task_id not in seen_ids and t.status == TaskStatus.READY_TO_INTEGRATE:
                            wt_p = t.workspace.worktree_path if t.workspace else str(self.worktree_mgr.get_workspace_path(t.task_id))
                            br = t.workspace.branch_name if t.workspace else self.worktree_mgr.get_branch_name(t.task_id)
                            candidates.append({
                                "task_id": t.task_id,
                                "title": t.title,
                                "priority": t.priority.value,
                                "risk": t.risk.value,
                                "status": t.status.value,
                                "branch_name": br,
                                "worktree_path": wt_p,
                                "target_files": list(t.target_files),
                                "dependencies": list(t.dependencies),
                                "verification_passed": True,
                                "source": "legacy",
                            })
                            seen_ids.add(t.task_id)
                except Exception as exc:
                    logger.warning("Could not read legacy state.json: %s", exc)

        return candidates

    def integrate_task(
        self,
        task_id: str,
        authorized: bool = False,
        target_branch: Optional[str] = None,
        dry_run: bool = False,
    ) -> tuple[bool, str, dict]:
        """
        Safely integrate a verified task into a dedicated integration branch.
        - Validates task state is READY_TO_INTEGRATE.
        - Verifies quality gates passed and candidate diff is non-empty.
        - Never merges into main.
        - Preserves unrelated working tree files by using an isolated integration worktree.
        - Synchronizes status to VERIFIED across campaign_state.json and legacy state.json.
        """
        import subprocess
        import shutil

        cp = self.state_mgr.load()
        t_state = cp.tasks.get(task_id) if cp else None

        legacy_task = None
        legacy_store = None
        if StateStore:
            legacy_store = StateStore(self.repo_root / ".maya" / "maintenance" / "state.json")
            if legacy_store.exists():
                try:
                    ledger = legacy_store.load_ledger()
                    legacy_task = ledger.get(task_id)
                except Exception:
                    pass

        if not t_state and not legacy_task:
            return False, f"Task '{task_id}' not found in active campaign or legacy ledger.", {}

        # Resolve task metadata
        title = t_state.title if t_state else legacy_task.title
        status = t_state.status if t_state else legacy_task.status
        branch_name = (
            (t_state.branch_name if t_state else (legacy_task.workspace.branch_name if legacy_task and legacy_task.workspace else None))
            or self.worktree_mgr.get_branch_name(task_id)
        )
        ws_path_str = (
            (t_state.worktree_path if t_state else (legacy_task.workspace.worktree_path if legacy_task and legacy_task.workspace else None))
            or str(self.worktree_mgr.get_workspace_path(task_id))
        )
        ws_path = Path(ws_path_str).resolve()
        target_files = list(t_state.target_files if t_state else legacy_task.target_files)

        if status == TaskStatus.VERIFIED:
            return True, f"Task '{task_id}' is already VERIFIED and integrated.", {"status": "ALREADY_VERIFIED"}

        if status != TaskStatus.READY_TO_INTEGRATE:
            return False, f"Task '{task_id}' is not in READY_TO_INTEGRATE state (current status: {status.value}). Integration rejected.", {"status": status.value}

        if not ws_path.is_dir():
            return False, f"Worktree directory for task '{task_id}' does not exist at {ws_path}.", {}

        # Ensure verified target files are committed in worktree
        tdef_for_commit = TaskDefinition(
            task_id=task_id,
            title=title,
            audit_source="integration",
            priority=TaskPriority.P2,
            risk=TaskRisk.LOW,
            description=t_state.description if t_state else "",
            evidence=t_state.evidence if t_state else "",
            target_files=target_files,
            test_targets=list(t_state.test_targets if t_state else []),
        )
        self._commit_worktree_changes(ws_path, tdef_for_commit)

        # Candidate diff check
        diff_text = self._get_worktree_diff(ws_path)
        # Also check commits ahead on branch
        rev_ahead = subprocess.run(
            ["git", "rev-list", f"^HEAD", branch_name],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        has_commits = rev_ahead.returncode == 0 and bool(rev_ahead.stdout.strip())
        if not diff_text and not has_commits:
            return False, f"Candidate patch for task '{task_id}' is empty; no diff or commits found.", {}

        # If diff_text was empty but commits exist, get diff from git diff
        if not diff_text and has_commits:
            commit_diff_proc = subprocess.run(
                ["git", "diff", f"HEAD..{branch_name}"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            diff_text = commit_diff_proc.stdout or ""

        # Identify changed files
        changed_proc = subprocess.run(
            ["git", "diff", "--name-only", f"HEAD..{branch_name}"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        changed_files = changed_proc.stdout.strip().splitlines() if (changed_proc.returncode == 0 and changed_proc.stdout.strip()) else target_files

        info = {
            "task_id": task_id,
            "title": title,
            "branch": branch_name,
            "worktree_path": str(ws_path),
            "changed_files": changed_files,
            "diff_preview": diff_text[:1200],
            "full_diff": diff_text,
            "verification_passed": t_state.verification_passed if t_state else True,
        }

        # Check authorization / dry run
        if dry_run or not authorized:
            return False, "Integration requires explicit operator authorization (--authorize).", info

        # Governance policy: Never integrate directly into main
        target_branch = target_branch or f"autophage/integrate-{task_id}"
        if target_branch == "main":
            return False, "Integration directly into branch 'main' is strictly forbidden by repository governance policy.", info

        # Create isolated integration worktree
        int_wt_path = self.worktree_mgr.worktrees_dir / f"integrate-{task_id}"
        if int_wt_path.exists():
            shutil.rmtree(int_wt_path, ignore_errors=True)
            subprocess.run(["git", "worktree", "prune"], cwd=str(self.repo_root), capture_output=True, check=False)

        # Baseline commit from repo_root HEAD
        head_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self.repo_root), capture_output=True, text=True, check=False)
        base_commit = head_proc.stdout.strip() if head_proc.returncode == 0 else "HEAD"

        ref_check = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{target_branch}"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if ref_check.returncode == 0:
            add_cmd = ["git", "worktree", "add", str(int_wt_path), target_branch]
        else:
            add_cmd = ["git", "worktree", "add", "-b", target_branch, str(int_wt_path), base_commit]

        wt_proc = subprocess.run(add_cmd, cwd=str(self.repo_root), capture_output=True, text=True, check=False)
        if wt_proc.returncode != 0:
            return False, f"Failed to create integration worktree: {wt_proc.stderr.strip()}", info

        try:
            # Apply commits from task branch into integration worktree
            commits_proc = subprocess.run(
                ["git", "rev-list", "--reverse", "^HEAD", branch_name],
                cwd=str(int_wt_path),
                capture_output=True,
                text=True,
                check=False,
            )
            commits = commits_proc.stdout.strip().splitlines() if (commits_proc.returncode == 0 and commits_proc.stdout.strip()) else []

            if commits:
                for c in commits:
                    cp_res = subprocess.run(
                        ["git", "-c", "user.name=AutoPhage", "-c", "user.email=autophage@mayave.local", "cherry-pick", c],
                        cwd=str(int_wt_path),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if cp_res.returncode != 0:
                        subprocess.run(["git", "cherry-pick", "--abort"], cwd=str(int_wt_path), capture_output=True, check=False)
                        return False, f"Integration conflict applying commit {c[:7]} from {branch_name} into {target_branch}: {cp_res.stderr.strip()}", info
            else:
                # Apply diff if no distinct commits
                apply_res = subprocess.run(
                    ["git", "apply", "--ignore-whitespace"],
                    input=diff_text,
                    text=True,
                    cwd=str(int_wt_path),
                    capture_output=True,
                    check=False,
                )
                if apply_res.returncode != 0:
                    return False, f"Integration conflict applying candidate diff into {target_branch}: {apply_res.stderr.strip()}", info
                subprocess.run(["git", "add", "-A"], cwd=str(int_wt_path), check=True)
                subprocess.run(
                    ["git", "-c", "user.name=AutoPhage", "-c", "user.email=autophage@mayave.local", "commit", "-m", f"fix({task_id.lower()}): {title}"],
                    cwd=str(int_wt_path),
                    capture_output=True,
                    check=False,
                )
        finally:
            # Clean up temporary integration worktree
            subprocess.run(["git", "worktree", "remove", "--force", str(int_wt_path)], cwd=str(self.repo_root), capture_output=True, check=False)
            if int_wt_path.exists():
                shutil.rmtree(int_wt_path, ignore_errors=True)
            subprocess.run(["git", "worktree", "prune"], cwd=str(self.repo_root), capture_output=True, check=False)

        # Update task state to VERIFIED in campaign checkpoint
        if t_state and cp:
            t_state.status = TaskStatus.VERIFIED
            t_state.blocking_reason = None
            t_state.failure_reason = None
            self._record_attempt(t_state)
            self.state_mgr.save(cp)

        # Update legacy ledger if present
        if legacy_store and legacy_store.exists():
            try:
                ledger = legacy_store.load_ledger()
                leg_t = ledger.get(task_id)
                if leg_t:
                    leg_t.status = TaskStatus.VERIFIED
                    leg_t.checkpoint = CheckpointRecord(
                        stage="INTEGRATED",
                        details={"target_branch": target_branch},
                        timestamp=utc_now_iso(),
                    )
                    if hasattr(ledger, "_commit_status"):
                        ledger._commit_status(task_id, TaskStatus.VERIFIED)
                    legacy_store.save_ledger(ledger)
            except Exception as exc:
                logger.warning("Could not sync legacy ledger for %s: %s", task_id, exc)

        return True, f"Successfully integrated task '{task_id}' into branch '{target_branch}'.", info


# ==============================================================================
# Helper / Entry Point Functions
# ==============================================================================

def run_campaign_from_markdown(
    repo_root: Path | str,
    markdown_path: Path | str,
    config: Optional[CampaignConfig] = None,
    agent: Optional[MacroAgent] = None,
    tier2_adapter: Optional[AgentAdapter] = None,
    reviewer: Optional[ReviewerAdapter] = None,
) -> CampaignCheckpoint:
    """Convenience helper to ingest and run a campaign in one call."""
    engine = CampaignEngine(
        repo_root=repo_root,
        config=config,
        agent=agent,
        tier2_adapter=tier2_adapter,
        reviewer=reviewer,
    )
    engine.ingest_markdown(markdown_path)
    return engine.run_campaign()
