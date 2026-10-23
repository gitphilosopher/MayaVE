"""
Adaptive Worker Scheduler for MayaVE Maintenance Coordinator.
Enforces initial concurrency 1, hard maximum 3, dynamic scaling upon approval blocks,
DAG dependency isolation, file mutual exclusion, process survival quarantine,
risk-tier limits, max task bounding, stop-on-failure safety, and session reporting.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

try:
    from .adapter import AgentAdapter, FakeAgentAdapter
    from .cache import ContextCache
    from .ledger import TaskLedger
    from .hub import CoordinatorAPI
    from .models import TaskDefinition, TaskRisk, TaskStatus, utc_now_iso
    from .orchestrator import ExecutionSummary, SingleTaskOrchestrator
    from .quality import QualityGateEngine
    from .reviewer import FakeReviewerAdapter, ReviewPolicy, ReviewerAdapter
    from .scheduler import DependencyScheduler
    from .store import StateStore
    from .terminal_approval import generate_session_summary, generate_task_report
    from .worktree import WorktreeManager
    from .routing import ComplexityRouter, WorkerConcurrencyGuard
except ImportError:
    from adapter import AgentAdapter, FakeAgentAdapter
    from cache import ContextCache
    from ledger import TaskLedger
    from hub import CoordinatorAPI
    from models import TaskDefinition, TaskRisk, TaskStatus, utc_now_iso
    from orchestrator import ExecutionSummary, SingleTaskOrchestrator
    from quality import QualityGateEngine
    from reviewer import FakeReviewerAdapter, ReviewPolicy, ReviewerAdapter
    from scheduler import DependencyScheduler
    from store import StateStore
    from terminal_approval import generate_session_summary, generate_task_report
    from worktree import WorktreeManager
    from routing import ComplexityRouter, WorkerConcurrencyGuard

logger = logging.getLogger("maya.maintenance.adaptive")


class SlotStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    BLOCKED_AWAITING_APPROVAL = "BLOCKED_AWAITING_APPROVAL"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class AdaptiveConfig:
    initial_concurrency: int = 1
    max_concurrency: int = 3
    poll_interval_sec: float = 0.05
    worker_timeout_sec: int = 600
    max_tasks: Optional[int] = None
    hard_timeout_sec: Optional[float] = 21600.0  # Default 6-hour unattended runtime limit (21600s)
    max_risk: TaskRisk = TaskRisk.MEDIUM
    stop_on_failure: bool = False

    def __post_init__(self) -> None:
        if self.initial_concurrency < 1:
            raise ValueError(f"initial_concurrency must be at least 1, got {self.initial_concurrency}")
        if self.max_concurrency < self.initial_concurrency:
            raise ValueError(
                f"max_concurrency ({self.max_concurrency}) cannot be less than initial_concurrency ({self.initial_concurrency})"
            )
        if self.max_tasks is not None and self.max_tasks < 1:
            raise ValueError(f"max_tasks must be at least 1, got {self.max_tasks}")
        if self.hard_timeout_sec is not None:
            if math.isnan(self.hard_timeout_sec) or math.isinf(self.hard_timeout_sec):
                raise ValueError(f"hard_timeout_sec must be a finite number, got {self.hard_timeout_sec}")
            if self.hard_timeout_sec <= 0:
                raise ValueError(f"hard_timeout_sec must be positive, got {self.hard_timeout_sec}")
            if self.hard_timeout_sec > 86400.0:
                raise ValueError(f"hard_timeout_sec cannot exceed 86400.0 seconds (24 hours), got {self.hard_timeout_sec}")
        if isinstance(self.max_risk, str):
            self.max_risk = TaskRisk(self.max_risk)
        if self.max_risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH):
            raise ValueError(f"{self.max_risk.value} risk tasks are excluded from unattended execution and require interactive approval.")


@dataclass
class WorkerSlot:
    slot_id: int
    task_id: Optional[str] = None
    status: SlotStatus = SlotStatus.IDLE
    worktree_path: Optional[str] = None
    process_pid: Optional[int] = None
    process_confirmed_terminated: bool = True
    can_release_slot: bool = True
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    start_time: Optional[float] = None
    thread: Optional[threading.Thread] = None
    summary: Optional[ExecutionSummary] = None

    def is_occupied(self) -> bool:
        """
        A slot is occupied if it is RUNNING, or if it cannot release its slot
        (e.g., process not confirmed terminated, or quarantined).
        """
        if self.status == SlotStatus.RUNNING:
            return True
        if not self.process_confirmed_terminated or not self.can_release_slot or self.quarantined:
            return True
        return False


@dataclass
class SessionState:
    session_id: str
    status: str  # "RUNNING", "INTERRUPTED", "EXPIRED", "COMPLETED", "MANUAL_STOP"
    requested_duration_sec: float
    start_time_iso: str
    deadline_iso: str
    deadline_timestamp: float
    halt_reason: Optional[str] = None
    resumed_count: int = 0
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "requested_duration_sec": self.requested_duration_sec,
            "start_time_iso": self.start_time_iso,
            "deadline_iso": self.deadline_iso,
            "deadline_timestamp": self.deadline_timestamp,
            "halt_reason": self.halt_reason,
            "resumed_count": self.resumed_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionState:
        return cls(
            session_id=data["session_id"],
            status=data.get("status", "RUNNING"),
            requested_duration_sec=float(data["requested_duration_sec"]),
            start_time_iso=data["start_time_iso"],
            deadline_iso=data["deadline_iso"],
            deadline_timestamp=float(data["deadline_timestamp"]),
            halt_reason=data.get("halt_reason"),
            resumed_count=int(data.get("resumed_count", 0)),
            updated_at=data.get("updated_at", utc_now_iso()),
        )


@dataclass
class AdaptiveSchedulerSummary:
    total_tasks: int = 0
    completed_tasks: int = 0
    verified_tasks: int = 0
    failed_tasks: int = 0
    blocked_tasks: int = 0
    peak_concurrency: int = 0
    quarantined_count: int = 0
    duration_sec: float = 0.0
    slot_summaries: List[dict] = field(default_factory=list)
    task_summaries: Dict[str, dict] = field(default_factory=dict)
    requested_duration_sec: Optional[float] = None
    session_start_time: Optional[str] = None
    session_deadline: Optional[str] = None
    halt_reason: Optional[str] = None
    remaining_duration_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "verified_tasks": self.verified_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "peak_concurrency": self.peak_concurrency,
            "quarantined_count": self.quarantined_count,
            "duration_sec": round(self.duration_sec, 2),
            "slot_summaries": self.slot_summaries,
            "task_summaries": self.task_summaries,
            "requested_duration_sec": self.requested_duration_sec,
            "session_start_time": self.session_start_time,
            "session_deadline": self.session_deadline,
            "halt_reason": self.halt_reason,
            "remaining_duration_sec": round(self.remaining_duration_sec, 2),
        }


def is_escalation_eligible(task: TaskDefinition) -> bool:
    """Check if task is eligible for adapter tier escalation under existing policy."""
    return False


class AdaptiveWorkerScheduler:
    """
    Adaptive worker scheduler managing parallel task execution across git worktrees.
    Enforces risk filtering (LOW prioritized before MEDIUM), max task bounding (cap 2),
    stop-on-failure safety, DAG dependency isolation, and process survival quarantine.
    """

    def __init__(
        self,
        repo_root: Path | str,
        store: StateStore,
        ledger: TaskLedger,
        worktree_mgr: WorktreeManager,
        config: Optional[AdaptiveConfig] = None,
        adapter: Optional[AgentAdapter] = None,
        quality_engine: Optional[QualityGateEngine] = None,
        reviewer: Optional[ReviewerAdapter] = None,
        review_policy: Optional[ReviewPolicy] = None,
        cache: Optional[ContextCache] = None,
        router: Optional[ComplexityRouter] = None,
        concurrency_guard: Optional[WorkerConcurrencyGuard] = None,
        local_adapter: Optional[AgentAdapter] = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.store = store
        self.ledger = ledger
        self.coordinator_api = CoordinatorAPI(store)
        self.worktree_mgr = worktree_mgr
        self.config = config or AdaptiveConfig()
        self.adapter = adapter or FakeAgentAdapter()
        self.quality_engine = quality_engine or QualityGateEngine(self.repo_root)
        self.reviewer = reviewer or FakeReviewerAdapter()
        self.review_policy = review_policy or ReviewPolicy()
        self.cache = cache
        self.router = router
        self.concurrency_guard = concurrency_guard
        self.local_adapter = local_adapter

        self._lock = threading.RLock()
        self.reviewer_lock = threading.Lock()

        # Initialize fixed worker slots up to max_concurrency
        self.slots: Dict[int, WorkerSlot] = {
            i: WorkerSlot(slot_id=i) for i in range(1, self.config.max_concurrency + 1)
        }

        self.claimed_tasks: Set[str] = set()
        self.paused_approval_tasks: Set[str] = set()
        self.deferred_tasks: Dict[str, float] = {}
        self.quarantined_workspaces: Dict[str, str] = {}
        self.task_summaries: Dict[str, ExecutionSummary] = {}
        self.peak_concurrency: int = 0
        self.tasks_completed_count: int = 0
        self.has_halted_due_to_failure: bool = False
        self.halt_reason: Optional[str] = None

    def validate_preflight_integrity(self) -> Tuple[bool, str]:
        """
        Validate persisted state, ownership, checkpoints, and worktrees before resuming.
        Returns (is_valid, reason_or_message).
        """
        with self._lock:
            try:
                # 1. State Store and Ledger validation
                if not self.ledger or len(self.ledger) == 0:
                    return False, "Ledger is empty or uninitialized."

                # 2. Check each task's worktree, branch ownership, and checkpoints
                for task in self.ledger:
                    # Checkpoints validation for non-PENDING tasks
                    if task.status not in (TaskStatus.PENDING, TaskStatus.NEEDS_APPROVAL):
                        if not task.history and not task.checkpoint:
                            return False, f"Task '{task.task_id}' is in state {task.status.value} but lacks checkpoint history."

                    # In-progress tasks must have an existing worktree
                    if task.status in (TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING):
                        if not task.workspace or not task.workspace.worktree_path:
                            return False, f"Task '{task.task_id}' is in state {task.status.value} but lacks workspace metadata."
                        if not Path(task.workspace.worktree_path).exists():
                            return False, f"Task '{task.task_id}' is in state {task.status.value} but its worktree does not exist on disk."

                    if task.workspace and task.workspace.worktree_path:
                        wt_path = Path(task.workspace.worktree_path).resolve()
                        try:
                            wt_path.relative_to(self.worktree_mgr.worktrees_dir)
                        except ValueError:
                            return False, f"Task '{task.task_id}' worktree path '{wt_path}' escapes worktrees directory."

                        if wt_path.exists():
                            # Verify worktree via supported inspect_workspace API
                            try:
                                ws_status = self.worktree_mgr.inspect_workspace(task.task_id)
                            except Exception as ws_err:
                                return False, f"Failed to inspect workspace for task '{task.task_id}': {ws_err}"

                            # Ownership validation: non-NEEDS_APPROVAL tasks must be MANAGED
                            if ws_status.ownership_status != "MANAGED" and task.status != TaskStatus.NEEDS_APPROVAL:
                                return False, (
                                    f"Worktree at {wt_path} has invalid ownership "
                                    f"'{ws_status.ownership_status}' but task is {task.status.value}."
                                )

                            # Verify branch ownership
                            if task.workspace.branch_name:
                                expected_branch = task.workspace.branch_name
                                if ws_status.ownership_status == "CONFLICT":
                                    return False, (
                                        f"Task '{task.task_id}' branch conflict: expected {expected_branch}, "
                                        f"but worktree is registered to conflicting branch '{ws_status.branch_name}'."
                                    )
                                if ws_status.registered_in_git and ws_status.branch_name != expected_branch:
                                    return False, (
                                        f"Task '{task.task_id}' branch mismatch: "
                                        f"expected {expected_branch}, got {ws_status.branch_name}"
                                    )

                return True, "All worktree baselines, ownership, checkpoints, and persisted state validated successfully."
            except Exception as e:
                return False, f"Integrity validation error: {e}"

    def active_running_workers_count(self) -> int:
        """Count of currently active RUNNING worker slots."""
        with self._lock:
            return sum(1 for s in self.slots.values() if s.status == SlotStatus.RUNNING)

    def total_occupied_slots_count(self) -> int:
        """Count of slots that are occupied (running or retained due to survival/quarantine)."""
        with self._lock:
            return sum(1 for s in self.slots.values() if s.is_occupied())

    def blocked_workers_count(self) -> int:
        """Count of workers or tasks that are safely blocked awaiting human approval."""
        with self._lock:
            slot_blocked = sum(
                1 for s in self.slots.values()
                if s.status == SlotStatus.BLOCKED_AWAITING_APPROVAL and s.process_confirmed_terminated
            )
            return max(slot_blocked, len(self.paused_approval_tasks))

    def current_allowed_capacity(self) -> int:
        """
        Calculate allowed active concurrent workers.
        Starts at initial_concurrency (1).
        Expands by 1 for each blocked awaiting-approval worker whose process is confirmed terminated,
        up to max_concurrency.
        """
        with self._lock:
            blocked = self.blocked_workers_count()
            capacity = min(self.config.initial_concurrency + blocked, self.config.max_concurrency)
            return capacity

    def mark_process_surviving(self, slot_id: int, pid: int) -> None:
        """
        Simulate or record that an agent process failed to confirm termination.
        Retains the slot and quarantines the workspace against reuse.
        """
        with self._lock:
            slot = self.slots.get(slot_id)
            if not slot:
                return
            slot.process_pid = pid
            slot.process_confirmed_terminated = False
            slot.can_release_slot = False
            slot.quarantined = True
            slot.quarantine_reason = f"Process PID {pid} surviving; unconfirmed termination."
            if slot.task_id:
                self.quarantined_workspaces[slot.task_id] = slot.quarantine_reason

    def confirm_process_terminated(self, slot_id: int) -> None:
        """
        Confirm that a surviving process has terminated, allowing slot release and unquarantine.
        """
        with self._lock:
            slot = self.slots.get(slot_id)
            if not slot:
                return
            slot.process_confirmed_terminated = True
            slot.can_release_slot = True
            slot.quarantined = False
            if slot.task_id and slot.task_id in self.quarantined_workspaces:
                del self.quarantined_workspaces[slot.task_id]

    def release_slot(self, slot_id: int) -> bool:
        """
        Safely release a slot whose task has completed or paused in approval,
        provided its process is confirmed terminated and it is not quarantined.
        """
        with self._lock:
            slot = self.slots.get(slot_id)
            if not slot:
                return False
            if slot.thread and slot.thread.is_alive():
                return False
            if not slot.process_confirmed_terminated or not slot.can_release_slot or slot.quarantined:
                return False
            slot.status = SlotStatus.IDLE
            slot.task_id = None
            slot.summary = None
            slot.thread = None
            return True

    def wait_for_active_workers(self, timeout: float = 5.0) -> None:
        """Wait for all active worker threads to terminate."""
        with self._lock:
            active_slots = [s for s in self.slots.values() if s.thread and s.thread.is_alive()]
        for slot in active_slots:
            if slot.thread:
                slot.thread.join(timeout=timeout)
                if slot.thread.is_alive():
                    with self._lock:
                        slot.process_confirmed_terminated = False
                        slot.can_release_slot = False
                        logger.warning(
                            "Worker thread for slot %d (task %s) still running after %.1fs timeout; slot retained.",
                            slot.slot_id,
                            slot.task_id,
                            timeout,
                        )

    def find_next_eligible_task(self) -> Optional[TaskDefinition]:
        """
        Select the next pending task whose dependencies are ALL verified,
        which is not currently claimed or quarantined, and whose target files
        do not overlap with any currently running task.
        Filters by config.max_risk and prioritizes LOW before MEDIUM.
        """
        with self._lock:
            # Gather target files currently being modified by active running workers
            active_target_files: Set[str] = set()
            for slot in self.slots.values():
                if slot.status == SlotStatus.RUNNING and slot.task_id:
                    running_task = self.ledger.get(slot.task_id)
                    if running_task:
                        active_target_files.update(running_task.target_files)

            # Evaluate pending tasks in priority/topological order with risk limits
            scheduler = DependencyScheduler(self.ledger, repo_root=self.repo_root)
            runnable = scheduler.get_eligible_tasks(max_risk=self.config.max_risk)

            for task in runnable:
                if task.task_id in self.claimed_tasks:
                    continue
                if task.task_id in self.quarantined_workspaces:
                    continue
                if task.task_id in self.paused_approval_tasks:
                    if task.status == TaskStatus.PENDING:
                        self.paused_approval_tasks.discard(task.task_id)
                    else:
                        continue
                # Exclude tasks currently in backoff / deferral (e.g. transport pause)
                if task.task_id in self.deferred_tasks:
                    import time
                    if time.monotonic() < self.deferred_tasks[task.task_id]:
                        continue
                    else:
                        del self.deferred_tasks[task.task_id]
                # Exclude tasks that are currently active in a worker slot
                if any(s.status == SlotStatus.RUNNING and s.task_id == task.task_id for s in self.slots.values()):
                    continue
                # Allow PENDING tasks or retryable FAILED tasks (with attempts remaining or escalation eligible)
                if task.status == TaskStatus.FAILED:
                    if task.attempts >= task.max_attempts and not is_escalation_eligible(task):
                        continue
                elif task.status != TaskStatus.PENDING:
                    continue

                # Strict verification of all prerequisites
                deps_satisfied = True
                for dep_id in task.dependencies:
                    dep = self.ledger.get(dep_id)
                    if not dep or dep.status != TaskStatus.VERIFIED:
                        deps_satisfied = False
                        break

                if not deps_satisfied:
                    continue

                # File mutual exclusion check
                file_conflict = any(f in active_target_files for f in task.target_files)
                if file_conflict:
                    continue

                return task

            return None

    def _get_free_slot(self) -> Optional[WorkerSlot]:
        """Find an idle slot that is not occupied."""
        with self._lock:
            for slot in self.slots.values():
                if not slot.is_occupied() and slot.status in (
                    SlotStatus.IDLE,
                    SlotStatus.COMPLETED,
                    SlotStatus.FAILED,
                ):
                    return slot
            return None

    def _worker_thread_main(
        self,
        slot_id: int,
        task: TaskDefinition,
        authorized: bool,
        dry_run: bool,
    ) -> None:
        """Execution body running inside the worker thread."""
        orchestrator = SingleTaskOrchestrator(
            repo_root=self.repo_root,
            store=self.store,
            ledger=self.ledger,
            worktree_mgr=self.worktree_mgr,
            adapter=self.adapter,
            quality_engine=self.quality_engine,
            reviewer=self.reviewer,
            review_policy=self.review_policy,
            cache=self.cache,
            reviewer_lock=self.reviewer_lock,
            timeout_seconds=self.config.worker_timeout_sec,
            router=self.router,
            concurrency_guard=self.concurrency_guard,
            local_adapter=self.local_adapter,
        )

        try:
            summary = orchestrator.execute_single_task(
                task_id=task.task_id,
                authorized=authorized,
                dry_run=dry_run,
                acquire_coarse_lock=False,
            )
        except Exception as e:
            logger.exception(f"Unhandled exception in worker slot {slot_id} on task {task.task_id}: {e}")
            summary = ExecutionSummary(
                task_id=task.task_id,
                success=False,
                final_status=TaskStatus.FAILED,
                dry_run=dry_run,
                authorized=authorized,
                duration_sec=0.0,
                message=f"Worker exception: {e}",
            )

        with self._lock:
            slot = self.slots[slot_id]
            slot.summary = summary
            self.task_summaries[task.task_id] = summary
            self.tasks_completed_count += 1

            # Refresh task state from store to ensure consistency
            updated_task = None
            try:
                latest_ledger = self.store.load_ledger()
                updated_task = latest_ledger.get(task.task_id)
                if updated_task:
                    CoordinatorAPI._copy_ledger_into(self.ledger, latest_ledger)
                    updated_task = self.ledger.get(task.task_id)
                    task_final_status = updated_task.status
                else:
                    task_final_status = summary.final_status
            except Exception:
                task_final_status = summary.final_status

            is_transport_failure = (
                summary.agent_result is not None
                and (
                    getattr(summary.agent_result, "transport_unavailable", False)
                    or (isinstance(summary.agent_result, dict) and summary.agent_result.get("transport_unavailable", False))
                )
            ) or (
                "transport_error" in (summary.message or "")
                or "Transport unavailable" in (summary.message or "")
                or "JobTransportError" in (summary.message or "")
                or "JobAuthenticationError" in (summary.message or "")
            )

            if summary.dry_run:
                slot.status = SlotStatus.COMPLETED
            elif is_transport_failure:
                # Operational transport pause: defer task with backoff, free slot, preserve repair budget
                import time
                self.deferred_tasks[task.task_id] = time.monotonic() + 5.0
                if task.task_id in self.claimed_tasks:
                    self.claimed_tasks.remove(task.task_id)
                slot.status = SlotStatus.IDLE
                slot.task_id = None
                slot.summary = None
            elif task_final_status == TaskStatus.NEEDS_APPROVAL:
                if slot.process_confirmed_terminated:
                    slot.status = SlotStatus.BLOCKED_AWAITING_APPROVAL
                    self.paused_approval_tasks.add(task.task_id)
                    if task.task_id in self.claimed_tasks:
                        self.claimed_tasks.remove(task.task_id)
                else:
                    slot.status = SlotStatus.RECOVERY_REQUIRED
            elif task_final_status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED):
                slot.status = SlotStatus.COMPLETED
            elif task_final_status == TaskStatus.RECOVERY_REQUIRED:
                slot.status = SlotStatus.RECOVERY_REQUIRED
            else:
                slot.status = SlotStatus.FAILED
                # If failure is recoverable with remaining attempts, allow retry
                attempts = updated_task.attempts if updated_task else task.attempts
                max_attempts = updated_task.max_attempts if updated_task else task.max_attempts
                if attempts < max_attempts and slot.process_confirmed_terminated:
                    if task.task_id in self.claimed_tasks:
                        self.claimed_tasks.remove(task.task_id)

            # Check stop-on-failure condition: permission blocks and transport pauses are operational pauses, not failures
            if task_final_status not in (
                TaskStatus.NEEDS_APPROVAL, TaskStatus.READY_TO_INTEGRATE,
                TaskStatus.VERIFIED, TaskStatus.PENDING,
            ) and not is_transport_failure:
                if slot.status in (SlotStatus.FAILED, SlotStatus.RECOVERY_REQUIRED) or not summary.success:
                    attempts = updated_task.attempts if updated_task else task.attempts
                    max_attempts = updated_task.max_attempts if updated_task else task.max_attempts
                    is_recoverable = (
                        (attempts < max_attempts)
                        and slot.process_confirmed_terminated
                        and (slot.status != SlotStatus.RECOVERY_REQUIRED)
                    )
                    if not is_recoverable and self.config.stop_on_failure:
                        self.has_halted_due_to_failure = True
                        self.halt_reason = summary.message or f"Task '{task.task_id}' exhausted retries or suffered unrecoverable failure."

            # Save per-task isolated markdown report
            try:
                reports_dir = self.repo_root / ".maya" / "reports"
                generate_task_report(
                    task_id=task.task_id,
                    title=task.title,
                    priority=task.priority.value,
                    risk=task.risk.value,
                    success=summary.success,
                    final_status=task_final_status.value if hasattr(task_final_status, "value") else str(task_final_status),
                    duration_sec=summary.duration_sec,
                    worktree_path=summary.worktree_path,
                    quality_report=summary.quality_report,
                    agent_result=summary.agent_result,
                    reports_dir=reports_dir,
                )
            except Exception as e:
                logger.warning("Failed to generate per-task report for %s: %s", task.task_id, e)

    def spawn_worker(
        self,
        task: TaskDefinition,
        authorized: bool = True,
        dry_run: bool = False,
    ) -> Optional[WorkerSlot]:
        """
        Spawn a worker in an idle slot to execute the given task.
        Returns the assigned WorkerSlot, or None if capacity, slots, or task caps are exhausted.
        """
        with self._lock:
            if self.config.stop_on_failure and self.has_halted_due_to_failure:
                return None

            running_count = self.active_running_workers_count()

            # Enforce max_tasks bounding
            if self.config.max_tasks is not None:
                if self.tasks_completed_count + running_count >= self.config.max_tasks:
                    return None

            capacity = self.current_allowed_capacity()

            if running_count >= capacity or running_count >= self.config.max_concurrency:
                return None

            slot = self._get_free_slot()
            if not slot:
                return None

            slot.task_id = task.task_id
            slot.status = SlotStatus.RUNNING
            slot.start_time = time.monotonic()
            slot.summary = None
            slot.can_release_slot = True
            slot.process_confirmed_terminated = True

            self.claimed_tasks.add(task.task_id)

            # Update peak concurrency
            current_running = running_count + 1
            if current_running > self.peak_concurrency:
                self.peak_concurrency = current_running

            # Launch thread
            t = threading.Thread(
                target=self._worker_thread_main,
                args=(slot.slot_id, task, authorized, dry_run),
                name=f"WorkerSlot-{slot.slot_id}-{task.task_id}",
                daemon=True,
            )
            slot.thread = t
            t.start()
            return slot

    def step(self, authorized: bool = True, dry_run: bool = False) -> int:
        """
        Perform one iteration of scheduler lifecycle:
        1. Harvest finished worker threads and release their slots if terminated.
        2. Determine current dynamic capacity.
        3. Spawn new eligible workers up to allowed capacity and task cap.
        Returns the number of workers actively running after the step.
        """
        with self._lock:
            # 1. Harvest completed or terminated threads
            for slot in self.slots.values():
                if slot.thread and not slot.thread.is_alive():
                    slot.thread.join(timeout=0.01)
                    slot.thread = None

                    # If slot finished and process is confirmed dead, slot can be released
                    if slot.status in (SlotStatus.COMPLETED, SlotStatus.FAILED):
                        if slot.process_confirmed_terminated and slot.can_release_slot:
                            slot.status = SlotStatus.IDLE
                            slot.task_id = None
                            slot.summary = None
                    elif slot.status in (SlotStatus.BLOCKED_AWAITING_APPROVAL, SlotStatus.RECOVERY_REQUIRED):
                        if slot.process_confirmed_terminated and slot.can_release_slot and not slot.quarantined:
                            slot.status = SlotStatus.IDLE
                            slot.task_id = None
                            slot.summary = None
                elif slot.thread is None and slot.status == SlotStatus.BLOCKED_AWAITING_APPROVAL:
                    # In single-worker mode (max_concurrency == 1), release idle blocked slots
                    if self.config.max_concurrency == 1:
                        if slot.process_confirmed_terminated and slot.can_release_slot and not slot.quarantined:
                            slot.status = SlotStatus.IDLE
                            slot.task_id = None
                            slot.summary = None

            if self.config.stop_on_failure and self.has_halted_due_to_failure:
                return self.active_running_workers_count()

            # 2. Spawn next tasks while capacity and max_tasks permit
            capacity = self.current_allowed_capacity()
            running = self.active_running_workers_count()

            while running < capacity and running < self.config.max_concurrency:
                if self.config.max_tasks is not None:
                    if self.tasks_completed_count + running >= self.config.max_tasks:
                        break

                next_task = self.find_next_eligible_task()
                if not next_task:
                    break

                spawned = self.spawn_worker(next_task, authorized=authorized, dry_run=dry_run)
                if not spawned:
                    break
                running = self.active_running_workers_count()

            return running

    @property
    def session_file_path(self) -> Path:
        return self.repo_root / ".maya" / "maintenance" / "active_session.json"

    def load_active_session(self) -> Optional[SessionState]:
        """Load active session state from disk if present."""
        if not self.session_file_path.exists():
            return None
        with open(self.session_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionState.from_dict(data)

    def save_active_session(self, session: SessionState) -> None:
        """Persist active session state atomically to disk."""
        try:
            self.session_file_path.parent.mkdir(parents=True, exist_ok=True)
            session.updated_at = utc_now_iso()
            tmp = self.session_file_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            os.replace(tmp, self.session_file_path)
        except Exception as e:
            logger.warning("Failed to save active session to %s: %s", self.session_file_path, e)

    def run_until_idle(
        self,
        authorized: bool = True,
        dry_run: bool = False,
        max_duration_sec: Optional[float] = None,
        force_new_session: bool = False,
        resume: bool = False,
    ) -> AdaptiveSchedulerSummary:
        """
        Execute adaptive scheduler loop until all eligible tasks are complete,
        task cap is reached, failure halts loop, or session deadline expires.
        Enforces absolute monotonic deadline in-process and bounds resumed sessions
        against their persisted original deadline without granting fresh time.
        """
        # 1. Resolve and validate duration
        raw_duration = self.config.hard_timeout_sec if max_duration_sec is None else max_duration_sec
        if raw_duration is None:
            raw_duration = 21600.0

        if math.isnan(raw_duration) or math.isinf(raw_duration):
            raise ValueError(f"Session duration must be a finite number, got {raw_duration}")
        if raw_duration <= 0:
            raise ValueError(f"Session duration must be a positive number, got {raw_duration}")
        if raw_duration > 86400.0:
            raise ValueError(f"Session duration cannot exceed 86400.0 seconds (24 hours), got {raw_duration}")

        requested_duration = float(raw_duration)

        # 2. Determine session lifecycle: New vs Resume
        now_utc = datetime.now(timezone.utc)
        now_ts = now_utc.timestamp()

        active_session: Optional[SessionState] = None
        session_file_exists = self.session_file_path.exists()
        if session_file_exists:
            try:
                active_session = self.load_active_session()
            except Exception as e:
                if not force_new_session:
                    raise ValueError(
                        f"Active session state at '{self.session_file_path}' is malformed or corrupted: {e}. "
                        "Refusing to grant fresh runtime. Run with --new-session to reset."
                    ) from e
                logger.warning("Active session file corrupted, but --new-session requested: starting fresh.")

        if resume:
            if not session_file_exists or active_session is None:
                raise ValueError(
                    f"Cannot resume: active session file not found at '{self.session_file_path}'. "
                    "Run with --new-session to start a fresh session."
                )
            if active_session.status == "COMPLETED":
                raise ValueError(
                    f"Cannot resume session '{active_session.session_id}': session already COMPLETED. "
                    "Run with --new-session to start a fresh session."
                )

        is_resuming = (
            not force_new_session
            and active_session is not None
            and (resume or active_session.status in ("RUNNING", "INTERRUPTED", "MANUAL_STOP"))
        )

        if is_resuming:
            assert active_session is not None
            remaining_sec = active_session.deadline_timestamp - now_ts
            active_session.resumed_count += 1
            active_session.status = "RUNNING"
            current_session = active_session
            effective_duration_sec = max(0.0, remaining_sec)
            logger.info(
                "Resuming session '%s': %.1fs remaining against original deadline %s",
                current_session.session_id,
                effective_duration_sec,
                current_session.deadline_iso,
            )
        else:
            session_id = now_utc.strftime("%Y%m%d-%H%M%S")
            deadline_ts = now_ts + requested_duration
            deadline_iso = datetime.fromtimestamp(deadline_ts, tz=timezone.utc).isoformat()
            current_session = SessionState(
                session_id=session_id,
                status="RUNNING",
                requested_duration_sec=requested_duration,
                start_time_iso=now_utc.isoformat(),
                deadline_iso=deadline_iso,
                deadline_timestamp=deadline_ts,
            )
            effective_duration_sec = requested_duration

        self.save_active_session(current_session)

        # 3. In-process monotonic clock deadline
        session_start_mono = time.monotonic()
        deadline_mono = session_start_mono + effective_duration_sec

        # 4. If deadline has already elapsed (e.g. while stopped), halt immediately
        if effective_duration_sec <= 0.0:
            logger.warning(
                "Session '%s' deadline (%s) has already expired. Halting without scheduling.",
                current_session.session_id,
                current_session.deadline_iso,
            )
            self.has_halted_due_to_failure = True
            self.halt_reason = f"Session runtime limit expired (deadline was {current_session.deadline_iso})"
            current_session.status = "EXPIRED"
            current_session.halt_reason = self.halt_reason
            self.save_active_session(current_session)
            return self._compile_summary(session_start_mono, current_session)

        # 5. Startup reconciliation & preflight integrity
        if not dry_run:
            with self.store.acquire_lock():
                self.worktree_mgr.reconcile_all(
                    self.ledger, coordinator_api=self.coordinator_api, lock_held=True
                )

            valid, integrity_reason = self.validate_preflight_integrity()
            if not valid:
                logger.error("Preflight integrity check failed before loop start: %s", integrity_reason)
                self.has_halted_due_to_failure = True
                self.halt_reason = f"Preflight integrity validation failed: {integrity_reason}"
                current_session.status = "INTERRUPTED"
                current_session.halt_reason = self.halt_reason
                self.save_active_session(current_session)
                return self._compile_summary(session_start_mono, current_session)

        # 6. Main scheduling loop
        try:
            while True:
                with self._lock:
                    if self.config.stop_on_failure and self.has_halted_due_to_failure:
                        logger.warning("Adaptive scheduler stopped on failure: %s", self.halt_reason)
                        current_session.status = "INTERRUPTED"
                        current_session.halt_reason = self.halt_reason
                        break

                    if self.config.max_tasks is not None and self.tasks_completed_count >= self.config.max_tasks:
                        logger.info("Adaptive scheduler reached max_tasks limit (%s); stopping loop.", self.config.max_tasks)
                        current_session.status = "COMPLETED"
                        break

                # Deadline check before scheduling
                time_remaining = deadline_mono - time.monotonic()
                if time_remaining <= 0:
                    logger.warning(
                        "Adaptive scheduler reached session deadline (budget: %.1fs); stopping new task scheduling.",
                        effective_duration_sec,
                    )
                    self.halt_reason = f"Session runtime limit expired: Hard timeout reached ({current_session.requested_duration_sec}s deadline reached)"
                    current_session.status = "EXPIRED"
                    current_session.halt_reason = self.halt_reason
                    break

                running = self.step(authorized=authorized, dry_run=dry_run)

                with self._lock:
                    has_running = running > 0
                    next_eligible = self.find_next_eligible_task()

                if not has_running and not next_eligible:
                    # No workers active and no tasks eligible to start
                    current_session.status = "COMPLETED"
                    break

                # Deadline check after scheduling
                time_remaining = deadline_mono - time.monotonic()
                if time_remaining <= 0:
                    logger.warning(
                        "Adaptive scheduler reached session deadline (budget: %.1fs); stopping new task scheduling.",
                        effective_duration_sec,
                    )
                    self.halt_reason = f"Session runtime limit expired: Hard timeout reached ({current_session.requested_duration_sec}s deadline reached)"
                    current_session.status = "EXPIRED"
                    current_session.halt_reason = self.halt_reason
                    break

                sleep_duration = min(self.config.poll_interval_sec, max(0.001, time_remaining))
                time.sleep(sleep_duration)

        except KeyboardInterrupt:
            logger.warning("Adaptive scheduler interrupted by operator (Ctrl+C). Initiating safe shutdown...")
            self.halt_reason = "Manual stop requested by operator (Ctrl+C)"
            current_session.status = "MANUAL_STOP"
            current_session.halt_reason = self.halt_reason
        finally:
            # Safe shutdown: wait for in-flight workers to terminate cleanly
            for slot in self.slots.values():
                if slot.thread and slot.thread.is_alive():
                    slot.thread.join(timeout=5.0)
                    if slot.thread.is_alive():
                        slot.process_confirmed_terminated = False
                        slot.can_release_slot = False
                        logger.warning(
                            "Worker thread for slot %d (task %s) still running after 5.0s shutdown timeout; slot retained.",
                            slot.slot_id,
                            slot.task_id,
                        )

            self.save_active_session(current_session)

        return self._compile_summary(session_start_mono, current_session)

    def _compile_summary(
        self,
        start_time: float,
        session: Optional[SessionState] = None,
    ) -> AdaptiveSchedulerSummary:
        duration = time.monotonic() - start_time

        # Compile final summary
        verified_count = sum(1 for t in self.ledger if t.status == TaskStatus.VERIFIED)
        failed_count = sum(1 for t in self.ledger if t.status == TaskStatus.FAILED)
        blocked_count = sum(
            1 for t in self.ledger
            if t.status in (TaskStatus.BLOCKED, TaskStatus.NEEDS_APPROVAL)
        )

        slot_summaries = [
            {
                "slot_id": s.slot_id,
                "status": s.status.value,
                "task_id": s.task_id,
                "process_confirmed_terminated": s.process_confirmed_terminated,
                "can_release_slot": s.can_release_slot,
                "quarantined": s.quarantined,
            }
            for s in self.slots.values()
        ]

        task_dict = {
            t_id: summary.to_dict() for t_id, summary in self.task_summaries.items()
        }

        requested_duration = session.requested_duration_sec if session else self.config.hard_timeout_sec
        start_iso = session.start_time_iso if session else None
        deadline_iso = session.deadline_iso if session else None
        halt_reason = self.halt_reason or (session.halt_reason if session else None)
        if session:
            remaining_sec = max(0.0, session.deadline_timestamp - datetime.now(timezone.utc).timestamp())
        else:
            remaining_sec = 0.0

        summary_obj = AdaptiveSchedulerSummary(
            total_tasks=len(self.ledger),
            completed_tasks=len(self.task_summaries),
            verified_tasks=verified_count,
            failed_tasks=failed_count,
            blocked_tasks=blocked_count,
            peak_concurrency=self.peak_concurrency,
            quarantined_count=len(self.quarantined_workspaces),
            duration_sec=duration,
            slot_summaries=slot_summaries,
            task_summaries=task_dict,
            requested_duration_sec=requested_duration,
            session_start_time=start_iso,
            session_deadline=deadline_iso,
            halt_reason=halt_reason,
            remaining_duration_sec=remaining_sec,
        )

        # Save durable session summary
        try:
            session_id = session.session_id if session else datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            reports_dir = self.repo_root / ".maya" / "reports"
            generate_session_summary(
                session_id=session_id,
                scheduler_summary=summary_obj.to_dict(),
                reports_dir=reports_dir,
            )
        except Exception as e:
            logger.warning("Failed to save session summary report: %s", e)

        return summary_obj
