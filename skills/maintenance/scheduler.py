"""
Deterministic, dependency-aware task scheduler for MayaVE Maintenance Coordinator.
Enforces risk-tier gating, unintegrated prerequisite code isolation, overlapping file mutual exclusion,
and priority/downstream-impact scheduling.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Tuple

try:
    from .ledger import TaskLedger
    from .models import TaskDefinition, TaskRisk, TaskStatus
except ImportError:
    from ledger import TaskLedger
    from models import TaskDefinition, TaskRisk, TaskStatus

RISK_RANK = {
    TaskRisk.LOW: 0,
    TaskRisk.MEDIUM: 1,
    TaskRisk.HIGH: 2,
    TaskRisk.VERY_HIGH: 3,
}


class DependencyScheduler:
    """Calculates task eligibility and schedules tasks according to dependency graph, risk tiers, and priorities."""

    def __init__(
        self,
        ledger: TaskLedger,
        repo_root: Path | str | None = None,
        check_unintegrated: bool = True,
    ) -> None:
        self._ledger = ledger
        self._repo_root = Path(repo_root).resolve() if repo_root else None
        self._check_unintegrated = check_unintegrated
        # Ensure graph integrity before scheduling
        self._ledger.validate_graph()

    @property
    def ledger(self) -> TaskLedger:
        return self._ledger

    def _is_task_unintegrated(self, task: TaskDefinition) -> bool:
        """
        Check if a verified task has unintegrated changes in its worktree that are not in repo baseline.
        """
        if not task.workspace:
            return False

        wt_path = Path(task.workspace.worktree_path)
        if not wt_path.exists():
            return False

        # 1. Dirty worktree (uncommitted changes)
        if task.workspace.is_dirty:
            return True

        repo_root = self._repo_root
        if not repo_root:
            if wt_path.parent.name == "worktrees" and wt_path.parent.parent.name == ".maya":
                repo_root = wt_path.parent.parent.parent
            else:
                repo_root = wt_path.parent.parent

        # 2. Check if all target files exist in repo baseline
        for tf in task.target_files:
            if not (repo_root / tf).exists():
                return True

        # 3. Check git status of worktree directly
        try:
            st = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(wt_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if st.returncode == 0 and st.stdout.strip():
                return True
        except Exception:
            pass

        # 4. Check if branch is merged into main
        if task.workspace.branch_name:
            try:
                res = subprocess.run(
                    ["git", "branch", "--merged", "main"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0:
                    merged = [b.strip().lstrip("*+ ") for b in res.stdout.splitlines()]
                    if task.workspace.branch_name not in merged:
                        return True
            except Exception:
                pass

        return False

    def check_task_eligibility(
        self,
        task_id: str,
        max_risk: TaskRisk | str | None = TaskRisk.MEDIUM,
    ) -> Tuple[bool, str]:
        """
        Evaluate if a task can be scheduled right now.
        Returns (is_eligible, reason_string).
        """
        task = self._ledger.get_task(task_id)

        # 1. State check
        if task.status == TaskStatus.VERIFIED:
            return False, "Task is already VERIFIED."
        if task.status in (TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.INTEGRATING):
            return False, f"Task is currently active in state {task.status.value}."
        if task.status == TaskStatus.READY_TO_INTEGRATE:
            return False, "Task has passed candidate verification and is awaiting approved integration."
        if task.status == TaskStatus.BLOCKED:
            return False, f"Task is BLOCKED: {task.blocking_reason or 'manual block'}"
        if task.status == TaskStatus.NEEDS_APPROVAL:
            return False, f"Task is pending human authorization (NEEDS_APPROVAL): {task.blocking_reason or 'Pending review'}"
        if task.status == TaskStatus.RECOVERY_REQUIRED:
            return False, f"Task requires recovery reconciliation (RECOVERY_REQUIRED): {task.blocking_reason or 'Interrupted'}"
        if task.status == TaskStatus.FAILED:
            if task.attempts >= task.max_attempts:
                return False, f"Task has reached maximum attempts ({task.attempts}/{task.max_attempts})."

        # 2. Risk check
        if task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH):
            return False, f"Task risk '{task.risk.value}' is excluded from unattended execution and requires interactive human authorization."

        if max_risk is not None:
            max_risk_enum = TaskRisk(max_risk) if isinstance(max_risk, str) else max_risk
            current_rank = RISK_RANK.get(task.risk, 99)
            allowed_rank = RISK_RANK.get(max_risk_enum, 99)
            if current_rank > allowed_rank:
                return False, f"Task risk '{task.risk.value}' exceeds allowed maximum '{max_risk_enum.value}'."

        # 3. Dependency checks
        for dep_id in task.dependencies:
            dep_task = self._ledger.get_task(dep_id)
            if dep_task.status != TaskStatus.VERIFIED:
                return (
                    False,
                    f"Prerequisite '{dep_id}' is not VERIFIED (current state: {dep_task.status.value}).",
                )

        # 4. Unintegrated prerequisite and overlapping file checks
        if self._check_unintegrated:
            for other_task in self._ledger:
                if other_task.status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.INTEGRATING, TaskStatus.VERIFIED) and self._is_task_unintegrated(other_task):
                    # Overlap with unintegrated verified task
                    overlap = set(task.target_files).intersection(set(other_task.target_files))
                    if overlap:
                        return (
                            False,
                            f"Task '{task.task_id}' overlaps target files ({', '.join(sorted(overlap))}) "
                            f"with unintegrated verified task '{other_task.task_id}'. Explicit baseline integration required.",
                        )
                    # Downstream dependency on unintegrated task
                    if other_task.task_id in task.dependencies:
                        return (
                            False,
                            f"Prerequisite '{other_task.task_id}' is VERIFIED in isolated worktree but unintegrated into main; "
                            f"prerequisite code is not present in baseline HEAD.",
                        )

        return True, "All prerequisites are VERIFIED and task is ready."

    def get_eligible_tasks(self, max_risk: TaskRisk | str | None = TaskRisk.MEDIUM) -> list[TaskDefinition]:
        """
        Return all tasks currently eligible for execution, sorted by:
        1. Risk tier: LOW (0) before MEDIUM (1).
        2. Priority rank: P0 (0), P1 (1), P2 (2), P3 (3).
        3. Downstream dependents count descending (prefer tasks that unblock more work).
        4. Stable task ID tie-breaking.
        """
        eligible: list[TaskDefinition] = []
        for task in self._ledger:
            is_ok, _ = self.check_task_eligibility(task.task_id, max_risk=max_risk)
            if is_ok:
                eligible.append(task)

        # Count downstream dependents for each task
        downstream_counts: dict[str, int] = {}
        for t in self._ledger:
            downstream_counts[t.task_id] = sum(
                1 for other in self._ledger if t.task_id in other.dependencies
            )

        # Deterministic sort:
        # 1. Risk rank: LOW (0) before MEDIUM (1) before HIGH (2)
        # 2. Attempts rank: unattempted (0) before retryable failures (1, 2)
        # 3. Priority rank: P0 (0), P1 (1), P2 (2), P3 (3)
        # 4. Downstream dependents count descending (prefer tasks that unblock more work)
        # 5. Stable task ID tie-breaking
        eligible.sort(
            key=lambda t: (
                RISK_RANK.get(t.risk, 99),
                t.attempts,
                t.priority.rank,
                -downstream_counts.get(t.task_id, 0),
                t.task_id,
            )
        )
        return eligible

    def get_next_task(self, max_risk: TaskRisk | str | None = TaskRisk.MEDIUM) -> TaskDefinition | None:
        """Return the single highest-priority eligible task, or None if no tasks can run."""
        eligible = self.get_eligible_tasks(max_risk=max_risk)
        return eligible[0] if eligible else None

    def get_blocked_tasks(self, max_risk: TaskRisk | str | None = TaskRisk.MEDIUM) -> list[tuple[TaskDefinition, str]]:
        """Return tasks that cannot run because dependencies are unfulfilled or blocked."""
        blocked: list[tuple[TaskDefinition, str]] = []
        for task in self._ledger:
            if task.status == TaskStatus.VERIFIED:
                continue
            is_ok, reason = self.check_task_eligibility(task.task_id, max_risk=max_risk)
            if not is_ok and task.status not in (TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING):
                blocked.append((task, reason))
        blocked.sort(key=lambda item: (RISK_RANK.get(item[0].risk, 99), item[0].priority.rank, item[0].task_id))
        return blocked

    def get_status_summary(self, max_risk: TaskRisk | str | None = TaskRisk.MEDIUM) -> dict[str, int]:
        """Return counts by task status."""
        summary: dict[str, int] = {status.value: 0 for status in TaskStatus}
        for task in self._ledger:
            summary[task.status.value] += 1
        summary["TOTAL"] = len(self._ledger)
        summary["ELIGIBLE"] = len(self.get_eligible_tasks(max_risk=max_risk))
        return summary
