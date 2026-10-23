#!/usr/bin/env python3
"""
MayaVE VE11 — Overnight Maintenance Coordinator CLI.
Milestone 1: Ledger, Dependency Resolution, Pre-Validation & Dry-Run Engine.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Ensure UTF-8 output on Windows consoles if available
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
_MAINTENANCE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MAINTENANCE_DIR.parent.parent
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .adapter import (
        AgentAdapter,
        AgentExecutionResult,
        AntigravityAgentAdapter,
        FakeAgentAdapter,
        LocalModelAgentAdapter,
        TieredAgentAdapter,
    )
    from .adaptive import (
        AdaptiveConfig,
        AdaptiveSchedulerSummary,
        AdaptiveWorkerScheduler,
        SlotStatus,
        WorkerSlot,
    )
    from .cache import ContextCache
    from .inventory import get_default_audit_inventory
    from .hub import CoordinatorAPI, compute_patch_sha256, compute_verification_sha256
    from .ledger import LedgerValidationError, TaskLedger
    from .models import CheckpointRecord, PlanProposal, TaskDefinition, TaskRisk, TaskStatus, utc_now_iso
    from .planner import LocalPlanner, PlannerConfig
    from .orchestrator import ExecutionSummary, SingleTaskOrchestrator
    from .quality import QualityGateEngine
    from .reviewer import (
        CopilotReviewerAdapter,
        FakeReviewerAdapter,
        ReviewPolicy,
        ReviewerAdapter,
    )
    from .scheduler import DependencyScheduler
    from .store import CorruptStateError, LockAcquisitionError, StateStore
    from .terminal_approval import (
        ApprovalStatus,
        TerminalApprovalManager,
    )
    from .validator import PreValidationStatus, PreValidator
    from .routing import ComplexityRouter, RoutingConfig, WorkerConcurrencyGuard
    from .registry import ModelRegistry, ProviderType, ModelStatus
    from .benchmark import Tier2BenchmarkHarness, generate_benchmark_report
    from .task_contract import TaskContractValidator, TaskContractValidationError
    from .worktree import (
        WorkspaceConflictError,
        WorkspaceError,
        WorkspaceValidationError,
        WorktreeManager,
    )
except ImportError:
    from adapter import (
        AgentAdapter,
        AgentExecutionResult,
        AntigravityAgentAdapter,
        FakeAgentAdapter,
        LocalModelAgentAdapter,
        TieredAgentAdapter,
    )
    from adaptive import (
        AdaptiveConfig,
        AdaptiveSchedulerSummary,
        AdaptiveWorkerScheduler,
        SlotStatus,
        WorkerSlot,
    )
    from cache import ContextCache
    from inventory import get_default_audit_inventory
    from hub import CoordinatorAPI, compute_patch_sha256, compute_verification_sha256
    from ledger import LedgerValidationError, TaskLedger
    from models import CheckpointRecord, PlanProposal, TaskDefinition, TaskRisk, TaskStatus, utc_now_iso
    from planner import LocalPlanner, PlannerConfig
    from orchestrator import ExecutionSummary, SingleTaskOrchestrator
    from quality import QualityGateEngine
    from reviewer import (
        CopilotReviewerAdapter,
        FakeReviewerAdapter,
        ReviewPolicy,
        ReviewerAdapter,
    )
    from scheduler import DependencyScheduler
    from store import CorruptStateError, LockAcquisitionError, StateStore
    from terminal_approval import (
        ApprovalStatus,
        TerminalApprovalManager,
    )
    from validator import PreValidationStatus, PreValidator
    from routing import ComplexityRouter, RoutingConfig, WorkerConcurrencyGuard
    from registry import ModelRegistry, ProviderType, ModelStatus
    from benchmark import Tier2BenchmarkHarness, generate_benchmark_report
    from task_contract import TaskContractValidator, TaskContractValidationError
    from worktree import (
        WorkspaceConflictError,
        WorkspaceError,
        WorkspaceValidationError,
        WorktreeManager,
    )


def _resolve_default_paths(custom_state: str | None, custom_root: str | None) -> tuple[Path, Path]:
    repo_root = Path(custom_root).resolve() if custom_root else _REPO_ROOT
    if custom_state:
        state_path = Path(custom_state).resolve()
    else:
        state_path = repo_root / ".maya" / "maintenance" / "state.json"
    return repo_root, state_path


def cmd_init(store: StateStore, force: bool = False) -> int:
    """Initialize state ledger with default audit tasks."""
    if store.exists() and not force:
        print(f"[ERROR] State file already exists at: {store.state_path}")
        print("Use --force with --init to overwrite an existing ledger.")
        return 1

    tasks = get_default_audit_inventory()
    ledger = TaskLedger(tasks)
    try:
        with store.acquire_lock(force_unlock=force):
            store.save_ledger(ledger)
        print(f"[SUCCESS] Initialized task ledger with {len(ledger)} audit tasks at: {store.state_path}")
        return 0
    except LockAcquisitionError as e:
        print(f"[ERROR] Cannot acquire lock: {e}")
        return 1


def _load_or_mock_ledger(store: StateStore) -> tuple[TaskLedger, bool]:
    """Load ledger from disk, or return in-memory default inventory if file does not exist yet."""
    if store.exists():
        return store.load_ledger(), True
    # In-memory default for dry-run simulation
    return TaskLedger(get_default_audit_inventory()), False


def cmd_status(store: StateStore, repo_root: Path | None = None) -> int:
    """Display comprehensive status table."""
    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[ERROR] State file is corrupt: {e}")
        return 2

    scheduler = DependencyScheduler(ledger, repo_root=repo_root)
    summary = scheduler.get_status_summary()

    persisted_msg = f"Persisted ({store.state_path})" if is_persisted else "In-Memory Simulation (Uninitialized)"
    print(f"\n=== MayaVE Maintenance Coordinator Status [{persisted_msg}] ===")
    print(
        f"Total: {summary['TOTAL']} | Eligible: {summary['ELIGIBLE']} | "
        f"Pending: {summary['PENDING']} | Ready to Integrate: {summary.get('READY_TO_INTEGRATE', 0)} | "
        f"Verified: {summary['VERIFIED']} | Blocked: {summary['BLOCKED']} | "
        f"Recovery Required: {summary.get('RECOVERY_REQUIRED', 0)} | "
        f"Failed: {summary['FAILED']} | Needs Approval: {summary['NEEDS_APPROVAL']}\n"
    )

    print(f"{'Task ID':<8} {'Pri':<4} {'Risk':<7} {'Status':<15} {'Dependencies':<18} {'Title'}")
    print("-" * 88)
    for task in ledger:
        deps_str = ", ".join(task.dependencies) if task.dependencies else "-"
        print(
            f"{task.task_id:<8} {task.priority.value:<4} {task.risk.value:<7} "
            f"{task.status.value:<15} {deps_str:<18} {task.title[:35]}"
        )
    print("-" * 88)

    recovery_tasks = [t for t in ledger if t.status == TaskStatus.RECOVERY_REQUIRED]
    if recovery_tasks:
        print("\nTasks Requiring Recovery:")
        for t in recovery_tasks:
            reason = t.blocking_reason or t.failure_reason or "Interrupted execution or failed verification."
            print(f"  - [{t.task_id}] {t.title}")
            print(f"      Reason : {reason}")
            print(f"      Action : Inspect worktree status and reconcile via '--reconcile'. Approvals cannot bypass RECOVERY_REQUIRED.")

    eligible = scheduler.get_eligible_tasks()
    if eligible:
        next_task = eligible[0]
        print(f"\nNext Eligible Task: [{next_task.task_id}] {next_task.title} (Priority: {next_task.priority.value})")
    else:
        print("\nNo tasks currently eligible for execution.")

    return 0


def cmd_next(store: StateStore, repo_root: Path | None = None) -> int:
    """Output only the next eligible task ID and details."""
    try:
        ledger, _ = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[ERROR] State file is corrupt: {e}")
        return 2

    scheduler = DependencyScheduler(ledger, repo_root=repo_root)
    next_task = scheduler.get_next_task()

    if not next_task:
        print("NONE")
        return 0

    print(f"TASK_ID={next_task.task_id}")
    print(f"TITLE={next_task.title}")
    print(f"PRIORITY={next_task.priority.value}")
    print(f"RISK={next_task.risk.value}")
    print(f"AUDIT_SOURCE={next_task.audit_source}")
    print(f"TARGET_FILES={','.join(next_task.target_files)}")
    print(f"TEST_TARGETS={','.join(next_task.test_targets)}")
    return 0


def cmd_dry_run(repo_root: Path, store: StateStore) -> int:
    """Perform non-destructive pre-validation, dependency evaluation, and execution simulation."""
    print(f"\n=== MayaVE Maintenance Coordinator — Dry-Run Simulation ===")
    print(f"Repository Root : {repo_root}")
    print(f"State Store     : {store.state_path} ({'Found' if store.exists() else 'In-Memory Default'})")

    validator = PreValidator(repo_root)

    # 1. Repo structure
    repo_ok, repo_msg = validator.check_repo_structure()
    if not repo_ok:
        print(f"[FAIL] Repository structure check failed: {repo_msg}")
        return 1
    print(f"[PASS] {repo_msg}")

    # 2. Git baseline
    git_info = validator.inspect_git_baseline()
    print(f"[INFO] Git Branch: {git_info.branch} (Commit: {git_info.head_sha[:8]})")
    if git_info.is_clean:
        print("[INFO] Working Tree: Clean (0 modified, 0 untracked).")
    else:
        print(
            f"[INFO] Working Tree: Contains pre-existing changes "
            f"({len(git_info.modified_files)} modified, {len(git_info.untracked_files)} untracked). "
            f"Preserving all user changes without modification."
        )

    # 3. Ledger loading
    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file corruption detected: {e}")
        return 2

    # 4. Dependency graph validation
    try:
        ledger.validate_graph()
        print(f"[PASS] Dependency graph validated (0 cycles, 0 missing dependencies across {len(ledger)} tasks).")
    except LedgerValidationError as e:
        print(f"[FAIL] Dependency graph validation error: {e}")
        return 1

    # 5. Pre-validation of audit findings
    report = validator.run_pre_validation(ledger)
    scheduler = DependencyScheduler(ledger, repo_root=repo_root)
    eligible = scheduler.get_eligible_tasks()
    blocked = scheduler.get_blocked_tasks()

    print("\n--- Finding Pre-Validation & Defect State ---")
    for check in report.tasks_checked:
        symbol = "+" if check.pre_validation_status == PreValidationStatus.CONFIRMED_UNPATCHED else "."
        print(
            f"[{symbol}] {check.task_id:<6} {check.priority.value:<3} "
            f"{check.pre_validation_status.value:<20} : {check.details}"
        )

    print("\n--- Scheduling & Eligibility Preview ---")
    print(f"Total Tasks Ingested : {len(ledger)}")
    print(f"Eligible For Next Run: {len(eligible)}")
    print(f"Blocked On Prereqs   : {len(blocked)}")

    if eligible:
        print("\nEligible Queue (Priority Order):")
        for i, t in enumerate(eligible, 1):
            deps = f"(Dependencies: {', '.join(t.dependencies)})" if t.dependencies else "(No dependencies)"
            print(f"  {i}. [{t.task_id}] {t.priority.value} - {t.title} {deps}")

        next_up = eligible[0]
        print(f"\n>>> NEXT SCHEDULED TASK WOULD BE: [{next_up.task_id}] {next_up.title} <<<")
        print(f"    Target Files: {', '.join(next_up.target_files)}")
        print(f"    Test Targets: {', '.join(next_up.test_targets)}")
    else:
        print("\n[NOTICE] No tasks are currently eligible to run.")

    if blocked:
        print("\nBlocked Queue Preview:")
        for t, reason in blocked:
            print(f"  - [{t.task_id}] {t.priority.value} : {reason}")

    print("\n[SAFETY CONFIRMATION]")
    print("[OK] Dry-run completed successfully in strictly read-only mode.")
    print("[OK] Zero agents or models were invoked.")
    print("[OK] Zero terminal test commands were executed.")
    print("[OK] Zero MayaVE application files were modified.")
    print("[OK] Git state and working tree were left completely untouched.")
    return 0


def cmd_prepare_workspace(repo_root: Path, store: StateStore, task_id: str) -> int:
    """Explicitly prepare an isolated Git worktree for the requested task."""
    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file is corrupt: {e}")
        return 2

    if task_id not in ledger:
        print(f"[FAIL] Task '{task_id}' not found in ledger.")
        return 1

    task = ledger.get_task(task_id)

    # 1. Status constraint: Only PENDING tasks can have workspaces prepared/reconfirmed
    if task.status != TaskStatus.PENDING:
        print(
            f"[FAIL] Task '{task_id}' has status '{task.status.value}'. "
            f"Workspace preparation is only permitted for PENDING tasks. "
            f"Held, recovery, or failed tasks must be resolved via --recover-task or --approve-request."
        )
        return 1

    # 2. Eligibility evaluation: Must be eligible under scheduler invariants
    scheduler = DependencyScheduler(ledger)
    eligible, reason = scheduler.check_task_eligibility(task_id)
    if not eligible:
        print(f"[FAIL] Task '{task_id}' is ineligible for workspace preparation: {reason}")
        return 1

    print(f"\n=== Preparing Isolated Workspace for Task [{task_id}] ===")
    print(f"Title       : {task.title}")
    print(f"Priority    : {task.priority.value}")
    print(f"Eligibility : Eligible")

    wt_manager = WorktreeManager(repo_root)
    ws_path = wt_manager.get_workspace_path(task_id)
    branch_name = wt_manager.get_branch_name(task_id)

    print(f"Target Worktree Path : {ws_path}")
    print(f"Target Branch Name   : {branch_name}")

    already_prepared = bool(task.workspace and Path(task.workspace.worktree_path).exists())
    if not is_persisted and already_prepared:
        store.save_ledger(ledger)
    try:
        with store.acquire_lock():
            metadata = wt_manager.prepare_workspace(task_id)
            ledger.update_workspace(task_id, metadata)
            result = CoordinatorAPI(store).transition_task(
                ledger, task_id, TaskStatus.PENDING, actor="coordinator",
                checkpoint_stage="WORKSPACE_PREPARED" if not already_prepared else "WORKSPACE_RECONFIRMED",
                checkpoint_details={
                    "worktree_path": str(metadata.worktree_path),
                    "branch_name": metadata.branch_name,
                    "base_commit": metadata.base_commit,
                },
                event_type="WORKSPACE_PREPARED" if not already_prepared else "WORKSPACE_RECONFIRMED",
                idempotency_key=f"workspace-prepared:{task_id}:{metadata.base_commit}:{'existing' if already_prepared else 'new'}",
                lock_held=True,
            )
            if not result.accepted:
                raise WorkspaceError(result.message)

        print(f"[SUCCESS] Workspace successfully prepared at: {metadata.worktree_path}")
        print(f"[INFO] Head Commit: {metadata.head_commit[:8]} | Branch: {metadata.branch_name}")
        print("[INFO] Workspace checkpoint saved; task remains PENDING until authorized execution starts.")
        return 0
    except (WorkspaceConflictError, WorkspaceValidationError, WorkspaceError) as e:
        print(f"[FAIL] Workspace preparation failed: {e}")
        return 1
    except LockAcquisitionError as e:
        print(f"[FAIL] Cannot acquire lock: {e}")
        return 1


def cmd_recover_task(
    repo_root: Path,
    store: StateStore,
    task_id: str,
    authorized: bool = False,
    reason: Optional[str] = None,
) -> int:
    """Explicitly authorize recovery/retry of a held or failed task, transitioning it to PENDING."""
    if not authorized:
        print("[FAIL] Task recovery requires explicit operator authorization (--authorize).")
        return 1

    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file is corrupt: {e}")
        return 2

    if task_id not in ledger:
        print(f"[FAIL] Task '{task_id}' not found in ledger.")
        return 1

    task = ledger.get_task(task_id)

    # 1. State check: Must be in a held/recoverable state
    if task.status not in (TaskStatus.NEEDS_APPROVAL, TaskStatus.RECOVERY_REQUIRED, TaskStatus.BLOCKED):
        print(
            f"[FAIL] Task '{task_id}' has status '{task.status.value}'; "
            f"only held tasks (NEEDS_APPROVAL, RECOVERY_REQUIRED, BLOCKED) can be recovered."
        )
        return 1

    # 2. Dependency check: Prerequisites must be VERIFIED
    for dep_id in task.dependencies:
        dep = ledger.get(dep_id)
        if not dep or dep.status != TaskStatus.VERIFIED:
            dep_status = dep.status.value if dep else "MISSING"
            print(f"[FAIL] Cannot recover task '{task_id}': prerequisite '{dep_id}' is not VERIFIED (status={dep_status}).")
            return 1

    # 3. Attempt limit check
    if task.attempts >= task.max_attempts:
        try:
            from .routing import is_escalation_eligible
        except ImportError:
            from routing import is_escalation_eligible
        if not is_escalation_eligible(task):
            print(f"[FAIL] Cannot recover task '{task_id}': maximum attempts ({task.attempts}/{task.max_attempts}) reached.")
            return 1

    recovery_reason = reason or "Operator authorized task recovery and retry."

    try:
        with store.acquire_lock():
            # If an open permission request exists in approvals.json, close it out as APPROVED
            approvals_path = repo_root / ".maya" / "maintenance" / "approvals.json"
            if approvals_path.is_file():
                try:
                    try:
                        from .terminal_approval import TerminalApprovalManager
                    except ImportError:
                        from terminal_approval import TerminalApprovalManager
                    mgr = TerminalApprovalManager(approvals_path, repo_root=repo_root, store=store)
                    for req in mgr.list_requests():
                        if req.task_id == task_id and req.status.value == "PENDING":
                            mgr.approve(req.request_id, reason=recovery_reason, sync_store=False)
                except Exception:
                    pass

            prior_status = task.status.value
            task.blocking_reason = None
            task.failure_reason = None
            if task.workspace:
                task.workspace.lifecycle_state = "PENDING"

            result = CoordinatorAPI(store).transition_task(
                ledger,
                task_id,
                TaskStatus.PENDING,
                actor="coordinator",
                checkpoint_stage="RECOVERY_AUTHORIZED",
                checkpoint_details={
                    "prior_status": prior_status,
                    "recovery_reason": recovery_reason,
                    "authorized_by": "operator",
                },
                event_type="TASK_RECOVERY_AUTHORIZED",
                idempotency_key=f"task-recovery-authorized:{task_id}:{task.attempts}:{int(time.time())}",
                lock_held=True,
            )
            if not result.accepted:
                print(f"[FAIL] Recovery transition rejected: {result.message}")
                return 1

        print(f"[SUCCESS] Task '{task_id}' recovered: {prior_status} -> PENDING.")
        print(f"[INFO] Reason: {recovery_reason}")
        print(f"[INFO] Event ID: {result.event_id}")
        return 0
    except LockAcquisitionError as e:
        print(f"[FAIL] Cannot acquire state lock: {e}")
        return 1


def cmd_workspace_status(repo_root: Path, store: StateStore, target_task_id: str | None = None) -> int:
    """Inspect and display workspace status for tasks."""
    try:
        ledger, _ = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file is corrupt: {e}")
        return 2

    wt_manager = WorktreeManager(repo_root)

    tasks_to_check = [target_task_id] if target_task_id else ledger.keys()

    print(f"\n=== MayaVE Task Workspace Status ===")
    print(f"{'Task ID':<8} {'Branch':<22} {'Registered':<12} {'Dirty?':<8} {'Ownership':<10} {'Path'}")
    print("-" * 90)

    count_found = 0
    for t_id in tasks_to_check:
        if t_id not in ledger:
            print(f"[WARN] Task ID '{t_id}' not found in ledger.")
            continue
        ws = wt_manager.inspect_workspace(t_id)
        if ws.exists_on_disk or ws.registered_in_git or target_task_id:
            count_found += 1
            print(
                f"{t_id:<8} {ws.branch_name:<22} {str(ws.registered_in_git):<12} "
                f"{str(ws.is_dirty):<8} {ws.ownership_status:<10} {ws.path}"
            )
            if ws.is_dirty:
                print(f"         Modified: {', '.join(ws.modified_files) or 'None'}")
                print(f"         Untracked: {', '.join(ws.untracked_files) or 'None'}")

    print("-" * 90)
    print(f"Workspaces displayed: {count_found}")
    return 0


def cmd_reconcile(repo_root: Path, store: StateStore) -> int:
    """Reconcile persisted state with actual Git worktrees on coordinator restart."""
    print(f"\n=== Reconciling Task State with Git Worktrees ===")
    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file is corrupt: {e}")
        return 2

    wt_manager = WorktreeManager(repo_root)

    with store.acquire_lock():
        reconciliations = wt_manager.reconcile_all(
            ledger, coordinator_api=CoordinatorAPI(store), lock_held=True
        )

    if not reconciliations:
        print("[OK] All task states are consistent with Git worktrees. Zero reconciliations required.")
        return 0

    print(f"[NOTICE] Reconciled {len(reconciliations)} task(s):")
    for item in reconciliations:
        print(f"  - [{item.task_id}] {item.previous_status.value} -> {item.new_status.value}")
        print(f"    Action: {item.action_taken}")
        print(f"    Reason: {item.reason}")

    return 0


def cmd_cleanup_preview(repo_root: Path, store: StateStore) -> int:
    """Display non-destructive cleanup preview of all workspaces."""
    print(f"\n=== Worktree Cleanup Preview (Strictly Read-Only) ===")
    try:
        ledger, _ = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file is corrupt: {e}")
        return 2

    wt_manager = WorktreeManager(repo_root)
    preview = wt_manager.get_cleanup_preview(ledger)

    if not preview:
        print("No worktrees or managed workspace directories detected under .maya/worktrees/.")
        return 0

    print(f"{'Task ID':<10} {'Task Status':<16} {'Dirty?':<8} {'Merged?':<8} {'Eligible?':<10} {'Recommendation':<25} {'Branch'}")
    print("-" * 105)
    for item in preview:
        t_id = item.task_id or "[ORPHAN]"
        print(
            f"{t_id:<10} {item.task_status:<16} {str(item.is_dirty):<8} {str(item.is_merged):<8} "
            f"{str(item.eligible):<10} {item.recommendation:<25} {item.branch_name}"
        )
        print(f"           Path    : {item.path}")
        print(f"           Reason  : {item.reason}")
        if item.blocking_reasons:
            print(f"           Blockers: {'; '.join(item.blocking_reasons)}")
        if item.actions:
            print(f"           Actions : {', '.join(item.actions)}")
    print("-" * 105)
    print("\n[SAFETY NOTICE] Preview only. No worktrees, branches, or files were modified or deleted.")
    return 0


def cmd_cleanup(
    repo_root: Path,
    store: StateStore,
    task_id: str | None = None,
    authorized: bool = False,
    dry_run: bool = False,
) -> int:
    """Execute or preview safe cleanup of a finished task's worktree and merged branch."""
    if dry_run or not task_id or task_id.upper() == "ALL":
        return cmd_cleanup_preview(repo_root, store)

    if not authorized:
        print("[ERROR] Cleanup execution requires explicit operator authorization (--authorize).")
        print("To preview cleanup without executing, run with --dry-run or --cleanup-preview.")
        return 1

    try:
        ledger, _ = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file is corrupt: {e}")
        return 2

    wt_manager = WorktreeManager(repo_root)
    ok, msg = wt_manager.cleanup_workspace(task_id, ledger, authorized=authorized)
    if not ok:
        print(f"[FAIL] {msg}")
        return 1

    print(f"[SUCCESS] {msg}")
    return 0


def cmd_pending_merges(repo_root: Path, store: StateStore, target_branch: str = "main") -> int:
    """Display visibility into pending and completed integration branches."""
    print(f"\n=== MayaVE Integration Branches & Pending Merges (Read-Only) ===")
    try:
        ledger, _ = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[FAIL] State file is corrupt: {e}")
        return 2

    wt_manager = WorktreeManager(repo_root)

    # 1. Discover all local branches in the Git repository
    br_proc = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    all_local_branches = set(br_proc.stdout.splitlines()) if br_proc.returncode == 0 else set()

    # Find candidate autophage branches
    int_branches = {b for b in all_local_branches if b.startswith("autophage/")}

    # Map ledger tasks by their workspace branch and expected integration branch
    ledger_tasks_by_branch: dict[str, list[TaskDefinition]] = {}
    for task in ledger:
        if task.workspace and task.workspace.branch_name:
            ledger_tasks_by_branch.setdefault(task.workspace.branch_name, []).append(task)
        int_b = f"autophage/integrate-{task.task_id}"
        ledger_tasks_by_branch.setdefault(int_b, []).append(task)
        int_branches.add(int_b)

    rows = []
    discrepancies = []

    for branch_name in sorted(int_branches):
        branch_exists = branch_name in all_local_branches
        associated_tasks = ledger_tasks_by_branch.get(branch_name, [])

        if not branch_exists:
            for t in associated_tasks:
                if t.status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.INTEGRATING, TaskStatus.VERIFIED):
                    discrepancies.append(
                        f"Task {t.task_id} ({t.status.value}) expects branch '{branch_name}', but branch does not exist in Git."
                    )
            continue

        task_id_str = "-"
        task_info = "[DISCREPANCY: No ledger task]"
        base_commit = "-"
        head_commit = "-"
        commits_ahead = 0
        wt_state = "MISSING"

        if associated_tasks:
            t = associated_tasks[0]
            task_id_str = t.task_id
            task_info = t.title[:35]
            if t.workspace and t.workspace.base_commit:
                base_commit = t.workspace.base_commit[:7]
            ws_status = wt_manager.inspect_workspace(t.task_id)
            if ws_status.exists_on_disk:
                wt_state = "DIRTY" if ws_status.is_dirty else "CLEAN"
            else:
                wt_state = "MISSING"
        else:
            discrepancies.append(
                f"Branch '{branch_name}' exists in Git but is not associated with any task in the ledger."
            )

        # Inspect Git commits
        head_proc = subprocess.run(
            ["git", "rev-parse", f"refs/heads/{branch_name}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if head_proc.returncode == 0:
            full_head = head_proc.stdout.strip()
            head_commit = full_head[:7]
            if base_commit != "-":
                cnt_proc = subprocess.run(
                    ["git", "rev-list", "--count", f"{base_commit}..{branch_name}"],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if cnt_proc.returncode == 0:
                    try:
                        commits_ahead = int(cnt_proc.stdout.strip())
                    except ValueError:
                        commits_ahead = 0

        # Check merge status
        is_merged = wt_manager.is_branch_merged(branch_name, target_branch)
        merge_status = "MERGED" if is_merged else "PENDING"

        rows.append({
            "branch": branch_name,
            "task_id": task_id_str,
            "title": task_info,
            "base": base_commit,
            "head": head_commit,
            "ahead": commits_ahead,
            "merge_status": merge_status,
            "worktree": wt_state,
        })

    if not rows and not discrepancies:
        print("No active integration or task branches detected.")
        return 0

    print(f"{'Branch Name':<32} {'Task':<8} {'Base':<8} {'Head':<8} {'Ahead':<6} {'Merge Status':<14} {'Worktree':<9} {'Description'}")
    print("-" * 115)
    for r in rows:
        print(
            f"{r['branch']:<32} {r['task_id']:<8} {r['base']:<8} {r['head']:<8} "
            f"{r['ahead']:<6} {r['merge_status']:<14} {r['worktree']:<9} {r['title']}"
        )
    print("-" * 115)

    if discrepancies:
        print("\n[DISCREPANCIES DETECTED]")
        for d in discrepancies:
            print(f"  ! {d}")

    print("\n[SAFETY NOTICE] Strictly an inspection and governance view. Zero merges performed.")
    return 0


def cmd_list_backups(store: StateStore) -> int:
    """List available rotating session state backups."""
    print(f"\n=== MayaVE Session State Backups ===")
    backups = store.list_backups()
    if not backups:
        print(f"No state backups found under: {store.backups_dir}")
        return 0

    print(f"{'Backup File':<45} {'Size (bytes)':<14} {'Modified (UTC)':<24}")
    print("-" * 85)
    for b in backups:
        print(f"{b['filename']:<45} {b['size_bytes']:<14} {b['modified_utc']:<24}")
    print("-" * 85)
    return 0


def cmd_restore_backup(store: StateStore, backup_name_or_path: str, authorized: bool = False) -> int:
    """Restore state.json from an earlier valid backup snapshot (requires --authorize)."""
    if not authorized:
        print("[ERROR] Restoring a state backup mutates active state and requires operator authorization (--authorize).")
        return 1

    try:
        store.restore_backup(backup_name_or_path, authorized=authorized)
        print(f"[SUCCESS] Restored state from backup '{backup_name_or_path}'.")
        return 0
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1


def cmd_validate(repo_root: Path, store: StateStore) -> int:
    """Run comprehensive non-destructive pre-validation check."""
    validator = PreValidator(repo_root)
    repo_ok, repo_msg = validator.check_repo_structure()
    if not repo_ok:
        print(f"[FAIL] {repo_msg}")
        return 1

    try:
        ledger, _ = _load_or_mock_ledger(store)
        ledger.validate_graph()
    except Exception as e:
        print(f"[FAIL] Ledger validation error: {e}")
        return 1

    report = validator.run_pre_validation(ledger)
    git_info = report.git_baseline

    print("\n=== Validation Report ===")
    print(f"Repo Valid   : True")
    print(f"Git Branch   : {git_info.branch if git_info else 'Unknown'}")
    print(f"Git Clean    : {git_info.is_clean if git_info else 'Unknown'}")
    print(f"Graph Valid  : {report.graph_valid}")
    print(f"Total Tasks  : {len(report.tasks_checked)}")

    unpatched = [c for c in report.tasks_checked if c.pre_validation_status == PreValidationStatus.CONFIRMED_UNPATCHED]
    patched = [c for c in report.tasks_checked if c.pre_validation_status == PreValidationStatus.ALREADY_PATCHED]
    missing = [c for c in report.tasks_checked if c.pre_validation_status == PreValidationStatus.TARGET_MISSING]
    ambiguous = [c for c in report.tasks_checked if c.pre_validation_status == PreValidationStatus.AMBIGUOUS]

    print(f"Active Defect Patterns Found: {len(unpatched)}")
    print(f"Already Patched / Modified   : {len(patched)}")
    print(f"Ambiguous / Untestable Regex : {len(ambiguous)}")
    print(f"Missing Targets              : {len(missing)}")
    return 0


def cmd_probe_models(repo_root: Path) -> int:
    """Probe runtime reachability and model availability across providers."""
    print("\n=== MayaVE Autophage Model Registry Preflight Probe ===")
    registry = ModelRegistry()
    probes = registry.probe_all()

    for provider, probe in probes.items():
        status_label = "[ONLINE]" if probe.reachable else "[OFFLINE]"
        print(f"\nProvider: {provider.value.upper()} {status_label}")
        print(f"  Target  : {probe.endpoint_or_target}")
        print(f"  Latency : {probe.latency_ms:.1f}ms")
        if probe.available_models:
            print(f"  Detected: {', '.join(probe.available_models)}")
        if probe.error_message:
            print(f"  Notice  : {probe.error_message}")

    print("\n--- Model Catalog Status ---")
    print(f"{'Model ID':<22} {'Tier':<24} {'Local?':<8} {'Status':<16} {'Cost/1k In/Out':<16}")
    print("-" * 88)
    for m in registry.list_models():
        cost_str = f"${m.cost_per_1k_input_tokens:.4f}/${m.cost_per_1k_output_tokens:.4f}"
        print(
            f"{m.model_id:<22} {m.tier.value:<24} {'Yes' if m.is_local else 'No':<8} "
            f"{m.status.value:<16} {cost_str:<16}"
        )
    return 0


def cmd_run_benchmark(repo_root: Path, output_file: str | None = None) -> int:
    """Run Tier 2 offline benchmark harness comparing candidate cloud models."""
    print("\n=== MayaVE Autophage Tier 2 Cloud Model Benchmark ===")
    harness = Tier2BenchmarkHarness()

    out_path = Path(output_file).resolve() if output_file else repo_root / ".maya" / "reports" / "tier2-benchmark-report.md"

    # Evaluate Candidate A (GLM-5.3-Flash Cloud) - Offline Unconfigured by default
    res_glm = harness.evaluate_candidate(
        candidate_id="glm-5.3-flash",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0002,
        task_replays=None,
    )

    # Evaluate Candidate B (Gemma 4 31B Cloud) - Offline Unconfigured by default
    res_gemma = harness.evaluate_candidate(
        candidate_id="gemma-4-31b",
        cost_per_1k_input=0.0002,
        cost_per_1k_output=0.0004,
        task_replays=None,
    )

    report = generate_benchmark_report([res_glm, res_gemma], output_path=out_path)
    print(f"[SUCCESS] Benchmark report generated at: {out_path}")
    print(f"GLM-5.3-Flash Status: {res_glm.status}")
    print(f"Gemma 4 31B   Status: {res_gemma.status}")
    print("Tier 2 Recommendation: Awaiting controlled live or recorded replay evidence.")
    return 0


def cmd_validate_contract(contract_path_str: str) -> int:
    """Validate a JSON task contract against the structured schema."""
    path = Path(contract_path_str).resolve()
    if not path.is_file():
        print(f"[ERROR] Task contract file not found: {path}")
        return 1

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON from '{path}': {e}")
        return 1

    if isinstance(data, list):
        ok, errors = TaskContractValidator.validate_batch(data)
    else:
        ok, errors = TaskContractValidator.validate_contract(data)

    if ok:
        print(f"[SUCCESS] Task contract at '{path}' is strictly VALID against schema v1.0.0.")
        return 0
    else:
        print(f"[FAIL] Task contract validation failed with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1


def cmd_plan_objective(
    repo_root: Path,
    store: StateStore,
    objective: str,
    max_tasks: int = 5,
    model: str | None = None,
) -> int:
    """Generate structured plan proposal via LocalPlanner using configured local model."""
    target_model = model or "qwen2.5-coder:3b"
    print(f"\n=== MayaVE Maintenance Planner (Milestone P5) ===")
    print(f"Target Planner : {target_model} (Local Ollama)")
    print(f"Max Plan Tasks : {min(max_tasks, 8)}")
    print(f"Objective      : {objective}")

    registry = ModelRegistry()
    probe_res = registry.probe_ollama()
    if not probe_res.reachable:
        print(f"[FAIL] Local Ollama daemon unreachable at {probe_res.endpoint_or_target}: {probe_res.error_message}")
        print("Please ensure Ollama is running ('ollama serve').")
        return 1

    desc = registry.get(target_model)
    if not desc or desc.status != ModelStatus.AVAILABLE:
        # Fallback to any installed local model if configured target is unavailable
        fallback_candidates = [m for m in probe_res.available_models if "qwen" in m or "llama" in m]
        if fallback_candidates:
            target_model = fallback_candidates[0]
            desc = registry.get(target_model)
        if not desc or desc.status != ModelStatus.AVAILABLE:
            print(f"[FAIL] Target planner model '{target_model}' is not available in local Ollama.")
            print("Installed models detected: " + ", ".join(probe_res.available_models))
            print(f"Please pull the model locally ('ollama pull {target_model}') before planning.")
            return 1
        print(f"[NOTICE] Using available local fallback planner model: {target_model}")

    known_task_ids: set[str] = set()
    if store.exists():
        try:
            ledger = store.load_ledger()
            known_task_ids = set(ledger.keys())
        except Exception:
            pass

    planner = LocalPlanner(PlannerConfig(model=target_model), repo_root=repo_root)
    print("\nDecomposing objective into structured task contract batch...")
    start_t = time.monotonic()
    proposal, errors = planner.generate_plan(
        objective=objective,
        max_tasks=max_tasks,
        known_task_ids=known_task_ids,
    )
    elapsed = round(time.monotonic() - start_t, 2)

    if not proposal:
        print(f"\n[FAIL] Plan generation failed after {elapsed}s with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    api = CoordinatorAPI(store)
    api.save_proposal(proposal)

    print(f"\n[SUCCESS] Plan proposal '{proposal.plan_id}' generated successfully in {elapsed}s!")
    print(f"Tasks Generated : {len(proposal.tasks)}")
    print(f"Tokens Used     : {proposal.input_tokens or 'N/A'} prompt, {proposal.output_tokens or 'N/A'} eval")
    print(f"Proposal Hash   : {proposal.proposal_hash[:16]}...")
    print(f"Proposal File   : {api.proposals_dir / f'{proposal.plan_id}.json'}")
    print(f"Status          : PROPOSED (Inert - awaiting human authorization)")

    print("\nProposed Tasks in Plan:")
    for t in proposal.tasks:
        deps = ", ".join(t.get("dependencies", [])) or "None"
        targets = ", ".join(t.get("target_files", []))
        print(f"  - [{t.get('task_id')}] (Pri: {t.get('priority')}, Risk: {t.get('risk')}): {t.get('title')}")
        print(f"      Targets: {targets}")
        print(f"      Deps   : {deps}")

    print("\n[NEXT STEPS]")
    print(f"  Inspect full plan : python -m skills.maintenance.coordinator --inspect-plan {proposal.plan_id}")
    print(f"  Approve & register: python -m skills.maintenance.coordinator --approve-plan {proposal.plan_id} --authorize")
    print(f"  Reject plan       : python -m skills.maintenance.coordinator --reject-plan {proposal.plan_id}")
    return 0


def cmd_plan_file(
    repo_root: Path,
    store: StateStore,
    file_path_str: str,
    max_tasks: int = 5,
    model: str | None = None,
) -> int:
    """Read objective from file and invoke cmd_plan_objective."""
    path = Path(file_path_str).resolve()
    if not path.is_file():
        print(f"[ERROR] Objective file not found: {path}")
        return 1
    try:
        objective = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to read objective file '{path}': {e}")
        return 1
    return cmd_plan_objective(repo_root, store, objective, max_tasks=max_tasks, model=model)


def cmd_ingest_backlog(
    repo_root: Path,
    store: StateStore,
    backlog_file_str: str,
    risk_policy: str = "manual",
    authorized: bool = False,
    stage_only: bool = False,
    output_manifest_str: str | None = None,
) -> int:
    """Ingest a Markdown bug backlog (.md) into canonical JSON manifest and maintenance queue."""
    try:
        from .manifest import ingest_backlog_workflow
    except ImportError:
        from manifest import ingest_backlog_workflow

    backlog_path = Path(backlog_file_str).resolve()
    output_manifest_path = Path(output_manifest_str).resolve() if output_manifest_str else None

    print(f"\n=== MayaVE Bug Backlog Intake ===")
    print(f"Backlog File : {backlog_path}")
    print(f"Risk Policy  : {risk_policy}")
    print(f"Authorized   : {authorized}")
    print(f"Stage Only   : {stage_only}")

    ok, info, msg = ingest_backlog_workflow(
        repo_root=repo_root,
        store=store,
        backlog_path=backlog_path,
        risk_policy=risk_policy,
        authorized=authorized,
        stage_only=stage_only,
        output_manifest_path=output_manifest_path,
    )

    if not ok:
        print(f"\n[FAIL] Backlog intake rejected:\n{msg}")
        return 1

    print(f"\n[SUCCESS] Backlog intake completed!")
    print(f"Manifest ID    : {info.get('manifest_id')}")
    print(f"Plan ID        : {info.get('plan_id')}")
    print(f"Tasks Parsed   : {info.get('tasks_count')}")
    print(f"Canonical JSON : {info.get('manifest_json')}")
    print(f"Status         : {info.get('status')}")

    if info.get("status") == "REGISTERED":
        reg_ids = info.get("registered_ids", [])
        held_ids = info.get("held_ids", [])
        print(f"\nRegistered as PENDING (Ready for queue): {', '.join(reg_ids) if reg_ids else 'None'}")
        if held_ids:
            print(f"Held in NEEDS_APPROVAL (Operator decision required): {', '.join(held_ids)}")
        print("\nNext step: Run batch repair queue:")
        if risk_policy == "auto-low-medium":
            print("  python -m skills.maintenance.coordinator --run-adaptive --authorize --risk-policy auto-low-medium --review-all")
        else:
            print("  python -m skills.maintenance.coordinator --run-adaptive --authorize --initial-workers 1 --max-workers 1")
    else:
        print(f"\nProposal staged under pending_proposals/. Zero modifications made to active ledger.")
        print("Next steps:")
        print(f"  Inspect plan: python -m skills.maintenance.coordinator --inspect-plan {info.get('plan_id')}")
        print(f"  Approve plan: python -m skills.maintenance.coordinator --approve-plan {info.get('plan_id')} --authorize")

    return 0


def cmd_list_plans(store: StateStore) -> int:
    """List all plan proposals in pending_proposals/."""
    api = CoordinatorAPI(store)
    proposals = api.list_proposals()

    print(f"\n=== MayaVE Maintenance Plan Proposals [{len(proposals)} total] ===")
    if not proposals:
        print("No plan proposals found under .maya/maintenance/pending_proposals/.")
        return 0

    print(f"{'Plan ID':<20} {'Status':<12} {'Tasks':<6} {'Created':<22} {'Objective'}")
    print("-" * 90)
    for p in proposals:
        obj_snippet = p.objective[:35].replace("\n", " ")
        if len(p.objective) > 35:
            obj_snippet += "..."
        print(f"{p.plan_id:<20} {p.status:<12} {len(p.tasks):<6} {p.created_at[:19]:<22} {obj_snippet}")
    print("-" * 90)
    return 0


def cmd_inspect_plan(store: StateStore, plan_id: str) -> int:
    """Inspect full details of a plan proposal without registration."""
    api = CoordinatorAPI(store)
    proposal = api.get_proposal(plan_id)
    if not proposal:
        print(f"[ERROR] Plan proposal '{plan_id}' not found.")
        return 1

    print(f"\n=== Inspection for Plan Proposal [{proposal.plan_id}] ===")
    print(f"Status        : {proposal.status}")
    print(f"Created At    : {proposal.created_at}")
    print(f"Proposal Hash : {proposal.proposal_hash}")
    if proposal.approved_at:
        print(f"Approved At   : {proposal.approved_at} (by {proposal.approved_by or 'operator'})")
    if proposal.rejection_reason:
        print(f"Rejected Due  : {proposal.rejection_reason}")
    if proposal.registered_task_ids:
        print(f"Registered IDs: {', '.join(proposal.registered_task_ids)}")
    print(f"Objective     :\n{proposal.objective}")
    print("\n--- Proposed Tasks ---")
    for idx, t in enumerate(proposal.tasks, start=1):
        print(f"\nTask #{idx}: [{t.get('task_id')}] {t.get('title')}")
        print(f"  Priority     : {t.get('priority')}")
        print(f"  Risk         : {t.get('risk')}")
        print(f"  Audit Source : {t.get('audit_source')}")
        print(f"  Description  : {t.get('description')}")
        print(f"  Target Files : {', '.join(t.get('target_files', []))}")
        print(f"  Test Targets : {', '.join(t.get('test_targets', []))}")
        print(f"  Dependencies : {', '.join(t.get('dependencies', [])) or 'None'}")
        print(f"  Criteria     :")
        for ac in t.get("acceptance_criteria", []):
            print(f"    - {ac}")

    print("\n[SAFETY NOTICE] Inspection only. Zero modifications or registrations made.")
    return 0


def cmd_approve_plan(
    store: StateStore,
    plan_id: str,
    authorized: bool = False,
) -> int:
    """Approve and register a plan proposal into active ledger (requires --authorize)."""
    api = CoordinatorAPI(store)
    ok, registered_tasks, msg = api.approve_and_register_plan(
        plan_id=plan_id,
        authorized=authorized,
    )
    if not ok:
        print(f"[FAIL] {msg}")
        return 1

    print(f"\n[SUCCESS] {msg}")
    print(f"Registered Task IDs in Ledger:")
    for tid in registered_tasks:
        print(f"  - {tid}")
    print("\nTasks are now PENDING in the active ledger. Run --status to view.")
    return 0


def cmd_reject_plan(
    store: StateStore,
    plan_id: str,
    reason: str = "Operator rejected",
) -> int:
    """Reject a plan proposal."""
    api = CoordinatorAPI(store)
    ok, msg = api.reject_plan(plan_id, reason=reason)
    if not ok:
        print(f"[FAIL] {msg}")
        return 1
    print(f"[SUCCESS] {msg}")
    return 0


def cmd_execute(
    repo_root: Path,
    store: StateStore,
    task_id: str | None = None,
    authorized: bool = False,
    dry_run: bool = False,
    adapter_name: str = "antigravity",
    reviewer_name: str = "copilot",
    max_reviews: int = 5,
    timeout_seconds: int = 600,
    agent_executable: str | None = None,
    route_complexity: bool = False,
    local_model: str = "qwen2.5-coder:3b",
) -> int:
    """Execute a single task via SingleTaskOrchestrator."""
    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[ERROR] State file is corrupt: {e}")
        return 2

    if not is_persisted and not dry_run:
        print(f"[ERROR] State store not initialized at: {store.state_path}")
        print("Run with --init first before executing tasks.")
        return 1

    if is_persisted:
        try:
            store.create_session_backup()
        except Exception:
            pass

    worktree_mgr = WorktreeManager(repo_root)
    quality_engine = QualityGateEngine(repo_root)
    cache = ContextCache(repo_root / ".maya" / "maintenance" / "cache.json")

    router = ComplexityRouter(config=RoutingConfig(local_model=local_model)) if route_complexity else None
    concurrency_guard = WorkerConcurrencyGuard(max_implementation_workers=1)

    if adapter_name == "fake":
        adapter: AgentAdapter = FakeAgentAdapter()
    elif adapter_name == "local":
        adapter = LocalModelAgentAdapter(model=local_model)
    elif adapter_name == "tiered":
        local_adapter = LocalModelAgentAdapter(model=local_model)
        escalation_adapter = AntigravityAgentAdapter(authorized=authorized, executable=agent_executable)
        adapter = TieredAgentAdapter(local_adapter=local_adapter, escalation_adapter=escalation_adapter)
    else:
        adapter = AntigravityAgentAdapter(authorized=authorized, executable=agent_executable)

    if reviewer_name == "fake":
        reviewer: Optional[ReviewerAdapter] = FakeReviewerAdapter()
    elif reviewer_name == "none":
        reviewer = None
    else:
        reviewer = CopilotReviewerAdapter()

    review_policy = ReviewPolicy(max_reviews_per_session=max_reviews)

    orchestrator = SingleTaskOrchestrator(
        repo_root=repo_root,
        store=store,
        ledger=ledger,
        worktree_mgr=worktree_mgr,
        adapter=adapter,
        quality_engine=quality_engine,
        reviewer=reviewer,
        review_policy=review_policy,
        cache=cache,
        timeout_seconds=timeout_seconds,
        router=router,
        concurrency_guard=concurrency_guard,
    )

    try:
        summary = orchestrator.execute_single_task(
            task_id=task_id,
            authorized=authorized,
            dry_run=dry_run,
        )
    except LockAcquisitionError as e:
        print(f"[ERROR] Cannot acquire execution lock: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[WARNING] Execution interrupted by operator (Ctrl+C). Safely halting...")
        if task_id and is_persisted and not dry_run:
            try:
                hub = CoordinatorAPI(store)
                reloaded = store.load_ledger()
                current_t = reloaded.get_task(task_id)
                if current_t and current_t.status in (TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING):
                    trans_res = hub.transition_task(
                        reloaded, task_id, TaskStatus.RECOVERY_REQUIRED, actor="coordinator",
                        blocking_reason="Interrupted by operator (KeyboardInterrupt). Requires recovery review.",
                        checkpoint_stage="USER_INTERRUPTED",
                        checkpoint_details={"interrupted_at": utc_now_iso()},
                        event_type="TASK_USER_INTERRUPTED",
                        idempotency_key=f"user-interrupt:{task_id}:{current_t.attempts}",
                    )
                    if not trans_res.accepted:
                        print(f"[ERROR] Could not transition interrupted task to recovery: {trans_res.message}")
                    else:
                        print(f"[INFO] Task {task_id} successfully transitioned to RECOVERY_REQUIRED.")
            except Exception as rec_err:
                print(f"[NOTICE] Could not persist interrupted recovery status: {rec_err}")
        return 130
    except Exception as e:
        print(f"[ERROR] Execution failed unexpectedly: {e}")
        return 1

    print("\n=== MayaVE Single-Task Execution Summary ===")
    print(f"Task ID      : {summary.task_id or 'None'}")
    print(f"Success      : {summary.success}")
    print(f"Final Status : {summary.final_status.value if summary.final_status else 'None'}")
    print(f"Dry Run      : {summary.dry_run}")
    print(f"Authorized   : {summary.authorized}")
    print(f"Duration     : {summary.duration_sec}s")
    if summary.worktree_path:
        print(f"Worktree     : {summary.worktree_path}")
    print(f"Message      : {summary.message}")

    if summary.quality_report:
        print("\n--- Quality Gate Results ---")
        for gate in summary.quality_report.get("results", []):
            mark = "[PASS]" if gate.get("passed") else "[FAIL]"
            print(f"  {mark} {gate.get('gate_name')}: {gate.get('message')}")

    if summary.agent_result and not summary.success:
        ag = summary.agent_result
        print("\n--- Agent Execution Diagnostics ---")
        print(f"  Exit Code   : {ag.get('exit_code')}")
        print(f"  Duration    : {ag.get('duration_sec')}s")
        if ag.get("timed_out"):
            print("  Status      : TIMED OUT")
        elif ag.get("interrupted"):
            print("  Status      : INTERRUPTED")
        elif ag.get("permission_denied"):
            print("  Status      : PERMISSION DENIED")
        if ag.get("error_message"):
            print(f"  Error       : {ag.get('error_message')}")
        stdout_excerpt = (ag.get("stdout_excerpt") or "").strip()
        if stdout_excerpt:
            lines = stdout_excerpt.splitlines()
            print(f"  Captured Agent Stdout ({ag.get('stdout_length', len(stdout_excerpt))} chars):")
            for line in lines[:50]:
                print(f"    | {line}")
            if len(lines) > 50:
                print(f"    | ... [{len(lines) - 50} more lines omitted for brevity] ...")
        stderr_excerpt = (ag.get("stderr_excerpt") or "").strip()
        if stderr_excerpt:
            lines = stderr_excerpt.splitlines()
            print(f"  Captured Agent Stderr ({ag.get('stderr_length', len(stderr_excerpt))} chars):")
            for line in lines[:30]:
                print(f"    | {line}")
            if len(lines) > 30:
                print(f"    | ... [{len(lines) - 30} more lines omitted for brevity] ...")

    if summary.review_result:
        print("\n--- Independent Review Results ---")
        rev = summary.review_result
        print(f"  Recommendation : {rev.get('recommendation')}")
        print(f"  Summary        : {rev.get('summary')}")
        for finding in rev.get("findings", []):
            print(f"  [{finding.get('severity')}] {finding.get('file')}: {finding.get('message')}")

    return 0 if summary.success else 1


def cmd_run_adaptive(
    repo_root: Path,
    store: StateStore,
    initial_workers: int = 1,
    max_workers: int = 1,
    authorized: bool = False,
    dry_run: bool = False,
    adapter_name: str = "antigravity",
    reviewer_name: str = "copilot",
    max_reviews: int = 5,
    timeout_seconds: int = 600,
    agent_executable: str | None = None,
    max_tasks: int | None = None,
    hard_timeout: float | None = 21600.0,
    max_risk: str = "MEDIUM",
    stop_on_failure: bool = True,
    force_new_session: bool = False,
    resume: bool = False,
    local_model: str = "qwen2.5-coder:3b",
    route_complexity: bool = False,
    risk_policy: str = "manual",
    review_all: bool = False,
) -> int:
    """Execute maintenance loop using AdaptiveWorkerScheduler with pilot safety bounds."""
    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[ERROR] State file is corrupt: {e}")
        return 2

    if not is_persisted and not dry_run:
        print(f"[ERROR] State store not initialized at: {store.state_path}")
        print("Run with --init first before executing tasks.")
        return 1

    if is_persisted:
        try:
            store.create_session_backup()
        except Exception:
            pass

    try:
        config = AdaptiveConfig(
            initial_concurrency=initial_workers,
            max_concurrency=max_workers,
            worker_timeout_sec=timeout_seconds,
            max_tasks=max_tasks,
            hard_timeout_sec=hard_timeout,
            max_risk=TaskRisk(max_risk),
            stop_on_failure=stop_on_failure,
        )
    except ValueError as e:
        print(f"[ERROR] Invalid adaptive worker configuration: {e}")
        return 1

    worktree_mgr = WorktreeManager(repo_root)
    quality_engine = QualityGateEngine(repo_root)
    cache = ContextCache(repo_root / ".maya" / "maintenance" / "cache.json")

    if adapter_name == "fake":
        adapter: AgentAdapter = FakeAgentAdapter()
    elif adapter_name == "local":
        adapter = LocalModelAgentAdapter(model=local_model)
    elif adapter_name == "tiered":
        local_adapter = LocalModelAgentAdapter(model=local_model)
        escalation_adapter = AntigravityAgentAdapter(authorized=authorized, executable=agent_executable)
        adapter = TieredAgentAdapter(local_adapter=local_adapter, escalation_adapter=escalation_adapter)
    else:
        adapter = AntigravityAgentAdapter(authorized=authorized, executable=agent_executable)

    if reviewer_name == "fake":
        reviewer: Optional[ReviewerAdapter] = FakeReviewerAdapter()
    elif reviewer_name == "none":
        reviewer = None
    else:
        reviewer = CopilotReviewerAdapter()

    review_policy = ReviewPolicy(max_reviews_per_session=max_reviews, review_all=review_all)

    router = None
    if route_complexity:
        routing_cfg = RoutingConfig(
            local_model=local_model,
            max_fallback_attempts=1,
            efficient_cloud_agent="implementer",
            advanced_cloud_agent="architect",
        )
        router = ComplexityRouter(config=routing_cfg)

    concurrency_guard = WorkerConcurrencyGuard(max_implementation_workers=1)
    local_adapter_instance = LocalModelAgentAdapter(model=local_model) if route_complexity else None

    scheduler = AdaptiveWorkerScheduler(
        repo_root=repo_root,
        store=store,
        ledger=ledger,
        worktree_mgr=worktree_mgr,
        config=config,
        adapter=adapter,
        quality_engine=quality_engine,
        reviewer=reviewer,
        review_policy=review_policy,
        cache=cache,
        router=router,
        concurrency_guard=concurrency_guard,
        local_adapter=local_adapter_instance,
        risk_policy=risk_policy,
    )

    print("\n=== MayaVE Adaptive Worker Scheduler Starting ===")
    print(f"Initial Concurrency : {config.initial_concurrency}")
    print(f"Max Concurrency     : {config.max_concurrency}")
    print(f"Max Tasks Cap       : {config.max_tasks}")
    print(f"Max Risk Tier       : {config.max_risk.value}")
    print(f"Stop on Failure     : {config.stop_on_failure}")
    print(f"Authorized          : {authorized}")
    print(f"Dry Run             : {dry_run}")
    print(f"Adapter             : {adapter_name}")
    print(f"Reviewer            : {reviewer_name} (Max budget: {max_reviews})")

    session_start = time.monotonic()
    try:
        summary = scheduler.run_until_idle(
            authorized=authorized,
            dry_run=dry_run,
            max_duration_sec=hard_timeout,
            force_new_session=force_new_session,
            resume=resume,
        )
    except ValueError as e:
        print(f"[ERROR] Session initialization failed: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[WARNING] Scheduler loop interrupted by operator (Ctrl+C). Safely halting...")
        scheduler.wait_for_active_workers(timeout=5.0)
        summary = scheduler._compile_summary(session_start)
        print("[INFO] Active workers stopped and state preserved safely.")

    print("\n=== MayaVE Adaptive Scheduler Execution Summary ===")
    print(f"Total Tasks        : {summary.total_tasks}")
    print(f"Tasks Executed     : {summary.completed_tasks}")
    print(f"Verified Tasks     : {summary.verified_tasks}")
    print(f"Failed Tasks       : {summary.failed_tasks}")
    print(f"Blocked Tasks      : {summary.blocked_tasks}")
    print(f"Peak Concurrency   : {summary.peak_concurrency}")
    print(f"Quarantined Count  : {summary.quarantined_count}")
    print(f"Duration           : {summary.duration_sec:.2f}s")

    return 0 if summary.failed_tasks == 0 else 1


def cmd_terminal_approval(repo_root: Path, store: StateStore | None = None) -> int:
    """Launch interactive terminal approval session."""
    approvals_path = repo_root / ".maya" / "maintenance" / "approvals.json"
    mgr = TerminalApprovalManager(approvals_path, repo_root=repo_root, store=store)
    count, reason = mgr.interactive_session()
    print(f"\nTerminal approval session closed ({count} decisions, result={reason}).")
    return 0


def cmd_list_approvals(repo_root: Path) -> int:
    """Display pending permission requests in a formatted table."""
    approvals_path = repo_root / ".maya" / "maintenance" / "approvals.json"
    mgr = TerminalApprovalManager(approvals_path, repo_root=repo_root)
    pending = mgr.list_requests(status=ApprovalStatus.PENDING)
    if not pending:
        print("\nNo pending permission requests awaiting operator approval.")
        return 0
    print(f"\n=== Pending Maintenance Permission Requests ({len(pending)}) ===")
    print(f"{'Request ID':<26} {'Task':<8} {'Risk':<7} {'Operation':<20} {'Target'}")
    print("-" * 88)
    for req in pending:
        print(f"{req.request_id:<26} {req.task_id:<8} {req.risk:<7} {req.operation[:18]:<20} {req.target[:25]}")
    return 0


def cmd_approve_request(
    repo_root: Path,
    request_id: str,
    scope: str | None = None,
    store: StateStore | None = None,
) -> int:
    """Approve a specific pending permission request."""
    approvals_path = repo_root / ".maya" / "maintenance" / "approvals.json"
    mgr = TerminalApprovalManager(approvals_path, repo_root=repo_root, store=store)
    try:
        req = mgr.approve(request_id, scope=scope, decision_by="operator")
        print(f"[SUCCESS] Approved request '{request_id}' for task '{req.task_id}'.")
        print(f"Scope: {req.decision_scope}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to approve request: {e}")
        return 1


def cmd_deny_request(repo_root: Path, request_id: str) -> int:
    """Deny a specific pending permission request."""
    approvals_path = repo_root / ".maya" / "maintenance" / "approvals.json"
    mgr = TerminalApprovalManager(approvals_path, repo_root=repo_root)
    try:
        req = mgr.deny(request_id, decision_by="operator")
        print(f"[SUCCESS] Denied request '{request_id}' for task '{req.task_id}'.")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to deny request: {e}")
        return 1


def cmd_inspect_request(repo_root: Path, request_id: str) -> int:
    """Inspect diagnostic details and worktree state for a request."""
    approvals_path = repo_root / ".maya" / "maintenance" / "approvals.json"
    if not approvals_path.exists():
        print(f"[ERROR] Approvals file not found at '{approvals_path}'.")
        return 1
    try:
        mgr = TerminalApprovalManager(approvals_path, repo_root=repo_root)
    except Exception as e:
        print(f"[ERROR] Failed to load approvals file: {e}")
        return 1
    info = mgr.inspect_request(request_id)
    if "error" in info:
        print(f"[ERROR] {info['error']}")
        return 1
    print(json.dumps(info, indent=2))
    return 0


def cmd_integrate(
    repo_root: Path,
    store: StateStore,
    task_id: str,
    authorized: bool = False,
) -> int:
    """Integrate an exactly approved candidate and mark VERIFIED only after post-checks."""
    try:
        ledger, is_persisted = _load_or_mock_ledger(store)
    except CorruptStateError as e:
        print(f"[ERROR] State file is corrupt: {e}")
        return 2

    task = ledger.get(task_id)
    if not task:
        print(f"[ERROR] Task '{task_id}' not found in ledger.")
        return 1

    if task.status == TaskStatus.VERIFIED:
        stage = task.checkpoint.stage if task.checkpoint else None
        if stage in ("INTEGRATED", "POST_INTEGRATION_VERIFIED"):
            print(f"[INFO] Task '{task_id}' is already VERIFIED after integration.")
            return 0
        print(f"[ERROR] Task '{task_id}' has VERIFIED state without post-integration evidence; recovery review is required.")
        return 1
    if task.status != TaskStatus.READY_TO_INTEGRATE:
        print(f"[ERROR] Task '{task_id}' is not READY_TO_INTEGRATE (current: {task.status.value}). Integration rejected.")
        return 1

    if not authorized:
        print("[ERROR] Integration requires explicit operator authorization (--authorize).")
        return 1

    # Check for any unresolved dependent tasks or pending revalidations (including transitive prerequisite revalidations)
    hub = CoordinatorAPI(store)
    reval_ok, reval_msg = hub.check_integration_revalidations(ledger, task_id)
    if not reval_ok:
        print(f"[ERROR] Cannot integrate task '{task_id}': {reval_msg}")
        return 1

    worktree_mgr = WorktreeManager(repo_root)
    ws_path = worktree_mgr.get_workspace_path(task_id)
    if not ws_path.exists():
        print(f"[ERROR] Worktree for task '{task_id}' does not exist at {ws_path}.")
        return 1

    baseline_commit = task.workspace.base_commit if task.workspace else ""
    if not baseline_commit:
        print("[ERROR] Candidate has no recorded baseline commit; recovery review is required.")
        return 1
    main_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, check=False
    )
    if main_head.returncode or main_head.stdout.strip() != baseline_commit:
        hub = CoordinatorAPI(store)
        if task.integration_approval:
            invalidated = hub.invalidate_integration_approval(
                ledger, task_id, "Main baseline changed after candidate preparation."
            )
            if not invalidated.accepted:
                print(f"[ERROR] Could not invalidate approval for stale baseline: {invalidated.message}")
                return 1
        result = hub.transition_task(
            ledger, task_id, TaskStatus.RECOVERY_REQUIRED, actor="coordinator",
            blocking_reason="Main baseline changed after candidate preparation; rebase/revalidation and fresh approval required.",
            checkpoint_stage="BASELINE_STALE",
            checkpoint_details={"expected_baseline": baseline_commit, "actual_baseline": main_head.stdout.strip()},
            event_type="INTEGRATION_BASELINE_STALE",
            idempotency_key=f"stale-baseline:{task_id}:{baseline_commit}:{main_head.stdout.strip()}",
        )
        print(f"[ERROR] Main baseline differs from approved candidate; task moved to {result.resulting_status if result.accepted else 'unchanged recovery state'}.")
        return 1

    # Candidate verification is repeated immediately before any integration write.
    quality_engine = QualityGateEngine(repo_root)
    qr = quality_engine.evaluate(task, ws_path)
    if not qr.passed:
        print(f"[ERROR] Quality gates failed for worktree: {qr.failure_reason}")
        result = CoordinatorAPI(store).transition_task(
            ledger, task_id, TaskStatus.RECOVERY_REQUIRED, actor="coordinator",
            failure_reason=qr.failure_reason or "Pre-integration verification failed.",
            checkpoint_stage="PRE_INTEGRATION_REVALIDATION_FAILED",
            checkpoint_details={"quality_report": qr.to_dict()},
            event_type="INTEGRATION_REVALIDATION_FAILED",
            idempotency_key=f"integration-revalidation-failed:{task_id}:{baseline_commit}",
        )
        if not result.accepted:
            print(f"[ERROR] Failed to persist recovery state: {result.message}")
        return 1

    try:
        patch_sha256 = compute_patch_sha256(ws_path, baseline_commit)
        verification_sha256 = compute_verification_sha256(patch_sha256, baseline_commit, qr, task.test_targets)
    except Exception as exc:
        print(f"[ERROR] Candidate artifact could not be independently fingerprinted: {exc}")
        return 1

    checkpoint_details = task.checkpoint.details if task.checkpoint else {}
    same_evidence = all(checkpoint_details.get(k) == v for k, v in (
        ("artifact_sha256", patch_sha256),
        ("baseline_commit", baseline_commit),
        ("verification_sha256", verification_sha256),
    ))
    hub = CoordinatorAPI(store)
    if not same_evidence:
        if task.integration_approval:
            invalidated = hub.invalidate_integration_approval(
                ledger, task_id, "Candidate patch, baseline, or verification evidence changed."
            )
            if not invalidated.accepted:
                print(f"[ERROR] Could not invalidate stale approval: {invalidated.message}")
                return 1
        to_verify = hub.transition_task(
            ledger, task_id, TaskStatus.VERIFYING, actor="coordinator",
            checkpoint_stage="INTEGRATION_REVALIDATING",
            checkpoint_details={"artifact_sha256": patch_sha256, "baseline_commit": baseline_commit},
            event_type="INTEGRATION_REVALIDATION_STARTED",
            idempotency_key=f"integration-revalidation-start:{task_id}:{patch_sha256}:{baseline_commit}",
        )
        if not to_verify.accepted:
            print(f"[ERROR] Candidate evidence is stale and could not be revalidated: {to_verify.message}")
            return 1
        updated = hub.transition_task(
            ledger, task_id, TaskStatus.READY_TO_INTEGRATE, actor="coordinator",
            checkpoint_stage="PRE_INTEGRATION_VERIFICATION",
            checkpoint_details={
                "artifact_sha256": patch_sha256,
                "baseline_commit": baseline_commit,
                "verification_sha256": verification_sha256,
                "quality_report": qr.to_dict(),
                "test_targets": list(task.test_targets),
            },
            event_type="PRE_INTEGRATION_VERIFICATION_PASSED",
            idempotency_key=f"pre-integration-verification:{task_id}:{patch_sha256}:{verification_sha256}",
        )
        if not updated.accepted:
            print(f"[ERROR] Could not persist refreshed verification evidence: {updated.message}")
            return 1
        task = ledger.get_task(task_id)
        print("[INFO] Candidate changed or lacked matching evidence; verification was refreshed. A new exact-artifact approval is required.")

    approvals_path = repo_root / ".maya" / "maintenance" / "approvals.json"
    approval_manager = TerminalApprovalManager(approvals_path, repo_root=repo_root, store=store)
    scope = (
        f"patch_sha256={patch_sha256};baseline_commit={baseline_commit};"
        f"verification_sha256={verification_sha256}"
    )
    approved = approval_manager.is_operation_approved(
        task_id, "integration", str(ws_path), scope,
        patch_sha256=patch_sha256,
        baseline_commit=baseline_commit,
        verification_sha256=verification_sha256,
    )
    if approved and not (task.integration_approval or {}).get("status") == "APPROVED":
        req = approval_manager.get_pending_request_for_task(task_id)
        approved_req = next((r for r in approval_manager.list_requests(status=ApprovalStatus.APPROVED)
                            if r.task_id == task_id and r.operation == "integration"
                            and r.patch_sha256 == patch_sha256 and r.baseline_commit == baseline_commit
                            and r.verification_sha256 == verification_sha256), None)
        if approved_req:
            sync_result = hub.approve_integration(
                task_id, approved_req.request_id, patch_sha256, baseline_commit,
                verification_sha256, approved_req.decision_by or "operator",
            )
            if not sync_result.accepted:
                print(f"[ERROR] Approved request is not authoritative yet: {sync_result.message}")
                return 1
            task = store.load_ledger().get_task(task_id)
    if not approved or (task.integration_approval or {}).get("status") != "APPROVED":
        req = approval_manager.create_request(
            task_id=task_id,
            risk=task.risk.value,
            operation="integration",
            target=str(ws_path),
            scope=scope,
            reason="Review the current candidate patch, baseline, and verification fingerprint before integration.",
            patch_sha256=patch_sha256,
            baseline_commit=baseline_commit,
            verification_sha256=verification_sha256,
        )
        print(f"[APPROVAL REQUIRED] Request {req.request_id} created for the exact patch and baseline.")
        print("Inspect and approve it with --inspect-request / --approve-request, then rerun --integrate-task with --authorize.")
        return 1

    approval = task.integration_approval
    expected_approval = {
        "request_id": approval.get("request_id"),
        "patch_sha256": patch_sha256,
        "baseline_commit": baseline_commit,
        "verification_sha256": verification_sha256,
    }
    if any(approval.get(k) != v for k, v in expected_approval.items()):
        invalidated = hub.invalidate_integration_approval(ledger, task_id, "Approval artifact tuple no longer matches.")
        print(f"[ERROR] Approval is stale and was invalidated: {invalidated.message}")
        return 1

    # Any existing primary-worktree change blocks integration; it may be user-owned.
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=str(repo_root), capture_output=True, text=True, check=False)
    if st.returncode or st.stdout.strip():
        print("[ERROR] Primary repository is dirty. Preserve those changes and integrate only from a clean baseline.")
        print("Hint: You can stash or commit uncommitted changes (e.g. docs/CHANGELOG.md) before integrating.")
        return 1

    # Dedicated Integration Branch Governance
    # Rule 1: Never commit directly on 'main'.
    # Rule 2: Prefer dedicated integration branch 'autophage/integrate-<task_id>' branched from baseline_commit.
    # Rule 3: Fail closed on existing-branch collisions or dirty states without modifying the repository.
    br_proc = subprocess.run(["git", "branch", "--show-current"], cwd=str(repo_root), capture_output=True, text=True, check=False)
    try:
        store.create_session_backup()
    except Exception:
        pass

    initial_branch = br_proc.stdout.strip()
    target_branch = f"autophage/integrate-{task_id}"
    created_integration_branch = False

    if initial_branch == "main" or initial_branch != target_branch:
        # Check if target_branch already exists
        ref_check = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{target_branch}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if ref_check.returncode == 0:
            existing_tip = ref_check.stdout.strip()
            # If target_branch exists and differs from baseline_commit, fail closed
            if existing_tip != baseline_commit:
                print(f"[ERROR] Dedicated integration branch '{target_branch}' already exists and points to {existing_tip[:7]} (expected baseline {baseline_commit[:7]}).")
                print(f"Operator action required: To prevent overwriting work, inspect, rename, or delete the existing '{target_branch}' branch before integrating.")
                return 1
            # Target branch exists and matches baseline_commit -> safe to switch
            co_proc = subprocess.run(["git", "checkout", target_branch], cwd=str(repo_root), capture_output=True, text=True, check=False)
            if co_proc.returncode != 0:
                print(f"[ERROR] Failed to switch to integration branch '{target_branch}': {co_proc.stderr or co_proc.stdout}")
                return 1
        else:
            # Create target_branch from baseline_commit
            co_proc = subprocess.run(
                ["git", "checkout", "-b", target_branch, baseline_commit],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if co_proc.returncode != 0:
                print(f"[ERROR] Failed to create dedicated integration branch '{target_branch}' from baseline {baseline_commit[:7]}: {co_proc.stderr or co_proc.stdout}")
                return 1
            created_integration_branch = True

    # Absolute invariant check: active branch MUST NOT be 'main'
    active_branch_proc = subprocess.run(["git", "branch", "--show-current"], cwd=str(repo_root), capture_output=True, text=True, check=False)
    active_branch = active_branch_proc.stdout.strip()
    if active_branch == "main":
        print("[CRITICAL] Integration directly on branch 'main' is forbidden by repository governance policy.")
        return 1

    def _rollback_branch_on_failure():
        if initial_branch and initial_branch != target_branch:
            subprocess.run(["git", "checkout", initial_branch], cwd=str(repo_root), capture_output=True, check=False)
            if created_integration_branch:
                subprocess.run(["git", "branch", "-D", target_branch], cwd=str(repo_root), capture_output=True, check=False)

    integrating = hub.transition_task(
        ledger, task_id, TaskStatus.INTEGRATING, actor="integration", authorized=True,
        checkpoint_stage="INTEGRATION_STARTED",
        checkpoint_details={"artifact_sha256": patch_sha256, "baseline_commit": baseline_commit,
                            "verification_sha256": verification_sha256,
                            "integration_approval": approval},
        event_type="INTEGRATION_STARTED",
        idempotency_key=f"integration-start:{task_id}:{patch_sha256}:{baseline_commit}",
        require_integration_approval=True,
    )
    if not integrating.accepted:
        print(f"[ERROR] Coordinator rejected integration: {integrating.error_code}: {integrating.message}")
        return 1
    ledger = store.load_ledger()
    task = ledger.get_task(task_id)

    # 1. Commit changes in worktree if uncommitted
    wt_st = subprocess.run(["git", "status", "--porcelain"], cwd=str(ws_path), capture_output=True, text=True, check=False)
    if wt_st.stdout.strip():
        subprocess.run(["git", "add", "-A"], cwd=str(ws_path), check=True)
        subprocess.run(
            ["git", "commit", "-m", f"fix({task_id.lower()}): {task.title}"],
            cwd=str(ws_path),
            check=True,
        )

    # Snapshot exact destinations so a failed post-check restores the pre-integration state.
    destinations = sorted(set(task.target_files) | {"docs/CHANGELOG.md"})
    backups: dict[str, bytes | None] = {}
    try:
        for rel in destinations:
            dst = repo_root / rel
            backups[rel] = dst.read_bytes() if dst.is_file() else None
        for tf in task.target_files:
            src = ws_path / tf
            dst = repo_root / tf
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                # Target file was deleted in candidate workspace
                # Validate deletion authorization: task.allow_file_deletion strictly required
                if not task.allow_file_deletion:
                    raise RuntimeError(f"Unauthorized candidate file deletion for '{tf}' during integration.")
                if tf.lower() == "docs/changelog.md":
                    raise RuntimeError(f"Changelog file '{tf}' cannot be deleted during integration.")
                # Verify repository boundary containment
                resolved_dst = dst.resolve()
                if not resolved_dst.is_relative_to(repo_root.resolve()):
                    raise RuntimeError(f"Deletion target '{tf}' escapes repository root.")
                if dst.exists():
                    dst.unlink()
    except Exception as exc:
        print(f"[ERROR] Candidate copy failed: {exc}")
        _restore_integration_files(repo_root, backups)
        _rollback_branch_on_failure()
        result = hub.transition_task(
            ledger, task_id, TaskStatus.RECOVERY_REQUIRED, actor="integration",
            failure_reason=f"Integration copy failed: {exc}", checkpoint_stage="INTEGRATION_COPY_FAILED",
            checkpoint_details={"artifact_sha256": patch_sha256}, event_type="INTEGRATION_FAILED",
            idempotency_key=f"integration-copy-failed:{task_id}:{patch_sha256}",
        )
        if not result.accepted:
            print(f"[ERROR] Could not persist copy-failure recovery state: {result.message}")
        return 1

    # 3. Merge changelog entries from worktree into main
    wt_changelog = ws_path / "docs" / "CHANGELOG.md"
    main_changelog = repo_root / "docs" / "CHANGELOG.md"
    if wt_changelog.exists() and main_changelog.exists():
        wt_lines = wt_changelog.read_text(encoding="utf-8", errors="replace").splitlines()
        main_lines = main_changelog.read_text(encoding="utf-8", errors="replace").splitlines()

        task_entries = []
        target_section = None
        current_section = None
        in_entry = False
        for line in wt_lines:
            if line.startswith("### "):
                current_section = line.strip() + "\n"
            if task_id in line and line.startswith("- "):
                in_entry = True
                task_entries.append(line)
                if current_section in ("### Features\n", "### Fixes\n", "### Improvements\n"):
                    target_section = current_section
            elif in_entry:
                if line.startswith("  ") or line.strip() == "":
                    task_entries.append(line)
                else:
                    break
        if task_entries:
            task_snippet = "\n".join(task_entries).strip()
            main_content = "\n".join(main_lines)
            if task_snippet not in main_content:
                if target_section and target_section in main_content:
                    main_content = main_content.replace(target_section, f"{target_section}{task_snippet}\n", 1)
                elif "### Fixes\n" in main_content:
                    main_content = main_content.replace("### Fixes\n", f"### Fixes\n{task_snippet}\n", 1)
                elif "### Improvements\n" in main_content:
                    main_content = main_content.replace("### Improvements\n", f"### Improvements\n{task_snippet}\n", 1)
                elif "### Features\n" in main_content:
                    main_content = main_content.replace("### Features\n", f"### Features\n{task_snippet}\n", 1)
                main_changelog.write_text(main_content, encoding="utf-8")

    # 4. Verify tests on integration branch
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(repo_root)}{os.pathsep}{env.get('PYTHONPATH', '')}"
    for tt in task.test_targets:
        test_file = repo_root / tt
        if test_file.exists():
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", "-o", "pythonpath=.", str(test_file)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if proc.returncode != 0:
                print(f"[ERROR] Tests failed on integration branch '{target_branch}' after integration: {proc.stderr or proc.stdout}")
                _restore_integration_files(repo_root, backups)
                _rollback_branch_on_failure()
                result = hub.transition_task(
                    ledger, task_id, TaskStatus.RECOVERY_REQUIRED, actor="integration",
                    failure_reason="Post-integration verification failed.",
                    checkpoint_stage="POST_INTEGRATION_CHECK_FAILED",
                    checkpoint_details={"test_target": tt, "exit_code": proc.returncode,
                                        "artifact_sha256": patch_sha256},
                    event_type="POST_INTEGRATION_VERIFICATION_FAILED",
                    idempotency_key=f"postcheck-failed:{task_id}:{patch_sha256}:{tt}",
                )
                if not result.accepted:
                    print(f"[ERROR] Recovery transition persistence failed: {result.message}")
                return 1

    # 5. Git commit on dedicated integration branch
    files_to_add = list(task.target_files) + ["docs/CHANGELOG.md"]
    subprocess.run(["git", "add", "-A", "--"] + files_to_add, cwd=str(repo_root), check=True)
    commit_msg = f"fix(core): {task.title.lower()} ({task.task_id})"
    c_proc = subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(repo_root), capture_output=True, text=True, check=False)
    if c_proc.returncode != 0 and "nothing to commit" not in c_proc.stdout:
        print(f"[ERROR] Failed to commit on integration branch '{target_branch}': {c_proc.stderr or c_proc.stdout}")
        subprocess.run(["git", "reset", "--", *files_to_add], cwd=str(repo_root), capture_output=True, check=False)
        _restore_integration_files(repo_root, backups)
        _rollback_branch_on_failure()
        result = hub.transition_task(
            ledger, task_id, TaskStatus.RECOVERY_REQUIRED, actor="integration",
            failure_reason="Integration commit failed.", checkpoint_stage="INTEGRATION_COMMIT_FAILED",
            checkpoint_details={"artifact_sha256": patch_sha256}, event_type="INTEGRATION_FAILED",
            idempotency_key=f"integration-commit-failed:{task_id}:{patch_sha256}",
        )
        if not result.accepted:
            print(f"[ERROR] Could not persist commit-failure recovery state: {result.message}")
        return 1

    if c_proc.returncode == 0 and "nothing to commit" not in c_proc.stdout:
        # Once the integration commit succeeds, recovery must not pretend it can
        # roll main back. Persist that post-checks passed before workspace cleanup.
        log_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, check=True)
        head_commit = log_proc.stdout.strip()
        result = hub.transition_task(
            ledger, task_id, TaskStatus.VERIFIED, actor="integration", authorized=True,
            checkpoint_stage="POST_INTEGRATION_VERIFIED",
            checkpoint_details={"postcheck_passed": True, "commit": head_commit, "branch": target_branch,
                                "artifact_sha256": patch_sha256, "baseline_commit": baseline_commit,
                                "verification_sha256": verification_sha256,
                                "approval_request_id": approval.get("request_id")},
            event_type="TASK_VERIFIED_AFTER_INTEGRATION",
            idempotency_key=f"task-verified:{task_id}:{head_commit}:{patch_sha256}",
        )
        if not result.accepted:
            print(f"[CRITICAL] Integration commit {head_commit[:7]} and checks succeeded, but final ledger state was not persisted: {result.message}")
            return 1

    log_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, check=True)
    head_commit = log_proc.stdout.strip()

    # 6. Synchronize worktree branch to target_branch
    reset_proc = subprocess.run(["git", "reset", "--hard", target_branch], cwd=str(ws_path), capture_output=True, check=False)
    if reset_proc.returncode:
        print("[ERROR] Integration commit passed post-checks, but workspace reset failed; task remains in recovery.")
        result = hub.transition_task(
            ledger, task_id, TaskStatus.RECOVERY_REQUIRED, actor="integration",
            failure_reason=f"Integration commit on '{target_branch}' succeeded but task workspace reset failed.",
            checkpoint_stage="INTEGRATION_WORKSPACE_RECOVERY_REQUIRED",
            checkpoint_details={"commit": head_commit, "artifact_sha256": patch_sha256},
            event_type="INTEGRATION_RECOVERY_REQUIRED",
            idempotency_key=f"integration-reset-failed:{task_id}:{head_commit}",
        )
        if not result.accepted:
            print(f"[ERROR] Failed to persist recovery state: {result.message}")
            return 1

    print(f"\n[SUCCESS] Task {task_id} integrated and verified on dedicated branch '{target_branch}' at commit {head_commit[:7]}.")
    print(f"Per repository governance, '{target_branch}' has NOT been merged into 'main'. Human authorization is required to merge into 'main'.")
    print(f"Verified files: {', '.join(files_to_add)}")

    # 7. Post-integration dependency revalidation assessment
    try:
        from .dependency import DependencyRegistry
        from .revalidation import RevalidationOrchestrator
    except ImportError:
        from dependency import DependencyRegistry
        from revalidation import RevalidationOrchestrator

    registry = DependencyRegistry(repo_root)
    reval_orch = RevalidationOrchestrator(store, registry, repo_root=repo_root)
    summary = reval_orch.process_change_event(
        changed_files=files_to_add,
        trigger_task_id=task_id,
        trigger_type="INTEGRATION_COMMIT",
    )
    if summary.invalidated_tasks:
        print(f"[REVALIDATION] Invalidation triggered for {len(summary.invalidated_tasks)} task(s): {', '.join(summary.invalidated_tasks)}")
    if summary.revalidation_tasks_created:
        print(f"[REVALIDATION] Created {len(summary.revalidation_tasks_created)} revalidation task(s): {', '.join(summary.revalidation_tasks_created)}")
    if summary.revalidation_tasks_updated:
        print(f"[REVALIDATION] Updated {len(summary.revalidation_tasks_updated)} revalidation task(s): {', '.join(summary.revalidation_tasks_updated)}")
    if summary.human_review_required:
        print(f"[REVALIDATION] Human review required for: {', '.join(summary.human_review_required)}")

    return 0


def _restore_integration_files(repo_root: Path, backups: dict[str, bytes | None]) -> None:
    """Restore only exact files Autophage snapshotted before integration began."""
    for rel, content in backups.items():
        target = repo_root / rel
        if content is None:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MayaVE VE11 Overnight Maintenance Coordinator (Milestone 3: Controlled Single-Task Execution & Quality Gates)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run scheduling simulation or execution simulation in read-only mode")
    parser.add_argument("--status", action="store_true", help="Display task queue status table and metrics")
    parser.add_argument("--next", action="store_true", help="Print only the next eligible task ID and attributes")
    parser.add_argument("--validate", action="store_true", help="Run comprehensive pre-validation checks")
    parser.add_argument("--init", action="store_true", help="Initialize persistent task ledger with default audit inventory")
    parser.add_argument("--force", action="store_true", help="Force operation (e.g. overwrite on --init or clear stale lock)")
    parser.add_argument("--state-path", type=str, default=None, help="Custom path to state.json")
    parser.add_argument("--repo-root", type=str, default=None, help="Custom path to repository root")
    parser.add_argument("--prepare-workspace", type=str, metavar="TASK_ID", help="Explicitly prepare an isolated Git worktree for TASK_ID")
    parser.add_argument("--workspace-status", nargs="?", const="ALL", metavar="TASK_ID", help="Inspect workspace status for a task or ALL tasks")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile task state on coordinator restart and detect interruptions")
    parser.add_argument("--cleanup-preview", action="store_true", help="Produce a non-destructive cleanup preview of all managed worktrees")
    parser.add_argument("--cleanup", nargs="?", const="ALL", metavar="TASK_ID", help="Safely remove finished worktree and delete its merged local branch (requires --authorize; use with --dry-run for preview)")
    parser.add_argument("--pending-merges", action="store_true", help="Display visibility into pending and completed integration branches (read-only governance view)")
    parser.add_argument("--list-backups", action="store_true", help="List available rotating session state backups")
    parser.add_argument("--restore-backup", type=str, metavar="BACKUP_PATH", help="Restore state.json from an earlier backup snapshot (requires --authorize)")
    parser.add_argument("--run-adaptive", action="store_true", help="Run maintenance loop with adaptive worker scaling")
    parser.add_argument("--initial-workers", type=int, default=1, help="Initial worker concurrency (default: 1)")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum concurrent implementation workers (default: 1 for pilot)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum number of tasks to execute per run (default: unlimited until empty queue or timeout)")
    parser.add_argument(
        "--hard-timeout", "--timeout-seconds",
        dest="hard_timeout",
        type=float,
        default=21600.0,
        help="Hard session timeout in seconds for unattended worker session (default: 21600 / 6h, max: 86400 / 24h)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing active or interrupted maintenance session against its original deadline",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new maintenance session regardless of previous session state",
    )
    parser.add_argument("--max-risk", choices=["LOW", "MEDIUM"], default="MEDIUM", help="Maximum risk level permitted for unattended execution (default: MEDIUM)")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True, help="Stop scheduling after a task failure or exception (default: True)")
    parser.add_argument("--terminal-approval", action="store_true", help="Launch interactive terminal approval session for pending permission requests")
    parser.add_argument("--list-approvals", action="store_true", help="List all pending permission requests awaiting operator review")
    parser.add_argument("--approve-request", type=str, metavar="REQ_ID", help="Approve specified permission request for its exact reviewed scope")
    parser.add_argument("--deny-request", type=str, metavar="REQ_ID", help="Deny specified permission request")
    parser.add_argument("--inspect-request", type=str, metavar="REQ_ID", help="Inspect diagnostic details and worktree state for a permission request")
    parser.add_argument("--approval-scope", type=str, default=None, help="Explicit scope for request approval (must match or refine reviewed scope)")
    parser.add_argument(
        "--recover-task",
        type=str,
        metavar="TASK_ID",
        help="Explicitly authorize recovery/retry of a held or failed task, transitioning it to PENDING (requires --authorize)",
    )
    parser.add_argument(
        "--recovery-reason",
        type=str,
        default=None,
        help="Explicit justification for recovering a held or failed task",
    )
    parser.add_argument("--execute-task", type=str, metavar="TASK_ID", help="Execute specified task through single-task orchestrator")
    parser.add_argument("--execute-next", action="store_true", help="Execute next eligible task through single-task orchestrator")
    parser.add_argument("--integrate-task", type=str, metavar="TASK_ID", default=None, help="Integrate verified task from worktree into main HEAD (requires --authorize)")
    parser.add_argument("--authorize", action="store_true", help="Explicit operator authorization required to invoke agent execution or integration")
    parser.add_argument("--agent-adapter", choices=["antigravity", "fake", "local", "tiered"], default="tiered", help="Agent adapter implementation to use (default: tiered — Local Ollama model with Antigravity fallback)")
    parser.add_argument("--route-complexity", action="store_true", help="Enable P3 complexity-aware dynamic model routing across tiers")
    parser.add_argument("--agent-executable", type=str, default=None, help="Custom path to Antigravity CLI (agy.exe) executable")
    parser.add_argument("--reviewer-adapter", choices=["copilot", "fake", "none"], default="copilot", help="Reviewer adapter implementation to use")
    parser.add_argument("--max-reviews", type=int, default=5, help="Maximum independent code reviews per session (default: 5)")
    parser.add_argument("--timeout", type=int, default=600, help="Agent execution timeout in seconds (default: 600)")
    parser.add_argument(
        "--local-model",
        type=str,
        default="qwen2.5-coder:3b",
        help="Local model identifier for Ollama execution (default: qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--probe-models",
        action="store_true",
        help="Probe local Ollama and Antigravity runtimes for model availability",
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Run Tier 2 cloud model offline benchmark harness",
    )
    parser.add_argument(
        "--validate-contract",
        type=str,
        metavar="PATH",
        help="Validate a JSON task contract file against the structured schema",
    )
    parser.add_argument(
        "--plan-objective",
        type=str,
        metavar="OBJECTIVE",
        help="Generate a structured maintenance plan using local qwen2.5-coder:3b planner",
    )
    parser.add_argument(
        "--plan-file",
        type=str,
        metavar="PATH",
        help="Generate a structured maintenance plan from an objective file",
    )
    parser.add_argument(
        "--ingest-backlog",
        type=str,
        default=None,
        metavar="PATH",
        help="Ingest a Markdown bug backlog (.md) into the canonical manifest and maintenance queue",
    )
    parser.add_argument(
        "--backlog-manifest",
        type=str,
        default=None,
        metavar="PATH",
        help="Custom output path for generated canonical JSON manifest",
    )
    parser.add_argument(
        "--risk-policy",
        choices=["auto-low-medium", "manual"],
        default="manual",
        help=(
            "Risk-based approval policy (default: manual; requires explicit operator approval for all tasks). "
            "Under 'auto-low-medium', LOW and MEDIUM risk tasks are automatically approved at task-approval gates "
            "with auditable evidence, while HIGH and uncertain risk tasks always remain held for operator decision. "
            "Note: when used with --ingest-backlog, --authorize is still strictly required to register tasks."
        ),
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Stage plan proposal without registering to ledger, even if --authorize is provided",
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        help="Enforce independent reviewer verification on all repair tasks (including LOW risk)",
    )
    parser.add_argument(
        "--max-plan-tasks",
        type=int,
        default=5,
        help="Maximum tasks to generate in a single plan (default: 5, hard limit: 8)",
    )
    parser.add_argument(
        "--list-plans",
        action="store_true",
        help="List all proposed, approved, and registered maintenance plans",
    )
    parser.add_argument(
        "--inspect-plan",
        type=str,
        metavar="PLAN_ID",
        help="Inspect details of a maintenance plan proposal",
    )
    parser.add_argument(
        "--approve-plan",
        type=str,
        metavar="PLAN_ID",
        help="Approve and register a plan proposal into the active ledger (requires --authorize)",
    )
    parser.add_argument(
        "--reject-plan",
        type=str,
        metavar="PLAN_ID",
        help="Reject a maintenance plan proposal with optional --plan-reason",
    )
    parser.add_argument(
        "--plan-reason",
        type=str,
        default="Operator rejected",
        help="Reason for rejecting a plan proposal (used with --reject-plan)",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch read-only terminal dashboard (Rich observer)",
    )
    parser.add_argument(
        "--operate",
        action="store_true",
        help="Launch interactive operator dashboard (Rich TUI with approval/operator controls)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Display trace events in non-TTY plain-text format",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable append-only JSONL tracing in .maya/maintenance/trace/",
    )
    parser.add_argument(
        "--replay",
        nargs="?",
        const="LATEST",
        metavar="SESSION_ID",
        help="Replay recorded trace session in terminal dashboard",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be at least 1.")

    if args.max_plan_tasks is not None:
        if args.max_plan_tasks < 1:
            parser.error("--max-plan-tasks must be at least 1.")
        if args.max_plan_tasks > 8:
            parser.error("--max-plan-tasks cannot exceed hard maximum limit of 8.")

    if args.resume and args.new_session:
        parser.error("Cannot combine --resume and --new-session.")

    if args.hard_timeout is not None:
        import math
        if math.isnan(args.hard_timeout) or math.isinf(args.hard_timeout) or args.hard_timeout <= 0:
            parser.error("--hard-timeout / --timeout-seconds must be a positive finite number.")
        if args.hard_timeout > 86400.0:
            parser.error("--hard-timeout / --timeout-seconds cannot exceed 86400.0 seconds (24 hours).")

    if args.local_model is not None:
        val = args.local_model.strip()
        if not val:
            parser.error("--local-model cannot be empty.")
        import re
        if not re.match(r"^[a-zA-Z0-9_.:-]+$", val):
            parser.error(f"Invalid model identifier for --local-model: '{args.local_model}'.")

    args.timeout_seconds = args.hard_timeout
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root, state_path = _resolve_default_paths(args.state_path, args.repo_root)

    if args.trace or args.tui or args.operate:
        try:
            from .trace import enable_trace
        except ImportError:
            try:
                from trace import enable_trace
            except ImportError:
                enable_trace = None
        if enable_trace:
            enable_trace(trace_dir=repo_root / ".maya" / "maintenance" / "trace")

    if args.tui or args.plain or args.replay or args.operate:
        try:
            from .tui import run_tui, run_plain
        except ImportError:
            from tui import run_tui, run_plain
        session_target = None
        if args.replay and args.replay != "LATEST":
            session_target = args.replay
        if args.plain:
            return run_plain(session_id=session_target, repo_root=repo_root, operate=bool(args.operate))
        return run_tui(session_id=session_target, repo_root=repo_root, replay=bool(args.replay), operate=bool(args.operate))

    store = StateStore(state_path)

    if args.terminal_approval:
        return cmd_terminal_approval(repo_root, store=store)

    if args.probe_models:
        return cmd_probe_models(repo_root)

    if args.run_benchmark:
        return cmd_run_benchmark(repo_root)

    if args.validate_contract:
        return cmd_validate_contract(args.validate_contract)

    if args.plan_objective:
        return cmd_plan_objective(
            repo_root=repo_root,
            store=store,
            objective=args.plan_objective,
            max_tasks=args.max_plan_tasks,
            model=args.local_model,
        )

    if args.plan_file:
        return cmd_plan_file(
            repo_root=repo_root,
            store=store,
            file_path_str=args.plan_file,
            max_tasks=args.max_plan_tasks,
            model=args.local_model,
        )

    if args.ingest_backlog:
        return cmd_ingest_backlog(
            repo_root=repo_root,
            store=store,
            backlog_file_str=args.ingest_backlog,
            risk_policy=args.risk_policy,
            authorized=args.authorize,
            stage_only=args.stage_only,
            output_manifest_str=args.backlog_manifest,
        )

    if args.list_plans:
        return cmd_list_plans(store=store)

    if args.inspect_plan:
        return cmd_inspect_plan(store=store, plan_id=args.inspect_plan)

    if args.approve_plan:
        return cmd_approve_plan(
            store=store,
            plan_id=args.approve_plan,
            authorized=args.authorize,
        )

    if args.reject_plan:
        return cmd_reject_plan(
            store=store,
            plan_id=args.reject_plan,
            reason=args.plan_reason,
        )

    if args.list_approvals:
        return cmd_list_approvals(repo_root)

    if args.approve_request:
        return cmd_approve_request(repo_root, args.approve_request, scope=args.approval_scope, store=store)

    if args.deny_request:
        return cmd_deny_request(repo_root, args.deny_request)

    if args.inspect_request:
        return cmd_inspect_request(repo_root, args.inspect_request)

    if args.recover_task:
        return cmd_recover_task(
            repo_root=repo_root,
            store=store,
            task_id=args.recover_task,
            authorized=args.authorize,
            reason=args.recovery_reason,
        )

    if args.init:
        return cmd_init(store, force=args.force)

    if args.status:
        return cmd_status(store, repo_root=repo_root)

    if args.next:
        return cmd_next(store, repo_root=repo_root)

    if args.validate:
        return cmd_validate(repo_root, store)

    if args.prepare_workspace:
        return cmd_prepare_workspace(repo_root, store, args.prepare_workspace)

    if args.workspace_status:
        target_id = None if args.workspace_status == "ALL" else args.workspace_status
        return cmd_workspace_status(repo_root, store, target_id)

    if args.reconcile:
        return cmd_reconcile(repo_root, store)

    if args.cleanup_preview:
        return cmd_cleanup_preview(repo_root, store)

    if args.cleanup:
        return cmd_cleanup(
            repo_root=repo_root,
            store=store,
            task_id=args.cleanup,
            authorized=args.authorize,
            dry_run=args.dry_run,
        )

    if args.pending_merges:
        return cmd_pending_merges(repo_root, store)

    if args.list_backups:
        return cmd_list_backups(store)

    if args.restore_backup:
        return cmd_restore_backup(store, args.restore_backup, authorized=args.authorize)

    if args.run_adaptive:
        return cmd_run_adaptive(
            repo_root=repo_root,
            store=store,
            initial_workers=args.initial_workers,
            max_workers=args.max_workers,
            authorized=args.authorize,
            dry_run=args.dry_run,
            adapter_name=args.agent_adapter,
            reviewer_name=args.reviewer_adapter,
            max_reviews=args.max_reviews,
            timeout_seconds=args.timeout,
            agent_executable=args.agent_executable,
            max_tasks=args.max_tasks,
            hard_timeout=args.hard_timeout,
            max_risk=args.max_risk,
            stop_on_failure=args.stop_on_failure,
            force_new_session=args.new_session,
            resume=args.resume,
            local_model=args.local_model,
            route_complexity=args.route_complexity,
            risk_policy=args.risk_policy,
            review_all=args.review_all,
        )

    if args.integrate_task:
        return cmd_integrate(
            repo_root=repo_root,
            store=store,
            task_id=args.integrate_task,
            authorized=args.authorize,
        )

    if args.execute_task:
        return cmd_execute(
            repo_root=repo_root,
            store=store,
            task_id=args.execute_task,
            authorized=args.authorize,
            dry_run=args.dry_run,
            adapter_name=args.agent_adapter,
            reviewer_name=args.reviewer_adapter,
            max_reviews=args.max_reviews,
            timeout_seconds=args.timeout,
            agent_executable=args.agent_executable,
            route_complexity=args.route_complexity,
            local_model=args.local_model,
        )

    if args.execute_next:
        return cmd_execute(
            repo_root=repo_root,
            store=store,
            task_id=None,
            authorized=args.authorize,
            dry_run=args.dry_run,
            adapter_name=args.agent_adapter,
            reviewer_name=args.reviewer_adapter,
            max_reviews=args.max_reviews,
            timeout_seconds=args.timeout,
            agent_executable=args.agent_executable,
            route_complexity=args.route_complexity,
            local_model=args.local_model,
        )

    if args.dry_run:
        return cmd_dry_run(repo_root, store)

    # Default action if no flags provided: display dry run
    return cmd_dry_run(repo_root, store)


if __name__ == "__main__":
    sys.exit(main())
