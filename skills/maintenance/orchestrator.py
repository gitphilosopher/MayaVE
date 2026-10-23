"""
Single-Task Execution Orchestrator for MayaVE Maintenance Coordinator.
Enforces lock exclusivity, supervised single-task execution, bounded prompting,
quality-gate evaluation, and non-destructive interruption recovery.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("maya.maintenance.orchestrator")

try:
    from .adapter import AgentAdapter, AgentExecutionResult, FakeAgentAdapter, LocalModelAgentAdapter, AntigravityAgentAdapter
    from .cache import ContextCache
    from .ledger import TaskLedger
    from .hub import CoordinatorAPI, compute_patch_sha256, compute_verification_sha256
    from .models import CheckpointRecord, EvidenceRecord, TaskDefinition, TaskRisk, TaskStatus, utc_now_iso
    from .quality import QualityGateEngine, QualityReport, freeze_task_tests, is_valid_defect_baseline
    from .reviewer import (
        FakeReviewerAdapter,
        ReviewPolicy,
        ReviewRecommendation,
        ReviewResult,
        ReviewerAdapter,
    )
    from .scheduler import DependencyScheduler
    from .store import StateStore
    from .worktree import WorktreeManager, safe_restore_worktree_attempt
    from .routing import ComplexityRouter, FallbackDecision, RoutingDecision, RoutingTier, WorkerConcurrencyGuard
    from .reviewer_safeguards import ReviewerSafeguardScanner, ScanReport
    from .terminal_approval import TerminalApprovalManager
except ImportError:
    from adapter import AgentAdapter, AgentExecutionResult, FakeAgentAdapter, LocalModelAgentAdapter, AntigravityAgentAdapter
    from cache import ContextCache
    from ledger import TaskLedger
    from hub import CoordinatorAPI, compute_patch_sha256, compute_verification_sha256
    from models import CheckpointRecord, EvidenceRecord, TaskDefinition, TaskRisk, TaskStatus, utc_now_iso
    from quality import QualityGateEngine, QualityReport, freeze_task_tests, is_valid_defect_baseline
    from reviewer import (
        FakeReviewerAdapter,
        ReviewPolicy,
        ReviewRecommendation,
        ReviewResult,
        ReviewerAdapter,
    )
    from scheduler import DependencyScheduler
    from store import StateStore
    from worktree import WorktreeManager, safe_restore_worktree_attempt
    from routing import ComplexityRouter, FallbackDecision, RoutingDecision, RoutingTier, WorkerConcurrencyGuard
    from reviewer_safeguards import ReviewerSafeguardScanner, ScanReport
    from terminal_approval import TerminalApprovalManager

try:
    from .trace import TraceEvent, emit as trace_emit
except ImportError:
    try:
        from trace import TraceEvent, emit as trace_emit
    except ImportError:
        def trace_emit(*args, **kwargs): pass
        class TraceEvent:
            def __init__(self, **kwargs): pass


@dataclass
class ExecutionSummary:
    task_id: Optional[str]
    success: bool
    final_status: Optional[TaskStatus]
    dry_run: bool
    authorized: bool
    duration_sec: float
    worktree_path: Optional[str] = None
    agent_result: Optional[dict] = None
    quality_report: Optional[dict] = None
    review_result: Optional[dict] = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "final_status": self.final_status.value if self.final_status else None,
            "dry_run": self.dry_run,
            "authorized": self.authorized,
            "duration_sec": round(self.duration_sec, 2),
            "worktree_path": self.worktree_path,
            "agent_result": self.agent_result,
            "quality_report": self.quality_report,
            "review_result": self.review_result,
            "message": self.message,
        }


class SingleTaskOrchestrator:
    """Coordinates the execution lifecycle for a single maintenance task."""

    def __init__(
        self,
        repo_root: Path | str,
        store: StateStore,
        ledger: TaskLedger,
        worktree_mgr: WorktreeManager,
        scheduler: Optional[DependencyScheduler] = None,
        adapter: Optional[AgentAdapter] = None,
        quality_engine: Optional[QualityGateEngine] = None,
        reviewer: Optional[ReviewerAdapter] = None,
        review_policy: Optional[ReviewPolicy] = None,
        cache: Optional[ContextCache] = None,
        timeout_seconds: int = 600,
        reviewer_lock: Optional[threading.Lock] = None,
        router: Optional[ComplexityRouter] = None,
        concurrency_guard: Optional[WorkerConcurrencyGuard] = None,
        local_adapter: Optional[AgentAdapter] = None,
        safeguard_scanner: Optional[ReviewerSafeguardScanner] = None,
        approval_manager: Optional[TerminalApprovalManager] = None,
        risk_policy: str = "manual",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.store = store
        self.coordinator_api = CoordinatorAPI(store)
        self.ledger = ledger
        self.worktree_mgr = worktree_mgr
        self.scheduler = scheduler or DependencyScheduler(self.ledger)
        self.adapter = adapter or FakeAgentAdapter()
        self._local_adapter = local_adapter
        self.quality_engine = quality_engine or QualityGateEngine(self.repo_root)
        self.reviewer = reviewer or FakeReviewerAdapter()
        self.review_policy = review_policy or ReviewPolicy()
        self.reviewer_lock = reviewer_lock
        self._review_cache: dict[tuple[str, str], ReviewResult] = {}
        cache_file = self.repo_root / ".maya" / "maintenance" / "cache.json"
        self.cache = cache or ContextCache(cache_file)
        self.timeout_seconds = timeout_seconds
        self.router = router
        self.concurrency_guard = concurrency_guard
        self.safeguard_scanner = safeguard_scanner or ReviewerSafeguardScanner()
        self.risk_policy = risk_policy
        if approval_manager is not None:
            self.approval_manager = approval_manager
        else:
            approvals_path = self.repo_root / ".maya" / "maintenance" / "approvals.json"
            self.approval_manager = TerminalApprovalManager(approvals_path, repo_root=self.repo_root, store=self.store)

    def build_prompt(self, task: TaskDefinition, worktree_path: Path) -> str:
        """
        Construct a self-contained, bounded implementation prompt for the coding agent.
        """
        target_files_str = "\n".join(f"- {f}" for f in task.target_files)
        test_targets_str = "\n".join(f"- {t}" for t in task.test_targets) if task.test_targets else "- None"
        criteria_str = "\n".join(f"- {c}" for c in task.acceptance_criteria) if task.acceptance_criteria else "- Fix confirmed defect"

        context_info = self.cache.get_task_context(task, worktree_path)

        previous_feedback = ""
        if task.checkpoint and isinstance(task.checkpoint.details, dict):
            last_gate_fail = task.checkpoint.details.get("last_quality_gate_failure")
            if last_gate_fail and isinstance(last_gate_fail, dict):
                prev_attempt = last_gate_fail.get("attempt", "?")
                summary_text = last_gate_fail.get("summary", "")
                previous_feedback = (
                    f"\n\n=== PREVIOUS ATTEMPT QUALITY GATE FAILURE (Attempt {prev_attempt}) ===\n"
                    f"Your previous repair proposal was applied to the worktree but failed quality gate verification:\n"
                    f"{summary_text}\n\n"
                    f"Please adjust your implementation to satisfy all verification requirements."
                )

        # Query bounded repair memory context
        repair_memory_snippet = ""
        try:
            try:
                from .repair_memory import RepairMemoryStore
            except ImportError:
                from repair_memory import RepairMemoryStore
            mem_store = RepairMemoryStore(self.repo_root / ".maya" / "maintenance" / "repair_memory.jsonl")
            formatted, selected_recs = mem_store.get_prompt_context_with_metadata(
                target_files=task.target_files,
                defect_category=task.audit_source,
                max_chars=800,
            )
            if formatted:
                repair_memory_snippet = f"\n\n{formatted}\n"
                if task.checkpoint and isinstance(task.checkpoint.details, dict):
                    task.checkpoint.details["memory_selected"] = [r.task_id for r in selected_recs]
        except Exception as mem_err:
            logger.debug("Could not retrieve repair memory for orchestrator prompt: %s", mem_err)

        prompt = (
            f"You are acting as the lightweight MayaVE Implementer role on Task {task.task_id}.\n\n"
            f"=== TASK SPECIFICATION ===\n"
            f"Task ID   : {task.task_id}\n"
            f"Title     : {task.title}\n"
            f"Priority  : {task.priority.value}\n"
            f"Risk      : {task.risk.value}\n"
            f"Audit Ref : {task.audit_source}\n\n"
            f"Description:\n{task.description}\n\n"
            f"Evidence of Defect:\n{task.evidence}{previous_feedback}{repair_memory_snippet}\n\n"
            f"Target Files Allowed to Modify:\n{target_files_str}\n\n"
            f"Verification Test Targets:\n{test_targets_str}\n\n"
            f"Acceptance Criteria:\n{criteria_str}\n\n"
            f"Verification Instructions:\n{task.verification_instructions}\n\n"
            f"Context Digest: {context_info['context_digest']}\n\n"
            f"=== STRICT OPERATIONAL INVARIANTS ===\n"
            f"1. NO COMMAND OR SHELL TOOLS (FILE TOOLS ONLY): Do NOT invoke `run_command`, terminal tools, shell commands (e.g. PowerShell, Bash, dir, ls, Get-ChildItem), or other command-execution tools during implementation. All shell command tools require interactive human permissions that are strictly unavailable in headless mode, triggering an automatic permission denial that halts execution immediately. Inspect source files and tests exclusively through available file-reading tools (e.g. `read_file`), and navigate using the task specification and known repository structure rather than listing directories through a shell.\n"
            f"2. MANDATORY TOOL-BACKED IMPLEMENTATION: You must perform actual implementation using your file-editing tools (e.g. `edit_file`, `write_file`) to directly modify the declared target files on disk. Do NOT merely describe a solution, propose code snippets, or return an explanation without writing changes to disk. Non-modifying turns will fail quality gates.\n"
            f"3. WORKTREE CONFINEMENT & GIT ISOLATION: All file reads, inspections, and modifications must remain strictly confined inside your current working directory (the assigned task worktree). NEVER follow `.git` worktree pointers or attempt to read, write, or inspect Git administrative paths (such as `.git/`, `.git/worktrees/...`, `HEAD`, `config`, or any file outside the worktree root). In headless mode, traversing outside the worktree triggers an automatic sandbox security denial and immediately terminates execution.\n"
            f"4. SCOPE BOUNDARY: You may ONLY modify files listed in 'Target Files Allowed to Modify' and 'docs/CHANGELOG.md'. Modifying any other file violates quality gates and causes immediate task rejection.\n"
            f"5. TEST POLICY & INTEGRITY: All tests must remain strictly under '.maya/tests/'. Do NOT create or modify any test files outside '.maya/tests/'. You must NEVER weaken, delete, or comment out test assertions, and never add test skips. The fix must satisfy all existing assertions.\n"
            f"6. CHANGELOG MAINTENANCE: Record your change in 'docs/CHANGELOG.md' under '### Fixes' or '### Improvements' in the same changeset.\n"
            f"7. GIT RESTRICTIONS: Do not run git stage, commit, push, or destructive commands (such as git reset --hard or git clean -fd).\n"
            f"8. HONEST REPORTING & FORBIDDEN COMMAND PREVENTION: If file-editing tools cannot complete the task, required tools are unavailable, or access is denied, report the specific limitation honestly and explicitly in your output instead of attempting a forbidden command. Do not fabricate successful changes or claim completion without edits."
        )
        return prompt

    def _save_diagnostic_report(
        self,
        task: TaskDefinition,
        agent_result: AgentExecutionResult,
        quality_report: Optional[QualityReport] = None,
        review_result: Optional[ReviewResult] = None,
    ) -> Optional[Path]:
        """
        Persist structured failure diagnostics to .maya/reports/ for auditing.
        Enforces path safety and strict directory confinement within .maya/reports/.
        """
        try:
            diag_dir = (Path(self.repo_root) / ".maya" / "reports").resolve()
            diag_dir.mkdir(parents=True, exist_ok=True)

            raw_id = task.task_id or "unknown_task"
            safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(raw_id).name)
            if not safe_id:
                safe_id = "sanitized_task"

            diag_path = (diag_dir / f"diagnostic-{safe_id}.json").resolve()

            if not diag_path.is_relative_to(diag_dir):
                logger.error("Path traversal detected in task_id '%s': resolved to '%s'", task.task_id, diag_path)
                return None

            diag_data = {
                "task_id": task.task_id,
                "timestamp": utc_now_iso(),
                "status": task.status.value,
                "failure_reason": task.failure_reason,
                "blocking_reason": task.blocking_reason,
                "agent_result": agent_result.to_dict(max_output_chars=8000),
                "quality_report": quality_report.to_dict() if quality_report else None,
                "review_result": review_result.to_dict() if review_result else None,
            }
            diag_path.write_text(json.dumps(diag_data, indent=2), encoding="utf-8")
            return diag_path
        except Exception as e:
            logger.warning("Failed to save diagnostic report for task %s: %s", getattr(task, "task_id", "?"), e)
            return None

    def _is_escalation_eligible(self, task: TaskDefinition) -> bool:
        """Check if task is eligible for adapter tier escalation."""
        return False

    def _update_checkpoint(
        self,
        task: TaskDefinition,
        stage: str,
        stage_details: dict[str, Any],
    ) -> None:
        """
        Update task.checkpoint preserving persistent metadata
        while cleanly updating stage and stage-specific details.
        Avoids mutating shared dictionaries or retaining stale transient execution fields.
        """
        current_details: dict[str, Any] = (
            dict(task.checkpoint.details)
            if (task.checkpoint and isinstance(task.checkpoint.details, dict))
            else {}
        )
        if stage == "AGENT_EXECUTION":
            for transient_key in (
                "exit_code",
                "stdout",
                "stderr",
                "duration_sec",
                "interrupted",
                "timed_out",
                "permission_denied",
                "error_message",
                "success",
            ):
                current_details.pop(transient_key, None)

        current_details.update(stage_details)
        task.checkpoint = CheckpointRecord(
            timestamp=utc_now_iso(),
            stage=stage,
            details=current_details,
        )

    def execute_single_task(
        self,
        task_id: Optional[str] = None,
        authorized: bool = False,
        dry_run: bool = False,
        acquire_coarse_lock: bool = True,
    ) -> ExecutionSummary:
        """
        Execute a single task through the controlled pipeline under lock exclusivity.
        """
        start_time = time.monotonic()

        lock_ctx = self.store.acquire_lock() if acquire_coarse_lock else contextlib.nullcontext()

        def _persist_task(t: TaskDefinition) -> None:
            if dry_run:
                return
            if t.status != self.ledger._status_snapshot.get(t.task_id):
                raise RuntimeError("Protected task status changes must use CoordinatorAPI.transition_task().")
            self.ledger.update(t)
            if acquire_coarse_lock:
                self.store.save_ledger(self.ledger)
            else:
                persisted = self.store.load_ledger()
                persisted.update(t)
                self.store.save_ledger(persisted)
                self.coordinator_api._copy_ledger_into(self.ledger, persisted)

        def _transition(
            status: TaskStatus,
            *,
            stage: str,
            details: dict,
            event_type: str,
            failure_reason: str | None = None,
            blocking_reason: str | None = None,
            idempotency_key: str,
        ) -> None:
            if dry_run:
                return
            result = self.coordinator_api.transition_task(
                self.ledger,
                task.task_id,
                status,
                actor="orchestrator",
                authorized=authorized,
                failure_reason=failure_reason,
                blocking_reason=blocking_reason,
                checkpoint_stage=stage,
                checkpoint_details=details,
                event_type=event_type,
                idempotency_key=idempotency_key,
                lock_held=acquire_coarse_lock,
            )
            if not result.accepted:
                raise RuntimeError(f"Coordinator rejected {event_type}: {result.error_code}: {result.message}")

            trace_emit(
                TraceEvent(
                    category="lifecycle",
                    type=event_type,
                    provenance="captured",
                    task_id=task.task_id if task else None,
                    severity="WARN" if (failure_reason or blocking_reason) else "INFO",
                    summary=f"Task transitioned to {status.name} [{stage}]: {failure_reason or blocking_reason or 'ok'}",
                    payload=details if isinstance(details, dict) else {},
                )
            )

        with lock_ctx:
            # 1. Reconcile interrupted or in-flight work on startup (skipped in dry-run mode)
            if acquire_coarse_lock and not dry_run:
                self.worktree_mgr.reconcile_all(
                    self.ledger, coordinator_api=self.coordinator_api, lock_held=True
                )

            # 2. Select task
            task: Optional[TaskDefinition] = None
            if task_id:
                task = self.ledger.get(task_id)
                if not task:
                    return ExecutionSummary(
                        task_id=task_id,
                        success=False,
                        final_status=None,
                        dry_run=dry_run,
                        authorized=authorized,
                        duration_sec=time.monotonic() - start_time,
                        message=f"Task '{task_id}' not found in ledger.",
                    )
                # Verify dependencies
                for dep_id in task.dependencies:
                    dep = self.ledger.get(dep_id)
                    if not dep or dep.status != TaskStatus.VERIFIED:
                        return ExecutionSummary(
                            task_id=task_id,
                            success=False,
                            final_status=task.status,
                            dry_run=dry_run,
                            authorized=authorized,
                            duration_sec=time.monotonic() - start_time,
                            message=f"Task '{task_id}' blocked by unverified dependency '{dep_id}' (status={dep.status if dep else 'MISSING'}).",
                        )
            else:
                task = self.scheduler.get_next_task()
                if not task:
                    return ExecutionSummary(
                        task_id=None,
                        success=True,
                        final_status=None,
                        dry_run=dry_run,
                        authorized=authorized,
                        duration_sec=time.monotonic() - start_time,
                        message="No eligible pending tasks available to execute.",
                    )

            # Check attempt limits
            if task.attempts >= task.max_attempts and not self._is_escalation_eligible(task):
                reason = f"Max execution attempts ({task.max_attempts}) reached."
                _transition(
                    TaskStatus.BLOCKED, stage="ATTEMPT_LIMIT", details={"attempts": task.attempts},
                    event_type="ATTEMPT_LIMIT_BLOCKED", blocking_reason=reason,
                    idempotency_key=f"attempt-limit:{task.task_id}:{task.attempts}",
                )
                return ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=dry_run,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    message=f"Task '{task.task_id}' is BLOCKED: max attempts reached.",
                )

            # 3. Check authorization for real execution under explicit risk policy
            effective_authorized = authorized
            if not dry_run and self.risk_policy == "auto-low-medium":
                try:
                    from .manifest import evaluate_risk_approval
                except ImportError:
                    from manifest import evaluate_risk_approval
                can_auto, policy_msg = evaluate_risk_approval(task.risk, risk_policy=self.risk_policy)
                if can_auto:
                    effective_authorized = True
                    task.history.append(
                        EvidenceRecord(
                            timestamp=utc_now_iso(),
                            stage="RISK_APPROVAL_POLICY",
                            details=f"Execution gate auto-approved by policy: {policy_msg}",
                            success=True,
                        )
                    )
                else:
                    # HIGH, VERY_HIGH, or uncertain risk: NEVER auto-approve
                    effective_authorized = False
                    reason = f"Execution held: {policy_msg}"
                    task.failure_reason = reason
                    if task.workspace:
                        task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                    _transition(
                        TaskStatus.NEEDS_APPROVAL,
                        stage="RISK_APPROVAL_POLICY",
                        details={"risk_policy": self.risk_policy, "risk": task.risk.value if hasattr(task.risk, "value") else str(task.risk)},
                        event_type="APPROVAL_HELD_BY_POLICY",
                        blocking_reason=policy_msg,
                        idempotency_key=f"risk-policy-hold:{task.task_id}",
                    )
                    return ExecutionSummary(
                        task_id=task.task_id,
                        success=False,
                        final_status=TaskStatus.NEEDS_APPROVAL,
                        dry_run=dry_run,
                        authorized=False,
                        duration_sec=time.monotonic() - start_time,
                        message=reason,
                    )

            if not dry_run and not effective_authorized:
                return ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=dry_run,
                    authorized=effective_authorized,
                    duration_sec=time.monotonic() - start_time,
                    message="Execution halted: Operator authorization (--authorize) required for real execution.",
                )

            # Check approval and blocker status for real execution
            if not dry_run and task.status == TaskStatus.NEEDS_APPROVAL:
                return ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=dry_run,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    message=f"Execution halted: Task '{task.task_id}' requires operator approval (NEEDS_APPROVAL): {task.blocking_reason}",
                )

            if not dry_run and task.status == TaskStatus.BLOCKED:
                return ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=dry_run,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    message=f"Execution halted: Task '{task.task_id}' is BLOCKED: {task.blocking_reason}",
                )

            if not dry_run and task.status == TaskStatus.RECOVERY_REQUIRED:
                return ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=dry_run,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    message=(
                        f"Execution halted: Task '{task.task_id}' requires recovery (RECOVERY_REQUIRED): "
                        f"{task.blocking_reason or task.failure_reason or 'Worktree needs operator recovery review.'}. "
                        "Worker execution cannot automatically resume on a worktree requiring recovery."
                    ),
                )

            # 4. Dry run early return (simulation without creating worktree or modifying state)
            if dry_run:
                duration = time.monotonic() - start_time
                ws_path = self.worktree_mgr.get_workspace_path(task.task_id)
                prompt = self.build_prompt(task, ws_path if ws_path.exists() else self.repo_root)
                return ExecutionSummary(
                    task_id=task.task_id,
                    success=True,
                    final_status=task.status,
                    dry_run=True,
                    authorized=authorized,
                    duration_sec=duration,
                    worktree_path=str(ws_path),
                    message=f"[DRY RUN] Task '{task.task_id}' prompt constructed. Execution simulated without modifying worktrees.",
                )

            guard_held = False

            def _return_summary(summary: ExecutionSummary) -> ExecutionSummary:
                nonlocal guard_held
                if guard_held and self.concurrency_guard and task:
                    self.concurrency_guard.release_implementation_slot(task.task_id)
                    guard_held = False
                return summary

            # Single-writer implementation concurrency enforcement
            if self.concurrency_guard and not self.concurrency_guard.acquire_implementation_slot(task.task_id):
                return ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=False,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    message="Execution rejected: Another implementation worker is active (single-writer invariant).",
                )
            guard_held = True

            # Complexity routing assessment (P3)
            routing_decision: Optional[RoutingDecision] = None
            if self.router:
                routing_decision = self.router.route_task(task, authorized=authorized)
                if (
                    routing_decision.selected_tier == RoutingTier.TIER_4_HUMAN_DECISION
                    or routing_decision.requires_human_approval
                ):
                    reason = "; ".join(routing_decision.reasons)
                    if task.status != TaskStatus.NEEDS_APPROVAL:
                        _transition(
                            TaskStatus.NEEDS_APPROVAL,
                            stage="ROUTING_APPROVAL_REQUIRED",
                            details=routing_decision.to_dict(),
                            event_type="ROUTING_APPROVAL_REQUIRED",
                            blocking_reason=reason,
                            idempotency_key=f"routing-approval:{task.task_id}:{task.attempts}",
                        )
                    return _return_summary(ExecutionSummary(
                        task_id=task.task_id,
                        success=False,
                        final_status=TaskStatus.NEEDS_APPROVAL,
                        dry_run=False,
                        authorized=authorized,
                        duration_sec=time.monotonic() - start_time,
                        message=f"Task '{task.task_id}' routed to human approval (Tier 4): {reason}",
                    ))

            # 5. Prepare isolated workspace
            ws_meta = self.worktree_mgr.prepare_workspace(task.task_id)
            worktree_path = Path(ws_meta.worktree_path)

            for rel in (task.target_files or []):
                src_f = self.repo_root / rel
                dst_f = worktree_path / rel
                if src_f.is_file() and not dst_f.is_file():
                    dst_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_f, dst_f)

            task.workspace = ws_meta
            self._update_checkpoint(
                task,
                stage="PREPARATION",
                stage_details={"worktree_path": str(worktree_path), "branch": ws_meta.branch_name},
            )
            _persist_task(task)

            # 5b. Freeze test targets and execute baseline verification
            frozen_hashes, freeze_errs = freeze_task_tests(task, worktree_path)
            if task.is_defect_task():
                if freeze_errs or not frozen_hashes:
                    reason = (
                        f"test doesn't prove the defect: {'; '.join(freeze_errs) if freeze_errs else 'no test targets specified'}"
                    )
                    task.failure_reason = reason
                    if task.workspace:
                        task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                    _transition(
                        TaskStatus.NEEDS_APPROVAL,
                        stage="BASELINE_VERIFICATION",
                        details={"error": "missing_or_invalid_test_targets", "freeze_errors": freeze_errs},
                        event_type="BASELINE_VERIFICATION_INCONCLUSIVE",
                        blocking_reason=reason,
                        idempotency_key=f"baseline-freeze-err:{task.task_id}",
                    )
                    return _return_summary(ExecutionSummary(
                        task_id=task.task_id,
                        success=False,
                        final_status=TaskStatus.NEEDS_APPROVAL,
                        dry_run=False,
                        authorized=authorized,
                        duration_sec=time.monotonic() - start_time,
                        worktree_path=str(worktree_path),
                        message=f"Task '{task.task_id}' halted: {reason}. Human approval required.",
                    ))
                task.frozen_tests = frozen_hashes
            elif frozen_hashes:
                task.frozen_tests = frozen_hashes

            # Run baseline verification against unpatched worktree
            baseline_gate_res = self.quality_engine.run_verification_tests(task, worktree_path)
            baseline_evidence = baseline_gate_res.to_dict()
            self._update_checkpoint(
                task,
                stage="BASELINE_VERIFICATION",
                stage_details={"baseline_verification": baseline_evidence},
            )
            _persist_task(task)

            trace_emit(
                TraceEvent(
                    category="quality",
                    type="quality.baseline_evaluated",
                    severity="INFO" if not (task.is_defect_task() and baseline_gate_res.passed) else "WARNING",
                    task_id=task.task_id,
                    summary=f"Baseline verification executed: passed={baseline_gate_res.passed}",
                    payload={
                        "passed": baseline_gate_res.passed,
                        "is_defect": task.is_defect_task(),
                        "exit_code": baseline_gate_res.details.get("exit_code"),
                    },
                )
            )

            if task.is_defect_task():
                # Defect task requires baseline failure to reproduce defect
                is_valid, verif_msg = is_valid_defect_baseline(baseline_gate_res.details)
                if not is_valid:
                    reason = f"test doesn't prove the defect: {verif_msg}"
                    task.failure_reason = reason
                    if task.workspace:
                        task.workspace.lifecycle_state = "NEEDS_APPROVAL"

                    # Register a durable PermissionRequest in approvals.json so hold can be audited/approved
                    if self.approval_manager:
                        try:
                            req_id = f"req-{task.task_id}-baseline-review"
                            if req_id not in self.approval_manager._requests:
                                self.approval_manager.create_request(
                                    task_id=task.task_id,
                                    risk=task.risk.value if hasattr(task.risk, "value") else str(task.risk),
                                    operation="baseline_verification",
                                    target=", ".join(task.test_targets or task.target_files or []),
                                    scope=f"baseline defect reproduction review for {task.task_id}: {verif_msg}",
                                    reason=reason,
                                    request_id=req_id,
                                    lock_held=acquire_coarse_lock,
                                )
                        except Exception as req_err:
                            logger.warning("Failed to record baseline approval request for %s: %s", task.task_id, req_err)

                    _transition(
                        TaskStatus.NEEDS_APPROVAL,
                        stage="BASELINE_VERIFICATION",
                        details={"baseline_verification": baseline_evidence, "rejection_reason": verif_msg},
                        event_type="BASELINE_VERIFICATION_INCONCLUSIVE",
                        blocking_reason=reason,
                        idempotency_key=f"baseline-verif-invalid:{task.task_id}",
                    )
                    return _return_summary(ExecutionSummary(
                        task_id=task.task_id,
                        success=False,
                        final_status=TaskStatus.NEEDS_APPROVAL,
                        dry_run=False,
                        authorized=authorized,
                        duration_sec=time.monotonic() - start_time,
                        worktree_path=str(worktree_path),
                        message=f"Task '{task.task_id}' halted: {reason}. Human approval required.",
                    ))

            # 6. Build bounded prompt
            prompt = self.build_prompt(task, worktree_path)

            # 7. Agent Execution
            task.attempts += 1
            exec_details = {"attempt": task.attempts, "timeout": self.timeout_seconds}
            if routing_decision:
                exec_details["routing_decision"] = routing_decision.to_dict()
            _transition(
                TaskStatus.IN_PROGRESS,
                stage="AGENT_EXECUTION",
                details=exec_details,
                event_type="AGENT_EXECUTION_STARTED",
                idempotency_key=f"agent-start:{task.task_id}:{task.attempts}",
            )

            # Resolve execution adapter based on routing decision if router is present
            active_adapter = self.adapter
            if routing_decision and routing_decision.is_available:
                if routing_decision.selected_tier in (
                    RoutingTier.TIER_2_EFFICIENT_CLOUD,
                    RoutingTier.TIER_3_ADVANCED_CLOUD,
                ):
                    # Cloud budget enforcement (R-03 / F-03 & P4 Persistent Accounting)
                    # Synchronize in-memory config from persistent ledger budget if available
                    if hasattr(self.ledger, "budget") and self.router and hasattr(self.router, "config"):
                        self.router.config.cloud_tasks_consumed = max(self.router.config.cloud_tasks_consumed, self.ledger.budget.cloud_tasks_consumed)
                        self.router.config.max_cloud_tasks_per_session = self.ledger.budget.max_cloud_tasks

                    # Check ceiling before work begins
                    if not self.router.config.has_cloud_budget():
                        reason = (
                            f"Cloud session budget exhausted ({self.router.config.cloud_tasks_consumed}/"
                            f"{self.router.config.max_cloud_tasks_per_session} tasks consumed); "
                            "human approval required."
                        )
                        task.failure_reason = reason
                        if task.status != TaskStatus.NEEDS_APPROVAL:
                            _transition(
                                TaskStatus.NEEDS_APPROVAL,
                                stage="CLOUD_BUDGET_EXHAUSTED",
                                details={
                                    "cloud_tasks_consumed": self.router.config.cloud_tasks_consumed,
                                    "max_cloud_tasks_per_session": self.router.config.max_cloud_tasks_per_session,
                                },
                                event_type="CLOUD_BUDGET_EXHAUSTED",
                                blocking_reason=reason,
                                idempotency_key=f"cloud-budget-exhausted:{task.task_id}:{task.attempts}",
                            )
                        return _return_summary(ExecutionSummary(
                            task_id=task.task_id,
                            success=False,
                            final_status=TaskStatus.NEEDS_APPROVAL,
                            dry_run=False,
                            authorized=authorized,
                            duration_sec=time.monotonic() - start_time,
                            message=f"Task '{task.task_id}' blocked: {reason}",
                        ))

                    # Account for cloud attempt before cloud work begins (atomic persistent or in-memory)
                    try:
                        if hasattr(self, "coordinator_api") and self.coordinator_api is not None:
                            ok = self.coordinator_api.consume_cloud_budget(
                                self.ledger,
                                task.task_id,
                                count=1,
                                idempotency_key=f"cloud-budget:{task.task_id}:{task.attempts}",
                                lock_held=acquire_coarse_lock,
                            )
                            if not ok:
                                raise RuntimeError("Persistent cloud budget limit reached.")
                            if hasattr(self.ledger, "budget"):
                                self.router.config.cloud_tasks_consumed = self.ledger.budget.cloud_tasks_consumed
                        else:
                            self.router.config.consume_cloud_budget()
                    except Exception as e:
                        reason = f"Cloud budget accounting failed: {e}; blocking cloud execution."
                        task.failure_reason = reason
                        if task.status != TaskStatus.NEEDS_APPROVAL:
                            _transition(
                                TaskStatus.NEEDS_APPROVAL,
                                stage="CLOUD_BUDGET_ACCOUNTING_ERROR",
                                details={"error": str(e)},
                                event_type="CLOUD_BUDGET_ACCOUNTING_ERROR",
                                blocking_reason=reason,
                                idempotency_key=f"cloud-budget-error:{task.task_id}:{task.attempts}",
                            )
                        return _return_summary(ExecutionSummary(
                            task_id=task.task_id,
                            success=False,
                            final_status=TaskStatus.NEEDS_APPROVAL,
                            dry_run=False,
                            authorized=authorized,
                            duration_sec=time.monotonic() - start_time,
                            message=f"Task '{task.task_id}' blocked: {reason}",
                        ))

                    if isinstance(self.adapter, FakeAgentAdapter):
                        active_adapter = self.adapter
                    else:
                        agent_role = (
                            self.router.config.efficient_cloud_agent
                            if routing_decision.selected_tier == RoutingTier.TIER_2_EFFICIENT_CLOUD
                            else self.router.config.advanced_cloud_agent
                        )
                        active_adapter = AntigravityAgentAdapter(
                            authorized=authorized,
                            mode="accept-edits",
                            agent=agent_role,
                        )
                elif routing_decision.selected_tier == RoutingTier.TIER_1_LOCAL:
                    if isinstance(self.adapter, FakeAgentAdapter):
                        active_adapter = self.adapter
                    elif hasattr(self, "_local_adapter") and self._local_adapter:
                        active_adapter = self._local_adapter
                    else:
                        active_adapter = LocalModelAgentAdapter(model=self.router.config.local_model)

            worker_adapter_name = active_adapter.__class__.__name__
            worker_id = "W1"
            exec_req_id = f"exec-{task.task_id}-a{task.attempts}"

            trace_emit(
                TraceEvent(
                    category="worker",
                    type="worker.dispatched",
                    task_id=task.task_id,
                    worker_id=worker_id,
                    request_id=exec_req_id,
                    sender="orchestrator",
                    recipient=worker_adapter_name,
                    summary=f"Dispatched attempt {task.attempts} to {worker_adapter_name}",
                    payload={"adapter": worker_adapter_name, "timeout": self.timeout_seconds},
                )
            )

            agent_result: AgentExecutionResult = active_adapter.execute(
                task=task,
                prompt=prompt,
                worktree_path=worktree_path,
                timeout_seconds=self.timeout_seconds,
            )

            tok_reported = agent_result.input_tokens is not None or agent_result.output_tokens is not None
            trace_emit(
                TraceEvent(
                    category="worker",
                    type="worker.completed" if agent_result.success else "worker.failed",
                    severity="INFO" if agent_result.success else "ERROR",
                    task_id=task.task_id,
                    worker_id=worker_id,
                    request_id=exec_req_id,
                    sender=worker_adapter_name,
                    recipient="orchestrator",
                    summary=f"Worker {worker_adapter_name} attempt {task.attempts} {'succeeded' if agent_result.success else 'failed'}: exit={agent_result.exit_code}",
                    usage={
                        "source": "provider" if tok_reported else "unknown",
                        "input_tokens": agent_result.input_tokens,
                        "output_tokens": agent_result.output_tokens,
                    },
                    latency_ms=int(agent_result.duration_sec * 1000),
                    latency_source="measured",
                    payload={"exit_code": agent_result.exit_code, "error": agent_result.error_message or ""},
                )
            )

            # Record persistent token and cost usage into ledger budget (P6 WS2)
            if agent_result.input_tokens is not None or agent_result.output_tokens is not None:
                in_tok = agent_result.input_tokens or 0
                out_tok = agent_result.output_tokens or 0
                cost_usd = 0.0
                if in_tok > 0 or out_tok > 0:
                    model_id = getattr(active_adapter, "model", None)
                    if model_id:
                        try:
                            from .registry import ModelRegistry
                        except ImportError:
                            from registry import ModelRegistry
                        desc = ModelRegistry().get(model_id)
                        if desc:
                            cost_usd = (in_tok / 1000.0 * desc.cost_per_1k_input_tokens) + (out_tok / 1000.0 * desc.cost_per_1k_output_tokens)
                if hasattr(self, "coordinator_api") and self.coordinator_api is not None and (in_tok > 0 or out_tok > 0):
                    try:
                        self.coordinator_api.record_token_usage(
                            self.ledger,
                            task.task_id,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            cost_usd=cost_usd,
                            idempotency_key=f"token-usage:{task.task_id}:{task.attempts}",
                            lock_held=acquire_coarse_lock,
                        )
                    except Exception as tok_err:
                        logger.warning("Failed to record persistent token usage for %s: %s", task.task_id, tok_err)

            # Record routing fallback decision if execution failed
            fallback_decision: Optional[FallbackDecision] = None
            if self.router and routing_decision and not agent_result.success:
                fallback_decision = self.router.evaluate_fallback(
                    task, routing_decision, agent_result, authorized=authorized
                )
                _transition(
                    task.status,
                    stage="ADAPTER_FALLBACK",
                    details=fallback_decision.to_dict(),
                    event_type="ADAPTER_FALLBACK_EVALUATED",
                    idempotency_key=f"fallback-eval:{task.task_id}:{task.attempts}",
                )

            # Record execution evidence
            task.history.append(
                EvidenceRecord(
                    timestamp=utc_now_iso(),
                    stage="AGENT_EXECUTION",
                    details=(
                        f"Attempt {task.attempts}: exit={agent_result.exit_code}, "
                        f"duration={agent_result.duration_sec:.1f}s, "
                        f"error={agent_result.error_message or 'None'}"
                    ),
                    success=agent_result.success,
                )
            )

            # Handle transport availability failure, execution interruptions, or failures
            is_transport = getattr(agent_result, "transport_unavailable", False) or (
                agent_result.error_message and any(
                    agent_result.error_message.startswith(f"[{cat}]")
                    for cat in (
                        "transport_error",
                        "TRANSPORT_UNAVAILABLE",
                        "TRANSPORT_TIMEOUT",
                    )
                )
            )

            if is_transport:
                # Transport availability failure: do NOT consume task's repair attempt budget!
                task.attempts = max(0, task.attempts - 1)
                task.failure_reason = agent_result.error_message or "Transport unavailable."
                if task.workspace:
                    task.workspace.lifecycle_state = "INTERRUPTED"
                _transition(
                    TaskStatus.RECOVERY_REQUIRED,
                    stage="INTERRUPTED",
                    details=agent_result.to_dict(),
                    event_type="WORKER_TRANSPORT_INTERRUPTED",
                    failure_reason=task.failure_reason,
                    idempotency_key=f"transport-interrupted:{task.task_id}:{task.attempts}",
                )
                self._save_diagnostic_report(task, agent_result)

                return _return_summary(ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=False,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    worktree_path=str(worktree_path),
                    agent_result=agent_result.to_dict(),
                    message=f"Task '{task.task_id}' deferred: {task.failure_reason}. Repair budget preserved.",
                ))

            if agent_result.interrupted or agent_result.timed_out:
                task.failure_reason = agent_result.error_message or "Agent execution interrupted or timed out."
                if task.workspace:
                    task.workspace.lifecycle_state = "INTERRUPTED"
                _transition(
                    TaskStatus.RECOVERY_REQUIRED, stage="INTERRUPTED", details=agent_result.to_dict(),
                    event_type="WORKER_INTERRUPTED", failure_reason=task.failure_reason,
                    idempotency_key=f"worker-interrupted:{task.task_id}:{task.attempts}",
                )
                self._save_diagnostic_report(task, agent_result)

                return _return_summary(ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=False,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    worktree_path=str(worktree_path),
                    agent_result=agent_result.to_dict(),
                    message=f"Task '{task.task_id}' interrupted: {task.failure_reason}. Worktree preserved for inspection.",
                ))

            if not agent_result.success:
                # Check for non-qualifying failure (security violation, permission denied, sandbox violation, max attempts)
                is_non_qualifying = (
                    fallback_decision is not None
                    and (not fallback_decision.qualifies)
                    and fallback_decision.requires_human_approval
                ) or agent_result.permission_denied

                if is_non_qualifying:
                    non_qual_reason = (
                        fallback_decision.non_qualifying_reason
                        if (fallback_decision and fallback_decision.non_qualifying_reason)
                        else "SECURITY_OR_POLICY_VIOLATION"
                    )
                    task.failure_reason = (
                        agent_result.error_message
                        or f"Non-qualifying execution failure: {non_qual_reason}. Human authorization required."
                    )
                    if task.workspace:
                        task.workspace.lifecycle_state = "BLOCKED"
                    self._update_checkpoint(
                        task,
                        stage="NON_QUALIFYING_FAILURE",
                        stage_details={
                            "last_non_qualifying_reason": non_qual_reason,
                            "permission_denied": agent_result.permission_denied,
                            "error": agent_result.error_message,
                        },
                    )
                    _transition(
                        TaskStatus.NEEDS_APPROVAL,
                        stage="NON_QUALIFYING_FAILURE",
                        details={
                            "agent_result": agent_result.to_dict(),
                            "fallback_decision": fallback_decision.to_dict() if fallback_decision else None,
                            "non_qualifying_reason": non_qual_reason,
                        },
                        event_type="NON_QUALIFYING_FAILURE_BLOCKED",
                        failure_reason=task.failure_reason,
                        blocking_reason=f"Non-qualifying failure ({non_qual_reason}): {task.failure_reason}",
                        idempotency_key=f"non-qualifying-fail:{task.task_id}:{task.attempts}",
                    )
                    self._save_diagnostic_report(task, agent_result)

                    return _return_summary(ExecutionSummary(
                        task_id=task.task_id,
                        success=False,
                        final_status=task.status,
                        dry_run=False,
                        authorized=authorized,
                        duration_sec=time.monotonic() - start_time,
                        worktree_path=str(worktree_path),
                        agent_result=agent_result.to_dict(),
                        message=f"Task '{task.task_id}' halted on non-qualifying failure ({non_qual_reason}): {task.failure_reason}. Human approval required.",
                    ))

                task.failure_reason = agent_result.error_message or f"Agent process failed with exit code {agent_result.exit_code}."
                if task.workspace:
                    task.workspace.lifecycle_state = "FAILED"
                _transition(
                    TaskStatus.FAILED, stage="FAILED", details=agent_result.to_dict(),
                    event_type="AGENT_EXECUTION_FAILED", failure_reason=task.failure_reason,
                    idempotency_key=f"agent-failed:{task.task_id}:{task.attempts}",
                )
                self._save_diagnostic_report(task, agent_result)

                return _return_summary(ExecutionSummary(
                    task_id=task.task_id,
                    success=False,
                    final_status=task.status,
                    dry_run=False,
                    authorized=authorized,
                    duration_sec=time.monotonic() - start_time,
                    worktree_path=str(worktree_path),
                    agent_result=agent_result.to_dict(),
                    message=f"Task '{task.task_id}' failed in agent execution: {task.failure_reason}. Worktree preserved.",
                ))

            # 8. Verification & Quality Gating
            _transition(
                TaskStatus.VERIFYING, stage="VERIFICATION",
                details={"quality_engine": "QualityGateEngine"},
                event_type="VERIFICATION_STARTED",
                idempotency_key=f"verification-start:{task.task_id}:{task.attempts}",
            )

            quality_report: QualityReport = self.quality_engine.evaluate(task, worktree_path)

            trace_emit(
                TraceEvent(
                    category="quality",
                    type="quality.evaluated",
                    task_id=task.task_id,
                    severity="INFO" if quality_report.passed else "WARN",
                    sender="quality",
                    recipient="orchestrator",
                    summary=f"Quality gates: {'PASSED' if quality_report.passed else 'FAILED'} (failed: {[r.gate_name for r in quality_report.results if not r.passed]})",
                    payload={"passed": quality_report.passed, "results": [{"gate": r.gate_name, "passed": r.passed, "message": r.message} for r in quality_report.results]},
                )
            )

            task.history.append(
                EvidenceRecord(
                    timestamp=utc_now_iso(),
                    stage="QUALITY_GATES",
                    details=f"All passed: {quality_report.passed}. Gates: {[r.gate_name for r in quality_report.results if not r.passed]}",
                    success=quality_report.passed,
                )
            )

            review_result: Optional[ReviewResult] = None
            scan_report: Optional[ScanReport] = None
            safeguard_approved = False
            final_status = TaskStatus.FAILED
            final_failure_reason: str | None = None
            final_blocking_reason: str | None = None

            if quality_report.passed:
                # Capture candidate diff for deterministic safeguard scan
                diff_proc = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=str(worktree_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                diff_text = diff_proc.stdout
                diff_hash = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()

                # Fail-closed if diff extraction failed
                if diff_proc.returncode != 0:
                    scan_report = ScanReport(
                        passed=False,
                        diff_sha256=diff_hash,
                        requires_approval=True,
                        approval_reason=f"Failed to extract candidate diff (exit {diff_proc.returncode}): {diff_proc.stderr}",
                    )
                else:
                    try:
                        scan_report = self.safeguard_scanner.scan_diff(
                            task,
                            diff_text,
                            worktree_path=worktree_path,
                        )
                    except Exception as scan_err:
                        logger.exception("Pre-review safeguard scanner error on %s: %s", task.task_id, scan_err)
                        scan_report = ScanReport(
                            passed=False,
                            diff_sha256=diff_hash,
                            requires_approval=True,
                            approval_reason=f"Safeguard scanner internal error: {scan_err}",
                        )

                # Emit safeguard scan trace event
                trace_emit(
                    TraceEvent(
                        category="review",
                        type="reviewer.safeguards_scanned",
                        task_id=task.task_id,
                        severity="INFO" if scan_report.passed else "WARN",
                        sender="reviewer_safeguards",
                        recipient="orchestrator",
                        summary=f"Safeguards scan: {'PASSED' if scan_report.passed else 'HELD'} (findings={len(scan_report.findings)}, limits_exceeded={len(scan_report.limits_exceeded)})",
                        payload=scan_report.to_dict(),
                    )
                )

                # Evaluate pre-review safeguard outcome
                safeguard_approved = False
                if scan_report.requires_approval:
                    approval_scope = scan_report.get_approval_scope()
                    if self.approval_manager:
                        safeguard_approved = self.approval_manager.is_operation_approved(
                            task_id=task.task_id,
                            operation="reviewer_safeguards",
                            target=str(worktree_path),
                            scope=approval_scope,
                            patch_sha256=scan_report.diff_sha256,
                        )

                    if not safeguard_approved:
                        if self.approval_manager:
                            try:
                                self.approval_manager.create_request(
                                    task_id=task.task_id,
                                    risk=task.risk.value if hasattr(task.risk, "value") else str(task.risk),
                                    operation="reviewer_safeguards",
                                    target=str(worktree_path),
                                    scope=approval_scope,
                                    reason=scan_report.approval_reason or "Reviewer safeguards triggered.",
                                    patch_sha256=scan_report.diff_sha256,
                                    lock_held=acquire_coarse_lock,
                                )
                            except Exception as req_err:
                                logger.warning("Failed to record permission request for %s: %s", task.task_id, req_err)

                        task.history.append(
                            EvidenceRecord(
                                timestamp=utc_now_iso(),
                                stage="REVIEWER_SAFEGUARDS",
                                details=f"Held by pre-review safeguards: {scan_report.approval_reason}. Scope: {approval_scope}",
                                success=False,
                            )
                        )

                        final_status = TaskStatus.NEEDS_APPROVAL
                        final_blocking_reason = f"Pre-review safeguards held patch: {scan_report.approval_reason}"
                        if task.workspace:
                            task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                    else:
                        task.history.append(
                            EvidenceRecord(
                                timestamp=utc_now_iso(),
                                stage="REVIEWER_SAFEGUARDS",
                                details=f"Explicit operator approval confirmed for scope: {approval_scope}",
                                success=True,
                            )
                        )
                else:
                    task.history.append(
                        EvidenceRecord(
                            timestamp=utc_now_iso(),
                            stage="REVIEWER_SAFEGUARDS",
                            details=(
                                f"Passed pre-review safeguards: files={scan_report.total_files}, "
                                f"+{scan_report.total_added}/-{scan_report.total_deleted} lines."
                            ),
                            success=True,
                        )
                    )

                # Only proceed to reviewer invocation or downstream readiness if not held by safeguards
                if not scan_report.requires_approval or safeguard_approved:
                    should_rev, rev_reason = self.review_policy.should_review(task)
                    if should_rev:
                        def _perform_review() -> Optional[ReviewResult]:
                            # Check review budget
                            if not self.review_policy.has_budget_available():
                                nonlocal final_status, final_blocking_reason
                                final_status = TaskStatus.NEEDS_APPROVAL
                                final_blocking_reason = (
                                    f"Review required ({rev_reason}), but session review budget "
                                    f"({self.review_policy.max_reviews_per_session}) is exhausted."
                                )
                                if task.workspace:
                                    task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                                return None

                            # Check for cached review on identical diff
                            cache_key = (task.task_id, diff_hash)
                            cached_rev = self._review_cache.get(cache_key)

                            if cached_rev:
                                return cached_rev
                            else:
                                self.review_policy.consume_review()
                                verif_summary = "\n".join(r.message for r in quality_report.results)
                                res = self.reviewer.review(
                                    task=task,
                                    diff=diff_text,
                                    verification_summary=verif_summary,
                                    worktree_path=worktree_path,
                                    timeout_seconds=self.timeout_seconds,
                                )
                                self._review_cache[cache_key] = res
                                return res

                        if self.reviewer_lock:
                            with self.reviewer_lock:
                                review_result = _perform_review()
                        else:
                            review_result = _perform_review()

                        if review_result is not None:
                            # Record evidence
                            task.history.append(
                                EvidenceRecord(
                                    timestamp=utc_now_iso(),
                                    stage="INDEPENDENT_REVIEW",
                                    details=(
                                        f"Recommendation={review_result.recommendation.value}: "
                                        f"{review_result.summary} (Mutated={review_result.worktree_mutated})"
                                    ),
                                    success=(review_result.recommendation == ReviewRecommendation.APPROVE),
                                )
                            )

                            # Evaluate review outcome
                            if review_result.worktree_mutated:
                                final_status = TaskStatus.FAILED
                                final_failure_reason = f"CRITICAL: Reviewer mutated worktree files: {review_result.error_message}"
                                if task.workspace:
                                    task.workspace.lifecycle_state = "FAILED"
                            elif review_result.auth_failed or review_result.rate_limited or not review_result.success:
                                final_status = TaskStatus.NEEDS_APPROVAL
                                final_blocking_reason = f"Independent review halted: {review_result.error_message or review_result.summary}"
                                if task.workspace:
                                    task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                            elif review_result.recommendation == ReviewRecommendation.REQUEST_CHANGES:
                                final_status = TaskStatus.FAILED
                                final_failure_reason = f"Independent review requested changes: {review_result.summary}"
                                if task.workspace:
                                    task.workspace.lifecycle_state = "FAILED"
                            elif review_result.recommendation == ReviewRecommendation.NEEDS_HUMAN_APPROVAL:
                                final_status = TaskStatus.NEEDS_APPROVAL
                                final_blocking_reason = f"Independent review requires human approval: {review_result.summary}"
                                if task.workspace:
                                    task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                            elif review_result.recommendation == ReviewRecommendation.APPROVE:
                                if task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH):
                                    final_status = TaskStatus.NEEDS_APPROVAL
                                    final_blocking_reason = f"{task.risk.value}-risk task requires human operator sign-off before downstream progression."
                                    if task.workspace:
                                        task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                                else:
                                    final_status = TaskStatus.READY_TO_INTEGRATE
                    else:
                        # Review skipped by policy (low risk)
                        task.history.append(
                            EvidenceRecord(
                                timestamp=utc_now_iso(),
                                stage="INDEPENDENT_REVIEW",
                                details=f"Skipped by policy: {rev_reason}",
                                success=True,
                            )
                        )
                        if quality_report.should_elevate_approval:
                            final_status = TaskStatus.NEEDS_APPROVAL
                            final_blocking_reason = quality_report.approval_reason
                            if task.workspace:
                                task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                        else:
                            final_status = TaskStatus.READY_TO_INTEGRATE
            else:
                if quality_report.should_elevate_approval:
                    final_status = TaskStatus.NEEDS_APPROVAL
                    final_blocking_reason = quality_report.approval_reason
                    if task.workspace:
                        task.workspace.lifecycle_state = "NEEDS_APPROVAL"
                else:
                    final_status = TaskStatus.FAILED
                    final_failure_reason = quality_report.failure_reason or "Quality gate verification failed."
                    if task.workspace:
                        task.workspace.lifecycle_state = "FAILED"

                    # Safely reset only attempt-modified files to baseline commit
                    base_sha = task.workspace.base_commit if task.workspace else None
                    safe_restore_worktree_attempt(worktree_path, task, base_commit=base_sha, repo_root=self.repo_root)

                # Extract concise, bounded diagnostics from failing gates for the next attempt
                failing_gates_summary = []
                for r in quality_report.results:
                    if not r.passed:
                        item = f"Gate '{r.gate_name}': {r.message}"
                        if r.details:
                            if "stdout_tail" in r.details and r.details["stdout_tail"]:
                                tail_str = "\n".join(r.details["stdout_tail"][-15:])
                                item += f"\nTest Output Excerpt:\n{tail_str}"
                            elif "unauthorized_files" in r.details:
                                item += f"\nUnauthorized Files: {r.details['unauthorized_files']}"
                            elif "deleted_assertions" in r.details:
                                item += f"\nDeleted Assertions: {r.details['deleted_assertions']}"
                        failing_gates_summary.append(item)

                gate_diag = {
                    "attempt": task.attempts,
                    "failing_gates": [r.gate_name for r in quality_report.results if not r.passed],
                    "summary": "\n---\n".join(failing_gates_summary)[:2000],
                }

            stage_details = {
                "status": final_status.value,
                "quality_report": quality_report.to_dict(),
                "review_result": review_result.to_dict() if review_result else None,
                "agent_result": agent_result.to_dict(),
            }
            if quality_report.passed:
                baseline = task.workspace.base_commit if task.workspace else ""
                artifact_sha = compute_patch_sha256(worktree_path, baseline)
                verification_sha = compute_verification_sha256(
                    artifact_sha, baseline, quality_report, task.test_targets
                )
                stage_details.update({
                    "artifact_sha256": artifact_sha,
                    "baseline_commit": baseline,
                    "verification_sha256": verification_sha,
                })
            if not quality_report.passed:
                stage_details["last_quality_gate_failure"] = gate_diag

            if task.workspace:
                task.workspace.lifecycle_state = (
                    "READY_TO_INTEGRATE" if final_status == TaskStatus.READY_TO_INTEGRATE else final_status.value
                )
            if final_status == TaskStatus.READY_TO_INTEGRATE:
                task.failure_reason = None
            if scan_report and scan_report.requires_approval and not safeguard_approved:
                stage_val = "REVIEWER_SAFEGUARD_HELD"
                event_val = "REVIEWER_SAFEGUARD_BLOCKED"
            elif quality_report.passed:
                stage_val = "PRE_INTEGRATION_VERIFICATION"
                event_val = "PRE_INTEGRATION_VERIFICATION_PASSED"
            elif quality_report.should_elevate_approval:
                stage_val = "INCONCLUSIVE_VERIFICATION"
                event_val = "INCONCLUSIVE_VERIFICATION_ELEVATED"
            else:
                stage_val = "QUALITY_FAILED"
                event_val = "QUALITY_VERIFICATION_FAILED"

            _transition(
                final_status,
                stage=stage_val,
                details=stage_details,
                event_type=event_val,
                failure_reason=final_failure_reason,
                blocking_reason=final_blocking_reason,
                idempotency_key=f"candidate-result:{task.task_id}:{task.attempts}",
            )
            self.cache.save_cache()

            duration = time.monotonic() - start_time
            overall_success = task.status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED)

            if overall_success and quality_report.passed:
                try:
                    try:
                        from .repair_memory import RepairMemoryStore, RepairOutcome
                    except ImportError:
                        from repair_memory import RepairMemoryStore, RepairOutcome
                    mem_store = RepairMemoryStore(self.repo_root / ".maya" / "maintenance" / "repair_memory.jsonl")
                    outcome = RepairOutcome(
                        task_id=task.task_id,
                        target_files=list(task.target_files),
                        defect_category=task.audit_source or "maintenance",
                        model_id=str(getattr(active_adapter, "model", "local")),
                        patch_sha256=str(stage_details.get("artifact_sha256", "")),
                        verification_sha256=str(stage_details.get("verification_sha256", "")),
                        test_targets=list(task.test_targets),
                        verification_passed=True,
                        duration_sec=duration,
                        diff_summary=task.title or task.description[:200],
                    )
                    mem_store.record_outcome(outcome)
                except Exception as mem_err:
                    logger.debug("Could not record repair memory outcome: %s", mem_err)

            if not overall_success and not quality_report.passed:
                try:
                    try:
                        from .repair_memory import RepairMemoryStore, RepairOutcome
                    except ImportError:
                        from repair_memory import RepairMemoryStore, RepairOutcome
                    mem_store = RepairMemoryStore(self.repo_root / ".maya" / "maintenance" / "repair_memory.jsonl")
                    fail_summary = gate_diag.get("summary", "Quality gate verification failed")[:200] if "gate_diag" in locals() else "Quality verification failed"
                    outcome = RepairOutcome(
                        task_id=task.task_id,
                        target_files=list(task.target_files),
                        defect_category=task.audit_source or "maintenance",
                        model_id=str(getattr(active_adapter, "model", "local")),
                        patch_sha256="",
                        verification_sha256="",
                        test_targets=list(task.test_targets),
                        verification_passed=False,
                        status="failed",
                        duration_sec=duration,
                        diff_summary=fail_summary,
                    )
                    mem_store.record_outcome(outcome)
                except Exception as mem_err:
                    logger.debug("Could not record failed repair memory outcome: %s", mem_err)

            if not overall_success:
                self._save_diagnostic_report(task, agent_result, quality_report, review_result)

            return _return_summary(ExecutionSummary(
                task_id=task.task_id,
                success=overall_success,
                final_status=task.status,
                dry_run=False,
                authorized=authorized,
                duration_sec=duration,
                worktree_path=str(worktree_path),
                agent_result=agent_result.to_dict(),
                quality_report=quality_report.to_dict(),
                review_result=review_result.to_dict() if review_result else None,
                message=(
                    f"Task '{task.task_id}' candidate execution concluded with status {task.status.value}."
                    f" (Gates: {quality_report.passed}, Review: {review_result.recommendation.value if review_result else 'SKIPPED'})"
                ),
            ))
