"""
skills/maintenance/operator_controls.py
Interactive Operator Controls & Safe Authorization Controller for MayaVE Autophage (Phase 6).

Key invariants:
- Authoritative binding revalidation at execution time (rejects stale requests,
  changed diff hashes, baseline commit mismatches, and verification fingerprint mismatches).
- Enforces strict safety gates: RECOVERY_REQUIRED tasks remain blocked, reviewer safeguard
  holds cannot be bypassed, and unmet dependencies reject approval.
- Duplicate approval prevention and authorization downgrade prevention.
- Rejection preserves evidence with an auditable reason.
- Read-only pending merges and worktree cleanup preview inspection without git mutations.
- Reuses existing approval APIs and locks; never directly mutates JSON files.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from .models import TaskDefinition, TaskStatus
    from .store import CorruptStateError, StateStore
    from .terminal_approval import ApprovalStatus, PermissionRequest, TerminalApprovalManager
    from .worktree import WorktreeManager
except ImportError:
    from models import TaskDefinition, TaskStatus
    from store import CorruptStateError, StateStore
    from terminal_approval import ApprovalStatus, PermissionRequest, TerminalApprovalManager
    from worktree import WorktreeManager

logger = logging.getLogger("maya.maintenance.operator_controls")


def compute_diff_text_sha256(diff_text: str) -> str:
    """Deterministically hash unified diff text matching reviewer safeguards scanner."""
    return hashlib.sha256(diff_text.encode("utf-8")).hexdigest()


@dataclass
class ApprovalValidationResult:
    valid: bool
    error: Optional[str] = None
    request: Optional[PermissionRequest] = None
    task: Optional[TaskDefinition] = None
    binding_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryDiagnostic:
    task_id: str
    title: str
    stage: str
    reason: str
    worktree_exists: bool
    is_dirty: bool
    integration_branch_exists: bool
    blockers: list[str] = field(default_factory=list)
    next_safe_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "stage": self.stage,
            "reason": self.reason,
            "worktree_exists": self.worktree_exists,
            "is_dirty": self.is_dirty,
            "integration_branch_exists": self.integration_branch_exists,
            "blockers": self.blockers,
            "next_safe_action": self.next_safe_action,
        }


class OperatorController:
    """
    Coordinates operator inspections, safety-gated authorizations,
    rejections, recovery visibility, and pending-merge management.
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        store: Optional[StateStore] = None,
        approval_manager: Optional[TerminalApprovalManager] = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.maintenance_dir = self.repo_root / ".maya" / "maintenance"
        self.approvals_path = self.maintenance_dir / "approvals.json"
        self.state_path = self.maintenance_dir / "state.json"

        if store:
            self.store = store
        elif self.state_path.exists():
            self.store = StateStore(self.state_path)
        else:
            self.store = StateStore(self.state_path)

        if approval_manager:
            self.approval_manager = approval_manager
        else:
            self.approval_manager = TerminalApprovalManager(
                self.approvals_path,
                repo_root=self.repo_root,
                store=self.store,
            )

        self.wt_manager = WorktreeManager(self.repo_root)

    def get_pending_approvals(self) -> list[PermissionRequest]:
        """Load fresh pending permission requests."""
        self.approval_manager.load()
        return self.approval_manager.list_requests(status=ApprovalStatus.PENDING)

    def inspect_request(self, request_id: str) -> dict[str, Any]:
        """
        Inspect request details, exact binding fields, and worktree state
        without modifying or approving anything.
        """
        self.approval_manager.load()
        req = self.approval_manager.get_request(request_id)
        if not req:
            return {"error": f"Request '{request_id}' not found."}

        info: dict[str, Any] = {
            "request_id": req.request_id,
            "task_id": req.task_id,
            "operation": req.operation,
            "risk": req.risk,
            "target": req.target,
            "scope": req.scope,
            "reason": req.reason,
            "requested_at": req.requested_at,
            "status": req.status.value if isinstance(req.status, ApprovalStatus) else str(req.status),
            "patch_sha256": req.patch_sha256,
            "baseline_commit": req.baseline_commit,
            "verification_sha256": req.verification_sha256,
            "decision_at": req.decision_at,
            "decision_by": req.decision_by,
            "decision_scope": req.decision_scope,
            "worktree_exists": False,
            "worktree_path": "",
            "git_status": "",
            "diff_summary": "",
            "bindings_valid": False,
            "validation_message": "",
        }

        # Check worktree details
        wt_path = self.repo_root / ".maya" / "worktrees" / req.task_id
        if wt_path.exists():
            info["worktree_exists"] = True
            info["worktree_path"] = str(wt_path)
            try:
                st_proc = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=str(wt_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                info["git_status"] = st_proc.stdout.strip()
            except Exception as e:
                info["git_status"] = f"Error querying status: {e}"

            try:
                diff_proc = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=str(wt_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if diff_proc.returncode == 0:
                    diff_text = diff_proc.stdout
                    curr_sha = compute_diff_text_sha256(diff_text)
                    lines = diff_text.splitlines()
                    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
                    deleted = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
                    info["diff_summary"] = f"+{added} / -{deleted} lines (current_diff_sha256={curr_sha[:10]}...)"
            except Exception:
                pass

        # Revalidate bindings without mutating
        val = self.validate_approval(request_id)
        info["bindings_valid"] = val.valid
        info["validation_message"] = val.error or "All binding requirements satisfied."
        if val.task:
            info["task_status"] = val.task.status.value
            info["task_stage"] = val.task.checkpoint.stage if val.task.checkpoint else "UNKNOWN"

        return info

    def validate_approval(self, request_id: str) -> ApprovalValidationResult:
        """
        Authoritatively validate a pending approval request against live state,
        exact cryptographic bindings, recovery rules, reviewer safeguards, and dependencies.
        Never mutates state.
        """
        self.approval_manager.load()
        req = self.approval_manager.get_request(request_id)
        if not req:
            return ApprovalValidationResult(valid=False, error=f"Request '{request_id}' not found.")

        # Duplicate approval prevention
        if req.status == ApprovalStatus.APPROVED:
            return ApprovalValidationResult(
                valid=False,
                error=f"Request '{request_id}' is already approved. Duplicate approval prohibited.",
                request=req,
            )

        if req.status != ApprovalStatus.PENDING:
            return ApprovalValidationResult(
                valid=False,
                error=f"Request '{request_id}' is not in PENDING status (current: {req.status.value}).",
                request=req,
            )

        # Check state store existence
        if not self.store or not self.store.exists():
            return ApprovalValidationResult(
                valid=False,
                error="State file does not exist. Cannot validate approval against state ledger.",
                request=req,
            )

        try:
            ledger = self.store.load_ledger()
        except CorruptStateError as e:
            return ApprovalValidationResult(
                valid=False,
                error=f"State ledger is corrupt: {e}",
                request=req,
            )

        if req.task_id not in ledger:
            return ApprovalValidationResult(
                valid=False,
                error=f"Task '{req.task_id}' not found in state ledger.",
                request=req,
            )

        task = ledger.get_task(req.task_id)

        # RECOVERY_REQUIRED restriction: approval cannot bypass recovery
        if task.status == TaskStatus.RECOVERY_REQUIRED:
            return ApprovalValidationResult(
                valid=False,
                error=(
                    f"Task '{req.task_id}' is in RECOVERY_REQUIRED status. "
                    "Approval cannot bypass recovery restrictions; manual inspection and reconciliation required."
                ),
                request=req,
                task=task,
            )

        # Dependency check: all upstream dependencies must be VERIFIED
        deps = getattr(task, "dependencies", []) or getattr(task, "depends_on", [])
        if deps:
            for dep_id in deps:
                if dep_id in ledger:
                    dep_task = ledger.get_task(dep_id)
                    if dep_task.status != TaskStatus.VERIFIED:
                        return ApprovalValidationResult(
                            valid=False,
                            error=(
                                f"Task '{req.task_id}' has unmet dependency '{dep_id}' "
                                f"(status={dep_task.status.value}). Approval cannot bypass dependency requirements."
                            ),
                            request=req,
                            task=task,
                        )

        # Operation-specific exact binding validation
        binding_details: dict[str, Any] = {}

        if req.operation == "integration":
            if task.status != TaskStatus.READY_TO_INTEGRATE:
                return ApprovalValidationResult(
                    valid=False,
                    error=f"Task '{req.task_id}' status is {task.status.value}, expected READY_TO_INTEGRATE for integration approval.",
                    request=req,
                    task=task,
                )

            cp = task.checkpoint.details if task.checkpoint else {}
            art_sha = cp.get("artifact_sha256")
            base_com = cp.get("baseline_commit")
            ver_sha = cp.get("verification_sha256")

            binding_details = {
                "checkpoint_artifact_sha256": art_sha,
                "request_patch_sha256": req.patch_sha256,
                "checkpoint_baseline_commit": base_com,
                "request_baseline_commit": req.baseline_commit,
                "checkpoint_verification_sha256": ver_sha,
                "request_verification_sha256": req.verification_sha256,
            }

            if req.patch_sha256 and art_sha != req.patch_sha256:
                return ApprovalValidationResult(
                    valid=False,
                    error=(
                        f"Patch hash mismatch: request has '{req.patch_sha256[:10]}...', "
                        f"checkpoint has '{art_sha[:10] if art_sha else 'None'}...'."
                    ),
                    request=req,
                    task=task,
                    binding_details=binding_details,
                )

            if req.baseline_commit and base_com != req.baseline_commit:
                return ApprovalValidationResult(
                    valid=False,
                    error=(
                        f"Baseline commit mismatch: request has '{req.baseline_commit[:10]}...', "
                        f"checkpoint has '{base_com[:10] if base_com else 'None'}...'."
                    ),
                    request=req,
                    task=task,
                    binding_details=binding_details,
                )

            if req.verification_sha256 and ver_sha != req.verification_sha256:
                return ApprovalValidationResult(
                    valid=False,
                    error=(
                        f"Verification fingerprint mismatch: request has '{req.verification_sha256[:10]}...', "
                        f"checkpoint has '{ver_sha[:10] if ver_sha else 'None'}...'."
                    ),
                    request=req,
                    task=task,
                    binding_details=binding_details,
                )

        elif req.operation == "reviewer_safeguards":
            if task.status != TaskStatus.NEEDS_APPROVAL:
                return ApprovalValidationResult(
                    valid=False,
                    error=f"Task '{req.task_id}' status is {task.status.value}, expected NEEDS_APPROVAL.",
                    request=req,
                    task=task,
                )

            wt_path = self.repo_root / ".maya" / "worktrees" / req.task_id
            if not wt_path.exists():
                return ApprovalValidationResult(
                    valid=False,
                    error=f"Worktree for task '{req.task_id}' not found at {wt_path}.",
                    request=req,
                    task=task,
                )

            # Revalidate candidate diff hash in worktree
            try:
                diff_proc = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=str(wt_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if diff_proc.returncode == 0:
                    curr_patch_sha = compute_diff_text_sha256(diff_proc.stdout)
                    binding_details["current_patch_sha256"] = curr_patch_sha
                    binding_details["request_patch_sha256"] = req.patch_sha256
                    if req.patch_sha256 and curr_patch_sha != req.patch_sha256:
                        return ApprovalValidationResult(
                            valid=False,
                            error=(
                                f"Changed diff hash: candidate patch in worktree has changed "
                                f"(request bound to '{req.patch_sha256[:10]}...', current is '{curr_patch_sha[:10]}...'). "
                                "Prior approval invalidated."
                            ),
                            request=req,
                            task=task,
                            binding_details=binding_details,
                        )
            except Exception as e:
                return ApprovalValidationResult(
                    valid=False,
                    error=f"Failed to inspect worktree candidate diff: {e}",
                    request=req,
                    task=task,
                )

        elif req.operation in ("workspace_adoption", "pre_validation"):
            if task.status not in (TaskStatus.NEEDS_APPROVAL, TaskStatus.PENDING):
                return ApprovalValidationResult(
                    valid=False,
                    error=f"Task '{req.task_id}' status is {task.status.value}, expected NEEDS_APPROVAL or PENDING.",
                    request=req,
                    task=task,
                )

        return ApprovalValidationResult(
            valid=True,
            error=None,
            request=req,
            task=task,
            binding_details=binding_details,
        )

    def execute_approval(
        self,
        request_id: str,
        reason: Optional[str] = None,
        decision_by: str = "operator",
    ) -> tuple[bool, str]:
        """
        Revalidate exact bindings and execute approval via TerminalApprovalManager.
        Synchronizes to store atomically under lock and records coordinator lifecycle event.
        """
        val = self.validate_approval(request_id)
        if not val.valid:
            logger.warning("Approval execution rejected for %s: %s", request_id, val.error)
            return False, val.error or "Validation failed."

        req = val.request
        assert req is not None

        try:
            decision_reason = reason or "Authorized by operator via dashboard."
            self.approval_manager.approve(
                request_id,
                scope=req.scope,
                reason=decision_reason,
                decision_by=decision_by,
                sync_store=True,
            )
            logger.info("Executed approval for request %s (task=%s)", request_id, req.task_id)
            return True, f"Successfully approved request '{request_id}' for task '{req.task_id}'."
        except Exception as exc:
            logger.error("Failed to execute approval for %s: %s", request_id, exc)
            return False, f"Failed to execute approval: {exc}"

    def execute_rejection(
        self,
        request_id: str,
        reason: Optional[str] = None,
        decision_by: str = "operator",
    ) -> tuple[bool, str]:
        """
        Reject a pending request with an auditable reason.
        Preserves evidence in approvals.json and records lifecycle denial event.
        Prevents downgrading an already APPROVED request.
        """
        self.approval_manager.load()
        req = self.approval_manager.get_request(request_id)
        if not req:
            return False, f"Request '{request_id}' not found."

        if req.status == ApprovalStatus.APPROVED:
            return False, f"Request '{request_id}' is already approved. Cannot downgrade an approved request."

        if req.status != ApprovalStatus.PENDING:
            return False, f"Request '{request_id}' is already {req.status.value}; cannot reject."

        audit_reason = (reason or "").strip()
        if not audit_reason:
            audit_reason = "Rejected by operator via dashboard."

        try:
            self.approval_manager.deny(
                request_id,
                reason=audit_reason,
                decision_by=decision_by,
            )
            logger.info("Executed rejection for request %s (task=%s, reason=%s)", request_id, req.task_id, audit_reason)
            return True, f"Successfully rejected request '{request_id}' for task '{req.task_id}': {audit_reason}"
        except Exception as exc:
            logger.error("Failed to execute rejection for %s: %s", request_id, exc)
            return False, f"Failed to execute rejection: {exc}"

    def get_recovery_diagnostics(self) -> list[RecoveryDiagnostic]:
        """
        Inspect all RECOVERY_REQUIRED tasks, actionable blockers, and safe next manual steps.
        Does not bypass or alter recovery status.
        """
        if not self.store or not self.store.exists():
            return []

        try:
            ledger = self.store.load_ledger()
        except Exception:
            return []

        diagnostics: list[RecoveryDiagnostic] = []
        for task in ledger:
            if task.status != TaskStatus.RECOVERY_REQUIRED:
                continue

            wt_path = self.repo_root / ".maya" / "worktrees" / task.task_id
            wt_exists = wt_path.exists()
            is_dirty = False
            if wt_exists:
                try:
                    st_proc = subprocess.run(
                        ["git", "status", "--short"],
                        cwd=str(wt_path),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if st_proc.returncode == 0 and st_proc.stdout.strip():
                        is_dirty = True
                except Exception:
                    pass

            int_branch = f"autophage/integrate-{task.task_id}"
            br_proc = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/heads/{int_branch}"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            has_int_branch = (br_proc.returncode == 0)

            stage = task.checkpoint.stage if task.checkpoint else "UNKNOWN"
            reason = task.blocking_reason or "Interrupted during in-flight lifecycle state"

            blockers: list[str] = []
            if is_dirty:
                blockers.append(f"Worktree .maya/worktrees/{task.task_id} has uncommitted modifications.")
            if has_int_branch and stage == "INTERRUPTED_INTEGRATION_DETECTED":
                blockers.append(f"Unverified integration commit on branch '{int_branch}' ahead of baseline.")
            if not blockers:
                blockers.append(f"Incomplete lifecycle stage: {stage} ({reason})")

            # Guidance for next safe manual action (no automated bypass)
            if has_int_branch and stage == "INTERRUPTED_INTEGRATION_DETECTED":
                next_action = (
                    f"Inspect git diff on '{int_branch}' against baseline. "
                    "Run 'python skills/maintenance/coordinator.py --reconcile' to verify."
                )
            elif is_dirty:
                next_action = (
                    f"Inspect uncommitted edits in '.maya/worktrees/{task.task_id}'. "
                    "Commit or reset manually, then run 'python skills/maintenance/coordinator.py --reconcile'."
                )
            else:
                next_action = (
                    "Run 'python skills/maintenance/coordinator.py --reconcile' to synchronize worktrees and ledger safely."
                )

            diagnostics.append(
                RecoveryDiagnostic(
                    task_id=task.task_id,
                    title=task.title,
                    stage=stage,
                    reason=reason,
                    worktree_exists=wt_exists,
                    is_dirty=is_dirty,
                    integration_branch_exists=has_int_branch,
                    blockers=blockers,
                    next_safe_action=next_action,
                )
            )

        return diagnostics

    def get_pending_merges_data(self, target_branch: str = "main") -> dict[str, Any]:
        """
        Read-only visibility into pending integration branches and worktree cleanup preview.
        Preserves human-controlled integration into main; never merges, rebases, or deletes branches.
        Labels ambiguous or unavailable evidence explicitly.
        """
        if not self.store or not self.store.exists():
            return {"merges": [], "cleanup": [], "discrepancies": []}

        try:
            ledger = self.store.load_ledger()
        except Exception:
            return {"merges": [], "cleanup": [], "discrepancies": []}

        br_proc = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        all_local_branches = set(br_proc.stdout.splitlines()) if br_proc.returncode == 0 else set()
        int_branches = {b for b in all_local_branches if b.startswith("autophage/")}

        ledger_tasks_by_branch: dict[str, list[TaskDefinition]] = {}
        for task in ledger:
            if task.workspace and task.workspace.branch_name:
                ledger_tasks_by_branch.setdefault(task.workspace.branch_name, []).append(task)
            int_b = f"autophage/integrate-{task.task_id}"
            ledger_tasks_by_branch.setdefault(int_b, []).append(task)
            int_branches.add(int_b)

        merges = []
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
            blockers = []

            if associated_tasks:
                t = associated_tasks[0]
                task_id_str = t.task_id
                task_info = t.title[:35]
                if t.workspace and t.workspace.base_commit:
                    base_commit = t.workspace.base_commit[:7]
                ws_status = self.wt_manager.inspect_workspace(t.task_id)
                if ws_status.exists_on_disk:
                    wt_state = "DIRTY" if ws_status.is_dirty else "CLEAN"
                else:
                    wt_state = "MISSING"
                if ws_status.is_dirty:
                    blockers.append("Worktree dirty")
            else:
                discrepancies.append(
                    f"Branch '{branch_name}' exists in Git but is not associated with any task in the ledger."
                )
                blockers.append("Unlinked branch")

            head_proc = subprocess.run(
                ["git", "rev-parse", f"refs/heads/{branch_name}"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if head_proc.returncode == 0:
                head_commit = head_proc.stdout.strip()[:7]
                if base_commit != "-":
                    cnt_proc = subprocess.run(
                        ["git", "rev-list", "--count", f"{base_commit}..{branch_name}"],
                        cwd=str(self.repo_root),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if cnt_proc.returncode == 0:
                        try:
                            commits_ahead = int(cnt_proc.stdout.strip())
                        except ValueError:
                            commits_ahead = 0
                    else:
                        commits_ahead = -1  # Ambiguous/unknown
            else:
                head_commit = "UNKNOWN"

            is_merged = self.wt_manager.is_branch_merged(branch_name, target_branch)
            merge_status = "MERGED" if is_merged else "PENDING"
            if not is_merged:
                blockers.append(f"Awaiting human merge into '{target_branch}'")

            merges.append({
                "branch": branch_name,
                "task_id": task_id_str,
                "title": task_info,
                "base": base_commit,
                "head": head_commit,
                "ahead": commits_ahead if commits_ahead >= 0 else "UNKNOWN",
                "merge_status": merge_status,
                "worktree": wt_state,
                "blockers": blockers,
            })

        try:
            cleanup_items = self.wt_manager.get_cleanup_preview(ledger, target_branch=target_branch)
        except Exception as e:
            logger.warning("Could not query cleanup preview: %s", e)
            cleanup_items = []
        cleanup = [
            {
                "task_id": item.task_id or "[ORPHAN]",
                "path": item.path,
                "branch_name": item.branch_name,
                "task_status": item.task_status,
                "is_dirty": item.is_dirty,
                "is_merged": item.is_merged,
                "eligible": item.eligible,
                "recommendation": item.recommendation,
                "reason": item.reason,
                "blocking_reasons": item.blocking_reasons,
            }
            for item in cleanup_items
        ]

        return {
            "merges": merges,
            "cleanup": cleanup,
            "discrepancies": discrepancies,
        }
