"""
Isolated Git Worktree Manager for MayaVE Maintenance Coordinator.
Enforces process isolation, strict ownership validation, and non-destructive recovery.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from .ledger import TaskLedger
    from .hub import CoordinatorAPI
    from .models import CheckpointRecord, TaskDefinition, TaskStatus, WorkspaceMetadata, utc_now_iso
    from .store import StateStore
except ImportError:
    from ledger import TaskLedger
    from hub import CoordinatorAPI
    from models import CheckpointRecord, TaskDefinition, TaskStatus, WorkspaceMetadata, utc_now_iso
    from store import StateStore

try:
    from .terminal_approval import ApprovalStatus, PermissionRequest, TerminalApprovalManager
except ImportError:
    try:
        from terminal_approval import ApprovalStatus, PermissionRequest, TerminalApprovalManager
    except ImportError:
        ApprovalStatus = None
        PermissionRequest = None
        TerminalApprovalManager = None

logger = logging.getLogger("maya.maintenance.worktree")


class WorkspaceError(Exception):
    """Base error for workspace operations."""
    pass


class WorkspaceConflictError(WorkspaceError):
    """Raised when an unexpected directory, worktree, or branch conflict is detected."""
    pass


class WorkspaceValidationError(WorkspaceError):
    """Raised when workspace preconditions are violated."""
    pass


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass
class WorkspaceStatus:
    task_id: str
    path: Path
    exists_on_disk: bool
    registered_in_git: bool
    branch_name: str
    head_commit: str
    is_dirty: bool
    modified_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    ownership_status: str = "MANAGED"  # MANAGED, ORPHAN, CONFLICT, MISSING


@dataclass
class CleanupPreviewItem:
    task_id: Optional[str]
    path: str
    branch_name: str
    is_dirty: bool
    task_status: str
    recommendation: str  # MANUAL_REVIEW_REQUIRED, PRESERVE_FOR_INSPECTION, SAFE_TO_CLEAN
    reason: str
    is_merged: bool = False
    eligible: bool = False
    blocking_reasons: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class ReconciliationItem:
    task_id: str
    previous_status: TaskStatus
    new_status: TaskStatus
    action_taken: str
    reason: str


class WorktreeManager:
    """Manages isolated Git worktrees under .maya/worktrees/."""

    def __init__(
        self,
        repo_root: Path | str,
        worktrees_dir: Path | str | None = None,
        approvals_path: Path | str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        if worktrees_dir:
            self.worktrees_dir = Path(worktrees_dir).resolve()
        else:
            self.worktrees_dir = (self.repo_root / ".maya" / "worktrees").resolve()
        if approvals_path:
            self.approvals_path = Path(approvals_path).resolve()
        else:
            self.approvals_path = (self.repo_root / ".maya" / "maintenance" / "approvals.json").resolve()

    def get_workspace_path(self, task_id: str) -> Path:
        """Resolve and validate the destination directory for a task workspace."""
        if not task_id or not _TASK_ID_RE.match(task_id):
            raise WorkspaceValidationError(f"Invalid task_id '{task_id}'. Cannot resolve workspace path.")
        path = (self.worktrees_dir / task_id).resolve()
        # Ensure no path traversal escaping worktrees_dir
        try:
            path.relative_to(self.worktrees_dir)
        except ValueError:
            raise WorkspaceValidationError(f"Path traversal detected for task_id '{task_id}'.")
        return path

    def get_branch_name(self, task_id: str) -> str:
        """Deterministic branch name per task."""
        if not task_id or not _TASK_ID_RE.match(task_id):
            raise WorkspaceValidationError(f"Invalid task_id '{task_id}'. Cannot construct branch name.")
        return f"maintenance/{task_id}"

    def list_git_worktrees(self) -> dict[Path, dict]:
        """
        Parse `git worktree list --porcelain` into a mapping of normalized paths to worktree info.
        """
        proc = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise WorkspaceError(f"Failed to query git worktrees: {proc.stderr}")

        worktrees: dict[Path, dict] = {}
        current: dict = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                if current and "worktree" in current:
                    wt_path = Path(current["worktree"]).resolve()
                    worktrees[wt_path] = current
                current = {}
                continue

            parts = line.split(maxsplit=1)
            key = parts[0]
            val = parts[1] if len(parts) > 1 else ""

            if key == "worktree":
                current["worktree"] = val
            elif key == "HEAD":
                current["HEAD"] = val
            elif key == "branch":
                # refs/heads/branch_name -> branch_name
                ref = val
                if ref.startswith("refs/heads/"):
                    current["branch"] = ref[len("refs/heads/") :]
                else:
                    current["branch"] = ref
            elif key == "bare":
                current["bare"] = True
            elif key == "detached":
                current["detached"] = True

        if current and "worktree" in current:
            wt_path = Path(current["worktree"]).resolve()
            worktrees[wt_path] = current

        return worktrees

    def branch_exists(self, branch_name: str) -> bool:
        """Check if local branch exists in the repository."""
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/heads/{branch_name}"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0

    def is_branch_merged(self, branch_name: str, target_branch: str = "main") -> bool:
        """Check if branch_name is merged into target_branch (branch is an ancestor of target)."""
        if not self.branch_exists(branch_name) or not self.branch_exists(target_branch):
            return False
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", branch_name, target_branch],
            cwd=str(self.repo_root),
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0

    def has_pending_approvals(self, task_id: str) -> bool:
        """Check if approvals.json contains any pending permission request for task_id."""
        if not self.approvals_path or not self.approvals_path.is_file():
            return False
        try:
            with open(self.approvals_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            requests = data.get("requests", {})
            for r in requests.values():
                if isinstance(r, dict) and r.get("task_id") == task_id and r.get("status") == "PENDING":
                    return True
        except Exception:
            pass
        return False

    def inspect_workspace(self, task_id: str) -> WorkspaceStatus:
        """Inspect the current on-disk and Git status of a task workspace."""
        path = self.get_workspace_path(task_id)
        branch = self.get_branch_name(task_id)
        exists_disk = path.is_dir()

        all_worktrees = self.list_git_worktrees()
        registered = path in all_worktrees

        head_commit = ""
        is_dirty = False
        modified: list[str] = []
        untracked: list[str] = []
        ownership = "MISSING"

        if registered:
            info = all_worktrees[path]
            head_commit = info.get("HEAD", "")
            actual_branch = info.get("branch", "")
            if actual_branch == branch:
                ownership = "MANAGED"
            else:
                ownership = "CONFLICT"

            # Check status inside the worktree
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(path),
                capture_output=True,
                text=True,
                check=False,
            )
            if status_proc.returncode == 0:
                for line in status_proc.stdout.splitlines():
                    if not line.strip():
                        continue
                    code = line[:2]
                    fname = line[2:].strip()
                    if code == "??":
                        untracked.append(fname)
                    else:
                        modified.append(fname)
                is_dirty = len(modified) > 0 or len(untracked) > 0
        elif exists_disk:
            ownership = "ORPHAN"

        return WorkspaceStatus(
            task_id=task_id,
            path=path,
            exists_on_disk=exists_disk,
            registered_in_git=registered,
            branch_name=branch,
            head_commit=head_commit,
            is_dirty=is_dirty,
            modified_files=modified,
            untracked_files=untracked,
            ownership_status=ownership,
        )

    def prepare_workspace(
        self,
        task_id: str,
        base_commit: str | None = None,
    ) -> WorkspaceMetadata:
        """
        Safely create and register an isolated Git worktree for the task.
        Guarantees:
        - Never overwrites an existing directory or worktree.
        - Detects conflicting branches.
        - Returns existing valid workspace metadata if already prepared.
        """
        path = self.get_workspace_path(task_id)
        branch_name = self.get_branch_name(task_id)

        # 1. Resolve base commit
        if not base_commit:
            sha_proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if sha_proc.returncode != 0:
                raise WorkspaceError(f"Could not determine baseline HEAD commit: {sha_proc.stderr}")
            base_commit = sha_proc.stdout.strip()

        # 2. Check if already prepared and registered
        current_status = self.inspect_workspace(task_id)
        if current_status.registered_in_git:
            if current_status.ownership_status == "MANAGED":
                self.sync_workspace_resources(task_id)
                return WorkspaceMetadata(
                    worktree_path=str(path),
                    branch_name=branch_name,
                    base_commit=base_commit,
                    head_commit=current_status.head_commit,
                    created_at=utc_now_iso(),
                    is_dirty=current_status.is_dirty,
                    lifecycle_state="ACTIVE",
                )
            else:
                raise WorkspaceConflictError(
                    f"Worktree directory '{path}' is registered to conflicting branch '{current_status.branch_name}'."
                )

        if current_status.exists_on_disk and not current_status.registered_in_git:
            raise WorkspaceConflictError(
                f"Directory '{path}' already exists on disk but is not a registered Git worktree. "
                "Preserving directory; manual inspection required."
            )

        # 3. Handle branch existence
        branch_exists = self.branch_exists(branch_name)

        # 4. Create worktree
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

        if branch_exists:
            cmd = ["git", "worktree", "add", str(path), branch_name]
        else:
            cmd = ["git", "worktree", "add", "-b", branch_name, str(path), base_commit]

        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise WorkspaceError(
                f"Failed to create Git worktree for '{task_id}': {proc.stderr.strip()}"
            )

        # Mirror local tests and custom agent definitions into worktree
        self.sync_workspace_resources(task_id)

        # 5. Query created worktree HEAD commit
        created_status = self.inspect_workspace(task_id)

        return WorkspaceMetadata(
            worktree_path=str(path),
            branch_name=branch_name,
            base_commit=base_commit,
            head_commit=created_status.head_commit,
            created_at=utc_now_iso(),
            is_dirty=created_status.is_dirty,
            lifecycle_state="PREPARED",
        )

    def sync_local_tests(self, task_id: str) -> bool:
        """
        Safely mirror untracked .maya/tests/ from repo_root to the task worktree.
        Preserves local-only test isolation while ensuring tests are present for execution.
        """
        path = self.get_workspace_path(task_id)
        src_tests = self.repo_root / ".maya" / "tests"
        dst_tests = path / ".maya" / "tests"

        if not src_tests.is_dir() or not path.is_dir():
            return False

        dst_tests.mkdir(parents=True, exist_ok=True)
        # Mirror test files, excluding pytest cache and pycache
        for item in src_tests.rglob("*"):
            if item.is_dir():
                continue
            if "__pycache__" in item.parts or ".pytest_cache" in item.parts:
                continue
            rel_part = item.relative_to(src_tests)
            dest_file = dst_tests / rel_part
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if not dest_file.exists() or dest_file.stat().st_mtime < item.stat().st_mtime:
                try:
                    shutil.copy2(item, dest_file)
                except Exception:
                    pass
        return True

    def sync_agents(self, task_id: str) -> bool:
        """
        Safely mirror untracked root .agents/ to the task worktree.
        Preserves root .agents/ contents exactly, including nested agent definitions and configuration files.
        Ensures custom agents (such as 'implementer', 'architect', 'reviewer') are available
        for Antigravity CLI discovery inside isolated task worktrees.

        Returns:
            True if .agents/ was successfully synchronized.
            False if the root .agents/ directory or the target worktree does not exist.

        Raises:
            WorkspaceError: If a filesystem error occurs during directory creation or file copy.
        """
        path = self.get_workspace_path(task_id)
        src_agents = self.repo_root / ".agents"
        dst_agents = path / ".agents"

        if not src_agents.is_dir() or not path.is_dir():
            return False

        try:
            dst_agents.mkdir(parents=True, exist_ok=True)
            for item in src_agents.rglob("*"):
                if item.is_dir():
                    continue
                if "__pycache__" in item.parts or ".pytest_cache" in item.parts:
                    continue
                rel_part = item.relative_to(src_agents)
                dest_file = dst_agents / rel_part
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                if not dest_file.exists() or dest_file.stat().st_mtime < item.stat().st_mtime:
                    shutil.copy2(item, dest_file)
            return True
        except OSError as e:
            raise WorkspaceError(f"Failed to synchronize .agents/ to worktree '{path}': {e}") from e

    def sync_workspace_resources(self, task_id: str) -> dict[str, bool]:
        """
        Synchronize untracked local resources (.maya/tests/ and .agents/)
        from the repository root into the specified task worktree.
        """
        tests_synced = self.sync_local_tests(task_id)
        agents_synced = self.sync_agents(task_id)
        return {
            "tests": tests_synced,
            "agents": agents_synced,
        }

    def _is_workspace_adoption_authorized(
        self,
        task: TaskDefinition,
        ws_status: WorkspaceStatus,
    ) -> bool:
        """
        Validate whether a pre-existing workspace for a PENDING task has been
        explicitly and legitimately authorized via a verified workspace_adoption approval.
        Fails closed on any inconsistency, missing record, or unapproved status.
        """
        try:
            # 1. Ownership status must be strictly MANAGED
            if ws_status.ownership_status != "MANAGED":
                return False

            # 2. Registered in Git
            if not ws_status.registered_in_git:
                return False

            # 3. Checkpoint must exist, be structurally valid, and record stage APPROVAL_GRANTED
            if not task.checkpoint or not isinstance(task.checkpoint, CheckpointRecord):
                return False
            if task.checkpoint.stage != "APPROVAL_GRANTED":
                return False
            if not isinstance(task.checkpoint.details, dict):
                return False

            # 4. Checkpoint operation must be workspace_adoption or reviewer_safeguards
            if task.checkpoint.details.get("operation") not in ("workspace_adoption", "reviewer_safeguards"):
                return False

            # 5. Checkpoint target must resolve to the same canonical path as ws_status.path
            cp_target = task.checkpoint.details.get("target")
            if not cp_target or not isinstance(cp_target, str):
                return False

            resolved_target = Path(cp_target).resolve()
            if resolved_target != ws_status.path.resolve():
                return False

            # Target must be contained within worktrees_dir
            try:
                resolved_target.relative_to(self.worktrees_dir.resolve())
            except ValueError:
                return False

            # Checkpoint must reference a valid request_id string
            req_id = task.checkpoint.details.get("request_id")
            if not req_id or not isinstance(req_id, str):
                return False

            # 6, 7 & 8. Authoritative persisted approval record cross-validation
            if not self.approvals_path.exists():
                return False

            try:
                with open(self.approvals_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return False

            if not isinstance(data, dict):
                return False

            requests_map = data.get("requests")
            if not isinstance(requests_map, dict):
                return False

            req_data = requests_map.get(req_id)
            if not isinstance(req_data, dict):
                return False

            # Validate using PermissionRequest and ApprovalStatus if available
            if PermissionRequest is not None and ApprovalStatus is not None:
                try:
                    req = PermissionRequest.from_dict(req_data)
                    if req.status != ApprovalStatus.APPROVED:
                        return False
                    if req.request_id != req_id:
                        return False
                    if req.task_id != task.task_id:
                        return False
                    if req.operation not in ("workspace_adoption", "reviewer_safeguards"):
                        return False
                    persisted_target = req.target
                except Exception:
                    return False
            else:
                if req_data.get("status") != "APPROVED":
                    return False
                if req_data.get("request_id") != req_id:
                    return False
                if req_data.get("task_id") != task.task_id:
                    return False
                if req_data.get("operation") not in ("workspace_adoption", "reviewer_safeguards"):
                    return False
                persisted_target = req_data.get("target")

            if not persisted_target or not isinstance(persisted_target, str):
                return False

            persisted_resolved = Path(persisted_target).resolve()
            if persisted_resolved != ws_status.path.resolve():
                return False

            return True
        except Exception:
            return False

    def reconcile_all(
        self,
        ledger: TaskLedger,
        coordinator_api=None,
        lock_held: bool = False,
    ) -> list[ReconciliationItem]:
        """
        Reconcile persisted task states with actual Git worktrees on coordinator restart.
        Ensures interrupted tasks are preserved and marked RECOVERY_REQUIRED.
        Guarantees VERIFIED tasks remain untouched.
        """
        if coordinator_api is None:
            # Compatibility for direct maintenance callers: reconciliation still flows
            # through CoordinatorAPI and its persisted transition/event checks.
            fallback_store = StateStore(self.repo_root / ".maya" / "maintenance" / "state.json")
            fallback_store.save_ledger(ledger)
            coordinator_api = CoordinatorAPI(fallback_store)
        elif not coordinator_api.store.exists():
            coordinator_api.store.save_ledger(ledger)
        if lock_held and coordinator_api.store.exists():
            persisted = coordinator_api.store.load_ledger()
            CoordinatorAPI._copy_ledger_into(ledger, persisted)

        reconciliations: list[ReconciliationItem] = []

        for task in ledger:
            ws_status = self.inspect_workspace(task.task_id)

            # Rule 1: A completed and verified task must remain verified after restart
            if task.status == TaskStatus.VERIFIED:
                continue

            # Rule 2: Interrupted in-progress tasks must be caught and marked RECOVERY_REQUIRED
            if task.status in (TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.INTEGRATING):
                prev_status = task.status
                blocking_reason = (
                    f"Coordinator restart: Interrupted while {prev_status.value}. "
                    f"Worktree preserved at {ws_status.path} (dirty={ws_status.is_dirty}). "
                    "Requires explicit recovery review."
                )
                if coordinator_api is None:
                    raise WorkspaceError("CoordinatorAPI is required to reconcile protected task state.")
                result = coordinator_api.transition_task(
                    ledger, task.task_id, TaskStatus.RECOVERY_REQUIRED, actor="recovery",
                    blocking_reason=blocking_reason,
                    checkpoint_stage="INTERRUPTED_RECOVERY_REQUIRED",
                    checkpoint_details={
                        "previous_status": prev_status.value,
                        "worktree_path": str(ws_status.path),
                        "is_dirty": ws_status.is_dirty,
                        "head_commit": ws_status.head_commit,
                    },
                    event_type="TASK_RECOVERY_REQUIRED",
                    idempotency_key=f"recovery:{task.task_id}:{prev_status.value}:{ws_status.head_commit}",
                    lock_held=lock_held,
                )
                if not result.accepted:
                    raise WorkspaceError(f"Coordinator rejected recovery transition for {task.task_id}: {result.message}")
                reconciliations.append(
                    ReconciliationItem(
                        task_id=task.task_id,
                        previous_status=prev_status,
                        new_status=TaskStatus.RECOVERY_REQUIRED,
                        action_taken="Marked RECOVERY_REQUIRED and recorded checkpoint",
                        reason="Interruption detected on restart while task was in flight.",
                    )
                )

            # Rule 2b: Interrupted integration with unverified commit on integration branch
            elif task.status == TaskStatus.READY_TO_INTEGRATE:
                int_branch = f"autophage/integrate-{task.task_id}"
                ref_check = subprocess.run(
                    ["git", "rev-parse", "--verify", f"refs/heads/{int_branch}"],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if ref_check.returncode == 0:
                    branch_tip = ref_check.stdout.strip()
                    base_commit = task.workspace.base_commit if task.workspace else ""
                    if branch_tip and base_commit and branch_tip != base_commit:
                        blocking_reason = (
                            f"Coordinator restart: Interrupted integration detected on branch '{int_branch}' "
                            f"(tip: {branch_tip[:7]}, baseline: {base_commit[:7]}). "
                            "Integration commit exists but was not verified. Requires explicit recovery review."
                        )
                        if coordinator_api is None:
                            raise WorkspaceError("CoordinatorAPI is required to reconcile protected task state.")
                        result = coordinator_api.transition_task(
                            ledger, task.task_id, TaskStatus.RECOVERY_REQUIRED, actor="recovery",
                            blocking_reason=blocking_reason,
                            checkpoint_stage="INTERRUPTED_INTEGRATION_DETECTED",
                            checkpoint_details={
                                "integration_branch": int_branch,
                                "branch_tip": branch_tip,
                                "baseline_commit": base_commit,
                            },
                            event_type="INTERRUPTED_INTEGRATION_DETECTED",
                            idempotency_key=f"interrupted-integration:{task.task_id}:{branch_tip}",
                            lock_held=lock_held,
                        )
                        if not result.accepted:
                            raise WorkspaceError(f"Coordinator rejected recovery transition for {task.task_id}: {result.message}")
                        reconciliations.append(
                            ReconciliationItem(
                                task_id=task.task_id,
                                previous_status=TaskStatus.READY_TO_INTEGRATE,
                                new_status=TaskStatus.RECOVERY_REQUIRED,
                                action_taken="Marked RECOVERY_REQUIRED and recorded checkpoint",
                                reason="Unverified integration commit detected on integration branch after interruption.",
                            )
                        )

            # Rule 3: Unscheduled worktree found for a PENDING or NEEDS_APPROVAL task
            elif task.status in (TaskStatus.PENDING, TaskStatus.NEEDS_APPROVAL) and ws_status.exists_on_disk:
                if self._is_workspace_adoption_authorized(task, ws_status):
                    continue
                prev_status = task.status
                blocking_reason = (
                    f"Unexpected worktree detected on disk for PENDING task at {ws_status.path}. "
                    "Requires human approval before scheduling."
                )
                if prev_status != TaskStatus.NEEDS_APPROVAL:
                    if coordinator_api is None:
                        raise WorkspaceError("CoordinatorAPI is required to reconcile protected task state.")
                    result = coordinator_api.transition_task(
                        ledger, task.task_id, TaskStatus.NEEDS_APPROVAL, actor="recovery",
                        blocking_reason=blocking_reason,
                        checkpoint_stage="UNSCHEDULED_WORKSPACE_DETECTED",
                        checkpoint_details={
                        "worktree_path": str(ws_status.path),
                        "is_dirty": ws_status.is_dirty,
                        },
                        event_type="UNSCHEDULED_WORKSPACE_DETECTED",
                        idempotency_key=f"workspace-adoption:{task.task_id}:{ws_status.head_commit}",
                        lock_held=lock_held,
                    )
                    if not result.accepted:
                        raise WorkspaceError(f"Coordinator rejected workspace recovery for {task.task_id}: {result.message}")
                if prev_status != TaskStatus.NEEDS_APPROVAL:
                    reconciliations.append(
                        ReconciliationItem(
                            task_id=task.task_id,
                            previous_status=prev_status,
                            new_status=TaskStatus.NEEDS_APPROVAL,
                            action_taken="Marked NEEDS_APPROVAL",
                            reason="Unexpected pre-existing workspace found for PENDING task.",
                        )
                    )

                # Ensure a durable PermissionRequest is registered in approvals.json
                if TerminalApprovalManager and self.approvals_path:
                    try:
                        store_to_use = coordinator_api.store if coordinator_api else None
                        mgr = TerminalApprovalManager(
                            self.approvals_path,
                            repo_root=self.repo_root,
                            store=store_to_use,
                        )
                        req_id = f"req-{task.task_id}-workspace-adoption"
                        if req_id not in mgr._requests:
                            mgr.create_request(
                                task_id=task.task_id,
                                risk=task.risk.value,
                                operation="workspace_adoption",
                                target=str(ws_status.path),
                                scope=f"adoption of pre-existing clean worktree on branch {self.get_branch_name(task.task_id)} synchronized to main",
                                reason=f"Unexpected pre-existing worktree detected at {ws_status.path}",
                                request_id=req_id,
                                lock_held=lock_held,
                            )
                    except Exception as req_err:
                        logger.warning("Failed to auto-register approval request for %s: %s", task.task_id, req_err)

            # Rule 4: Preserved worktree detected for a task in RECOVERY_REQUIRED
            elif task.status == TaskStatus.RECOVERY_REQUIRED and ws_status.exists_on_disk:
                created_req = False
                if TerminalApprovalManager and self.approvals_path:
                    try:
                        store_to_use = coordinator_api.store if coordinator_api else None
                        mgr = TerminalApprovalManager(
                            self.approvals_path,
                            repo_root=self.repo_root,
                            store=store_to_use,
                        )
                        req_id = f"req-{task.task_id}-recovery"
                        if req_id not in mgr._requests:
                            mgr.create_request(
                                task_id=task.task_id,
                                risk=task.risk.value,
                                operation="recovery",
                                target=str(ws_status.path),
                                scope=f"recovery review for preserved worktree on branch {self.get_branch_name(task.task_id)}",
                                reason=f"Worktree requires recovery: {task.blocking_reason or task.failure_reason or 'Interrupted execution'}",
                                request_id=req_id,
                                lock_held=lock_held,
                            )
                            created_req = True
                    except Exception as req_err:
                        logger.warning("Failed to auto-register recovery approval request for %s: %s", task.task_id, req_err)

                # First-pass reconciliation for this worktree in RECOVERY_REQUIRED
                if created_req or not task.checkpoint or task.checkpoint.stage != "RECOVERY_RECONCILED":
                    if coordinator_api and coordinator_api.store and lock_held and coordinator_api.store.exists():
                        persisted = coordinator_api.store.load_ledger()
                        CoordinatorAPI._copy_ledger_into(ledger, persisted)

                    task.blocking_reason = (
                        f"Preserved worktree at {ws_status.path} (dirty={ws_status.is_dirty}, "
                        f"head={ws_status.head_commit[:7] if ws_status.head_commit else 'None'}). "
                        "Requires explicit operator recovery review before execution."
                    )
                    task.checkpoint = CheckpointRecord(
                        timestamp=utc_now_iso(),
                        stage="RECOVERY_RECONCILED",
                        details={
                            "worktree_path": str(ws_status.path),
                            "is_dirty": ws_status.is_dirty,
                            "head_commit": ws_status.head_commit,
                            "recovery_request_id": f"req-{task.task_id}-recovery",
                        },
                    )
                    if coordinator_api and coordinator_api.store:
                        coordinator_api.store.save_ledger(ledger)
                    reconciliations.append(
                        ReconciliationItem(
                            task_id=task.task_id,
                            previous_status=TaskStatus.RECOVERY_REQUIRED,
                            new_status=TaskStatus.RECOVERY_REQUIRED,
                            action_taken="Preserved RECOVERY_REQUIRED and existing worktree contents",
                            reason=f"Worktree preserved at {ws_status.path}; recovery approval request registered.",
                        )
                    )

            # Rule 5: Task in NEEDS_APPROVAL lacking an active PermissionRequest
            elif task.status == TaskStatus.NEEDS_APPROVAL:
                if TerminalApprovalManager and self.approvals_path:
                    try:
                        store_to_use = coordinator_api.store if coordinator_api else None
                        mgr = TerminalApprovalManager(
                            self.approvals_path,
                            repo_root=self.repo_root,
                            store=store_to_use,
                        )
                        existing_req = mgr.get_pending_request_for_task(task.task_id)
                        if not existing_req:
                            is_baseline = "test doesn't prove the defect" in (task.blocking_reason or "")
                            op = "baseline_verification" if is_baseline else "operator_review"
                            req_id = f"req-{task.task_id}-baseline-review" if is_baseline else f"req-{task.task_id}-hold-review"
                            if req_id not in mgr._requests:
                                mgr.create_request(
                                    task_id=task.task_id,
                                    risk=task.risk.value if hasattr(task.risk, "value") else str(task.risk),
                                    operation=op,
                                    target=str(ws_status.path) if ws_status.exists_on_disk else ", ".join(task.target_files or []),
                                    scope=f"review for task in {task.status.value}: {task.blocking_reason or 'Pending review'}",
                                    reason=task.blocking_reason or task.failure_reason or "Task held in NEEDS_APPROVAL",
                                    request_id=req_id,
                                    lock_held=lock_held,
                                )
                    except Exception as req_err:
                        logger.warning("Failed to auto-register hold review request for %s: %s", task.task_id, req_err)

        return reconciliations

    def get_cleanup_preview(self, ledger: TaskLedger, target_branch: str = "main") -> list[CleanupPreviewItem]:
        """
        Produce a non-destructive cleanup preview of all managed and orphan worktrees.
        Evaluates the 5 explicit eligibility conditions before recommending SAFE_TO_CLEAN.
        Never deletes anything.
        """
        preview: list[CleanupPreviewItem] = []
        all_git_wts = self.list_git_worktrees()

        # 1. Inspect on-disk directories under .maya/worktrees/
        on_disk_dirs: set[Path] = set()
        if self.worktrees_dir.is_dir():
            for entry in self.worktrees_dir.iterdir():
                if entry.is_dir():
                    on_disk_dirs.add(entry.resolve())

        # Combine with registered worktrees located within worktrees_dir
        combined_paths = set(on_disk_dirs)
        for wt_path in all_git_wts:
            try:
                wt_path.relative_to(self.worktrees_dir)
                combined_paths.add(wt_path)
            except ValueError:
                # Primary working tree or external worktree
                pass

        for wt_path in sorted(combined_paths):
            task_id = wt_path.name
            task: TaskDefinition | None = None
            if task_id in ledger:
                task = ledger.get_task(task_id)

            status = self.inspect_workspace(task_id)
            blocking_reasons: list[str] = []
            actions: list[str] = []
            is_merged = False

            if task is None:
                blocking_reasons.append("Worktree is not registered in task ledger (unregistered orphan).")
                preview.append(
                    CleanupPreviewItem(
                        task_id=None,
                        path=str(wt_path),
                        branch_name=status.branch_name,
                        is_dirty=status.is_dirty,
                        task_status="UNREGISTERED_ORPHAN",
                        recommendation="MANUAL_REVIEW_REQUIRED",
                        reason="Worktree is not registered in task ledger. Preserve for manual inspection.",
                        is_merged=False,
                        eligible=False,
                        blocking_reasons=blocking_reasons,
                        actions=[],
                    )
                )
                continue

            # Condition 1: Task must be in terminal state permitting cleanup (VERIFIED)
            if task.status != TaskStatus.VERIFIED:
                blocking_reasons.append(
                    f"Task status is '{task.status.value}' (only VERIFIED tasks are eligible for cleanup)."
                )

            # Condition 2: Worktree must be clean (no untracked, modified or staged files)
            if status.is_dirty:
                blocking_reasons.append(
                    f"Worktree has {len(status.modified_files)} modified and {len(status.untracked_files)} untracked files."
                )

            # Condition 3: Worktree branch must be verified merged into target_branch
            if status.branch_name:
                is_merged = self.is_branch_merged(status.branch_name, target_branch)
                if not is_merged:
                    blocking_reasons.append(
                        f"Branch '{status.branch_name}' is not verified merged into target branch '{target_branch}'."
                    )
            else:
                blocking_reasons.append("No active Git branch identified for worktree.")

            # Condition 4: Managed ownership and branch naming match Autophage conventions
            expected_branch = self.get_branch_name(task.task_id)
            if status.ownership_status != "MANAGED":
                blocking_reasons.append(
                    f"Worktree ownership status is '{status.ownership_status}' (expected 'MANAGED')."
                )
            if status.branch_name and status.branch_name != expected_branch:
                blocking_reasons.append(
                    f"Branch name '{status.branch_name}' does not match expected Autophage branch '{expected_branch}'."
                )

            # Condition 5: No active task, pending approval or unresolved recovery references the workspace
            if task.status in (TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.INTEGRATING, TaskStatus.RECOVERY_REQUIRED):
                blocking_reasons.append(
                    f"Task {task.task_id} is currently active or in recovery ({task.status.value})."
                )
            if self.has_pending_approvals(task.task_id):
                blocking_reasons.append(
                    f"Task {task.task_id} has pending approval requests in approvals.json."
                )

            eligible = (len(blocking_reasons) == 0)

            if eligible:
                actions = [
                    f"git worktree remove {wt_path}",
                    f"git branch -d {status.branch_name}",
                ]
                rec = "SAFE_TO_CLEAN"
                reason = (
                    f"Task {task_id} is VERIFIED, branch '{status.branch_name}' is merged into '{target_branch}', "
                    "worktree is clean, and no pending approvals exist."
                )
            elif status.is_dirty:
                rec = "MANUAL_REVIEW_REQUIRED"
                reason = (
                    f"Worktree is dirty ({len(status.modified_files)} modified, "
                    f"{len(status.untracked_files)} untracked). Must preserve uncommitted work."
                )
            else:
                rec = "PRESERVE_FOR_INSPECTION"
                reason = "; ".join(blocking_reasons)

            preview.append(
                CleanupPreviewItem(
                    task_id=task_id,
                    path=str(wt_path),
                    branch_name=status.branch_name,
                    is_dirty=status.is_dirty,
                    task_status=task.status.value,
                    recommendation=rec,
                    reason=reason,
                    is_merged=is_merged,
                    eligible=eligible,
                    blocking_reasons=blocking_reasons,
                    actions=actions,
                )
            )

        return preview

    def cleanup_workspace(
        self,
        task_id: str,
        ledger: TaskLedger,
        target_branch: str = "main",
        authorized: bool = False,
    ) -> tuple[bool, str]:
        """
        Safely remove a finished task's worktree and delete its merged local branch.
        Fails closed unless all 5 safety conditions are met and authorized=True.
        Never force-deletes. Preserves task ledger records, event history, and trace data.
        """
        if not authorized:
            return False, f"Operator authorization required to cleanup worktree for task {task_id}."

        if task_id not in ledger:
            return False, f"Task {task_id} not found in task ledger; cannot clean up unmanaged worktree."

        task = ledger.get_task(task_id)
        status = self.inspect_workspace(task_id)
        wt_path = self.get_workspace_path(task_id)

        # Condition 1: Task must be VERIFIED
        if task.status != TaskStatus.VERIFIED:
            return False, f"Refusing cleanup: Task {task_id} is in status '{task.status.value}' (only VERIFIED tasks can be cleaned)."

        # Condition 2: Worktree must be clean
        if status.is_dirty:
            return False, f"Refusing cleanup: Worktree at {wt_path} has uncommitted/untracked changes."

        # Condition 3: Branch must be verified merged
        if not status.branch_name:
            return False, f"Refusing cleanup: Worktree at {wt_path} has no active branch."
        if not self.is_branch_merged(status.branch_name, target_branch):
            return False, f"Refusing cleanup: Branch '{status.branch_name}' is not verified merged into '{target_branch}'."

        # Condition 4: Managed ownership and branch naming match
        expected_branch = self.get_branch_name(task_id)
        if status.ownership_status != "MANAGED" or status.branch_name != expected_branch:
            return False, f"Refusing cleanup: Worktree ownership or branch '{status.branch_name}' does not match expected '{expected_branch}'."

        # Condition 5: No active task, pending approval or unresolved recovery
        if self.has_pending_approvals(task_id):
            return False, f"Refusing cleanup: Task {task_id} has pending approval requests."

        # Execute removal cleanly:
        # Step A: git worktree remove <path>
        if status.registered_in_git or status.exists_on_disk:
            res_wt = subprocess.run(
                ["git", "worktree", "remove", str(wt_path)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if res_wt.returncode != 0:
                return False, f"git worktree remove failed: {res_wt.stderr.strip() or res_wt.stdout.strip()}"

        # Step B: Prune local branch using safe -d (never force -D)
        if self.branch_exists(status.branch_name):
            res_br = subprocess.run(
                ["git", "branch", "-d", status.branch_name],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if res_br.returncode != 0:
                return False, f"git branch -d failed (branch not pruned): {res_br.stderr.strip() or res_br.stdout.strip()}"

        return True, f"Successfully cleaned worktree and pruned merged branch '{status.branch_name}' for task {task_id}."


def safe_restore_worktree_attempt(
    worktree_path: Path | str,
    task: TaskDefinition,
    base_commit: Optional[str] = None,
    repo_root: Optional[Path | str] = None,
) -> list[str]:
    """
    Safely rollback only the files modified or created during the current task attempt,
    preserving all unrelated tracked files, pre-existing modifications, task evidence,
    and policy-compliant untracked local test files.

    Target scope:
    - task.target_files
    - docs/CHANGELOG.md (if touched by attempt)
    - declared candidate test files in task.test_targets (if created/modified by attempt)

    Returns the list of restored/removed relative file paths.
    """
    wt = Path(worktree_path).resolve()
    if not wt.is_dir():
        logger.warning("Worktree path does not exist for attempt rollback: %s", worktree_path)
        return []

    # Resolve primary repository root
    r_root: Optional[Path] = None
    if repo_root is not None:
        r_root = Path(repo_root).resolve()
    else:
        git_common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(wt),
            capture_output=True,
            text=True,
            check=False,
        )
        if git_common.returncode == 0 and git_common.stdout.strip():
            raw_common = git_common.stdout.strip()
            common_path = Path(raw_common) if Path(raw_common).is_absolute() else (wt / raw_common).resolve()
            r_root = common_path.parent
        else:
            for parent in wt.parents:
                if (parent / ".maya").is_dir():
                    r_root = parent
                    break

    restored: list[str] = []
    scope_files = set(task.target_files or [])
    scope_files.add("docs/CHANGELOG.md")
    if task.test_targets:
        scope_files.update(task.test_targets)

    commit_ref = base_commit or "HEAD"

    for rel_str in sorted(scope_files):
        rel_posix = Path(rel_str).as_posix()
        target_path = (wt / rel_str).resolve()

        # Path traversal guard
        if not target_path.is_relative_to(wt):
            logger.warning("Path traversal attempt in safe_restore_worktree_attempt: %s", rel_str)
            continue

        # Check if file existed in Git at commit_ref
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{commit_ref}:{rel_posix}"],
            cwd=str(wt),
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            # File existed at commit_ref -> checkout only this specific file from commit_ref
            res = subprocess.run(
                ["git", "checkout", commit_ref, "--", rel_posix],
                cwd=str(wt),
                capture_output=True,
                check=False,
            )
            if res.returncode == 0:
                restored.append(rel_posix)
        else:
            # File did not exist at commit_ref
            # Handle policy-compliant untracked local test files under .maya/tests/
            if rel_posix.startswith(".maya/tests/"):
                canonical_test = (r_root / rel_posix).resolve() if r_root else None
                if canonical_test and canonical_test.is_file() and canonical_test.is_relative_to(r_root):
                    # Policy-compliant untracked local test mirrored from primary repository
                    # Restore from canonical baseline if missing or modified by attempt
                    try:
                        needs_restore = (
                            not target_path.is_file()
                            or target_path.read_bytes() != canonical_test.read_bytes()
                        )
                        if needs_restore:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(canonical_test, target_path)
                            restored.append(rel_posix)
                    except Exception as e:
                        logger.warning("Failed to restore canonical test file %s: %s", rel_str, e)
                else:
                    # File does NOT exist in canonical repo_root/.maya/tests/
                    # Genuinely disposable attempt artifact created during execution -> unlink
                    if target_path.is_file():
                        try:
                            target_path.unlink()
                            restored.append(rel_posix)
                        except Exception as e:
                            logger.warning("Failed to remove attempt-created test file %s: %s", rel_str, e)
            else:
                # Production target file did not exist at commit_ref -> attempt-created file, unlink it
                if target_path.is_file():
                    try:
                        target_path.unlink()
                        restored.append(rel_posix)
                    except Exception as e:
                        logger.warning("Failed to remove attempt-created file %s: %s", rel_str, e)

    return restored
