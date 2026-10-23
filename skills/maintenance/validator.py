"""
Pre-validation framework for MayaVE Maintenance Coordinator.
Performs non-destructive, read-only checks on repository baseline and audit findings.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from .ledger import TaskLedger
    from .models import PreValidationStatus, TaskDefinition, TaskPriority, TaskStatus
except ImportError:
    from ledger import TaskLedger
    from models import PreValidationStatus, TaskDefinition, TaskPriority, TaskStatus


@dataclass
class GitBaseline:
    branch: str
    head_sha: str
    is_clean: bool
    modified_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)


@dataclass
class TaskFindingCheck:
    task_id: str
    title: str
    priority: TaskPriority
    status: TaskStatus
    pre_validation_status: PreValidationStatus
    details: str
    target_files_present: list[str] = field(default_factory=list)
    target_files_missing: list[str] = field(default_factory=list)


@dataclass
class PreValidationReport:
    repo_root: Path
    git_baseline: Optional[GitBaseline]
    graph_valid: bool
    graph_error: Optional[str]
    tasks_checked: list[TaskFindingCheck] = field(default_factory=list)
    eligible_count: int = 0
    blocked_count: int = 0
    verified_count: int = 0


class PreValidator:
    """Non-destructive environment and defect pre-validator."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).resolve()

    def check_repo_structure(self) -> tuple[bool, str]:
        """Verify essential repository anchors exist."""
        git_dir = self.repo_root / ".git"
        changelog = self.repo_root / "docs" / "CHANGELOG.md"
        agents_md = self.repo_root / ".agents" / "AGENTS.md"

        if not git_dir.exists():
            return False, f"Git directory missing at: {git_dir}"
        if not changelog.exists():
            return False, f"Changelog missing at: {changelog}"
        if not agents_md.exists():
            return False, f"AGENTS.md missing at: {agents_md}"

        return True, "Repository root structure verified."

    def inspect_git_baseline(self) -> GitBaseline:
        """Inspect Git branch, commit, and working tree without modifying any state."""
        # 1. Branch
        branch_proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        branch = branch_proc.stdout.strip() or "HEAD_DETACHED"

        # 2. SHA
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        head_sha = sha_proc.stdout.strip() or "UNKNOWN"

        # 3. Status
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )

        modified: list[str] = []
        untracked: list[str] = []

        for raw_line in status_proc.stdout.splitlines():
            if not raw_line.strip():
                continue
            code = raw_line[:2]
            file_name = raw_line[2:].strip()
            if code == "??":
                untracked.append(file_name)
            else:
                modified.append(file_name)

        is_clean = len(modified) == 0 and len(untracked) == 0

        return GitBaseline(
            branch=branch,
            head_sha=head_sha,
            is_clean=is_clean,
            modified_files=modified,
            untracked_files=untracked,
        )

    def validate_task_finding(self, task: TaskDefinition) -> TaskFindingCheck:
        """
        Inspect task targets and pre-validation regex pattern.
        Determines if the audit finding is confirmed active in the current code.
        """
        present: list[str] = []
        missing: list[str] = []

        for rel_path in task.target_files:
            target_path = self.repo_root / rel_path
            if target_path.exists():
                present.append(rel_path)
            else:
                missing.append(rel_path)

        # Evaluate target existence
        if not present and missing:
            return TaskFindingCheck(
                task_id=task.task_id,
                title=task.title,
                priority=task.priority,
                status=task.status,
                pre_validation_status=PreValidationStatus.TARGET_MISSING,
                details=f"All declared target files are missing: {', '.join(missing)}",
                target_files_present=present,
                target_files_missing=missing,
            )

        # Evaluate pattern check if defined
        if task.pre_validation_pattern:
            pattern_file = self.repo_root / task.pre_validation_pattern.file_path
            if not pattern_file.exists():
                return TaskFindingCheck(
                    task_id=task.task_id,
                    title=task.title,
                    priority=task.priority,
                    status=task.status,
                    pre_validation_status=PreValidationStatus.TARGET_MISSING,
                    details=f"Pattern target file missing: {task.pre_validation_pattern.file_path}",
                    target_files_present=present,
                    target_files_missing=missing,
                )

            try:
                content = pattern_file.read_text(encoding="utf-8", errors="ignore")
                matches = re.search(task.pre_validation_pattern.pattern, content)
                if matches:
                    status = PreValidationStatus.CONFIRMED_UNPATCHED
                    details = (
                        f"Pattern matched in '{task.pre_validation_pattern.file_path}': "
                        f"{task.pre_validation_pattern.description}"
                    )
                else:
                    status = PreValidationStatus.ALREADY_PATCHED
                    details = (
                        f"Pattern not found in '{task.pre_validation_pattern.file_path}'. "
                        "Finding may already be resolved or modified."
                    )
            except Exception as e:
                status = PreValidationStatus.AMBIGUOUS
                details = f"Failed to inspect pattern file: {e}"

            return TaskFindingCheck(
                task_id=task.task_id,
                title=task.title,
                priority=task.priority,
                status=task.status,
                pre_validation_status=status,
                details=details,
                target_files_present=present,
                target_files_missing=missing,
            )

        # No pattern defined; present files established
        return TaskFindingCheck(
            task_id=task.task_id,
            title=task.title,
            priority=task.priority,
            status=task.status,
            pre_validation_status=PreValidationStatus.AMBIGUOUS,
            details=f"Target files present ({len(present)}/{len(task.target_files)}), no automated pattern defined.",
            target_files_present=present,
            target_files_missing=missing,
        )

    def run_pre_validation(self, ledger: TaskLedger) -> PreValidationReport:
        """Run complete non-destructive validation suite."""
        _, _ = self.check_repo_structure()
        git_info = self.inspect_git_baseline()

        graph_valid = True
        graph_err = None
        try:
            ledger.validate_graph()
        except Exception as e:
            graph_valid = False
            graph_err = str(e)

        task_checks: list[TaskFindingCheck] = []
        for task in ledger:
            task_checks.append(self.validate_task_finding(task))

        # Sort by priority rank then task_id
        task_checks.sort(key=lambda c: (c.priority.rank, c.task_id))

        return PreValidationReport(
            repo_root=self.repo_root,
            git_baseline=git_info,
            graph_valid=graph_valid,
            graph_error=graph_err,
            tasks_checked=task_checks,
            eligible_count=0,
            blocked_count=0,
            verified_count=0,
        )
