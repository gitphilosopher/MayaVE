"""
Mechanical Fix-Quality Gates for MayaVE Maintenance Coordinator.
Enforces diff boundaries, prevents test weakening, verifies local-only test policy,
runs targeted pytest checks in the worktree, and validates changelog updates.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from .models import TaskDefinition, TaskRisk
except ImportError:
    from models import TaskDefinition, TaskRisk


@dataclass
class QualityGateResult:
    gate_name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class QualityReport:
    passed: bool
    results: list[QualityGateResult] = field(default_factory=list)
    should_elevate_approval: bool = False
    approval_reason: Optional[str] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
            "should_elevate_approval": self.should_elevate_approval,
            "approval_reason": self.approval_reason,
            "failure_reason": self.failure_reason,
        }


class QualityGateEngine:
    """Evaluates worktree modifications against strict repository quality invariants."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).resolve()

    def _get_python_executable(self) -> str:
        venv_py = self.repo_root / ".venv" / "Scripts" / "python.exe"
        if venv_py.is_file():
            return str(venv_py)
        return sys.executable

    def check_diff_bounds(self, task: TaskDefinition, worktree_path: Path) -> QualityGateResult:
        """
        Ensure modifications are strictly confined to task.target_files, docs/CHANGELOG.md,
        and .maya/tests/. Any unauthorized file change fails immediately.
        """
        # Run git status --porcelain
        proc = subprocess.run(
            ["git", "status", "--porcelain", "-u"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return QualityGateResult(
                gate_name="diff_bounds",
                passed=False,
                message=f"Git status query failed in worktree: {proc.stderr}",
            )

        changed_files: list[str] = []
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            fname = line[2:].strip()
            # Handle possible rename "a -> b"
            if " -> " in fname:
                fname = fname.split(" -> ")[1].strip()
            changed_files.append(fname.replace("\\", "/"))

        if not changed_files:
            # Check diff against main if already committed on branch
            diff_proc = subprocess.run(
                ["git", "diff", "--name-only", "main...HEAD"],
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if diff_proc.returncode == 0:
                for line in diff_proc.stdout.splitlines():
                    if line.strip():
                        changed_files.append(line.strip().replace("\\", "/"))

        if not changed_files:
            return QualityGateResult(
                gate_name="diff_bounds",
                passed=False,
                message="No files were modified or created by the agent execution.",
                details={"changed_files": []},
            )

        # Build allowed normalized set
        allowed_targets = {t.replace("\\", "/") for t in task.target_files}
        allowed_targets.add("docs/CHANGELOG.md")

        unauthorized: list[str] = []
        for f in changed_files:
            # Check if file is explicitly allowed or under .maya/tests/
            if f in allowed_targets:
                continue
            if f.startswith(".maya/tests/"):
                continue
            unauthorized.append(f)

        if unauthorized:
            return QualityGateResult(
                gate_name="diff_bounds",
                passed=False,
                message=f"Diff boundary violated: unauthorized changes to {unauthorized}",
                details={
                    "changed_files": changed_files,
                    "unauthorized_files": unauthorized,
                    "allowed_targets": list(allowed_targets),
                },
            )

        return QualityGateResult(
            gate_name="diff_bounds",
            passed=True,
            message=f"Diff bounds verified: {len(changed_files)} file(s) modified within authorized scope.",
            details={"changed_files": changed_files},
        )

    def check_test_policy_and_staging(self, worktree_path: Path) -> QualityGateResult:
        """
        Verify that no test files were created outside .maya/tests/ and no changes are staged.
        """
        # 1. Check for staged files
        proc_staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_staged.returncode == 0 and proc_staged.stdout.strip():
            staged = [s.strip() for s in proc_staged.stdout.splitlines() if s.strip()]
            return QualityGateResult(
                gate_name="local_test_policy_and_staging",
                passed=False,
                message=f"Staging violation: agent staged files in index: {staged}",
                details={"staged_files": staged},
            )

        # 2. Check for test files created outside .maya/tests/
        proc_status = subprocess.run(
            ["git", "status", "--porcelain", "-u"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_status.returncode == 0:
            disallowed_test_locations = []
            for line in proc_status.stdout.splitlines():
                if not line.strip():
                    continue
                fname = line[2:].strip().replace("\\", "/")
                # Check if it looks like a test file outside .maya/tests/
                base_name = os.path.basename(fname)
                is_test_file = (
                    base_name.startswith("test_")
                    or base_name.endswith("_test.py")
                    or base_name.endswith(".test.js")
                    or base_name.endswith(".spec.js")
                )
                if is_test_file and not fname.startswith(".maya/tests/"):
                    disallowed_test_locations.append(fname)
                elif (fname.startswith("tests/") or fname.startswith("test/")) and not fname.startswith(".maya/tests/"):
                    disallowed_test_locations.append(fname)

            if disallowed_test_locations:
                return QualityGateResult(
                    gate_name="local_test_policy_and_staging",
                    passed=False,
                    message=f"Mandatory Local Test Policy violated: tests created outside .maya/tests/: {disallowed_test_locations}",
                    details={"disallowed_test_files": disallowed_test_locations},
                )

        return QualityGateResult(
            gate_name="local_test_policy_and_staging",
            passed=True,
            message="Local test policy and staging rules verified.",
        )

    def check_no_weakened_tests(self, worktree_path: Path) -> QualityGateResult:
        """
        Inspect diff of test files under .maya/tests/ to prevent weakening assertions,
        introducing skips, or removing test coverage. Checks both Git diff (if tracked)
        and baseline filesystem diff (for untracked local-only tests).
        """
        deleted_assertions = []
        added_skips = []

        assertion_patterns = [
            re.compile(r"^\-\s*assert\b"),
            re.compile(r"^\-\s*pytest\.raises\b"),
            re.compile(r"^\-\s*self\.assert[A-Za-z]+\b"),
            re.compile(r"^\-\s*expect\(.*?\)\.to"),
        ]
        skip_patterns = [
            re.compile(r"^\+\s*@pytest\.mark\.skip\b"),
            re.compile(r"^\+\s*pytest\.skip\b"),
            re.compile(r"^\+\s*test\.skip\b"),
        ]

        # 1. Check Git diff for tracked tests
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--", ".maya/tests/"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in proc.stdout.splitlines():
                for pat in assertion_patterns:
                    if pat.search(line):
                        deleted_assertions.append(line.strip())
                for pat in skip_patterns:
                    if pat.search(line):
                        added_skips.append(line.strip())

        # 2. Check filesystem baseline diff for local-only untracked tests
        import difflib
        src_tests = self.repo_root / ".maya" / "tests"
        wt_tests = worktree_path / ".maya" / "tests"
        if wt_tests.is_dir() and src_tests.is_dir():
            for wt_file in wt_tests.rglob("*.py"):
                if "__pycache__" in wt_file.parts:
                    continue
                rel = wt_file.relative_to(wt_tests)
                base_file = src_tests / rel
                if base_file.is_file():
                    try:
                        base_lines = base_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                        wt_lines = wt_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                        diff_lines = list(difflib.unified_diff(base_lines, wt_lines))
                        for line in diff_lines:
                            for pat in assertion_patterns:
                                if pat.search(line):
                                    deleted_assertions.append(line.strip())
                            for pat in skip_patterns:
                                if pat.search(line):
                                    added_skips.append(line.strip())
                    except Exception:
                        pass

        if deleted_assertions or added_skips:
            return QualityGateResult(
                gate_name="no_weakened_tests",
                passed=False,
                message=(
                    f"Test weakening detected: {len(deleted_assertions)} deleted assertion(s), "
                    f"{len(added_skips)} added skip(s)."
                ),
                details={
                    "deleted_assertions": deleted_assertions,
                    "added_skips": added_skips,
                },
            )

        return QualityGateResult(
            gate_name="no_weakened_tests",
            passed=True,
            message="Test assertion integrity preserved; no weakened tests or skips detected.",
        )

    def check_changelog_entry(self, worktree_path: Path) -> QualityGateResult:
        """
        Verify that docs/CHANGELOG.md was updated with a recorded entry under Fixes or Improvements.
        """
        changelog_path = worktree_path / "docs" / "CHANGELOG.md"
        if not changelog_path.is_file():
            return QualityGateResult(
                gate_name="changelog_entry",
                passed=False,
                message="docs/CHANGELOG.md does not exist in worktree.",
            )

        proc = subprocess.run(
            ["git", "diff", "HEAD", "--", "docs/CHANGELOG.md"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return QualityGateResult(
                gate_name="changelog_entry",
                passed=False,
                message=f"Failed to query changelog diff: {proc.stderr}",
            )

        diff = proc.stdout
        if not diff.strip():
            # If already committed on branch, check diff against main
            proc = subprocess.run(
                ["git", "diff", "main...HEAD", "--", "docs/CHANGELOG.md"],
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                check=False,
            )
            diff = proc.stdout

        if not diff.strip():
            return QualityGateResult(
                gate_name="changelog_entry",
                passed=False,
                message="docs/CHANGELOG.md was not updated by the task execution.",
            )

        # Verify added lines exist
        added_lines = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        if not added_lines:
            return QualityGateResult(
                gate_name="changelog_entry",
                passed=False,
                message="docs/CHANGELOG.md diff contains no additions.",
            )

        return QualityGateResult(
            gate_name="changelog_entry",
            passed=True,
            message=f"Changelog updated with {len(added_lines)} line(s).",
            details={"added_lines_sample": added_lines[:5]},
        )

    def check_defect_resolved(self, task: TaskDefinition, worktree_path: Path) -> QualityGateResult:
        """
        Re-evaluate the pre-validation pattern against the target file.
        If the unpatched defect regex still matches, the fix is incomplete.
        """
        if not task.pre_validation_pattern:
            return QualityGateResult(
                gate_name="defect_resolved",
                passed=True,
                message="No pre-validation pattern defined for task; defect resolution assumed.",
            )

        pat = task.pre_validation_pattern
        target = worktree_path / pat.file_path
        if not target.is_file():
            return QualityGateResult(
                gate_name="defect_resolved",
                passed=False,
                message=f"Target file {pat.file_path} missing in worktree.",
            )

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            compiled = re.compile(pat.pattern, re.MULTILINE)
            if compiled.search(content):
                # If the pattern is a function/class/export signature rather than an unpatched defect line,
                # verify that the declared target files were actively modified in the worktree.
                is_decl = bool(re.search(r"^\s*(def|class|export|function|const|let|var)\b", pat.pattern.strip()))
                if is_decl:
                    targets_to_check = [str(f) for f in task.target_files if (worktree_path / f).exists()] or [str(pat.file_path)]
                    diff_proc = subprocess.run(
                        ["git", "diff", "HEAD", "--"] + targets_to_check,
                        cwd=str(worktree_path),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                    if (diff_proc.stdout or "").strip():
                        return QualityGateResult(
                            gate_name="defect_resolved",
                            passed=True,
                            message=f"Target file(s) modified in worktree around '{pat.description}'.",
                        )
                return QualityGateResult(
                    gate_name="defect_resolved",
                    passed=False,
                    message=f"Defect pattern '{pat.description}' still found in {pat.file_path}.",
                    details={"pattern": pat.pattern, "file": pat.file_path},
                )
        except Exception as e:
            return QualityGateResult(
                gate_name="defect_resolved",
                passed=False,
                message=f"Error evaluating defect pattern: {e}",
            )

        return QualityGateResult(
            gate_name="defect_resolved",
            passed=True,
            message=f"Defect pattern '{pat.description}' successfully resolved in {pat.file_path}.",
        )

    def run_verification_tests(self, task: TaskDefinition, worktree_path: Path) -> QualityGateResult:
        """
        Execute task.test_targets inside the worktree using the repository Python virtualenv.
        """
        if not task.test_targets:
            return QualityGateResult(
                gate_name="verification_tests",
                passed=True,
                message="No test targets specified for this task.",
            )

        py_exe = self._get_python_executable()
        # Resolve test paths relative to worktree
        resolved_targets = []
        for t in task.test_targets:
            target_path = (worktree_path / t).resolve()
            if target_path.exists():
                resolved_targets.append(str(target_path))
            else:
                # If target does not exist on disk, report failure
                return QualityGateResult(
                    gate_name="verification_tests",
                    passed=False,
                    message=f"Declared test target does not exist: {t}",
                    details={"missing_target": t},
                )

        env = os.environ.copy()
        # Include worktree in PYTHONPATH so tests import worktree modules
        orig_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{str(worktree_path)}{os.pathsep}{orig_pypath}"

        cmd = [py_exe, "-m", "pytest", "-v", "--rootdir=.", "-o", "pythonpath=."] + resolved_targets
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                env=env,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name="verification_tests",
                passed=False,
                message="Target verification pytest run timed out after 180s.",
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="verification_tests",
                passed=False,
                message=f"Failed to execute verification tests: {e}",
            )

        passed = (proc.returncode == 0)
        return QualityGateResult(
            gate_name="verification_tests",
            passed=passed,
            message=f"Pytest verification {'PASSED' if passed else 'FAILED'} with return code {proc.returncode}.",
            details={
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
            },
        )

    def evaluate(self, task: TaskDefinition, worktree_path: Path) -> QualityReport:
        """
        Run the complete battery of quality gates and return an aggregated report.
        """
        results: list[QualityGateResult] = []

        # 1. Diff Bounds Check
        r_bounds = self.check_diff_bounds(task, worktree_path)
        results.append(r_bounds)

        # 2. Local-Only Test Policy & Staging Check
        r_policy = self.check_test_policy_and_staging(worktree_path)
        results.append(r_policy)

        # 3. Test Weakening Check
        r_weakening = self.check_no_weakened_tests(worktree_path)
        results.append(r_weakening)

        # 4. Defect Resolution Check
        r_resolved = self.check_defect_resolved(task, worktree_path)
        results.append(r_resolved)

        # 5. Changelog Check
        r_changelog = self.check_changelog_entry(worktree_path)
        results.append(r_changelog)

        # 6. Targeted Verification Tests
        r_tests = self.run_verification_tests(task, worktree_path)
        results.append(r_tests)

        all_passed = all(r.passed for r in results)
        failure_reason = None
        if not all_passed:
            failed_gates = [r.gate_name for r in results if not r.passed]
            failure_reason = f"Quality gate(s) failed: {', '.join(failed_gates)}"

        # Evaluate approval elevation:
        # High-risk tasks or cases with passing tests but boundary/changelog anomalies
        elevate = False
        approval_reason = None
        if task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH) and all_passed:
            elevate = True
            approval_reason = f"{task.risk.value}-risk task requires human operator sign-off before downstream progression."

        return QualityReport(
            passed=all_passed,
            results=results,
            should_elevate_approval=elevate,
            approval_reason=approval_reason,
            failure_reason=failure_reason,
        )
