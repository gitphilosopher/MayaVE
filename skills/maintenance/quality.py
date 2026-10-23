"""
Mechanical Fix-Quality Gates for MayaVE Maintenance Coordinator.
Enforces diff boundaries, prevents test weakening, verifies local-only test policy,
runs targeted pytest checks in the worktree, and validates changelog updates.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from .models import TaskDefinition, TaskRisk, utc_now_iso
except ImportError:
    from models import TaskDefinition, TaskRisk, utc_now_iso

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


def freeze_task_tests(task: TaskDefinition, worktree_path: Path) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and freeze test targets before code modifications.
    Computes cryptographic SHA-256 hashes for all declared test target files.
    Returns (frozen_metadata, errors).
    """
    if task.is_defect_task() and not task.test_targets:
        return {}, ["Defect task requires non-empty 'test_targets' to establish baseline fail-to-pass evidence."]

    hashes: dict[str, str] = {}
    for t in (task.test_targets or []):
        target_path = (worktree_path / t).resolve()
        if not target_path.is_file():
            return {}, [f"Declared test target '{t}' does not exist on disk in worktree."]
        try:
            content = target_path.read_bytes()
            hashes[t] = hashlib.sha256(content).hexdigest()
        except Exception as e:
            return {}, [f"Could not read test target '{t}': {e}"]

    metadata = {
        "task_id": task.task_id,
        "frozen_at": utc_now_iso(),
        "targets": list(task.test_targets or []),
        "hashes": hashes,
        "origin": getattr(task, "audit_source", "") or "task_preparation",
    }
    return metadata, []


def parse_pytest_execution(
    stdout: str,
    stderr: str,
    exit_code: int,
    junit_xml_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Parse pytest execution into structured test results.
    Prefers JUnit XML if available and falls back to robust stdout/stderr analysis.
    """
    total = 0
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    collection_errors: list[dict[str, str]] = []
    xml_parsed = False

    if junit_xml_path and junit_xml_path.exists():
        try:
            tree = ET.parse(junit_xml_path)
            root = tree.getroot()
            suite = root.find("testsuite") if root.tag != "testsuite" else root
            if suite is not None:
                total = int(suite.attrib.get("tests", 0))
                errors = int(suite.attrib.get("errors", 0))
                failed = int(suite.attrib.get("failures", 0))
                skipped = int(suite.attrib.get("skipped", 0))
                passed = max(0, total - failed - errors - skipped)

                for tc in suite.findall("testcase"):
                    tc_name = f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}"
                    for failure in tc.findall("failure"):
                        failures.append({
                            "test": tc_name,
                            "message": (failure.attrib.get("message") or failure.text or "").strip()[:500],
                            "type": failure.attrib.get("type", "failure"),
                        })
                    for error in tc.findall("error"):
                        failures.append({
                            "test": tc_name,
                            "message": (error.attrib.get("message") or error.text or "").strip()[:500],
                            "type": error.attrib.get("type", "error"),
                        })

                # Check top-level errors (collection errors)
                for err in suite.findall("error"):
                    collection_errors.append({
                        "file": err.attrib.get("name", ""),
                        "message": (err.attrib.get("message") or err.text or "").strip()[:500],
                    })

                xml_parsed = True
        except Exception:
            xml_parsed = False

    if not xml_parsed:
        # Fallback to stdout/stderr text analysis
        text = f"{stdout}\n{stderr}"

        # Check summary line: e.g. "=== 1 failed, 1 passed in 0.16s ==="
        m_summary = re.search(r"=+ (.*?) in [\d.]+s =+", text)
        if m_summary:
            summary_str = m_summary.group(1)
            m_fail = re.search(r"(\d+)\s+failed", summary_str)
            m_pass = re.search(r"(\d+)\s+passed", summary_str)
            m_err = re.search(r"(\d+)\s+error", summary_str)
            m_skip = re.search(r"(\d+)\s+skipped", summary_str)

            if m_fail:
                failed = int(m_fail.group(1))
            if m_pass:
                passed = int(m_pass.group(1))
            if m_err:
                errors = int(m_err.group(1))
            if m_skip:
                skipped = int(m_skip.group(1))
            total = failed + passed + errors + skipped
        else:
            m_col = re.search(r"collected\s+(\d+)\s+item", text)
            if m_col:
                total = int(m_col.group(1))

        # Extract failed test lines
        for match in re.finditer(r"^FAILED\s+(.*?)(?:\s+-\s+(.*))?$", text, re.MULTILINE):
            failures.append({
                "test": match.group(1).strip(),
                "message": (match.group(2) or "").strip(),
                "type": "failure",
            })
        if not failed and failures:
            failed = len(failures)

        # Check for collection errors
        for match in re.finditer(r"ERROR collecting\s+(.*)", text):
            collection_errors.append({
                "file": match.group(1).strip(),
                "message": "Collection error",
            })
        if not errors and collection_errors:
            errors = len(collection_errors)

        if total == 0 and (failed > 0 or passed > 0 or errors > 0):
            total = failed + passed + errors + skipped

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "collected": total,
        "failures": failures,
        "collection_errors": collection_errors,
        "has_failures": failed > 0,
        "has_errors": errors > 0,
        "has_collection_errors": len(collection_errors) > 0,
    }


def is_valid_defect_baseline(details: dict) -> tuple[bool, str]:
    """
    Evaluates whether a pre-patch baseline test execution conclusively proves the defect.
    Returns (is_valid, reason).
    """
    if not isinstance(details, dict):
        return False, "Missing details dictionary from verification runner."

    exit_code = details.get("exit_code")
    test_results = details.get("test_results")

    # 1. Runner errors / timeouts
    if exit_code is None or exit_code == -1:
        return False, "Test runner failed or timed out during baseline execution."

    # 2. No tests collected
    if exit_code == 5:
        return False, "No tests collected in declared test targets."

    # 3. Missing structured test results
    if not test_results or not isinstance(test_results, dict):
        return False, "Missing structured test results from verification runner."

    # 4. Import / Collection errors
    if test_results.get("errors", 0) > 0 or test_results.get("collection_errors"):
        err_msg = ""
        if test_results.get("collection_errors"):
            err_msg = f": {test_results['collection_errors'][0].get('message', '')}"
        return False, f"Import or collection error encountered in test target{err_msg}."

    # 5. Tests collected check
    total_tests = test_results.get("total", 0)
    if total_tests == 0:
        return False, "Zero tests were executed in declared test targets."

    # 6. Baseline passed check (tests did not fail)
    failed_tests = test_results.get("failed", 0)
    if exit_code == 0 or failed_tests == 0:
        return False, "Baseline tests passed before patch was applied; does not prove defect."

    # 7. Unhandled exit codes
    if exit_code != 1:
        return False, f"Test runner exited with unexpected exit code {exit_code}."

    # 8. Valid defect proof
    failure_names = [f.get("test") for f in test_results.get("failures", []) if f.get("test")]
    detail_str = f" ({', '.join(failure_names[:3])})" if failure_names else ""
    return True, f"Defect confirmed: {failed_tests} of {total_tests} test(s) failed with assertion error{detail_str}."


class QualityGateEngine:
    """Evaluates worktree modifications against strict repository quality invariants."""

    def __init__(self, repo_root: Path | str) -> None:
        self.repo_root = Path(repo_root).resolve()

    def _get_python_executable(self) -> str:
        venv_py = self.repo_root / ".venv" / "Scripts" / "python.exe"
        if venv_py.is_file():
            return str(venv_py)
        return sys.executable

    def _get_ruff_executable(self) -> Optional[str]:
        candidates = [
            str(self.repo_root / ".venv" / "Scripts" / "ruff.exe"),
            str(self.repo_root / ".venv" / "bin" / "ruff"),
            str(Path(sys.executable).parent / "ruff.exe"),
            str(Path(sys.executable).parent / "ruff"),
            shutil.which("ruff"),
        ]
        for c in candidates:
            if c and Path(c).is_file():
                return str(Path(c).resolve())
        return None

    def _ensure_frontend_node_modules(self, worktree_path: Path) -> None:
        wt_front = worktree_path / "frontend"
        if not wt_front.is_dir():
            return
        wt_nm = wt_front / "node_modules"
        if wt_nm.exists():
            return
        repo_nm = self.repo_root / "frontend" / "node_modules"
        if repo_nm.is_dir():
            try:
                import _winapi
                _winapi.CreateJunction(str(repo_nm), str(wt_nm))
            except Exception:
                try:
                    os.symlink(repo_nm, wt_nm, target_is_directory=True)
                except Exception:
                    pass

    def _get_changed_files(self, task: TaskDefinition, worktree_path: Path) -> list[str]:
        changed = set()
        try:
            diff_proc = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if diff_proc.returncode == 0 and diff_proc.stdout.strip():
                for line in diff_proc.stdout.splitlines():
                    s = line.strip().replace("\\", "/")
                    if s:
                        changed.add(s)
        except Exception:
            pass

        try:
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(worktree_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if status_proc.returncode == 0 and status_proc.stdout.strip():
                for line in status_proc.stdout.splitlines():
                    if len(line) >= 4:
                        p = line[3:].strip().strip('"').replace("\\", "/")
                        if p:
                            changed.add(p)
        except Exception:
            pass

        if not changed and task.target_files:
            for tf in task.target_files:
                norm_tf = tf.replace("\\", "/").strip()
                if (worktree_path / norm_tf).exists():
                    changed.add(norm_tf)

        return sorted(list(changed))

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
            # Check if file is explicitly allowed or under .maya/
            if f in allowed_targets:
                continue
            if f.startswith(".maya/"):
                continue
            if "__pycache__" in f or f.endswith(".pyc"):
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

    def check_test_integrity(self, task: TaskDefinition, worktree_path: Path) -> QualityGateResult:
        """
        Verify that frozen test targets were not modified, weakened, or deleted during execution.
        """
        if not task.is_defect_task() and not task.frozen_tests:
            return QualityGateResult(
                gate_name="test_integrity",
                passed=True,
                message="Test integrity verification not required for non-defect task.",
            )

        if task.is_defect_task() and not task.frozen_tests:
            return QualityGateResult(
                gate_name="test_integrity",
                passed=False,
                message="test doesn't prove the defect: test targets were not frozen prior to execution.",
                details={"error": "unfrozen_tests"},
            )

        hashes = task.frozen_tests.get("hashes", {})
        for t, orig_hash in hashes.items():
            target_path = (worktree_path / t).resolve()
            if not target_path.is_file():
                return QualityGateResult(
                    gate_name="test_integrity",
                    passed=False,
                    message=f"Test target '{t}' was deleted during execution (tampering rejected).",
                    details={"target": t, "tampered": True},
                )
            cur_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
            if cur_hash != orig_hash:
                return QualityGateResult(
                    gate_name="test_integrity",
                    passed=False,
                    message=f"Test target '{t}' was modified after baseline freezing (tampering rejected).",
                    details={"target": t, "original_hash": orig_hash, "current_hash": cur_hash, "tampered": True},
                )

        return QualityGateResult(
            gate_name="test_integrity",
            passed=True,
            message="All frozen test targets verified unmodified.",
            details={"frozen_count": len(hashes)},
        )

    def check_python_static(self, task: TaskDefinition, worktree_path: Path) -> list[QualityGateResult]:
        """
        Deterministic static quality gates for changed Python files:
        1. py_compile check on each changed .py file.
        2. ruff check --select F on each changed .py file.
        Fails closed if a required tool is unavailable.
        """
        changed_files = self._get_changed_files(task, worktree_path)
        py_files = [f for f in changed_files if f.endswith(".py")]

        if not py_files:
            return [
                QualityGateResult(
                    gate_name="static_py_compile",
                    passed=True,
                    message="No Python files modified in worktree.",
                    details={"scope": "none"},
                )
            ]

        results: list[QualityGateResult] = []
        py_exe = self._get_python_executable()
        ruff_exe = self._get_ruff_executable()

        for rel in py_files:
            abs_path = worktree_path / rel
            if not abs_path.is_file():
                continue

            # 1. py_compile check
            try:
                compile_proc = subprocess.run(
                    [py_exe, "-m", "py_compile", str(abs_path)],
                    cwd=str(worktree_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if compile_proc.returncode != 0:
                    err = compile_proc.stderr.strip() or compile_proc.stdout.strip()
                    results.append(
                        QualityGateResult(
                            gate_name="static_py_compile",
                            passed=False,
                            message=f"Python compilation failed for '{rel}': {err[:300]}",
                            details={"command": f"python -m py_compile {rel}", "scope": rel, "exit_code": compile_proc.returncode, "diagnostics": err},
                        )
                    )
                else:
                    results.append(
                        QualityGateResult(
                            gate_name="static_py_compile",
                            passed=True,
                            message=f"Python compilation passed for '{rel}'.",
                            details={"command": f"python -m py_compile {rel}", "scope": rel, "exit_code": 0},
                        )
                    )
            except Exception as exc:
                results.append(
                    QualityGateResult(
                        gate_name="static_py_compile",
                        passed=False,
                        message=f"Failed to run py_compile on '{rel}': {exc}",
                        details={"scope": rel, "error": str(exc)},
                    )
                )

            # 2. ruff check --select F
            if not ruff_exe:
                results.append(
                    QualityGateResult(
                        gate_name="static_ruff",
                        passed=False,
                        message="Required static analysis tool 'ruff' is unavailable in environment.",
                        details={"command": "ruff check --select F", "scope": rel, "exit_code": 1, "diagnostics": "Tool 'ruff' not found"},
                    )
                )
            else:
                try:
                    ruff_proc = subprocess.run(
                        [ruff_exe, "check", "--select", "F", "--no-cache", str(abs_path)],
                        cwd=str(worktree_path),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if ruff_proc.returncode != 0:
                        out = ruff_proc.stdout.strip() or ruff_proc.stderr.strip()
                        results.append(
                            QualityGateResult(
                                gate_name="static_ruff",
                                passed=False,
                                message=f"Ruff F-checks failed for '{rel}': {out[:300]}",
                                details={"command": f"ruff check --select F {rel}", "scope": rel, "exit_code": ruff_proc.returncode, "diagnostics": out},
                            )
                        )
                    else:
                        results.append(
                            QualityGateResult(
                                gate_name="static_ruff",
                                passed=True,
                                message=f"Ruff F-checks passed for '{rel}'.",
                                details={"command": f"ruff check --select F {rel}", "scope": rel, "exit_code": 0},
                            )
                        )
                except Exception as exc:
                    results.append(
                        QualityGateResult(
                            gate_name="static_ruff",
                            passed=False,
                            message=f"Failed to execute ruff on '{rel}': {exc}",
                            details={"scope": rel, "error": str(exc)},
                        )
                    )

        return results

    def check_javascript_static(self, task: TaskDefinition, worktree_path: Path) -> list[QualityGateResult]:
        """
        Deterministic static quality gates for changed JavaScript and frontend files:
        1. node --check syntax check on each changed .js file.
        2. npm run build bundle/import check if any frontend file was modified.
        Fails closed if a required tool is unavailable.
        """
        changed_files = self._get_changed_files(task, worktree_path)
        js_files = [f for f in changed_files if f.endswith(".js")]
        frontend_modified = any(f.startswith("frontend/") for f in changed_files)

        if not js_files and not frontend_modified:
            return [
                QualityGateResult(
                    gate_name="static_javascript",
                    passed=True,
                    message="No JavaScript or frontend files modified in worktree.",
                    details={"scope": "none"},
                )
            ]

        results: list[QualityGateResult] = []
        node_exe = shutil.which("node")

        if js_files:
            if not node_exe:
                return [
                    QualityGateResult(
                        gate_name="static_js_syntax",
                        passed=False,
                        message="Required JavaScript runtime 'node' is unavailable in environment.",
                        details={"command": "node --check", "scope": "node", "exit_code": 1, "diagnostics": "Tool 'node' not found"},
                    )
                ]

            for rel in js_files:
                abs_path = worktree_path / rel
                if not abs_path.is_file():
                    continue
                try:
                    proc = subprocess.run(
                        [node_exe, "--check", str(abs_path)],
                        cwd=str(worktree_path),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode != 0:
                        err = proc.stderr.strip() or proc.stdout.strip()
                        results.append(
                            QualityGateResult(
                                gate_name="static_js_syntax",
                                passed=False,
                                message=f"JavaScript syntax check failed for '{rel}': {err[:300]}",
                                details={"command": f"node --check {rel}", "scope": rel, "exit_code": proc.returncode, "diagnostics": err},
                            )
                        )
                    else:
                        results.append(
                            QualityGateResult(
                                gate_name="static_js_syntax",
                                passed=True,
                                message=f"JavaScript syntax check passed for '{rel}'.",
                                details={"command": f"node --check {rel}", "scope": rel, "exit_code": 0},
                            )
                        )
                except Exception as exc:
                    results.append(
                        QualityGateResult(
                            gate_name="static_js_syntax",
                            passed=False,
                            message=f"Failed to run node --check on '{rel}': {exc}",
                            details={"scope": rel, "error": str(exc)},
                        )
                    )

        if frontend_modified and (worktree_path / "frontend" / "package.json").is_file():
            npm_exe = shutil.which("npm.cmd") or shutil.which("npm")
            if not npm_exe:
                results.append(
                    QualityGateResult(
                        gate_name="static_frontend_build",
                        passed=False,
                        message="Required tool 'npm' is unavailable to verify frontend build.",
                        details={"command": "npm run build", "scope": "frontend", "exit_code": 1, "diagnostics": "Tool 'npm' not found"},
                    )
                )
            else:
                self._ensure_frontend_node_modules(worktree_path)
                try:
                    build_proc = subprocess.run(
                        [npm_exe, "run", "build"],
                        cwd=str(worktree_path / "frontend"),
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=180,
                    )
                    if build_proc.returncode != 0:
                        err = (build_proc.stderr or build_proc.stdout).strip()
                        results.append(
                            QualityGateResult(
                                gate_name="static_frontend_build",
                                passed=False,
                                message=f"Frontend build verification failed: {err[:300]}",
                                details={"command": "npm run build", "scope": "frontend", "exit_code": build_proc.returncode, "diagnostics": err[-1500:]},
                            )
                        )
                    else:
                        results.append(
                            QualityGateResult(
                                gate_name="static_frontend_build",
                                passed=True,
                                message="Frontend build verification passed cleanly.",
                                details={"command": "npm run build", "scope": "frontend", "exit_code": 0},
                            )
                        )
                except subprocess.TimeoutExpired:
                    results.append(
                        QualityGateResult(
                            gate_name="static_frontend_build",
                            passed=False,
                            message="Frontend build verification timed out after 180s.",
                            details={"command": "npm run build", "scope": "frontend", "exit_code": 124},
                        )
                    )
                except Exception as exc:
                    results.append(
                        QualityGateResult(
                            gate_name="static_frontend_build",
                            passed=False,
                            message=f"Frontend build failed to execute: {exc}",
                            details={"scope": "frontend", "error": str(exc)},
                        )
                    )

        return results

    def check_defect_resolved(
        self,
        task: TaskDefinition,
        worktree_path: Path,
        patched_test_result: Optional[QualityGateResult] = None,
    ) -> QualityGateResult:
        """
        Verify defect resolution using evidence-based fail-to-pass test execution.
        Requires:
        1. Baseline test failure (exit code != 0) proving defect reproduction.
        2. Patched test success (exit code == 0) proving defect resolution.
        3. Optional pre-validation pattern sanity check if defined.
        For non-defect tasks (feature/improvement), exempt from fail-to-pass.
        """
        if not task.is_defect_task():
            return QualityGateResult(
                gate_name="defect_resolved",
                passed=True,
                message=f"Task classified as non-defect ({task.task_type}); fail-to-pass verification exempt.",
                details={"task_type": task.task_type, "fail_to_pass": "exempt"},
            )

        # Retrieve baseline verification evidence from checkpoint stage details
        baseline = None
        if task.checkpoint and getattr(task.checkpoint, "stage_details", None) and isinstance(task.checkpoint.stage_details, dict):
            baseline = task.checkpoint.stage_details.get("baseline_verification")
        if not baseline and task.checkpoint and getattr(task.checkpoint, "details", None) and isinstance(task.checkpoint.details, dict):
            baseline = task.checkpoint.details.get("baseline_verification")
        if not baseline:
            for rec in reversed(task.history):
                if rec.stage == "BASELINE_VERIFICATION":
                    if isinstance(rec.details, str):
                        try:
                            baseline = json.loads(rec.details)
                            break
                        except Exception:
                            pass
                    elif isinstance(rec.details, dict):
                        baseline = rec.details.get("baseline_verification") or rec.details
                        if baseline:
                            break

        if not baseline:
            return QualityGateResult(
                gate_name="defect_resolved",
                passed=False,
                message="test doesn't prove the defect: no baseline failure evidence recorded prior to patch.",
                details={"error": "missing_baseline_evidence"},
            )

        if baseline.get("passed") is True:
            return QualityGateResult(
                gate_name="defect_resolved",
                passed=False,
                message="test doesn't prove the defect: baseline tests passed before patch was applied.",
                details=baseline,
            )

        # Baseline confirmed failed! Now check patched verification test results
        if patched_test_result is None:
            patched_test_result = self.run_verification_tests(task, worktree_path)

        if not patched_test_result.passed:
            return QualityGateResult(
                gate_name="defect_resolved",
                passed=False,
                message=f"Defect unresolved: post-patch verification tests failed (exit code {patched_test_result.details.get('exit_code')}).",
                details={"baseline": baseline, "patched": patched_test_result.details},
            )

        # Secondary sanity check: pre_validation_pattern if defined
        if task.pre_validation_pattern:
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
                is_decl = bool(re.search(r"^\s*(def|class|export|function|const|let|var)\b", pat.pattern.strip()))
                if not is_decl and compiled.search(content):
                    return QualityGateResult(
                        gate_name="defect_resolved",
                        passed=False,
                        message=f"Pre-validation defect pattern '{pat.description}' still found in {pat.file_path}.",
                        details={"pattern": pat.pattern, "file": pat.file_path},
                    )
            except Exception as e:
                return QualityGateResult(
                    gate_name="defect_resolved",
                    passed=False,
                    message=f"Error evaluating defect pattern: {e}",
                )

        b_exit = baseline.get("exit_code")
        if b_exit is None and isinstance(baseline.get("details"), dict):
            b_exit = baseline["details"].get("exit_code")

        p_exit = patched_test_result.details.get("exit_code", 0) if isinstance(patched_test_result.details, dict) else 0

        return QualityGateResult(
            gate_name="defect_resolved",
            passed=True,
            message="Defect resolution verified via evidence-based fail-to-pass test execution.",
            details={
                "baseline_exit_code": b_exit,
                "patched_exit_code": p_exit,
                "fail_to_pass_verified": True,
            },
        )


    def run_verification_tests(self, task: TaskDefinition, worktree_path: Path) -> QualityGateResult:
        """
        Execute task.test_targets inside the worktree using the repository Python virtualenv.
        Populates structured test_results in details to unambiguously distinguish assertion
        failures from import errors, timeouts, or empty test targets.
        """
        if not task.test_targets:
            if task.is_defect_task():
                test_results = {
                    "total": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                    "failures": [], "collection_errors": [{"file": "", "message": "no test targets specified"}],
                }
                return QualityGateResult(
                    gate_name="verification_tests",
                    passed=False,
                    message="test doesn't prove the defect: no test targets specified for defect-fix task.",
                    details={"missing_target": True, "exit_code": -1, "test_results": test_results},
                )
            test_results = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "failures": [], "collection_errors": []}
            return QualityGateResult(
                gate_name="verification_tests",
                passed=True,
                message="No test targets specified for this non-defect task.",
                details={"exit_code": 0, "test_results": test_results},
            )

        py_exe = self._get_python_executable()
        # Resolve test paths relative to worktree
        resolved_targets = []
        for t in task.test_targets:
            target_path = (worktree_path / t).resolve()
            if target_path.exists():
                resolved_targets.append(str(target_path))
            else:
                test_results = {
                    "total": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                    "failures": [], "collection_errors": [{"file": t, "message": f"Declared test target does not exist: {t}"}],
                }
                return QualityGateResult(
                    gate_name="verification_tests",
                    passed=False,
                    message=f"Declared test target does not exist: {t}",
                    details={"missing_target": t, "exit_code": -1, "test_results": test_results},
                )

        env = os.environ.copy()
        orig_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{str(worktree_path)}{os.pathsep}{orig_pypath}"

        xml_tmp = None
        try:
            import tempfile
            tmp_dir = self.repo_root / ".maya" / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path_str = tempfile.mkstemp(prefix="pytest_junit_", suffix=".xml", dir=str(tmp_dir))
            os.close(fd)
            xml_tmp = Path(tmp_path_str)
        except Exception:
            xml_tmp = None

        cmd = [py_exe, "-m", "pytest", "-v", "--rootdir=.", "-o", "pythonpath=."]
        if xml_tmp:
            cmd.append(f"--junitxml={str(xml_tmp)}")
        cmd.extend(resolved_targets)

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
            test_results = {
                "total": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                "failures": [], "collection_errors": [{"file": "", "message": "Target verification pytest run timed out after 180s."}],
            }
            return QualityGateResult(
                gate_name="verification_tests",
                passed=False,
                message="Target verification pytest run timed out after 180s.",
                details={"exit_code": -1, "stdout_tail": "", "stderr_tail": "Timed out", "test_results": test_results},
            )
        except Exception as e:
            test_results = {
                "total": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0,
                "failures": [], "collection_errors": [{"file": "", "message": str(e)}],
            }
            return QualityGateResult(
                gate_name="verification_tests",
                passed=False,
                message=f"Failed to execute verification tests: {e}",
                details={"exit_code": -1, "stdout_tail": "", "stderr_tail": str(e), "test_results": test_results},
            )
        finally:
            pass

        try:
            test_results = parse_pytest_execution(proc.stdout, proc.stderr, proc.returncode, xml_tmp)
        finally:
            if xml_tmp and xml_tmp.exists():
                try:
                    xml_tmp.unlink(missing_ok=True)
                except OSError:
                    pass

        # Post-patch verification passes ONLY when all tests pass with zero failures and zero errors
        passed = (
            proc.returncode == 0
            and test_results.get("errors", 0) == 0
            and test_results.get("failed", 0) == 0
            and test_results.get("total", 0) > 0
        )
        return QualityGateResult(
            gate_name="verification_tests",
            passed=passed,
            message=f"Pytest verification {'PASSED' if passed else 'FAILED'} with return code {proc.returncode}.",
            details={
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
                "test_results": test_results,
            },
        )

    def evaluate(self, task: TaskDefinition, worktree_path: Path) -> QualityReport:
        """
        Run the complete battery of quality gates and return an aggregated report.
        """
        results: list[QualityGateResult] = []

        # 1. Diff Bounds Check
        results.append(self.check_diff_bounds(task, worktree_path))

        # 2. Local-Only Test Policy & Staging Check
        results.append(self.check_test_policy_and_staging(worktree_path))

        # 3. Test Weakening Check
        results.append(self.check_no_weakened_tests(worktree_path))

        # 4. Test Integrity Check (tampering defense against modifying frozen tests)
        r_integrity = self.check_test_integrity(task, worktree_path)
        results.append(r_integrity)

        # 5. Deterministic Static Quality Gates (py_compile and ruff F-checks for changed Python files)
        results.extend(self.check_python_static(task, worktree_path))

        # 6. JavaScript Static Gates (node --check syntax and frontend build verification)
        results.extend(self.check_javascript_static(task, worktree_path))

        # 7. Targeted Verification Tests (run against patched worktree)
        r_tests = self.run_verification_tests(task, worktree_path)
        results.append(r_tests)

        # 8. Evidence-Based Defect Resolution Check (baseline fail-to-pass)
        r_resolved = self.check_defect_resolved(task, worktree_path, patched_test_result=r_tests)
        results.append(r_resolved)

        # 9. Changelog Check
        results.append(self.check_changelog_entry(worktree_path))

        all_passed = all(r.passed for r in results)
        failure_reason = None
        if not all_passed:
            failed_gates = [r.gate_name for r in results if not r.passed]
            failure_reason = f"Quality gate(s) failed: {', '.join(failed_gates)}"

        for r in results:
            trace_emit(
                TraceEvent(
                    category="quality",
                    type="quality.gate_result",
                    task_id=task.task_id,
                    severity="INFO" if r.passed else "WARN",
                    summary=f"Gate {r.gate_name}: {'PASS' if r.passed else 'FAIL'} - {r.message[:100]}",
                    payload={"gate_name": r.gate_name, "passed": r.passed, "message": r.message},
                )
            )

        # Evaluate approval elevation:
        # Inconclusive evidence, test tampering, or high risk
        elevate = False
        approval_reason = None

        if not r_integrity.passed:
            elevate = True
            approval_reason = f"Verification inconclusive: {r_integrity.message}"
        elif not r_resolved.passed and "test doesn't prove the defect" in r_resolved.message:
            elevate = True
            approval_reason = r_resolved.message
        elif task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH) and all_passed:
            elevate = True
            approval_reason = f"{task.risk.value}-risk task requires human operator sign-off before downstream progression."

        return QualityReport(
            passed=all_passed,
            results=results,
            should_elevate_approval=elevate,
            approval_reason=approval_reason,
            failure_reason=failure_reason,
        )
