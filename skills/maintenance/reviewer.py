"""
Read-Only Copilot Quality Reviewer Integration for MayaVE Maintenance Coordinator.
Enforces read-only permissions, risk-based review scheduling, review budget limits,
evidence-based verification, and worktree tamper detection.
"""

from __future__ import annotations

import abc
import enum
import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    from .adapter import kill_process_tree
    from .models import TaskDefinition, TaskRisk
except ImportError:
    from adapter import kill_process_tree
    from models import TaskDefinition, TaskRisk


class ReviewRecommendation(str, enum.Enum):
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    NEEDS_HUMAN_APPROVAL = "NEEDS_HUMAN_APPROVAL"
    ERROR = "ERROR"


@dataclass
class ReviewFinding:
    severity: str  # CRITICAL, WARNING, SUGGESTION
    category: str  # CORRECTNESS, REGRESSION, ARCHITECTURE, PERFORMANCE, SECURITY
    file: str
    message: str
    evidence: str = ""
    line_range: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "file": self.file,
            "line_range": self.line_range,
            "message": self.message,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewFinding:
        return cls(
            severity=data.get("severity", "WARNING"),
            category=data.get("category", "CORRECTNESS"),
            file=data.get("file", ""),
            line_range=data.get("line_range"),
            message=data.get("message", ""),
            evidence=data.get("evidence", ""),
        )


@dataclass
class ReviewResult:
    success: bool
    recommendation: ReviewRecommendation
    findings: list[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    duration_sec: float = 0.0
    raw_output: str = ""
    error_message: Optional[str] = None
    credit_usage: Optional[float] = None
    diff_sha256: Optional[str] = None
    worktree_mutated: bool = False
    auth_failed: bool = False
    rate_limited: bool = False

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "recommendation": self.recommendation.value,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "duration_sec": round(self.duration_sec, 2),
            "error_message": self.error_message,
            "diff_sha256": self.diff_sha256,
            "worktree_mutated": self.worktree_mutated,
            "auth_failed": self.auth_failed,
            "rate_limited": self.rate_limited,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewResult:
        findings = [ReviewFinding.from_dict(f) for f in data.get("findings", [])]
        return cls(
            success=data.get("success", False),
            recommendation=ReviewRecommendation(data.get("recommendation", "ERROR")),
            findings=findings,
            summary=data.get("summary", ""),
            duration_sec=float(data.get("duration_sec", 0.0)),
            raw_output=data.get("raw_output", ""),
            error_message=data.get("error_message"),
            credit_usage=data.get("credit_usage"),
            diff_sha256=data.get("diff_sha256"),
            worktree_mutated=data.get("worktree_mutated", False),
            auth_failed=data.get("auth_failed", False),
            rate_limited=data.get("rate_limited", False),
        )


def capture_worktree_fingerprint(worktree_path: Path) -> dict[str, tuple[float, int]]:
    """Capture mtime and size for all files in worktree to detect unexpected mutations."""
    fingerprint: dict[str, tuple[float, int]] = {}
    if not worktree_path.is_dir():
        return fingerprint

    for item in worktree_path.rglob("*"):
        if item.is_file():
            # Skip .git metadata internally modified by git status/locks
            if ".git" in item.parts:
                continue
            try:
                rel = str(item.relative_to(worktree_path))
                stat = item.stat()
                fingerprint[rel] = (stat.st_mtime, stat.st_size)
            except Exception:
                pass
    return fingerprint


def detect_worktree_mutations(worktree_path: Path, before: dict[str, tuple[float, int]]) -> list[str]:
    """Compare worktree state against before-fingerprint. Returns list of modified/added/deleted files."""
    after = capture_worktree_fingerprint(worktree_path)
    mutations: list[str] = []

    # Check modified or deleted
    for path_str, (mtime, size) in before.items():
        if path_str not in after:
            mutations.append(f"DELETED: {path_str}")
        elif after[path_str] != (mtime, size):
            mutations.append(f"MODIFIED: {path_str}")

    # Check newly created
    for path_str in after:
        if path_str not in before:
            mutations.append(f"CREATED: {path_str}")

    return mutations


class ReviewPolicy:
    """Evaluates whether an independent code review is required according to risk and budget."""

    def __init__(self, max_reviews_per_session: int = 5) -> None:
        self.max_reviews_per_session = max_reviews_per_session
        self.reviews_conducted = 0

    def should_review(self, task: TaskDefinition) -> tuple[bool, str]:
        """
        Determine if review is warranted based on task risk and touched subsystems.
        """
        # Low risk: automated checks and tests suffice
        if task.risk == TaskRisk.LOW:
            return False, "Low-risk task satisfied by targeted automated tests."

        # High and Very High risk: always requires review
        if task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH):
            return True, f"{task.risk.value}-risk task requires independent architectural and security review."

        # Medium risk: review if touching core, services, brain, or concurrency
        core_subsystems = ["core/", "brain/", "services/", "main.py", "config/"]
        for f in task.target_files:
            norm_f = f.replace("\\", "/")
            if any(norm_f.startswith(sub) for sub in core_subsystems):
                return True, f"Medium-risk task touches architectural subsystem ({f})."

        return True, "Medium-risk task requires independent verification."

    def has_budget_available(self) -> bool:
        return self.reviews_conducted < self.max_reviews_per_session

    def consume_review(self) -> None:
        self.reviews_conducted += 1


class ReviewerAdapter(abc.ABC):
    """Abstract interface for independent code review."""

    @abc.abstractmethod
    def review(
        self,
        task: TaskDefinition,
        diff: str,
        verification_summary: str,
        worktree_path: Path,
        timeout_seconds: int = 300,
    ) -> ReviewResult:
        """Execute independent read-only review of task diff within worktree."""
        pass


class FakeReviewerAdapter(ReviewerAdapter):
    """Deterministic mock reviewer for testing."""

    def __init__(
        self,
        default_recommendation: ReviewRecommendation = ReviewRecommendation.APPROVE,
        default_summary: str = "Fake review: changes are clean and verified.",
        findings_map: dict[str, list[ReviewFinding]] | None = None,
        timeout_on: set[str] | None = None,
        auth_error_on: set[str] | None = None,
        rate_limit_on: set[str] | None = None,
        malformed_output_on: set[str] | None = None,
        mutations: dict[str, Callable[[Path], None]] | None = None,
    ) -> None:
        self.default_recommendation = default_recommendation
        self.default_summary = default_summary
        self.findings_map = findings_map or {}
        self.timeout_on = timeout_on or set()
        self.auth_error_on = auth_error_on or set()
        self.rate_limit_on = rate_limit_on or set()
        self.malformed_output_on = malformed_output_on or set()
        self.mutations = mutations or {}
        self.history: list[dict] = []
        self.call_count = 0

    def review(
        self,
        task: TaskDefinition,
        diff: str,
        verification_summary: str,
        worktree_path: Path,
        timeout_seconds: int = 300,
    ) -> ReviewResult:
        self.call_count += 1
        start_time = time.monotonic()
        before_fingerprint = capture_worktree_fingerprint(worktree_path)

        diff_hasher = hashlib.sha256()
        diff_hasher.update(diff.encode("utf-8"))
        diff_sha = diff_hasher.hexdigest()

        self.history.append({
            "task_id": task.task_id,
            "diff_sha256": diff_sha,
            "timeout_seconds": timeout_seconds,
            "worktree_path": str(worktree_path),
        })

        if task.task_id in self.timeout_on:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="Review process timed out.",
                duration_sec=0.01,
                error_message=f"Reviewer timed out after {timeout_seconds}s.",
                diff_sha256=diff_sha,
            )

        if task.task_id in self.auth_error_on:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="Authentication failed: Copilot not logged in.",
                duration_sec=0.01,
                error_message="GitHub Copilot CLI authentication failure (not logged in).",
                diff_sha256=diff_sha,
                auth_failed=True,
            )

        if task.task_id in self.rate_limit_on:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="Usage limit reached: AI credits exhausted.",
                duration_sec=0.01,
                error_message="Copilot usage limit reached: AI credits exhausted.",
                diff_sha256=diff_sha,
                rate_limited=True,
            )

        if task.task_id in self.malformed_output_on:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="Reviewer returned malformed non-JSON output.",
                duration_sec=0.01,
                raw_output="<<<CORRUPT OUTPUT WITHOUT VALID STRUCTURE>>>",
                error_message="Malformed reviewer output: could not extract structured recommendation.",
                diff_sha256=diff_sha,
            )

        # Apply simulated mutation to test tamper detection
        if task.task_id in self.mutations:
            mutation_fn = self.mutations[task.task_id]
            mutation_fn(worktree_path)

        # Tamper detection
        mutations = detect_worktree_mutations(worktree_path, before_fingerprint)
        if mutations:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="Read-only violation: Reviewer modified files in worktree.",
                duration_sec=time.monotonic() - start_time,
                error_message=f"Reviewer mutated worktree files: {mutations}",
                diff_sha256=diff_sha,
                worktree_mutated=True,
            )

        findings = self.findings_map.get(task.task_id, [])
        recommendation = self.default_recommendation
        if any(f.severity == "CRITICAL" for f in findings):
            recommendation = ReviewRecommendation.REQUEST_CHANGES
        elif any(f.severity == "WARNING" for f in findings) and recommendation == ReviewRecommendation.APPROVE:
            recommendation = ReviewRecommendation.NEEDS_HUMAN_APPROVAL

        return ReviewResult(
            success=True,
            recommendation=recommendation,
            findings=findings,
            summary=self.default_summary,
            duration_sec=time.monotonic() - start_time,
            diff_sha256=diff_sha,
        )


class CopilotReviewerAdapter(ReviewerAdapter):
    """
    Read-only GitHub Copilot CLI Reviewer Adapter.
    Enforces non-interactive execution, denied write/shell permissions,
    timeout handling with process tree cleanup, and worktree tamper detection.
    """

    def __init__(
        self,
        executable: str | Path | None = None,
        model: str = "auto",
    ) -> None:
        self.model = model
        if executable:
            self.executable = str(executable)
        else:
            self.executable = shutil.which("copilot.exe") or shutil.which("copilot") or "copilot.exe"

    def build_review_prompt(self, task: TaskDefinition, diff: str, verification_summary: str) -> str:
        return (
            f"You are the MayaVE Independent Read-Only Quality Reviewer acting on Task {task.task_id}.\n"
            f"CRITICAL CONSTRAINT: You are in READ-ONLY mode. Do NOT write files or execute shell commands.\n\n"
            f"=== TASK CONTEXT ===\n"
            f"Task ID   : {task.task_id}\n"
            f"Title     : {task.title}\n"
            f"Risk Level: {task.risk.value}\n"
            f"Target Scope: {', '.join(task.target_files)}\n"
            f"Description: {task.description}\n\n"
            f"=== VERIFICATION SUMMARY ===\n"
            f"{verification_summary}\n\n"
            f"=== IMPLEMENTATION DIFF ===\n"
            f"{diff}\n\n"
            f"=== REVIEW INSTRUCTIONS ===\n"
            f"Analyze the diff for:\n"
            f"1. Root-cause correctness: Did the fix actually resolve the issue?\n"
            f"2. Scope discipline: Did the fix touch only declared files without unrelated refactoring?\n"
            f"3. Regression risk: Are there unhandled edge cases or subtle regressions?\n"
            f"4. Security and performance: Does it violate local-first or workstation resource bounds?\n\n"
            f"Return ONLY valid JSON matching this exact schema:\n"
            f"{{\n"
            f'  "recommendation": "APPROVE" | "REQUEST_CHANGES" | "NEEDS_HUMAN_APPROVAL",\n'
            f'  "summary": "<concise summary of review>",\n'
            f'  "findings": [\n'
            f"    {{\n"
            f'      "severity": "CRITICAL" | "WARNING" | "SUGGESTION",\n'
            f'      "category": "CORRECTNESS" | "REGRESSION" | "ARCHITECTURE" | "PERFORMANCE" | "SECURITY",\n'
            f'      "file": "<affected file>",\n'
            f'      "line_range": "<optional lines>",\n'
            f'      "message": "<clear description of issue>",\n'
            f'      "evidence": "<exact quote or proof>"\n'
            f"    }}\n"
            f"  ]\n"
            f"}}\n"
        )

    def review(
        self,
        task: TaskDefinition,
        diff: str,
        verification_summary: str,
        worktree_path: Path,
        timeout_seconds: int = 300,
    ) -> ReviewResult:
        start_time = time.monotonic()

        diff_hasher = hashlib.sha256()
        diff_hasher.update(diff.encode("utf-8"))
        diff_sha = diff_hasher.hexdigest()

        if not worktree_path.is_dir():
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                error_message=f"Worktree path does not exist: {worktree_path}",
                diff_sha256=diff_sha,
            )

        # Defense-in-depth: Deterministic pre-review security and diff limit scan
        try:
            try:
                from .reviewer_safeguards import ReviewerSafeguardScanner
            except ImportError:
                from reviewer_safeguards import ReviewerSafeguardScanner
            safeguard_report = ReviewerSafeguardScanner().scan_diff(task, diff, worktree_path=worktree_path)
            if safeguard_report.requires_approval:
                return ReviewResult(
                    success=False,
                    recommendation=ReviewRecommendation.NEEDS_HUMAN_APPROVAL,
                    summary=f"Pre-review safeguards held patch: {safeguard_report.approval_reason}",
                    error_message=f"Reviewer safeguards blocked external review: {safeguard_report.approval_reason}",
                    duration_sec=time.monotonic() - start_time,
                    diff_sha256=diff_sha,
                )
        except Exception as scan_err:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                error_message=f"Reviewer safeguard scanner failed closed: {scan_err}",
                diff_sha256=diff_sha,
            )

        resolved_exe = shutil.which(self.executable) or self.executable
        if not resolved_exe or not Path(str(resolved_exe)).exists():
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                error_message=f"GitHub Copilot CLI binary '{self.executable}' not found in PATH.",
                diff_sha256=diff_sha,
            )

        # Baseline worktree snapshot for tamper detection
        before_fingerprint = capture_worktree_fingerprint(worktree_path)

        prompt = self.build_review_prompt(task, diff, verification_summary)

        # Build strictly bounded, read-only invocation command
        cmd = [
            str(resolved_exe),
            "-p",
            prompt,
            "-C",
            str(worktree_path),
            "--silent",
            "--deny-tool=write",
            "--deny-tool=shell",
            "--no-ask-user",
            "--disallow-temp-dir",
            "--no-custom-instructions",
        ]
        if self.model and self.model != "auto":
            cmd.extend(["--model", self.model])

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(worktree_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                error_message=f"Failed to spawn Copilot CLI: {e}",
                duration_sec=time.monotonic() - start_time,
                diff_sha256=diff_sha,
            )

        timed_out = False
        stdout_data = ""
        stderr_data = ""

        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(proc)
            try:
                stdout_data, stderr_data = proc.communicate(timeout=5)
            except Exception:
                pass
            exit_code = 124
        except Exception as e:
            kill_process_tree(proc)
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                error_message=f"Subprocess communication error: {e}",
                duration_sec=time.monotonic() - start_time,
                diff_sha256=diff_sha,
            )

        duration = time.monotonic() - start_time

        # Post-review tamper inspection
        mutations = detect_worktree_mutations(worktree_path, before_fingerprint)
        if mutations:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="CRITICAL: Reviewer attempted to mutate worktree files.",
                duration_sec=duration,
                raw_output=stdout_data,
                error_message=f"Worktree tampering detected during review: {mutations}",
                diff_sha256=diff_sha,
                worktree_mutated=True,
            )

        if timed_out:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary=f"Copilot review timed out after {timeout_seconds}s.",
                duration_sec=duration,
                error_message=f"Copilot review process exceeded timeout of {timeout_seconds}s.",
                diff_sha256=diff_sha,
            )

        combined_output = f"{stdout_data}\n{stderr_data}".lower()

        # Check for authentication errors
        if "not logged in" in combined_output or "run 'copilot login'" in combined_output or "authentication required" in combined_output:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="Copilot CLI authentication failure.",
                duration_sec=duration,
                raw_output=stdout_data,
                error_message="GitHub Copilot CLI authentication failure: session is not logged in.",
                diff_sha256=diff_sha,
                auth_failed=True,
            )

        # Check for usage/credit limit exhaustion
        if "session limit" in combined_output or "ai credits" in combined_output and "exhaust" in combined_output or "rate limit" in combined_output:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary="Copilot CLI usage limit or AI credits exhausted.",
                duration_sec=duration,
                raw_output=stdout_data,
                error_message="Copilot CLI session limit or credit quota exhausted.",
                diff_sha256=diff_sha,
                rate_limited=True,
            )

        if exit_code != 0:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.ERROR,
                summary=f"Copilot CLI exited with error code {exit_code}.",
                duration_sec=duration,
                raw_output=stdout_data,
                error_message=f"Process exited with code {exit_code}: {stderr_data.strip()[:300]}",
                diff_sha256=diff_sha,
            )

        # Extract structured JSON from output
        json_dict = self._extract_json(stdout_data)
        if not json_dict:
            return ReviewResult(
                success=False,
                recommendation=ReviewRecommendation.NEEDS_HUMAN_APPROVAL,
                summary="Copilot output could not be parsed as structured JSON.",
                duration_sec=duration,
                raw_output=stdout_data,
                error_message="Malformed review output: expected JSON schema.",
                diff_sha256=diff_sha,
            )

        raw_rec = json_dict.get("recommendation", "NEEDS_HUMAN_APPROVAL").upper()
        rec = ReviewRecommendation.NEEDS_HUMAN_APPROVAL
        if raw_rec == "APPROVE":
            rec = ReviewRecommendation.APPROVE
        elif raw_rec == "REQUEST_CHANGES":
            rec = ReviewRecommendation.REQUEST_CHANGES

        findings_raw = json_dict.get("findings", [])
        findings = []
        if isinstance(findings_raw, list):
            for item in findings_raw:
                if isinstance(item, dict):
                    findings.append(ReviewFinding.from_dict(item))

        # Enforce that critical findings block approval
        if any(f.severity == "CRITICAL" for f in findings):
            rec = ReviewRecommendation.REQUEST_CHANGES

        return ReviewResult(
            success=True,
            recommendation=rec,
            findings=findings,
            summary=json_dict.get("summary", "Review completed."),
            duration_sec=duration,
            raw_output=stdout_data,
            diff_sha256=diff_sha,
        )

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON dictionary from text or fenced code blocks."""
        if not text:
            return None

        # Try fenced code block first
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try direct bracket search
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        return None
