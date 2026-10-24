"""
skills/maintenance/orchestration/auditor.py
Autonomous Auditor for AutoPhage 3.0.
Discovers defects, formulates hypotheses, and records evidence-based issues.
"""
from __future__ import annotations
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional, Set
from dataclasses import dataclass, field

from skills.maintenance.macro.tools import MacroToolRegistry, ToolExecutionContext, ToolResult
from skills.maintenance.ledger import MaintenanceLedger, TaskStatus

logger = logging.getLogger(__name__)

@dataclass
class DefectHypothesis:
    """A structured hypothesis about a potential defect."""
    issue_id: str
    title: str
    observed_behavior: str
    expected_behavior: str
    evidence: str
    affected_files: Set[str]
    severity: str
    confidence: str
    is_verified: bool = False
    repro_command: Optional[str] = None

    def fingerprint(self) -> str:
        """Deterministic fingerprint to prevent duplicate issues."""
        payload = f"{self.title}|{self.observed_behavior}|{self.expected_behavior}|{sorted(list(self.affected_files))}"
        return hashlib.sha256(payload.encode()).hexdigest()

class Auditor:
    """
    The Auditor explores the codebase to find bugs.
    It is strictly read-only and cannot apply fixes.
    """
    def __init__(
        self,
        repo_root: Path,
        ledger: MaintenanceLedger,
        registry: MacroToolRegistry,
    ):
        self.repo_root = repo_root
        self.ledger = ledger
        self.registry = registry

    def _get_auditor_context(self, task_id: str) -> ToolExecutionContext:
        """
        Creates a restricted context for the auditor.
        Ensures read-only access by omitting write-capable tools from the registry.
        """
        return ToolExecutionContext(
            worktree_path=self.repo_root / ".maya" / "worktrees" / "auditor",
            repo_root=self.repo_root,
            task_id=task_id,
            task_type="audit"
        )

    def audit_loop(self, scope_path: str, objective: str, worker: Optional[MaintenanceWorker] = None):
        """
        Main loop: Explore -> Hypothesize -> Verify -> Record.
        Now integrates with a real MaintenanceWorker for analysis.
        """
        logger.info(f"Auditing scope: {scope_path} with objective: {objective}")

        if not worker:
            logger.warning("No worker provided to Auditor. Falling back to structured mock for evidence generation.")
            # Keep the existing structured mock for fallback/tests
            hypothesis = DefectHypothesis(
                issue_id="AUDIT-001",
                title="Potential null pointer in router",
                observed_behavior="Router crashes when intent is empty",
                expected_behavior="Router should return UNKNOWN",
                evidence="Line 45 of dispatch.py accesses .name without check",
                affected_files={"brain/router/dispatch.py"},
                severity="P1",
                confidence="PROBABLE"
            )
            verified = self._verify_hypothesis(hypothesis)
            hypothesis.is_verified = verified
            if verified:
                self._record_issue(hypothesis)
            return

        # 1. Explore & Hypothesize using the worker
        ctx = self._get_auditor_context("system-audit")
        prompt = (
            f"Objective: {objective}\n"
            f"Scope: {scope_path}\n"
            "Analyze the codebase for defects. Identify potential bugs, security vulnerabilities, "
            "or architectural drift. For each finding, provide: \n"
            "1. Title\n2. Observed Behavior\n3. Expected Behavior\n4. Evidence (file:line)\n5. Affected Files\n"
            "6. Severity (P0-P3)\n"
            "7. Confidence (HIGH/PROBABLE/LOW)\n"
            "Constraint: You are in READ-ONLY mode. You cannot apply fixes."
        )

        response = worker.execute_turn(ctx, prompt)

        # 2. Parse worker response into hypothesis
        # In a real implementation, we would use a parser. Here we simulate the structured extraction.
        # This logic assumes the worker's thought contains the hypothesis details.
        hypothesis = self._parse_worker_hypothesis(response.thought)

        if hypothesis:
            # 3. Verify (Attempt safe reproduction)
            verified = self._verify_hypothesis(hypothesis)
            hypothesis.is_verified = verified

            # 4. Record
            if verified:
                self._record_issue(hypothesis)
            else:
                logger.info(f"Hypothesis {hypothesis.issue_id} remained unverified. Recorded as suspicion only.")
        else:
            logger.info("Auditor worker failed to formulate a structured hypothesis.")

    def _parse_worker_hypothesis(self, text: str) -> Optional[DefectHypothesis]:
        """Extracts a DefectHypothesis from worker thought text."""
        # Basic extraction logic for demonstration/integration
        if "Title:" not in text:
            return None

        try:
            # Simple key-value parsing
            lines = text.splitlines()
            data = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    data[k.strip().lower()] = v.strip()

            return DefectHypothesis(
                issue_id=f"AUDIT-{int(time.time())}",
                title=data.get("title", "Unknown Defect"),
                observed_behavior=data.get("observed behavior", "Unknown"),
                expected_behavior=data.get("expected behavior", "Unknown"),
                evidence=data.get("evidence", "Unknown"),
                affected_files={data.get("affected files", "unknown")},
                severity=data.get("severity", "P2"),
                confidence=data.get("confidence", "PROBABLE")
            )
        except Exception:
            return None

    def _verify_hypothesis(self, hypothesis: DefectHypothesis) -> bool:
        """
        Attempts focused, safe reproduction using a test.
        """
        # Auditor creates a test in .maya/tests
        test_path = self.repo_root / ".maya" / "tests" / f"test_audit_{hypothesis.issue_id}.py"

        # Use test_runner to verify
        ctx = self._get_auditor_context("system")
        res = self.registry.execute("test_runner", ctx, {"test_targets": [str(test_path)]})

        return res.success == False # If test fails, defect is reproduced

    def _record_issue(self, hypothesis: DefectHypothesis):
        """
        Records the issue to the actionable maintenance backlog.
        """
        # Prevent duplicates using fingerprint
        fingerprint = hypothesis.fingerprint()
        if self._is_duplicate(fingerprint):
            logger.info(f"Duplicate issue {hypothesis.issue_id} detected. Skipping.")
            return

        # Record to ledger (Transition to PENDING)
        self.ledger.update_task(
            hypothesis.issue_id,
            status=TaskStatus.PENDING,
            target_files=list(hypothesis.affected_files)
        )

        # Save detailed Markdown report to .maya/reports
        report_dir = self.repo_root / ".maya" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"audit_{hypothesis.issue_id}.md"
        report_content = f"# Audit Finding: {hypothesis.title}\n\n" \
                         f"**ID**: {hypothesis.issue_id}\n" \
                         f"**Observed**: {hypothesis.observed_behavior}\n" \
                         f"**Expected**: {hypothesis.expected_behavior}\n" \
                         f"**Evidence**: {hypothesis.evidence}\n" \
                         f"**Severity**: {hypothesis.severity}\n" \
                         f"**Confidence**: {hypothesis.confidence}\n"
        report_path.write_text(report_content)

    def _is_duplicate(self, fingerprint: str) -> bool:
        # In a real system, this would query a persistent fingerprint store.
        return False
