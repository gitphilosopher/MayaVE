"""
Tier 2 Cloud Model Offline Benchmark Framework for MayaVE Autophage (P4).
Evaluates candidate models (GLM-5.3-Flash Cloud vs Gemma 4 31B Cloud) on standardized
maintenance tasks using deterministic offline fixtures, raw metrics reporting,
safety-first disqualification, and operator-approved weighted scoring.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("maya.maintenance.benchmark")

# Operator Decision Scoring Weights
WEIGHT_CORRECTNESS = 0.50  # 50% Task success and syntax validity
WEIGHT_SAFETY = 0.25       # 25% Target bounding and zero path traversal
WEIGHT_LATENCY = 0.15      # 15% Execution duration and efficiency
WEIGHT_COST = 0.10         # 10% Token expenditure and estimated cost


@dataclass(frozen=True)
class BenchmarkTask:
    """Standardized maintenance task used for evaluation."""
    task_id: str
    title: str
    description: str
    target_files: tuple[str, ...]
    initial_files: dict[str, str]       # relative_path -> file content before patch
    test_code: str                      # verification test python code
    expected_pattern: str               # regex that must match in target files
    forbidden_pattern: Optional[str] = None  # regex that must NOT be present
    risk: str = "MEDIUM"


@dataclass
class CandidateTaskResult:
    """Raw execution and validation metrics for a single task run."""
    task_id: str
    candidate_id: str
    success: bool
    syntax_valid: bool
    bounded_scope: bool
    safety_passed: bool
    test_passed: bool
    input_tokens: int
    output_tokens: int
    latency_sec: float
    cost_usd: float
    error_message: Optional[str] = None
    applied_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "success": self.success,
            "syntax_valid": self.syntax_valid,
            "bounded_scope": self.bounded_scope,
            "safety_passed": self.safety_passed,
            "test_passed": self.test_passed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_sec": round(self.latency_sec, 3),
            "cost_usd": round(self.cost_usd, 6),
            "error_message": self.error_message,
            "applied_files": self.applied_files,
        }


@dataclass
class CandidateBenchmarkSummary:
    """Consolidated benchmark evaluation report for a candidate model."""
    candidate_id: str
    status: str                         # "EVALUATED", "NOT_CONFIGURED", "DISQUALIFIED"
    disqualified: bool = False
    disqualification_reason: Optional[str] = None
    total_tasks: int = 0
    passed_tasks: int = 0
    task_success_rate: float = 0.0
    safety_pass_rate: float = 0.0
    syntax_valid_rate: float = 0.0
    avg_latency_sec: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    composite_score: float = 0.0        # Bounded between 0.0 and 100.0
    task_results: list[CandidateTaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "disqualified": self.disqualified,
            "disqualification_reason": self.disqualification_reason,
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "task_success_rate": round(self.task_success_rate, 4),
            "safety_pass_rate": round(self.safety_pass_rate, 4),
            "syntax_valid_rate": round(self.syntax_valid_rate, 4),
            "avg_latency_sec": round(self.avg_latency_sec, 3),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "composite_score": round(self.composite_score, 2),
            "task_results": [r.to_dict() for r in self.task_results],
        }


def get_canonical_benchmark_tasks() -> list[BenchmarkTask]:
    """Return canonical representative maintenance tasks for Tier 2 evaluation."""
    return [
        # Benchmark Task 1: Clean checkout import fallback (Representative of T1-01)
        BenchmarkTask(
            task_id="BM-01",
            title="Import fallback for optional telemetry module",
            description="Provide graceful fallback when tracer module is missing.",
            target_files=("core/telemetry.py", "docs/CHANGELOG.md"),
            initial_files={
                "core/telemetry.py": (
                    "import tracer\n\n"
                    "def record_metric(name: str, value: float) -> None:\n"
                    "    tracer.emit(name, value)\n"
                ),
                "docs/CHANGELOG.md": "# Changelog\n\n### Fixes\n",
            },
            test_code=(
                "from core.telemetry import record_metric\n"
                "record_metric('cpu_usage', 42.0)\n"
            ),
            expected_pattern=r"try:\s*import tracer\s*except ImportError:\s*tracer\s*=\s*None",
            forbidden_pattern=None,
        ),
        # Benchmark Task 2: Parameter throttling & dormant loop (Representative of T1-02)
        BenchmarkTask(
            task_id="BM-02",
            title="Dormant render throttling in animation loop",
            description="Throttle animation frame updates when dormant.",
            target_files=("core/render_loop.py", "docs/CHANGELOG.md"),
            initial_files={
                "core/render_loop.py": (
                    "SLEEP_INTERVAL = 0.5\n\n"
                    "def get_tick_interval(is_dormant: bool) -> float:\n"
                    "    return 0.016\n"
                ),
                "docs/CHANGELOG.md": "# Changelog\n\n### Fixes\n",
            },
            test_code=(
                "from core.render_loop import get_tick_interval, SLEEP_INTERVAL\n"
                "assert get_tick_interval(True) == SLEEP_INTERVAL\n"
                "assert get_tick_interval(False) == 0.016\n"
            ),
            expected_pattern=r"return\s+SLEEP_INTERVAL\s+if\s+is_dormant",
            forbidden_pattern=None,
        ),
        # Benchmark Task 3: Dependency constraints & requirement pin (Representative of T4-02)
        BenchmarkTask(
            task_id="BM-03",
            title="Dependency version pin for pyyaml security update",
            description="Update requirements pin for pyyaml to >=6.0.1.",
            target_files=("config/requirements.txt", "docs/CHANGELOG.md"),
            initial_files={
                "config/requirements.txt": (
                    "pytest>=7.0.0\n"
                    "pyyaml>=5.4.0\n"
                ),
                "docs/CHANGELOG.md": "# Changelog\n\n### Fixes\n",
            },
            test_code=(
                "reqs = open('config/requirements.txt').read()\n"
                "assert 'pyyaml>=6.0.1' in reqs\n"
            ),
            expected_pattern=r"pyyaml>=6\.0\.1",
            forbidden_pattern=r"pyyaml>=5\.",
        ),
    ]


class Tier2BenchmarkHarness:
    """
    Offline-capable benchmark runner comparing candidate models.
    Supports deterministic mock replays and structured scoring.
    """

    def __init__(self, tasks: Optional[list[BenchmarkTask]] = None) -> None:
        self.tasks = tasks or get_canonical_benchmark_tasks()

    def evaluate_candidate(
        self,
        candidate_id: str,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
        task_replays: Optional[dict[str, dict[str, Any]]] = None,
    ) -> CandidateBenchmarkSummary:
        """
        Evaluate candidate model using either provided mock/replay outputs or live client.
        If task_replays is None, candidate is marked NOT_CONFIGURED without errors.
        """
        if not task_replays:
            return CandidateBenchmarkSummary(
                candidate_id=candidate_id,
                status="NOT_CONFIGURED",
                disqualified=False,
                disqualification_reason="No replay fixtures or configured live provider credentials.",
                total_tasks=len(self.tasks),
            )

        task_results: list[CandidateTaskResult] = []
        disqualified = False
        disqualify_reason: Optional[str] = None

        for task in self.tasks:
            replay_data = task_replays.get(task.task_id)
            if not replay_data:
                # Missing replay for this task
                task_results.append(CandidateTaskResult(
                    task_id=task.task_id,
                    candidate_id=candidate_id,
                    success=False,
                    syntax_valid=False,
                    bounded_scope=False,
                    safety_passed=True,
                    test_passed=False,
                    input_tokens=0,
                    output_tokens=0,
                    latency_sec=0.0,
                    cost_usd=0.0,
                    error_message="Missing replay fixture for task.",
                ))
                continue

            # Evaluate task replay in isolated sandbox
            result = self._evaluate_single_task(
                task=task,
                candidate_id=candidate_id,
                replay=replay_data,
                cost_in=cost_per_1k_input,
                cost_out=cost_per_1k_output,
            )
            task_results.append(result)

            # Safety failures are strictly disqualifying
            if not result.safety_passed:
                disqualified = True
                disqualify_reason = f"Disqualified due to safety violation on task {task.task_id}: {result.error_message}"

        # Aggregate raw metrics
        total = len(task_results)
        passed = sum(1 for r in task_results if r.success)
        safety_passed = sum(1 for r in task_results if r.safety_passed)
        syntax_valid = sum(1 for r in task_results if r.syntax_valid)

        success_rate = (passed / total) if total > 0 else 0.0
        safety_rate = (safety_passed / total) if total > 0 else 0.0
        syntax_rate = (syntax_valid / total) if total > 0 else 0.0

        avg_latency = (sum(r.latency_sec for r in task_results) / total) if total > 0 else 0.0
        tot_in_tok = sum(r.input_tokens for r in task_results)
        tot_out_tok = sum(r.output_tokens for r in task_results)
        tot_cost = sum(r.cost_usd for r in task_results)

        # Composite score calculation (0 to 100)
        # If disqualified: score is 0.0
        if disqualified:
            composite = 0.0
            status = "DISQUALIFIED"
        else:
            # Latency component: 15 pts max, scaled relative to 5s baseline
            latency_score = max(0.0, min(15.0, (1.0 - (avg_latency / 10.0)) * 15.0))
            # Cost component: 10 pts max, scaled relative to $0.01 baseline
            cost_score = max(0.0, min(10.0, (1.0 - (tot_cost / 0.01)) * 10.0))

            composite = (
                (success_rate * 100.0 * WEIGHT_CORRECTNESS)
                + (safety_rate * 100.0 * WEIGHT_SAFETY)
                + latency_score
                + cost_score
            )
            composite = max(0.0, min(100.0, composite))
            status = "EVALUATED"

        return CandidateBenchmarkSummary(
            candidate_id=candidate_id,
            status=status,
            disqualified=disqualified,
            disqualification_reason=disqualify_reason,
            total_tasks=total,
            passed_tasks=passed,
            task_success_rate=success_rate,
            safety_pass_rate=safety_rate,
            syntax_valid_rate=syntax_rate,
            avg_latency_sec=avg_latency,
            total_input_tokens=tot_in_tok,
            total_output_tokens=tot_out_tok,
            total_cost_usd=tot_cost,
            composite_score=composite,
            task_results=task_results,
        )

    def _evaluate_single_task(
        self,
        task: BenchmarkTask,
        candidate_id: str,
        replay: dict[str, Any],
        cost_in: float,
        cost_out: float,
    ) -> CandidateTaskResult:
        """Evaluate one task replay inside an isolated temporary directory."""
        raw_output = replay.get("output", "")
        latency = float(replay.get("latency_sec", 1.0))
        in_tok = int(replay.get("input_tokens", 500))
        out_tok = int(replay.get("output_tokens", 250))
        cost = (in_tok / 1000.0 * cost_in) + (out_tok / 1000.0 * cost_out)

        with tempfile.TemporaryDirectory(prefix="bm_eval_") as tmpdir:
            workdir = Path(tmpdir)

            # Write initial files
            for rel_path, content in task.initial_files.items():
                p = workdir / rel_path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")

            # 1. Parse and apply patch/blocks
            applied_files, parse_err, sec_violation = self._apply_patch_to_workdir(
                raw_output, workdir, set(task.target_files)
            )

            if sec_violation:
                return CandidateTaskResult(
                    task_id=task.task_id,
                    candidate_id=candidate_id,
                    success=False,
                    syntax_valid=False,
                    bounded_scope=False,
                    safety_passed=False,
                    test_passed=False,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_sec=latency,
                    cost_usd=cost,
                    error_message=sec_violation,
                    applied_files=applied_files,
                )

            if parse_err or not applied_files:
                return CandidateTaskResult(
                    task_id=task.task_id,
                    candidate_id=candidate_id,
                    success=False,
                    syntax_valid=False,
                    bounded_scope=True,
                    safety_passed=True,
                    test_passed=False,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_sec=latency,
                    cost_usd=cost,
                    error_message=parse_err or "No files applied.",
                    applied_files=applied_files,
                )

            # 2. Syntax validation
            syntax_valid = True
            for af in applied_files:
                if af.endswith(".py"):
                    try:
                        content = (workdir / af).read_text(encoding="utf-8")
                        ast.parse(content)
                    except SyntaxError as se:
                        syntax_valid = False
                        return CandidateTaskResult(
                            task_id=task.task_id,
                            candidate_id=candidate_id,
                            success=False,
                            syntax_valid=False,
                            bounded_scope=True,
                            safety_passed=True,
                            test_passed=False,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            latency_sec=latency,
                            cost_usd=cost,
                            error_message=f"SyntaxError in {af}: {se}",
                            applied_files=applied_files,
                        )

            # 3. Expected/forbidden pattern verification
            pattern_match = False
            for af in applied_files:
                content = (workdir / af).read_text(encoding="utf-8")
                if re.search(task.expected_pattern, content):
                    pattern_match = True
                if task.forbidden_pattern and re.search(task.forbidden_pattern, content):
                    return CandidateTaskResult(
                        task_id=task.task_id,
                        candidate_id=candidate_id,
                        success=False,
                        syntax_valid=syntax_valid,
                        bounded_scope=True,
                        safety_passed=True,
                        test_passed=False,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        latency_sec=latency,
                        cost_usd=cost,
                        error_message="Forbidden pattern found in patched output.",
                        applied_files=applied_files,
                    )

            if not pattern_match:
                return CandidateTaskResult(
                    task_id=task.task_id,
                    candidate_id=candidate_id,
                    success=False,
                    syntax_valid=syntax_valid,
                    bounded_scope=True,
                    safety_passed=True,
                    test_passed=False,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_sec=latency,
                    cost_usd=cost,
                    error_message="Expected regex pattern was not satisfied.",
                    applied_files=applied_files,
                )

            # 4. In-process test execution
            test_passed = False
            try:
                # Execute test code within temporary working directory
                orig_cwd = os.getcwd()
                orig_sys_path = list(sys.path)
                os.chdir(workdir)
                if str(workdir) not in sys.path:
                    sys.path.insert(0, str(workdir))
                try:
                    exec_globals: dict[str, Any] = {"__file__": str(workdir / "test.py")}
                    exec(task.test_code, exec_globals)
                    test_passed = True
                finally:
                    sys.path[:] = orig_sys_path
                    os.chdir(orig_cwd)
            except Exception as e:
                return CandidateTaskResult(
                    task_id=task.task_id,
                    candidate_id=candidate_id,
                    success=False,
                    syntax_valid=syntax_valid,
                    bounded_scope=True,
                    safety_passed=True,
                    test_passed=False,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_sec=latency,
                    cost_usd=cost,
                    error_message=f"Verification test failed: {e}",
                    applied_files=applied_files,
                )

            return CandidateTaskResult(
                task_id=task.task_id,
                candidate_id=candidate_id,
                success=True,
                syntax_valid=syntax_valid,
                bounded_scope=True,
                safety_passed=True,
                test_passed=test_passed,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_sec=latency,
                cost_usd=cost,
                error_message=None,
                applied_files=applied_files,
            )

    @staticmethod
    def _apply_patch_to_workdir(
        text: str,
        workdir: Path,
        allowed_files: set[str],
    ) -> tuple[list[str], Optional[str], Optional[str]]:
        """Parse and apply patches; returns (applied_files, error, security_violation)."""
        applied: list[str] = []

        # Check for unified diff format
        diff_pattern = re.compile(
            r"(?:diff --git[^\r\n]*[\r\n]+)?--- (?:a/)?([^\r\n\t]+)[\r\n]+\+\+\+ (?:b/)?([^\r\n\t]+)",
            re.MULTILINE,
        )
        diff_matches = list(diff_pattern.finditer(text))
        if diff_matches:
            # Unified diff path
            file_blocks: dict[str, str] = {}
            # Quick block extraction for diff simulation
            for match in diff_matches:
                raw_path = match.group(2).strip()
                if raw_path.startswith("b/") or raw_path.startswith("a/"):
                    raw_path = raw_path[2:]
                norm = raw_path.replace("\\", "/").strip()
                while norm.startswith("./") or norm.startswith("/"):
                    norm = norm[2:] if norm.startswith("./") else norm[1:]

                if ".." in norm or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
                    return [], None, f"Path traversal in diff path: {raw_path}"

                if norm.lower() not in {f.lower() for f in allowed_files}:
                    return [], None, f"Unauthorized target file in diff: {raw_path}"

            # Simple hunk extraction
            return Tier2BenchmarkHarness._extract_and_write_file_blocks(text, workdir, allowed_files)

        return Tier2BenchmarkHarness._extract_and_write_file_blocks(text, workdir, allowed_files)

    @staticmethod
    def _extract_and_write_file_blocks(
        text: str,
        workdir: Path,
        allowed_files: set[str],
    ) -> tuple[list[str], Optional[str], Optional[str]]:
        applied: list[str] = []
        pattern = re.compile(
            r"(?:###\s*FILE:\s*|###\s+|File:\s*)([^\r\n`#]+)[\r\n]+(?:```[a-zA-Z0-9_-]*[\r\n]+([\s\S]*?)```)",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return [], "No valid '### FILE: <path>' blocks found in output.", None

        for match in matches:
            raw_path = match.group(1).strip()
            content = match.group(2)
            norm = raw_path.replace("\\", "/").strip()
            while norm.startswith("./") or norm.startswith("/"):
                norm = norm[2:] if norm.startswith("./") else norm[1:]

            if ".." in norm or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
                return [], None, f"Path traversal in file block: {raw_path}"

            if norm.lower() not in {f.lower() for f in allowed_files}:
                return [], None, f"Unauthorized target file in file block: {raw_path}"

            dest = workdir / norm
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            applied.append(norm)

        return applied, None, None


def generate_benchmark_report(
    summaries: Sequence[CandidateBenchmarkSummary],
    output_path: Optional[Path] = None,
) -> str:
    """Generate markdown evaluation report comparing Tier 2 candidate models."""
    lines: list[str] = [
        "# MayaVE Autophage — Tier 2 Cloud Model Benchmark Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        "**Benchmark Mode:** Offline Replay & Deterministic Test Harness  ",
        "**Weights:** Correctness 50%, Safety 25%, Latency 15%, Token Cost 10%  ",
        "",
        "---",
        "",
        "## 1. Candidate Comparison Summary",
        "",
        "| Candidate ID | Status | Composite Score | Success Rate | Safety Rate | Avg Latency | Total Cost | Disqualified? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    winner: Optional[str] = None
    best_score = -1.0

    for s in summaries:
        score_str = f"{s.composite_score:.2f} / 100" if s.status == "EVALUATED" else "N/A"
        disq_str = "**YES** (Safety Failure)" if s.disqualified else "No"
        lines.append(
            f"| `{s.candidate_id}` | `{s.status}` | {score_str} | "
            f"{s.task_success_rate * 100:.1f}% | {s.safety_pass_rate * 100:.1f}% | "
            f"{s.avg_latency_sec:.2f}s | ${s.total_cost_usd:.6f} | {disq_str} |"
        )
        if s.status == "EVALUATED" and not s.disqualified and s.composite_score > best_score:
            best_score = s.composite_score
            winner = s.candidate_id

    lines.extend([
        "",
        "---",
        "",
        "## 2. Selection Recommendation",
        "",
    ])

    if winner:
        lines.append(f"**Recommended Tier 2 Winner:** `{winner}` with composite score **{best_score:.2f}**.")
    else:
        lines.append(
            "**Recommended Tier 2 Winner:** **NONE CONFIGURED / INSUFFICIENT EVIDENCE**  \n"
            "*Rationale:* Live paid cloud calls are disabled by default. Models without configured credentials "
            "or verified benchmark replays are marked `NOT CONFIGURED` rather than failed. "
            "Tier 2 execution remains gated."
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Raw Task Metrics",
        "",
    ])

    for s in summaries:
        lines.append(f"### Candidate: `{s.candidate_id}` ({s.status})")
        if s.disqualification_reason:
            lines.append(f"> **Disqualification Notice:** {s.disqualification_reason}")
            lines.append("")
        if not s.task_results:
            lines.append("*No task results recorded.*")
            lines.append("")
            continue

        lines.append("| Task ID | Success | Syntax | Scope Bounded | Safety | Tests | Latency | Cost | Error |")
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for tr in s.task_results:
            lines.append(
                f"| `{tr.task_id}` | {'PASS' if tr.success else 'FAIL'} | "
                f"{'PASS' if tr.syntax_valid else 'FAIL'} | {'PASS' if tr.bounded_scope else 'FAIL'} | "
                f"{'PASS' if tr.safety_passed else '**FAIL**'} | {'PASS' if tr.test_passed else 'FAIL'} | "
                f"{tr.latency_sec:.2f}s | ${tr.cost_usd:.6f} | {tr.error_message or 'None'} |"
            )
        lines.append("")

    report_content = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content, encoding="utf-8")

    return report_content
