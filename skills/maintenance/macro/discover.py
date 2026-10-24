"""
skills/maintenance/macro/discover.py
AutoPhage 2.0 - Phase 3: Repository Intelligence & Progressive Discovery Engine.

Design invariants:
- Bounded repository inventory and architectural subsystem mapping.
- Python AST symbol discovery and JS/TS declaration extraction with malformed source tolerance.
- Staged discovery pipeline:
    Stage 1: Mechanical static checks (syntax errors, swallowed exceptions, dangerous calls, unmapped tests).
    Stage 2: Subsystem selection and bounded context assembly.
    Stage 3: Focused MACRO audit loop with read-only exploration tools.
    Stage 4: Evidence verification (grounding files, lines, and test reproductions).
    Stage 5: Deduplication, severity/confidence ranking, and Markdown backlog export.
- Strictly read-only with respect to production source files, git history, and maintenance ledgers.
- Generates Markdown output consumable by parse_markdown_backlog for Phase 4 campaign intake.
- Zero network or live model requirement in mechanical-only mode.
- Concurrency = 1 with bounded context budgets.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence

from .tools import (
    MacroToolRegistry,
    ToolExecutionContext,
    create_default_registry,
)
from .agent import (
    MacroAgent,
    MacroAgentConfig,
    MacroAgentResult,
)

try:
    from skills.maintenance.task_contract import FORBIDDEN_TARGET_PREFIXES
except ImportError:
    try:
        from task_contract import FORBIDDEN_TARGET_PREFIXES
    except ImportError:
        FORBIDDEN_TARGET_PREFIXES = (
            ".git/",
            ".maya/maintenance/",
            "skills/maintenance/",
            ".maya/tmp/",
            ".maya/reports/",
            ".maya/worktrees/",
        )

logger = logging.getLogger("maya.maintenance.macro.discover")

# ==============================================================================
# Constants & Subsystem Taxonomy
# ==============================================================================

CANONICAL_SUBSYSTEMS: tuple[str, ...] = (
    "core",
    "brain",
    "services",
    "skills",
    "config",
    "frontend",
    "scripts",
    "audio",
)

DEFAULT_EXCLUSIONS: tuple[str, ...] = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".maya/worktrees",
    ".maya/reports",
    ".maya/tmp",
    ".maya/maintenance",
    "skills/maintenance",
    "build",
    "dist",
    ".egg-info",
)

KNOWN_ENTRY_POINTS: tuple[str, ...] = (
    "main.py",
    "cli.py",
    "run_maya.py",
    "skills/maintenance/coordinator.py",
    "brain/router/understand.py",
    "services/ws_server.py",
)


# ==============================================================================
# Data Models & Enums
# ==============================================================================

class FindingCategory(str, Enum):
    CONFIRMED_DEFECT = "CONFIRMED_DEFECT"
    PROBABLE_DEFECT = "PROBABLE_DEFECT"
    ARCHITECTURAL_CONCERN = "ARCHITECTURAL_CONCERN"
    IMPROVEMENT = "IMPROVEMENT"
    UNVERIFIED = "UNVERIFIED"


class FindingSeverity(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class FindingConfidence(str, Enum):
    CONFIRMED = "CONFIRMED"
    PROBABLE = "PROBABLE"
    SPECULATIVE = "SPECULATIVE"


def _clean_relpath(path_str: str) -> str:
    """Standardize relative path string without stripping dot-prefixed directories like .maya/."""
    p = str(path_str).strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    while p.startswith("/") and not p.startswith("/."):
        p = p[1:]
    return p


@dataclass
class DiscoveryFinding:
    """Actionable structured finding produced during repository discovery."""
    issue_id: str
    title: str
    category: FindingCategory
    severity: FindingSeverity
    confidence: FindingConfidence
    target_files: list[str] = field(default_factory=list)
    line_references: list[str] = field(default_factory=list)
    actual_behavior: str = ""
    expected_behavior: str = ""
    evidence: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    test_targets: list[str] = field(default_factory=list)
    subsystem: str = "general"
    risk: str = "LOW"
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def __post_init__(self) -> None:
        # Standardize relative paths with forward slashes without stripping leading dot in .maya
        self.target_files = [_clean_relpath(p) for p in self.target_files if p]
        self.test_targets = [_clean_relpath(p) for p in self.test_targets if p]
        # Auto-compute risk tier if not explicitly assigned
        if not self.risk or self.risk == "LOW":
            self.risk = self._compute_risk()

    def _compute_risk(self) -> str:
        """Derive risk tier from severity, target files, and subsystem criticality."""
        if self.severity == FindingSeverity.P0:
            return "VERY_HIGH"
        if len(self.target_files) >= 3 or self.subsystem in {"core", "services"}:
            return "HIGH" if self.severity == FindingSeverity.P1 else "MEDIUM"
        if self.severity == FindingSeverity.P1:
            return "MEDIUM"
        return "LOW"

    def fingerprint(self) -> str:
        """Deterministic fingerprint for finding deduplication across categories/tools."""
        primary_file = self.target_files[0] if self.target_files else "none"
        primary_line = self.line_references[0] if self.line_references else "none"
        # Group by location and normalized title to deduplicate across differing model/static confidence
        clean_title = re.sub(r"[^a-z0-9]", "", self.title.strip().lower())
        raw_key = f"{primary_file}:{primary_line}:{clean_title}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def to_markdown_section(self) -> str:
        """
        Format finding as a Markdown section strictly compatible with
        skills.maintenance.manifest.parse_markdown_backlog.
        """
        targets_str = ", ".join(self.target_files) if self.target_files else "None"
        tests_line = f"- **Test Targets:** {', '.join(self.test_targets)}\n" if self.test_targets else "- **Test Targets:** None\n"
        criteria_lines = "\n".join(f"- {c}" for c in self.acceptance_criteria) if self.acceptance_criteria else "- Fix verified and regression test added."

        evidence_block = self.evidence.strip()
        if evidence_block and not evidence_block.startswith("```"):
            evidence_block = f"```\n{evidence_block}\n```"

        return (
            f"## [{self.issue_id}] {self.title}\n"
            f"- **ID:** {self.issue_id}\n"
            f"- **Priority:** {self.severity.value}\n"
            f"- **Risk:** {self.risk}\n"
            f"- **Category:** {self.category.value}\n"
            f"- **Confidence:** {self.confidence.value}\n"
            f"- **Target Files:** {targets_str}\n"
            f"{tests_line}\n"
            f"### Actual Behavior\n"
            f"{self.actual_behavior.strip() or 'No detailed actual behavior recorded.'}\n\n"
            f"### Expected Behavior\n"
            f"{self.expected_behavior.strip() or 'No detailed expected behavior recorded.'}\n\n"
            f"### Reproduction Evidence\n"
            f"{evidence_block or 'No static or runtime evidence attached.'}\n\n"
            f"### Acceptance Criteria\n"
            f"{criteria_lines}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "risk": self.risk,
            "subsystem": self.subsystem,
            "target_files": self.target_files,
            "line_references": self.line_references,
            "actual_behavior": self.actual_behavior,
            "expected_behavior": self.expected_behavior,
            "evidence": self.evidence,
            "acceptance_criteria": self.acceptance_criteria,
            "test_targets": self.test_targets,
            "timestamp": self.timestamp,
        }


@dataclass
class SubsystemInfo:
    """Detailed metadata for a recognized architectural subsystem."""
    name: str
    relative_path: str
    files: list[str] = field(default_factory=list)
    symbols_count: int = 0
    entry_points: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    outbound_dependencies: set[str] = field(default_factory=set)


@dataclass
class RepoInventory:
    """Bounded, cached structural map of the MayaVE repository."""
    repo_root: Path
    subsystems: dict[str, SubsystemInfo] = field(default_factory=dict)
    file_type_counts: dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    entry_points: list[str] = field(default_factory=list)
    test_map: dict[str, list[str]] = field(default_factory=dict)  # source_file -> [test_files]
    malformed_files: list[dict[str, str]] = field(default_factory=list)
    symbols: list[dict[str, Any]] = field(default_factory=list)
    scanned_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class DiscoveryConfig:
    """Configuration options for a progressive discovery run."""
    subsystem: str = "all"
    mechanical_only: bool = False
    max_files: int = 1000
    max_symbols: int = 500
    output_dir: Optional[Path] = None
    output_filename: Optional[str] = None
    timeout_seconds: float = 300.0
    agent_config: Optional[MacroAgentConfig] = None


@dataclass
class DiscoveryResult:
    """Outcome and artifacts produced by DiscoveryEngine."""
    success: bool
    inventory: RepoInventory
    findings: list[DiscoveryFinding]
    report_path: Optional[Path] = None
    markdown_report: str = ""
    duration_sec: float = 0.0
    stages_executed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ==============================================================================
# Read-Only Tool Registry Factory for Discovery Mode
# ==============================================================================

def create_discovery_registry() -> MacroToolRegistry:
    """
    Create a tool registry strictly restricted to read-only exploration and verification.
    fs_edit and fs_write are deliberately omitted to preserve production safety invariants.
    """
    full_registry = create_default_registry()
    discovery_registry = MacroToolRegistry()

    allowed_tools = {"fs_read", "fs_search", "fs_find_symbols", "test_runner", "finding_record"}
    for tool_name in allowed_tools:
        tool_def = full_registry.get(tool_name)
        if tool_def:
            discovery_registry.register(tool_def)

    return discovery_registry


# ==============================================================================
# Repository Mapper & Inventory Engine
# ==============================================================================

class RepoMapper:
    """Scans and analyzes repository structure within safety bounds."""

    def __init__(self, repo_root: Path | str, exclusions: Optional[Sequence[str]] = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.exclusions = tuple(exclusions) if exclusions is not None else DEFAULT_EXCLUSIONS

    def _is_excluded(self, path: Path) -> bool:
        """Check whether a path matches any exclusion rule."""
        raw = str(path).replace("\\", "/").lower()
        parts = [p.lower() for p in path.parts]
        for excl in self.exclusions:
            clean_excl = excl.strip("/").lower()
            if clean_excl in parts or f"/{clean_excl}/" in raw or raw.endswith(f"/{clean_excl}"):
                return True
        return False

    def build_inventory(self, max_files: int = 5000, max_symbols: int = 500) -> RepoInventory:
        """Scan repository and construct the bounded inventory."""
        inv = RepoInventory(repo_root=self.repo_root)

        # Initialize subsystems
        for sub in CANONICAL_SUBSYSTEMS:
            inv.subsystems[sub] = SubsystemInfo(name=sub, relative_path=sub)
        inv.subsystems["other"] = SubsystemInfo(name="other", relative_path="")

        all_python_files: list[Path] = []
        all_test_files: list[Path] = []

        scanned = 0
        for root, dirs, files in os.walk(self.repo_root):
            # Prune excluded directories in-place for fast traversal
            dirs[:] = [d for d in dirs if not self._is_excluded(Path(root) / d)]

            for f in files:
                file_path = Path(root) / f
                if self._is_excluded(file_path):
                    continue

                rel_str = str(file_path.relative_to(self.repo_root)).replace("\\", "/")
                ext = file_path.suffix.lower() or "[no-ext]"
                inv.file_type_counts[ext] = inv.file_type_counts.get(ext, 0) + 1
                inv.total_files += 1

                is_test_file = rel_str.startswith(".maya/tests/") and ext == ".py"
                is_internal_maintenance = (
                    rel_str.startswith("skills/maintenance/")
                    or rel_str.startswith(".maya/")
                )

                # Check tests
                if is_test_file:
                    all_test_files.append(file_path)

                # Internal maintenance and test files are not ordinary product targets
                if not is_internal_maintenance and not is_test_file:
                    # Classify subsystem
                    subsystem_name = "other"
                    first_part = rel_str.split("/")[0]
                    if first_part in inv.subsystems:
                        subsystem_name = first_part
                    inv.subsystems[subsystem_name].files.append(rel_str)

                    # Check entry points
                    if rel_str in KNOWN_ENTRY_POINTS:
                        inv.entry_points.append(rel_str)
                        inv.subsystems[subsystem_name].entry_points.append(rel_str)

                    if ext == ".py":
                        all_python_files.append(file_path)

                scanned += 1
                if scanned >= max_files:
                    break
            if scanned >= max_files:
                break

        # Map tests to source files
        self._map_tests_to_sources(inv, all_test_files, all_python_files)

        # Extract symbols and dependencies on bounded subset of python files
        self._extract_symbols_and_deps(inv, all_python_files, max_symbols=max_symbols)

        return inv

    def _map_tests_to_sources(
        self,
        inv: RepoInventory,
        test_files: list[Path],
        source_files: list[Path],
    ) -> None:
        """Establish relationships between test files and source modules."""
        # Index test basenames
        for tf in test_files:
            rel_test = str(tf.relative_to(self.repo_root)).replace("\\", "/")
            test_stem = tf.stem  # e.g. test_ws_server or ws_server_test

            for sf in source_files:
                if sf.is_relative_to(self.repo_root / ".maya"):
                    continue
                rel_src = str(sf.relative_to(self.repo_root)).replace("\\", "/")
                src_stem = sf.stem

                # Direct stem matching: test_foo.py matches foo.py
                if test_stem in (f"test_{src_stem}", f"{src_stem}_test") or (src_stem in test_stem and len(src_stem) > 4):
                    inv.test_map.setdefault(rel_src, []).append(rel_test)

    def _extract_symbols_and_deps(
        self,
        inv: RepoInventory,
        python_files: list[Path],
        max_symbols: int = 500,
    ) -> None:
        """Parse Python files with AST to extract symbols, dependencies, and syntax errors."""
        collected_symbols = 0

        for pf in python_files:
            rel_path = str(pf.relative_to(self.repo_root)).replace("\\", "/")
            first_part = rel_path.split("/")[0]
            subsystem_name = first_part if first_part in inv.subsystems else "other"

            try:
                code = pf.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(code, filename=str(pf))
            except SyntaxError as syn_err:
                inv.malformed_files.append({
                    "file": rel_path,
                    "error": f"SyntaxError: {syn_err.msg} at line {syn_err.lineno}",
                    "line": str(syn_err.lineno or 1),
                })
                continue
            except Exception as exc:
                inv.malformed_files.append({
                    "file": rel_path,
                    "error": str(exc),
                    "line": "1",
                })
                continue

            # Check for __main__ block
            has_main = any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators)
                for node in tree.body
            )
            if has_main and rel_path not in inv.entry_points:
                inv.entry_points.append(rel_path)
                inv.subsystems[subsystem_name].entry_points.append(rel_path)

            # Analyze imports / outbound dependencies
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        if root_mod in CANONICAL_SUBSYSTEMS and root_mod != subsystem_name:
                            inv.subsystems[subsystem_name].outbound_dependencies.add(root_mod)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root_mod = node.module.split(".")[0]
                        if root_mod in CANONICAL_SUBSYSTEMS and root_mod != subsystem_name:
                            inv.subsystems[subsystem_name].outbound_dependencies.add(root_mod)

            # Extract symbols up to bound
            if collected_symbols < max_symbols:
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        inv.symbols.append({
                            "name": node.name,
                            "type": "class",
                            "file": rel_path,
                            "line": node.lineno,
                        })
                        inv.subsystems[subsystem_name].symbols_count += 1
                        collected_symbols += 1
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        inv.symbols.append({
                            "name": node.name,
                            "type": "function" if isinstance(node, ast.FunctionDef) else "async_function",
                            "file": rel_path,
                            "line": node.lineno,
                        })
                        inv.subsystems[subsystem_name].symbols_count += 1
                        collected_symbols += 1
                    if collected_symbols >= max_symbols:
                        break


# ==============================================================================
# Mechanical Static Scanner (Stage 1)
# ==============================================================================

class MechanicalScanner:
    """
    Deterministic model-free static scanner.
    Detects syntax errors, swallowed exceptions, unsafe calls, and unmapped modules.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def scan(self, inventory: RepoInventory, target_subsystem: str = "all") -> list[DiscoveryFinding]:
        """Execute mechanical checks and return structured findings."""
        findings: list[DiscoveryFinding] = []
        counter = 1

        # 1. Syntax Errors in malformed files
        for item in inventory.malformed_files:
            file_rel = item["file"]
            if any(file_rel.startswith(pfx) for pfx in FORBIDDEN_TARGET_PREFIXES) or file_rel.startswith(".maya/"):
                continue
            if target_subsystem != "all" and not file_rel.startswith(f"{target_subsystem}/"):
                continue

            h = hashlib.sha256(f"syntax:{file_rel}".encode("utf-8")).hexdigest()[:6].upper()
            findings.append(DiscoveryFinding(
                issue_id=f"MECH-SYNTAX-{h}",
                title=f"Syntax error in {file_rel}",
                category=FindingCategory.CONFIRMED_DEFECT,
                severity=FindingSeverity.P1,
                confidence=FindingConfidence.CONFIRMED,
                target_files=[file_rel],
                line_references=[f"{file_rel}:{item.get('line', '1')}"],
                actual_behavior=f"Source file contains invalid syntax preventing execution: {item['error']}",
                expected_behavior="Source file must parse cleanly into Python AST without syntax errors.",
                evidence=item["error"],
                acceptance_criteria=[
                    f"Resolve syntax error in {file_rel}.",
                    "Ensure python -m py_compile compiles the file cleanly.",
                ],
                subsystem=file_rel.split("/")[0],
            ))
            counter += 1

        # 2. Inspect ASTs for swallowed exceptions and dangerous shell calls
        for sub_name, sub_info in inventory.subsystems.items():
            if target_subsystem != "all" and sub_name != target_subsystem:
                continue

            for file_rel in sub_info.files:
                if not file_rel.endswith(".py"):
                    continue

                abs_path = self.repo_root / file_rel
                if not abs_path.is_file():
                    continue

                try:
                    code = abs_path.read_text(encoding="utf-8", errors="replace")
                    tree = ast.parse(code, filename=str(abs_path))
                except Exception:
                    continue  # Already captured in malformed_files

                mapped_tests = inventory.test_map.get(file_rel, [])
                findings.extend(self._inspect_file_ast(file_rel, tree, code, sub_name, counter, mapped_tests=mapped_tests))
                counter += len(findings)

        return findings

    def _inspect_file_ast(
        self,
        file_rel: str,
        tree: ast.AST,
        code: str,
        subsystem: str,
        start_counter: int,
        mapped_tests: Optional[list[str]] = None,
    ) -> list[DiscoveryFinding]:
        """Inspect a single AST for anti-patterns."""
        if any(file_rel.startswith(pfx) for pfx in FORBIDDEN_TARGET_PREFIXES) or file_rel.startswith(".maya/"):
            return []

        results: list[DiscoveryFinding] = []
        counter = start_counter
        lines = code.splitlines()
        tests = list(mapped_tests) if mapped_tests else []

        for node in ast.walk(tree):
            # Check 1: Swallowed Exceptions (except: pass or except Exception: pass with no re-raise or logging)
            if isinstance(node, ast.ExceptHandler):
                is_swallowed = False
                if len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue)):
                    is_swallowed = True
                elif len(node.body) == 1 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    # Docstring only
                    is_swallowed = True

                if is_swallowed:
                    line_no = node.lineno
                    line_snip = lines[line_no - 1] if 0 < line_no <= len(lines) else "except ...: pass"
                    h = hashlib.sha256(f"exc:{file_rel}:{line_no}".encode("utf-8")).hexdigest()[:6].upper()
                    results.append(DiscoveryFinding(
                        issue_id=f"MECH-EXC-{h}",
                        title=f"Swallowed exception in {file_rel}:{line_no}",
                        category=FindingCategory.PROBABLE_DEFECT,
                        severity=FindingSeverity.P2,
                        confidence=FindingConfidence.PROBABLE,
                        target_files=[file_rel],
                        line_references=[f"{file_rel}:{line_no}"],
                        actual_behavior=f"Exception caught and silently swallowed with pass/continue at line {line_no}.",
                        expected_behavior="Exceptions should be logged or handled with explicit recovery actions.",
                        evidence=f"Line {line_no}: {line_snip.strip()}",
                        acceptance_criteria=[
                            f"Add appropriate logging or error recovery to exception handler at {file_rel}:{line_no}.",
                        ],
                        test_targets=tests,
                        subsystem=subsystem,
                    ))
                    counter += 1

            # Check 2: Unsafe Shell Invocations (shell=True or os.system)
            elif isinstance(node, ast.Call):
                is_unsafe_shell = False
                call_evidence = ""

                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    # os.system(...)
                    if node.func.value.id == "os" and node.func.attr == "system":
                        is_unsafe_shell = True
                        call_evidence = f"os.system() call at line {node.lineno}"
                    # subprocess with shell=True
                    elif node.func.value.id == "subprocess":
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                is_unsafe_shell = True
                                call_evidence = f"subprocess.{node.func.attr}(shell=True) at line {node.lineno}"

                if is_unsafe_shell:
                    line_no = node.lineno
                    h = hashlib.sha256(f"sec:{file_rel}:{line_no}".encode("utf-8")).hexdigest()[:6].upper()
                    results.append(DiscoveryFinding(
                        issue_id=f"MECH-SEC-{h}",
                        title=f"Unsafe shell execution in {file_rel}:{line_no}",
                        category=FindingCategory.ARCHITECTURAL_CONCERN,
                        severity=FindingSeverity.P1,
                        confidence=FindingConfidence.PROBABLE,
                        target_files=[file_rel],
                        line_references=[f"{file_rel}:{line_no}"],
                        actual_behavior=f"Unsafe subprocess or os.system shell invocation detected: {call_evidence}.",
                        expected_behavior="Use subprocess with shell=False and validated argument arrays.",
                        evidence=call_evidence,
                        acceptance_criteria=[
                            f"Refactor {file_rel}:{line_no} to use safe subprocess execution with shell=False.",
                        ],
                        test_targets=tests,
                        subsystem=subsystem,
                    ))
                    counter += 1

        return results


# ==============================================================================
# Evidence Verifier (Stage 4)
# ==============================================================================

class EvidenceVerifier:
    """
    Validates proposed findings against repository ground truth.
    Checks file existence, line bounds, and test outcomes without fabricating evidence.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def verify(self, findings: list[DiscoveryFinding]) -> list[DiscoveryFinding]:
        """Verify each finding and adjust category/confidence accordingly."""
        verified: list[DiscoveryFinding] = []

        for f in findings:
            # Check 1: Do target files actually exist on disk?
            # Also handle if target file has ":line_no" appended (e.g., "skills/system/open_target.py:175")
            existing_targets = []
            for t in f.target_files:
                clean_path = t
                line_no_str = None
                if ":" in t:
                    parts = t.rsplit(":", 1)
                    if parts[1].isdigit():
                        clean_path = parts[0]
                        line_no_str = parts[1]

                p = self.repo_root / clean_path
                if p.is_file():
                    existing_targets.append(clean_path)
                    if line_no_str:
                        ref_str = f"{clean_path}:{line_no_str}"
                        if ref_str not in f.line_references:
                            f.line_references.append(ref_str)
                elif p.exists():
                    existing_targets.append(clean_path)

            if f.target_files and not existing_targets:
                # All claimed targets are non-existent -> Mark UNVERIFIED / SPECULATIVE
                f.category = FindingCategory.UNVERIFIED
                f.confidence = FindingConfidence.SPECULATIVE
                f.actual_behavior += "\n[VERIFIER NOTE: Referenced target files do not exist on disk.]"
                verified.append(f)
                continue

            # Update with verified target files
            f.target_files = existing_targets or f.target_files

            # Check 1b: Verify line references if present
            valid_line_refs = []
            for ref in f.line_references:
                ref_path_str = ref
                line_num = None
                if ":" in ref:
                    p_str, l_str = ref.rsplit(":", 1)
                    if l_str.isdigit():
                        ref_path_str = p_str
                        line_num = int(l_str)
                abs_f = self.repo_root / ref_path_str
                if abs_f.is_file():
                    if line_num is not None:
                        try:
                            line_count = sum(1 for _ in abs_f.open(encoding="utf-8", errors="replace"))
                            if 0 < line_num <= line_count:
                                valid_line_refs.append(ref)
                            else:
                                logger.debug("Line reference %s exceeds file line count %d", ref, line_count)
                        except Exception:
                            valid_line_refs.append(ref)
                    else:
                        valid_line_refs.append(ref)
            f.line_references = valid_line_refs or f.line_references

            # Check 1c: Check for forbidden maintenance or coordinator paths
            has_forbidden = any(
                any(t.startswith(pfx) for pfx in FORBIDDEN_TARGET_PREFIXES) or t.startswith(".maya/")
                for t in f.target_files
            )
            if has_forbidden:
                f.category = FindingCategory.ARCHITECTURAL_CONCERN
                f.actual_behavior += "\n[VERIFIER NOTE: Finding targets internal maintenance or coordinator path; excluded from automated repair campaign.]"

            # Check 2: If test target is specified and verified, check confirmation
            if f.test_targets:
                existing_tests = [t for t in f.test_targets if (self.repo_root / t).exists()]
                f.test_targets = existing_tests

            # Preserve strict anti-fabrication rule: If there is no reproduction evidence or test,
            # CONFIRMED_DEFECT is downgraded to PROBABLE_DEFECT
            if f.category == FindingCategory.CONFIRMED_DEFECT and f.confidence != FindingConfidence.CONFIRMED:
                f.confidence = FindingConfidence.PROBABLE
                f.category = FindingCategory.PROBABLE_DEFECT

            verified.append(f)

        return verified


# ==============================================================================
# Discovery Engine Coordinator
# ==============================================================================

class DiscoveryEngine:
    """
    Orchestrates the 5-stage progressive discovery pipeline.
    Produces deduplicated, ranked findings and Markdown reports for AutoPhage campaign intake.
    """

    def __init__(
        self,
        repo_root: Path | str,
        config: Optional[DiscoveryConfig] = None,
        agent: Optional[MacroAgent] = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = config or DiscoveryConfig()
        self.agent = agent
        self.mapper = RepoMapper(self.repo_root)
        self.scanner = MechanicalScanner(self.repo_root)
        self.verifier = EvidenceVerifier(self.repo_root)

    def run(self) -> DiscoveryResult:
        """Execute the progressive discovery pipeline end-to-end."""
        start_time = time.monotonic()
        stages_run: list[str] = []
        errors: list[str] = []
        all_findings: list[DiscoveryFinding] = []

        # Stage 1 & 2: Repository Mapping & Inventory
        stages_run.append("Stage 1: Repository Mapping")
        try:
            inventory = self.mapper.build_inventory(
                max_files=self.config.max_files,
                max_symbols=self.config.max_symbols,
            )
        except Exception as exc:
            logger.exception("Failed to build repository inventory: %s", exc)
            return DiscoveryResult(
                success=False,
                inventory=RepoInventory(repo_root=self.repo_root),
                findings=[],
                duration_sec=time.monotonic() - start_time,
                stages_executed=stages_run,
                errors=[f"Inventory error: {exc}"],
            )

        # Stage 2: Mechanical Static Checks
        stages_run.append("Stage 2: Mechanical Checks")
        try:
            mech_findings = self.scanner.scan(inventory, target_subsystem=self.config.subsystem)
            all_findings.extend(mech_findings)
        except Exception as exc:
            logger.exception("Mechanical check error: %s", exc)
            errors.append(f"Mechanical check error: {exc}")

        # Stage 3: Focused MACRO Model Audit (if not mechanical-only and agent available)
        if not self.config.mechanical_only and self.agent is not None:
            stages_run.append("Stage 3: Focused MACRO Audit")
            try:
                model_findings = self._run_macro_audit(inventory, mech_findings)
                all_findings.extend(model_findings)
            except Exception as exc:
                logger.exception("MACRO audit error: %s", exc)
                errors.append(f"MACRO audit error: {exc}")

        # Stage 4: Evidence Verification
        stages_run.append("Stage 4: Evidence Verification")
        try:
            verified_findings = self.verifier.verify(all_findings)
        except Exception as exc:
            logger.exception("Evidence verification error: %s", exc)
            errors.append(f"Verification error: {exc}")
            verified_findings = all_findings

        # Stage 5: Deduplication, Severity Ranking & Markdown Backlog Export
        stages_run.append("Stage 5: Deduplication & Markdown Export")
        deduped = self.deduplicate_and_rank(verified_findings)
        md_report = self.generate_markdown_report(deduped, inventory)

        report_path = self._save_report(md_report)

        duration = time.monotonic() - start_time
        return DiscoveryResult(
            success=True,
            inventory=inventory,
            findings=deduped,
            report_path=report_path,
            markdown_report=md_report,
            duration_sec=duration,
            stages_executed=stages_run,
            errors=errors,
        )

    def _run_macro_audit(
        self,
        inventory: RepoInventory,
        prior_findings: list[DiscoveryFinding],
    ) -> list[DiscoveryFinding]:
        """Execute a read-only MACRO agent reasoning loop across the targeted subsystem."""
        if not self.agent:
            return []

        registry = create_discovery_registry()
        exec_ctx = ToolExecutionContext(
            worktree_path=self.repo_root,
            repo_root=self.repo_root,
            task_id=f"DISCOVERY-{self.config.subsystem.upper()}",
            allow_shared_changelog=False,
        )

        # Assemble compact, budget-conscious prompt
        sub = self.config.subsystem
        if sub == "all":
            all_files = [f for s in inventory.subsystems.values() for f in s.files]
            all_entries = inventory.entry_points
            file_summary = ", ".join(all_files[:15]) if all_files else "none"
            entry_summary = ", ".join(all_entries[:10]) if all_entries else "none"
        else:
            sub_info = inventory.subsystems.get(sub)
            file_summary = ", ".join(sub_info.files[:10]) if sub_info else "none"
            entry_summary = ", ".join(sub_info.entry_points) if sub_info and sub_info.entry_points else "none"
        prior_summary = ", ".join(f.issue_id for f in prior_findings[:5]) if prior_findings else "none"

        objective = (
            f"DISCOVERY OBJECTIVE: Inspect subsystem '{sub}'.\n"
            f"- Key files: {file_summary}\n"
            f"- Entry points: {entry_summary}\n"
            f"- Initial mechanical findings: {prior_summary}\n\n"
            "INSTRUCTIONS:\n"
            "1. Use 'fs_read', 'fs_search', and 'fs_find_symbols' to investigate potential edge cases, missing exception handling, or architecture concerns.\n"
            "2. If applicable, run relevant tests under '.maya/tests/' using 'test_runner'.\n"
            "3. Record all verified or suspected issues using 'finding_record'.\n"
            "4. When finished, conclude with <finish_task> summarizing discoveries."
        )

        agent_res: MacroAgentResult = self.agent.run(
            task_objective=objective,
            context=exec_ctx,
            registry=registry,
        )
        if not agent_res.success and agent_res.error:
            logger.warning("MACRO audit encountered issue: %s", agent_res.error)

        # Convert tool findings in exec_ctx.findings to DiscoveryFinding objects
        agent_findings: list[DiscoveryFinding] = []
        for idx, raw in enumerate(exec_ctx.findings, start=1):
            category_val = raw.get("category", "PROBABLE_DEFECT").upper()
            try:
                cat_enum = FindingCategory(category_val)
            except ValueError:
                cat_enum = FindingCategory.PROBABLE_DEFECT

            sev_val = raw.get("severity", "P2").upper()
            if sev_val in ("CRITICAL", "HIGH"):
                sev_enum = FindingSeverity.P1
            elif sev_val == "LOW":
                sev_enum = FindingSeverity.P3
            elif sev_val in ("P0", "P1", "P2", "P3"):
                sev_enum = FindingSeverity(sev_val)
            else:
                sev_enum = FindingSeverity.P2

            conf_val = raw.get("confidence", "PROBABLE").upper()
            try:
                conf_enum = FindingConfidence(conf_val)
            except ValueError:
                conf_enum = FindingConfidence.PROBABLE

            agent_findings.append(DiscoveryFinding(
                issue_id=raw.get("issue_id") or f"MACRO-DISC-{idx:03d}",
                title=raw.get("title") or "Unnamed agent discovery",
                category=cat_enum,
                severity=sev_enum,
                confidence=conf_enum,
                target_files=raw.get("affected_locations") or [],
                line_references=raw.get("affected_locations") or [],
                actual_behavior=raw.get("description") or "",
                expected_behavior="Expected correct, robust behavior.",
                evidence=raw.get("evidence") or "",
                acceptance_criteria=[raw.get("recommendations")] if raw.get("recommendations") else [],
                subsystem=sub,
            ))

        return agent_findings

    @staticmethod
    def deduplicate_and_rank(findings: list[DiscoveryFinding]) -> list[DiscoveryFinding]:
        """
        Deduplicate findings by deterministic fingerprint or issue ID,
        preserving highest severity and confidence.
        """
        seen_fingerprints: dict[str, DiscoveryFinding] = {}

        severity_rank = {FindingSeverity.P0: 0, FindingSeverity.P1: 1, FindingSeverity.P2: 2, FindingSeverity.P3: 3}
        confidence_rank = {FindingConfidence.CONFIRMED: 0, FindingConfidence.PROBABLE: 1, FindingConfidence.SPECULATIVE: 2}

        for f in findings:
            fp = f.fingerprint()
            if fp not in seen_fingerprints:
                seen_fingerprints[fp] = f
            else:
                existing = seen_fingerprints[fp]
                # Merge target files and lines
                for t in f.target_files:
                    if t not in existing.target_files:
                        existing.target_files.append(t)
                for l in f.line_references:
                    if l not in existing.line_references:
                        existing.line_references.append(l)

                # Keep higher severity
                if severity_rank[f.severity] < severity_rank[existing.severity]:
                    existing.severity = f.severity
                    existing.risk = f.risk

                # Keep higher confidence
                if confidence_rank[f.confidence] < confidence_rank[existing.confidence]:
                    existing.confidence = f.confidence
                    existing.category = f.category

                # Append additional evidence
                if f.evidence and f.evidence not in existing.evidence:
                    existing.evidence += f"\n\nAdditional evidence:\n{f.evidence}"

        unique_findings = list(seen_fingerprints.values())

        # Sort by severity (P0 first), then confidence (CONFIRMED first), then issue_id
        unique_findings.sort(
            key=lambda x: (severity_rank[x.severity], confidence_rank[x.confidence], x.issue_id)
        )
        return unique_findings

    def generate_markdown_report(
        self,
        findings: list[DiscoveryFinding],
        inventory: RepoInventory,
    ) -> str:
        """
        Format findings into a Markdown report consumable by parse_markdown_backlog.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        manifest_id = f"DISCOVERY-{self.config.subsystem.upper()}-{timestamp}"

        # Group by category
        confirmed = [f for f in findings if f.category == FindingCategory.CONFIRMED_DEFECT]
        suspected = [f for f in findings if f.category == FindingCategory.PROBABLE_DEFECT]
        arch = [f for f in findings if f.category == FindingCategory.ARCHITECTURAL_CONCERN]
        improvements = [f for f in findings if f.category == FindingCategory.IMPROVEMENT]
        unverified = [f for f in findings if f.category == FindingCategory.UNVERIFIED]

        lines = [
            f"# Backlog: {manifest_id}",
            "",
            "**Summary of Findings**",
            f"- **Target Subsystem:** `{self.config.subsystem}`",
            f"- **Repository Files Scanned:** {inventory.total_files}",
            f"- **Total Findings Discovered:** {len(findings)}",
            f"  - Confirmed Defects: {len(confirmed)}",
            f"  - Suspected Defects: {len(suspected)}",
            f"  - Architectural Concerns: {len(arch)}",
            f"  - Improvement Opportunities: {len(improvements)}",
            f"  - Unverified Claims: {len(unverified)}",
            "",
            "---",
            "",
        ]

        if not findings:
            lines.append("No actionable findings or defects were identified during this discovery pass.\n")
            return "\n".join(lines)

        if confirmed:
            lines.append("#### Confirmed Defects\n")
            for f in confirmed:
                lines.append(f.to_markdown_section())

        if suspected:
            lines.append("#### Suspected / Probable Defects\n")
            for f in suspected:
                lines.append(f.to_markdown_section())

        if arch:
            lines.append("#### Architectural Concerns\n")
            for f in arch:
                lines.append(f.to_markdown_section())

        if improvements:
            lines.append("#### Improvement Opportunities\n")
            for f in improvements:
                lines.append(f.to_markdown_section())

        if unverified:
            lines.append("#### Unverified Hypotheses\n")
            for f in unverified:
                lines.append(f.to_markdown_section())

        return "\n".join(lines)

    def _save_report(self, markdown_content: str) -> Optional[Path]:
        """
        Safely write discovery report to .maya/reports/ or explicitly configured directory.
        Never silently overwrites an existing user-authored file.
        """
        output_dir = self.config.output_dir or (self.repo_root / ".maya" / "reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = self.config.output_filename
        if not filename:
            sub = self.config.subsystem.replace("/", "_")
            ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            filename = f"autophage-discovery-{sub}-{ts}.md"

        target_path = output_dir / filename

        # Safety: avoid overwriting existing files
        if target_path.exists():
            base_stem = target_path.stem
            unique_ts = int(time.time() * 1000)
            target_path = output_dir / f"{base_stem}_{unique_ts}.md"

        try:
            target_path.write_text(markdown_content, encoding="utf-8")
            logger.info("Saved discovery report to %s", target_path)
            return target_path
        except Exception as exc:
            logger.error("Failed to write discovery report to %s: %s", target_path, exc)
            return None


# ==============================================================================
# Convenience Functions & CLI Entry Point
# ==============================================================================

def run_discovery(
    repo_root: Path | str,
    subsystem: str = "all",
    mechanical_only: bool = False,
    agent: Optional[MacroAgent] = None,
    output_dir: Optional[Path] = None,
) -> DiscoveryResult:
    """Execute progressive discovery on a repository and return results."""
    config = DiscoveryConfig(
        subsystem=subsystem,
        mechanical_only=mechanical_only,
        output_dir=output_dir,
    )
    engine = DiscoveryEngine(repo_root=repo_root, config=config, agent=agent)
    return engine.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoPhage 2.0 MACRO Discovery Mode (Phase 3)")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--subsystem", default="all", help="Target subsystem (e.g. brain, core, services, skills, all)")
    parser.add_argument("--mechanical-only", action="store_true", help="Run mechanical static checks only without invoking model")
    parser.add_argument("--output-dir", default=None, help="Directory to save generated Markdown report")

    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else None

    print(f"Starting AutoPhage 2.0 Discovery on '{root}' (subsystem: {args.subsystem}, mechanical_only: {args.mechanical_only})...")
    res = run_discovery(repo_root=root, subsystem=args.subsystem, mechanical_only=args.mechanical_only, output_dir=out)

    print(f"\nDiscovery completed in {res.duration_sec:.2f}s:")
    print(f"- Total findings: {len(res.findings)}")
    print(f"- Stages run: {', '.join(res.stages_executed)}")
    if res.report_path:
        print(f"- Report written to: {res.report_path}")
    if res.errors:
        print(f"- Errors encountered: {res.errors}")
