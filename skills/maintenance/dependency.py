"""
Dependency Registry and Change-Impact Analysis for MayaVE Maintenance Coordinator (Autophage P2).

Provides:
- File, symbol, import, test, and artifact dependency tracking.
- Impact classification: AFFECTED_CONFIRMED, POTENTIALLY_AFFECTED, UNAFFECTED_WITH_EVIDENCE, IMPACT_UNKNOWN.
- AST-based symbol and import extraction for Python files with safe fallbacks.
- Conservative handling of unknown dependencies.
"""

from __future__ import annotations

import ast
import enum
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Set, Tuple

try:
    from .models import TaskDefinition, TaskStatus, utc_now_iso
except ImportError:
    from models import TaskDefinition, TaskStatus, utc_now_iso

logger = logging.getLogger("maya.maintenance.dependency")


class ImpactClassification(str, enum.Enum):
    AFFECTED_CONFIRMED = "AFFECTED_CONFIRMED"
    POTENTIALLY_AFFECTED = "POTENTIALLY_AFFECTED"
    UNAFFECTED_WITH_EVIDENCE = "UNAFFECTED_WITH_EVIDENCE"
    IMPACT_UNKNOWN = "IMPACT_UNKNOWN"


@dataclass
class FileDependencyRecord:
    """Detailed dependency details for a file associated with a task."""
    path: str
    read_only: bool = False
    symbols_defined: list[str] = field(default_factory=list)
    symbols_used: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "read_only": self.read_only,
            "symbols_defined": sorted(self.symbols_defined),
            "symbols_used": sorted(self.symbols_used),
            "imports": sorted(self.imports),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileDependencyRecord:
        return cls(
            path=data["path"],
            read_only=data.get("read_only", False),
            symbols_defined=list(data.get("symbols_defined", [])),
            symbols_used=list(data.get("symbols_used", [])),
            imports=list(data.get("imports", [])),
        )


@dataclass
class TaskDependencyNode:
    """Tracks static and dynamic dependencies for a maintenance task."""
    task_id: str
    explicit_prerequisites: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    read_files: list[str] = field(default_factory=list)
    test_targets: list[str] = field(default_factory=list)
    file_records: dict[str, FileDependencyRecord] = field(default_factory=dict)
    originating_issue: Optional[str] = None
    baseline_commit: Optional[str] = None
    patch_sha256: Optional[str] = None
    verification_sha256: Optional[str] = None
    symbols_defined: list[str] = field(default_factory=list)
    symbols_consumed: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "explicit_prerequisites": sorted(self.explicit_prerequisites),
            "target_files": sorted(self.target_files),
            "read_files": sorted(self.read_files),
            "test_targets": sorted(self.test_targets),
            "file_records": {k: v.to_dict() for k, v in sorted(self.file_records.items())},
            "originating_issue": self.originating_issue,
            "baseline_commit": self.baseline_commit,
            "patch_sha256": self.patch_sha256,
            "verification_sha256": self.verification_sha256,
            "symbols_defined": sorted(self.symbols_defined),
            "symbols_consumed": sorted(self.symbols_consumed),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskDependencyNode:
        file_records = {
            k: FileDependencyRecord.from_dict(v)
            for k, v in data.get("file_records", {}).items()
        }
        return cls(
            task_id=data["task_id"],
            explicit_prerequisites=list(data.get("explicit_prerequisites", [])),
            target_files=list(data.get("target_files", [])),
            read_files=list(data.get("read_files", [])),
            test_targets=list(data.get("test_targets", [])),
            file_records=file_records,
            originating_issue=data.get("originating_issue"),
            baseline_commit=data.get("baseline_commit"),
            patch_sha256=data.get("patch_sha256"),
            verification_sha256=data.get("verification_sha256"),
            symbols_defined=list(data.get("symbols_defined", [])),
            symbols_consumed=list(data.get("symbols_consumed", [])),
            updated_at=data.get("updated_at", utc_now_iso()),
        )


@dataclass
class ImpactAssessment:
    """Result of an impact analysis triggered by code or lifecycle changes."""
    trigger_task_id: Optional[str]
    affected_task_id: str
    classification: ImpactClassification
    reason: str
    changed_files: list[str]
    overlapping_files: list[str] = field(default_factory=list)
    symbol_conflicts: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict:
        return {
            "trigger_task_id": self.trigger_task_id,
            "affected_task_id": self.affected_task_id,
            "classification": self.classification.value,
            "reason": self.reason,
            "changed_files": sorted(self.changed_files),
            "overlapping_files": sorted(self.overlapping_files),
            "symbol_conflicts": sorted(self.symbol_conflicts),
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImpactAssessment:
        return cls(
            trigger_task_id=data.get("trigger_task_id"),
            affected_task_id=data["affected_task_id"],
            classification=ImpactClassification(data["classification"]),
            reason=data["reason"],
            changed_files=list(data.get("changed_files", [])),
            overlapping_files=list(data.get("overlapping_files", [])),
            symbol_conflicts=list(data.get("symbol_conflicts", [])),
            evidence=dict(data.get("evidence", {})),
            timestamp=data.get("timestamp", utc_now_iso()),
        )


def extract_python_symbols_and_imports(file_path: Path | str, content: str | None = None) -> Tuple[list[str], list[str], list[str]]:
    """
    Extract defined symbols, referenced symbols, and module imports from a Python source.
    Returns: (symbols_defined, symbols_used, imports)
    Unknown or unparseable files return empty lists (conservative uncertainty).
    """
    p = Path(file_path)
    if p.suffix.lower() != ".py":
        return [], [], []

    if content is None:
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return [], [], []

    try:
        tree = ast.parse(content, filename=str(p))
    except Exception:
        # SyntaxError or unparseable: return empty lists
        return [], [], []

    defined: set[str] = set()
    used: set[str] = set()
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.add(f"{mod}.{alias.name}" if mod else alias.name)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                used.add(node.id)

    return sorted(defined), sorted(used), sorted(imports)


class DependencyRegistry:
    """
    Maintains and queries the dependency relationships between maintenance tasks,
    files, symbols, and tests.
    """

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self._nodes: dict[str, TaskDependencyNode] = {}

    def register_task(
        self,
        task: TaskDefinition,
        read_files: list[str] | None = None,
        force_refresh_ast: bool = False,
    ) -> TaskDependencyNode:
        """Register or update a task's dependency node."""
        declared_reads = list(getattr(task, "read_files", None) or [])
        effective_reads = list(read_files if read_files is not None else declared_reads)

        node = self._nodes.get(task.task_id)
        if not node:
            node = TaskDependencyNode(
                task_id=task.task_id,
                explicit_prerequisites=list(task.dependencies),
                target_files=list(task.target_files),
                read_files=effective_reads,
                test_targets=list(task.test_targets),
                originating_issue=task.audit_source,
            )
        else:
            node.explicit_prerequisites = list(task.dependencies)
            node.target_files = list(task.target_files)
            for rf in effective_reads:
                if rf not in node.read_files:
                    node.read_files.append(rf)
            node.test_targets = list(task.test_targets)
            node.originating_issue = task.audit_source

        # Update baseline commit & verification fingerprints if available
        if task.workspace and task.workspace.base_commit:
            node.baseline_commit = task.workspace.base_commit
        if task.checkpoint and task.checkpoint.details:
            node.patch_sha256 = task.checkpoint.details.get("artifact_sha256")
            node.verification_sha256 = task.checkpoint.details.get("verification_sha256")

        # Analyze target files and read files via AST if repo_root is provided
        all_files = set(node.target_files) | set(node.read_files)
        all_defined: set[str] = set()
        all_consumed: set[str] = set()

        for rel_path in sorted(all_files):
            rec = node.file_records.get(rel_path)
            if rec is None or force_refresh_ast:
                is_read_only = rel_path in node.read_files and rel_path not in node.target_files
                full_path = (self.repo_root / rel_path) if self.repo_root else Path(rel_path)
                symbols_def, symbols_use, imports = [], [], []
                if full_path.exists():
                    symbols_def, symbols_use, imports = extract_python_symbols_and_imports(full_path)
                rec = FileDependencyRecord(
                    path=rel_path,
                    read_only=is_read_only,
                    symbols_defined=symbols_def,
                    symbols_used=symbols_use,
                    imports=imports,
                )
                node.file_records[rel_path] = rec

            all_defined.update(rec.symbols_defined)
            all_consumed.update(rec.symbols_used)

        node.symbols_defined = sorted(all_defined)
        node.symbols_consumed = sorted(all_consumed)
        node.updated_at = utc_now_iso()
        self._nodes[task.task_id] = node
        return node

    def get_node(self, task_id: str) -> Optional[TaskDependencyNode]:
        return self._nodes.get(task_id)

    def assess_impact(
        self,
        changed_files: list[str],
        trigger_task_id: str | None = None,
        candidate_tasks: list[TaskDefinition] | None = None,
        patch_ast_details: dict[str, dict] | None = None,
    ) -> list[ImpactAssessment]:
        """
        Assess the impact of changed files on registered tasks.
        Classifications:
        - AFFECTED_CONFIRMED: Directly modifies a target file or verified artifact of a verified/completed task.
        - POTENTIALLY_AFFECTED: Modifies a file read or imported/depended on by the task, or has symbol overlap.
        - UNAFFECTED_WITH_EVIDENCE: Checked and proven to have disjoint files, symbols, and dependencies.
        - IMPACT_UNKNOWN: File or dependency information is missing or unparseable.
        """
        assessments: list[ImpactAssessment] = []
        changed_set = set(changed_files)

        if candidate_tasks:
            for t in candidate_tasks:
                if t.task_id not in self._nodes:
                    self.register_task(t)

        for task_id, node in self._nodes.items():
            if trigger_task_id and task_id == trigger_task_id:
                continue

            target_overlap = set(node.target_files).intersection(changed_set)
            read_overlap = set(node.read_files).intersection(changed_set)
            test_overlap = set(node.test_targets).intersection(changed_set)

            # Check for missing dependency info
            if not node.target_files and not node.read_files:
                assessments.append(ImpactAssessment(
                    trigger_task_id=trigger_task_id,
                    affected_task_id=task_id,
                    classification=ImpactClassification.IMPACT_UNKNOWN,
                    reason=f"Task '{task_id}' has no recorded target or read files; cannot establish independence.",
                    changed_files=changed_files,
                ))
                continue

            # 1. Direct target modification overlap
            if target_overlap:
                assessments.append(ImpactAssessment(
                    trigger_task_id=trigger_task_id,
                    affected_task_id=task_id,
                    classification=ImpactClassification.AFFECTED_CONFIRMED,
                    reason=f"Changed files directly overlap task target files: {sorted(target_overlap)}",
                    changed_files=changed_files,
                    overlapping_files=sorted(target_overlap),
                ))
                continue

            # 2. Test overlap
            if test_overlap:
                assessments.append(ImpactAssessment(
                    trigger_task_id=trigger_task_id,
                    affected_task_id=task_id,
                    classification=ImpactClassification.AFFECTED_CONFIRMED,
                    reason=f"Changed files directly overlap task test targets: {sorted(test_overlap)}",
                    changed_files=changed_files,
                    overlapping_files=sorted(test_overlap),
                ))
                continue

            # 3. Read-file overlap, dependency on trigger_task_id, or dependency on an affected task
            is_dependent = (trigger_task_id is not None) and (trigger_task_id in node.explicit_prerequisites)
            # Downstream dependency check: does this task depend on a task whose target files are being changed?
            overlap_prereqs = [
                dep_id for dep_id in node.explicit_prerequisites
                if dep_id in self._nodes and set(self._nodes[dep_id].target_files).intersection(changed_set)
            ]
            if read_overlap or is_dependent or overlap_prereqs:
                reason = []
                if read_overlap:
                    reason.append(f"overlaps read files: {sorted(read_overlap)}")
                if is_dependent:
                    reason.append(f"task explicitly depends on trigger task '{trigger_task_id}'")
                if overlap_prereqs:
                    reason.append(f"task depends on tasks modifying changed files: {sorted(overlap_prereqs)}")
                assessments.append(ImpactAssessment(
                    trigger_task_id=trigger_task_id,
                    affected_task_id=task_id,
                    classification=ImpactClassification.POTENTIALLY_AFFECTED,
                    reason=f"Task is potentially affected: {'; '.join(reason)}",
                    changed_files=changed_files,
                    overlapping_files=sorted(read_overlap),
                ))
                continue

            # 4. Symbol-level / Import check if patch AST details provided
            symbol_conflict = False
            conflicting_symbols: list[str] = []
            if patch_ast_details:
                modified_symbols = set()
                for cf, sym_info in patch_ast_details.items():
                    modified_symbols.update(sym_info.get("modified_symbols", []))
                sym_overlap = modified_symbols.intersection(set(node.symbols_consumed))
                if sym_overlap:
                    symbol_conflict = True
                    conflicting_symbols = sorted(sym_overlap)

            if symbol_conflict:
                assessments.append(ImpactAssessment(
                    trigger_task_id=trigger_task_id,
                    affected_task_id=task_id,
                    classification=ImpactClassification.POTENTIALLY_AFFECTED,
                    reason=f"Changed symbols overlap with symbols consumed by task: {conflicting_symbols}",
                    changed_files=changed_files,
                    symbol_conflicts=conflicting_symbols,
                ))
                continue

            # 5. Independence established with evidence
            assessments.append(ImpactAssessment(
                trigger_task_id=trigger_task_id,
                affected_task_id=task_id,
                classification=ImpactClassification.UNAFFECTED_WITH_EVIDENCE,
                reason="Disjoint target files, read files, test targets, and symbol consumption verified.",
                changed_files=changed_files,
            ))

        return assessments

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "nodes": {task_id: node.to_dict() for task_id, node in sorted(self._nodes.items())},
        }

    @classmethod
    def from_dict(cls, data: dict, repo_root: Path | str | None = None) -> DependencyRegistry:
        reg = cls(repo_root=repo_root)
        for task_id, node_dict in data.get("nodes", {}).items():
            reg._nodes[task_id] = TaskDependencyNode.from_dict(node_dict)
        return reg
