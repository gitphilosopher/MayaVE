"""
Structured Task Contract Schema & Strict Validation for MayaVE Autophage (P4).
Defines a versioned, machine-verifiable contract format for planner-generated tasks.
Validates scope boundaries, risk classification, test locations, and DAG integrity.
Does not introduce a runtime autonomous planning loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from .models import (
        PreValidationPattern,
        TaskDefinition,
        TaskPriority,
        TaskRisk,
        TaskStatus,
        utc_now_iso,
    )
except ImportError:
    from models import (
        PreValidationPattern,
        TaskDefinition,
        TaskPriority,
        TaskRisk,
        TaskStatus,
        utc_now_iso,
    )

CONTRACT_SCHEMA_VERSION = "1.0.0"
_TASK_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")

FORBIDDEN_TARGET_PREFIXES = (
    ".git/",
    ".maya/maintenance/",
    "skills/maintenance/",
    ".maya/tmp/",
    ".maya/reports/",
    ".maya/worktrees/",
)
FORBIDDEN_TARGET_EXACT = {
    ".env",
    "state.json",
    "cache.json",
    "approvals.json",
    "active_session.json",
    "repair_memory.jsonl",
}
ALLOWED_SUBSYSTEM_ROOTS = {
    "frontend/",
    "services/",
    "core/",
    "brain/",
    "config/",
    "docs/",
    "scripts/",
    "skills/",
}



class TaskContractValidationError(Exception):
    """Raised when a task contract violates schema or safety invariants."""
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class TaskContractValidator:
    """
    Validates task definitions against the formal Autophage contract schema.
    Enforces local-only test policy, target containment, explicit risk levels,
    and DAG cycle freedom.
    """

    @classmethod
    def validate_contract(cls, data: dict[str, Any], is_planner: bool = False) -> tuple[bool, list[str]]:
        """
        Validate a single task contract dictionary.
        Returns (is_valid, list_of_error_strings).
        """
        errors: list[str] = []

        if not isinstance(data, dict):
            return False, ["Task contract payload must be a JSON dictionary."]

        # 1. task_id
        task_id = data.get("task_id")
        if not task_id or not isinstance(task_id, str):
            errors.append("Field 'task_id' must be a non-empty string.")
        elif not _TASK_ID_REGEX.match(task_id) or len(task_id) > 64:
            errors.append(
                f"Field 'task_id' '{task_id}' is invalid. Must match '^[A-Za-z0-9_-]+$' (max 64 chars)."
            )

        # 2. title
        title = data.get("title")
        if not title or not isinstance(title, str) or len(title.strip()) < 5 or len(title) > 140:
            errors.append("Field 'title' must be a string between 5 and 140 characters.")

        # 3. audit_source
        audit_source = data.get("audit_source")
        if not audit_source or not isinstance(audit_source, str):
            errors.append("Field 'audit_source' must be a non-empty string.")

        # 4. priority
        priority = data.get("priority")
        valid_priorities = {p.value for p in TaskPriority}
        if priority not in valid_priorities:
            errors.append(f"Field 'priority' '{priority}' is invalid. Must be one of {sorted(valid_priorities)}.")

        # 5. risk (Mandatory fail-closed on None or invalid)
        risk = data.get("risk")
        valid_risks = {r.value for r in TaskRisk}
        if risk not in valid_risks:
            errors.append(f"Field 'risk' '{risk}' is invalid. Must be explicitly specified from {sorted(valid_risks)}.")
        elif is_planner and risk in {TaskRisk.HIGH.value, TaskRisk.VERY_HIGH.value}:
            errors.append(
                f"Planner is not permitted to propose '{risk}' risk tasks without explicit operator elevation."
            )

        # 6. description
        description = data.get("description")
        if not description or not isinstance(description, str) or not description.strip():
            errors.append("Field 'description' must be a non-empty string.")

        # 7. target_files
        target_files = data.get("target_files")
        if not isinstance(target_files, list) or len(target_files) == 0:
            errors.append("Field 'target_files' must be a non-empty list of relative file paths.")
        else:
            for tf in target_files:
                if not isinstance(tf, str) or not tf.strip():
                    errors.append(f"Target file entry '{tf}' is empty or not a string.")
                    continue
                norm = tf.replace("\\", "/").strip()
                if ".." in norm or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
                    errors.append(f"Target file '{tf}' violates path containment (path traversal forbidden).")
                while norm.startswith("./"):
                    norm = norm[2:]

                if is_planner:
                    if any(norm.startswith(p) for p in FORBIDDEN_TARGET_PREFIXES):
                        errors.append(f"Target file '{tf}' modifies forbidden maintenance/system path.")
                    elif norm.lower() in FORBIDDEN_TARGET_EXACT or Path(norm).name.lower() in FORBIDDEN_TARGET_EXACT:
                        errors.append(f"Target file '{tf}' targets security-sensitive or coordinator state file.")
                    elif norm.startswith(".maya/tests/"):
                        errors.append(f"Target file '{tf}' cannot target test directory; tests belong in 'test_targets'.")
                    elif not any(norm.startswith(sub) for sub in ALLOWED_SUBSYSTEM_ROOTS):
                        errors.append(f"Target file '{tf}' is outside allowed subsystems ({sorted(ALLOWED_SUBSYSTEM_ROOTS)}).")

        # 8. test_targets (Mandatory Local Test Policy)
        test_targets = data.get("test_targets", [])
        if not isinstance(test_targets, list):
            errors.append("Field 'test_targets' must be a list of file paths.")
        else:
            for tt in test_targets:
                if not isinstance(tt, str) or not tt.strip():
                    errors.append(f"Test target entry '{tt}' is empty or not a string.")
                    continue
                norm = tt.replace("\\", "/").strip()
                while norm.startswith("./"):
                    norm = norm[2:]
                # Mandatory invariant: All tests belong under .maya/tests/
                if not norm.startswith(".maya/tests/"):
                    errors.append(
                        f"Test target '{tt}' violates local-only test policy. "
                        "All tests must reside strictly under '.maya/tests/'."
                    )
                if ".." in norm:
                    errors.append(f"Test target '{tt}' contains forbidden traversal ('..').")

        # 8b. read_files (Optional)
        read_files = data.get("read_files")
        if read_files is not None:
            if not isinstance(read_files, list):
                errors.append("Field 'read_files' must be a list of file path strings.")
            else:
                for rf in read_files:
                    if not isinstance(rf, str) or not rf.strip():
                        errors.append(f"Read file entry '{rf}' is empty or not a string.")
                        continue
                    norm = rf.replace("\\", "/").strip()
                    while norm.startswith("./"):
                        norm = norm[2:]
                    if any(norm.startswith(p) for p in FORBIDDEN_TARGET_PREFIXES) or norm.lower() in FORBIDDEN_TARGET_EXACT:
                        errors.append(f"Read file '{rf}' targets forbidden path.")
                    if norm.startswith("/") or ":" in norm or ".." in norm:
                        errors.append(f"Read file '{rf}' must be relative and cannot contain traversal ('..').")

        # 9. dependencies
        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list):
            errors.append("Field 'dependencies' must be a list of task ID strings.")
        else:
            for dep in dependencies:
                if not isinstance(dep, str) or not dep.strip():
                    errors.append(f"Dependency '{dep}' must be a non-empty string.")
                elif dep == task_id:
                    errors.append(f"Task '{task_id}' cannot declare self-dependency on '{dep}'.")

        # 10. acceptance_criteria
        acceptance_criteria = data.get("acceptance_criteria")
        if not isinstance(acceptance_criteria, list) or len(acceptance_criteria) == 0:
            errors.append("Field 'acceptance_criteria' must be a non-empty list of criterion strings.")
        else:
            for ac in acceptance_criteria:
                if not isinstance(ac, str) or not ac.strip():
                    errors.append("Acceptance criterion entry must be a non-empty string.")

        # 11. pre_validation_pattern (Optional)
        pattern_data = data.get("pre_validation_pattern")
        if pattern_data is not None:
            if not isinstance(pattern_data, dict):
                errors.append("Field 'pre_validation_pattern' must be a dictionary.")
            else:
                fp = pattern_data.get("file_path")
                pat = pattern_data.get("pattern")
                if not fp or not isinstance(fp, str):
                    errors.append("pre_validation_pattern.file_path must be a non-empty string.")
                elif target_files and fp not in target_files:
                    errors.append(f"pre_validation_pattern.file_path '{fp}' must be one of target_files.")

                if not pat or not isinstance(pat, str):
                    errors.append("pre_validation_pattern.pattern must be a non-empty regex string.")
                else:
                    try:
                        re.compile(pat)
                    except re.error as re_err:
                        errors.append(f"pre_validation_pattern.pattern '{pat}' is invalid regex: {re_err}.")

        is_valid = len(errors) == 0
        return is_valid, errors

    @classmethod
    def validate_batch(cls, contracts: list[dict[str, Any]], is_planner: bool = False) -> tuple[bool, list[str]]:
        """
        Validate a list of task contracts including inter-task dependencies and cycle detection.
        When is_planner=True, enforces planner-specific constraints (risk levels, forbidden prefixes,
        state files, allowed subsystems, test target policy) across every contract in the batch.
        Returns (is_valid, list_of_error_strings).
        """
        all_errors: list[str] = []
        task_ids: set[str] = set()

        # Step 1: Validate each contract individually
        for idx, item in enumerate(contracts):
            ok, errs = cls.validate_contract(item, is_planner=is_planner)
            if not ok:
                all_errors.extend(f"Contract[{idx}]: {e}" for e in errs)
            else:
                tid = item["task_id"]
                if tid in task_ids:
                    all_errors.append(f"Duplicate task_id '{tid}' in batch.")
                task_ids.add(tid)

        if all_errors:
            return False, all_errors

        # Step 2: Cycle detection across batch dependencies
        dep_graph: dict[str, list[str]] = {
            item["task_id"]: list(item.get("dependencies", []))
            for item in contracts
        }

        # DFS cycle check
        visited: dict[str, int] = {tid: 0 for tid in dep_graph}

        def _dfs(node: str, path: list[str]) -> bool:
            visited[node] = 1
            for dep in dep_graph.get(node, []):
                if dep not in dep_graph:
                    # External or upstream dependency, acceptable if in pre-existing ledger
                    continue
                if visited.get(dep) == 1:
                    cycle = " -> ".join(path + [dep])
                    all_errors.append(f"Dependency cycle detected in batch: {cycle}")
                    return False
                if visited.get(dep) == 0:
                    if not _dfs(dep, path + [dep]):
                        return False
            visited[node] = 2
            return True

        for tid in dep_graph:
            if visited[tid] == 0:
                _dfs(tid, [tid])

        return len(all_errors) == 0, all_errors

    @classmethod
    def validate_plan(
        cls,
        plan_data: dict[str, Any] | list[dict[str, Any]],
        max_tasks: int = 5,
        known_task_ids: set[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate an end-to-end plan proposal.
        Enforces task count limits (1 <= count <= min(max_tasks, 8)), unique task IDs,
        DAG cycle freedom, dependency consistency, target containment, and strict planner safety invariants.
        """
        errors: list[str] = []

        if isinstance(plan_data, dict):
            objective = plan_data.get("objective")
            if not objective or not isinstance(objective, str) or not objective.strip():
                errors.append("Plan proposal missing valid 'objective' string.")
            tasks = plan_data.get("tasks")
            if tasks is None or not isinstance(tasks, list):
                errors.append("Plan proposal 'tasks' field must be a list of task objects.")
                return False, errors
        elif isinstance(plan_data, list):
            tasks = plan_data
        else:
            return False, ["Plan proposal payload must be a JSON dictionary or list of task objects."]

        # Task count limits: default max 5, hard limit 8
        hard_limit = 8
        configured_limit = min(max_tasks, hard_limit)

        if len(tasks) < 1:
            errors.append("Plan must contain at least 1 task.")
            return False, errors

        if len(tasks) > hard_limit:
            errors.append(f"Plan task count ({len(tasks)}) exceeds hard maximum limit of {hard_limit} tasks.")
            return False, errors

        if len(tasks) > configured_limit:
            errors.append(f"Plan task count ({len(tasks)}) exceeds configured maximum limit of {configured_limit} tasks.")
            return False, errors

        # Validate each task individually with is_planner=True
        plan_task_ids: set[str] = set()
        for idx, task_item in enumerate(tasks):
            ok, errs = cls.validate_contract(task_item, is_planner=True)
            if not ok:
                errors.extend(f"Task[{idx}]: {e}" for e in errs)
            elif isinstance(task_item, dict) and "task_id" in task_item:
                tid = task_item["task_id"]
                if tid in plan_task_ids:
                    errors.append(f"Duplicate task_id '{tid}' in plan proposal.")
                plan_task_ids.add(tid)

        if errors:
            return False, errors

        # Batch acyclicity and dependencies check with planner restrictions propagated
        ok_batch, batch_errs = cls.validate_batch(tasks, is_planner=True)
        if not ok_batch:
            errors.extend(batch_errs)

        # Validate dependency references exist within the plan or known_task_ids
        valid_deps = set(plan_task_ids)
        if known_task_ids:
            valid_deps.update(known_task_ids)

        for task_item in tasks:
            for dep in task_item.get("dependencies", []):
                if dep not in valid_deps:
                    errors.append(f"Task '{task_item.get('task_id')}' references unknown dependency '{dep}'.")

        return len(errors) == 0, errors

    @classmethod
    def create_task_definition(cls, contract_data: dict[str, Any]) -> TaskDefinition:
        """
        Convert a validated contract dictionary into a typed TaskDefinition.
        Raises TaskContractValidationError if validation fails.
        """
        ok, errors = cls.validate_contract(contract_data)
        if not ok:
            raise TaskContractValidationError(errors)

        pattern = None
        pat_data = contract_data.get("pre_validation_pattern")
        if pat_data:
            pattern = PreValidationPattern(
                file_path=pat_data["file_path"],
                pattern=pat_data["pattern"],
                description=pat_data.get("description", ""),
            )

        return TaskDefinition(
            task_id=contract_data["task_id"],
            title=contract_data["title"],
            audit_source=contract_data["audit_source"],
            priority=TaskPriority(contract_data["priority"]),
            risk=TaskRisk(contract_data["risk"]),
            description=contract_data["description"],
            evidence=contract_data.get("evidence", ""),
            target_files=list(contract_data["target_files"]),
            test_targets=list(contract_data.get("test_targets", [])),
            read_files=list(contract_data.get("read_files", [])),
            dependencies=list(contract_data.get("dependencies", [])),
            acceptance_criteria=list(contract_data["acceptance_criteria"]),
            verification_instructions=contract_data.get("verification_instructions", ""),
            pre_validation_pattern=pattern,
            status=TaskStatus.PENDING,
            attempts=0,
            max_attempts=int(contract_data.get("max_attempts", 3)),
            allow_file_deletion=bool(contract_data.get("allow_file_deletion", False)),
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
