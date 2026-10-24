"""
skills/maintenance/macro/tools.py
Foundational Typed Tool Registry & Execution Engine for AutoPhage 2.0 (MACRO Phase 1).

Design invariants:
- Typed tool definitions with schema generation and input validation.
- Strict filesystem containment within the assigned worktree (normally under .maya/worktrees/<task_id>).
- Shared-file write access permitted exclusively for docs/CHANGELOG.md in accordance with AGENTS.md.
- Strict rejection of traversal outside permitted roots, alternate drive escapes, symlink escapes,
  and any access to .git/ internals.
- Exact-match requirement for edits: rejects stale (0 matches) and ambiguous (>1 matches) edits.
- Preserves existing line endings (\r\n vs \n) and UTF-8 encoding.
- Subprocess execution without shell interpolation (shell=False) strictly confined to worktree cwd.
- test_runner executes focused pytest targets under .maya/tests/ with JUnit XML parsing and timeouts.
- finding_record collects structured, typed findings preserving stable issue identifiers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import fnmatch
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Optional, Union

logger = logging.getLogger("maya.maintenance.macro.tools")

# Compatibility import for QualityGateEngine.parse_pytest_execution
try:
    from skills.maintenance.quality import parse_pytest_execution
except ImportError:
    try:
        from quality import parse_pytest_execution
    except ImportError:
        def parse_pytest_execution(*args, **kwargs):
            return {
                "total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0,
                "failures": [], "collection_errors": [],
            }

# Compatibility import for forbidden target definitions
try:
    from skills.maintenance.task_contract import FORBIDDEN_TARGET_EXACT, FORBIDDEN_TARGET_PREFIXES
except ImportError:
    try:
        from task_contract import FORBIDDEN_TARGET_EXACT, FORBIDDEN_TARGET_PREFIXES
    except ImportError:
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


# ==============================================================================
# Exceptions
# ==============================================================================

class MacroToolError(Exception):
    """Base exception for all MACRO tool errors."""
    pass


class PathTraversalError(MacroToolError):
    """Raised when a path escapes the permitted worktree or repository root."""
    pass


class ForbiddenPathError(MacroToolError):
    """Raised when a path attempts to access protected or forbidden repository paths."""
    pass


class ToolValidationError(MacroToolError):
    """Raised when tool arguments fail validation or are missing."""
    pass


# ==============================================================================
# Data Models & Results
# ==============================================================================

@dataclass
class ToolResult:
    """Standardized result returned by all MACRO tool invocations."""
    success: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "truncated": self.truncated,
        }


@dataclass
class ToolParameter:
    """Typed parameter definition for tool schema registration."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    items_type: Optional[str] = None
    enum: Optional[list[str]] = None


@dataclass
class ToolDefinition:
    """Tool metadata and execution binding."""
    name: str
    description: str
    parameters: list[ToolParameter]
    execute: Callable[..., ToolResult]
    category: str = "general"

    def to_schema(self) -> dict[str, Any]:
        """Convert parameter specifications to standard JSON Schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.items_type and p.type == "array":
                prop["items"] = {"type": p.items_type}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default

            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolExecutionContext:
    """
    Execution context providing worktree isolation, repository boundaries,
    and operational limits to tools.
    """
    worktree_path: Path
    repo_root: Path
    task_id: Optional[str] = None
    task_type: str = "general"
    target_files: list[str] = field(default_factory=list)
    test_targets: list[str] = field(default_factory=list)
    require_repair_evidence: Optional[bool] = None
    require_tool_execution: bool = False
    allow_shared_changelog: bool = True
    max_read_chars: int = 100_000
    max_search_matches: int = 100
    max_diff_chars: int = 100_000
    test_timeout_sec: float = 180.0
    findings: list[dict[str, Any]] = field(default_factory=list)
    is_cancelled: bool = False
    tools_executed: list[dict[str, Any]] = field(default_factory=list)
    files_modified: set[str] = field(default_factory=set)
    tests_passed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.worktree_path = self.worktree_path.resolve()
        self.repo_root = self.repo_root.resolve()

    def is_repair_task(self) -> bool:
        if self.require_repair_evidence is not None:
            return bool(self.require_repair_evidence)
        return str(self.task_type).lower() in ("defect", "repair", "bug")


# ==============================================================================
# PathGuard - Security, Containment & Boundary Validation
# ==============================================================================

class PathGuard:
    """Enforces strict path containment, traversal rejection, and forbidden target rules."""

    @staticmethod
    def is_git_path(path: Path | str) -> bool:
        """Check if path targets .git internal files or directories."""
        raw = str(path).replace("\\", "/").lower()
        parts = [p.lower() for p in Path(raw).parts]
        return ".git" in parts or raw.startswith(".git/") or "/.git/" in raw

    @staticmethod
    def normalize_relpath(path_str: str | Path) -> str:
        """Standardize relative path string without leading slashes or dots."""
        p = str(path_str).strip().replace("\\", "/")
        while p.startswith("./") or p.startswith("/"):
            p = p.lstrip("./")
        return p

    @classmethod
    def validate_read_path(
        cls,
        path_str: str | Path,
        context: ToolExecutionContext,
    ) -> Path:
        """
        Validate path for reading. Allows reading from worktree_path or repo_root,
        strictly prohibiting .git internals and external filesystem escapes.
        """
        if cls.is_git_path(path_str):
            raise ForbiddenPathError(f"Access to Git internal path '{path_str}' is strictly forbidden.")

        norm_rel = cls.normalize_relpath(path_str)
        if norm_rel in FORBIDDEN_TARGET_EXACT:
            raise ForbiddenPathError(f"Access to protected target '{norm_rel}' is forbidden.")

        # Check if the path exists (or is a symlink) in worktree_path
        wt_target = context.worktree_path / norm_rel
        if wt_target.is_symlink() or wt_target.exists():
            candidate = wt_target.resolve()
            if not cls._is_subpath(candidate, context.worktree_path) and not cls._is_subpath(candidate, context.repo_root):
                raise PathTraversalError(
                    f"Path '{path_str}' resolves outside permitted boundary: {candidate}"
                )
            if cls.is_git_path(candidate):
                raise ForbiddenPathError(f"Resolved path '{candidate}' penetrates a .git directory.")
            return candidate

        # If not present in worktree, fallback to repo_root resolution
        candidate_repo = (context.repo_root / norm_rel).resolve()
        if not cls._is_subpath(candidate_repo, context.repo_root):
            raise PathTraversalError(
                f"Path '{path_str}' resolves outside permitted boundary: {candidate_repo}"
            )
        candidate = candidate_repo

        if cls.is_git_path(candidate):
            raise ForbiddenPathError(f"Resolved path '{candidate}' penetrates a .git directory.")

        return candidate

    @classmethod
    def validate_write_path(
        cls,
        path_str: str | Path,
        context: ToolExecutionContext,
    ) -> Path:
        """
        Validate path for writing or editing. Must be strictly confined to worktree_path,
        with the sole approved exception of docs/CHANGELOG.md.
        """
        if cls.is_git_path(path_str):
            raise ForbiddenPathError(f"Modifying Git internal path '{path_str}' is strictly forbidden.")

        norm_rel = cls.normalize_relpath(path_str)

        # Check exact forbidden targets
        if norm_rel in FORBIDDEN_TARGET_EXACT:
            raise ForbiddenPathError(f"Modifying protected target '{norm_rel}' is forbidden.")

        # Resolve candidate in worktree_path
        candidate = (context.worktree_path / norm_rel).resolve()

        # Check changelog exception
        is_changelog = (norm_rel == "docs/CHANGELOG.md")
        if is_changelog and context.allow_shared_changelog:
            # Changelog can be written in worktree or repo_root
            if cls._is_subpath(candidate, context.worktree_path):
                return candidate
            candidate_repo = (context.repo_root / "docs/CHANGELOG.md").resolve()
            return candidate_repo

        # Check prefix restrictions for ordinary task modifications
        for pfx in FORBIDDEN_TARGET_PREFIXES:
            if norm_rel.startswith(pfx) or f"/{pfx}" in norm_rel:
                raise ForbiddenPathError(
                    f"Modifying protected subsystem or internal path '{norm_rel}' is forbidden."
                )

        # Enforce worktree confinement
        if not cls._is_subpath(candidate, context.worktree_path):
            raise PathTraversalError(
                f"Write target '{path_str}' resolves outside permitted worktree boundary: {candidate}"
            )

        if cls.is_git_path(candidate):
            raise ForbiddenPathError(f"Resolved write target '{candidate}' penetrates a .git directory.")

        return candidate

    @staticmethod
    def _is_subpath(candidate: Path, parent: Path) -> bool:
        """Safe subpath check accounting for Windows drive letters and case."""
        try:
            cand_res = candidate.resolve()
            parent_res = parent.resolve()
            # On Windows, drives must match
            if os.name == "nt":
                if cand_res.drive.lower() != parent_res.drive.lower():
                    return False
            cand_res.relative_to(parent_res)
            return True
        except (ValueError, RuntimeError):
            return False


# ==============================================================================
# Sandboxed Subprocess Runner
# ==============================================================================

def run_sandboxed_command(
    cmd: list[str],
    cwd: Path,
    timeout: float = 60.0,
    max_output_chars: int = 50_000,
    env: Optional[dict[str, str]] = None,
) -> tuple[int, str, str, float, bool]:
    """
    Execute a subprocess strictly bounded by cwd, timeout, and character limits.
    shell=False is strictly enforced to prevent shell interpolation.
    Returns (exit_code, stdout, stderr, duration_sec, timed_out).
    """
    if not isinstance(cmd, (list, tuple)) or not cmd:
        raise ValueError("Command must be a non-empty list of string arguments.")

    start_time = time.monotonic()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout,
            env=env,
            check=False,
        )
        duration = time.monotonic() - start_time
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = proc.returncode

        if len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars] + f"\n... [stdout truncated at {max_output_chars} chars]"
        if len(stderr) > max_output_chars:
            stderr = stderr[:max_output_chars] + f"\n... [stderr truncated at {max_output_chars} chars]"

        return exit_code, stdout, stderr, duration, False

    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start_time
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return -1, stdout, f"Execution timed out after {timeout:.1f}s", duration, True
    except Exception as exc:
        duration = time.monotonic() - start_time
        return -1, "", str(exc), duration, False


# ==============================================================================
# Tool Implementations
# ==============================================================================

def _fs_read_impl(
    context: ToolExecutionContext,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_lines: int = 1000,
    max_chars: Optional[int] = None,
) -> ToolResult:
    """Read a file within permitted boundary with line numbering and truncation."""
    try:
        resolved_path = PathGuard.validate_read_path(path, context)
    except (PathTraversalError, ForbiddenPathError) as e:
        return ToolResult(success=False, output="", error=f"Access denied: {e}")

    if not resolved_path.exists():
        return ToolResult(success=False, output="", error=f"File not found: '{path}'")
    if resolved_path.is_dir():
        return ToolResult(success=False, output="", error=f"Target '{path}' is a directory, not a file.")

    char_limit = max_chars or context.max_read_chars

    try:
        raw_content = resolved_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Failed to read file '{path}': {e}")

    lines = raw_content.splitlines()
    total_lines = len(lines)

    # Validate range
    s_line = start_line if start_line is not None else 1
    e_line = end_line if end_line is not None else total_lines

    if s_line < 1:
        return ToolResult(success=False, output="", error=f"Invalid start_line {s_line}; must be >= 1.")
    if e_line < s_line:
        return ToolResult(success=False, output="", error=f"Invalid line range: end_line {e_line} < start_line {s_line}.")

    if total_lines == 0:
        return ToolResult(
            success=True,
            output="[Empty file]",
            data={"total_lines": 0, "lines_returned": 0, "path": str(resolved_path)},
        )

    # Slice lines (1-indexed)
    idx_start = max(0, s_line - 1)
    idx_end = min(total_lines, e_line)
    selected_lines = lines[idx_start:idx_end]

    truncated = False
    if len(selected_lines) > max_lines:
        selected_lines = selected_lines[:max_lines]
        truncated = True

    # Format numbered lines
    formatted_rows = []
    width = len(str(min(total_lines, s_line + len(selected_lines))))
    for i, line_content in enumerate(selected_lines, start=s_line):
        formatted_rows.append(f"{str(i).rjust(width)} | {line_content}")

    out_text = "\n".join(formatted_rows)
    if len(out_text) > char_limit:
        out_text = out_text[:char_limit] + f"\n... [Output truncated at {char_limit} characters]"
        truncated = True

    if truncated:
        out_text += f"\n[Output truncated: showing lines {s_line}-{s_line + len(selected_lines) - 1} of {total_lines}]"

    return ToolResult(
        success=True,
        output=out_text,
        data={
            "path": PathGuard.normalize_relpath(resolved_path),
            "total_lines": total_lines,
            "start_line": s_line,
            "end_line": s_line + len(selected_lines) - 1,
            "lines_returned": len(selected_lines),
        },
        truncated=truncated,
    )


def _fs_search_impl(
    context: ToolExecutionContext,
    query: str,
    path_pattern: Optional[str] = None,
    is_regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 50,
) -> ToolResult:
    """Search repository files by text or regex, excluding non-code and temporary trees."""
    if not query:
        return ToolResult(success=False, output="", error="Search query cannot be empty.")

    root_dir = context.worktree_path if context.worktree_path.is_dir() else context.repo_root
    pattern_filter = path_pattern or "*"

    # Compile regex if requested
    compiled_regex = None
    if is_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled_regex = re.compile(query, flags)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regular expression '{query}': {e}")
    else:
        needle = query if case_sensitive else query.lower()

    # Directories to strictly exclude
    EXCLUDED_DIRS = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        ".ruff_cache", "node_modules", "worktrees", ".maya/tmp",
        "build", "dist",
    }

    matches: list[dict[str, Any]] = []
    total_scanned_files = 0
    truncated = False

    for root, dirs, files in os.walk(str(root_dir)):
        # Prune excluded directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIRS
            and not d.startswith(".git")
            and not (Path(root) / d).name in EXCLUDED_DIRS
        ]

        for file in files:
            # Check pattern filter
            rel_file = os.path.relpath(os.path.join(root, file), str(root_dir)).replace("\\", "/")
            if not fnmatch.fnmatch(file, pattern_filter) and not fnmatch.fnmatch(rel_file, pattern_filter):
                continue

            full_p = Path(root) / file
            if PathGuard.is_git_path(full_p):
                continue

            # Skip large files (> 2MB) or common binaries
            if full_p.stat().st_size > 2 * 1024 * 1024:
                continue

            total_scanned_files += 1
            try:
                content = full_p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for line_idx, line in enumerate(content.splitlines(), start=1):
                is_match = False
                if compiled_regex:
                    if compiled_regex.search(line):
                        is_match = True
                else:
                    target_line = line if case_sensitive else line.lower()
                    if needle in target_line:
                        is_match = True

                if is_match:
                    matches.append({
                        "file": rel_file,
                        "line": line_idx,
                        "content": line.strip()[:200],
                    })
                    if len(matches) >= max_matches:
                        truncated = True
                        break
            if truncated:
                break
        if truncated:
            break

    if not matches:
        return ToolResult(
            success=True,
            output=f"No matches found for query '{query}' (scanned {total_scanned_files} files).",
            data={"matches": [], "total_matches": 0, "scanned_files": total_scanned_files},
        )

    lines_out = [f"Found {len(matches)} match(es) (scanned {total_scanned_files} files):"]
    for m in matches:
        lines_out.append(f"{m['file']}:{m['line']}: {m['content']}")

    if truncated:
        lines_out.append(f"... [Truncated at {max_matches} matches]")

    return ToolResult(
        success=True,
        output="\n".join(lines_out),
        data={"matches": matches, "total_matches": len(matches), "scanned_files": total_scanned_files},
        truncated=truncated,
    )


def _fs_find_symbols_impl(
    context: ToolExecutionContext,
    path: Optional[str] = None,
    name_filter: Optional[str] = None,
    symbol_type: Optional[str] = None,
    max_symbols: int = 100,
) -> ToolResult:
    """Discover Python (AST) and JS/TS symbols, handling malformed syntax gracefully."""
    root_dir = context.worktree_path if context.worktree_path.is_dir() else context.repo_root

    target_files: list[Path] = []
    if path:
        try:
            resolved = PathGuard.validate_read_path(path, context)
            if resolved.is_file():
                target_files = [resolved]
            elif resolved.is_dir():
                target_files = [p for p in resolved.rglob("*") if p.is_file()]
        except (PathTraversalError, ForbiddenPathError) as e:
            return ToolResult(success=False, output="", error=str(e))
    else:
        # Default scan across relevant files
        for p in root_dir.rglob("*"):
            if p.is_file() and not PathGuard.is_git_path(p):
                if any(part in {".venv", "__pycache__", "node_modules", ".maya/tmp", ".maya/worktrees"} for part in p.parts):
                    continue
                if p.suffix in {".py", ".js", ".jsx", ".ts", ".tsx"}:
                    target_files.append(p)

    symbols: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    truncated = False

    sym_filter = (symbol_type or "all").lower()
    needle = name_filter.lower() if name_filter else None

    for f in target_files:
        rel_path = os.path.relpath(str(f), str(root_dir)).replace("\\", "/")

        if f.suffix == ".py":
            try:
                code = f.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(code, filename=str(f))

                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        if sym_filter in {"all", "class"}:
                            if not needle or needle in node.name.lower():
                                symbols.append({
                                    "name": node.name,
                                    "type": "class",
                                    "file": rel_path,
                                    "line": node.lineno,
                                })
                        # Check methods inside class
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if sym_filter in {"all", "method", "function"}:
                                    if not needle or needle in item.name.lower():
                                        symbols.append({
                                            "name": f"{node.name}.{item.name}",
                                            "type": "method",
                                            "file": rel_path,
                                            "line": item.lineno,
                                        })

                    elif isinstance(node, ast.FunctionDef):
                        if sym_filter in {"all", "function"}:
                            if not needle or needle in node.name.lower():
                                symbols.append({
                                    "name": node.name,
                                    "type": "function",
                                    "file": rel_path,
                                    "line": node.lineno,
                                })

                    elif isinstance(node, ast.AsyncFunctionDef):
                        if sym_filter in {"all", "function", "async_function"}:
                            if not needle or needle in node.name.lower():
                                symbols.append({
                                    "name": node.name,
                                    "type": "async_function",
                                    "file": rel_path,
                                    "line": node.lineno,
                                })

                    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                        if sym_filter in {"all", "variable"}:
                            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                            for t in targets:
                                if isinstance(t, ast.Name):
                                    if not needle or needle in t.id.lower():
                                        symbols.append({
                                            "name": t.id,
                                            "type": "variable",
                                            "file": rel_path,
                                            "line": node.lineno,
                                        })

                    if len(symbols) >= max_symbols:
                        truncated = True
                        break
            except SyntaxError as syn_err:
                errors.append({"file": rel_path, "error": f"SyntaxError: {syn_err.msg} at line {syn_err.lineno}"})
            except Exception as e:
                errors.append({"file": rel_path, "error": str(e)})

        elif f.suffix in {".js", ".jsx", ".ts", ".tsx"}:
            try:
                code = f.read_text(encoding="utf-8", errors="replace")
                for line_idx, line in enumerate(code.splitlines(), start=1):
                    # Simple heuristic regex for JS/TS function and class declarations
                    m_fn = re.search(r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)", line)
                    if m_fn and sym_filter in {"all", "function"}:
                        name = m_fn.group(1)
                        if not needle or needle in name.lower():
                            symbols.append({"name": name, "type": "function", "file": rel_path, "line": line_idx})

                    m_cls = re.search(r"(?:export\s+)?(?:default\s+)?class\s+([A-Za-z0-9_$]+)", line)
                    if m_cls and sym_filter in {"all", "class"}:
                        name = m_cls.group(1)
                        if not needle or needle in name.lower():
                            symbols.append({"name": name, "type": "class", "file": rel_path, "line": line_idx})

                    if len(symbols) >= max_symbols:
                        truncated = True
                        break
            except Exception as e:
                errors.append({"file": rel_path, "error": str(e)})

        if truncated:
            break

    out_lines = [f"Discovered {len(symbols)} symbol(s):"]
    for s in symbols:
        out_lines.append(f"  [{s['type']}] {s['name']} ({s['file']}:{s['line']})")

    if errors:
        out_lines.append(f"\nEncountered {len(errors)} non-fatal parsing error(s):")
        for err in errors[:5]:
            out_lines.append(f"  {err['file']}: {err['error']}")

    if truncated:
        out_lines.append(f"... [Truncated at {max_symbols} symbols]")

    return ToolResult(
        success=True,
        output="\n".join(out_lines),
        data={"symbols": symbols, "total_symbols": len(symbols), "errors": errors},
        truncated=truncated,
    )


def _fs_edit_impl(
    context: ToolExecutionContext,
    path: str,
    old_text: str,
    new_text: str,
) -> ToolResult:
    """Apply an exact-match edit to a file within the worktree boundary."""
    if not old_text:
        return ToolResult(success=False, output="", error="old_text cannot be empty.")

    try:
        resolved_path = PathGuard.validate_write_path(path, context)
    except (PathTraversalError, ForbiddenPathError) as e:
        return ToolResult(success=False, output="", error=f"Security violation: {e}")

    if not resolved_path.exists():
        return ToolResult(success=False, output="", error=f"Target file '{path}' does not exist.")
    if resolved_path.is_dir():
        return ToolResult(success=False, output="", error=f"Target '{path}' is a directory, not a file.")

    try:
        raw_bytes = resolved_path.read_bytes()
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Failed to read file '{path}': {e}")

    # Detect newline convention
    is_crlf = b"\r\n" in raw_bytes

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("utf-8", errors="replace")

    # Count occurrences
    occurrences = content.count(old_text)

    if occurrences == 0:
        # Try normalizing newlines in old_text for matching
        normalized_content = content.replace("\r\n", "\n")
        normalized_old = old_text.replace("\r\n", "\n")
        if normalized_old in normalized_content:
            occurrences = normalized_content.count(normalized_old)
            content = normalized_content
            old_text = normalized_old

    if occurrences == 0:
        return ToolResult(
            success=False,
            output="",
            error=(
                f"Target old_text not found in '{path}'. "
                "Edit rejected as stale or mismatched. Inspect the latest file content before editing."
            ),
            data={"path": str(resolved_path), "occurrences": 0},
        )

    if occurrences > 1:
        return ToolResult(
            success=False,
            output="",
            error=(
                f"Target old_text matches {occurrences} locations in '{path}'. "
                "Ambiguous match rejected; provide more surrounding context to target a unique occurrence."
            ),
            data={"path": str(resolved_path), "occurrences": occurrences},
        )

    # Single unambiguous match! Perform replacement
    new_content = content.replace(old_text, new_text, 1)

    # Re-apply CRLF if the original file was CRLF
    if is_crlf and "\r\n" not in new_content:
        new_content = new_content.replace("\n", "\r\n")

    try:
        resolved_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Failed to write changes to '{path}': {e}")

    norm_rel = PathGuard.normalize_relpath(resolved_path)
    return ToolResult(
        success=True,
        output=f"Successfully edited '{norm_rel}': replaced 1 unambiguous match.",
        data={
            "path": norm_rel,
            "occurrences": 1,
            "bytes_written": len(new_content.encode("utf-8")),
            "newline": "CRLF" if is_crlf else "LF",
        },
    )


def _fs_write_impl(
    context: ToolExecutionContext,
    path: str,
    content: str,
    overwrite: bool = False,
) -> ToolResult:
    """Create or overwrite a file strictly within the worktree boundary."""
    try:
        resolved_path = PathGuard.validate_write_path(path, context)
    except (PathTraversalError, ForbiddenPathError) as e:
        return ToolResult(success=False, output="", error=f"Security violation: {e}")

    existed = resolved_path.exists()
    if existed and not overwrite:
        return ToolResult(
            success=False,
            output="",
            error=f"File '{path}' already exists and overwrite=False. Set overwrite=True to replace.",
            data={"path": str(resolved_path), "existed": True},
        )

    try:
        # Create parent directories safely
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Failed to write file '{path}': {e}")

    operation = "overwritten" if existed else "created"
    norm_rel = PathGuard.normalize_relpath(resolved_path)
    return ToolResult(
        success=True,
        output=f"File '{norm_rel}' {operation} successfully ({len(content.encode('utf-8'))} bytes).",
        data={
            "path": norm_rel,
            "operation": operation,
            "bytes": len(content.encode("utf-8")),
        },
    )


def _git_diff_impl(
    context: ToolExecutionContext,
    paths: Optional[Union[list[str], str]] = None,
    staged: bool = False,
    base_commit: str = "HEAD",
    max_lines: int = 500,
) -> ToolResult:
    """Inspect changes in the assigned worktree via git diff."""
    wt = context.worktree_path if context.worktree_path.is_dir() else context.repo_root

    cmd = ["git", "diff"]
    if staged:
        cmd.append("--cached")
    elif base_commit:
        cmd.append(base_commit)

    path_args: list[str] = []
    if paths:
        if isinstance(paths, str):
            path_args = [paths]
        elif isinstance(paths, (list, tuple)):
            path_args = list(paths)

    if path_args:
        cmd.append("--")
        cmd.extend(path_args)

    exit_code, stdout, stderr, duration, timed_out = run_sandboxed_command(
        cmd=cmd,
        cwd=wt,
        timeout=30.0,
        max_output_chars=context.max_diff_chars,
    )

    if exit_code != 0:
        return ToolResult(
            success=False,
            output="",
            error=f"git diff failed (exit code {exit_code}): {stderr.strip() or stdout.strip()}",
            data={"exit_code": exit_code, "stderr": stderr},
        )

    diff_text = stdout.strip()
    if not diff_text:
        return ToolResult(
            success=True,
            output="No changes detected in worktree.",
            data={"diff": "", "lines": 0},
        )

    lines = diff_text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    out = "\n".join(lines)
    if truncated:
        out += f"\n... [Diff truncated at {max_lines} lines]"

    return ToolResult(
        success=True,
        output=out,
        data={"lines": len(lines), "duration_sec": duration},
        truncated=truncated,
    )


def _test_runner_impl(
    context: ToolExecutionContext,
    test_targets: Union[list[str], str],
    filter_expr: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> ToolResult:
    """Execute focused pytest targets under .maya/tests/ using repository venv."""
    if not test_targets:
        return ToolResult(
            success=False,
            output="",
            error="No test targets specified. Focused test targets under '.maya/tests/' are required.",
        )

    targets: list[str] = [test_targets] if isinstance(test_targets, str) else list(test_targets)

    # Validate that every target is strictly under .maya/tests/
    for t in targets:
        norm = PathGuard.normalize_relpath(t)
        if not norm.startswith(".maya/tests/") and not norm.startswith(".maya\\tests\\"):
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Invalid test target '{t}'. Per repository test policy, "
                    "all tests must reside strictly under '.maya/tests/'."
                ),
            )

    wt = context.worktree_path if context.worktree_path.is_dir() else context.repo_root

    # Resolve python executable in repo .venv
    py_exe = context.repo_root / ".venv" / "Scripts" / "python.exe"
    if not py_exe.is_file():
        py_exe = Path(sys.executable)

    # Set up temporary JUnit XML output
    tmp_xml: Optional[Path] = None
    tmp_dir = context.repo_root / ".maya" / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(prefix="pytest_junit_", suffix=".xml", dir=str(tmp_dir))
        os.close(fd)
        tmp_xml = Path(tmp_path_str)
    except Exception:
        tmp_xml = None

    cmd = [
        str(py_exe),
        "-m",
        "pytest",
        "-v",
        "--rootdir=.",
        "-o",
        "pythonpath=.",
    ]
    if tmp_xml:
        cmd.append(f"--junitxml={str(tmp_xml)}")
    if filter_expr:
        cmd.extend(["-k", filter_expr])

    cmd.extend(targets)

    timeout = timeout_seconds or context.test_timeout_sec

    env = os.environ.copy()
    orig_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{str(wt)}{os.pathsep}{str(context.repo_root)}{os.pathsep}{orig_pypath}"

    try:
        exit_code, stdout, stderr, duration, timed_out = run_sandboxed_command(
            cmd=cmd,
            cwd=wt,
            timeout=timeout,
            max_output_chars=context.max_read_chars,
            env=env,
        )

        if timed_out:
            return ToolResult(
                success=False,
                output="",
                error=f"Test runner timed out after {timeout:.1f}s.",
                data={"timed_out": True, "duration_sec": duration, "exit_code": -1},
            )

        # Parse structured results
        structured = parse_pytest_execution(
            junit_xml_path=tmp_xml,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

        # Format human/model readable summary
        total = structured.get("total", 0)
        passed = structured.get("passed", 0)
        failed = structured.get("failed", 0)
        errors_count = structured.get("errors", 0)

        status_str = "PASSED" if exit_code == 0 else "FAILED"
        summary_lines = [
            f"Pytest {status_str} (exit code {exit_code}) in {duration:.2f}s",
            f"Results: {total} total, {passed} passed, {failed} failed, {errors_count} error(s).",
        ]

        failures = structured.get("failures", [])
        if failures:
            summary_lines.append("\nFailures:")
            for f in failures[:5]:
                summary_lines.append(f"  - {f.get('test')}: {f.get('message', '')[:300]}")

        col_errors = structured.get("collection_errors", [])
        if col_errors:
            summary_lines.append("\nCollection Errors:")
            for c in col_errors[:3]:
                summary_lines.append(f"  - {c.get('file')}: {c.get('message', '')[:300]}")

        return ToolResult(
            success=(exit_code == 0),
            output="\n".join(summary_lines),
            data={
                "exit_code": exit_code,
                "duration_sec": duration,
                "structured": structured,
                "targets": targets,
            },
        )

    finally:
        if tmp_xml and tmp_xml.is_file():
            try:
                tmp_xml.unlink(missing_ok=True)
            except Exception:
                pass


def _finding_record_impl(
    context: ToolExecutionContext,
    title: str,
    category: str,
    severity: str,
    confidence: str,
    issue_id: Optional[str] = None,
    description: str = "",
    affected_locations: Optional[Union[list[str], str]] = None,
    evidence: str = "",
    recommendations: str = "",
) -> ToolResult:
    """Record a structured finding preserving stable issue identifiers."""
    ALLOWED_CATEGORIES = {
        "DEFECT", "ARCHITECTURAL_CONCERN", "PERFORMANCE", "SECURITY", "IMPROVEMENT",
    }
    ALLOWED_SEVERITIES = {
        "P0", "P1", "P2", "P3", "CRITICAL", "HIGH", "MEDIUM", "LOW",
    }
    ALLOWED_CONFIDENCES = {
        "CONFIRMED", "PROBABLE", "SPECULATIVE",
    }

    cat_upper = category.upper()
    sev_upper = severity.upper()
    conf_upper = confidence.upper()

    if cat_upper not in ALLOWED_CATEGORIES:
        return ToolResult(
            success=False,
            output="",
            error=f"Invalid category '{category}'. Allowed: {sorted(ALLOWED_CATEGORIES)}",
        )
    if sev_upper not in ALLOWED_SEVERITIES:
        return ToolResult(
            success=False,
            output="",
            error=f"Invalid severity '{severity}'. Allowed: {sorted(ALLOWED_SEVERITIES)}",
        )
    if conf_upper not in ALLOWED_CONFIDENCES:
        return ToolResult(
            success=False,
            output="",
            error=f"Invalid confidence '{confidence}'. Allowed: {sorted(ALLOWED_CONFIDENCES)}",
        )

    locs: list[str] = []
    if affected_locations:
        if isinstance(affected_locations, str):
            locs = [affected_locations]
        elif isinstance(affected_locations, (list, tuple)):
            locs = list(affected_locations)

    finding_record = {
        "issue_id": issue_id or f"FINDING-{int(time.time())}-{len(context.findings) + 1}",
        "title": title.strip(),
        "category": cat_upper,
        "severity": sev_upper,
        "confidence": conf_upper,
        "description": description.strip(),
        "affected_locations": locs,
        "evidence": evidence.strip(),
        "recommendations": recommendations.strip(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    context.findings.append(finding_record)

    return ToolResult(
        success=True,
        output=(
            f"Recorded finding [{finding_record['issue_id']}]: "
            f"'{finding_record['title']}' ({cat_upper}, {sev_upper}, {conf_upper})."
        ),
        data={"finding": finding_record, "total_recorded": len(context.findings)},
    )


# ==============================================================================
# Central MacroToolRegistry
# ==============================================================================

class MacroToolRegistry:
    """Central registry and dispatch engine for MACRO tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool definition."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Export tool definitions as standard JSON Schema functions."""
        return [t.to_schema() for t in self._tools.values()]

    def execute(
        self,
        name: str,
        context: ToolExecutionContext,
        arguments: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Dispatch and execute a tool with input validation and exception isolation.
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool '{name}'. Available tools: {list(self._tools.keys())}",
            )

        args = arguments or {}

        # Validate required arguments
        for param in tool.parameters:
            if param.required and param.name not in args:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Missing required parameter '{param.name}' for tool '{name}'.",
                )

        try:
            res = tool.execute(context=context, **args)
        except Exception as exc:
            logger.exception("Unexpected execution error in tool '%s'", name)
            res = ToolResult(
                success=False,
                output="",
                error=f"Internal tool error in '{name}': {exc}",
            )

        # Track execution and outcomes in execution context
        if hasattr(context, "tools_executed") and isinstance(context.tools_executed, list):
            context.tools_executed.append({
                "tool": name,
                "success": res.success,
                "error": res.error,
                "args": args,
            })
        if res.success:
            if name in ("fs_write", "fs_edit", "fs_patch"):
                if hasattr(context, "files_modified"):
                    target_p = args.get("path") or args.get("file_path")
                    if target_p:
                        context.files_modified.add(PathGuard.normalize_relpath(str(target_p)))
            elif name == "test_runner":
                if hasattr(context, "tests_passed"):
                    target_t = args.get("test_targets") or args.get("target") or "tests"
                    if isinstance(target_t, list):
                        for tt in target_t:
                            context.tests_passed.append(PathGuard.normalize_relpath(str(tt)))
                    else:
                        context.tests_passed.append(PathGuard.normalize_relpath(str(target_t)))

        return res


def create_default_registry() -> MacroToolRegistry:
    """Create and populate registry with all Phase 1 MACRO tools."""
    registry = MacroToolRegistry()

    # 1. fs_read
    registry.register(ToolDefinition(
        name="fs_read",
        description="Read a file with line numbers and optional line-range selection within the permitted boundary.",
        category="filesystem",
        parameters=[
            ToolParameter("path", "string", "Relative path to file within worktree or repository root.", required=True),
            ToolParameter("start_line", "integer", "1-indexed starting line number.", required=False, default=1),
            ToolParameter("end_line", "integer", "1-indexed ending line number (inclusive).", required=False),
            ToolParameter("max_lines", "integer", "Maximum lines to return (safety bound).", required=False, default=1000),
            ToolParameter("max_chars", "integer", "Maximum characters to return (safety bound).", required=False),
        ],
        execute=_fs_read_impl,
    ))

    # 2. fs_search
    registry.register(ToolDefinition(
        name="fs_search",
        description="Search repository files by text or regex, excluding non-code and temporary directories.",
        category="filesystem",
        parameters=[
            ToolParameter("query", "string", "Text or regular expression to search for.", required=True),
            ToolParameter("path_pattern", "string", "Glob filter for file paths (e.g. '*.py' or 'services/*').", required=False, default="*"),
            ToolParameter("is_regex", "boolean", "Whether query is a regular expression.", required=False, default=False),
            ToolParameter("case_sensitive", "boolean", "Whether to perform case-sensitive search.", required=False, default=False),
            ToolParameter("max_matches", "integer", "Maximum matching lines to return.", required=False, default=50),
        ],
        execute=_fs_search_impl,
    ))

    # 3. fs_find_symbols
    registry.register(ToolDefinition(
        name="fs_find_symbols",
        description="Discover Python AST symbols and JS/TS declarations, handling syntax errors gracefully.",
        category="filesystem",
        parameters=[
            ToolParameter("path", "string", "Optional file or directory to inspect (defaults to codebase scan).", required=False),
            ToolParameter("name_filter", "string", "Substring filter for symbol name.", required=False),
            ToolParameter("symbol_type", "string", "Filter by type: 'class', 'function', 'async_function', 'method', 'variable', or 'all'.", required=False, default="all", enum=["class", "function", "async_function", "method", "variable", "all"]),
            ToolParameter("max_symbols", "integer", "Maximum symbols to return.", required=False, default=100),
        ],
        execute=_fs_find_symbols_impl,
    ))

    # 4. fs_edit
    registry.register(ToolDefinition(
        name="fs_edit",
        description="Apply an exact-match replacement to a file in the worktree, rejecting ambiguous or stale edits.",
        category="filesystem",
        parameters=[
            ToolParameter("path", "string", "Relative path to file in worktree.", required=True),
            ToolParameter("old_text", "string", "Exact existing text to be replaced (must match uniquely).", required=True),
            ToolParameter("new_text", "string", "Replacement text.", required=True),
        ],
        execute=_fs_edit_impl,
    ))

    # 5. fs_write
    registry.register(ToolDefinition(
        name="fs_write",
        description="Create or overwrite a file strictly within the worktree boundary.",
        category="filesystem",
        parameters=[
            ToolParameter("path", "string", "Relative path to file in worktree.", required=True),
            ToolParameter("content", "string", "Full text content to write.", required=True),
            ToolParameter("overwrite", "boolean", "Allow overwriting existing file.", required=False, default=False),
        ],
        execute=_fs_write_impl,
    ))

    # 6. git_diff
    registry.register(ToolDefinition(
        name="git_diff",
        description="Inspect git diff in the assigned worktree with optional path filtering.",
        category="git",
        parameters=[
            ToolParameter("paths", "array", "Optional list of files or directories to filter diff.", required=False, items_type="string"),
            ToolParameter("staged", "boolean", "Whether to inspect staged changes (--cached).", required=False, default=False),
            ToolParameter("base_commit", "string", "Base commit to diff against.", required=False, default="HEAD"),
            ToolParameter("max_lines", "integer", "Maximum diff lines to return.", required=False, default=500),
        ],
        execute=_git_diff_impl,
    ))

    # 7. test_runner
    registry.register(ToolDefinition(
        name="test_runner",
        description="Execute focused pytest targets strictly under .maya/tests/ using repository environment.",
        category="verification",
        parameters=[
            ToolParameter("test_targets", "array", "Focused test file(s) or directory under '.maya/tests/'.", required=True, items_type="string"),
            ToolParameter("filter_expr", "string", "Pytest keyword filter expression (-k).", required=False),
            ToolParameter("timeout_seconds", "number", "Subprocess execution timeout in seconds.", required=False, default=180.0),
        ],
        execute=_test_runner_impl,
    ))

    # 8. finding_record
    registry.register(ToolDefinition(
        name="finding_record",
        description="Record a structured finding preserving stable issue identifiers and categories.",
        category="discovery",
        parameters=[
            ToolParameter("title", "string", "Concise title describing the finding.", required=True),
            ToolParameter("category", "string", "Category: 'DEFECT', 'ARCHITECTURAL_CONCERN', 'PERFORMANCE', 'SECURITY', 'IMPROVEMENT'.", required=True, enum=["DEFECT", "ARCHITECTURAL_CONCERN", "PERFORMANCE", "SECURITY", "IMPROVEMENT"]),
            ToolParameter("severity", "string", "Severity: 'P0', 'P1', 'P2', 'P3' (or 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW').", required=True, enum=["P0", "P1", "P2", "P3", "CRITICAL", "HIGH", "MEDIUM", "LOW"]),
            ToolParameter("confidence", "string", "Confidence: 'CONFIRMED', 'PROBABLE', 'SPECULATIVE'.", required=True, enum=["CONFIRMED", "PROBABLE", "SPECULATIVE"]),
            ToolParameter("issue_id", "string", "Optional stable identifier (e.g. 'BUG-F10-LAUNCH-PASSTHROUGH').", required=False),
            ToolParameter("description", "string", "Detailed description of the finding.", required=False, default=""),
            ToolParameter("affected_locations", "array", "List of affected file paths or line ranges.", required=False, items_type="string"),
            ToolParameter("evidence", "string", "Reproduction excerpt or failure evidence.", required=False, default=""),
            ToolParameter("recommendations", "string", "Suggested resolution or fix strategy.", required=False, default=""),
        ],
        execute=_finding_record_impl,
    ))

    return registry
