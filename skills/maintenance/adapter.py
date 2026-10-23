"""
Agent Adapter Interface and Implementations for MayaVE Maintenance Coordinator.
Provides strict boundaries, timeout management, non-interactive execution,
and safe test mocks.
"""

from __future__ import annotations

import abc
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from .models import TaskDefinition, TaskPriority, TaskRisk
except ImportError:
    from models import TaskDefinition, TaskPriority, TaskRisk


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Safely terminate a process and all its descendants on Windows or Unix."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


@dataclass
class AgentExecutionResult:
    """Standardized result returned by any agent adapter."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    interrupted: bool = False
    timed_out: bool = False
    permission_denied: bool = False
    transport_unavailable: bool = False
    error_message: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_dict(self, max_output_chars: int = 4000) -> dict:
        stdout_excerpt = self.stdout or ""
        if len(stdout_excerpt) > max_output_chars:
            head = max_output_chars // 2
            tail = max_output_chars - head
            stdout_excerpt = (
                f"{stdout_excerpt[:head]}\n... [truncated {len(stdout_excerpt) - max_output_chars} chars] ...\n{stdout_excerpt[-tail:]}"
            )

        stderr_excerpt = self.stderr or ""
        if len(stderr_excerpt) > max_output_chars:
            head = max_output_chars // 2
            tail = max_output_chars - head
            stderr_excerpt = (
                f"{stderr_excerpt[:head]}\n... [truncated {len(stderr_excerpt) - max_output_chars} chars] ...\n{stderr_excerpt[-tail:]}"
            )

        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout_length": len(self.stdout or ""),
            "stderr_length": len(self.stderr or ""),
            "stdout_excerpt": stdout_excerpt,
            "stderr_excerpt": stderr_excerpt,
            "duration_sec": round(self.duration_sec, 2),
            "interrupted": self.interrupted,
            "timed_out": self.timed_out,
            "permission_denied": self.permission_denied,
            "transport_unavailable": self.transport_unavailable,
            "error_message": self.error_message,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class AgentAdapter(abc.ABC):
    """Abstract interface for executing an agent on a task inside an isolated worktree."""

    @abc.abstractmethod
    def execute(
        self,
        task: TaskDefinition,
        prompt: str,
        worktree_path: Path,
        timeout_seconds: int = 600,
    ) -> AgentExecutionResult:
        """
        Execute the agent on the specified task inside worktree_path.
        Must not mutate repository state outside worktree_path.
        """
        pass


class FakeAgentAdapter(AgentAdapter):
    """
    Test and simulation adapter.
    Allows injecting file mutations, simulating timeouts, and asserting prompt contents
    without launching real processes or consuming quota.
    """

    def __init__(
        self,
        default_success: bool = True,
        default_exit_code: int = 0,
        default_stdout: str = "Fake agent execution completed.",
        default_stderr: str = "",
        timeout_on: set[str] | None = None,
        fail_on: set[str] | None = None,
        mutations: dict[str, Callable[[Path], None]] | None = None,
    ) -> None:
        self.default_success = default_success
        self.default_exit_code = default_exit_code
        self.default_stdout = default_stdout
        self.default_stderr = default_stderr
        self.timeout_on = timeout_on or set()
        self.fail_on = fail_on or set()
        self.mutations = mutations or {}
        self.history: list[dict] = []

    def execute(
        self,
        task: TaskDefinition,
        prompt: str,
        worktree_path: Path,
        timeout_seconds: int = 600,
    ) -> AgentExecutionResult:
        start_time = time.monotonic()
        call_record = {
            "task_id": task.task_id,
            "prompt": prompt,
            "worktree_path": str(worktree_path),
            "timeout_seconds": timeout_seconds,
        }
        self.history.append(call_record)

        if task.task_id in self.timeout_on:
            duration = min(0.01, float(timeout_seconds))
            return AgentExecutionResult(
                success=False,
                exit_code=124,
                stdout="",
                stderr="Execution timed out",
                duration_sec=duration,
                interrupted=True,
                timed_out=True,
                error_message=f"Task {task.task_id} timed out after {timeout_seconds}s",
            )

        if task.task_id in self.fail_on:
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Agent failed to resolve task {task.task_id}",
                duration_sec=0.01,
                interrupted=False,
                timed_out=False,
                error_message=f"Simulated agent failure for task {task.task_id}",
            )

        # Apply mutation if registered for this task
        if task.task_id in self.mutations:
            mutation_fn = self.mutations[task.task_id]
            try:
                mutation_fn(worktree_path)
            except Exception as e:
                return AgentExecutionResult(
                    success=False,
                    exit_code=1,
                    stdout="",
                    stderr=f"Mutation hook failed: {e}",
                    duration_sec=time.monotonic() - start_time,
                    error_message=str(e),
                )

        duration = time.monotonic() - start_time
        return AgentExecutionResult(
            success=self.default_success,
            exit_code=self.default_exit_code,
            stdout=self.default_stdout,
            stderr=self.default_stderr,
            duration_sec=duration,
            interrupted=False,
            timed_out=False,
        )


def resolve_antigravity_executable(custom_path: str | Path | None = None) -> str:
    """
    Deterministically resolve the Antigravity CLI executable path:
    1. Explicit custom path or argument if provided and valid.
    2. Environment variable AGY_PATH, ANTIGRAVITY_PATH, AGY_BIN, or ANTIGRAVITY_BIN.
    3. Standard PATH lookup via shutil.which('agy.exe') or shutil.which('agy').
    4. Well-known user installation directories:
       - ~/.gemini/bin/agy.exe
       - ~/.gemini/bin/agy
       - ~/.gemini/antigravity/bin/agy.exe
       - AppData/Local/Programs/antigravity/Antigravity.exe
       - AppData/Roaming/Antigravity/bin/agy.exe
    5. Fallback string "agy.exe".
    """
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.is_file():
            return str(p)
        which_p = shutil.which(str(custom_path))
        if which_p:
            return which_p
        return str(custom_path)

    for env_var in ("AGY_PATH", "ANTIGRAVITY_PATH", "AGY_BIN", "ANTIGRAVITY_BIN"):
        val = os.environ.get(env_var)
        if val:
            p = Path(val).expanduser().resolve()
            if p.is_file():
                return str(p)
            which_p = shutil.which(val)
            if which_p:
                return which_p

    for name in ("agy.exe", "agy"):
        found = shutil.which(name)
        if found:
            return found

    # Check well-known user paths
    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.extend([
            Path(local_app_data) / "agy" / "bin" / "agy.exe",
            Path(local_app_data) / "agy" / "bin" / "agy",
        ])

    home = Path.home()
    candidates.extend([
        home / "AppData" / "Local" / "agy" / "bin" / "agy.exe",
        home / "AppData" / "Local" / "agy" / "bin" / "agy",
        home / ".gemini" / "bin" / "agy.exe",
        home / ".gemini" / "bin" / "agy",
        home / ".gemini" / "antigravity" / "bin" / "agy.exe",
        home / ".gemini" / "antigravity" / "bin" / "agy",
        home / "AppData" / "Local" / "Programs" / "antigravity" / "Antigravity.exe",
        home / "AppData" / "Roaming" / "Antigravity" / "bin" / "agy.exe",
    ])
    for c in candidates:
        if c.is_file():
            return str(c)

    return "agy.exe"


SUPPORTED_ANTIGRAVITY_MODES: tuple[str, ...] = ("accept-edits", "auto", "plan")


class AntigravityAgentAdapter(AgentAdapter):
    """
    Adapter for invoking Antigravity CLI (agy.exe) headlessly in supervised single-task mode.
    Strictly disabled by default; requires explicit authorization (--authorize).
    """

    def __init__(
        self,
        authorized: bool = False,
        executable: str | Path | None = None,
        mode: str = "accept-edits",
        agent: Optional[str] = None,
    ) -> None:
        if mode not in SUPPORTED_ANTIGRAVITY_MODES:
            raise ValueError(
                f"Unsupported Antigravity mode '{mode}'. "
                f"Supported modes are: {', '.join(SUPPORTED_ANTIGRAVITY_MODES)}"
            )
        self.authorized = authorized
        self.executable = resolve_antigravity_executable(executable)
        self.mode = mode
        self.agent = agent

    def execute(
        self,
        task: TaskDefinition,
        prompt: str,
        worktree_path: Path,
        timeout_seconds: int = 600,
    ) -> AgentExecutionResult:
        if not self.authorized:
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="Authorization rejected",
                duration_sec=0.0,
                error_message=(
                    "Execution prohibited: Antigravity agent invocation requires explicit operator "
                    "authorization (--authorize). Unattended autonomous execution is disabled."
                ),
            )

        if not worktree_path.is_dir():
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="Invalid worktree path",
                duration_sec=0.0,
                error_message=f"Target worktree path does not exist or is not a directory: {worktree_path}",
            )

        resolved_exe = resolve_antigravity_executable(self.executable)
        if not (shutil.which(resolved_exe) or Path(resolved_exe).is_file()):
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="Executable not found",
                duration_sec=0.0,
                error_message=f"Antigravity CLI binary '{self.executable}' not found in PATH or standard installation locations.",
            )

        # Strictly bounded invocation command
        cmd = [
            str(resolved_exe),
            "-p",
            prompt,
        ]
        if self.agent:
            cmd.extend(["--agent", self.agent])
        cmd.extend(["--mode", self.mode])

        start_time = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(worktree_path),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_sec=time.monotonic() - start_time,
                error_message=f"Failed to spawn agent process: {e}",
            )

        timed_out = False
        interrupted = False
        stdout_data = ""
        stderr_data = ""

        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            interrupted = True
            kill_process_tree(proc)
            try:
                stdout_data, stderr_data = proc.communicate(timeout=5)
            except Exception:
                pass
            exit_code = 124
        except KeyboardInterrupt:
            interrupted = True
            kill_process_tree(proc)
            try:
                stdout_data, stderr_data = proc.communicate(timeout=5)
            except Exception:
                pass
            exit_code = 130
            raise
        except Exception as e:
            interrupted = True
            kill_process_tree(proc)
            exit_code = 1
            stderr_data = f"Process execution failed: {e}"

        duration = time.monotonic() - start_time

        # Detect headless permission denial in CLI output
        denial_indicators = (
            "auto-denied",
            "permission check failed",
            "required the \"command\" permission",
            "required the 'command' permission",
            "user denied permission",
            "soft-denying tool confirmation",
            "soft denied: path outside cwd",
            "path outside cwd",
        )
        combined_output = f"{stderr_data or ''}\n{stdout_data or ''}".lower()
        permission_denied = any(indicator in combined_output for indicator in denial_indicators)

        success = (exit_code == 0) and not timed_out and not interrupted and not permission_denied

        error_msg = None
        if timed_out:
            error_msg = f"Agent process exceeded timeout of {timeout_seconds}s and was terminated."
        elif exit_code != 0:
            error_msg = f"Agent process exited with non-zero status code: {exit_code}."
        elif permission_denied:
            error_msg = (
                "Headless tool permission denied: agent attempted an unauthorized command "
                "or path outside worktree that was auto-denied."
            )

        return AgentExecutionResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout_data or "",
            stderr=stderr_data or "",
            duration_sec=duration,
            interrupted=interrupted,
            timed_out=timed_out,
            permission_denied=permission_denied,
            error_message=error_msg,
        )


class LocalModelAgentAdapter(AgentAdapter):
    """
    Adapter for invoking a local LLM (defaults to Qwen-2.5-Coder 3B via Ollama)
    to perform targeted code modifications on LOW-risk maintenance tasks.
    Enforces strict diff bounds, unified diff parsing, and path containment.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5-coder:3b",
        temperature: float = 0.2,
        client: Optional[Any] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._client = client

    def _call_ollama(self, prompt: str, timeout_seconds: int = 120) -> tuple[str, Optional[int], Optional[int]]:
        """
        Call Ollama /api/generate endpoint synchronously.
        Returns (response_text, input_tokens, output_tokens).
        """
        if self._client is not None:
            res = self._client.generate(prompt=prompt, model=self.model)
            if isinstance(res, tuple):
                return res
            if isinstance(res, dict):
                return (
                    res.get("response", ""),
                    res.get("prompt_eval_count"),
                    res.get("eval_count"),
                )
            return str(res), None, None

        url = f"{self.base_url}/api/generate"
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (
                    data.get("response", ""),
                    data.get("prompt_eval_count"),
                    data.get("eval_count"),
                )
        except urllib.error.URLError as e:
            raise ConnectionError(f"Ollama connection error at {url}: {e}") from e
        except TimeoutError as e:
            raise TimeoutError(f"Ollama request timed out after {timeout_seconds}s") from e

    @staticmethod
    def _normalize_path(raw_path: str) -> str:
        """Normalize target file paths consistently."""
        if not raw_path:
            return ""
        norm = raw_path.replace("\\", "/").strip()
        while norm.startswith("./") or norm.startswith("/"):
            norm = norm[2:] if norm.startswith("./") else norm[1:]
        return norm

    @staticmethod
    def _is_distinctive_context_line(line: str) -> bool:
        """
        Check if a context line is distinctive enough for fuzzy offset matching (F-01).
        Excludes whitespace, short tokens, and common boilerplate/control statements.
        """
        s = line.strip()
        if len(s) < 3:
            return False
        trivial = {
            "pass", "return", "return True", "return False", "return None",
            "break", "continue", "{", "}", "(", ")", "[", "]",
            "else:", "try:", "except:", "finally:", "do", "end",
        }
        return s not in trivial

    def _parse_unified_diff(
        self,
        text: str,
        allowed_files: set[str],
        worktree_path: Path,
    ) -> tuple[dict[str, str], set[str], Optional[str], Optional[str]]:
        """
        Extract and apply unified diff format patches from model output.
        Enforces F-01 (distinctive context threshold >= 2) and F-02 (file deletion detection).
        Returns (modified_files_dict, deleted_files_set, error_message, security_violation_message).
        """
        diff_pattern = re.compile(
            r"(?:diff --git[^\r\n]*[\r\n]+)?--- (?:a/)?([^\r\n\t]+)[\r\n]+\+\+\+ (?:b/)?([^\r\n\t]+)[\r\n]+"
            r"((?:@@ -[\d,]+ \+[\d,]+ @@[^\r\n]*[\r\n]+(?:(?: [^\r\n]*|-(?!--)[^\r\n]*|\+(?!\+\+)[^\r\n]*|\\[^\r\n]*)(?:[\r\n]+|$))*)+)",
            re.MULTILINE,
        )

        matches = list(diff_pattern.finditer(text))
        if not matches:
            return {}, set(), None, None

        results: dict[str, str] = {}
        deleted_files: set[str] = set()

        for match in matches:
            old_path_raw = match.group(1).strip()
            new_path_raw = match.group(2).strip()
            diff_body = match.group(3)

            is_deletion = (new_path_raw == "/dev/null" or new_path_raw == "b//dev/null")

            # Determine target file name
            target_raw = old_path_raw if is_deletion else new_path_raw
            if target_raw.startswith("b/") or target_raw.startswith("a/"):
                target_raw = target_raw[2:]
            norm_path = self._normalize_path(target_raw)

            # 1. Security traversal check
            if (
                ".." in norm_path
                or norm_path.startswith("/")
                or (len(norm_path) >= 2 and norm_path[1] == ":")
            ):
                return {}, set(), None, f"Local model attempted forbidden path traversal in diff: {target_raw}"

            # 2. Scope boundary check
            if norm_path.lower() not in {f.lower() for f in allowed_files}:
                return {}, set(), None, f"Local model attempted unauthorized modification outside target files in diff: {target_raw}"

            # If deletion diff
            if is_deletion:
                deleted_files.add(norm_path)
                results.pop(norm_path, None)
                continue

            # Read existing file content from worktree
            file_path = worktree_path / norm_path
            orig_lines: list[str] = []
            if file_path.is_file():
                try:
                    orig_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception as e:
                    return {}, set(), f"Could not read existing file '{norm_path}' for patching: {e}", None

            # Parse hunks
            hunk_pattern = re.compile(
                r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\r\n]*$",
                re.MULTILINE,
            )
            hunk_starts = list(hunk_pattern.finditer(diff_body))
            if not hunk_starts:
                return {}, set(), f"Malformed diff: no valid hunk headers in diff for {norm_path}", None

            # Split hunk text
            hunks = []
            for i, h_match in enumerate(hunk_starts):
                start_idx = h_match.start()
                end_idx = hunk_starts[i + 1].start() if i + 1 < len(hunk_starts) else len(diff_body)
                old_start = int(h_match.group(1))
                raw_lines = diff_body[start_idx:end_idx].splitlines()[1:]  # Skip header
                hunk_lines = [l for l in raw_lines if not l.startswith("--- ") and not l.startswith("+++ ") and not l.startswith("diff --git")]
                hunks.append((old_start, hunk_lines))

            # Apply hunks to lines
            new_lines = list(orig_lines)
            offset = 0

            for old_start, hunk_lines in hunks:
                # Separate expected context/deleted lines from additions
                expected_block: list[str] = []
                replacement_block: list[str] = []

                for l in hunk_lines:
                    if not l:
                        expected_block.append("")
                        replacement_block.append("")
                    elif l.startswith(" "):
                        expected_block.append(l[1:])
                        replacement_block.append(l[1:])
                    elif l.startswith("-"):
                        expected_block.append(l[1:])
                    elif l.startswith("+"):
                        replacement_block.append(l[1:])

                # If file was empty and all lines are additions
                if not new_lines and not expected_block:
                    new_lines = replacement_block
                    continue

                # Locate match index in new_lines
                target_idx = (old_start - 1) + offset
                match_pos = -1

                # Try exact position first
                if 0 <= target_idx <= len(new_lines) - len(expected_block):
                    if new_lines[target_idx:target_idx + len(expected_block)] == expected_block:
                        match_pos = target_idx

                # If not exact, evaluate context distinctiveness before sliding-window search (F-01)
                if match_pos == -1 and expected_block:
                    distinctive_count = sum(1 for line in expected_block if self._is_distinctive_context_line(line))
                    if distinctive_count < 2:
                        return (
                            {},
                            set(),
                            f"Hunk failed to apply to '{norm_path}' near line {old_start}: context has only {distinctive_count} distinctive line(s) (< 2 required for fuzzy search). Guessing rejected.",
                            None,
                        )

                    search_start = max(0, target_idx - 15)
                    search_end = min(len(new_lines) - len(expected_block) + 1, target_idx + 15)
                    candidate_matches = []
                    for pos in range(search_start, search_end):
                        if new_lines[pos:pos + len(expected_block)] == expected_block:
                            candidate_matches.append(pos)

                    if len(candidate_matches) == 1:
                        match_pos = candidate_matches[0]
                    elif len(candidate_matches) > 1:
                        return (
                            {},
                            set(),
                            f"Hunk failed to apply to '{norm_path}' near line {old_start}: ambiguous fuzzy match ({len(candidate_matches)} occurrences found in +-15 lines).",
                            None,
                        )

                if match_pos == -1 and expected_block:
                    return {}, set(), f"Hunk failed to apply to '{norm_path}' near line {old_start}", None

                if match_pos != -1:
                    new_lines[match_pos:match_pos + len(expected_block)] = replacement_block
                    offset += (len(replacement_block) - len(expected_block))

            patched_content = "\n".join(new_lines) + ("\n" if new_lines else "")
            results[norm_path] = patched_content

        return results, deleted_files, None, None

    def _parse_file_blocks(self, text: str) -> dict[str, str]:
        """
        Extract modified file paths and contents from LLM markdown response.
        Recognizes headings formatted as:
          ### FILE: <path>  OR  ### <path>
          ```<lang>
          <content>
          ```
        """
        results: dict[str, str] = {}
        pattern = re.compile(
            r"(?:###\s*FILE:\s*|###\s+|File:\s*)([^\r\n`#]+)[\r\n]+(?:```[a-zA-Z0-9_-]*[\r\n]+([\s\S]*?)```)",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            raw_path = match.group(1).strip()
            content = match.group(2)
            norm_path = self._normalize_path(raw_path)
            if norm_path:
                results[norm_path] = content

        return results

    def execute(
        self,
        task: TaskDefinition,
        prompt: str,
        worktree_path: Path,
        timeout_seconds: int = 180,
    ) -> AgentExecutionResult:
        start_time = time.monotonic()

        if not worktree_path.is_dir():
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr="Invalid worktree path",
                duration_sec=0.0,
                error_message=f"Target worktree path does not exist or is not a directory: {worktree_path}",
            )

        # 1. Read existing target file contents to provide full context
        file_contexts = []
        for tf in task.target_files:
            target_file = worktree_path / tf
            if target_file.exists():
                try:
                    f_content = target_file.read_text(encoding="utf-8", errors="replace")
                    file_contexts.append(f"--- File: {tf} ---\n{f_content}\n")
                except Exception as e:
                    file_contexts.append(f"--- File: {tf} (Error reading: {e}) ---\n")

        # Include changelog context
        changelog_file = worktree_path / "docs" / "CHANGELOG.md"
        changelog_snippet = ""
        if changelog_file.exists():
            try:
                ch_lines = changelog_file.read_text(encoding="utf-8", errors="replace").splitlines()
                changelog_snippet = "\n".join(ch_lines[:60])
            except Exception:
                pass

        # 2. Extract orchestrator retry feedback and task context from prompt
        retry_feedback = ""
        if prompt:
            # Check for standard previous attempt quality gate failure section
            marker = "=== PREVIOUS ATTEMPT QUALITY GATE FAILURE"
            if marker in prompt:
                idx = prompt.find(marker)
                end_markers = [
                    "Target Files Allowed to Modify:",
                    "Verification Test Targets:",
                    "=== STRICT OPERATIONAL INVARIANTS ===",
                ]
                end_pos = len(prompt)
                for em in end_markers:
                    p = prompt.find(em, idx)
                    if p != -1 and p < end_pos:
                        end_pos = p
                extracted = prompt[idx:end_pos].strip()
                # Bound feedback size to prevent context overflow (max 2500 chars)
                retry_feedback = f"\n\n{extracted[:2500]}\n"
            elif "=== TASK SPECIFICATION ===" not in prompt and prompt.strip():
                # Custom caller-supplied prompt
                bounded_custom = prompt.strip()[:2500]
                retry_feedback = f"\n\n=== ORCHESTRATOR RETRY & CONTEXT FEEDBACK ===\n{bounded_custom}\n"

        # Fallback to checkpoint details if prompt didn't contain feedback but checkpoint has failure history
        if not retry_feedback and task.checkpoint and isinstance(task.checkpoint.details, dict):
            last_gate_fail = task.checkpoint.details.get("last_quality_gate_failure")
            if last_gate_fail and isinstance(last_gate_fail, dict):
                prev_attempt = last_gate_fail.get("attempt", "?")
                summary_text = last_gate_fail.get("summary", "")[:2000]
                retry_feedback = (
                    f"\n\n=== PREVIOUS ATTEMPT QUALITY GATE FAILURE (Attempt {prev_attempt}) ===\n"
                    f"Your previous repair proposal was applied to the worktree but failed quality gate verification:\n"
                    f"{summary_text}\n\n"
                    f"Please adjust your implementation to satisfy all verification requirements.\n"
                )

        # 3. Build local model prompt tailored for Qwen-Coder / small coder models
        test_targets_str = ", ".join(task.test_targets) if task.test_targets else "None"
        criteria_str = "\n".join(f"- {ac}" for ac in task.acceptance_criteria) if task.acceptance_criteria else "- Fix confirmed defect"

        local_prompt = f"""You are the automated MayaVE Maintenance Engine. Your task is to resolve a verified defect in an isolated git worktree.

=== TASK DEFINITION ===
Task ID: {task.task_id}
Title: {task.title}
Priority: {task.priority.name}
Risk: {task.risk.name}
Description: {task.description}
Evidence: {task.evidence or 'None'}{retry_feedback}
Acceptance Criteria:
{criteria_str}
Allowed Target Files: {', '.join(task.target_files)}
Verification Test Targets: {test_targets_str}

=== EXISTING FILE CONTENT ===
{''.join(file_contexts)}

=== EXISTING CHANGELOG (docs/CHANGELOG.md excerpt) ===
{changelog_snippet}

=== INSTRUCTIONS ===
1. You may provide modifications using either standard unified diff format (preferred) or complete updated file blocks.
2. If using unified diff, format as:
```diff
--- a/<relative_path>
+++ b/<relative_path>
@@ -line,count +line,count @@
-old code
+new code
```
3. If using full file blocks, format as:
### FILE: <relative_path>
```python
<full updated content>
```
4. Also update docs/CHANGELOG.md under '### Fixes' or '### Improvements' with a concise entry, formatted as a diff hunk or full file block.
5. ONLY modify declared target files and docs/CHANGELOG.md. Do not modify any other files.
"""

        in_tokens: Optional[int] = None
        out_tokens: Optional[int] = None

        try:
            call_timeout = min(timeout_seconds, 120)
            res = self._call_ollama(local_prompt, timeout_seconds=call_timeout)
            if isinstance(res, tuple):
                response_text, in_tokens, out_tokens = res
            else:
                response_text = str(res)
        except TimeoutError as te:
            return AgentExecutionResult(
                success=False,
                exit_code=124,
                stdout="",
                stderr=str(te),
                duration_sec=time.monotonic() - start_time,
                timed_out=True,
                error_message=f"Local model request timed out after {call_timeout}s.",
            )
        except ConnectionError as ce:
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(ce),
                duration_sec=time.monotonic() - start_time,
                transport_unavailable=True,
                error_message=(
                    f"Local model runtime (Ollama) is unavailable at {self.base_url}: {ce}. "
                    f"Ensure 'ollama serve' is running and model '{self.model}' is available."
                ),
            )
        except Exception as e:
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_sec=time.monotonic() - start_time,
                error_message=f"Local model execution failed: {e}",
            )

        duration = time.monotonic() - start_time
        allowed_files = set(self._normalize_path(f) for f in task.target_files) | {"docs/changelog.md"}

        # 3. First attempt: Strict Unified Diff extraction
        modified_files, deleted_files, diff_err, sec_violation = self._parse_unified_diff(
            response_text, allowed_files, worktree_path
        )

        if sec_violation:
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout=response_text,
                stderr=sec_violation,
                duration_sec=duration,
                permission_denied=True,
                error_message=sec_violation,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )

        # 4. Second attempt: Fallback to Markdown file blocks if no diff found
        if not modified_files and not deleted_files and not diff_err:
            file_blocks = self._parse_file_blocks(response_text)
            if file_blocks:
                for rel_path, content in file_blocks.items():
                    norm_path = self._normalize_path(rel_path)
                    if (
                        ".." in norm_path
                        or norm_path.startswith("/")
                        or (len(norm_path) >= 2 and norm_path[1] == ":")
                    ):
                        return AgentExecutionResult(
                            success=False,
                            exit_code=1,
                            stdout=response_text,
                            stderr=f"Path traversal attempted: {rel_path}",
                            duration_sec=duration,
                            permission_denied=True,
                            error_message=f"Local model attempted forbidden path traversal: {rel_path}",
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                        )
                    if norm_path.lower() not in {f.lower() for f in allowed_files}:
                        return AgentExecutionResult(
                            success=False,
                            exit_code=1,
                            stdout=response_text,
                            stderr=f"Unauthorized file modification: {rel_path}",
                            duration_sec=duration,
                            permission_denied=True,
                            error_message=f"Local model attempted unauthorized modification outside target files: {rel_path}",
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                        )
                    modified_files[norm_path] = content

        # If neither format succeeded
        if not modified_files and not deleted_files:
            err_detail = diff_err or "Local model generated no parseable unified diff or file blocks."
            return AgentExecutionResult(
                success=False,
                exit_code=1,
                stdout=response_text,
                stderr=err_detail,
                duration_sec=duration,
                error_message=err_detail,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )

        # 5. Pre-application security and containment validation across ALL files
        # Check all modifications for worktree containment
        resolved_wt = worktree_path.resolve()
        for rel_path in modified_files:
            dest = (worktree_path / rel_path).resolve()
            if not str(dest).startswith(str(resolved_wt)):
                sec_msg = f"Path traversal attempted outside worktree: {rel_path}"
                return AgentExecutionResult(
                    success=False,
                    exit_code=1,
                    stdout=response_text,
                    stderr=sec_msg,
                    duration_sec=duration,
                    permission_denied=True,
                    error_message=sec_msg,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )

        # Check all deletions for scope, containment, and explicit authorization (F-02)
        for rel_path in deleted_files:
            dest = (worktree_path / rel_path).resolve()
            if not str(dest).startswith(str(resolved_wt)):
                sec_msg = f"Path traversal attempted in file deletion outside worktree: {rel_path}"
                return AgentExecutionResult(
                    success=False,
                    exit_code=1,
                    stdout=response_text,
                    stderr=sec_msg,
                    duration_sec=duration,
                    permission_denied=True,
                    error_message=sec_msg,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            if rel_path.lower() == "docs/changelog.md":
                sec_msg = "Model attempted forbidden deletion of changelog file."
                return AgentExecutionResult(
                    success=False,
                    exit_code=1,
                    stdout=response_text,
                    stderr=sec_msg,
                    duration_sec=duration,
                    permission_denied=True,
                    error_message=sec_msg,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )

            # Deletion authorization check
            desc_lower = (task.description + " " + " ".join(task.acceptance_criteria)).lower()
            file_base = Path(rel_path).name.lower()
            del_keywords = ("delete", "deletion", "remove", "removal", "unlink", "prune", "deprecate", "clean up")
            is_del_authorized = getattr(task, "allow_file_deletion", False) or (
                any(kw in desc_lower for kw in del_keywords) and (file_base in desc_lower or rel_path.lower() in desc_lower)
            )

            if not is_del_authorized:
                sec_msg = f"Local model attempted unauthorized file deletion of '{rel_path}'. Task does not authorize file deletion."
                return AgentExecutionResult(
                    success=False,
                    exit_code=1,
                    stdout=response_text,
                    stderr=sec_msg,
                    duration_sec=duration,
                    permission_denied=True,
                    error_message=sec_msg,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )

        # 6. Apply modifications and deletions atomically to worktree on disk
        for rel_path, content in modified_files.items():
            dest = worktree_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel_path.lower() == "docs/changelog.md" and "### Fixes" in content and not content.startswith("# Changelog"):
                try:
                    orig_cl = dest.read_text(encoding="utf-8", errors="replace")
                    clean_snip = content.strip()
                    if clean_snip not in orig_cl and "### Fixes" in orig_cl:
                        entry_lines = [l for l in clean_snip.splitlines() if not l.startswith("#")]
                        if entry_lines:
                            insert_text = "\n".join(entry_lines).strip()
                            orig_cl = orig_cl.replace("### Fixes\n", f"### Fixes\n{insert_text}\n", 1)
                            dest.write_text(orig_cl, encoding="utf-8")
                            continue
                except Exception:
                    pass

            dest.write_text(content, encoding="utf-8")

        for rel_path in deleted_files:
            dest = worktree_path / rel_path
            if dest.is_file() or dest.is_symlink():
                dest.unlink(missing_ok=True)

        return AgentExecutionResult(
            success=True,
            exit_code=0,
            stdout=response_text,
            stderr="",
            duration_sec=duration,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )


class TieredAgentAdapter(AgentAdapter):
    """
    Tiered Agent Adapter:
    - Routes appropriate LOW-risk tasks on initial attempts to LocalModelAgentAdapter.
    - Escalates complex tasks, failed local attempts (retries), or higher-risk tasks to AntigravityAgentAdapter.
    """

    def __init__(
        self,
        local_adapter: Optional[AgentAdapter] = None,
        escalation_adapter: Optional[AgentAdapter] = None,
        prefer_local_for_low_risk: bool = True,
    ) -> None:
        self.local_adapter = local_adapter
        self.escalation_adapter = escalation_adapter
        self.prefer_local_for_low_risk = prefer_local_for_low_risk
        self.history: list[dict[str, Any]] = []

    def execute(
        self,
        task: TaskDefinition,
        prompt: str,
        worktree_path: Path,
        timeout_seconds: int = 600,
    ) -> AgentExecutionResult:
        start_time = time.monotonic()

        is_low_risk = getattr(task, "risk", None) == TaskRisk.LOW
        is_initial_attempt = getattr(task, "attempts", 0) <= 1
        is_bounded_scope = len(getattr(task, "target_files", [])) <= 2

        qualifies_for_local = (
            self.prefer_local_for_low_risk
            and self.local_adapter is not None
            and is_low_risk
            and is_initial_attempt
            and is_bounded_scope
        )

        if qualifies_for_local:
            local_res = self.local_adapter.execute(
                task=task,
                prompt=prompt,
                worktree_path=worktree_path,
                timeout_seconds=min(timeout_seconds, 120),
            )
            self.history.append({
                "task_id": task.task_id,
                "tier": "local",
                "success": local_res.success,
                "error": local_res.error_message,
            })

            if local_res.success:
                return local_res

            escalation_reason = local_res.error_message or "Local model produced non-viable result"
            self.history.append({
                "task_id": task.task_id,
                "tier": "escalation",
                "reason": escalation_reason,
            })

            if self.escalation_adapter is not None:
                try:
                    subprocess.run(["git", "checkout", "."], cwd=str(worktree_path), capture_output=True, check=False)
                    subprocess.run(["git", "clean", "-fd"], cwd=str(worktree_path), capture_output=True, check=False)
                except Exception:
                    pass

                escalation_context = (
                    f"\n\n=== ESCALATION CONTEXT ===\n"
                    f"This task was escalated from a local model execution.\n"
                    f"Reason for escalation: {escalation_reason}\n"
                    f"Worktree has been sanitized to clean baseline HEAD to ensure safe implementation.\n"
                )
                esc_res = self.escalation_adapter.execute(
                    task=task,
                    prompt=prompt + escalation_context,
                    worktree_path=worktree_path,
                    timeout_seconds=timeout_seconds,
                )
                esc_res.stdout = f"[ESCALATED FROM LOCAL MODEL: {escalation_reason}]\n" + (esc_res.stdout or "")
                return esc_res

            return local_res

        if self.escalation_adapter is not None:
            # On retry escalation, sanitize any partial local edits from previous attempts
            if getattr(task, "attempts", 0) > 1:
                try:
                    subprocess.run(["git", "checkout", "."], cwd=str(worktree_path), capture_output=True, check=False)
                    subprocess.run(["git", "clean", "-fd"], cwd=str(worktree_path), capture_output=True, check=False)
                except Exception:
                    pass

            self.history.append({
                "task_id": task.task_id,
                "tier": "escalation",
                "reason": f"Direct route (risk={getattr(task.risk, 'value', task.risk)}, attempts={getattr(task, 'attempts', 0)})",
            })
            retry_note = ""
            if getattr(task, "attempts", 0) > 1:
                retry_note = f"\n\n=== RETRY CONTEXT ===\nAttempt {task.attempts}/{task.max_attempts}. Worktree has been sanitized to clean baseline HEAD.\n"
            return self.escalation_adapter.execute(
                task=task,
                prompt=prompt + retry_note,
                worktree_path=worktree_path,
                timeout_seconds=timeout_seconds,
            )

        if self.local_adapter is not None:
            return self.local_adapter.execute(
                task=task,
                prompt=prompt,
                worktree_path=worktree_path,
                timeout_seconds=timeout_seconds,
            )

        return AgentExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="No adapter configured",
            duration_sec=time.monotonic() - start_time,
            error_message="TieredAgentAdapter has no valid backend adapter configured.",
        )
