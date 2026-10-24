"""
skills/maintenance/macro/agent.py
MACRO Autonomous Reasoning Loop & Local Model Adapter (AutoPhage 2.0 Phase 2).

Design invariants:
- Bounded Think -> Tool -> Observe -> Reflect multi-turn reasoning loop.
- Local model integration via Ollama (/api/generate) with zero external network dependencies.
- Pluggable transport parameter for deterministic mock testing.
- Strict single-concurrency execution per agent instance (concurrency = 1).
- Resilient tool-call parser supporting <tool_call> tags, JSON markdown blocks, and finish tags.
- Loop prevention: detects repeated identical failed tool calls and breaks infinite loops.
- Configurable turn limits, wall-clock timeouts, malformed-response recovery, and cancellation.
- Integrates strictly with verified Phase 1 MacroToolRegistry and ToolExecutionContext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional, Tuple

from .context import MacroContextConfig, MacroContextManager
from .tools import (
    MacroToolRegistry,
    ToolExecutionContext,
    ToolResult,
    create_default_registry,
)

logger = logging.getLogger("maya.maintenance.macro.agent")


# ==============================================================================
# Configuration & Data Models
# ==============================================================================

@dataclass
class MacroAgentConfig:
    """Operational parameters and resource ceilings for MACRO agent reasoning loop."""
    model: str = "qwen2.5-coder:3b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    num_ctx: int = 4096
    num_predict: int = 512
    max_turn_timeout_seconds: float = 60.0
    timeout_seconds: float = 300.0
    max_turns: int = 15
    keep_alive: str = "5m"
    unload_on_exit: bool = True
    stop_sequences: list[str] = field(default_factory=lambda: ["</tool_call>", "</finish_task>"])
    max_consecutive_malformed: int = 3
    max_identical_failed_calls: int = 3
    max_consecutive_premature_finishes: int = 3
    max_context_chars: int = 24_000


@dataclass
class AgentStep:
    """Detailed record of a single step in the agent's execution trajectory."""
    turn: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[ToolResult] = None
    duration_sec: float = 0.0
    is_finish: bool = False
    finish_summary: Optional[str] = None


@dataclass
class MacroAgentResult:
    """Outcome and trajectory summary of a complete MACRO reasoning loop."""
    success: bool
    status: str  # "COMPLETED", "TURN_LIMIT_EXCEEDED", "TIMEOUT", "CANCELLED", "MALFORMED_OUTPUT_EXHAUSTED", "REPEATED_FAILURE_EXHAUSTED", "PREMATURE_COMPLETION_EXHAUSTED", "OLLAMA_ERROR", "CONCURRENCY_BLOCKED"
    summary: str
    turns: int
    duration_sec: float
    steps: list[AgentStep] = field(default_factory=list)
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    error_message: Optional[str] = None

    @property
    def error(self) -> Optional[str]:
        return self.error_message

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "summary": self.summary,
            "turns": self.turns,
            "duration_sec": self.duration_sec,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "error_message": self.error_message,
            "steps_count": len(self.steps),
        }


# ==============================================================================
# Tool Call & Response Parser
# ==============================================================================

class ToolCallParser:
    """
    Parses agent thoughts, tool invocations, and completion signals from model responses.
    Handles varied model formats (<tool_call>, markdown fences, inline JSON, <finish_task>).
    """

    @classmethod
    def parse(
        cls,
        text: str,
    ) -> Tuple[str, Optional[str], dict[str, Any], bool, Optional[str], Optional[str]]:
        """
        Parse raw model response.
        Returns:
            (thought, tool_name, tool_args, is_finished, finish_summary, parse_error)
        """
        raw = text.strip()
        if not raw:
            return "", None, {}, False, None, "Empty response received from model."

        # 1. First, search for actionable tool calls (<tool_call>, markdown fences, inline JSON).
        # Even if the model emitted a premature <finish_task> tag or prose in the same turn,
        # an executable tool call takes priority.

        # 1a. Explicit <tool_call> tags
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL | re.IGNORECASE)
        if tc_match:
            thought = raw[:tc_match.start()].strip()
            thought = cls._clean_thought(thought)
            body = tc_match.group(1).strip()
            tool_name, tool_args, err = cls._extract_json_call(body)
            if err:
                return thought, None, {}, False, None, f"Malformed JSON in <tool_call>: {err}"
            return thought, tool_name, tool_args, False, None, None

        # 1a-stop. Stop-sequence-truncated <tool_call> tags without closing tag
        tc_unclosed = re.search(r"<tool_call>(.*)$", raw, re.DOTALL | re.IGNORECASE)
        if tc_unclosed:
            body = tc_unclosed.group(1).strip()
            tool_name, tool_args, err = cls._extract_json_call(body)
            if not err and tool_name:
                thought = raw[:tc_unclosed.start()].strip()
                thought = cls._clean_thought(thought)
                return thought, tool_name, tool_args, False, None, None
            elif err and ("name" in body or "tool" in body or "action" in body):
                thought = raw[:tc_unclosed.start()].strip()
                thought = cls._clean_thought(thought)
                return thought, None, {}, False, None, f"Malformed JSON in unclosed <tool_call>: {err}"

        # 1b. Markdown json / tool_call fences: ```json ... ```
        fence_match = re.search(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        if fence_match:
            thought = raw[:fence_match.start()].strip()
            thought = cls._clean_thought(thought)
            body = fence_match.group(1).strip()
            tool_name, tool_args, err = cls._extract_json_call(body)
            if not err and tool_name:
                return thought, tool_name, tool_args, False, None, None
            elif err and ("name" in body or "tool" in body or "action" in body):
                return thought, None, {}, False, None, f"Malformed JSON in fenced tool call: {err}"

        # 1c. Inline balanced JSON objects with "name", "tool", or "action"
        for start_idx, end_idx, candidate_json in cls._find_balanced_json_objects(raw):
            tool_name, tool_args, err = cls._extract_json_call(candidate_json)
            if not err and tool_name:
                thought = raw[:start_idx].strip()
                thought = cls._clean_thought(thought)
                return thought, tool_name, tool_args, False, None, None

        # 2. If NO tool call was found, check for completion signals

        # 2a. Explicit <finish_task>summary</finish_task>
        finish_match = re.search(r"<finish_task>(.*?)</finish_task>", raw, re.DOTALL | re.IGNORECASE)
        if finish_match:
            finish_summary = finish_match.group(1).strip()
            if not finish_summary or len(finish_summary) < 3:
                return cls._clean_thought(raw[:finish_match.start()].strip()), None, {}, False, None, "Empty or truncated completion summary in <finish_task>."
            thought = raw[:finish_match.start()].strip()
            thought = cls._clean_thought(thought)
            return thought, None, {}, True, finish_summary, None

        # 2a-stop. Stop-sequence-truncated <finish_task> without closing tag
        finish_unclosed = re.search(r"<finish_task>(.*)$", raw, re.DOTALL | re.IGNORECASE)
        if finish_unclosed:
            finish_summary = finish_unclosed.group(1).strip()
            if not finish_summary or len(finish_summary) < 3:
                return cls._clean_thought(raw[:finish_unclosed.start()].strip()), None, {}, False, None, "Empty or truncated completion summary in <finish_task>."
            thought = raw[:finish_unclosed.start()].strip()
            thought = cls._clean_thought(thought)
            return thought, None, {}, True, finish_summary, None

        # 2b. Alternative finish patterns (e.g. TASK_COMPLETED: ...)
        alt_finish = re.search(r"^(?:TASK_COMPLETED|FINAL_ANSWER|COMPLETED)\s*:\s*(.*)$", raw, re.MULTILINE | re.IGNORECASE)
        if alt_finish:
            finish_summary = alt_finish.group(1).strip()
            if not finish_summary or len(finish_summary) < 3:
                return cls._clean_thought(raw[:alt_finish.start()].strip()), None, {}, False, None, "Empty or truncated completion summary in alternative completion marker."
            thought = raw[:alt_finish.start()].strip()
            return cls._clean_thought(thought), None, {}, True, finish_summary, None

        # 3. Neither tool call nor finish found
        thought = cls._clean_thought(raw)
        return (
            thought,
            None,
            {},
            False,
            None,
            "No valid <tool_call> tag or <finish_task> completion signal found in response.",
        )

    @classmethod
    def _find_balanced_json_objects(cls, text: str) -> list[Tuple[int, int, str]]:
        """Find balanced JSON { ... } slices in text, returning list of (start_idx, end_idx, json_str)."""
        results: list[Tuple[int, int, str]] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == '{':
                depth = 0
                in_str = False
                escape = False
                start_idx = i
                for j in range(i, n):
                    c = text[j]
                    if escape:
                        escape = False
                        continue
                    if c == '\\':
                        escape = True
                        continue
                    if c == '"':
                        in_str = not in_str
                        continue
                    if not in_str:
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                results.append((start_idx, j + 1, text[start_idx:j + 1]))
                                i = j
                                break
            i += 1
        return results

    @classmethod
    def _clean_thought(cls, text: str) -> str:
        """Strip <thought>, <tool_call>, and <finish_task> XML tags if explicitly emitted by model."""
        cleaned = re.sub(r"</?thought>", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<finish_task>.*?</finish_task>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?finish_task>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?tool_call>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @classmethod
    def _extract_json_call(cls, json_str: str) -> Tuple[Optional[str], dict[str, Any], Optional[str]]:
        """Parse JSON dictionary and extract tool name and arguments."""
        try:
            # Strip markdown code blocks if still present inside tags
            s = json_str.strip()
            if s.startswith("```"):
                s = re.sub(r"^```[a-z]*\s*", "", s, flags=re.IGNORECASE)
                s = re.sub(r"\s*```$", "", s)
            data = json.loads(s)
        except Exception as e:
            return None, {}, str(e)

        if not isinstance(data, dict):
            return None, {}, "Payload must be a JSON object dictionary."

        # Extract tool name
        tool_name = data.get("name") or data.get("tool") or data.get("action")
        if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
            return None, {}, "Missing or empty 'name' or 'tool' string in tool call payload."

        # Extract arguments
        args = data.get("arguments") or data.get("args") or data.get("action_input") or {}
        if not isinstance(args, dict):
            return None, {}, f"Tool arguments for '{tool_name}' must be a JSON object dictionary."

        # Normalize common model variations: if a parameter value is a single-element list of string, unpack it
        normalized_args: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
                normalized_args[k] = v[0]
            else:
                normalized_args[k] = v

        return tool_name.strip(), normalized_args, None


# ==============================================================================
# Ollama Model Client / Transport
# ==============================================================================

class OllamaTransport:
    """Synchronous HTTP client for local Ollama /api/generate endpoint."""

    @classmethod
    def generate(
        cls,
        prompt: str,
        config: MacroAgentConfig,
        timeout: Optional[float] = None,
    ) -> Tuple[str, Optional[int], Optional[int], Optional[str]]:
        """
        Call Ollama generate endpoint.
        Returns (response_text, input_tokens, output_tokens, done_reason).
        """
        url = f"{config.base_url.rstrip('/')}/api/generate"
        options: dict[str, Any] = {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "num_predict": getattr(config, "num_predict", 512),
        }
        stop_seqs = getattr(config, "stop_sequences", None)
        if stop_seqs:
            options["stop"] = list(stop_seqs)

        payload = json.dumps({
            "model": config.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": config.keep_alive,
            "options": options,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        timeout_sec = timeout if timeout is not None else getattr(config, "max_turn_timeout_seconds", 60.0)
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (
                    data.get("response", ""),
                    data.get("prompt_eval_count"),
                    data.get("eval_count"),
                    data.get("done_reason"),
                )
        except urllib.error.URLError as e:
            raise ConnectionError(f"Ollama connection error at {url}: {e}") from e
        except TimeoutError as e:
            raise TimeoutError(f"Ollama request timed out after {timeout_sec:.1f}s") from e

    @classmethod
    def unload(cls, config: MacroAgentConfig, timeout: float = 5.0) -> bool:
        """
        Request Ollama to evict model from memory by setting keep_alive to 0.
        Returns True if successfully acknowledged, False otherwise.
        Never raises an exception; failures are logged and do not mask primary outcomes.
        """
        url = f"{config.base_url.rstrip('/')}/api/generate"
        payload = json.dumps({
            "model": config.model,
            "keep_alive": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            logger.info("Evicted model '%s' from memory (keep_alive: 0).", config.model)
            return True
        except Exception as exc:
            logger.debug("Ollama unload request for '%s' non-critical failure: %s", config.model, exc)
            return False


# ==============================================================================
# MACRO Autonomous Agent Engine
# ==============================================================================

class MacroAgent:
    """
    MACRO Autonomous Maintenance Agent.
    Executes a bounded Think -> Tool -> Observe -> Reflect reasoning loop
    using the local Ollama model and verified sandboxed tools.
    """

    def __init__(
        self,
        config: Optional[MacroAgentConfig] = None,
        transport: Optional[Callable[[str, MacroAgentConfig], Tuple[str, Optional[int], Optional[int]]]] = None,
    ) -> None:
        self.config = config or MacroAgentConfig()
        self._transport = transport
        self._execution_lock = threading.Lock()
        self._is_cancelled = False

    def cancel(self) -> None:
        """Signal the running agent loop to halt immediately."""
        self._is_cancelled = True

    def _build_system_prompt(self, registry: MacroToolRegistry) -> str:
        """Construct the pinned system prompt detailing identity, boundaries, and tool calling format."""
        tools_summary = []
        for t in registry.list_tools():
            param_desc = ", ".join(f"{p.name} ({p.type}{'*' if p.required else ''})" for p in t.parameters)
            tools_summary.append(f"- **{t.name}**: {t.description}\n  Parameters: {param_desc}")

        tools_str = "\n".join(tools_summary)

        return (
            "You are MACRO (Macrophage-inspired Maintenance Agent), MayaVE's project-aware autonomous coding agent.\n"
            "Your objective is to investigate defects, plan repairs, inspect code, run focused tests, and record verifiable fixes.\n\n"
            "=== STRICT OPERATIONAL INVARIANTS ===\n"
            "1. WORKTREE CONFINEMENT: You may ONLY inspect or modify files within your assigned task worktree. Never attempt to read or modify .git/ internals, .env, or files outside the permitted boundary.\n"
            "2. EXACT-MATCH EDITS: When using 'fs_edit', the old_text must match uniquely (exact single match) in the target file. Provide sufficient surrounding context to disambiguate.\n"
            "3. TEST INTEGRITY: All test targets must reside strictly under '.maya/tests/'. Never weaken, delete, or comment out existing test assertions.\n"
            "4. CHANGELOG SYNCHRONIZATION: Record all verified fixes in 'docs/CHANGELOG.md' under '### Fixes' or '### Improvements'.\n"
            "5. EVIDENCE-BASED COMPLETION: Do not guess or fabricate successful repairs. Verify changes using 'test_runner' and 'git_diff' before concluding.\n\n"
            "=== TOOL CALLING PROTOCOL ===\n"
            "To invoke a tool, output your reasoning inside <thought>...</thought>, followed by a <tool_call> block with JSON:\n"
            "<thought>\n"
            "Explain your reasoning, hypothesis, or diagnostic goal here.\n"
            "</thought>\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param1": "val1"}}\n'
            "</tool_call>\n\n"
            "When the task is complete and verified, conclude with:\n"
            "<finish_task>\n"
            "Concise summary of actions taken, verified test outcomes, and final state.\n"
            "</finish_task>\n\n"
            "=== AVAILABLE TOOLS ===\n"
            f"{tools_str}"
        )

    def _validate_completion_evidence(
        self,
        context: ToolExecutionContext,
    ) -> Tuple[bool, Optional[str]]:
        """
        Independently verify whether sufficient verifiable evidence exists to justify task completion.
        Rejects premature finish claims when tools have not been executed or when repair outcomes
        are not established in the worktree.
        """
        tools_executed = getattr(context, "tools_executed", [])
        successful_tools = [t for t in tools_executed if t.get("success")]

        is_repair = getattr(context, "is_repair_task", lambda: False)()

        # For repair tasks, we strictly require verifiable repair outcomes:
        if is_repair:
            if not successful_tools:
                return (
                    False,
                    "No tools have been executed. A defect repair cannot be marked complete without inspecting files and executing tools."
                )

            # Independently verify file modification (via recorded modified files or worktree git diff)
            modified_files = getattr(context, "files_modified", set())
            has_worktree_diff = False
            worktree_path = getattr(context, "worktree_path", None)
            worktree_modified_files: set[str] = set()
            if worktree_path and Path(worktree_path).exists():
                try:
                    res = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=str(worktree_path),
                        capture_output=True,
                        text=True,
                        timeout=5.0,
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        has_worktree_diff = True
                        for line in res.stdout.splitlines():
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                clean_p = parts[1].replace("\\", "/").strip().lstrip("./")
                                worktree_modified_files.add(clean_p)
                except Exception:
                    pass

            if not modified_files and not has_worktree_diff:
                return (
                    False,
                    "No target files have been modified in the worktree. A defect repair requires applying targeted edits to resolve the issue."
                )

            # Task-specific target files validation: Unrelated file modifications cannot satisfy repair completion
            target_files = getattr(context, "target_files", [])
            if target_files:
                norm_targets = {t.replace("\\", "/").strip().lstrip("./") for t in target_files}
                norm_modified = {str(m).replace("\\", "/").strip().lstrip("./") for m in modified_files} | worktree_modified_files
                if not (norm_targets & norm_modified):
                    return (
                        False,
                        f"None of the declared target files ({', '.join(target_files)}) have been modified in the worktree. "
                        "Modifying unrelated files does not satisfy defect repair requirements."
                    )

            # If test targets were specified, verify test_runner was run and passed
            test_targets = getattr(context, "test_targets", [])
            test_runs = [t for t in tools_executed if t.get("tool") == "test_runner"]
            if test_targets and not test_runs:
                return (
                    False,
                    "No verification tests were executed with 'test_runner'. A defect repair must be verified by running focused tests."
                )

            if test_runs:
                last_test = test_runs[-1]
                if not last_test.get("success"):
                    err_details = last_test.get("error") or "test runner reported failures"
                    return (
                        False,
                        f"The most recent verification test execution failed ({err_details}). Fix all test failures before concluding."
                    )

            return True, None

        # For non-repair tasks:
        if getattr(context, "require_tool_execution", False) and not successful_tools:
            return False, "No tools have been executed. Task requires at least one successful tool execution."

        return True, None

    def run(
        self,
        task_objective: str,
        context: ToolExecutionContext,
        registry: Optional[MacroToolRegistry] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> MacroAgentResult:
        """
        Execute the iterative reasoning loop until the objective is accomplished,
        the turn limit is reached, or a timeout/cancellation occurs.
        """
        if not self._execution_lock.acquire(blocking=False):
            return MacroAgentResult(
                success=False,
                status="CONCURRENCY_BLOCKED",
                summary="Rejected: Another MACRO reasoning loop is currently active (concurrency = 1).",
                turns=0,
                duration_sec=0.0,
                steps=[],
                error_message="Another MACRO reasoning loop is active on this agent instance.",
            )

        self._is_cancelled = False
        start_time = time.monotonic()
        active_registry = registry or create_default_registry()

        # Initialize sliding window context manager
        system_prompt = self._build_system_prompt(active_registry)
        ctx_config = MacroContextConfig(max_context_chars=self.config.max_context_chars)
        context_mgr = MacroContextManager(
            system_prompt=system_prompt,
            task_objective=task_objective,
            config=ctx_config,
        )

        steps: list[AgentStep] = []
        consecutive_malformed = 0
        consecutive_premature = 0
        last_failed_signature: Optional[Tuple[str, str]] = None
        identical_failed_count = 0
        total_in_tokens = 0
        total_out_tokens = 0
        current_turn = 0
        final_status = "TURN_LIMIT_EXCEEDED"
        final_summary = ""
        error_msg: Optional[str] = None

        try:
            while current_turn < self.config.max_turns:
                current_turn += 1
                turn_start = time.monotonic()

                # 1. Check cancellation
                if self._is_cancelled or getattr(context, "is_cancelled", False):
                    final_status = "CANCELLED"
                    final_summary = f"Execution cancelled by operator after {current_turn - 1} turns."
                    break

                # 2. Check wall-clock timeout
                elapsed = time.monotonic() - start_time
                if elapsed >= self.config.timeout_seconds:
                    final_status = "TIMEOUT"
                    final_summary = f"Execution exceeded timeout ceiling of {self.config.timeout_seconds:.1f}s."
                    break

                # 3. Assemble prompt with sliding window compaction
                prompt_text = context_mgr.build_prompt()

                if on_progress:
                    on_progress(f"  [Turn {current_turn}/{self.config.max_turns}] Generating reasoning step with {self.config.model}...")

                # 4. Invoke local model via transport
                # Multi-constraint timeout: bounded by per-turn cap, remaining task budget, and optional campaign deadline
                remaining_task_budget = self.config.timeout_seconds - elapsed
                if remaining_task_budget <= 0:
                    final_status = "TIMEOUT"
                    final_summary = f"Execution exceeded task wall-clock budget of {self.config.timeout_seconds:.1f}s."
                    break

                campaign_deadline = getattr(context, "campaign_deadline_monotonic", None)
                if campaign_deadline is not None:
                    remaining_campaign_budget = campaign_deadline - time.monotonic()
                    if remaining_campaign_budget <= 0:
                        final_status = "TIMEOUT"
                        final_summary = "Execution halted: campaign deadline expired."
                        break
                    remaining_budget = min(remaining_task_budget, remaining_campaign_budget)
                else:
                    remaining_budget = remaining_task_budget

                if remaining_budget <= 0:
                    final_status = "TIMEOUT"
                    final_summary = "Execution halted: deadline expired."
                    break

                call_timeout = min(self.config.max_turn_timeout_seconds, remaining_budget)

                done_reason = None
                try:
                    if self._transport is not None:
                        t_res = self._transport(prompt_text, self.config)
                        if isinstance(t_res, tuple) and len(t_res) == 4:
                            response_text, in_tok, out_tok, done_reason = t_res
                        else:
                            response_text, in_tok, out_tok = t_res[:3]
                            done_reason = None
                    else:
                        response_text, in_tok, out_tok, done_reason = OllamaTransport.generate(
                            prompt=prompt_text,
                            config=self.config,
                            timeout=call_timeout,
                        )

                    if in_tok:
                        total_in_tokens += in_tok
                    if out_tok:
                        total_out_tokens += out_tok

                except TimeoutError as e:
                    elapsed_now = time.monotonic() - start_time
                    if elapsed_now >= self.config.timeout_seconds or (campaign_deadline and time.monotonic() >= campaign_deadline):
                        final_status = "TIMEOUT"
                        error_msg = f"Execution exceeded timeout ceiling: {e}"
                        final_summary = error_msg
                        break
                    if on_progress:
                        on_progress(f"  [Turn {current_turn}] Inference timed out: {e}")
                    consecutive_malformed += 1
                    step = AgentStep(
                        turn=current_turn,
                        thought="[Inference Timed Out]",
                        duration_sec=time.monotonic() - turn_start,
                    )
                    steps.append(step)
                    if consecutive_malformed >= self.config.max_consecutive_malformed:
                        final_status = "TIMEOUT"
                        error_msg = f"Inference timed out repeatedly ({consecutive_malformed} consecutive turns): {e}"
                        final_summary = error_msg
                        break
                    context_mgr.add_system_note(
                        f"Turn {current_turn} inference timed out ({e}). "
                        "Your previous response took too long. Emit concise, single tool calls without large text blocks."
                    )
                    continue

                except Exception as e:
                    final_status = "OLLAMA_ERROR"
                    error_msg = f"Local model invocation failed: {e}"
                    final_summary = f"Halted on Ollama error: {e}"
                    break

                # Post-inference cancellation check
                if self._is_cancelled or getattr(context, "is_cancelled", False):
                    final_status = "CANCELLED"
                    final_summary = f"Execution cancelled by operator during turn {current_turn}."
                    break

                # 5. Parse model response
                thought, tool_name, tool_args, is_finished, finish_summary, parse_err = ToolCallParser.parse(response_text)

                # If output was truncated by token limit, handle incomplete completions and payloads fail-closed
                if done_reason == "length":
                    if is_finished and "</finish_task>" not in response_text.lower():
                        is_finished = False
                        finish_summary = None
                        parse_err = (
                            f"Output truncated: token limit ({self.config.num_predict} tokens) "
                            "reached before <finish_task> completion."
                        )
                    elif parse_err:
                        parse_err = (
                            f"Output truncated: token limit ({self.config.num_predict} tokens) "
                            "reached before payload completion."
                        )

                # 6. Check for completion signal
                if is_finished:
                    valid_finish, rejection_reason = self._validate_completion_evidence(context)
                    if not valid_finish:
                        if on_progress:
                            on_progress(f"  [Turn {current_turn}] Completion claim rejected: {rejection_reason}")
                        consecutive_premature += 1
                        step = AgentStep(
                            turn=current_turn,
                            thought=thought,
                            duration_sec=time.monotonic() - turn_start,
                            is_finish=False,
                            finish_summary=f"REJECTED: {rejection_reason}",
                        )
                        steps.append(step)

                        if consecutive_premature >= self.config.max_consecutive_premature_finishes:
                            final_status = "PREMATURE_COMPLETION_EXHAUSTED"
                            error_msg = (
                                f"Terminated: Model made {consecutive_premature} consecutive premature "
                                f"<finish_task> claims without establishing verifiable completion evidence. "
                                f"Last reason: {rejection_reason}"
                            )
                            final_summary = error_msg
                            break

                        guidance = (
                            f"Turn {current_turn} <finish_task> rejected: {rejection_reason}. "
                            "You must inspect files, apply exact edits using tools, and verify tests before concluding."
                        )
                        context_mgr.add_system_note(guidance)
                        continue

                    # Legitimate completion established
                    if on_progress:
                        on_progress(f"  [Turn {current_turn}] Legitimate task completion verified and accepted.")
                    consecutive_premature = 0
                    step = AgentStep(
                        turn=current_turn,
                        thought=thought,
                        duration_sec=time.monotonic() - turn_start,
                        is_finish=True,
                        finish_summary=finish_summary or "Task finished.",
                    )
                    steps.append(step)
                    final_status = "COMPLETED"
                    final_summary = finish_summary or "Task completed successfully."
                    return MacroAgentResult(
                        success=True,
                        status=final_status,
                        summary=final_summary,
                        turns=current_turn,
                        duration_sec=time.monotonic() - start_time,
                        steps=steps,
                        total_input_tokens=total_in_tokens or None,
                        total_output_tokens=total_out_tokens or None,
                    )

                # 7. Handle malformed response recovery
                if parse_err:
                    if on_progress:
                        on_progress(f"  [Turn {current_turn}] Model response syntax error: {parse_err}")
                    consecutive_malformed += 1
                    step = AgentStep(
                        turn=current_turn,
                        thought=thought,
                        duration_sec=time.monotonic() - turn_start,
                    )
                    steps.append(step)

                    if consecutive_malformed >= self.config.max_consecutive_malformed:
                        final_status = "MALFORMED_OUTPUT_EXHAUSTED"
                        error_msg = f"Exceeded {self.config.max_consecutive_malformed} consecutive malformed model responses."
                        final_summary = f"Terminated: {error_msg}"
                        break

                    # Inject recovery guidance note
                    guidance = (
                        f"Turn {current_turn} parser error: {parse_err} "
                        "You must format your action as <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> "
                        "or conclude with <finish_task>summary</finish_task>."
                    )
                    context_mgr.add_system_note(guidance)
                    continue

                # Valid tool call encountered: reset malformed & premature counters
                consecutive_malformed = 0
                consecutive_premature = 0

                # 8. Loop Prevention: Detect repeated identical failed calls
                call_sig = (tool_name or "", json.dumps(tool_args, sort_keys=True))

                # 9. Execute tool through registry
                if on_progress:
                    clean_args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in tool_args.items()) if tool_args else ""
                    on_progress(f"  [Turn {current_turn}] Tool invocation: {tool_name}({clean_args_str})")
                tool_result: ToolResult = active_registry.execute(tool_name or "", context, tool_args)
                step_duration = time.monotonic() - turn_start
                if on_progress:
                    res_tag = "SUCCESS" if tool_result.success else f"FAILED ({tool_result.error})"
                    on_progress(f"  [Turn {current_turn}] Tool result: {res_tag}")

                # Evaluate loop condition on failures
                if not tool_result.success:
                    if call_sig == last_failed_signature:
                        identical_failed_count += 1
                        if identical_failed_count >= self.config.max_identical_failed_calls:
                            final_status = "REPEATED_FAILURE_EXHAUSTED"
                            error_msg = f"Terminated: Tool '{tool_name}' failed {identical_failed_count} consecutive times with identical arguments."
                            final_summary = error_msg
                            step = AgentStep(
                                turn=current_turn,
                                thought=thought,
                                tool_name=tool_name,
                                tool_args=tool_args,
                                tool_result=tool_result,
                                duration_sec=step_duration,
                            )
                            steps.append(step)
                            break
                        else:
                            context_mgr.add_system_note(
                                f"Loop warning: Tool '{tool_name}' failed with the exact same arguments on Turn {current_turn - 1}. "
                                "You must alter your arguments or select a different tool."
                            )
                    else:
                        last_failed_signature = call_sig
                        identical_failed_count = 1
                else:
                    last_failed_signature = None
                    identical_failed_count = 0

                # Record step in trajectory
                step = AgentStep(
                    turn=current_turn,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result,
                    duration_sec=step_duration,
                )
                steps.append(step)

                # Feed turn into sliding window context
                context_mgr.add_turn(
                    turn_index=current_turn,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_output=tool_result.output if tool_result.success else (tool_result.error or tool_result.output),
                    tool_success=tool_result.success,
                )

            # Loop ended without completion signal
            if final_status == "TURN_LIMIT_EXCEEDED" and not final_summary:
                final_summary = f"Reached maximum iteration limit of {self.config.max_turns} turns without task completion."

            return MacroAgentResult(
                success=(final_status == "COMPLETED"),
                status=final_status,
                summary=final_summary,
                turns=current_turn,
                duration_sec=time.monotonic() - start_time,
                steps=steps,
                total_input_tokens=total_in_tokens or None,
                total_output_tokens=total_out_tokens or None,
                error_message=error_msg,
            )

        finally:
            self._execution_lock.release()
            if getattr(self.config, "unload_on_exit", True) and self._transport is None:
                try:
                    OllamaTransport.unload(self.config, timeout=5.0)
                except Exception as exc:
                    logger.debug("Cleanup exception caught: %s", exc)
