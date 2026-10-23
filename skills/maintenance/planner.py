"""
Controlled Local Planner for MayaVE Autophage (P5).
Integrates local qwen2.5-coder:3b via Ollama for bounded task decomposition.
Produces structured, schema-compliant task proposals without autonomous self-authorization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from .models import PlanProposal, TaskRisk, TaskPriority, utc_now_iso
    from .task_contract import (
        CONTRACT_SCHEMA_VERSION,
        TaskContractValidationError,
        TaskContractValidator,
        ALLOWED_SUBSYSTEM_ROOTS,
        FORBIDDEN_TARGET_PREFIXES,
        FORBIDDEN_TARGET_EXACT,
    )
except ImportError:
    from models import PlanProposal, TaskRisk, TaskPriority, utc_now_iso
    from task_contract import (
        CONTRACT_SCHEMA_VERSION,
        TaskContractValidationError,
        TaskContractValidator,
        ALLOWED_SUBSYSTEM_ROOTS,
        FORBIDDEN_TARGET_PREFIXES,
        FORBIDDEN_TARGET_EXACT,
    )

logger = logging.getLogger("maya.maintenance.planner")


def compute_proposal_hash(objective: str, tasks: list[dict[str, Any]]) -> str:
    """Compute deterministic SHA-256 hash of canonical plan proposal content."""
    payload = json.dumps(
        {"objective": objective.strip(), "tasks": tasks},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for local qwen2.5-coder:3b planner engine."""
    model: str = "qwen2.5-coder:3b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.1
    max_context_tokens: int = 4096
    max_completion_tokens: int = 2048
    timeout_seconds: float = 120.0
    default_max_tasks: int = 5
    hard_max_tasks: int = 8


class PlannerPromptEngine:
    """Builds bounded, deterministic planning prompts embedding schema and safety rules."""

    @classmethod
    def collect_repository_summary(cls, repo_root: Path, max_chars: int = 3000) -> str:
        """Collect bounded inventory summary of allowed subsystems."""
        summary_lines = []
        try:
            for sub in sorted(ALLOWED_SUBSYSTEM_ROOTS):
                sub_dir = repo_root / sub.rstrip("/")
                if sub_dir.is_dir():
                    files = [
                        p.relative_to(repo_root).as_posix()
                        for p in sub_dir.rglob("*")
                        if p.is_file() and not p.name.startswith(".") and "__pycache__" not in p.parts
                    ]
                    # Sample top 8 files per subsystem
                    sample = files[:8]
                    more = f" ... (+{len(files) - 8} more)" if len(files) > 8 else ""
                    summary_lines.append(f"Subsystem '{sub}': {len(files)} files: {', '.join(sample)}{more}")
        except Exception as e:
            summary_lines.append(f"Subsystems: {', '.join(sorted(ALLOWED_SUBSYSTEM_ROOTS))}")

        full_summary = "\n".join(summary_lines)
        if len(full_summary) > max_chars:
            return full_summary[:max_chars] + "\n...(truncated context)"
        return full_summary

    @classmethod
    def build_prompt(
        cls,
        objective: str,
        repo_root: Path,
        max_tasks: int = 5,
        known_task_ids: set[str] | None = None,
    ) -> str:
        """Construct the prompt enforcing schema v1.0.0, bounded task counts, and test policy."""
        subsystem_summary = cls.collect_repository_summary(repo_root)
        task_cap = min(max_tasks, 8)
        existing_info = f"Existing Task IDs: {', '.join(sorted(known_task_ids))}" if known_task_ids else "No prior tasks."

        return f"""You are the MayaVE Maintenance Planner (Milestone P5).
Your objective is to decompose an explicit maintenance objective into a structured, dependency-ordered task plan.

=== MANDATORY SAFETY & CONTRACT RULES ===
1. You propose tasks ONLY. You do NOT execute tools, shell commands, or modify files.
2. Output MUST be valid JSON with this exact structure:
{{
  "objective": "{objective}",
  "tasks": [
    {{
      "task_id": "T5-01",
      "title": "Short descriptive title (5-140 chars)",
      "audit_source": "Explicit issue/source reference",
      "priority": "P1",
      "risk": "LOW",
      "description": "Clear explanation of required fix",
      "target_files": ["services/example.py"],
      "test_targets": [".maya/tests/test_example.py"],
      "dependencies": [],
      "acceptance_criteria": ["Criteria 1", "Criteria 2"]
    }}
  ]
}}
3. Task count limit: Output at most {task_cap} tasks (minimum 1, hard max 8).
4. Priorities: "P0", "P1", "P2", "P3".
5. Risk: Must be "LOW" or "MEDIUM" only. "HIGH" and "VERY_HIGH" are strictly forbidden.
6. Target Files: Must reside in allowed subsystems: {sorted(ALLOWED_SUBSYSTEM_ROOTS)}.
   NEVER target skills/maintenance/, .maya/maintenance/, state.json, cache.json, .env, or credentials.
7. Test Targets: All test files MUST strictly reside under '.maya/tests/'.
8. Dependencies: Must form an acyclic directed graph (no circular dependencies).
9. Output JSON ONLY. No conversational preamble, explanation, or markdown fences.

=== REPOSITORY SUBSYSTEMS ===
{subsystem_summary}

=== EXISTING CONTEXT ===
{existing_info}

=== MAINTENANCE OBJECTIVE ===
{objective.strip()}
"""


class LocalPlanner:
    """
    Controlled local Planner engine powered by qwen2.5-coder:3b.
    Validates and produces PlanProposal instances for coordinator review.
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
        repo_root: Path | None = None,
        transport: Optional[Callable[[str, PlannerConfig], tuple[str, Optional[int], Optional[int]]]] = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self._transport = transport

    def _call_ollama(self, prompt: str) -> tuple[str, Optional[int], Optional[int]]:
        """Call Ollama /api/generate with structured JSON format and bounded context."""
        if self._transport is not None:
            return self._transport(prompt, self.config)

        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        payload = json.dumps({
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "0",  # Release weights after planning to preserve LOQ workstation memory
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_completion_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (
                    data.get("response", ""),
                    data.get("prompt_eval_count"),
                    data.get("eval_count"),
                )
        except urllib.error.URLError as e:
            raise ConnectionError(f"Ollama connection error at {url}: {e.reason}") from e
        except TimeoutError as e:
            raise TimeoutError(f"Ollama request timed out after {self.config.timeout_seconds}s") from e

    def generate_plan(
        self,
        objective: str,
        max_tasks: int = 5,
        known_task_ids: set[str] | None = None,
    ) -> tuple[Optional[PlanProposal], list[str]]:
        """
        Generate and validate a structured task plan from an explicit maintenance objective.
        Returns (PlanProposal, list_of_errors). If errors occur, PlanProposal is None.
        """
        if not objective or not isinstance(objective, str) or not objective.strip():
            return None, ["Maintenance objective must be a non-empty string."]

        clean_objective = objective.strip()
        task_cap = min(max(1, max_tasks), self.config.hard_max_tasks)

        prompt = PlannerPromptEngine.build_prompt(
            objective=clean_objective,
            repo_root=self.repo_root,
            max_tasks=task_cap,
            known_task_ids=known_task_ids,
        )

        try:
            response_text, in_tokens, out_tokens = self._call_ollama(prompt)
        except Exception as e:
            return None, [f"Planner model invocation failed: {e}"]

        if not response_text or not response_text.strip():
            return None, ["Planner returned empty response."]

        # Clean markdown code fences if emitted by model
        raw_text = response_text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        try:
            parsed_json = json.loads(raw_text)
        except Exception as e:
            return None, [f"Planner generated invalid JSON: {e}"]

        # Extract tasks
        tasks: list[dict[str, Any]] = []
        if isinstance(parsed_json, dict):
            if "tasks" in parsed_json and isinstance(parsed_json["tasks"], list):
                tasks = parsed_json["tasks"]
            elif "task_id" in parsed_json:
                tasks = [parsed_json]
            else:
                return None, ["Planner JSON missing 'tasks' array or valid task objects."]
        elif isinstance(parsed_json, list):
            tasks = parsed_json
        else:
            return None, ["Planner JSON root must be an object or list."]

        # Validate through TaskContractValidator
        plan_payload = {"objective": clean_objective, "tasks": tasks}
        is_valid, validation_errors = TaskContractValidator.validate_plan(
            plan_payload,
            max_tasks=task_cap,
            known_task_ids=known_task_ids,
        )

        if not is_valid:
            return None, validation_errors

        proposal_hash = compute_proposal_hash(clean_objective, tasks)
        plan_id = f"PLAN-{uuid.uuid4().hex[:12].upper()}"

        proposal = PlanProposal(
            plan_id=plan_id,
            objective=clean_objective,
            tasks=tasks,
            status="PROPOSED",
            proposal_hash=proposal_hash,
            created_at=utc_now_iso(),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            max_tasks=task_cap,
        )

        return proposal, []
