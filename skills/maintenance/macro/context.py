"""
skills/maintenance/macro/context.py
Sliding-Window Context Management for MACRO (AutoPhage 2.0 Phase 2).

Design invariants:
- Pinned System Instructions: Never pruned or truncated.
- Pinned Task Specification: Objective, target files, acceptance criteria are never lost.
- Sliding Window with Summarization: When context budget is approached, older intermediate
  tool outputs are compressed while keeping recent turns in full fidelity.
- Pinned Findings & Notes: Accumulated discoveries and safety notes preserved.
- Bounded token/character budget aligned with local model context limits (e.g. 4096 / 8192 tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MacroContextConfig:
    """Configuration for MACRO prompt context budgeting and sliding window."""
    max_context_chars: int = 24_000  # ~6,000 tokens safe for 8k/4k context
    min_recent_turns: int = 2        # Keep at least last N turns in full detail
    max_tool_output_chars: int = 1500  # Cap on raw tool outputs stored in historical turns
    summary_snippet_chars: int = 200   # Length of compressed tool output for pruned turns


@dataclass
class ContextTurn:
    """Represents a single iteration turn in the agent's interaction history."""
    turn_index: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    tool_success: bool = True
    is_pruned: bool = False


class MacroContextManager:
    """
    Manages the multi-turn interaction context for MACRO.
    Enforces pinned system/task prompts while dynamically compressing and sliding
    older turns to stay within the local model's context budget.
    """

    def __init__(
        self,
        system_prompt: str,
        task_objective: str,
        config: Optional[MacroContextConfig] = None,
    ) -> None:
        self.config = config or MacroContextConfig()
        self.system_prompt = system_prompt.strip()
        self.task_objective = task_objective.strip()
        self.turns: list[ContextTurn] = []
        self.system_notes: list[str] = []

    def add_system_note(self, note: str) -> None:
        """Inject a transient or persistent system guidance note into the context."""
        cleaned = note.strip()
        if cleaned and cleaned not in self.system_notes:
            self.system_notes.append(cleaned)

    def add_turn(
        self,
        turn_index: int,
        thought: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict[str, Any]] = None,
        tool_output: str = "",
        tool_success: bool = True,
    ) -> None:
        """Record an executed turn with thought, tool call, and tool result."""
        # Cap raw tool output if excessively large
        capped_output = tool_output.strip()
        if len(capped_output) > self.config.max_tool_output_chars:
            capped_output = (
                capped_output[:self.config.max_tool_output_chars]
                + f"\n... [Output truncated for turn context ({len(tool_output)} total chars)]"
            )

        turn = ContextTurn(
            turn_index=turn_index,
            thought=thought.strip(),
            tool_name=tool_name,
            tool_args=tool_args or {},
            tool_output=capped_output,
            tool_success=tool_success,
            is_pruned=False,
        )
        self.turns.append(turn)

    def _render_turn(self, turn: ContextTurn, full_detail: bool) -> str:
        """Render a single turn into text formatted for model consumption."""
        parts = [f"--- Turn {turn.turn_index} ---"]
        if turn.thought:
            parts.append(f"<thought>\n{turn.thought}\n</thought>")

        if turn.tool_name:
            import json
            args_str = json.dumps(turn.tool_args)
            parts.append(f"<tool_call>\n{{\"name\": \"{turn.tool_name}\", \"arguments\": {args_str}}}\n</tool_call>")

            status_str = "SUCCESS" if turn.tool_success else "FAILED"
            if full_detail and not turn.is_pruned:
                body = turn.tool_output or "[No output]"
            else:
                snippet = turn.tool_output[:self.config.summary_snippet_chars].replace("\n", " ").strip()
                body = f"[Summary ({status_str}): {snippet}...]"

            parts.append(f"<tool_result name=\"{turn.tool_name}\" status=\"{status_str}\">\n{body}\n</tool_result>")

        return "\n".join(parts)

    def build_prompt(self) -> str:
        """
        Assemble the complete prompt payload.
        If character budget is exceeded, progressively compresses older turns.
        """
        pinned_header = (
            f"=== SYSTEM INSTRUCTIONS ===\n{self.system_prompt}\n\n"
            f"=== TASK OBJECTIVE ===\n{self.task_objective}\n"
        )

        notes_section = ""
        if self.system_notes:
            notes_section = "\n=== GUIDANCE NOTES ===\n" + "\n".join(f"- {n}" for n in self.system_notes) + "\n"

        if not self.turns:
            return f"{pinned_header}{notes_section}\n=== INTERACTION HISTORY ===\nNo prior actions taken."

        # Pass 1: Render all turns in full detail
        rendered_turns = [self._render_turn(t, full_detail=not t.is_pruned) for t in self.turns]
        history_body = "\n\n".join(rendered_turns)
        full_candidate = f"{pinned_header}{notes_section}\n=== INTERACTION HISTORY ===\n{history_body}\n\n=== NEXT ACTION ==="

        if len(full_candidate) <= self.config.max_context_chars:
            return full_candidate

        # Pass 2: Prune older turns (beyond min_recent_turns) to concise summaries
        num_turns = len(self.turns)
        prune_cutoff = max(0, num_turns - self.config.min_recent_turns)

        for i in range(prune_cutoff):
            self.turns[i].is_pruned = True

        rendered_turns = [self._render_turn(t, full_detail=not t.is_pruned) for t in self.turns]
        history_body = "\n\n".join(rendered_turns)
        full_candidate = f"{pinned_header}{notes_section}\n=== INTERACTION HISTORY ===\n{history_body}\n\n=== NEXT ACTION ==="

        if len(full_candidate) <= self.config.max_context_chars:
            return full_candidate

        # Pass 3: Hard window sliding (omit oldest pruned turns completely)
        retained_turns = list(self.turns)
        omitted_count = 0

        while len(full_candidate) > self.config.max_context_chars and len(retained_turns) > self.config.min_recent_turns:
            retained_turns.pop(0)
            omitted_count += 1
            omitted_marker = f"[... Older turns 1 to {omitted_count} omitted for context budget ...]\n\n"
            rendered = [self._render_turn(t, full_detail=not t.is_pruned) for t in retained_turns]
            history_body = omitted_marker + "\n\n".join(rendered)
            full_candidate = f"{pinned_header}{notes_section}\n=== INTERACTION HISTORY ===\n{history_body}\n\n=== NEXT ACTION ==="

        # Pass 4: If still over budget, compact tool outputs even in retained recent turns
        if len(full_candidate) > self.config.max_context_chars and retained_turns:
            rendered = [self._render_turn(t, full_detail=False) for t in retained_turns]
            omitted_marker = f"[... Older turns 1 to {omitted_count} omitted for context budget ...]\n\n" if omitted_count > 0 else ""
            history_body = omitted_marker + "\n\n".join(rendered)
            full_candidate = f"{pinned_header}{notes_section}\n=== INTERACTION HISTORY ===\n{history_body}\n\n=== NEXT ACTION ==="

        return full_candidate

    def get_history_summary(self) -> list[dict[str, Any]]:
        """Return structured turn metadata for diagnostics and reporting."""
        return [
            {
                "turn": t.turn_index,
                "thought": t.thought[:200],
                "tool": t.tool_name,
                "success": t.tool_success,
                "is_pruned": t.is_pruned,
            }
            for t in self.turns
        ]
