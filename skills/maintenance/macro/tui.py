"""
skills/maintenance/macro/tui.py
Minimalist Claude Code-Style Rich TUI for AutoPhage 2.0 (Phase 5).

Strictly adheres to the project color palette:
- Purple: rgb(180,156,255)  - brand, highlight, active selection
- Gray:   rgb(160,157,165)  - secondary text, rules, borders, metadata
- White:  white             - primary text, commands, outputs
- Mint:   rgb(128,222,184)  - verified, passing tests, clean worktrees
- Amber:  rgb(244,201,125)  - warnings, pending approvals, in-progress tasks
- Coral:  rgb(255,128,146)  - blocked, failures, errors, recovery required
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

# Mandatory color palette
PURPLE = "rgb(180,156,255)"
GRAY = "rgb(160,157,165)"
WHITE = "white"
MINT = "rgb(128,222,184)"
AMBER = "rgb(244,201,125)"
CORAL = "rgb(255,128,146)"
ACCENT = f"bold {PURPLE}"

GLYPH_REPLACEMENTS = {
    "\u2713": "[OK]",
    "\u2714": "[OK]",
    "\u26a0": "[!]",
    "\u2717": "[X]",
    "\u2718": "[X]",
    "\u2022": "*",
    "\u25b6": ">>",
    "\u2500": "-",
    "\u2500\u2500": "--",
    "\u2014": "-",
    "\u2013": "-",
    "\u2026": "...",
}


def to_ascii_safe(text: Any, target_encoding: str = "cp1252") -> str:
    """
    Convert text to an ASCII-safe representation under Windows console encodings (e.g. cp1252).
    Maps common Unicode status glyphs to ASCII equivalents and replaces unencodable characters.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    for glyph, repl in GLYPH_REPLACEMENTS.items():
        if glyph in text:
            text = text.replace(glyph, repl)
    try:
        text.encode(target_encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode("ascii", errors="replace").decode("ascii")


def get_status_style(status_value: str) -> str:
    """Return matching palette color style for a status string."""
    val = str(status_value).upper()
    if any(k in val for k in ("FAILED", "BLOCKED", "ERROR", "CORRUPT", "CRITICAL", "UNAVAILABLE")):
        return CORAL
    if any(k in val for k in ("IN_PROGRESS", "PLANNING", "VERIFYING", "NEEDS_APPROVAL", "REVIEW_PENDING", "PAUSED", "WARNING", "REQUEST_CHANGES")):
        return AMBER
    if any(k in val for k in ("READY_TO_INTEGRATE", "VERIFIED", "COMPLETED", "APPROVE", "PASS", "CLEAN", "AVAILABLE")):
        return MINT
    return GRAY


class MacroTUI:
    """
    Minimalist, keyboard-first Claude Code-style Rich interface for AutoPhage 2.0.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console(highlight=False)

    def print_banner(self, branch: str = "main", repo_name: str = "VE11") -> None:
        """Render clean, compact Claude Code-style header."""
        title = Text.assemble(
            ("-- ", GRAY),
            ("AutoPhage 2.0", f"bold {PURPLE}"),
            (" - MACRO Maintenance Agent ", GRAY),
            (f"[{repo_name} / {branch}]", PURPLE),
            (" --", GRAY),
        )
        self.console.print()
        self.console.print(title)
        self.console.print(
            Text(
                "Type a command or slash shortcut (/help, /scan, /fix, /improve, /review, /status, /resume, /quit)",
                style=GRAY,
            )
        )
        self.console.print()

    def print_help(self) -> None:
        """Render concise usage and command reference."""
        table = Table(
            box=box.SIMPLE_HEAD,
            header_style=f"bold {PURPLE}",
            show_edge=False,
            padding=(0, 2),
        )
        table.add_column("Command", style=f"bold {WHITE}")
        table.add_column("Description", style=GRAY)
        table.add_column("Example", style=PURPLE)

        table.add_row(
            "scan",
            "Progressive repository discovery and Markdown backlog generation",
            "autophage scan [--subsystem core]",
        )
        table.add_row(
            "fix <findings.md>",
            "Execute multi-issue campaign from Markdown backlog (Mode B)",
            "autophage fix .maya/maintenance/findings.md",
        )
        table.add_row(
            "fix --issue <id>",
            "Execute single issue in isolated worktree with review (Mode A)",
            "autophage fix --issue BUG-001 [--backlog findings.md]",
        )
        table.add_row(
            "improve",
            "Discover and prioritize improvement opportunities (read-only audit)",
            "autophage improve [--focus skills]",
        )
        table.add_row(
            "review [target]",
            "Perform read-only review of a diff file, branch, or worktree",
            "autophage review candidate.diff",
        )
        table.add_row(
            "status",
            "Show campaign state, active worktrees, and model availability",
            "autophage status",
        )
        table.add_row(
            "resume",
            "Safely resume interrupted campaign skipping verified tasks",
            "autophage resume",
        )
        table.add_row(
            "integrate [task_id]",
            "Integrate verified tasks into dedicated integration branch",
            "autophage integrate [BUG-001] [--authorize]",
        )
        table.add_row(
            "help",
            "Show this command reference",
            "/help",
        )
        table.add_row(
            "quit / exit",
            "Exit interactive shell",
            "/quit",
        )

        self.console.print(table)
        self.console.print()

    def print_task_header(
        self,
        task_id: str,
        title: str,
        priority: str,
        risk: str,
        target_files: Sequence[str] = (),
        test_targets: Sequence[str] = (),
    ) -> None:
        """Print task execution header in Claude Code style."""
        self.console.print(
            Text.assemble(
                (">> ", f"bold {PURPLE}"),
                (f"[{to_ascii_safe(task_id)}] ", f"bold {WHITE}"),
                (f"{to_ascii_safe(title)} ", WHITE),
                (f"(Priority: {to_ascii_safe(priority)}, Risk: {to_ascii_safe(risk)})", GRAY),
            )
        )
        targets_str = ", ".join(to_ascii_safe(t) for t in target_files) if target_files else "none specified"
        tests_str = ", ".join(to_ascii_safe(t) for t in test_targets) if test_targets else "none specified"
        self.console.print(
            Text(f"  Target: {targets_str} | Test: {tests_str}", style=GRAY)
        )

    def print_step(self, message: str, step_type: str = "info") -> None:
        """Print an execution step with matching palette styling."""
        safe_msg = to_ascii_safe(message)
        if step_type == "tool":
            self.console.print(Text(f"  tool: {safe_msg}", style=GRAY))
        elif step_type in ("pass", "verified", "ok"):
            self.console.print(Text(f"  [OK] {safe_msg}", style=MINT))
        elif step_type in ("warn", "approval"):
            self.console.print(Text(f"  [!] {safe_msg}", style=AMBER))
        elif step_type in ("fail", "error"):
            self.console.print(Text(f"  [X] {safe_msg}", style=CORAL))
        else:
            self.console.print(Text(f"  * {safe_msg}", style=GRAY))

    def print_task_result(
        self,
        task_id: str,
        status: str,
        tier: str = "",
        attempts: int = 1,
        details: str = "",
    ) -> None:
        """Print final state of task execution."""
        style = get_status_style(status)
        symbol = "[OK]" if style == MINT else ("[X]" if style == CORAL else "[!]")
        safe_id = to_ascii_safe(task_id)
        safe_stat = to_ascii_safe(status)
        safe_tier = to_ascii_safe(tier)
        safe_details = to_ascii_safe(details)
        msg = f"{symbol} [{safe_id}] Status: {safe_stat} (Tier: {safe_tier}, Attempts: {attempts})"
        if safe_details:
            msg += f" - {safe_details}"
        self.console.print(Text(f"  {msg}", style=style))
        self.console.print()

    def print_status(
        self,
        checkpoint: Any,
        worktrees: Sequence[str] = (),
        model_statuses: Optional[dict[str, str]] = None,
    ) -> None:
        """Render campaign state, tasks table, and worker health."""
        if not checkpoint:
            self.console.print(Text("No active campaign found in state.", style=GRAY))
            self.console.print()
        else:
            camp_status = getattr(checkpoint, "status", "UNKNOWN")
            camp_status_val = getattr(camp_status, "value", str(camp_status))
            c_style = get_status_style(camp_status_val)

            summary_text = Text.assemble(
                ("Campaign ID: ", GRAY),
                (str(getattr(checkpoint, "campaign_id", "N/A")), f"bold {WHITE}"),
                ("  Status: ", GRAY),
                (camp_status_val, f"bold {c_style}"),
                ("  Completed: ", GRAY),
                (str(getattr(checkpoint, "completed_count", 0)), MINT),
                ("  Failed: ", GRAY),
                (str(getattr(checkpoint, "failed_count", 0)), CORAL),
                ("  Blocked: ", GRAY),
                (str(getattr(checkpoint, "blocked_count", 0)), AMBER),
            )
            self.console.print(summary_text)
            self.console.print()

            # Tasks table
            tasks = getattr(checkpoint, "tasks", {})
            if tasks:
                table = Table(
                    box=box.SIMPLE_HEAD,
                    header_style=f"bold {PURPLE}",
                    show_edge=False,
                    padding=(0, 1),
                )
                table.add_column("Task ID", style=f"bold {WHITE}")
                table.add_column("Priority", style=GRAY)
                table.add_column("Risk", style=GRAY)
                table.add_column("Tier", style=GRAY)
                table.add_column("Status", style=WHITE)
                table.add_column("Att.", justify="right", style=GRAY)
                table.add_column("Notes / Reason", style=GRAY)

                for tid, t in tasks.items():
                    stat_val = getattr(getattr(t, "status", ""), "value", str(getattr(t, "status", "")))
                    prio_val = getattr(getattr(t, "priority", ""), "value", str(getattr(t, "priority", "")))
                    risk_val = getattr(getattr(t, "risk", ""), "value", str(getattr(t, "risk", "")))
                    tier_val = getattr(getattr(t, "worker_tier", ""), "value", str(getattr(t, "worker_tier", "")))
                    notes = to_ascii_safe(getattr(t, "blocking_reason", "") or getattr(t, "failure_reason", "") or "")

                    table.add_row(
                        to_ascii_safe(tid),
                        to_ascii_safe(prio_val),
                        to_ascii_safe(risk_val),
                        to_ascii_safe(tier_val),
                        Text(to_ascii_safe(stat_val), style=get_status_style(stat_val)),
                        str(getattr(t, "attempts", 0)),
                        notes[:60] + ("..." if len(notes) > 60 else ""),
                    )

                self.console.print(table)
                self.console.print()

        # Active worktrees
        if worktrees:
            self.console.print(Text.assemble(("Active Worktrees: ", GRAY), (to_ascii_safe(", ".join(worktrees)), WHITE)))
            self.console.print()

        # Model policy status
        if model_statuses:
            m_table = Table(
                box=box.SIMPLE_HEAD,
                header_style=f"bold {PURPLE}",
                show_edge=False,
                padding=(0, 2),
            )
            m_table.add_column("Role", style=f"bold {WHITE}")
            m_table.add_column("Configured Model / Executable", style=GRAY)
            m_table.add_column("Status", style=WHITE)

            for role, info in model_statuses.items():
                target, status_val = info.split("|") if "|" in info else (info, "UNKNOWN")
                m_table.add_row(
                    to_ascii_safe(role),
                    to_ascii_safe(target),
                    Text(to_ascii_safe(status_val), style=get_status_style(status_val)),
                )
            self.console.print(m_table)
            self.console.print()

    def print_findings(self, findings: Sequence[Any], report_path: Optional[Path] = None) -> None:
        """Render discovery findings table."""
        if not findings:
            self.console.print(Text("[OK] No defects or issues discovered.", style=MINT))
            self.console.print()
            return

        table = Table(
            box=box.SIMPLE_HEAD,
            header_style=f"bold {PURPLE}",
            show_edge=False,
            padding=(0, 1),
        )
        table.add_column("Issue ID", style=f"bold {WHITE}")
        table.add_column("Subsystem", style=GRAY)
        table.add_column("Severity", style=WHITE)
        table.add_column("Confidence", style=GRAY)
        table.add_column("Location", style=GRAY)
        table.add_column("Summary", style=WHITE)

        for f in findings:
            sev_val = getattr(getattr(f, "severity", ""), "value", str(getattr(f, "severity", "")))
            conf_val = getattr(getattr(f, "confidence", ""), "value", str(getattr(f, "confidence", "")))
            locs = (
                getattr(f, "line_references", None)
                or getattr(f, "target_files", None)
                or getattr(f, "affected_locations", None)
                or []
            )
            loc_str = str(locs[0]) if locs else ""
            desc = getattr(f, "title", "") or getattr(f, "description", "") or getattr(f, "actual_behavior", "")

            table.add_row(
                to_ascii_safe(getattr(f, "issue_id", "N/A")),
                to_ascii_safe(getattr(f, "subsystem", "unknown")),
                Text(to_ascii_safe(sev_val), style=get_status_style(sev_val)),
                Text(to_ascii_safe(conf_val), style=get_status_style(conf_val)),
                to_ascii_safe(loc_str[:40]),
                to_ascii_safe(desc[:50] + ("..." if len(desc) > 50 else "")),
            )

        self.console.print(table)
        self.console.print()

        if report_path:
            self.console.print(
                Text.assemble(
                    ("Findings written to: ", GRAY),
                    (to_ascii_safe(str(report_path)), PURPLE),
                )
            )
            self.console.print()

    def print_review(self, recommendation: str, summary: str, findings: Sequence[Any] = ()) -> None:
        """Render independent code review report."""
        r_style = get_status_style(recommendation)
        self.console.print(
            Text.assemble(
                ("Review Recommendation: ", f"bold {WHITE}"),
                (to_ascii_safe(recommendation), f"bold {r_style}"),
            )
        )
        if summary:
            self.console.print(Text(f"Summary: {to_ascii_safe(summary)}", style=GRAY))
        self.console.print()

        if findings:
            table = Table(
                box=box.SIMPLE_HEAD,
                header_style=f"bold {PURPLE}",
                show_edge=False,
                padding=(0, 1),
            )
            table.add_column("Severity", style=WHITE)
            table.add_column("Category", style=GRAY)
            table.add_column("File / Location", style=GRAY)
            table.add_column("Message", style=WHITE)

            for f in findings:
                sev = getattr(f, "severity", "WARNING")
                cat = getattr(f, "category", "")
                fl = getattr(f, "file", "")
                msg = getattr(f, "message", "")
                table.add_row(
                    Text(to_ascii_safe(sev), style=get_status_style(sev)),
                    to_ascii_safe(cat),
                    to_ascii_safe(fl),
                    to_ascii_safe(msg[:60]),
                )

            self.console.print(table)
            self.console.print()

    def print_integration_candidates(self, candidates: Sequence[dict]) -> None:
        """Render table of tasks eligible for integration (READY_TO_INTEGRATE)."""
        table = Table(
            box=box.SIMPLE_HEAD,
            header_style=f"bold {PURPLE}",
            show_edge=False,
            padding=(0, 2),
        )
        table.add_column("Task ID", style=f"bold {WHITE}")
        table.add_column("Pri", style=GRAY)
        table.add_column("Risk", style=GRAY)
        table.add_column("Status", style=MINT)
        table.add_column("Branch", style=PURPLE)
        table.add_column("Target Files", style=GRAY)
        table.add_column("Title", style=WHITE)

        for c in candidates:
            targets_str = ", ".join(c.get("target_files", []))[:30]
            table.add_row(
                to_ascii_safe(c.get("task_id", "")),
                to_ascii_safe(c.get("priority", "")),
                to_ascii_safe(c.get("risk", "")),
                to_ascii_safe(c.get("status", "")),
                to_ascii_safe(c.get("branch_name", "")),
                to_ascii_safe(targets_str),
                to_ascii_safe(c.get("title", "")[:40]),
            )

        self.console.print(table)
        self.console.print()

    def print_integration_diff(self, info: dict, full_diff: bool = False) -> None:
        """Display proposed patch diff and task verification details for human review."""
        task_id = info.get("task_id", "")
        title = info.get("title", "")
        branch = info.get("branch", "")
        changed = ", ".join(info.get("changed_files", []))
        diff_text = info.get("full_diff" if full_diff else "diff_preview", "")

        tip_sha = info.get("branch_tip_sha", "")
        target_b = info.get("target_branch", "")
        header = f"[bold {PURPLE}]Candidate Integration:[/] [bold {WHITE}]{task_id}[/] - {title}\n"
        meta_parts = [
            f"[{GRAY}]Source Branch:[/] [{WHITE}]{branch}[/]",
        ]
        if tip_sha:
            meta_parts.append(f"[{GRAY}]Tip SHA:[/] [{WHITE}]{tip_sha}[/]")
        if target_b:
            meta_parts.append(f"[{GRAY}]Target Branch:[/] [{WHITE}]{target_b}[/]")
        meta_parts.append(f"[{GRAY}]Changed Files:[/] [{WHITE}]{changed}[/]")
        meta = "  |  ".join(meta_parts) + "\n"
        self.console.print(header + meta)

        if diff_text:
            diff_panel = Panel(
                Text(to_ascii_safe(diff_text), style=GRAY),
                title=f"Proposed Diff ({task_id})",
                border_style=PURPLE,
                box=box.ROUNDED,
                padding=(1, 2),
            )
            self.console.print(diff_panel)
        else:
            self.console.print(f"[{GRAY}]No diff content available.[/]\n")

    def interactive_repl(self, dispatcher: Callable[[list[str]], int]) -> int:
        """
        Interactive line-input loop (macro>).
        Falls back cleanly for non-interactive or headless environments.
        """
        if not sys.stdin.isatty():
            self.console.print(
                Text(
                    "Non-interactive terminal detected. Run commands with CLI arguments (e.g. autophage status).",
                    style=GRAY,
                )
            )
            return 0

        self.print_banner()

        while True:
            try:
                raw = input("macro> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                self.console.print(Text("Exiting AutoPhage.", style=GRAY))
                break

            if not raw:
                continue

            # Strip leading slash if user used slash-command syntax (/fix, /status, etc.)
            cmd_line = raw[1:] if raw.startswith("/") else raw
            tokens = cmd_line.split()
            if not tokens:
                continue

            command = tokens[0].lower()
            if command in ("quit", "exit", "q"):
                self.console.print(Text("Exiting AutoPhage.", style=GRAY))
                break
            if command == "help":
                self.print_help()
                continue

            try:
                dispatcher(tokens)
            except Exception as exc:
                self.console.print(Text(f"Command error: {exc}", style=CORAL))

        return 0
