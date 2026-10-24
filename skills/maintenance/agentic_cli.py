"""
skills/maintenance/agentic_cli.py
AutoPhage 3.0 Agentic CLI.
An interactive, stream-centric workspace for MayaVE maintenance.
"""
from __future__ import annotations
import sys
import os
import time
import json
import uuid
from pathlib import Path
from typing import Any, Optional, List, Dict
from datetime import datetime, timezone

# UI Imports
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns
from rich.box import ROUNDED
from rich.console import Group

# Internal Imports
try:
    from .ledger import MaintenanceLedger, TaskEntry, TaskStatus
    from .hub import CoordinatorAPI
    from .store import StateStore
    from .models import TaskRisk, utc_now_iso
    from .operator_controls import OperatorController
    from .trace import TraceEvent, read_trace_body
except ImportError:
    from ledger import MaintenanceLedger, TaskEntry, TaskStatus
    from hub import CoordinatorAPI
    from store import StateStore
    from models import TaskRisk, utc_now_iso
    from operator_controls import OperatorController
    from trace import TraceEvent, read_trace_body

# Color Palette (Preserved from TUI)
PURPLE = "rgb(180,156,255)"
GRAY = "rgb(160,157,165)"
WHITE = "white"
MINT = "rgb(128,222,184)"
AMBER = "rgb(244,201,125)"
CORAL = "rgb(255,128,146)"
ACCENT = f"bold {PURPLE}"

class AgenticCLIState:
    """State management for the AutoPhage 3.0 CLI workspace."""
    def __init__(self, repo_root: Path, operate: bool = False):
        self.repo_root = repo_root.resolve()
        self.maintenance_dir = self.repo_root / ".maya" / "maintenance"
        self.store = StateStore(self.maintenance_dir / "state.json")
        self.controller = OperatorController(repo_root=self.repo_root)
        self.operate_mode = operate
        self.active_task_id: Optional[str] = None
        self.scroll_offset = 0
        self.follow = True
        self.paused = False
        self.modal_mode: Optional[str] = None # "APPROVAL", "INSPECT", "MESSAGE"
        self.modal_data: Any = None
        self.status_banner = ""
        self.last_banner_time = 0.0

    def get_active_session_info(self) -> dict:
        session_file = self.maintenance_dir / "active_session.json"
        if session_file.exists():
            try:
                return json.loads(session_file.read_text(encoding="utf-8"))
            except Exception: pass
        return {}

    def get_ledger(self) -> MaintenanceLedger:
        return self.store.load_ledger()

class StreamRenderer:
    """Renders the agentic interaction stream."""

    @staticmethod
    def render_worker_block(role: str, content: str, style: str = WHITE) -> Panel:
        """Renders a block of activity from a specific worker role."""
        role_map = {
            "planner": "[Planner]",
            "macro": "[MacroAgent]",
            "reviewer": "[Reviewer]",
            "coordinator": "[Coordinator]",
        }
        label = role_map.get(role.lower(), f"[{role}]")

        header = Text(f"{label} ", style=ACCENT)
        return Panel(
            Text(content, style=style),
            title=header,
            title_align="left",
            border_style=GRAY,
            box=ROUNDED,
            padding=(0, 1)
        )

    @staticmethod
    def render_diff_block(diff_text: str) -> Panel:
        """Renders a git-style diff with additions/removals colored."""
        text = Text()
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                text.append(line + "\n", style="green")
            elif line.startswith("-") and not line.startswith("---"):
                text.append(line + "\n", style="dim red")
            else:
                text.append(line + "\n", style=GRAY)

        return Panel(text, title=Text("File Edit", style=GRAY), border_style=GRAY, box=ROUNDED)

    @staticmethod
    def render_status_dock(state: AgenticCLIState) -> Columns:
        """Renders the persistent status dock at the bottom."""
        status = "IDLE"
        color = WHITE

        if state.active_task_id:
            ledger = state.get_ledger()
            task = ledger.get(state.active_task_id)
            if task:
                status = task.status.value
                if status == "IN_PROGRESS": color = PURPLE
                elif status == "VERIFIED": color = MINT
                elif status == "FAILED": color = CORAL
                elif status == "NEEDS_APPROVAL": color = AMBER

        dock = Text()
        dock.append(f" STATUS: {status} ", style=f"bold {color}")
        dock.append(f"  |  MODE: {'OPERATE' if state.operate_mode else 'OBSERVE'} ", style=GRAY)
        dock.append(f"  |  TABS: Tab to cycle views", style=GRAY)

        return Columns([Rule(characters="─", style=GRAY), dock])

def render_header(state: AgenticCLIState) -> Columns:
    """Renders the EVE logo and session header."""
    logo = Text()
    logo.append(" ▐▛███▛█   AutoPhage 3.0\n", style=ACCENT)
    logo.append("▝▜██████▀  gemma4:31b-cloud · medium effort · Token usage\n", style=WHITE)

    session = state.get_active_session_info()
    workdir = state.repo_root

    return Columns([
        logo,
        Text(f" {workdir}", style=GRAY),
        Rule(characters="=", style=PURPLE)
    ])

def render_main_view(state: AgenticCLIState) -> Layout:
    """The primary agentic interaction stream."""
    layout = Layout()

    trace_dir = state.maintenance_dir / "trace"
    events = []
    if trace_dir.exists():
        latest_trace = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if latest_trace:
            try:
                with open(latest_trace[0], "r", encoding="utf-8") as f:
                    for line in f:
                        ev = json.loads(line)
                        events.append(ev)
            except Exception: pass

    stream_content = []
    for ev in events[-20:]:
        role = ev.get("sender", "unknown").split(":")[0]
        summary = ev.get("summary", "")

        if "diff" in ev.get("type", "").lower() or "edit" in ev.get("type", "").lower():
            diff = ev.get("payload", {}).get("diff", "No diff available")
            stream_content.append(StreamRenderer.render_diff_block(diff))
        elif "tool" in ev.get("type", "").lower():
            stream_content.append(StreamRenderer.render_worker_block(role, f"Tool Call: {summary}", style=GRAY))
        else:
            stream_content.append(StreamRenderer.render_worker_block(role, summary))

    if not stream_content:
        stream_content.append(Panel(Text("No active agentic stream. Start a task to begin.", style=GRAY), border_style=GRAY))

    layout.split_column(
        Layout(Group(*stream_content), ratio=1),
        Layout(StreamRenderer.render_status_dock(state), size=2)
    )
    return layout

def run_agentic_cli(repo_root: Path, operate: bool = False):
    """Entry point for the AutoPhage 3.0 Agentic CLI."""
    console = Console()
    state = AgenticCLIState(repo_root, operate=operate)

    try:
        # Create a base layout and then split it
        main_layout = Layout()
        main_layout.split_column(
            Layout(render_header(state), size=3),
            Layout(render_main_view(state), ratio=1)
        )

        with Live(
            main_layout,
            console=console,
            refresh_per_second=4,
            screen=True
        ) as live:
            while True:
                live.update(main_layout)
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--operate", action="store_true")
    args = parser.parse_args()
    run_agentic_cli(Path.cwd(), operate=args.operate)
