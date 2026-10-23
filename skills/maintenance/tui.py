"""
skills/maintenance/tui.py
Trustworthy, Read-Only Terminal Dashboard for MayaVE Autophage (Phase 1).

Design invariants:
- Strictly read-only: never modifies state.json, never acquires StateStore locks,
  and never imports CoordinatorAPI mutating methods.
- Observes append-only trace JSONL logs (.maya/maintenance/trace/<session_id>.jsonl)
  and polls state.json / active_session.json.
- Handles partial final lines, missing trace files, and rotated sessions.
- Compact 3-panel Rich layout (Operations, Trace, Usage/Status).
- Windows non-blocking keyboard controls via msvcrt (space=pause, f=follow,
  i/Enter=inspect, /=search, t/w/c/s=filter, q=quit).
- Plain-text non-TTY fallback for CI, headless runs, or pipes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    from .trace import TraceEvent, read_trace_body, get_global_trace
except ImportError:
    from trace import TraceEvent, read_trace_body, get_global_trace


SEVERITY_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "CRIT"]
SEVERITY_ORDER = {sev: idx for idx, sev in enumerate(SEVERITY_LEVELS)}


@dataclass
class FilterConfig:
    task_id: Optional[str] = None
    worker_id: Optional[str] = None
    category: Optional[str] = None
    min_severity: str = "INFO"
    search_query: str = ""

    def matches(self, event: TraceEvent) -> bool:
        if self.task_id and event.task_id != self.task_id:
            return False
        if self.worker_id and event.worker_id != self.worker_id:
            return False
        if self.category and event.category != self.category:
            return False
        ev_sev = event.severity.upper() if event.severity else "INFO"
        ev_order = SEVERITY_ORDER.get(ev_sev, 1)
        min_order = SEVERITY_ORDER.get(self.min_severity.upper(), 1)
        if ev_order < min_order:
            return False
        if self.search_query:
            q = self.search_query.lower()
            in_summary = q in (event.summary or "").lower()
            in_type = q in (event.type or "").lower()
            in_sender = q in (event.sender or "").lower()
            if not (in_summary or in_type or in_sender):
                return False
        return True


class TraceTailReader:
    """Non-blocking, rotation-tolerant reader for JSONL trace files."""

    def __init__(self, trace_path: Optional[Path] = None, trace_dir: Optional[Path] = None) -> None:
        self.trace_path = trace_path
        self.trace_dir = trace_dir
        self.file_pos = 0
        self.line_buffer = ""
        self.last_mtime: float = 0.0

    def resolve_latest_trace_file(self) -> Optional[Path]:
        if self.trace_path and self.trace_path.is_file():
            return self.trace_path
        if not self.trace_dir or not self.trace_dir.is_dir():
            return None
        try:
            jsonl_files = sorted(
                self.trace_dir.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return jsonl_files[0] if jsonl_files else None
        except Exception:
            return None

    def read_new_events(self) -> list[TraceEvent]:
        target = self.resolve_latest_trace_file()
        if not target or not target.is_file():
            return []

        # If file rotated
        if self.trace_path and target != self.trace_path:
            self.trace_path = target
            self.file_pos = 0
            self.line_buffer = ""

        self.trace_path = target
        events: list[TraceEvent] = []

        try:
            stat = target.stat()
            # If file was truncated or replaced
            if stat.st_size < self.file_pos:
                self.file_pos = 0
                self.line_buffer = ""

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.file_pos)
                chunk = f.read()
                self.file_pos = f.tell()

            if not chunk:
                return []

            full_chunk = self.line_buffer + chunk
            lines = full_chunk.split("\n")
            # If chunk does not end in newline, the last segment is incomplete
            if not full_chunk.endswith("\n"):
                self.line_buffer = lines.pop()
            else:
                self.line_buffer = ""

            for line in lines:
                s = line.strip()
                if not s:
                    continue
                try:
                    data = json.loads(s)
                    if isinstance(data, dict):
                        events.append(TraceEvent.from_dict(data))
                except Exception:
                    # Ignore malformed or truncated lines
                    pass
        except Exception:
            pass

        return events


class DashboardState:
    """In-memory view state for Autophage terminal dashboard."""

    def __init__(self, repo_root: Path, session_id: Optional[str] = None) -> None:
        self.repo_root = repo_root.resolve()
        self.maintenance_dir = self.repo_root / ".maya" / "maintenance"
        self.trace_dir = self.maintenance_dir / "trace"
        self.session_id = session_id or ""

        # Trace events
        self.events: list[TraceEvent] = []
        self.tail_reader = TraceTailReader(
            trace_path=(self.trace_dir / f"{session_id}.jsonl") if session_id else None,
            trace_dir=self.trace_dir,
        )

        # View settings
        self.follow: bool = True
        self.paused: bool = False
        self.scroll_offset: int = 0  # 0 = tail
        self.selected_index: int = -1
        self.inspecting: bool = False
        self.inspection_content: str = ""
        self.inspection_title: str = ""

        # Filters
        self.filters = FilterConfig()

        # Cached state.json
        self.state_mtime: float = 0.0
        self.cached_tasks: dict[str, Any] = {}
        self.cached_budget: dict[str, Any] = {}

        # Cached active_session.json
        self.session_mtime: float = 0.0
        self.cached_session: dict[str, Any] = {}

        # Cumulative tokens from events (provider vs unknown)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.has_provider_tokens: bool = False

        self.last_poll_time = time.monotonic()
        self.lag_sec: float = 0.0

    def refresh(self) -> None:
        now = time.monotonic()
        self.lag_sec = max(0.0, now - self.last_poll_time)
        self.last_poll_time = now

        # Read trace events
        new_evs = self.tail_reader.read_new_events()
        if new_evs:
            for ev in new_evs:
                self.events.append(ev)
                # Aggregate tokens if provider-reported
                usage = ev.usage if isinstance(ev.usage, dict) else {}
                if usage.get("source") == "provider":
                    inp = usage.get("input_tokens")
                    out = usage.get("output_tokens")
                    if inp is not None:
                        self.total_input_tokens += int(inp)
                        self.has_provider_tokens = True
                    if out is not None:
                        self.total_output_tokens += int(out)
                        self.has_provider_tokens = True

        # Poll active_session.json
        session_file = self.maintenance_dir / "active_session.json"
        if session_file.is_file():
            try:
                mtime = session_file.stat().st_mtime
                if mtime != self.session_mtime:
                    self.session_mtime = mtime
                    self.cached_session = json.loads(session_file.read_text(encoding="utf-8", errors="replace"))
                    if not self.session_id and self.cached_session.get("session_id"):
                        self.session_id = str(self.cached_session["session_id"])
            except Exception:
                pass

        # Poll state.json
        state_file = self.maintenance_dir / "state.json"
        if state_file.is_file():
            try:
                mtime = state_file.stat().st_mtime
                if mtime != self.state_mtime:
                    self.state_mtime = mtime
                    raw = json.loads(state_file.read_text(encoding="utf-8", errors="replace"))
                    self.cached_tasks = raw.get("tasks", {})
                    self.cached_budget = raw.get("budget", {})
            except Exception:
                pass

    def get_filtered_events(self) -> list[TraceEvent]:
        return [ev for ev in self.events if self.filters.matches(ev)]


def render_header(state: DashboardState) -> Panel:
    session_str = state.session_id or (state.cached_session.get("session_id") or "UNKNOWN")
    status_str = state.cached_session.get("status", "IDLE")

    # Elapsed / Duration
    start_iso = state.cached_session.get("start_time_iso")
    req_dur = state.cached_session.get("requested_duration_sec", 21600.0)
    elapsed_str = "--:--:--"
    if start_iso:
        try:
            st = datetime.fromisoformat(start_iso)
            now = datetime.now(timezone.utc)
            delta_s = int((now - st).total_seconds())
            if delta_s >= 0:
                h = delta_s // 3600
                m = (delta_s % 3600) // 60
                s = delta_s % 60
                dur_h = int(req_dur) // 3600
                dur_m = (int(req_dur) % 3600) // 60
                elapsed_str = f"{h:02d}:{m:02d}:{s:02d} / {dur_h:02d}:{dur_m:02d}:00"
        except Exception:
            pass

    # Cloud tasks
    cloud_consumed = state.cached_budget.get("cloud_tasks_consumed", 0)
    cloud_max = state.cached_budget.get("max_cloud_tasks", 10)
    cloud_str = f"cloud {cloud_consumed}/{cloud_max}"

    # Token summary
    if state.has_provider_tokens:
        tok_str = f"tok {state.total_input_tokens:,} in / {state.total_output_tokens:,} out [provider]"
    elif state.cached_budget.get("total_input_tokens", 0) > 0:
        b_in = state.cached_budget.get("total_input_tokens", 0)
        b_out = state.cached_budget.get("total_output_tokens", 0)
        tok_str = f"tok {b_in:,} in / {b_out:,} out [ledger]"
    else:
        tok_str = "tok unknown"

    obs_str = "PAUSED" if state.paused else "OBSERVING"

    title_text = Text()
    title_text.append(" AUTOPHAGE ", style="bold cyan")
    title_text.append(f"─ session {session_str} ─ ", style="dim")
    status_color = "green" if status_str == "RUNNING" else ("yellow" if status_str == "COMPLETED" else "white")
    title_text.append(f"{status_str} ", style=f"bold {status_color}")
    title_text.append(f"─ {elapsed_str} ─ {cloud_str} ─ {tok_str} ─ ", style="dim")
    title_text.append(f"{obs_str} ", style="bold magenta" if state.paused else "bold green")

    return Panel(title_text, style="cyan", border_style="dim", height=3)


def render_operations(state: DashboardState) -> Panel:
    # 1. Tasks Table
    tasks_table = Table(box=None, expand=True, padding=(0, 1), show_header=True, header_style="bold underline")
    tasks_table.add_column("Task ID", style="bold", width=9)
    tasks_table.add_column("Prio", width=5)
    tasks_table.add_column("Risk", width=6)
    tasks_table.add_column("Status", width=15)
    tasks_table.add_column("Attempts", width=8)
    tasks_table.add_column("Details / Target Files", style="dim")

    status_styles = {
        "VERIFIED": "bold green",
        "IN_PROGRESS": "bold cyan",
        "VERIFYING": "bold yellow",
        "NEEDS_APPROVAL": "bold magenta",
        "BLOCKED": "bold red",
        "FAILED": "red",
        "PENDING": "dim",
        "RECOVERY_REQUIRED": "bold red",
    }

    if not state.cached_tasks:
        tasks_table.add_row("-", "-", "-", "NO TASKS", "-", "No active task ledger found")
    else:
        # Show up to 4 relevant tasks
        sorted_tasks = sorted(
            state.cached_tasks.values(),
            key=lambda t: (
                0 if t.get("status") in ("IN_PROGRESS", "VERIFYING") else (
                    1 if t.get("status") == "NEEDS_APPROVAL" else (
                        2 if t.get("status") == "BLOCKED" else 3
                    )
                )
            )
        )
        for t in sorted_tasks[:4]:
            t_id = t.get("task_id", "?")
            prio = t.get("priority", "P2")
            risk = t.get("risk", "LOW")
            st = t.get("status", "PENDING")
            att = f"a{t.get('attempts', 0)}/{t.get('max_attempts', 3)}"
            st_style = status_styles.get(st, "white")

            details = ""
            if st == "NEEDS_APPROVAL":
                details = f"[bold magenta]{t.get('blocking_reason', 'Requires Approval')}[/bold magenta]"
            elif st == "BLOCKED":
                details = f"[red]{t.get('blocking_reason', 'Blocked')}[/red]"
            else:
                tf = t.get("target_files", [])
                details = ", ".join(tf[:3]) if tf else (t.get("title", "")[:50])

            tasks_table.add_row(t_id, prio, risk, f"[{st_style}]{st}[/{st_style}]", att, details)

    # 2. Worker slots
    workers_table = Table(box=None, expand=True, padding=(0, 1), show_header=False)
    workers_table.add_column("Slot", style="bold", width=8)
    workers_table.add_column("Icon", width=3)
    workers_table.add_column("Description", style="dim")

    # Find active task
    active_task = None
    for t in state.cached_tasks.values():
        if t.get("status") in ("IN_PROGRESS", "VERIFYING"):
            active_task = t
            break

    workers_table.add_row("MACRO", "[green]●[/green]", "scheduler  cap 1/1  concurrency=single-writer  halt: -")
    if active_task:
        w1_desc = f"Task {active_task.get('task_id')}  stage={active_task.get('status')}  attempt={active_task.get('attempts', 1)}"
        workers_table.add_row("W1", "[green]●[/green]", w1_desc)
    else:
        workers_table.add_row("W1", "[dim]○[/dim]", "idle  (waiting for eligible claim)")
    workers_table.add_row("REV", "[dim]○[/dim]", "reviewer  idle (invoked for qualifying risk/policy gates)")

    content = Group(tasks_table, Text("─" * 40, style="dim"), workers_table)
    return Panel(content, title="[bold]OPERATIONS[/bold]", border_style="dim", height=9)


def render_trace(state: DashboardState, max_rows: int = 14) -> Panel:
    filtered = state.get_filtered_events()
    table = Table(box=None, expand=True, padding=(0, 1), show_header=True, header_style="bold underline")
    table.add_column("Time", width=12, style="dim")
    table.add_column("Sev", width=5)
    table.add_column("Provenance", width=11)
    table.add_column("Source → Dest", width=22)
    table.add_column("Summary", style="white")
    table.add_column("Usage / Tokens", width=18, style="dim")

    if not filtered:
        table.add_row("--:--:--.---", "INFO", "[captured]", "-", "No trace events matching filter", "-")
    else:
        total = len(filtered)
        if state.follow:
            start_idx = max(0, total - max_rows)
            slice_evs = filtered[start_idx:]
        else:
            # Scrolled view
            offset = max(0, min(state.scroll_offset, max(0, total - max_rows)))
            slice_evs = filtered[offset : offset + max_rows]

        sev_colors = {
            "DEBUG": "dim",
            "INFO": "green",
            "WARN": "yellow",
            "ERROR": "bold red",
            "CRIT": "bold magenta",
        }

        for ev in slice_evs:
            # Format time
            ts_str = ev.ts
            if "T" in ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    time_display = dt.strftime("%H:%M:%S.%f")[:12]
                except Exception:
                    time_display = ts_str[-12:]
            else:
                time_display = ts_str[:12] or "--:--:--.---"

            sev = ev.severity.upper() if ev.severity else "INFO"
            sev_col = sev_colors.get(sev, "white")

            prov = f"[{ev.provenance}]" if ev.provenance else "[captured]"

            sd = f"{ev.sender or 'sys'} → {ev.recipient or '*'}"[:22]

            summary = ev.summary or ""
            if ev.payload and "body_ref" in ev.payload:
                summary += " [i]"

            # Usage string
            usage = ev.usage if isinstance(ev.usage, dict) else {}
            u_src = usage.get("source", "unknown")
            if u_src == "provider":
                inp = usage.get("input_tokens")
                out = usage.get("output_tokens")
                usage_str = f"in={inp or '?'} out={out or '?'}"
            elif u_src == "unknown":
                usage_str = "not reported"
            else:
                usage_str = u_src

            if ev.latency_ms is not None:
                usage_str += f" ({ev.latency_ms / 1000.0:.1f}s)"

            table.add_row(
                time_display,
                f"[{sev_col}]{sev}[/{sev_col}]",
                f"[dim]{prov}[/dim]",
                sd,
                summary,
                usage_str,
            )

    f_str = f"task={state.filters.task_id or '*'} worker={state.filters.worker_id or '*'} cat={state.filters.category or '*'} sev>={state.filters.min_severity}"
    if state.filters.search_query:
        f_str += f" search='{state.filters.search_query}'"

    follow_mode = "[bold green]follow ▶[/bold green]" if state.follow else "[yellow]paused (scroll)[/yellow]"
    title = f"TRACE  {follow_mode}  filter: {f_str} ({len(filtered)}/{len(state.events)} events)"
    return Panel(table, title=title, border_style="dim")


def render_footer(state: DashboardState) -> Panel:
    lag_str = f"lag: {state.lag_sec:.1f}s"
    view_str = "PAUSED" if state.paused else ("LIVE" if state.follow else "SCROLL")
    footer_text = Text()
    footer_text.append("[space]", style="bold cyan")
    footer_text.append("pause view  ", style="dim")
    footer_text.append("[f]", style="bold cyan")
    footer_text.append("ollow  ", style="dim")
    footer_text.append("[/]", style="bold cyan")
    footer_text.append("search  ", style="dim")
    footer_text.append("[t]", style="bold cyan")
    footer_text.append("ask  ", style="dim")
    footer_text.append("[w]", style="bold cyan")
    footer_text.append("orker  ", style="dim")
    footer_text.append("[c]", style="bold cyan")
    footer_text.append("at  ", style="dim")
    footer_text.append("[s]", style="bold cyan")
    footer_text.append("ev  ", style="dim")
    footer_text.append("[i]", style="bold cyan")
    footer_text.append("nspect  ", style="dim")
    footer_text.append("[0]", style="bold cyan")
    footer_text.append("clear  ", style="dim")
    footer_text.append("[q]", style="bold cyan")
    footer_text.append("quit  │  ", style="dim")
    footer_text.append(f"view: {view_str}  {lag_str}", style="bold")

    return Panel(footer_text, style="white", border_style="dim", height=3)


def render_inspect_popup(state: DashboardState) -> Panel:
    content = Text(state.inspection_content or "No body or payload content available.", style="white")
    return Panel(
        content,
        title=f"[bold cyan]INSPECT: {state.inspection_title}[/bold cyan] ([dim]Press Esc/i/q to return[/dim])",
        border_style="bold cyan",
    )


def render_dashboard(state: DashboardState, console: Console) -> Layout:
    layout = Layout()
    if state.inspecting:
        layout.split_column(
            Layout(render_header(state), size=3),
            Layout(render_inspect_popup(state)),
            Layout(render_footer(state), size=3),
        )
        return layout

    layout.split_column(
        Layout(render_header(state), size=3),
        Layout(render_operations(state), size=10),
        Layout(render_trace(state), ratio=1),
        Layout(render_footer(state), size=3),
    )
    return layout


def poll_keyboard(state: DashboardState) -> bool:
    """
    Non-blocking keyboard poller for Windows (msvcrt).
    Returns False when exit is requested ('q' or Ctrl+C); True to continue.
    """
    if not HAS_MSVCRT:
        return True

    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch == "\x03":  # Ctrl+C
            return False

        if state.inspecting:
            if ch in ("\x1b", "i", "q", "\r", "\n"):
                state.inspecting = False
            continue

        if ch == "q":
            return False
        elif ch == " ":
            state.paused = not state.paused
        elif ch == "f":
            state.follow = not state.follow
            if state.follow:
                state.scroll_offset = 0
        elif ch == "0":
            state.filters = FilterConfig()
        elif ch == "s":
            # Cycle severity
            cur = state.filters.min_severity
            idx = SEVERITY_LEVELS.index(cur) if cur in SEVERITY_LEVELS else 1
            state.filters.min_severity = SEVERITY_LEVELS[(idx + 1) % len(SEVERITY_LEVELS)]
        elif ch == "c":
            # Cycle category
            cats = [None, "lifecycle", "worker", "request", "planner", "quality", "review"]
            cur_idx = cats.index(state.filters.category) if state.filters.category in cats else 0
            state.filters.category = cats[(cur_idx + 1) % len(cats)]
        elif ch == "t":
            # Cycle task
            known_tasks = list(state.cached_tasks.keys())
            if known_tasks:
                tasks_cycle = [None] + known_tasks
                cur_idx = tasks_cycle.index(state.filters.task_id) if state.filters.task_id in tasks_cycle else 0
                state.filters.task_id = tasks_cycle[(cur_idx + 1) % len(tasks_cycle)]
        elif ch in ("i", "\r"):
            # Inspect last / selected event
            filtered = state.get_filtered_events()
            if filtered:
                ev = filtered[-1]
                body_ref = ev.payload.get("body_ref") if ev.payload else None
                body_text = read_trace_body(body_ref) if body_ref else None
                details = {
                    "seq": ev.seq,
                    "ts": ev.ts,
                    "type": ev.type,
                    "severity": ev.severity,
                    "provenance": ev.provenance,
                    "task_id": ev.task_id,
                    "worker_id": ev.worker_id,
                    "request_id": ev.request_id,
                    "latency_ms": ev.latency_ms,
                    "usage": ev.usage,
                    "summary": ev.summary,
                    "payload": ev.payload,
                }
                disp = json.dumps(details, indent=2)
                if body_text:
                    disp += f"\n\n=== ATTACHED BODY ({body_ref}) ===\n{body_text}"
                state.inspection_content = disp
                state.inspection_title = f"{ev.type} (req={ev.request_id or 'none'})"
                state.inspecting = True
        elif ch == "\x1b":
            # Escape sequence / arrows
            if msvcrt.kbhit():
                ch2 = msvcrt.getwch()
                if ch2 == "[":
                    if msvcrt.kbhit():
                        ch3 = msvcrt.getwch()
                        filtered = state.get_filtered_events()
                        total = len(filtered)
                        if ch3 == "A":  # Up
                            state.follow = False
                            state.scroll_offset = max(0, state.scroll_offset - 1)
                        elif ch3 == "B":  # Down
                            state.follow = False
                            state.scroll_offset = min(max(0, total - 10), state.scroll_offset + 1)
        elif ch == "\xe0":  # Windows special key prefix (arrows/PgUp/PgDn)
            if msvcrt.kbhit():
                ch2 = msvcrt.getwch()
                filtered = state.get_filtered_events()
                total = len(filtered)
                if ch2 == "H":  # Up
                    state.follow = False
                    state.scroll_offset = max(0, state.scroll_offset - 1)
                elif ch2 == "P":  # Down
                    state.follow = False
                    state.scroll_offset = min(max(0, total - 10), state.scroll_offset + 1)
                elif ch2 == "I":  # PgUp
                    state.follow = False
                    state.scroll_offset = max(0, state.scroll_offset - 8)
                elif ch2 == "Q":  # PgDn
                    state.follow = False
                    state.scroll_offset = min(max(0, total - 10), state.scroll_offset + 8)

    return True


def run_plain(
    session_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
    max_events: int = 100,
    follow: bool = False,
) -> int:
    """Non-interactive plain-text fallback for non-TTY, CI, or pipes."""
    root = repo_root or Path.cwd()
    state = DashboardState(root, session_id=session_id)
    state.refresh()

    sys.stdout.write("=== MAYA AUTOPHAGE TRACE REPORT (PLAIN) ===\n")
    sys.stdout.write(f"Session: {state.session_id or 'UNKNOWN'} | Active Tasks: {len(state.cached_tasks)}\n\n")

    events = state.events
    if not events:
        sys.stdout.write("No trace events recorded.\n")
        return 0

    to_show = events[-max_events:] if max_events > 0 else events
    for ev in to_show:
        u_src = ev.usage.get("source", "unknown") if isinstance(ev.usage, dict) else "unknown"
        tok_str = ""
        if u_src == "provider":
            tok_str = f" [tokens: in={ev.usage.get('input_tokens')} out={ev.usage.get('output_tokens')}]"
        elif u_src == "unknown":
            tok_str = " [tokens: not reported]"

        lat_str = f" ({ev.latency_ms / 1000.0:.2f}s)" if ev.latency_ms is not None else ""
        sys.stdout.write(
            f"{ev.ts} [{ev.severity:5s}] [{ev.provenance:9s}] {ev.sender} -> {ev.recipient or '*'}: "
            f"{ev.summary}{tok_str}{lat_str}\n"
        )

    sys.stdout.flush()
    return 0


def run_tui(
    session_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
    plain: bool = False,
    replay: bool = False,
    max_fps: int = 10,
) -> int:
    """
    Launch Autophage terminal dashboard.
    Falls back to plain text if plain=True or stdout is not a TTY.
    """
    root = (repo_root or Path.cwd()).resolve()

    if plain or not sys.stdout.isatty():
        return run_plain(session_id=session_id, repo_root=root, follow=not replay)

    console = Console()
    state = DashboardState(root, session_id=session_id)
    if replay:
        state.follow = False

    sleep_sec = 1.0 / max(1, max_fps)

    try:
        with Live(render_dashboard(state, console), console=console, refresh_per_second=max_fps, screen=True) as live:
            while True:
                if not replay and not state.paused:
                    state.refresh()
                elif replay and not state.events:
                    state.refresh()

                live.update(render_dashboard(state, console))

                if not poll_keyboard(state):
                    break

                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        pass

    console.print("[dim]Autophage TUI exited. Background coordinator and worktrees preserved.[/dim]")
    return 0


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    run_tui(session_id=sid)
