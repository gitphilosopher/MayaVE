"""
skills/maintenance/tui.py
Trustworthy, Interactive Terminal Dashboard for MayaVE Autophage (Phases 1 & 6).

Design invariants:
- Read-only by default (observer mode): never modifies state.json, never acquires
  StateStore locks, and never executes mutating coordinator actions.
- Interactive operator mode enabled strictly via explicit --operate flag.
- Safe operator controls: exact binding revalidation (patch SHA-256, baseline commit,
  verification fingerprint), confirmation modals before consequential actions,
  prevention of duplicate approvals or downgrades, and strict enforcement that
  RECOVERY_REQUIRED tasks remain blocked.
- Exposes pending merges, worktree cleanliness, and non-destructive cleanup previews.
- Windows non-blocking keyboard controls via msvcrt (space=pause, f=follow,
  i/Enter=inspect, Tab/1-4=view tabs, a=approve, d=deny, q=quit).
- Plain-text non-TTY fallback for CI, headless runs, or pipes.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    from .trace import TraceEvent, read_trace_body
    from .operator_controls import OperatorController
except ImportError:
    from trace import TraceEvent, read_trace_body
    from operator_controls import OperatorController


SEVERITY_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "CRIT"]
SEVERITY_ORDER = {sev: idx for idx, sev in enumerate(SEVERITY_LEVELS)}
TAB_NAMES = ["OPERATIONS", "APPROVALS", "MERGES", "RECOVERY"]

# Theme: lavender + cool gray + white as the base, with three soft semantic
# accents (tuned to sit beside the lavender) used only to signal state.
PURPLE = "rgb(180,156,255)"  # brand / active
GRAY = "rgb(160,157,165)"    # secondary text, rules, labels
WHITE = "white"              # primary text
MINT = "rgb(128,222,184)"    # ok: running, verified, clean, merged
AMBER = "rgb(244,201,125)"   # caution: needs approval, warnings, operate mode
CORAL = "rgb(255,128,146)"   # danger: blocked, failed, recovery, errors
ACCENT = f"bold {PURPLE}"


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

        if self.trace_path and target != self.trace_path:
            self.trace_path = target
            self.file_pos = 0
            self.line_buffer = ""

        self.trace_path = target
        events: list[TraceEvent] = []

        try:
            stat = target.stat()
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
                    pass
        except Exception:
            pass

        return events


class DashboardState:
    """In-memory view state for Autophage terminal dashboard."""

    def __init__(
        self,
        repo_root: Path,
        session_id: Optional[str] = None,
        operate: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.maintenance_dir = self.repo_root / ".maya" / "maintenance"
        self.trace_dir = self.maintenance_dir / "trace"
        self.session_id = session_id or ""
        self.operate_mode: bool = operate
        self.active_tab: str = "OPERATIONS"

        # Trace events
        self.events: list[TraceEvent] = []
        self.tail_reader = TraceTailReader(
            trace_path=(self.trace_dir / f"{session_id}.jsonl") if session_id else None,
            trace_dir=self.trace_dir,
        )

        # View settings
        self.follow: bool = True
        self.paused: bool = False
        self.scroll_offset: int = 0
        self.selected_index: int = -1

        # Modals / Popups
        self.inspecting: bool = False
        self.modal_mode: Optional[str] = None  # None, "INSPECT", "CONFIRM_APPROVE", "CONFIRM_DENY", "MESSAGE"
        self.inspection_content: str = ""
        self.inspection_title: str = ""
        self.modal_req: Any = None
        self.status_banner: str = ""
        self.status_banner_time: float = 0.0

        # Operator controls & Approvals
        self.controller = OperatorController(repo_root=self.repo_root)
        self.pending_requests: list[Any] = []
        self.selected_request_idx: int = 0
        self.cached_recovery: list[dict[str, Any]] = []
        self.cached_merges: dict[str, Any] = {}
        self.last_aux_poll: float = 0.0

        # Filters
        self.filters = FilterConfig()

        # Cached state.json
        self.state_mtime: float = 0.0
        self.cached_tasks: dict[str, Any] = {}
        self.cached_budget: dict[str, Any] = {}

        # Cached active_session.json
        self.session_mtime: float = 0.0
        self.cached_session: dict[str, Any] = {}

        # Cumulative tokens from events (provider vs ledger vs unknown)
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

        # Poll approvals and recovery diagnostics
        try:
            self.pending_requests = self.controller.get_pending_approvals()
            if self.pending_requests:
                self.selected_request_idx = max(0, min(self.selected_request_idx, len(self.pending_requests) - 1))
            else:
                self.selected_request_idx = 0
        except Exception:
            self.pending_requests = []

        # Poll auxiliary recovery diagnostics and merges every 1.5 seconds
        if (now - self.last_aux_poll) > 1.5:
            self.last_aux_poll = now
            try:
                self.cached_recovery = [d.to_dict() for d in self.controller.get_recovery_diagnostics()]
            except Exception:
                self.cached_recovery = []

            if self.active_tab == "MERGES":
                try:
                    self.cached_merges = self.controller.get_pending_merges_data()
                except Exception:
                    self.cached_merges = {}

    def get_filtered_events(self) -> list[TraceEvent]:
        return [ev for ev in self.events if self.filters.matches(ev)]


def _table(**kw: Any) -> Table:
    return Table(box=None, expand=True, padding=(0, 1), show_header=True,
                 header_style=GRAY, pad_edge=False, **kw)


def _risk_style(risk: Any) -> str:
    r = str(risk or "").upper()
    if r in ("HIGH", "CRITICAL", "CRIT"):
        return f"bold {CORAL}"
    if r in ("MEDIUM", "MED"):
        return AMBER
    return WHITE


def _rule() -> Rule:
    return Rule(characters="─", style=GRAY)


def _section(title: str, body: Any, meta: str = "") -> Group:
    head = Text()
    head.append(title, style=ACCENT)
    if meta:
        head.append(f"  {meta}", style=GRAY)
    return Group(head, body)


def render_header(state: DashboardState) -> Group:
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

    cloud_consumed = state.cached_budget.get("cloud_tasks_consumed", 0)
    cloud_max = state.cached_budget.get("max_cloud_tasks", 10)
    cloud_str = f"cloud {cloud_consumed}/{cloud_max}"

    # Token summary with provenance
    if state.has_provider_tokens:
        tok_str = f"tok {state.total_input_tokens:,} in / {state.total_output_tokens:,} out [provider]"
    elif state.cached_budget.get("total_input_tokens", 0) > 0:
        b_in = state.cached_budget.get("total_input_tokens", 0)
        b_out = state.cached_budget.get("total_output_tokens", 0)
        tok_str = f"tok {b_in:,} in / {b_out:,} out [ledger]"
    else:
        tok_str = "tok unknown"

    line1 = Text(no_wrap=True, overflow="ellipsis")
    line1.append("autophage", style=ACCENT)
    line1.append(f"  {session_str}", style=GRAY)
    line1.append(f"  {status_str.lower()}", style=f"bold {MINT}" if status_str == "RUNNING" else WHITE)
    line1.append(f"  {elapsed_str}  {cloud_str}  {tok_str}", style=GRAY)
    if state.operate_mode:
        line1.append("  operate", style=f"bold {AMBER}")
    elif state.paused:
        line1.append("  paused", style=AMBER)
    else:
        line1.append("  observe", style=GRAY)

    # Inactive tabs light up when they hold something that needs attention.
    tabs = [
        ("1 ops", state.active_tab == "OPERATIONS", GRAY),
        (f"2 approvals ({len(state.pending_requests)})", state.active_tab == "APPROVALS",
         AMBER if state.pending_requests else GRAY),
        ("3 merges", state.active_tab == "MERGES", GRAY),
        (f"4 recovery ({len(state.cached_recovery)})", state.active_tab == "RECOVERY",
         CORAL if state.cached_recovery else GRAY),
    ]
    line2 = Text(no_wrap=True, overflow="ellipsis")
    for name, is_active, idle_style in tabs:
        line2.append(name + "   ", style=ACCENT if is_active else idle_style)

    return Group(line1, line2, _rule())


def render_operations(state: DashboardState) -> Group:
    tasks_table = _table()
    tasks_table.add_column("task", style=GRAY, width=9)
    tasks_table.add_column("prio", style=WHITE, width=5)
    tasks_table.add_column("risk", style=WHITE, width=6)
    tasks_table.add_column("status", width=18)
    tasks_table.add_column("tries", style=WHITE, width=6)
    tasks_table.add_column("details", style=GRAY)

    status_styles = {
        "VERIFIED": MINT,
        "IN_PROGRESS": PURPLE,
        "VERIFYING": PURPLE,
        "READY_TO_INTEGRATE": PURPLE,
        "NEEDS_APPROVAL": f"bold {AMBER}",
        "BLOCKED": f"bold {CORAL}",
        "FAILED": f"bold {CORAL}",
        "PENDING": GRAY,
        "RECOVERY_REQUIRED": f"bold {CORAL}",
    }

    if not state.cached_tasks:
        tasks_table.add_row("-", "-", "-", Text("NO TASKS", style=GRAY), "-", "No active task ledger found")
    else:
        sorted_tasks = sorted(
            state.cached_tasks.values(),
            key=lambda t: (
                0 if t.get("status") in ("IN_PROGRESS", "VERIFYING", "READY_TO_INTEGRATE") else (
                    1 if t.get("status") == "RECOVERY_REQUIRED" else (
                        2 if t.get("status") == "NEEDS_APPROVAL" else (
                            3 if t.get("status") == "BLOCKED" else 4
                        )
                    )
                )
            )
        )
        for t in sorted_tasks[:4]:
            t_id = t.get("task_id", "?")
            prio = t.get("priority", "P2")
            risk = t.get("risk", "LOW")
            st = t.get("status", "PENDING")
            att = f"{t.get('attempts', 0)}/{t.get('max_attempts', 3)}"

            if st == "NEEDS_APPROVAL":
                details = Text(str(t.get("blocking_reason", "Requires Operator Approval")), style=AMBER)
            elif st == "RECOVERY_REQUIRED":
                details = Text(str(t.get("blocking_reason", "Recovery required - in-flight interruption")), style=CORAL)
            elif st == "BLOCKED":
                details = Text(str(t.get("blocking_reason", "Blocked")), style=CORAL)
            else:
                tf = t.get("target_files", [])
                details = Text(", ".join(tf[:3]) if tf else (t.get("title", "")[:50]), style=GRAY)

            tasks_table.add_row(t_id, prio, Text(str(risk), style=_risk_style(risk)), Text(st, style=status_styles.get(st, WHITE)), att, details)

    workers_table = Table(box=None, expand=True, padding=(0, 1), show_header=False, pad_edge=False)
    workers_table.add_column("slot", style=WHITE, width=6)
    workers_table.add_column("icon", width=2)
    workers_table.add_column("desc", style=GRAY)

    active_task = None
    for t in state.cached_tasks.values():
        if t.get("status") in ("IN_PROGRESS", "VERIFYING", "INTEGRATING"):
            active_task = t
            break

    on, off = Text("●", style=MINT), Text("○", style=GRAY)
    workers_table.add_row("macro", on, "scheduler  cap 1/1  single-writer  halt: -")
    if active_task:
        w1_desc = f"task {active_task.get('task_id')}  stage={active_task.get('status')}  attempt={active_task.get('attempts', 1)}"
        workers_table.add_row("w1", on, w1_desc)
    else:
        workers_table.add_row("w1", off, "idle  (waiting for eligible claim)")
    workers_table.add_row("rev", off, "reviewer idle (invoked for qualifying risk/safeguard gates)")

    return _section("operations", Group(tasks_table, _rule(), workers_table))


def render_approvals(state: DashboardState) -> Group:
    if not state.pending_requests:
        body = Group(
            Text("\nNo pending permission requests.", style=WHITE),
            Text("Tasks requiring authorization register binding requests here automatically.", style=GRAY),
        )
        return _section("approvals", body, "0 pending")

    table = _table()
    table.add_column("", width=1, style=PURPLE)
    table.add_column("request", width=22)
    table.add_column("task", width=8)
    table.add_column("risk", width=6)
    table.add_column("operation", width=18)
    table.add_column("patch", width=10)
    table.add_column("scope / reason")

    for idx, req in enumerate(state.pending_requests[:4]):
        is_sel = (idx == state.selected_request_idx)
        patch_disp = (req.patch_sha256[:8] + "..") if req.patch_sha256 else "N/A"
        scope_disp = f"[{req.scope}] {req.reason}"[:50]
        table.add_row(
            ">" if is_sel else "",
            req.request_id, req.task_id, Text(str(req.risk), style=_risk_style(req.risk)),
            req.operation, patch_disp, scope_disp,
            style=WHITE if is_sel else GRAY,
        )

    sel_req = state.pending_requests[state.selected_request_idx]
    patch_s = sel_req.patch_sha256 or "N/A"
    base_s = sel_req.baseline_commit or "N/A"
    ver_s = sel_req.verification_sha256 or "N/A"
    detail = Text()
    detail.append(f"{sel_req.request_id}", style=ACCENT)
    detail.append(f"  task {sel_req.task_id}  op {sel_req.operation}  risk {sel_req.risk}\n", style=WHITE)
    detail.append(f"patch={patch_s[:10]}..  base={base_s[:10]}..  ver={ver_s[:10]}..\n", style=GRAY)
    detail.append(f"scope: {sel_req.scope}  reason: {sel_req.reason}", style=GRAY)

    return _section("approvals", Group(table, _rule(), detail), f"{len(state.pending_requests)} pending")


def render_merges(state: DashboardState) -> Group:
    merges_info = state.cached_merges.get("merges", [])
    cleanup_info = state.cached_merges.get("cleanup", [])

    table = _table()
    table.add_column("branch", width=26, style=WHITE)
    table.add_column("task", width=8, style=GRAY)
    table.add_column("base..head", width=16, style=GRAY)
    table.add_column("ahead", width=6, style=WHITE)
    table.add_column("merge", width=13)
    table.add_column("worktree", width=9)
    table.add_column("blockers", style=GRAY)

    if not merges_info:
        table.add_row("-", "-", "-", "-", Text("NONE", style=GRAY), "-", "No active integration branches found")
    else:
        for m in merges_info[:3]:
            st = m.get("merge_status", "PENDING")
            wt = m.get("worktree", "-")
            blockers = ", ".join(m.get("blockers", [])) or "none"
            table.add_row(
                m.get("branch", "?"),
                m.get("task_id", "-"),
                f"{m.get('base', '-')}..{m.get('head', '-')}",
                str(m.get("ahead", 0)),
                Text(st, style=MINT if st == "MERGED" else AMBER),
                Text(str(wt), style=CORAL if wt == "DIRTY" else (MINT if wt == "CLEAN" else GRAY)),
                blockers,
            )

    safe_cleanup_count = sum(1 for c in cleanup_info if c.get("eligible"))
    cleanup_text = Text()
    cleanup_text.append(f"cleanup preview: {len(cleanup_info)} inspected, {safe_cleanup_count} safe. ", style=WHITE)
    cleanup_text.append("read-only; deletion requires CLI --cleanup <task> --authorize", style=GRAY)

    return _section("merges", Group(table, _rule(), cleanup_text))


def render_recovery(state: DashboardState) -> Group:
    recovery_tasks = state.cached_recovery

    table = _table()
    table.add_column("task", width=9, style=GRAY)
    table.add_column("stage", width=22, style=WHITE)
    table.add_column("worktree", width=9, style=WHITE)
    table.add_column("blocker", width=34, style=CORAL)
    table.add_column("next safe action", style=WHITE)

    if not recovery_tasks:
        table.add_row("-", "-", "-", "No tasks in RECOVERY_REQUIRED", "All tasks healthy")
    else:
        for r in recovery_tasks[:3]:
            wt_desc = "DIRTY" if r.get("is_dirty") else ("EXISTS" if r.get("worktree_exists") else "MISSING")
            table.add_row(
                r.get("task_id", "?"),
                r.get("stage", "UNKNOWN"),
                Text(wt_desc, style=CORAL if wt_desc == "DIRTY" else (AMBER if wt_desc == "MISSING" else WHITE)),
                "; ".join(r.get("blockers", []))[:33],
                r.get("next_safe_action", "")[:45],
            )

    invariant = Text(
        "RECOVERY_REQUIRED tasks stay blocked; automated bypass is forbidden. "
        "Review diagnostics, then run 'coordinator --reconcile'.",
        style=GRAY,
    )
    return _section("recovery", Group(table, _rule(), invariant), f"{len(recovery_tasks)} blocked")


def render_trace(state: DashboardState, max_rows: int = 14) -> Group:
    filtered = state.get_filtered_events()
    table = _table()
    table.add_column("time", width=12, style=GRAY)
    table.add_column("sev", width=5)
    table.add_column("prov", width=10, style=GRAY)
    table.add_column("route", width=22, style=GRAY)
    table.add_column("summary", style=WHITE)
    table.add_column("usage", width=18, style=GRAY)

    if not filtered:
        table.add_row("--:--:--.---", Text("INFO", style=GRAY), "captured", "-", "No trace events matching filter", "-")
    else:
        total = len(filtered)
        if state.follow:
            slice_evs = filtered[max(0, total - max_rows):]
        else:
            offset = max(0, min(state.scroll_offset, max(0, total - max_rows)))
            slice_evs = filtered[offset : offset + max_rows]

        sev_styles = {
            "DEBUG": GRAY,
            "INFO": GRAY,
            "WARN": AMBER,
            "ERROR": f"bold {CORAL}",
            "CRIT": f"bold {WHITE} on {CORAL}",
        }

        for ev in slice_evs:
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
            prov = ev.provenance or "captured"
            sd = f"{ev.sender or 'sys'} → {ev.recipient or '*'}"[:22]

            summary = ev.summary or ""
            if ev.payload and "body_ref" in ev.payload:
                summary += " ›"

            usage = ev.usage if isinstance(ev.usage, dict) else {}
            u_src = usage.get("source", "unknown")
            if u_src == "provider":
                usage_str = f"in={usage.get('input_tokens') or '?'} out={usage.get('output_tokens') or '?'}"
            elif u_src == "unknown":
                usage_str = "not reported"
            else:
                usage_str = u_src
            if ev.latency_ms is not None:
                usage_str += f" ({ev.latency_ms / 1000.0:.1f}s)"

            table.add_row(time_display, Text(sev, style=sev_styles.get(sev, WHITE)), prov, sd, summary, usage_str)

    f_str = f"task={state.filters.task_id or '*'} worker={state.filters.worker_id or '*'} cat={state.filters.category or '*'} sev>={state.filters.min_severity}"
    if state.filters.search_query:
        f_str += f" search='{state.filters.search_query}'"

    head = Text()
    head.append("trace", style=ACCENT)
    head.append("  follow" if state.follow else "  scroll", style=MINT if state.follow else AMBER)
    head.append(f"  {f_str}  {len(filtered)}/{len(state.events)}", style=GRAY)
    return Group(head, table)


def render_footer(state: DashboardState) -> Group:
    lag_str = f"lag {state.lag_sec:.1f}s"
    view_str = "paused" if state.paused else ("live" if state.follow else "scroll")
    t = Text(no_wrap=True, overflow="ellipsis")

    def key(k: str, label: str) -> None:
        t.append(k, style=WHITE)
        t.append(f" {label}   ", style=GRAY)

    if state.modal_mode == "CONFIRM_APPROVE":
        key("y/enter", "confirm approve")
        key("n/esc/q", "cancel")
    elif state.modal_mode == "CONFIRM_DENY":
        key("y/enter", "confirm reject")
        key("n/esc/q", "cancel")
    elif state.modal_mode in ("INSPECT", "MESSAGE"):
        key("esc/enter/q", "close")
    else:
        if state.operate_mode:
            key("a", "approve")
            key("d", "deny")
        else:
            key("space", "pause")
        key("i", "inspect")
        key("tab/1-4", "view")
        key("f", "follow")
        if not state.operate_mode:
            key("0", "clear")
        key("q", "quit")
        t.append("operate" if state.operate_mode else "observe", style=f"bold {AMBER}" if state.operate_mode else GRAY)
        t.append(f"  {view_str}  {lag_str}", style=GRAY)

    if state.status_banner:
        t.append(f"  {state.status_banner}", style=f"bold {AMBER}")

    return Group(_rule(), t)


def _popup(title: str, body: Text, border: str = PURPLE) -> Panel:
    return Panel(body, title=Text(title, style=f"bold {border}"), title_align="left",
                 box=box.ROUNDED, border_style=border, padding=(1, 2))


def render_confirm_approve_popup(state: DashboardState) -> Panel:
    req = state.modal_req
    text = Text()
    text.append("Authorize a consequential maintenance operation.\n\n", style=WHITE)
    if req:
        for label, value, style in (
            ("operation", req.operation, WHITE),
            ("task", req.task_id, WHITE),
            ("request", req.request_id, GRAY),
            ("risk", req.risk, WHITE),
            ("patch sha", req.patch_sha256 or "N/A", PURPLE),
            ("baseline", req.baseline_commit or "N/A", GRAY),
            ("verify sha", req.verification_sha256 or "N/A", GRAY),
            ("scope", req.scope, GRAY),
            ("reason", req.reason, GRAY),
        ):
            text.append(f"{label:<11}", style=GRAY)
            text.append(f"{value}\n", style=style)
        text.append("\n")
    text.append("Bindings are revalidated on confirmation.\n\n", style=GRAY)
    text.append("y / enter", style=f"bold {MINT}")
    text.append(" confirm   ", style=GRAY)
    text.append("n / esc / q", style=WHITE)
    text.append(" cancel", style=GRAY)
    return _popup("confirm approval", text, AMBER)


def render_confirm_deny_popup(state: DashboardState) -> Panel:
    req = state.modal_req
    text = Text()
    text.append("Reject this permission request.\n\n", style=WHITE)
    if req:
        for label, value in (("operation", req.operation), ("task", req.task_id), ("request", req.request_id)):
            text.append(f"{label:<11}", style=GRAY)
            text.append(f"{value}\n", style=WHITE)
        text.append("\n")
    text.append("An auditable rejection is kept in approvals.json and the event ledger.\n\n", style=GRAY)
    text.append("y / enter", style=f"bold {CORAL}")
    text.append(" confirm   ", style=GRAY)
    text.append("n / esc / q", style=WHITE)
    text.append(" cancel", style=GRAY)
    return _popup("confirm rejection", text, CORAL)


def render_inspect_popup(state: DashboardState) -> Panel:
    content = Text(state.inspection_content or "No body or payload content available.", style=WHITE)
    return _popup(f"inspect  {state.inspection_title}", content)


def render_message_popup(state: DashboardState) -> Panel:
    content = Text(state.inspection_content, style=WHITE)
    return _popup(state.inspection_title or "message", content)


def render_dashboard(state: DashboardState, console: Console) -> Layout:
    layout = Layout()

    popup: Optional[Panel] = None
    if state.modal_mode == "CONFIRM_APPROVE":
        popup = render_confirm_approve_popup(state)
    elif state.modal_mode == "CONFIRM_DENY":
        popup = render_confirm_deny_popup(state)
    elif state.modal_mode == "MESSAGE":
        popup = render_message_popup(state)
    elif state.inspecting or state.modal_mode == "INSPECT":
        popup = render_inspect_popup(state)

    if popup is not None:
        layout.split_column(
            Layout(render_header(state), size=3),
            Layout(popup),
            Layout(render_footer(state), size=2),
        )
        return layout

    if state.active_tab == "APPROVALS":
        main_panel = render_approvals(state)
    elif state.active_tab == "MERGES":
        main_panel = render_merges(state)
    elif state.active_tab == "RECOVERY":
        main_panel = render_recovery(state)
    else:
        main_panel = render_operations(state)

    layout.split_column(
        Layout(render_header(state), size=3),
        Layout(main_panel, size=10),
        Layout(_rule(), size=1),
        Layout(render_trace(state), ratio=1),
        Layout(render_footer(state), size=2),
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

        # Clear transient status banner on any keystroke
        if state.status_banner:
            state.status_banner = ""

        # Modal navigation
        if state.modal_mode == "CONFIRM_APPROVE":
            if ch in ("y", "Y", "\r", "\n"):
                if state.modal_req:
                    ok, msg = state.controller.execute_approval(state.modal_req.request_id)
                    state.modal_mode = "MESSAGE"
                    state.modal_title = "Approval Succeeded" if ok else "Approval Rejected"
                    state.inspection_content = msg
                    state.refresh()
                else:
                    state.modal_mode = None
            elif ch in ("n", "N", "\x1b", "q"):
                state.modal_mode = None
                state.status_banner = "Approval cancelled."
            continue

        if state.modal_mode == "CONFIRM_DENY":
            if ch in ("y", "Y", "\r", "\n"):
                if state.modal_req:
                    ok, msg = state.controller.execute_rejection(
                        state.modal_req.request_id,
                        reason="Operator rejected via dashboard.",
                    )
                    state.modal_mode = "MESSAGE"
                    state.modal_title = "Request Rejected" if ok else "Rejection Failed"
                    state.inspection_content = msg
                    state.refresh()
                else:
                    state.modal_mode = None
            elif ch in ("n", "N", "\x1b", "q"):
                state.modal_mode = None
                state.status_banner = "Rejection cancelled."
            continue

        if state.modal_mode in ("INSPECT", "MESSAGE") or state.inspecting:
            if ch in ("\x1b", "i", "q", "\r", "\n"):
                state.inspecting = False
                state.modal_mode = None
            continue

        # Standard navigation & tab switching
        if ch == "q":
            return False
        elif ch == "\t":
            # Cycle tabs
            cur_idx = TAB_NAMES.index(state.active_tab) if state.active_tab in TAB_NAMES else 0
            state.active_tab = TAB_NAMES[(cur_idx + 1) % len(TAB_NAMES)]
        elif ch == "1":
            state.active_tab = "OPERATIONS"
        elif ch == "2":
            state.active_tab = "APPROVALS"
        elif ch == "3":
            state.active_tab = "MERGES"
            state.cached_merges = state.controller.get_pending_merges_data()
        elif ch == "4":
            state.active_tab = "RECOVERY"
        elif ch == "m":
            state.active_tab = "MERGES"
            state.cached_merges = state.controller.get_pending_merges_data()
        elif ch == "r":
            state.active_tab = "RECOVERY"
        elif ch == " ":
            state.paused = not state.paused
        elif ch == "f":
            state.follow = not state.follow
            if state.follow:
                state.scroll_offset = 0
        elif ch == "0":
            state.filters = FilterConfig()
        elif ch == "s":
            cur = state.filters.min_severity
            idx = SEVERITY_LEVELS.index(cur) if cur in SEVERITY_LEVELS else 1
            state.filters.min_severity = SEVERITY_LEVELS[(idx + 1) % len(SEVERITY_LEVELS)]
        elif ch == "c":
            cats = [None, "lifecycle", "worker", "request", "planner", "quality", "review", "approval"]
            cur_idx = cats.index(state.filters.category) if state.filters.category in cats else 0
            state.filters.category = cats[(cur_idx + 1) % len(cats)]
        elif ch == "t":
            known_tasks = list(state.cached_tasks.keys())
            if known_tasks:
                tasks_cycle = [None] + known_tasks
                cur_idx = tasks_cycle.index(state.filters.task_id) if state.filters.task_id in tasks_cycle else 0
                state.filters.task_id = tasks_cycle[(cur_idx + 1) % len(tasks_cycle)]

        # Approval actions (operate mode only)
        elif ch == "a":
            if not state.operate_mode:
                state.status_banner = "Read-only mode. Launch with --operate to enable actions."
            elif not state.pending_requests:
                state.status_banner = "No pending approval requests."
            else:
                req = state.pending_requests[state.selected_request_idx]
                state.modal_req = req
                state.modal_mode = "CONFIRM_APPROVE"

        elif ch == "d":
            if not state.operate_mode:
                state.status_banner = "Read-only mode. Launch with --operate to enable actions."
            elif not state.pending_requests:
                state.status_banner = "No pending approval requests."
            else:
                req = state.pending_requests[state.selected_request_idx]
                state.modal_req = req
                state.modal_mode = "CONFIRM_DENY"

        elif ch in ("j", "k") and state.active_tab == "APPROVALS":
            if ch == "k":
                state.selected_request_idx = max(0, state.selected_request_idx - 1)
            elif ch == "j":
                state.selected_request_idx = min(len(state.pending_requests) - 1, state.selected_request_idx + 1)

        elif ch in ("i", "\r"):
            # Inspect tab item or trace event
            if state.active_tab == "APPROVALS" and state.pending_requests:
                req = state.pending_requests[state.selected_request_idx]
                info = state.controller.inspect_request(req.request_id)
                state.inspection_content = json.dumps(info, indent=2)
                state.inspection_title = f"Approval Request {req.request_id}"
                state.inspecting = True
            elif state.active_tab == "RECOVERY" and state.cached_recovery:
                state.inspection_content = json.dumps(state.cached_recovery, indent=2)
                state.inspection_title = "Recovery Diagnostics"
                state.inspecting = True
            else:
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

        elif ch == "\xe0":  # Windows arrow keys
            if msvcrt.kbhit():
                ch2 = msvcrt.getwch()
                if state.active_tab == "APPROVALS" and state.pending_requests:
                    if ch2 == "H":  # Up
                        state.selected_request_idx = max(0, state.selected_request_idx - 1)
                    elif ch2 == "P":  # Down
                        state.selected_request_idx = min(len(state.pending_requests) - 1, state.selected_request_idx + 1)
                else:
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
    operate: bool = False,
) -> int:
    """Non-interactive plain-text fallback for non-TTY, CI, or pipes."""
    root = repo_root or Path.cwd()
    state = DashboardState(root, session_id=session_id, operate=operate)
    state.refresh()

    sys.stdout.write("=== MAYA AUTOPHAGE TRACE REPORT (PLAIN) ===\n")
    if operate:
        sys.stdout.write("Mode: OPERATOR CONTROLS\n")
    sys.stdout.write(f"Session: {state.session_id or 'UNKNOWN'} | Active Tasks: {len(state.cached_tasks)}\n\n")

    # 1. Pending Approvals
    if state.pending_requests:
        sys.stdout.write(f"--- Pending Approval Requests ({len(state.pending_requests)}) ---\n")
        for req in state.pending_requests:
            sys.stdout.write(
                f"  [{req.request_id}] Task: {req.task_id} | Risk: {req.risk} | Op: {req.operation} | "
                f"Patch: {req.patch_sha256 or 'N/A'}\n"
                f"    Scope: {req.scope}\n"
                f"    Reason: {req.reason}\n"
            )
        sys.stdout.write("\n")

    # 2. Recovery Tasks
    if state.cached_recovery:
        sys.stdout.write(f"--- Recovery Required Tasks ({len(state.cached_recovery)}) ---\n")
        for r in state.cached_recovery:
            sys.stdout.write(
                f"  Task {r.get('task_id')}: {r.get('stage')} (Dirty={r.get('is_dirty')})\n"
                f"    Blocker: {'; '.join(r.get('blockers', []))}\n"
                f"    Next Safe Action: {r.get('next_safe_action')}\n"
            )
        sys.stdout.write("\n")

    # 3. Trace Events
    events = state.events
    sys.stdout.write(f"--- Recent Trace Events ({len(events)}) ---\n")
    if not events:
        sys.stdout.write("No trace events recorded.\n")
    else:
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
    operate: bool = False,
    max_fps: int = 10,
) -> int:
    """
    Launch Autophage terminal dashboard.
    Falls back to plain text if plain=True or stdout is not a TTY.
    """
    root = (repo_root or Path.cwd()).resolve()

    if plain or not sys.stdout.isatty():
        return run_plain(session_id=session_id, repo_root=root, follow=not replay, operate=operate)

    console = Console()
    state = DashboardState(root, session_id=session_id, operate=operate)
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

    console.print("Autophage TUI exited. Background coordinator and worktrees preserved.", style=GRAY)
    return 0


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    run_tui(session_id=sid)