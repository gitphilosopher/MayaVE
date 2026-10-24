"""
skills/maintenance/agentic_cli.py
AutoPhage 3.0 Agentic CLI - read-only observer workspace.
"""
from __future__ import annotations

import argparse
import io
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
except ImportError:  # non-Windows
    HAS_MSVCRT = False

from rich.box import ROUNDED
from rich.color import Color
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout

PURPLE = "rgb(180,156,255)"
GRAY = "rgb(160,157,165)"
WHITE = "white"
MINT = "rgb(128,222,184)"
AMBER = "rgb(244,201,125)"
CORAL = "rgb(255,128,146)"
ACCENT = f"bold {PURPLE}"

LOGO_MAX_WIDTH = 11
LOGO_MAX_HEIGHT_PX = 11
ALPHA_CUTOFF = 10
STREAM_ROWS = 20


# ==============================================================================
# Paths & Utils
# ==============================================================================

def resolve_repo_root(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# ==============================================================================
# Logo Rendering
# ==============================================================================

@dataclass
class LogoResult:
    text: Optional[Text]
    source: Optional[Path]
    error: Optional[str] = None


def render_image_ansi(image_path: Path) -> Text:
    from PIL import Image
    with Image.open(image_path) as im:
        rgba = im.convert("RGBA")
    iw, ih = rgba.size

    # Use the target width and calculate height based on terminal cell aspect ratio (~0.45)
    w = LOGO_MAX_WIDTH
    h = int(w * (ih / iw) * 0.45)

    rgba = rgba.resize((w, h), getattr(Image, "Resampling", Image).NEAREST)
    px = rgba.load()
    out = Text(no_wrap=True, overflow="crop")
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a < ALPHA_CUTOFF:
                out.append(" ")
            else:
                out.append("█", Style(color=Color.from_rgb(r, g, b)))
        out.append("\n")
    return out


def load_logo(repo_root: Path) -> LogoResult:
    path = repo_root / "docs" / "eve.png"
    if not path.is_file(): return LogoResult(None, path, "missing")
    try:
        return LogoResult(render_image_ansi(path), path)
    except Exception as e:
        return LogoResult(None, path, str(e))


# ==============================================================================
# Stream Tail
# ==============================================================================

class TraceTail:
    def __init__(self) -> None:
        self.path: Optional[Path] = None
        self.pos = 0
        self.buf = ""

    def switch(self, path: Optional[Path]) -> bool:
        if path == self.path: return False
        self.path, self.pos, self.buf = path, 0, ""
        return True

    def read_new(self) -> list[dict]:
        if not self.path or not self.path.is_file(): return []
        try:
            size = self.path.stat().st_size
            if size < self.pos: self.pos, self.buf = 0, ""
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
            self.pos += len(data)
        except OSError: return []
        if not data: return []
        lines = (self.buf + data.decode("utf-8", errors="replace")).split("\n")
        self.buf = lines.pop()
        events = []
        for line in lines:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict): events.append(obj)
            except Exception: continue
        return events


# ==============================================================================
# State
# ==============================================================================

class AgenticCLIState:
    def __init__(self, repo_root: Path, operate: bool = False) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.maintenance_dir = self.repo_root / ".maya" / "maintenance"
        self.trace_dir = self.maintenance_dir / "trace"
        self.session_file = self.maintenance_dir / "active_session.json"
        self.state_file = self.maintenance_dir / "state.json"
        self.operate_mode = operate

        self.tail = TraceTail()
        self.events: list[dict] = []
        self.in_tokens = 0
        self.out_tokens = 0
        self.has_provider_tokens = False
        self.session: dict[str, Any] = {}
        self.budget: dict[str, Any] = {}

        self.follow = True
        self.paused = False
        self.scroll = 0
        self.last_refresh = time.monotonic()
        self.lag_sec = 0.0
        self.banner = ""

    def refresh(self) -> None:
        now = time.monotonic()
        self.lag_sec = max(0.0, now - self.last_refresh)
        self.last_refresh = now
        self.session = _read_json(self.session_file) or {}
        state = _read_json(self.state_file) or {}
        self.budget = state.get("budget", {}) if isinstance(state.get("budget"), dict) else {}

        # Discovery
        target = None
        sid = self.session.get("session_id")
        if sid:
            cand = self.trace_dir / f"{sid}.jsonl"
            if cand.is_file(): target = cand
        if not target and self.trace_dir.is_dir():
            files = sorted(self.trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files: target = files[0]

        if self.tail.switch(target):
            self.events, self.in_tokens, self.out_tokens, self.has_provider_tokens = [], 0, 0, False
        for ev in self.tail.read_new():
            self.events.append(ev)
            usage = ev.get("usage")
            if isinstance(usage, dict) and usage.get("source") == "provider":
                self.in_tokens += int(usage.get("input_tokens", 0))
                self.out_tokens += int(usage.get("output_tokens", 0))
                self.has_provider_tokens = True

    def observed_model(self) -> Optional[str]:
        for ev in reversed(self.events):
            m = (ev.get("payload") or {}).get("model")
            if m: return str(m)
            if str(ev.get("type", "")).endswith("request.started") and ev.get("recipient"):
                return str(ev["recipient"])
        return None

    def usage_label(self) -> str:
        if self.has_provider_tokens:
            return f"in {self.in_tokens:,} / out {self.out_tokens:,} tokens [provider]"
        b_in, b_out = self.budget.get("total_input_tokens", 0), self.budget.get("total_output_tokens", 0)
        if b_in or b_out: return f"in {int(b_in):,} / out {int(b_out):,} tokens [budget]"
        return "Usage unavailable"

# ==============================================================================
# Rendering (Restored UI)
# ==============================================================================

def render_header(state: AgenticCLIState, logo: LogoResult) -> Columns:
    # Compact Header: Logo on left, info on right
    info = Text()
    info.append(" AutoPhage 3.0\n", style=ACCENT)

    model = state.observed_model()
    usage = state.usage_label()
    info.append(f" model: {model or 'Unknown'}  ·  {usage}\n", style=WHITE)

    # Second line: Working Directory
    info.append(f" {state.repo_root}", style=GRAY)

    logo_text = logo.text if logo.text else Text(" [LOGO] ", style=CORAL)
    return Columns([
        logo_text,
        info,
        Rule(characters="=", style=PURPLE)
    ])

def render_worker_block(role: str, content: str, style: str = WHITE) -> Panel:
    role_map = {"planner": "[Planner]", "macro": "[MacroAgent]", "reviewer": "[Reviewer]", "coordinator": "[Coordinator]"}
    label = role_map.get(role.lower(), f"[{role}]")
    return Panel(Text(content, style=style), title=Text(f"{label} ", style=ACCENT), title_align="left", border_style=GRAY, box=ROUNDED, padding=(0, 1))

def render_diff_block(diff_text: str) -> Panel:
    text = Text()
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"): text.append(line + "\n", style="green")
        elif line.startswith("-") and not line.startswith("---"): text.append(line + "\n", style="dim red")
        else: text.append(line + "\n", style=GRAY)
    return Panel(text, title=Text("File Edit", style=GRAY), border_style=GRAY, box=ROUNDED)

def render_main_view(state: AgenticCLIState) -> Layout:
    layout = Layout()
    events = state.events

    stream_content = []
    # Tail last N events
    for ev in events[-STREAM_ROWS:]:
        role = ev.get("sender", "unknown").split(":")[0]
        summary = ev.get("summary", "")
        if "diff" in ev.get("type", "").lower() or "edit" in ev.get("type", "").lower():
            diff = ev.get("payload", {}).get("diff", "No diff available")
            stream_content.append(render_diff_block(diff))
        elif "tool" in ev.get("type", "").lower():
            stream_content.append(render_worker_block(role, f"Tool Call: {summary}", style=GRAY))
        else:
            stream_content.append(render_worker_block(role, summary))

    if not stream_content:
        stream_content.append(Panel(Text("No active agentic stream. Start a task to begin.", style=GRAY), border_style=GRAY))

    layout.split_column(
        Layout(Group(*stream_content), ratio=1),
        Layout(render_status_dock(state), size=2)
    )
    return layout

def render_status_dock(state: AgenticCLIState) -> Columns:
    status = "IDLE"
    color = WHITE
    # Simple honest status from session
    if state.session.get("status") == "RUNNING":
        status = "RUNNING"
        color = PURPLE

    dock = Text()
    dock.append(f" STATUS: {status} ", style=f"bold {color}")
    dock.append(f"  |  MODE: {'OPERATE' if state.operate_mode else 'OBSERVE'} ", style=GRAY)
    dock.append(f"  |  TABS: Tab to cycle views", style=GRAY)
    return Columns([Rule(characters="─", style=GRAY), dock])

def render_frame(state: AgenticCLIState, logo: LogoResult) -> Group:
    return Group(render_header(state, logo), render_main_view(state))

# ==============================================================================
# Entry Point
# ==============================================================================

def poll_keyboard(state: AgenticCLIState) -> bool:
    if not HAS_MSVCRT: return True
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("q", "\x03"): return False
        if ch == " ": state.paused = not state.paused
        if ch == "f": state.follow = not state.follow
    return True

def run_agentic_cli(repo_root: Path, operate: bool = False, once: bool = False):
    state = AgenticCLIState(repo_root, operate=operate)
    logo = load_logo(state.repo_root)

    console = Console()
    try:
        if once or not sys.stdout.isatty():
            Console(highlight=False).print(render_frame(state, logo))
            return
        with Live(render_frame(state, logo), console=console, refresh_per_second=4, screen=True) as live:
            while True:
                if not state.paused: state.refresh()
                live.update(render_frame(state, logo))
                if not poll_keyboard(state): break
                time.sleep(0.2)
    except KeyboardInterrupt: pass

def main():
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--operate", action="store_true"); parser.add_argument("--once", action="store_true")
    parser.add_argument("--trace-file", default=None, help="Pin a specific trace .jsonl file")
    args = parser.parse_args()
    run_agentic_cli(Path.cwd(), operate=args.operate, once=args.once, trace_file=Path(args.trace_file).resolve() if args.trace_file else None)

if __name__ == "__main__":
    main()
