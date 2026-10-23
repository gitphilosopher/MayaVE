"""
skills/maintenance/trace.py
Trustworthy, append-only JSON Lines trace infrastructure for MayaVE Autophage.

Design principles from .maya/reports/autophage-tui-design.md:
- Thread-safe, non-blocking append to .maya/maintenance/trace/<session_id>.jsonl
- Off by default; zero overhead when disabled
- Typed event schema with strict provenance (captured, ledger, derived)
- Provider-reported usage vs unknown (never synthesize zero for unknown tokens)
- Automatic credential, token, and secret redaction before persistence
- Strips ANSI escape codes to prevent terminal injection
- Bounded inline payloads with safe, isolated side files for large bodies
- Swallows write failures safely (tracing must NEVER fail a task)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("maya.maintenance.trace")

# Credential and token patterns for redaction
_SENSITIVE_PATTERNS = [
    re.compile(r"(api[_-]?key|token|secret|password|passwd|auth|bearer)\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    re.compile(r"(bearer\s+)[A-Za-z0-9_\-\.]{16,}", re.IGNORECASE),
    re.compile(r"(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})", re.IGNORECASE),
    re.compile(r"(https?://[^:@\s]+:[^@\s]+@[^\s]+)", re.IGNORECASE),  # URLs with user:pass
    re.compile(r"^[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|CRED|AUTH)[A-Z0-9_]*\s*=\s*.+$", re.MULTILINE),  # .env lines
]

# ANSI CSI, OSC, and escape code removal
_ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\].*?(?:\x07|\x1B\\))"
)

_VALID_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

MAX_BODY_CHARS = 65536  # 64 KB cap for side-file bodies


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and non-printable control characters."""
    if not text or not isinstance(text, str):
        return ""
    clean = _ANSI_ESCAPE_RE.sub("", text)
    # Strip dangerous control chars < 32 except newline (\n) and tab (\t)
    return "".join(c for c in clean if c == "\n" or c == "\t" or ord(c) >= 32)


def redact_text(text: str) -> tuple[str, int]:
    """
    Sanitize text by redacting credentials, secrets, and environment tokens.
    Replaces user home directory with '~' and strips ANSI escapes.
    Returns (sanitized_text, redactions_count).
    """
    if not text or not isinstance(text, str):
        return "", 0

    redactions = 0
    clean = strip_ansi(text)

    # Redact home directory paths
    try:
        home_dir = str(Path.home().resolve())
        if home_dir and home_dir in clean:
            clean = clean.replace(home_dir, "~")
            redactions += 1
    except Exception:
        pass

    # Apply sensitive credential regexes
    for pat in _SENSITIVE_PATTERNS:
        matches = len(pat.findall(clean))
        if matches > 0:
            redactions += matches
            clean = pat.sub(r"[REDACTED]", clean)

    return clean, redactions


@dataclass
class TraceEvent:
    """Standardized event record for Autophage maintenance tracing."""
    v: int = 1
    seq: int = 0
    ts: str = ""
    mono_ms: int = 0
    session_id: str = ""
    pid: int = 0
    category: str = "system"  # lifecycle, worker, request, tool, planner, routing, quality, review, approval, budget, recovery, system
    type: str = ""
    severity: str = "INFO"    # DEBUG, INFO, WARN, ERROR, CRIT
    provenance: str = "captured"  # captured, ledger, derived
    sender: str = "orchestrator"
    recipient: str = ""
    task_id: Optional[str] = None
    worker_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    summary: str = ""
    usage: dict[str, Any] = field(default_factory=lambda: {"source": "unknown"})
    latency_ms: Optional[int] = None
    latency_source: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "seq": self.seq,
            "ts": self.ts,
            "mono_ms": self.mono_ms,
            "session_id": self.session_id,
            "pid": self.pid,
            "category": self.category,
            "type": self.type,
            "severity": self.severity,
            "provenance": self.provenance,
            "sender": self.sender,
            "recipient": self.recipient,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "summary": self.summary,
            "usage": dict(self.usage),
            "latency_ms": self.latency_ms,
            "latency_source": self.latency_source,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        return cls(
            v=int(data.get("v", 1)),
            seq=int(data.get("seq", 0)),
            ts=str(data.get("ts", "")),
            mono_ms=int(data.get("mono_ms", 0)),
            session_id=str(data.get("session_id", "")),
            pid=int(data.get("pid", 0)),
            category=str(data.get("category", "system")),
            type=str(data.get("type", "")),
            severity=str(data.get("severity", "INFO")),
            provenance=str(data.get("provenance", "captured")),
            sender=str(data.get("sender", "orchestrator")),
            recipient=str(data.get("recipient", "")),
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            request_id=data.get("request_id"),
            correlation_id=data.get("correlation_id"),
            summary=str(data.get("summary", "")),
            usage=dict(data.get("usage", {"source": "unknown"})),
            latency_ms=data.get("latency_ms"),
            latency_source=data.get("latency_source"),
            payload=dict(data.get("payload", {})),
        )


class TraceManager:
    """Thread-safe append-only trace file manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._trace_dir: Optional[Path] = None
        self._seq = 0
        self._dropped_count = 0

    def enable(
        self,
        trace_dir: Optional[Path | str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._enabled = True
            if session_id:
                self._session_id = str(session_id)
            if trace_dir:
                self._trace_dir = Path(trace_dir).resolve()
            else:
                repo_root = Path(__file__).resolve().parent.parent.parent
                self._trace_dir = (repo_root / ".maya" / "maintenance" / "trace").resolve()
            try:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                (self._trace_dir / "bodies").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning("Could not create trace directory %s: %s", self._trace_dir, e)

    def disable(self) -> None:
        with self._lock:
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def trace_dir(self) -> Optional[Path]:
        return self._trace_dir

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def get_trace_file_path(self, session_id: Optional[str] = None) -> Path:
        sid = session_id or self._session_id
        if self._trace_dir:
            return self._trace_dir / f"{sid}.jsonl"
        repo_root = Path(__file__).resolve().parent.parent.parent
        return repo_root / ".maya" / "maintenance" / "trace" / f"{sid}.jsonl"

    def emit(self, event: TraceEvent | dict[str, Any]) -> None:
        """
        Record an event to the append-only JSONL trace file.
        Swallows write failures safely and increments dropped_count.
        Never raises exceptions or modifies task state.
        """
        if not self._enabled:
            return

        with self._lock:
            try:
                self._seq += 1
                if isinstance(event, dict):
                    ev_dict = dict(event)
                else:
                    ev_dict = event.to_dict()

                ev_dict["v"] = 1
                ev_dict["seq"] = self._seq
                if not ev_dict.get("ts"):
                    ev_dict["ts"] = datetime.now(timezone.utc).isoformat()
                if not ev_dict.get("mono_ms"):
                    ev_dict["mono_ms"] = int(time.monotonic() * 1000)
                if not ev_dict.get("session_id"):
                    ev_dict["session_id"] = self._session_id
                if not ev_dict.get("pid"):
                    ev_dict["pid"] = os.getpid()

                # Clean and redact summary
                raw_summary = str(ev_dict.get("summary", ""))
                clean_summary, red_count = redact_text(raw_summary)
                ev_dict["summary"] = clean_summary[:200]

                # Ensure usage semantics: never 0 for unknown
                usage = ev_dict.get("usage")
                if not usage or not isinstance(usage, dict):
                    ev_dict["usage"] = {"source": "unknown"}
                elif usage.get("source") == "unknown":
                    if usage.get("input_tokens") == 0:
                        usage["input_tokens"] = None
                    if usage.get("output_tokens") == 0:
                        usage["output_tokens"] = None

                trace_file = self.get_trace_file_path()
                trace_file.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(ev_dict, ensure_ascii=False)
                with open(trace_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
            except Exception as exc:
                self._dropped_count += 1
                logger.debug("Trace write dropped (count=%d): %s", self._dropped_count, exc)

    def write_body(
        self,
        request_id: str,
        body_type: str,
        content: str,
    ) -> tuple[Optional[str], bool, int]:
        """
        Write a bounded, redacted prompt or response side-file into .maya/maintenance/trace/bodies/.
        Returns (body_filename, was_truncated, redaction_count).
        """
        if not self._enabled or not self._trace_dir:
            return None, False, 0

        if not request_id or not _VALID_ID_RE.match(request_id):
            return None, False, 0

        clean_type = "req" if body_type == "req" else "resp"
        filename = f"{request_id}.{clean_type}.txt"
        bodies_dir = (self._trace_dir / "bodies").resolve()

        with self._lock:
            try:
                bodies_dir.mkdir(parents=True, exist_ok=True)
                target_path = (bodies_dir / filename).resolve()
                if not target_path.is_relative_to(bodies_dir):
                    logger.warning("Trace body path traversal rejected: %s", filename)
                    return None, False, 0

                clean_content, red_count = redact_text(content)
                truncated = False
                if len(clean_content) > MAX_BODY_CHARS:
                    truncated = True
                    head_chars = MAX_BODY_CHARS // 2
                    tail_chars = MAX_BODY_CHARS // 2
                    clean_content = (
                        clean_content[:head_chars]
                        + f"\n\n... [TRUNCATED {len(clean_content) - MAX_BODY_CHARS} CHARACTERS] ...\n\n"
                        + clean_content[-tail_chars:]
                    )

                target_path.write_text(clean_content, encoding="utf-8")
                return filename, truncated, red_count
            except Exception as e:
                self._dropped_count += 1
                logger.debug("Failed to write trace body: %s", e)
                return None, False, 0

    def read_body(self, body_ref: str) -> Optional[str]:
        """Read a side-file body safely with strict path containment."""
        if not body_ref or not self._trace_dir:
            return None
        # Must be simple filename
        if "/" in body_ref or "\\" in body_ref or ".." in body_ref:
            return None
        bodies_dir = (self._trace_dir / "bodies").resolve()
        target_path = (bodies_dir / body_ref).resolve()
        try:
            if not target_path.is_relative_to(bodies_dir) or not target_path.is_file():
                return None
            return target_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def prune_sessions(self, max_sessions: int = 10) -> int:
        """Prune older trace sessions keeping only the newest max_sessions."""
        if not self._trace_dir or not self._trace_dir.is_dir():
            return 0
        with self._lock:
            try:
                jsonl_files = sorted(
                    self._trace_dir.glob("*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                to_delete = jsonl_files[max_sessions:]
                deleted = 0
                for f in to_delete:
                    try:
                        f.unlink()
                        deleted += 1
                    except Exception:
                        pass
                return deleted
            except Exception:
                return 0


# Global singleton instance
_GLOBAL_TRACE = TraceManager()

# Check environment variable on startup
if os.environ.get("MAYA_TRACE") == "1":
    _GLOBAL_TRACE.enable()


def emit(event: TraceEvent | dict[str, Any]) -> None:
    _GLOBAL_TRACE.emit(event)


def enable_trace(trace_dir: Optional[Path | str] = None, session_id: Optional[str] = None) -> None:
    _GLOBAL_TRACE.enable(trace_dir=trace_dir, session_id=session_id)


def disable_trace() -> None:
    _GLOBAL_TRACE.disable()


def is_trace_enabled() -> bool:
    return _GLOBAL_TRACE.is_enabled


def write_trace_body(request_id: str, body_type: str, content: str) -> tuple[Optional[str], bool, int]:
    return _GLOBAL_TRACE.write_body(request_id, body_type, content)


def read_trace_body(body_ref: str) -> Optional[str]:
    return _GLOBAL_TRACE.read_body(body_ref)


def get_dropped_count() -> int:
    return _GLOBAL_TRACE.dropped_count


def get_global_trace() -> TraceManager:
    return _GLOBAL_TRACE
