"""
Deterministic Pre-Review Security Scan and Tier-Based Diff Limits for MayaVE Autophage.
Enforces deterministic security checks and diff bounding before candidate patches reach
external reviewers (Copilot, etc.) or downstream integration.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    from .models import TaskDefinition, TaskRisk
except ImportError:
    from models import TaskDefinition, TaskRisk

logger = logging.getLogger("maya.maintenance.reviewer_safeguards")


class SafeguardSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ==============================================================================
# Secret Detection & Redaction
# ==============================================================================

# High-precision patterns for structured tokens and keys
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OPENAI_KEY", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("GITHUB_TOKEN", re.compile(r"(gh[pousr]_[A-Za-z0-9_]{36,}|github_pat_[A-Za-z0-9_]{82})")),
    ("AWS_KEY_ID", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GOOGLE_API_KEY", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("SLACK_TOKEN", re.compile(r"xox[baprs]-[0-9A-Za-z_-]{10,}")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z0-9_\- ]+PRIVATE KEY-----")),
    (
        "GENERIC_SECRET_ASSIGNMENT",
        re.compile(
            r"(?i)(api[_-]?key|secret|password|auth[_-]?token|access[_-]?token)\s*[:=]\s*[\"']([A-Za-z0-9+/=_\-.]{16,})[\"']"
        ),
    ),
]

_NON_SECRET_PLACEHOLDERS = {
    "todo", "replace_me", "your_key", "dummy", "test", "change_me",
    "example", "00000000", "placeholder", "none", "secret", "password",
}


def redact_secret(text: str) -> str:
    """
    Mask all detected secrets and high-entropy credential tokens in text.
    Never exposes raw secret values in findings, messages, or traces.
    """
    if not text:
        return ""

    sanitized = text

    # Mask private key blocks
    sanitized = re.sub(
        r"-----BEGIN [A-Z0-9_\- ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9_\- ]+PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY_BLOCK]",
        sanitized,
    )

    # Mask individual key patterns
    for name, pattern in _SECRET_PATTERNS:
        if name == "GENERIC_SECRET_ASSIGNMENT":
            def _replace_assignment(m: re.Match) -> str:
                var_name = m.group(1)
                secret_val = m.group(2)
                if secret_val.lower() in _NON_SECRET_PLACEHOLDERS or len(set(secret_val)) <= 3:
                    return m.group(0)
                shorthash = hashlib.sha256(secret_val.encode("utf-8")).hexdigest()[:8]
                return f'{var_name}="[REDACTED_SECRET:{shorthash}]"'
            sanitized = pattern.sub(_replace_assignment, sanitized)
        else:
            def _replace_token(m: re.Match, token_name=name) -> str:
                token = m.group(0)
                prefix = token[:4] if len(token) > 4 else "***"
                shorthash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
                return f"{prefix}***[REDACTED:{token_name}:{shorthash}]***"
            sanitized = pattern.sub(_replace_token, sanitized)

    return sanitized


# ==============================================================================
# Structured Diff Model & Parser
# ==============================================================================

@dataclass
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: list[str]
    added_lines: list[tuple[int, str]] = field(default_factory=list)  # (new_lineno, content)
    deleted_lines: list[tuple[int, str]] = field(default_factory=list)  # (old_lineno, content)


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False
    hunks: list[DiffHunk] = field(default_factory=list)
    added_lines_count: int = 0
    deleted_lines_count: int = 0

    @property
    def total_changed_lines(self) -> int:
        return self.added_lines_count + self.deleted_lines_count


@dataclass
class ParsedDiff:
    raw_diff: str
    files: list[FileDiff]
    total_added: int
    total_deleted: int
    total_files: int
    diff_sha256: str


def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """
    Deterministically parse a unified git diff into structured file diffs and hunks.
    """
    diff_hasher = hashlib.sha256()
    diff_hasher.update(diff_text.encode("utf-8"))
    diff_sha256 = diff_hasher.hexdigest()

    files: list[FileDiff] = []
    lines = diff_text.splitlines()
    i = 0
    n = len(lines)

    current_file: Optional[FileDiff] = None
    current_hunk: Optional[DiffHunk] = None
    curr_old_line = 0
    curr_new_line = 0

    total_added = 0
    total_deleted = 0

    while i < n:
        line = lines[i]

        if line.startswith("diff --git "):
            parts = line.split(" ")
            old_p = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else ""
            new_p = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else ""
            current_file = FileDiff(old_path=old_p, new_path=new_p)
            files.append(current_file)
            current_hunk = None
            i += 1
            continue

        if current_file is None:
            i += 1
            continue

        if line.startswith("new file mode "):
            current_file.is_new = True
            i += 1
            continue

        if line.startswith("deleted file mode "):
            current_file.is_deleted = True
            i += 1
            continue

        if line.startswith("Binary files "):
            current_file.is_binary = True
            i += 1
            continue

        if line.startswith("--- "):
            if line.startswith("--- /dev/null"):
                current_file.is_new = True
            else:
                current_file.old_path = line[4:].strip().lstrip("a/")
            i += 1
            continue

        if line.startswith("+++ "):
            if line.startswith("+++ /dev/null"):
                current_file.is_deleted = True
            else:
                current_file.new_path = line[4:].strip().lstrip("b/")
            i += 1
            continue

        if line.startswith("@@ "):
            # Hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_match = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$", line)
            if hunk_match:
                old_start = int(hunk_match.group(1))
                old_count = int(hunk_match.group(2) or 1)
                new_start = int(hunk_match.group(3))
                new_count = int(hunk_match.group(4) or 1)
                header = hunk_match.group(5)
                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=header,
                    lines=[],
                )
                current_file.hunks.append(current_hunk)
                curr_old_line = old_start
                curr_new_line = new_start
            i += 1
            continue

        if current_hunk is not None:
            current_hunk.lines.append(line)
            if line.startswith("+") and not line.startswith("+++"):
                content = line[1:]
                current_hunk.added_lines.append((curr_new_line, content))
                current_file.added_lines_count += 1
                total_added += 1
                curr_new_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                content = line[1:]
                current_hunk.deleted_lines.append((curr_old_line, content))
                current_file.deleted_lines_count += 1
                total_deleted += 1
                curr_old_line += 1
            else:
                curr_old_line += 1
                curr_new_line += 1

        i += 1

    return ParsedDiff(
        raw_diff=diff_text,
        files=files,
        total_added=total_added,
        total_deleted=total_deleted,
        total_files=len(files),
        diff_sha256=diff_sha256,
    )


# ==============================================================================
# Tier-Based Diff Limits
# ==============================================================================

@dataclass
class TierDiffLimits:
    """Configurable boundaries for actual candidate diffs per task risk tier."""
    max_files: int
    max_added_lines: int
    max_deleted_lines: int
    max_total_lines: int
    max_file_lines: int  # Max added+deleted lines in any single file


DEFAULT_TIER_LIMITS: dict[TaskRisk, TierDiffLimits] = {
    TaskRisk.LOW: TierDiffLimits(
        max_files=4,
        max_added_lines=200,
        max_deleted_lines=60,
        max_total_lines=250,
        max_file_lines=150,
    ),
    TaskRisk.MEDIUM: TierDiffLimits(
        max_files=6,
        max_added_lines=400,
        max_deleted_lines=150,
        max_total_lines=550,
        max_file_lines=300,
    ),
    TaskRisk.HIGH: TierDiffLimits(
        max_files=10,
        max_added_lines=700,
        max_deleted_lines=350,
        max_total_lines=900,
        max_file_lines=500,
    ),
    TaskRisk.VERY_HIGH: TierDiffLimits(
        max_files=12,
        max_added_lines=1000,
        max_deleted_lines=500,
        max_total_lines=1200,
        max_file_lines=600,
    ),
}


# ==============================================================================
# Findings & Scan Report Model
# ==============================================================================

@dataclass
class SafeguardFinding:
    rule_id: str
    severity: SafeguardSeverity
    category: str  # NETWORK, SUBPROCESS, MANIFEST, SECRETS, DELETION, GOVERNANCE, DIFF_LIMIT
    file: str
    line_number: Optional[int]
    message: str
    snippet: str = ""  # Always sanitized & redacted
    location: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value if isinstance(self.severity, SafeguardSeverity) else str(self.severity),
            "category": self.category,
            "file": self.file,
            "line_number": self.line_number,
            "message": self.message,
            "snippet": self.snippet,
            "location": self.location,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SafeguardFinding:
        return cls(
            rule_id=data.get("rule_id", "UNKNOWN"),
            severity=SafeguardSeverity(data.get("severity", SafeguardSeverity.HIGH.value)),
            category=data.get("category", "SECURITY"),
            file=data.get("file", ""),
            line_number=data.get("line_number"),
            message=data.get("message", ""),
            snippet=data.get("snippet", ""),
            location=data.get("location", ""),
        )


@dataclass
class ScanReport:
    passed: bool
    findings: list[SafeguardFinding] = field(default_factory=list)
    diff_sha256: str = ""
    total_added: int = 0
    total_deleted: int = 0
    total_files: int = 0
    limits_exceeded: list[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_reason: Optional[str] = None

    def get_approval_scope(self) -> str:
        """
        Deterministic scope string bound to the exact findings and patch hash.
        Used for exact matching in PermissionRequest and TerminalApprovalManager.
        """
        rule_set = sorted(set(f.rule_id for f in self.findings))
        if self.limits_exceeded and "DIFF_LIMIT_EXCEEDED" not in rule_set:
            rule_set.append("DIFF_LIMIT_EXCEEDED")
        rules_str = ",".join(rule_set) if rule_set else "NONE"
        return f"rules={rules_str};diff_sha256={self.diff_sha256}"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
            "diff_sha256": self.diff_sha256,
            "total_added": self.total_added,
            "total_deleted": self.total_deleted,
            "total_files": self.total_files,
            "limits_exceeded": self.limits_exceeded,
            "requires_approval": self.requires_approval,
            "approval_reason": self.approval_reason,
            "scope": self.get_approval_scope(),
        }


# ==============================================================================
# Security Detection Rules & Scanner
# ==============================================================================

# Governance files whose modification requires immediate human approval
_GOVERNANCE_FILES = {
    "skills/maintenance/terminal_approval.py",
    "skills/maintenance/hub.py",
    "skills/maintenance/quality.py",
    "skills/maintenance/worktree.py",
    "skills/maintenance/task_contract.py",
    "skills/maintenance/reviewer_safeguards.py",
    "skills/maintenance/store.py",
    "skills/maintenance/coordinator.py",
}

# Manifest & package dependency files
_MANIFEST_PATTERNS = [
    re.compile(r"^requirements([-_a-zA-Z0-9]*)\.txt$"),
    re.compile(r"^pyproject\.toml$"),
    re.compile(r"^setup\.(py|cfg)$"),
    re.compile(r"^Pipfile(\.lock)?$"),
    re.compile(r"^package(-lock)?\.json$"),
    re.compile(r"^(pnpm-lock\.yaml|yarn\.lock)$"),
    re.compile(r"^environment\.ya?ml$"),
]

# Network modules and calls
_NETWORK_MODULES = {
    "socket", "urllib", "requests", "httpx", "aiohttp", "websocket",
    "http.client", "ftplib", "poplib", "imaplib", "smtplib", "xmlrpc",
    "paramiko", "asyncio.open_connection",
}

_NETWORK_FUNCTIONS = {
    "socket.socket", "socket.create_connection",
    "requests.get", "requests.post", "requests.put", "requests.delete", "requests.request",
    "httpx.get", "httpx.post", "httpx.Client", "httpx.AsyncClient",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "aiohttp.ClientSession", "fetch", "axios", "XMLHttpRequest", "WebSocket",
}

# Subprocess & shell execution calls
_SUBPROCESS_CALLS = {
    "subprocess.run", "subprocess.Popen", "subprocess.call",
    "subprocess.check_call", "subprocess.check_output",
    "subprocess.getstatusoutput", "subprocess.getoutput",
    "os.system", "os.popen", "os.spawn", "os.spawnl", "os.spawnv",
    "os.startfile", "exec", "eval",
}

_SUSPICIOUS_DELETION_PATTERNS = [
    ("ASSERTION_REMOVAL", re.compile(r"^\s*assert\s+")),
    ("TEST_REMOVAL", re.compile(r"^\s*def\s+test_")),
    ("SECURITY_GATE_REMOVAL", re.compile(r"(is_operation_approved|allow_file_deletion|require_authorization|verify_patch|FORBIDDEN_)")),
]


class ReviewerSafeguardScanner:
    """
    Deterministic pre-review security scanner and tier-based diff limit enforcer.
    Evaluates candidate diffs prior to external reviewer handoff.
    Fails closed on parse/diff errors, secret exposures, or policy violations.
    """

    def __init__(self, limits: dict[TaskRisk, TierDiffLimits] | None = None) -> None:
        self.limits = limits or DEFAULT_TIER_LIMITS

    def scan_diff(
        self,
        task: TaskDefinition,
        diff_text: str,
        worktree_path: Optional[Path] = None,
    ) -> ScanReport:
        """
        Execute deterministic security scan and tier limit checks on the candidate diff.
        """
        # 1. Validation & Fail-Closed Guard
        if not diff_text or not diff_text.strip():
            # Empty diff: fail closed
            finding = SafeguardFinding(
                rule_id="EMPTY_DIFF",
                severity=SafeguardSeverity.HIGH,
                category="DIFF_VALIDATION",
                file="",
                line_number=None,
                message="Candidate diff is empty or missing; cannot verify patch safety.",
            )
            return ScanReport(
                passed=False,
                findings=[finding],
                diff_sha256="",
                requires_approval=True,
                approval_reason="Candidate diff is empty or missing.",
            )

        try:
            parsed = parse_unified_diff(diff_text)
        except Exception as e:
            logger.exception("Failed to parse candidate diff: %s", e)
            finding = SafeguardFinding(
                rule_id="DIFF_PARSE_ERROR",
                severity=SafeguardSeverity.CRITICAL,
                category="DIFF_VALIDATION",
                file="",
                line_number=None,
                message=f"Candidate diff failed structured parsing: {e}",
            )
            return ScanReport(
                passed=False,
                findings=[finding],
                diff_sha256="",
                requires_approval=True,
                approval_reason=f"Diff parsing error: {e}",
            )

        findings: list[SafeguardFinding] = []
        limits_exceeded: list[str] = []

        # 2. Tier-Based Diff Limits Enforcement
        risk_tier = task.risk if isinstance(task.risk, TaskRisk) else TaskRisk(task.risk)
        tier_bounds = self.limits.get(risk_tier, DEFAULT_TIER_LIMITS[TaskRisk.MEDIUM])

        # Exclude generated/ignored files (.maya/ runtime, __pycache__) from file counts
        audited_files = [
            f for f in parsed.files
            if not f.new_path.startswith(".maya/")
            and "__pycache__" not in f.new_path
            and not f.new_path.endswith(".pyc")
        ]

        if len(audited_files) > tier_bounds.max_files:
            msg = f"Changed files count ({len(audited_files)}) exceeds {risk_tier.value} limit ({tier_bounds.max_files})."
            limits_exceeded.append(msg)
            findings.append(SafeguardFinding(
                rule_id="DIFF_LIMIT_EXCEEDED",
                severity=SafeguardSeverity.HIGH,
                category="DIFF_LIMIT",
                file="",
                line_number=None,
                message=msg,
            ))

        if parsed.total_added > tier_bounds.max_added_lines:
            msg = f"Added lines ({parsed.total_added}) exceeds {risk_tier.value} limit ({tier_bounds.max_added_lines})."
            limits_exceeded.append(msg)
            findings.append(SafeguardFinding(
                rule_id="DIFF_LIMIT_EXCEEDED",
                severity=SafeguardSeverity.HIGH,
                category="DIFF_LIMIT",
                file="",
                line_number=None,
                message=msg,
            ))

        if parsed.total_deleted > tier_bounds.max_deleted_lines:
            msg = f"Deleted lines ({parsed.total_deleted}) exceeds {risk_tier.value} limit ({tier_bounds.max_deleted_lines})."
            limits_exceeded.append(msg)
            findings.append(SafeguardFinding(
                rule_id="DIFF_LIMIT_EXCEEDED",
                severity=SafeguardSeverity.HIGH,
                category="DIFF_LIMIT",
                file="",
                line_number=None,
                message=msg,
            ))

        total_lines = parsed.total_added + parsed.total_deleted
        if total_lines > tier_bounds.max_total_lines:
            msg = f"Total changed lines ({total_lines}) exceeds {risk_tier.value} limit ({tier_bounds.max_total_lines})."
            limits_exceeded.append(msg)
            findings.append(SafeguardFinding(
                rule_id="DIFF_LIMIT_EXCEEDED",
                severity=SafeguardSeverity.HIGH,
                category="DIFF_LIMIT",
                file="",
                line_number=None,
                message=msg,
            ))

        # Check per-file lines
        for f in audited_files:
            if f.total_changed_lines > tier_bounds.max_file_lines:
                msg = f"Changed lines in '{f.new_path}' ({f.total_changed_lines}) exceeds {risk_tier.value} per-file limit ({tier_bounds.max_file_lines})."
                limits_exceeded.append(msg)
                findings.append(SafeguardFinding(
                    rule_id="DIFF_LIMIT_EXCEEDED",
                    severity=SafeguardSeverity.HIGH,
                    category="DIFF_LIMIT",
                    file=f.new_path,
                    line_number=None,
                    message=msg,
                ))

        # 3. Security Rule Scans on Each File Diff
        for file_diff in parsed.files:
            fpath = file_diff.new_path or file_diff.old_path
            norm_path = fpath.replace("\\", "/")
            basename = Path(norm_path).name

            # Skip untracked runtime .maya files from security scan (handled by local policy)
            if norm_path.startswith(".maya/"):
                continue

            # --- Rule GOV001: Core Governance or Enforcement File Modified ---
            if norm_path in _GOVERNANCE_FILES:
                findings.append(SafeguardFinding(
                    rule_id="GOV001",
                    severity=SafeguardSeverity.CRITICAL,
                    category="GOVERNANCE",
                    file=norm_path,
                    line_number=None,
                    message=f"Candidate patch modifies protected maintenance governance component '{norm_path}'.",
                ))

            # --- Rule DEP001: Package Manifest or Dependency Modification ---
            if any(p.match(basename) for p in _MANIFEST_PATTERNS):
                findings.append(SafeguardFinding(
                    rule_id="DEP001",
                    severity=SafeguardSeverity.HIGH,
                    category="MANIFEST",
                    file=norm_path,
                    line_number=None,
                    message=f"Candidate patch modifies package manifest/dependencies in '{norm_path}'.",
                ))

            # Scan Added Lines and Deleted Lines across Hunks
            is_py = norm_path.endswith(".py")

            for hunk in file_diff.hunks:
                # A. Scan added lines for secrets, network, subprocess
                for lineno, content in hunk.added_lines:
                    stripped = content.strip()
                    if not stripped:
                        continue

                    # SEC001: Secret / Credential Detection
                    self._scan_for_secrets(content, norm_path, lineno, findings)

                    # NET001: Network Access
                    self._scan_for_network(content, norm_path, lineno, findings, is_python=is_py)

                    # SUB001: Subprocess / Shell Execution
                    self._scan_for_subprocess(content, norm_path, lineno, findings, is_python=is_py)

                # B. Scan deleted lines for suspicious removals of safeguards/tests
                for lineno, content in hunk.deleted_lines:
                    self._scan_for_suspicious_deletions(content, norm_path, lineno, findings)

            # C. AST-level verification for Python files
            if is_py and worktree_path is not None:
                full_path = worktree_path / norm_path
                if full_path.is_file():
                    self._ast_scan_python_file(full_path, norm_path, file_diff, findings)

        # 4. Evaluation and Approval Hold Assessment
        high_critical = [
            f for f in findings
            if f.severity in (SafeguardSeverity.CRITICAL, SafeguardSeverity.HIGH)
        ]

        requires_approval = len(high_critical) > 0 or len(limits_exceeded) > 0
        passed = not requires_approval

        approval_reason = None
        if requires_approval:
            parts = []
            if limits_exceeded:
                parts.append(f"Diff limits exceeded: {'; '.join(limits_exceeded)}")
            if high_critical:
                rule_summaries = [f"{f.rule_id} in {f.file or 'diff'}: {f.message}" for f in high_critical[:3]]
                if len(high_critical) > 3:
                    rule_summaries.append(f"(+{len(high_critical) - 3} more finding(s))")
                parts.append(f"Security findings: {'; '.join(rule_summaries)}")
            approval_reason = "; ".join(parts)

        return ScanReport(
            passed=passed,
            findings=findings,
            diff_sha256=parsed.diff_sha256,
            total_added=parsed.total_added,
            total_deleted=parsed.total_deleted,
            total_files=len(audited_files),
            limits_exceeded=limits_exceeded,
            requires_approval=requires_approval,
            approval_reason=approval_reason,
        )

    # --------------------------------------------------------------------------
    # Rule Scanners
    # --------------------------------------------------------------------------

    def _scan_for_secrets(
        self,
        line: str,
        file_path: str,
        lineno: int,
        findings: list[SafeguardFinding],
    ) -> None:
        """Scan line for secrets with mandatory redaction."""
        for name, pattern in _SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                if name == "GENERIC_SECRET_ASSIGNMENT":
                    secret_val = match.group(2)
                    if secret_val.lower() in _NON_SECRET_PLACEHOLDERS or len(set(secret_val)) <= 3:
                        continue
                # Sanitize snippet completely
                redacted_snippet = redact_secret(line.strip())
                findings.append(SafeguardFinding(
                    rule_id="SEC001",
                    severity=SafeguardSeverity.HIGH,
                    category="SECRETS",
                    file=file_path,
                    line_number=lineno,
                    message=f"Potentially exposed credential or secret token detected ({name}).",
                    snippet=redacted_snippet,
                    location=f"{file_path}:{lineno}",
                ))
                break

    def _scan_for_network(
        self,
        line: str,
        file_path: str,
        lineno: int,
        findings: list[SafeguardFinding],
        is_python: bool,
    ) -> None:
        """Scan line for newly introduced outbound network calls or imports."""
        stripped = line.strip()

        if is_python:
            # Check Python imports
            if stripped.startswith("import ") or stripped.startswith("from "):
                for mod in _NETWORK_MODULES:
                    if re.search(rf"\b(import|from)\s+{re.escape(mod)}\b", stripped):
                        findings.append(SafeguardFinding(
                            rule_id="NET001",
                            severity=SafeguardSeverity.HIGH,
                            category="NETWORK",
                            file=file_path,
                            line_number=lineno,
                            message=f"Newly introduced network module import: '{mod}'.",
                            snippet=stripped[:120],
                            location=f"{file_path}:{lineno}",
                        ))
                        return

        # Check function invocations
        for net_fn in _NETWORK_FUNCTIONS:
            if net_fn in stripped:
                findings.append(SafeguardFinding(
                    rule_id="NET001",
                    severity=SafeguardSeverity.HIGH,
                    category="NETWORK",
                    file=file_path,
                    line_number=lineno,
                    message=f"Newly introduced outbound network call: '{net_fn}'.",
                    snippet=stripped[:120],
                    location=f"{file_path}:{lineno}",
                ))
                return

    def _scan_for_subprocess(
        self,
        line: str,
        file_path: str,
        lineno: int,
        findings: list[SafeguardFinding],
        is_python: bool,
    ) -> None:
        """Scan line for subprocess, shell invocation, or os.startfile."""
        stripped = line.strip()

        # Check shell=True
        if "shell=True" in stripped.replace(" ", ""):
            findings.append(SafeguardFinding(
                rule_id="SUB001",
                severity=SafeguardSeverity.HIGH,
                category="SUBPROCESS",
                file=file_path,
                line_number=lineno,
                message="Subprocess shell invocation ('shell=True') detected.",
                snippet=stripped[:120],
                location=f"{file_path}:{lineno}",
            ))
            return

        # Check subprocess/shell function calls
        for sub_call in _SUBPROCESS_CALLS:
            if sub_call in stripped:
                findings.append(SafeguardFinding(
                    rule_id="SUB001",
                    severity=SafeguardSeverity.HIGH,
                    category="SUBPROCESS",
                    file=file_path,
                    line_number=lineno,
                    message=f"Subprocess, shell, or process execution call detected: '{sub_call}'.",
                    snippet=stripped[:120],
                    location=f"{file_path}:{lineno}",
                ))
                return

    def _scan_for_suspicious_deletions(
        self,
        line: str,
        file_path: str,
        lineno: int,
        findings: list[SafeguardFinding],
    ) -> None:
        """Scan deleted line for removal of assertions, tests, or security enforcement."""
        stripped = line.strip()
        for rule_suffix, pat in _SUSPICIOUS_DELETION_PATTERNS:
            if pat.search(stripped):
                findings.append(SafeguardFinding(
                    rule_id=f"DEL_{rule_suffix}",
                    severity=SafeguardSeverity.HIGH,
                    category="DELETION",
                    file=file_path,
                    line_number=lineno,
                    message=f"Suspicious deletion: removal of {rule_suffix.lower().replace('_', ' ')}.",
                    snippet=stripped[:120],
                    location=f"{file_path}:{lineno}",
                ))
                break

    def _ast_scan_python_file(
        self,
        full_path: Path,
        rel_path: str,
        file_diff: FileDiff,
        findings: list[SafeguardFinding],
    ) -> None:
        """
        Perform AST analysis on the patched worktree Python file to detect structured
        AST nodes in newly added line ranges that simple text checks might miss.
        """
        try:
            tree = ast.parse(full_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            # If full file is invalid Python, py_compile quality gate already caught it
            return

        added_linenos = set()
        for hunk in file_diff.hunks:
            for lno, _ in hunk.added_lines:
                added_linenos.add(lno)

        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", None)
            if lineno not in added_linenos:
                continue

            # AST Check: Subprocess Call with shell=True
            if isinstance(node, ast.Call):
                call_name = self._resolve_ast_func_name(node.func)
                for kw in node.keywords:
                    if kw.arg == "shell":
                        if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            # Avoid duplicate finding if line regex already caught it
                            if not any(f.file == rel_path and f.line_number == lineno and f.rule_id == "SUB001" for f in findings):
                                findings.append(SafeguardFinding(
                                    rule_id="SUB001",
                                    severity=SafeguardSeverity.HIGH,
                                    category="SUBPROCESS",
                                    file=rel_path,
                                    line_number=lineno,
                                    message=f"AST verification: '{call_name}' invoked with shell=True.",
                                    location=f"{rel_path}:{lineno}",
                                ))

            # AST Check: Dynamic eval/exec
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    if not any(f.file == rel_path and f.line_number == lineno and f.rule_id == "SUB001" for f in findings):
                        findings.append(SafeguardFinding(
                            rule_id="SUB001",
                            severity=SafeguardSeverity.HIGH,
                            category="SUBPROCESS",
                            file=rel_path,
                            line_number=lineno,
                            message=f"AST verification: Dynamic execution function '{node.func.id}' introduced.",
                            location=f"{rel_path}:{lineno}",
                        ))

    @staticmethod
    def _resolve_ast_func_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val_name = ReviewerSafeguardScanner._resolve_ast_func_name(node.value)
            return f"{val_name}.{node.attr}" if val_name else node.attr
        return ""
