"""
MayaVE VE11 — Persistent Terminal Approval Manager & Session Reporting.

Provides durable, file-backed permission request tracking, interactive operator
prompts (Approve, Deny, Inspect, Skip, Exit), exact-scope validation, and session
summary generation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    from .models import TaskStatus
    from .store import StateStore
    from .hub import CoordinatorAPI
except ImportError:
    from models import TaskStatus
    from store import StateStore
    from hub import CoordinatorAPI

logger = logging.getLogger("maya.maintenance.terminal_approval")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    SKIPPED = "SKIPPED"


@dataclass
class PermissionRequest:
    request_id: str
    task_id: str
    risk: str
    operation: str
    target: str
    scope: str
    reason: str
    requested_at: str = field(default_factory=_utc_now_iso)
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_at: Optional[str] = None
    decision_by: Optional[str] = None
    decision_scope: Optional[str] = None
    decision_reason: Optional[str] = None
    patch_sha256: Optional[str] = None
    baseline_commit: Optional[str] = None
    verification_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "risk": self.risk,
            "operation": self.operation,
            "target": self.target,
            "scope": self.scope,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "status": self.status.value if isinstance(self.status, ApprovalStatus) else str(self.status),
            "decision_at": self.decision_at,
            "decision_by": self.decision_by,
            "decision_scope": self.decision_scope,
            "decision_reason": self.decision_reason,
            "patch_sha256": self.patch_sha256,
            "baseline_commit": self.baseline_commit,
            "verification_sha256": self.verification_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PermissionRequest:
        status_val = data.get("status", ApprovalStatus.PENDING.value)
        try:
            status = ApprovalStatus(status_val)
        except ValueError:
            status = ApprovalStatus.PENDING
        return cls(
            request_id=data["request_id"],
            task_id=data["task_id"],
            risk=data.get("risk", "MEDIUM"),
            operation=data["operation"],
            target=data["target"],
            scope=data["scope"],
            reason=data.get("reason", ""),
            requested_at=data.get("requested_at", _utc_now_iso()),
            status=status,
            decision_at=data.get("decision_at"),
            decision_by=data.get("decision_by"),
            decision_scope=data.get("decision_scope"),
            decision_reason=data.get("decision_reason"),
            patch_sha256=data.get("patch_sha256"),
            baseline_commit=data.get("baseline_commit"),
            verification_sha256=data.get("verification_sha256"),
        )


class TerminalApprovalManager:
    """
    Manages persistent terminal permission requests and interactive operator review.
    Guarantees that approvals apply only to the exact reviewed operation and scope.
    """

    def __init__(
        self,
        approvals_path: Path | str,
        repo_root: Path | str | None = None,
        store: Optional[StateStore] = None,
    ) -> None:
        self.approvals_path = Path(approvals_path).resolve()
        self.repo_root = Path(repo_root).resolve() if repo_root else self.approvals_path.parent.parent.parent
        if store:
            self.store = store
        else:
            default_state = self.repo_root / ".maya" / "maintenance" / "state.json"
            self.store = StateStore(default_state) if default_state.exists() else None
        self._requests: Dict[str, PermissionRequest] = {}
        self.load()

    def sync_approval_to_store(self, req: PermissionRequest) -> bool:
        """
        Synchronize an approved permission request to state.json.
        Transitions the task from NEEDS_APPROVAL or RECOVERY_REQUIRED to PENDING, clears blocking_reason,
        records an explicit checkpoint, and persists atomically under lock.
        """
        if not self.store or not self.store.state_path.exists():
            return False
        if req.operation == "integration":
            result = CoordinatorAPI(self.store).approve_integration(
                task_id=req.task_id,
                request_id=req.request_id,
                patch_sha256=req.patch_sha256 or "",
                baseline_commit=req.baseline_commit or "",
                verification_sha256=req.verification_sha256 or "",
                approved_by=req.decision_by or "operator",
            )
            if not result.accepted:
                logger.error("Integration approval was not persisted: %s", result.message)
            return result.accepted
        ledger = self.store.load_ledger()
        if req.task_id not in ledger:
            logger.warning("Task %s not found in ledger during approval sync.", req.task_id)
            return False
        task = ledger.get_task(req.task_id)
        if task.status not in (TaskStatus.NEEDS_APPROVAL, TaskStatus.RECOVERY_REQUIRED):
            return False
        result = CoordinatorAPI(self.store).transition_task(
            ledger, task.task_id, TaskStatus.PENDING, actor="approval",
            checkpoint_stage="APPROVAL_GRANTED",
            checkpoint_details={
                "request_id": req.request_id,
                "operation": req.operation,
                "target": req.target,
                "decision_scope": req.decision_scope,
                "decision_by": req.decision_by,
            },
            event_type="APPROVAL_GRANTED",
            idempotency_key=f"approval-transition:{req.request_id}",
        )
        if result.accepted:
            logger.info("Synchronized approval of %s to state.json for task %s.", req.request_id, task.task_id)
        else:
            logger.error("Approval transition was not persisted: %s", result.message)
        return result.accepted

    def _record_lifecycle_event(self, req: PermissionRequest, event_type: str) -> bool:
        if not self.store or not self.store.exists():
            return True
        try:
            ledger = self.store.load_ledger()
            if req.task_id not in ledger:
                return False
            result = CoordinatorAPI(self.store).record_event(
                ledger,
                req.task_id,
                event_type,
                "approval",
                details={
                    "request_id": req.request_id,
                    "operation": req.operation,
                    "target": req.target,
                    "decision_scope": req.decision_scope or req.scope,
                    "decision_by": req.decision_by,
                    "patch_sha256": req.patch_sha256,
                    "baseline_commit": req.baseline_commit,
                    "verification_sha256": req.verification_sha256,
                },
                idempotency_key=f"approval-event:{req.request_id}:{event_type}",
            )
            return result.accepted
        except Exception as exc:
            logger.error("Could not persist %s event for request %s: %s", event_type, req.request_id, exc)
            return False

    def load(self) -> None:
        if not self.approvals_path.exists():
            self._requests = {}
            return
        try:
            with open(self.approvals_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._requests = {
                req_id: PermissionRequest.from_dict(d)
                for req_id, d in data.get("requests", {}).items()
            }
        except Exception as e:
            logger.warning("Failed to load approvals file %s: %s", self.approvals_path, e)
            self._requests = {}

    def save(self) -> None:
        self.approvals_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "requests": {req_id: req.to_dict() for req_id, req in self._requests.items()},
        }
        tmp_path = self.approvals_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.approvals_path)

    def create_request(
        self,
        task_id: str,
        risk: str,
        operation: str,
        target: str,
        scope: str,
        reason: str,
        request_id: Optional[str] = None,
        patch_sha256: Optional[str] = None,
        baseline_commit: Optional[str] = None,
        verification_sha256: Optional[str] = None,
    ) -> PermissionRequest:
        if not request_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            request_id = f"req-{task_id}-{ts}"

        # If a pending request for the exact operation and scope already exists, return it
        for existing in self._requests.values():
            if (
                existing.task_id == task_id
                and existing.operation == operation
                and existing.target == target
                and existing.scope == scope
                and existing.status == ApprovalStatus.PENDING
                and existing.patch_sha256 == patch_sha256
                and existing.baseline_commit == baseline_commit
                and existing.verification_sha256 == verification_sha256
            ):
                return existing

        req = PermissionRequest(
            request_id=request_id,
            task_id=task_id,
            risk=risk,
            operation=operation,
            target=target,
            scope=scope,
            reason=reason,
            patch_sha256=patch_sha256,
            baseline_commit=baseline_commit,
            verification_sha256=verification_sha256,
        )
        self._requests[request_id] = req
        self.save()
        if not self._record_lifecycle_event(req, "PERMISSION_REQUEST_CREATED"):
            raise RuntimeError("Permission request was saved, but its coordinator lifecycle event could not be persisted.")
        return req

    def get_request(self, request_id: str) -> Optional[PermissionRequest]:
        return self._requests.get(request_id)

    def get_pending_request_for_task(self, task_id: str) -> Optional[PermissionRequest]:
        for req in self._requests.values():
            if req.task_id == task_id and req.status == ApprovalStatus.PENDING:
                return req
        return None

    def list_requests(self, status: Optional[ApprovalStatus] = None) -> List[PermissionRequest]:
        reqs = list(self._requests.values())
        if status is not None:
            reqs = [r for r in reqs if r.status == status]
        reqs.sort(key=lambda r: r.requested_at)
        return reqs

    def approve(
        self,
        request_id: str,
        scope: Optional[str] = None,
        reason: Optional[str] = None,
        decision_by: str = "operator",
        sync_store: bool = True,
    ) -> PermissionRequest:
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Permission request '{request_id}' not found.")
        if req.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot approve request '{request_id}' with status {req.status.value}.")

        if req.operation == "integration" and scope is not None and scope != req.scope:
            raise ValueError("Integration approval scope is immutable and must match the reviewed patch and baseline.")

        # Approval must explicitly specify or confirm the reviewed scope
        approved_scope = scope if scope is not None else req.scope
        if scope is not None and scope != req.scope:
            # Operator restricted or changed scope
            approved_scope = scope

        req.status = ApprovalStatus.APPROVED
        req.decision_at = _utc_now_iso()
        req.decision_by = decision_by
        req.decision_scope = approved_scope
        req.decision_reason = reason or "Authorized by operator."
        self.save()
        if sync_store:
            synced = self.sync_approval_to_store(req)
            if req.operation == "integration" and not synced:
                raise RuntimeError("Approval was recorded, but coordinator state could not confirm it; integration remains blocked.")
            if not synced and not self._record_lifecycle_event(req, "PERMISSION_APPROVED"):
                raise RuntimeError("Approval was recorded, but its coordinator lifecycle event could not be persisted.")
        return req

    def deny(
        self,
        request_id: str,
        reason: Optional[str] = None,
        decision_by: str = "operator",
    ) -> PermissionRequest:
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Permission request '{request_id}' not found.")
        if req.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot deny request '{request_id}' with status {req.status.value}.")

        req.status = ApprovalStatus.DENIED
        req.decision_at = _utc_now_iso()
        req.decision_by = decision_by
        req.decision_reason = reason or "Denied by operator."
        self.save()
        if not self._record_lifecycle_event(req, "PERMISSION_DENIED"):
            raise RuntimeError("Denial was recorded, but its coordinator lifecycle event could not be persisted.")
        return req

    def skip(
        self,
        request_id: str,
        reason: Optional[str] = None,
        decision_by: str = "operator",
    ) -> PermissionRequest:
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Permission request '{request_id}' not found.")
        if req.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot skip request '{request_id}' with status {req.status.value}.")

        req.status = ApprovalStatus.SKIPPED
        req.decision_at = _utc_now_iso()
        req.decision_by = decision_by
        req.decision_reason = reason or "Skipped by operator for current session."
        self.save()
        if not self._record_lifecycle_event(req, "PERMISSION_SKIPPED"):
            raise RuntimeError("Skip decision was recorded, but its coordinator lifecycle event could not be persisted.")
        return req

    def is_operation_approved(
        self,
        task_id: str,
        operation: str,
        target: str,
        scope: str,
        patch_sha256: Optional[str] = None,
        baseline_commit: Optional[str] = None,
        verification_sha256: Optional[str] = None,
    ) -> bool:
        """
        Check if an exact reviewed operation and scope has been approved.
        Any changed target or broadened scope returns False and requires fresh approval.
        """
        for req in self._requests.values():
            if (
                req.task_id == task_id
                and req.operation == operation
                and req.target == target
                and req.status == ApprovalStatus.APPROVED
                and (patch_sha256 is None or req.patch_sha256 == patch_sha256)
                and (baseline_commit is None or req.baseline_commit == baseline_commit)
                and (verification_sha256 is None or req.verification_sha256 == verification_sha256)
            ):
                # Verify exact scope match
                if req.decision_scope == scope:
                    return True
        return False

    def format_request_prompt(self, req: PermissionRequest) -> str:
        lines = [
            "=" * 78,
            "MAINTENANCE TERMINAL APPROVAL REQUIRED",
            "=" * 78,
            f"Request ID : {req.request_id}",
            f"Task ID    : {req.task_id}",
            f"Risk       : {req.risk}",
            f"Operation  : {req.operation}",
            f"Target     : {req.target}",
            f"Scope      : {req.scope}",
            f"Patch SHA  : {req.patch_sha256 or 'N/A'}",
            f"Baseline   : {req.baseline_commit or 'N/A'}",
            f"Verify SHA : {req.verification_sha256 or 'N/A'}",
            f"Reason     : {req.reason}",
            f"Requested  : {req.requested_at}",
            "-" * 78,
            "Available decisions:",
            "  [A]pprove : Authorize this exact operation and scope",
            "  [D]eny    : Reject this request and keep task blocked",
            "  [I]nspect : Inspect target details, worktree status, and git diff",
            "  [S]kip    : Skip this task for the current session without failing",
            "  [E]xit    : Safely exit terminal approval session",
            "=" * 78,
        ]
        return "\n".join(lines)

    def inspect_request(self, request_id: str) -> dict:
        req = self._requests.get(request_id)
        if not req:
            return {"error": f"Request '{request_id}' not found."}

        info: dict = {
            "request": req.to_dict(),
            "worktree_exists": False,
            "git_status": "",
            "target_files_in_main": {},
        }

        wt_path = self.repo_root / ".maya" / "worktrees" / req.task_id
        if wt_path.exists():
            info["worktree_exists"] = True
            info["worktree_path"] = str(wt_path)
            try:
                proc = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=str(wt_path),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                info["git_status"] = proc.stdout.strip()
            except Exception as e:
                info["git_status"] = f"Error querying git status: {e}"

        return info

    def interactive_session(
        self,
        input_fn: Optional[Callable[[str], str]] = None,
        output_fn: Optional[Callable[[str], None]] = None,
    ) -> Tuple[int, str]:
        """
        Run interactive approval session for all pending requests.
        Returns (decisions_count, exit_reason).
        """
        _input = input_fn or input
        _output = output_fn or print

        pending = self.list_requests(status=ApprovalStatus.PENDING)
        if not pending:
            _output("\nNo pending permission requests awaiting operator approval.")
            return 0, "NO_PENDING"

        decisions_made = 0
        for req in pending:
            while True:
                _output("\n" + self.format_request_prompt(req))
                try:
                    choice = _input("Enter decision [A/D/I/S/E]: ").strip().upper()
                except (EOFError, KeyboardInterrupt):
                    _output("\nSession safely interrupted by operator.")
                    return decisions_made, "INTERRUPTED"

                if choice in ("A", "APPROVE"):
                    self.approve(req.request_id, decision_by="operator")
                    _output(f"[APPROVED] Request {req.request_id} authorized.")
                    decisions_made += 1
                    break
                elif choice in ("D", "DENY"):
                    self.deny(req.request_id, decision_by="operator")
                    _output(f"[DENIED] Request {req.request_id} rejected.")
                    decisions_made += 1
                    break
                elif choice in ("I", "INSPECT"):
                    inspection = self.inspect_request(req.request_id)
                    _output("\n--- Diagnostic Inspection ---")
                    _output(json.dumps(inspection, indent=2))
                    _output("-----------------------------")
                    continue
                elif choice in ("S", "SKIP"):
                    self.skip(req.request_id, decision_by="operator")
                    _output(f"[SKIPPED] Request {req.request_id} skipped for now.")
                    decisions_made += 1
                    break
                elif choice in ("E", "EXIT", "Q", "QUIT"):
                    _output("\nExiting terminal approval session.")
                    return decisions_made, "OPERATOR_EXIT"
                else:
                    _output("Invalid decision choice. Enter A, D, I, S, or E.")

        return decisions_made, "ALL_PROCESSED"


def generate_task_report(
    task_id: str,
    title: str,
    priority: str,
    risk: str,
    success: bool,
    final_status: str,
    duration_sec: float,
    worktree_path: Optional[str],
    quality_report: Optional[dict],
    agent_result: Optional[dict],
    reports_dir: Path,
) -> Path:
    """Save an isolated markdown report for a completed task."""
    reports_dir = Path(reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_file = reports_dir / f"task-{task_id}-{ts}.md"

    lines = [
        f"# MayaVE Task Execution Report: {task_id}",
        "",
        f"- **Task ID**: {task_id}",
        f"- **Title**: {title}",
        f"- **Priority**: {priority}",
        f"- **Risk**: {risk}",
        f"- **Success**: {success}",
        f"- **Final Status**: {final_status}",
        f"- **Duration**: {duration_sec:.2f}s",
        f"- **Worktree**: {worktree_path or 'N/A'}",
        f"- **Generated**: {_utc_now_iso()}",
        "",
        "## Quality Gate Summary",
    ]

    if quality_report and quality_report.get("results"):
        lines.append("| Gate Name | Result | Details |")
        lines.append("|---|---|---|")
        for g in quality_report["results"]:
            status_str = "PASS" if g.get("passed") else "FAIL"
            lines.append(f"| `{g.get('gate_name')}` | **{status_str}** | {g.get('message', '')} |")
    else:
        lines.append("No automated quality gate results recorded.")

    lines.append("")
    lines.append("## Agent Subprocess Summary")
    if agent_result:
        lines.append(f"- Exit Code: `{agent_result.get('exit_code')}`")
        lines.append(f"- Duration: `{agent_result.get('duration_sec')}s`")
        lines.append(f"- Timed Out: `{agent_result.get('timed_out')}`")
        lines.append(f"- Permission Denied: `{agent_result.get('permission_denied')}`")
    else:
        lines.append("No agent subprocess results recorded.")

    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_file


def generate_session_summary(
    session_id: str,
    scheduler_summary: dict,
    reports_dir: Path,
) -> Tuple[Path, Path]:
    """Save durable session summary in JSON and Markdown formats."""
    reports_dir = Path(reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_file = reports_dir / f"session-{session_id}.json"
    md_file = reports_dir / f"session-{session_id}.md"

    # 1. JSON output
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(scheduler_summary, f, indent=2)

    # 2. Markdown output
    lines = [
        f"# MayaVE Maintenance Worker Session Summary: {session_id}",
        "",
        f"- **Session ID**: {session_id}",
        f"- **Completed Tasks**: {scheduler_summary.get('completed_tasks', 0)}",
        f"- **Verified Tasks**: {scheduler_summary.get('verified_tasks', 0)}",
        f"- **Failed Tasks**: {scheduler_summary.get('failed_tasks', 0)}",
        f"- **Blocked Tasks**: {scheduler_summary.get('blocked_tasks', 0)}",
        f"- **Peak Concurrency**: {scheduler_summary.get('peak_concurrency', 0)}",
        f"- **Total Duration**: {scheduler_summary.get('duration_sec', 0.0):.2f}s",
    ]

    if scheduler_summary.get("requested_duration_sec"):
        req_dur = float(scheduler_summary["requested_duration_sec"])
        lines.append(f"- **Requested Duration**: {req_dur:.1f}s ({req_dur/3600:.1f}h)")
    if scheduler_summary.get("session_deadline"):
        lines.append(f"- **Session Deadline**: {scheduler_summary['session_deadline']}")
    if scheduler_summary.get("remaining_duration_sec") is not None:
        lines.append(f"- **Remaining Duration**: {scheduler_summary['remaining_duration_sec']:.1f}s")
    if scheduler_summary.get("halt_reason"):
        lines.append(f"- **Shutdown Reason**: {scheduler_summary['halt_reason']}")

    lines.extend([
        f"- **Generated**: {_utc_now_iso()}",
        "",
        "## Executed Task Summaries",
    ])

    task_summaries = scheduler_summary.get("task_summaries", {})
    if task_summaries:
        lines.append("| Task ID | Success | Final Status | Duration | Message |")
        lines.append("|---|---|---|---|---|")
        for tid, tsum in task_summaries.items():
            lines.append(
                f"| `{tid}` | {tsum.get('success')} | {tsum.get('final_status')} | "
                f"{tsum.get('duration_sec', 0.0):.2f}s | {tsum.get('message', '')} |"
            )
    else:
        lines.append("No tasks were executed during this session.")

    md_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_file, md_file
