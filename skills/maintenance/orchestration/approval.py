"""
skills/maintenance/orchestration/approval.py
Secure approval management for AutoPhage 3.0.
Binds permissions to task, operation, and specific target artifacts.
"""
from __future__ import annotations
import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

@dataclass
class ApprovalToken:
    """A secure binding between a task and a specific permitted operation."""
    token_id: str
    task_id: str
    operation: str
    targets: Set[str]
    parameters_hash: str
    timestamp: float
    authorized_by: str = "operator"

    def __post_init__(self):
        # Normalize targets to prevent evasion via path variations
        self.targets = {t.replace("\\", "/").strip("/") for t in self.targets}

    def matches(self, operation: str, targets: Set[str], params_hash: str) -> bool:
        if self.operation != operation:
            return False

        normalized_targets = {t.replace("\\", "/").strip("/") for t in targets}
        if not normalized_targets.issubset(self.targets):
            return False

        if self.parameters_hash != params_hash:
            return False

        return True

class ApprovalManager:
    """
    Handles the lifecycle of operation approvals.
    Replaces boolean 'has_approval' with bound evidence checks.
    """
    def __init__(self, approvals_path: Path):
        self.approvals_path = approvals_path
        self.approvals_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_approvals(self) -> dict[str, ApprovalToken]:
        if not self.approvals_path.exists():
            return {}
        try:
            with open(self.approvals.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    tid: ApprovalToken(
                        token_id=tok["token_id"],
                        task_id=tid,
                        operation=tok["operation"],
                        targets=set(tok["targets"]),
                        parameters_hash=tok["parameters_hash"],
                        timestamp=tok["timestamp"],
                        authorized_by=tok.get("authorized_by", "operator")
                    )
                    for tid, tok in data.items()
                }
        except Exception as e:
            logger.error(f"Failed to load approvals: {e}")
            return {}

    def _save_approvals(self, approvals: dict[str, ApprovalToken]) -> None:
        data = {
            tid: {
                "token_id": t.token_id,
                "operation": t.operation,
                "targets": list(t.targets),
                "parameters_hash": t.parameters_hash,
                "timestamp": t.timestamp,
                "authorized_by": t.authorized_by,
            }
            for tid, t in approvals.items()
        }
        tmp = self.approvals_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.approvals_path)

    def grant_approval(self, task_id: str, operation: str, targets: Set[str], params: dict[str, Any]) -> str:
        """Grants a bound approval for a specific operation."""
        param_str = json.dumps(params, sort_keys=True)
        p_hash = hashlib.sha256(param_str.encode()).hexdigest()

        token_id = f"tok_{int(time.time())}_{hashlib.md5(task_id.encode()).hexdigest()[:6]}"
        token = ApprovalToken(
            token_id=token_id,
            task_id=task_id,
            operation=operation,
            targets=targets,
            parameters_hash=p_hash,
            timestamp=time.time()
        )

        approvals = self._load_approvals()
        approvals[task_id] = token
        self._save_approvals(approvals)
        return token_id

    def verify_approval(self, task_id: str, operation: str, targets: Set[str], params: dict[str, Any]) -> bool:
        """Re-validates approval immediately before execution."""
        approvals = self._load_approvals()
        token = approvals.get(task_id)
        if not token:
            return False

        param_str = json.dumps(params, sort_keys=True)
        p_hash = hashlib.sha256(param_str.encode()).hexdigest()

        return token.matches(operation, targets, p_hash)

    def consume_approval(self, task_id: str) -> None:
        """One-time consumption to prevent replay attacks."""
        approvals = self._load_approvals()
        if task_id in approvals:
            del approvals[task_id]
            self._save_approvals(approvals)
