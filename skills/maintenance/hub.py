"""Coordinator-owned task transitions, lifecycle events, and integration approval."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .ledger import LedgerValidationError, TaskLedger
    from .models import (
        CheckpointRecord,
        EvidenceRecord,
        PlanProposal,
        TaskDefinition,
        TaskStatus,
        utc_now_iso,
    )
    from .planner import compute_proposal_hash
    from .store import StateStore
    from .task_contract import TaskContractValidator
except ImportError:
    from ledger import LedgerValidationError, TaskLedger
    from models import (
        CheckpointRecord,
        EvidenceRecord,
        PlanProposal,
        TaskDefinition,
        TaskStatus,
        utc_now_iso,
    )
    from planner import compute_proposal_hash
    from store import StateStore
    from task_contract import TaskContractValidator


EVENT_SCHEMA_VERSION = 1
MAX_LIFECYCLE_EVENTS = 5000

ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_APPROVAL, TaskStatus.BLOCKED},
    TaskStatus.IN_PROGRESS: {
        TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.FAILED,
        TaskStatus.NEEDS_APPROVAL, TaskStatus.RECOVERY_REQUIRED, TaskStatus.BLOCKED,
    },
    TaskStatus.VERIFYING: {
        TaskStatus.READY_TO_INTEGRATE, TaskStatus.FAILED, TaskStatus.NEEDS_APPROVAL,
        TaskStatus.RECOVERY_REQUIRED,
    },
    TaskStatus.READY_TO_INTEGRATE: {
        TaskStatus.VERIFYING, TaskStatus.INTEGRATING, TaskStatus.RECOVERY_REQUIRED,
    },
    TaskStatus.INTEGRATING: {
        TaskStatus.VERIFIED, TaskStatus.RECOVERY_REQUIRED, TaskStatus.FAILED,
    },
    TaskStatus.FAILED: {
        TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.NEEDS_APPROVAL,
    },
    TaskStatus.NEEDS_APPROVAL: {TaskStatus.PENDING, TaskStatus.RECOVERY_REQUIRED},
    TaskStatus.RECOVERY_REQUIRED: {
        TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_APPROVAL,
    },
    TaskStatus.BLOCKED: {TaskStatus.PENDING, TaskStatus.NEEDS_APPROVAL},
    TaskStatus.VERIFIED: {TaskStatus.RECOVERY_REQUIRED, TaskStatus.NEEDS_APPROVAL},
}

ACTOR_TRANSITIONS: dict[str, set[TaskStatus]] = {
    "orchestrator": {
        TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.READY_TO_INTEGRATE,
        TaskStatus.FAILED, TaskStatus.NEEDS_APPROVAL, TaskStatus.RECOVERY_REQUIRED,
        TaskStatus.BLOCKED,
    },
    "recovery": {TaskStatus.RECOVERY_REQUIRED, TaskStatus.NEEDS_APPROVAL},
    "approval": {TaskStatus.PENDING},
    "integration": {
        TaskStatus.INTEGRATING, TaskStatus.VERIFIED, TaskStatus.RECOVERY_REQUIRED,
        TaskStatus.FAILED,
    },
    "coordinator": {
        TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED,
        TaskStatus.NEEDS_APPROVAL, TaskStatus.RECOVERY_REQUIRED, TaskStatus.READY_TO_INTEGRATE,
        TaskStatus.VERIFYING,
    },
}


@dataclass(frozen=True)
class TransitionResult:
    accepted: bool
    task_id: str
    previous_status: str | None
    resulting_status: str | None
    event_id: str | None
    error_code: str | None = None
    message: str = ""
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "task_id": self.task_id,
            "previous_status": self.previous_status,
            "resulting_status": self.resulting_status,
            "event_id": self.event_id,
            "error_code": self.error_code,
            "message": self.message,
            "duplicate": self.duplicate,
        }


class CoordinatorAPI:
    """The persistence boundary for protected task status and lifecycle updates."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    @staticmethod
    def _event_id(idempotency_key: str | None) -> str:
        if idempotency_key:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mayave-autophage:{idempotency_key}"))
        return str(uuid.uuid4())

    @staticmethod
    def _find_event(ledger: TaskLedger, key: str | None) -> dict | None:
        if not key:
            return None
        for event in reversed(ledger.events):
            if event.get("idempotency_key") == key:
                return event
        return None

    @staticmethod
    def _event(
        task_id: str,
        event_type: str,
        previous: str | None,
        resulting: str | None,
        actor: str,
        outcome: str,
        idempotency_key: str | None,
        details: dict | None = None,
        error_code: str | None = None,
        message: str | None = None,
    ) -> dict:
        data = dict(details or {})
        return {
            "event_id": CoordinatorAPI._event_id(idempotency_key),
            "schema_version": EVENT_SCHEMA_VERSION,
            "task_id": task_id,
            "event_type": event_type,
            "previous_status": previous,
            "resulting_status": resulting,
            "timestamp": utc_now_iso(),
            "actor": actor,
            "baseline_sha256": data.pop("baseline_sha256", None),
            "baseline_commit": data.pop("baseline_commit", None),
            "artifact_sha256": data.pop("artifact_sha256", None),
            "correlation_id": data.pop("correlation_id", None),
            "idempotency_key": idempotency_key,
            "outcome": outcome,
            "error_code": error_code,
            "message": message,
            "details": data,
        }

    @staticmethod
    def _copy_ledger_into(target: TaskLedger, source: TaskLedger) -> None:
        """Publish persisted state back while preserving caller-held TaskDefinition references."""
        for task_id, persisted in source.items():
            current = target.get(task_id)
            if current is None:
                target._tasks[task_id] = persisted
            else:
                for name in persisted.__dataclass_fields__:
                    setattr(current, name, copy.deepcopy(getattr(persisted, name)))
        target.events = copy.deepcopy(source.events)
        target._status_snapshot = dict(source._status_snapshot)
        if hasattr(source, "budget"):
            target.budget = copy.deepcopy(source.budget)

    def _persist_candidate(
        self,
        current: TaskLedger,
        candidate: TaskLedger,
        lock_held: bool,
        changed_task_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        if lock_held:
            try:
                self.store.save_ledger(candidate)
            except Exception as exc:
                raise LedgerValidationError("Could not atomically persist coordinator state and lifecycle event.") from exc
        else:
            with self.store.acquire_lock():
                latest = self.store.load_ledger()
                quarantined = latest.quarantine_legacy_unintegrated_verified()
                if changed_task_id and changed_task_id in latest and changed_task_id in current:
                    expected = current._status_snapshot.get(changed_task_id)
                    actual = latest.get_task(changed_task_id).status
                    if expected is not None and actual != expected and changed_task_id in quarantined:
                        candidate_task = candidate.get_task(changed_task_id)
                        candidate_task.status = actual
                        candidate_task.blocking_reason = latest.get_task(changed_task_id).blocking_reason
                        candidate._commit_status(changed_task_id, actual)
                    quarantine_event = next((e for e in latest.events if e.get("task_id") == changed_task_id
                                             and e.get("event_type") == "LEGACY_VERIFICATION_QUARANTINED"), None)
                    if quarantine_event:
                        candidate.events = [e for e in candidate.events if e.get("event_id") != quarantine_event.get("event_id")]
                        candidate.events.append(quarantine_event)
                    elif expected is not None and actual != expected:
                        if idempotency_key and self._find_event(latest, idempotency_key):
                            raise LedgerValidationError("Transition idempotency key was already persisted; reload current task state.")
                        raise LedgerValidationError(
                            f"Task '{changed_task_id}' changed concurrently ({expected.value} -> {actual.value}); reload and review."
                        )
                # Merge other workers' task updates and append only unseen events.
                for task_id, latest_task in latest.items():
                    if task_id != changed_task_id:
                        candidate._tasks[task_id] = latest_task
                        candidate._status_snapshot[task_id] = latest._status_snapshot.get(task_id, latest_task.status)
                merged_events = list(latest.events)
                seen_ids = {event.get("event_id") for event in merged_events}
                merged_events.extend(event for event in candidate.events if event.get("event_id") not in seen_ids)
                candidate.events = merged_events[-MAX_LIFECYCLE_EVENTS:]
                if hasattr(candidate, "budget") and hasattr(latest, "budget"):
                    candidate.budget.cloud_tasks_consumed = max(candidate.budget.cloud_tasks_consumed, latest.budget.cloud_tasks_consumed)
                    candidate.budget.total_input_tokens = max(candidate.budget.total_input_tokens, latest.budget.total_input_tokens)
                    candidate.budget.total_output_tokens = max(candidate.budget.total_output_tokens, latest.budget.total_output_tokens)
                    candidate.budget.estimated_cost_usd = max(candidate.budget.estimated_cost_usd, latest.budget.estimated_cost_usd)
                if changed_task_id is None:
                    for task_id, latest_task in latest.items():
                        candidate._tasks[task_id] = latest_task
                        candidate._status_snapshot[task_id] = latest._status_snapshot.get(task_id, latest_task.status)
                try:
                    self.store.save_ledger(candidate)
                except Exception:
                    raise LedgerValidationError(
                        "Could not atomically persist coordinator state and lifecycle event."
                    ) from None
        self._copy_ledger_into(current, candidate)

    def record_event(
        self,
        ledger: TaskLedger,
        task_id: str,
        event_type: str,
        actor: str,
        *,
        details: dict | None = None,
        idempotency_key: str | None = None,
        lock_held: bool = False,
    ) -> TransitionResult:
        """Persist a structured non-transition lifecycle event in the same ledger file."""
        try:
            ledger.validate_status_integrity()
            previous = ledger.get_task(task_id).status.value
            existing = self._find_event(ledger, idempotency_key)
            if existing:
                same = (
                    existing.get("task_id") == task_id
                    and existing.get("event_type") == event_type
                    and existing.get("actor") == actor
                )
                if same:
                    if not lock_held:
                        persisted = self.store.load_ledger()
                        if self._find_event(persisted, idempotency_key):
                            self._copy_ledger_into(ledger, persisted)
                if same and not lock_held:
                    persisted = self.store.load_ledger()
                    if self._find_event(persisted, idempotency_key):
                        self._copy_ledger_into(ledger, persisted)
                return TransitionResult(
                    same, task_id, previous, previous, existing.get("event_id"),
                    None if same else "IDEMPOTENCY_CONFLICT",
                    "Duplicate event already recorded." if same else "Idempotency key was used for a different event.",
                    duplicate=True,
                )
            if idempotency_key is not None and lock_held:
                persisted = self.store.load_ledger()
                committed = self._find_event(persisted, idempotency_key)
                if committed:
                    same = (
                        committed.get("task_id") == task_id
                        and committed.get("event_type") == event_type
                        and committed.get("actor") == actor
                    )
                    return TransitionResult(
                        same, task_id, previous, previous, committed.get("event_id"),
                        None if same else "IDEMPOTENCY_CONFLICT",
                        "Duplicate event already recorded." if same else "Idempotency key was used for a different event.",
                        duplicate=True,
                    )
            candidate = TaskLedger.from_dict(ledger.to_dict())
            event = self._event(task_id, event_type, previous, previous, actor, "accepted", idempotency_key, details)
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held, idempotency_key=idempotency_key)
            return TransitionResult(True, task_id, previous, previous, event["event_id"], message="Event recorded.")
        except Exception as exc:
            return TransitionResult(False, task_id, None, None, None, "PERSISTENCE_OR_STATE_ERROR", str(exc))

    def transition_task(
        self,
        ledger: TaskLedger,
        task_id: str,
        new_status: TaskStatus | str,
        actor: str,
        *,
        authorized: bool = False,
        reason: str | None = None,
        failure_reason: str | None = None,
        blocking_reason: str | None = None,
        checkpoint_stage: str | None = None,
        checkpoint_details: dict | None = None,
        event_type: str = "TASK_TRANSITION",
        idempotency_key: str | None = None,
        lock_held: bool = False,
        require_integration_approval: bool = False,
    ) -> TransitionResult:
        """Validate, record, and atomically persist one authoritative task transition."""
        try:
            ledger.validate_status_integrity()
            task = ledger.get_task(task_id)
            original_state = self.store.load_ledger()
            persisted_task = original_state.get(task_id)
            if persisted_task is not None and persisted_task.status != task.status:
                self._copy_ledger_into(ledger, original_state)
                return self._reject(
                    ledger, task_id, actor, persisted_task.status, "STALE_TASK_STATE",
                    "In-memory task state is stale; reload before requesting a transition.",
                    idempotency_key, lock_held,
                )
            old = task.status
            try:
                target = new_status if isinstance(new_status, TaskStatus) else TaskStatus(new_status)
            except (ValueError, TypeError):
                return self._reject(ledger, task_id, actor, old, "UNKNOWN_STATE", f"Unknown task state: {new_status!r}.", idempotency_key, lock_held)

            existing = self._find_event(ledger, idempotency_key)
            if existing:
                same = (
                    existing.get("task_id") == task_id
                    and existing.get("event_type") == event_type
                    and existing.get("resulting_status") == target.value
                )
                still_current = same and old == target
                return TransitionResult(
                    still_current, task_id, existing.get("previous_status"), existing.get("resulting_status"),
                    existing.get("event_id"), None if still_current else ("ALREADY_PROCESSED" if same else "IDEMPOTENCY_CONFLICT"),
                    "Duplicate transition already recorded." if still_current else (
                        "This transition was already processed; inspect current task state." if same
                        else "Idempotency key was used for a different transition."
                    ),
                    duplicate=True,
                )
            if idempotency_key and not lock_held:
                persisted = self.store.load_ledger()
                committed = self._find_event(persisted, idempotency_key)
                if committed:
                    same = (
                        committed.get("task_id") == task_id
                        and committed.get("event_type") == event_type
                        and committed.get("resulting_status") == target.value
                    )
                    current_status = persisted.get_task(task_id).status
                    succeeded = same and current_status == target
                    if succeeded:
                        self._copy_ledger_into(ledger, persisted)
                    return TransitionResult(
                        succeeded, task_id, committed.get("previous_status"), committed.get("resulting_status"),
                        committed.get("event_id"), None if succeeded else ("ALREADY_PROCESSED" if same else "IDEMPOTENCY_CONFLICT"),
                        "Duplicate transition already recorded." if succeeded else "Idempotency key was already used; inspect current task state.",
                        duplicate=True,
                    )
            if idempotency_key and lock_held:
                persisted = self.store.load_ledger()
                committed = self._find_event(persisted, idempotency_key)
                if committed:
                    same = (
                        committed.get("task_id") == task_id
                        and committed.get("event_type") == event_type
                        and committed.get("resulting_status") == target.value
                    )
                    current_status = persisted.get_task(task_id).status
                    succeeded = same and current_status == target
                    if succeeded:
                        self._copy_ledger_into(ledger, persisted)
                    return TransitionResult(
                        succeeded, task_id, committed.get("previous_status"), committed.get("resulting_status"),
                        committed.get("event_id"), None if succeeded else ("ALREADY_PROCESSED" if same else "IDEMPOTENCY_CONFLICT"),
                        "Duplicate transition already recorded." if succeeded else "Idempotency key was already used; inspect current task state.",
                        duplicate=True,
                    )
            actor_targets = ACTOR_TRANSITIONS.get(actor, set())
            if target not in actor_targets:
                return self._reject(ledger, task_id, actor, old, "ACTOR_NOT_PERMITTED", f"Actor '{actor}' cannot request {target.value}.", idempotency_key, lock_held, target.value)
            if target not in ALLOWED_TRANSITIONS.get(old, set()):
                return self._reject(ledger, task_id, actor, old, "INVALID_TRANSITION", f"Transition {old.value} -> {target.value} is not permitted.", idempotency_key, lock_held, target.value)
            if target in (TaskStatus.IN_PROGRESS, TaskStatus.INTEGRATING) and not authorized:
                return self._reject(ledger, task_id, actor, old, "AUTHORIZATION_REQUIRED", f"Transition to {target.value} requires explicit operator authorization.", idempotency_key, lock_held, target.value)
            if target == TaskStatus.VERIFIED:
                details = checkpoint_details or {}
                if actor != "integration" or not details.get("postcheck_passed"):
                    return self._reject(ledger, task_id, actor, old, "POSTCHECK_REQUIRED", "VERIFIED is reserved for successful integration with passing post-integration checks.", idempotency_key, lock_held, target.value)
            if target == TaskStatus.INTEGRATING and require_integration_approval:
                expected = (checkpoint_details or {}).get("integration_approval") or {}
                saved = task.integration_approval or {}
                if not authorized or any(saved.get(k) != expected.get(k) for k in (
                    "request_id", "patch_sha256", "baseline_commit", "verification_sha256"
                )) or saved.get("status") != "APPROVED":
                    return self._reject(ledger, task_id, actor, old, "INTEGRATION_APPROVAL_INVALID", "No valid approval matches the task, patch, baseline, and verification evidence.", idempotency_key, lock_held, target.value)

            candidate = TaskLedger.from_dict(ledger.to_dict())
            candidate_task = candidate.get_task(task_id)
            candidate_task.status = target
            candidate_task.updated_at = utc_now_iso()
            if failure_reason is not None:
                candidate_task.failure_reason = failure_reason
            if blocking_reason is not None:
                candidate_task.blocking_reason = blocking_reason
            if target in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.READY_TO_INTEGRATE, TaskStatus.VERIFIED):
                candidate_task.blocking_reason = None
            if checkpoint_stage:
                timestamp = utc_now_iso()
                candidate_task.checkpoint = CheckpointRecord(timestamp, checkpoint_stage, checkpoint_details or {})
                candidate_task.history.append(EvidenceRecord(
                    timestamp, checkpoint_stage, json.dumps(checkpoint_details or {}, sort_keys=True),
                    target not in (TaskStatus.FAILED, TaskStatus.RECOVERY_REQUIRED, TaskStatus.NEEDS_APPROVAL),
                ))
            candidate._commit_status(task_id, target)
            event = self._event(
                task_id, event_type, old.value, target.value, actor, "accepted", idempotency_key,
                {**(checkpoint_details or {}), "reason": reason},
            )
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held, changed_task_id=task_id, idempotency_key=idempotency_key)
            return TransitionResult(True, task_id, old.value, target.value, event["event_id"], message="Transition persisted.")
        except Exception as exc:
            return TransitionResult(False, task_id, None, None, None, "PERSISTENCE_OR_STATE_ERROR", str(exc))

    def _reject(
        self,
        ledger: TaskLedger,
        task_id: str,
        actor: str,
        old: TaskStatus,
        error_code: str,
        message: str,
        key: str | None,
        lock_held: bool,
        target: str | None = None,
    ) -> TransitionResult:
        try:
            candidate = TaskLedger.from_dict(ledger.to_dict())
            event = self._event(task_id, "TRANSITION_REJECTED", old.value, old.value, actor, "rejected", key,
                                {"requested_status": target}, error_code, message)
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held, changed_task_id=task_id)
            return TransitionResult(False, task_id, old.value, old.value, event["event_id"], error_code, message)
        except Exception as exc:
            return TransitionResult(False, task_id, old.value, old.value, None, "PERSISTENCE_FAILED", f"{message} Rejection event could not be persisted: {exc}")

    def approve_integration(
        self,
        task_id: str,
        request_id: str,
        patch_sha256: str,
        baseline_commit: str,
        verification_sha256: str,
        approved_by: str,
    ) -> TransitionResult:
        """Persist a human-approved integration artifact tuple in the authoritative ledger."""
        try:
            with self.store.acquire_lock():
                ledger = self.store.load_ledger()
                task = ledger.get_task(task_id)
                if task.status != TaskStatus.READY_TO_INTEGRATE:
                    return self._reject(ledger, task_id, "approval", task.status, "TASK_NOT_READY", "Task is not ready for integration approval.", f"approval:{request_id}", True)
                cp = task.checkpoint.details if task.checkpoint else {}
                if any(cp.get(k) != v for k, v in (
                    ("artifact_sha256", patch_sha256),
                    ("baseline_commit", baseline_commit),
                    ("verification_sha256", verification_sha256),
                )):
                    return self._reject(ledger, task_id, "approval", task.status, "APPROVAL_ARTIFACT_MISMATCH", "Approval does not match the current verified artifact tuple.", f"approval:{request_id}", True)
                candidate = TaskLedger.from_dict(ledger.to_dict())
                candidate_task = candidate.get_task(task_id)
                candidate_task.integration_approval = {
                    "status": "APPROVED",
                    "request_id": request_id,
                    "patch_sha256": patch_sha256,
                    "baseline_commit": baseline_commit,
                    "verification_sha256": verification_sha256,
                    "approved_by": approved_by,
                    "approved_at": utc_now_iso(),
                }
                event = self._event(task_id, "INTEGRATION_APPROVED", task.status.value, task.status.value, "approval", "accepted",
                                    f"approval:{request_id}", {
                                        "artifact_sha256": patch_sha256,
                                        "baseline_sha256": baseline_commit,
                                        "verification_sha256": verification_sha256,
                                        "request_id": request_id,
                                        "approved_by": approved_by,
                                    })
                candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
                self.store.save_ledger(candidate)
            return TransitionResult(True, task_id, task.status.value, task.status.value, event["event_id"], message="Exact integration approval persisted.")
        except Exception as exc:
            return TransitionResult(False, task_id, None, None, None, "PERSISTENCE_OR_STATE_ERROR", str(exc))

    def invalidate_integration_approval(
        self,
        ledger: TaskLedger,
        task_id: str,
        reason: str,
        *,
        lock_held: bool = False,
    ) -> TransitionResult:
        try:
            ledger.validate_status_integrity()
            candidate = TaskLedger.from_dict(ledger.to_dict())
            task = candidate.get_task(task_id)
            previous = task.status.value
            approval = task.integration_approval
            task.integration_approval = None
            event = self._event(task_id, "APPROVAL_INVALIDATED", previous, previous, "coordinator", "accepted", None,
                                {"reason": reason, "previous_approval": approval})
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held, changed_task_id=task_id)
            return TransitionResult(True, task_id, previous, previous, event["event_id"], message="Approval invalidated.")
        except Exception as exc:
            return TransitionResult(False, task_id, None, None, None, "PERSISTENCE_OR_STATE_ERROR", str(exc))

    def invalidate_verification_evidence(
        self,
        ledger: TaskLedger,
        task_id: str,
        reason: str,
        trigger_change: dict | None = None,
        *,
        lock_held: bool = False,
    ) -> TransitionResult:
        """Invalidate verified or candidate verification evidence when an underlying dependency changes."""
        try:
            ledger.validate_status_integrity()
            candidate = TaskLedger.from_dict(ledger.to_dict())
            task = candidate.get_task(task_id)
            previous = task.status.value

            # Clear approval
            task.integration_approval = None

            # Mark checkpoint as invalidated
            prev_cp = task.checkpoint.to_dict() if task.checkpoint else {}
            task.checkpoint = CheckpointRecord(
                timestamp=utc_now_iso(),
                stage="VERIFICATION_INVALIDATED",
                details={
                    "reason": reason,
                    "previous_checkpoint": prev_cp,
                    "trigger_change": trigger_change or {},
                },
            )
            task.history.append(EvidenceRecord(
                timestamp=utc_now_iso(),
                stage="VERIFICATION_INVALIDATED",
                details=json.dumps({"reason": reason, "trigger_change": trigger_change or {}}, sort_keys=True),
                success=False,
            ))

            target_status = TaskStatus.RECOVERY_REQUIRED if task.status == TaskStatus.VERIFIED else (
                TaskStatus.PENDING if task.status in (TaskStatus.READY_TO_INTEGRATE, TaskStatus.NEEDS_APPROVAL) else task.status
            )
            candidate_task = candidate.get_task(task_id)
            candidate_task.status = target_status
            candidate_task.blocking_reason = f"Verification invalidated: {reason}"
            candidate._commit_status(task_id, target_status)

            event = self._event(
                task_id,
                "VERIFICATION_INVALIDATED",
                previous,
                target_status.value,
                "coordinator",
                "accepted",
                idempotency_key=f"invalidation:{task_id}:{trigger_change.get('trigger_task_id') if trigger_change else 'manual'}:{hashlib.sha256(reason.encode('utf-8')).hexdigest()[:8]}",
                details={"reason": reason, "trigger_change": trigger_change or {}},
            )
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held, changed_task_id=task_id)
            return TransitionResult(True, task_id, previous, target_status.value, event["event_id"], message=f"Verification evidence invalidated: {reason}")
        except Exception as exc:
            return TransitionResult(False, task_id, None, None, None, "PERSISTENCE_OR_STATE_ERROR", str(exc))

    def create_or_update_revalidation_task(
        self,
        ledger: TaskLedger,
        original_task_id: str,
        reason: str,
        trigger_task_id: str | None = None,
        trigger_change_details: dict | None = None,
        *,
        lock_held: bool = False,
    ) -> tuple[TaskDefinition, TransitionResult]:
        """
        Create a deduplicated revalidation task or update an existing follow-up revalidation task.
        Maintains explicit causal relationships and bounded attempt budgets.
        """
        reval_task_id = f"REVAL-{original_task_id}"
        orig_task = ledger.get_task(original_task_id)

        # Check if reval task already exists
        if reval_task_id in ledger:
            existing = ledger.get_task(reval_task_id)
            # If already pending, in-progress, or verifying, update evidence rather than spawning duplicate
            candidate = TaskLedger.from_dict(ledger.to_dict())
            cand_task = candidate.get_task(reval_task_id)
            cand_task.description = f"Revalidation for {original_task_id}: {reason}"
            cand_task.evidence = json.dumps({
                "trigger_task_id": trigger_task_id,
                "trigger_details": trigger_change_details or {},
                "original_task": original_task_id,
                "reason": reason,
            }, sort_keys=True)
            cand_task.updated_at = utc_now_iso()

            event = self._event(
                reval_task_id,
                "REVALIDATION_UPDATED",
                cand_task.status.value,
                cand_task.status.value,
                "coordinator",
                "accepted",
                idempotency_key=f"reval-updated:{reval_task_id}:{trigger_task_id or 'none'}",
                details={"reason": reason, "trigger_task_id": trigger_task_id},
            )
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held, changed_task_id=reval_task_id)
            return cand_task, TransitionResult(True, reval_task_id, cand_task.status.value, cand_task.status.value, event["event_id"], message="Revalidation task updated.")

        # Create new revalidation task
        new_task = TaskDefinition(
            task_id=reval_task_id,
            title=f"Revalidate {orig_task.title} ({original_task_id})",
            audit_source=f"REVALUATION / {orig_task.audit_source or original_task_id}",
            priority=orig_task.priority,
            risk=orig_task.risk,
            description=f"Automatic bounded revalidation of {original_task_id} triggered by {trigger_task_id or 'subsequent change'}: {reason}",
            evidence=json.dumps({
                "original_task_id": original_task_id,
                "trigger_task_id": trigger_task_id,
                "trigger_details": trigger_change_details or {},
                "reason": reason,
            }, sort_keys=True),
            target_files=list(orig_task.target_files),
            test_targets=list(orig_task.test_targets),
            dependencies=[dep for dep in orig_task.dependencies if dep != reval_task_id and dep != original_task_id],
            max_attempts=orig_task.max_attempts,
            status=TaskStatus.PENDING,
        )
        # Ensure no cyclic dependency is introduced
        deps = [dep for dep in orig_task.dependencies if dep != reval_task_id and dep != original_task_id]
        if trigger_task_id and trigger_task_id in ledger and trigger_task_id != reval_task_id and trigger_task_id != original_task_id:
            # Check if trigger_task_id depends on reval_task_id (would create cycle)
            trigger_task = ledger.get_task(trigger_task_id)
            if reval_task_id not in trigger_task.dependencies and trigger_task_id not in deps:
                deps.append(trigger_task_id)
        new_task.dependencies = deps

        candidate = TaskLedger.from_dict(ledger.to_dict())
        candidate.add_task(new_task)
        event = self._event(
            reval_task_id,
            "REVALIDATION_TASK_CREATED",
            None,
            TaskStatus.PENDING.value,
            "coordinator",
            "accepted",
            idempotency_key=f"reval-created:{reval_task_id}:{trigger_task_id or 'none'}",
            details={"original_task_id": original_task_id, "trigger_task_id": trigger_task_id, "reason": reason},
        )
        candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
        self._persist_candidate(ledger, candidate, lock_held, changed_task_id=reval_task_id)
        return new_task, TransitionResult(True, reval_task_id, None, TaskStatus.PENDING.value, event["event_id"], message="Revalidation task created.")

    def check_integration_revalidations(self, ledger: TaskLedger, task_id: str) -> tuple[bool, str]:
        """
        Transitive revalidation gate check before integration.
        Ensures that:
        1. Task's own revalidation task (REVAL-<task_id>), if present, is VERIFIED.
        2. All direct and transitive prerequisite dependencies are VERIFIED.
        3. For each prerequisite dependency, any corresponding revalidation task
           (REVAL-<dep_id>) is VERIFIED.
        Returns (is_ready, failure_reason).
        """
        task = ledger.get(task_id)
        if not task:
            return False, f"Task '{task_id}' not found in ledger."

        # Check own revalidation
        own_reval = f"REVAL-{task_id}"
        if own_reval in ledger and ledger.get(own_reval).status != TaskStatus.VERIFIED:
            status_val = ledger.get(own_reval).status.value
            return False, f"Pending revalidation '{own_reval}' has not been verified (status: {status_val})."

        # Collect direct and transitive prerequisites
        visited: set[str] = set()

        def _collect(tid: str) -> None:
            t = ledger.get(tid)
            if not t:
                return
            for d in t.dependencies:
                if d not in visited:
                    visited.add(d)
                    _collect(d)

        _collect(task_id)

        # Check all prerequisites and their revalidation tasks
        for dep_id in sorted(visited):
            dep_task = ledger.get(dep_id)
            if not dep_task:
                return False, f"Prerequisite task '{dep_id}' not found in ledger."
            if dep_task.status != TaskStatus.VERIFIED:
                return False, f"Prerequisite task '{dep_id}' is unresolved (status: {dep_task.status.value})."

            dep_reval = f"REVAL-{dep_id}"
            if dep_reval in ledger and ledger.get(dep_reval).status != TaskStatus.VERIFIED:
                dep_reval_status = ledger.get(dep_reval).status.value
                return False, f"Prerequisite '{dep_id}' has unverified revalidation '{dep_reval}' (status: {dep_reval_status})."

        return True, ""

    def consume_cloud_budget(
        self,
        ledger: TaskLedger,
        task_id: str,
        count: int = 1,
        *,
        idempotency_key: str | None = None,
        lock_held: bool = False,
    ) -> bool:
        """
        Atomically check and increment consumed cloud task quota in persistent ledger.
        Deduplicates by idempotency_key to prevent double-charging on retries/restarts.
        Returns True if budget was available and consumed; False if budget limit reached.
        """
        ledger.validate_status_integrity()
        if idempotency_key:
            existing = self._find_event(ledger, idempotency_key)
            if existing and existing.get("event_type") == "CLOUD_BUDGET_CONSUMED":
                return True

        if not lock_held:
            with self.store.acquire_lock():
                latest = self.store.load_ledger()
                if idempotency_key and self._find_event(latest, idempotency_key):
                    self._copy_ledger_into(ledger, latest)
                    return True
                if latest.budget.cloud_tasks_consumed + count > latest.budget.max_cloud_tasks:
                    return False
                candidate = TaskLedger.from_dict(latest.to_dict())
                candidate.budget.cloud_tasks_consumed += count
                candidate.budget.last_updated = utc_now_iso()
                event = self._event(
                    task_id,
                    "CLOUD_BUDGET_CONSUMED",
                    None,
                    None,
                    "coordinator",
                    "accepted",
                    idempotency_key=idempotency_key,
                    details={
                        "consumed": count,
                        "cloud_tasks_consumed": candidate.budget.cloud_tasks_consumed,
                        "max_cloud_tasks": candidate.budget.max_cloud_tasks,
                    },
                )
                candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
                self._persist_candidate(ledger, candidate, lock_held=True, idempotency_key=idempotency_key)
                return True
        else:
            if ledger.budget.cloud_tasks_consumed + count > ledger.budget.max_cloud_tasks:
                return False
            candidate = TaskLedger.from_dict(ledger.to_dict())
            candidate.budget.cloud_tasks_consumed += count
            candidate.budget.last_updated = utc_now_iso()
            event = self._event(
                task_id,
                "CLOUD_BUDGET_CONSUMED",
                None,
                None,
                "coordinator",
                "accepted",
                idempotency_key=idempotency_key,
                details={
                    "consumed": count,
                    "cloud_tasks_consumed": candidate.budget.cloud_tasks_consumed,
                    "max_cloud_tasks": candidate.budget.max_cloud_tasks,
                },
            )
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held=True, idempotency_key=idempotency_key)
            return True

    def record_token_usage(
        self,
        ledger: TaskLedger,
        task_id: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None = None,
        *,
        idempotency_key: str | None = None,
        lock_held: bool = False,
    ) -> bool:
        """
        Atomically record token usage and cost metrics in persistent ledger.
        If tokens are unknown (None), usage remains explicitly unknown and is not recorded as zero.
        """
        if input_tokens is None or output_tokens is None:
            return False
        ledger.validate_status_integrity()
        if idempotency_key:
            existing = self._find_event(ledger, idempotency_key)
            if existing and existing.get("event_type") == "TOKEN_USAGE_RECORDED":
                return True

        cost = cost_usd if (cost_usd is not None and cost_usd >= 0.0) else 0.0

        if not lock_held:
            with self.store.acquire_lock():
                latest = self.store.load_ledger()
                if idempotency_key and self._find_event(latest, idempotency_key):
                    self._copy_ledger_into(ledger, latest)
                    return True
                candidate = TaskLedger.from_dict(latest.to_dict())
                candidate.budget.total_input_tokens += int(input_tokens)
                candidate.budget.total_output_tokens += int(output_tokens)
                candidate.budget.estimated_cost_usd += float(cost)
                candidate.budget.last_updated = utc_now_iso()
                event = self._event(
                    task_id,
                    "TOKEN_USAGE_RECORDED",
                    None,
                    None,
                    "coordinator",
                    "accepted",
                    idempotency_key=idempotency_key,
                    details={
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "cost_usd": float(cost),
                        "total_input_tokens": candidate.budget.total_input_tokens,
                        "total_output_tokens": candidate.budget.total_output_tokens,
                        "total_cost_usd": candidate.budget.estimated_cost_usd,
                    },
                )
                candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
                self._persist_candidate(ledger, candidate, lock_held=True, idempotency_key=idempotency_key)
                return True
        else:
            candidate = TaskLedger.from_dict(ledger.to_dict())
            candidate.budget.total_input_tokens += int(input_tokens)
            candidate.budget.total_output_tokens += int(output_tokens)
            candidate.budget.estimated_cost_usd += float(cost)
            candidate.budget.last_updated = utc_now_iso()
            event = self._event(
                task_id,
                "TOKEN_USAGE_RECORDED",
                None,
                None,
                "coordinator",
                "accepted",
                idempotency_key=idempotency_key,
                details={
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "cost_usd": float(cost),
                    "total_input_tokens": candidate.budget.total_input_tokens,
                    "total_output_tokens": candidate.budget.total_output_tokens,
                    "total_cost_usd": candidate.budget.estimated_cost_usd,
                },
            )
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held=True, idempotency_key=idempotency_key)
            return True

    @property
    def proposals_dir(self) -> Path:
        """Directory for storing inert plan proposals awaiting authorization."""
        p = Path(self.store.state_path).parent / "pending_proposals"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _write_proposal_atomic(self, proposal: PlanProposal) -> None:
        """Atomically persist proposal to disk using temporary file replacement."""
        p_path = self.proposals_dir / f"{proposal.plan_id}.json"
        temp_file = p_path.with_name(f"{p_path.name}.tmp.{os.getpid()}_{uuid.uuid4().hex[:8]}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(proposal.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(5):
                try:
                    os.replace(temp_file, p_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def save_proposal(self, proposal: PlanProposal, lock_held: bool = False) -> bool:
        """Persist a plan proposal and log PLAN_PROPOSED event."""
        self._write_proposal_atomic(proposal)

        # Record planner tokens into ledger budget if available
        if (proposal.input_tokens or proposal.output_tokens) and self.store.exists():
            try:
                p_ledger = self.store.load_ledger()
                self.record_token_usage(
                    p_ledger,
                    f"plan:{proposal.plan_id}",
                    input_tokens=proposal.input_tokens or 0,
                    output_tokens=proposal.output_tokens or 0,
                    cost_usd=0.0,
                    idempotency_key=f"token-usage:plan:{proposal.plan_id}",
                    lock_held=lock_held,
                )
            except Exception:
                pass

        try:
            if not self.store.exists():
                return True
            ledger = self.store.load_ledger()
            idempotency_key = f"plan-proposed:{proposal.plan_id}"
            if self._find_event(ledger, idempotency_key):
                return True

            event = self._event(
                task_id=f"plan:{proposal.plan_id}",
                event_type="PLAN_PROPOSED",
                previous=None,
                resulting="PROPOSED",
                actor="planner",
                outcome="accepted",
                idempotency_key=idempotency_key,
                details={
                    "plan_id": proposal.plan_id,
                    "proposal_hash": proposal.proposal_hash,
                    "task_count": len(proposal.tasks),
                    "objective": proposal.objective,
                },
            )
            candidate = TaskLedger.from_dict(ledger.to_dict())
            candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
            self._persist_candidate(ledger, candidate, lock_held=lock_held, idempotency_key=idempotency_key)
        except Exception:
            pass
        return True

    def get_proposal(self, plan_id: str) -> Optional[PlanProposal]:
        """Load proposal by plan_id."""
        clean_id = Path(plan_id).name.replace(".json", "")
        p_path = self.proposals_dir / f"{clean_id}.json"
        if not p_path.is_file():
            return None
        try:
            data = json.loads(p_path.read_text(encoding="utf-8"))
            return PlanProposal.from_dict(data)
        except Exception:
            return None

    def list_proposals(self) -> list[PlanProposal]:
        """List all plan proposals ordered by creation date."""
        proposals: list[PlanProposal] = []
        for p in self.proposals_dir.glob("PLAN-*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                proposals.append(PlanProposal.from_dict(data))
            except Exception:
                pass
        proposals.sort(key=lambda x: x.created_at, reverse=True)
        return proposals

    def approve_and_register_plan(
        self,
        plan_id: str,
        authorized: bool = False,
        operator_identity: str = "operator",
        lock_held: bool = False,
    ) -> tuple[bool, list[str], str]:
        """
        Approve and atomically register a proposed task plan into the active ledger as PENDING.
        Enforces:
        - Mandatory human authorization (--authorize required).
        - Stable proposal hash verification (anti-tampering).
        - Replay prevention (cannot re-register REGISTERED proposals).
        - Re-validation of contracts against schema v1.0.0 and constraints.
        - Collision checks against existing task IDs in ledger.
        """
        if not authorized:
            return False, [], "Explicit human authorization required (--authorize) to approve and register plan."

        proposal = self.get_proposal(plan_id)
        if not proposal:
            return False, [], f"Plan proposal '{plan_id}' not found."

        if proposal.status == "REGISTERED":
            return False, [], f"Proposal '{plan_id}' has already been registered (replay prevented)."

        if proposal.status == "REJECTED":
            return False, [], f"Proposal '{plan_id}' was rejected ({proposal.rejection_reason or 'No reason provided'})."

        # Verify proposal hash against canonical payload
        computed_hash = compute_proposal_hash(proposal.objective, proposal.tasks)
        if proposal.proposal_hash and computed_hash != proposal.proposal_hash:
            return False, [], f"Proposal '{plan_id}' failed hash verification: content tampering detected."

        # Idempotent crash recovery check: if this exact plan was already registered in the ledger
        # (e.g. process terminated between ledger commit and proposal file update), reconcile proposal
        if self.store.exists():
            current_ledger = self.store.load_ledger()
            idemp_reg = f"plan-registered:{proposal.plan_id}"
            has_reg_event = (
                bool(self._find_event(current_ledger, idemp_reg))
                or idemp_reg in getattr(current_ledger, "applied_idempotency_keys", set())
            )
            if has_reg_event:
                proposal.status = "REGISTERED"
                proposal.approved_at = proposal.approved_at or utc_now_iso()
                proposal.approved_by = proposal.approved_by or operator_identity
                proposal.registered_task_ids = [t["task_id"] for t in proposal.tasks if "task_id" in t]
                self._write_proposal_atomic(proposal)
                return True, proposal.registered_task_ids, f"Plan '{proposal.plan_id}' was already registered in ledger; proposal record reconciled."

        # Re-validate plan contracts deterministically using proposal's configured max_tasks (closing F-GAP-02)
        plan_max_tasks = getattr(proposal, "max_tasks", None) or len(proposal.tasks) or 5
        is_valid, validation_errors = TaskContractValidator.validate_plan(
            {"objective": proposal.objective, "tasks": proposal.tasks},
            max_tasks=plan_max_tasks,
        )
        if not is_valid:
            return False, [], f"Proposal failed validation: {'; '.join(validation_errors)}"

        def _do_registration() -> tuple[bool, list[str], str]:
            if not self.store.exists():
                return False, [], f"State store does not exist at: {self.store.state_path}"

            ledger = self.store.load_ledger()
            candidate = TaskLedger.from_dict(ledger.to_dict())

            # Check ID collisions against current ledger
            task_ids_to_add: list[str] = []
            new_tasks: list[TaskDefinition] = []
            for t_dict in proposal.tasks:
                tid = t_dict.get("task_id")
                if tid in candidate:
                    return False, [], f"Task ID collision: task '{tid}' already exists in ledger."
                t_def = TaskContractValidator.create_task_definition(t_dict)
                t_def.status = TaskStatus.PENDING
                new_tasks.append(t_def)
                task_ids_to_add.append(t_def.task_id)

            # Register tasks in candidate ledger
            for t_def in new_tasks:
                candidate.add_task(t_def)

            # Log events
            idemp_app = f"plan-approved:{proposal.plan_id}"
            idemp_reg = f"plan-registered:{proposal.plan_id}"

            event_app = self._event(
                task_id=f"plan:{proposal.plan_id}",
                event_type="PLAN_APPROVED",
                previous="PROPOSED",
                resulting="APPROVED",
                actor=operator_identity,
                outcome="accepted",
                idempotency_key=idemp_app,
                details={
                    "plan_id": proposal.plan_id,
                    "proposal_hash": proposal.proposal_hash,
                    "task_count": len(new_tasks),
                },
            )
            event_reg = self._event(
                task_id=f"plan:{proposal.plan_id}",
                event_type="PLAN_REGISTERED",
                previous="APPROVED",
                resulting="REGISTERED",
                actor=operator_identity,
                outcome="accepted",
                idempotency_key=idemp_reg,
                details={
                    "plan_id": proposal.plan_id,
                    "registered_tasks": task_ids_to_add,
                },
            )
            candidate.events = (candidate.events + [event_app, event_reg])[-MAX_LIFECYCLE_EVENTS:]

            # Persist candidate ledger atomically
            self._persist_candidate(ledger, candidate, lock_held=True, idempotency_key=idemp_reg)

            # Update proposal record
            proposal.status = "REGISTERED"
            proposal.approved_at = utc_now_iso()
            proposal.approved_by = operator_identity
            proposal.registered_task_ids = task_ids_to_add
            self._write_proposal_atomic(proposal)

            return True, task_ids_to_add, f"Plan '{proposal.plan_id}' successfully approved and registered {len(new_tasks)} task(s)."

        if lock_held:
            return _do_registration()
        with self.store.acquire_lock():
            return _do_registration()

    def reject_plan(
        self,
        plan_id: str,
        reason: str = "Operator rejected",
        operator_identity: str = "operator",
        lock_held: bool = False,
    ) -> tuple[bool, str]:
        """Reject a proposed plan and record PLAN_REJECTED lifecycle event."""
        proposal = self.get_proposal(plan_id)
        if not proposal:
            return False, f"Plan proposal '{plan_id}' not found."

        if proposal.status == "REGISTERED":
            return False, f"Proposal '{plan_id}' is already registered and cannot be rejected."

        proposal.status = "REJECTED"
        proposal.rejection_reason = reason
        self._write_proposal_atomic(proposal)

        # Record event in ledger
        try:
            if self.store.exists():
                def _record():
                    ledger = self.store.load_ledger()
                    candidate = TaskLedger.from_dict(ledger.to_dict())
                    idemp_rej = f"plan-rejected:{proposal.plan_id}"
                    event = self._event(
                        task_id=f"plan:{proposal.plan_id}",
                        event_type="PLAN_REJECTED",
                        previous="PROPOSED",
                        resulting="REJECTED",
                        actor=operator_identity,
                        outcome="accepted",
                        idempotency_key=idemp_rej,
                        details={"plan_id": proposal.plan_id, "reason": reason},
                    )
                    candidate.events = (candidate.events + [event])[-MAX_LIFECYCLE_EVENTS:]
                    self._persist_candidate(ledger, candidate, lock_held=True, idempotency_key=idemp_rej)

                if lock_held:
                    _record()
                else:
                    with self.store.acquire_lock():
                        _record()
        except Exception:
            pass

        return True, f"Plan '{proposal.plan_id}' rejected: {reason}"



def compute_patch_sha256(worktree: Path | str, baseline_commit: str) -> str:
    """Hash all tracked and untracked candidate changes relative to the recorded base."""
    root = Path(worktree).resolve()
    diff = subprocess.run(
        ["git", "diff", "--binary", baseline_commit, "--"], cwd=str(root),
        capture_output=True, check=False,
    )
    if diff.returncode:
        raise RuntimeError(f"Cannot compute patch against baseline {baseline_commit}: {diff.stderr.decode(errors='replace')}")
    h = hashlib.sha256()
    h.update(b"MAYAVE_AUTOPHAGE_PATCH_V1\0")
    h.update(diff.stdout)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--binary", baseline_commit, "--"], cwd=str(root),
        capture_output=True, check=False,
    )
    if staged.returncode:
        raise RuntimeError(f"Cannot compute staged patch against baseline {baseline_commit}: {staged.stderr.decode(errors='replace')}")
    h.update(b"STAGED\0")
    h.update(staged.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=str(root),
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if untracked.returncode:
        raise RuntimeError(f"Cannot enumerate untracked patch files: {untracked.stderr}")
    for rel in sorted(p for p in untracked.stdout.splitlines() if p):
        full = (root / rel).resolve()
        if not full.is_relative_to(root) or not full.is_file():
            raise RuntimeError(f"Unsafe or unreadable untracked patch path: {rel}")
        content = full.read_bytes()
        h.update(rel.replace("\\", "/").encode("utf-8") + b"\0")
        h.update(hashlib.sha256(content).digest())
    return h.hexdigest()


def compute_verification_sha256(
    patch_sha256: str,
    baseline_commit: str,
    quality_report: Any,
    test_targets: list[str],
) -> str:
    """Fingerprint required verification evidence without unstable test output text."""
    gates = getattr(quality_report, "results", [])
    payload = {
        "schema_version": 1,
        "patch_sha256": patch_sha256,
        "baseline_commit": baseline_commit,
        "test_targets": sorted(test_targets),
        "gates": sorted((str(r.gate_name), bool(r.passed)) for r in gates),
        "passed": bool(getattr(quality_report, "passed", False)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
