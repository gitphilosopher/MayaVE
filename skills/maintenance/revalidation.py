"""
Bounded Automatic Revalidation Orchestrator for MayaVE Maintenance Coordinator (Autophage P2).

Handles:
- Processing impact assessments and triggering CoordinatorAPI invalidations.
- Automated creation or deduplicated update of bounded revalidation tasks.
- Topological downstream scheduling order.
- Cycle prevention, recursion limits, and idempotent resume after interruption.
- Verification failure repair routing escalation without bypassing approval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

try:
    from .dependency import DependencyRegistry, ImpactAssessment, ImpactClassification
    from .hub import CoordinatorAPI, TransitionResult
    from .ledger import TaskLedger
    from .models import TaskDefinition, TaskPriority, TaskRisk, TaskStatus, utc_now_iso
    from .store import StateStore
except ImportError:
    from dependency import DependencyRegistry, ImpactAssessment, ImpactClassification
    from hub import CoordinatorAPI, TransitionResult
    from ledger import TaskLedger
    from models import TaskDefinition, TaskPriority, TaskRisk, TaskStatus, utc_now_iso
    from store import StateStore

logger = logging.getLogger("maya.maintenance.revalidation")

# Maximum revalidation cascade depth to prevent infinite loops
MAX_REVALIDATION_DEPTH = 5


@dataclass
class RevalidationSummary:
    """Summary of an automated revalidation evaluation cycle."""
    trigger_task_id: Optional[str]
    invalidated_tasks: list[str] = field(default_factory=list)
    revalidation_tasks_created: list[str] = field(default_factory=list)
    revalidation_tasks_updated: list[str] = field(default_factory=list)
    skipped_unaffected: list[str] = field(default_factory=list)
    human_review_required: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trigger_task_id": self.trigger_task_id,
            "invalidated_tasks": sorted(self.invalidated_tasks),
            "revalidation_tasks_created": sorted(self.revalidation_tasks_created),
            "revalidation_tasks_updated": sorted(self.revalidation_tasks_updated),
            "skipped_unaffected": sorted(self.skipped_unaffected),
            "human_review_required": sorted(self.human_review_required),
            "details": self.details,
        }


class RevalidationOrchestrator:
    """
    Coordinates change-impact detection, evidence invalidation, and bounded revalidation scheduling.
    """

    def __init__(
        self,
        store: StateStore,
        registry: DependencyRegistry,
        repo_root: Path | str | None = None,
        max_depth: int = MAX_REVALIDATION_DEPTH,
    ) -> None:
        self.store = store
        self.registry = registry
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.max_depth = max_depth

    def calculate_cascade_depth(
        self,
        task_id: str,
        trigger_task_id: str | None,
        ledger: TaskLedger,
    ) -> tuple[int, bool]:
        """
        Calculate cascade depth for an affected task branch.
        Returns (depth, is_safe). If is_safe is False, depth could not be established safely.
        """
        try:
            # 1. Prefix depth of the affected task itself
            affected_prefix = 0
            curr = task_id
            while curr and curr.startswith("REVAL-"):
                affected_prefix += 1
                curr = curr[len("REVAL-"):]
                if affected_prefix > self.max_depth + 2:
                    return affected_prefix, True

            # 2. Trigger task depth (prefix + causal)
            trigger_depth = 0
            if trigger_task_id:
                tcurr = trigger_task_id
                while tcurr and tcurr.startswith("REVAL-"):
                    trigger_depth += 1
                    tcurr = tcurr[len("REVAL-"):]
                    if trigger_depth > self.max_depth + 2:
                        return trigger_depth, True

                # Follow causal chain of trigger_task_id in ledger
                visited_trig: set[str] = set()
                chain_curr = trigger_task_id
                causal_trig_steps = 0
                while chain_curr:
                    if chain_curr in visited_trig:
                        # Cycle detected in trigger causal chain -> unsafe
                        return self.max_depth + 1, False
                    visited_trig.add(chain_curr)
                    if chain_curr not in ledger:
                        break
                    t_obj = ledger.get(chain_curr)
                    if not t_obj or not t_obj.evidence:
                        break
                    ev_text = t_obj.evidence.strip()
                    if ev_text.startswith("{"):
                        try:
                            ev_data = json.loads(ev_text)
                            if isinstance(ev_data, dict) and "trigger_task_id" in ev_data:
                                parent_trig = ev_data.get("trigger_task_id")
                                if parent_trig:
                                    if parent_trig in visited_trig or parent_trig == chain_curr:
                                        # Self-reference or cycle in trigger chain: unsafe
                                        return self.max_depth + 1, False
                                    causal_trig_steps += 1
                                    chain_curr = parent_trig
                                    if causal_trig_steps > self.max_depth + 2:
                                        break
                                    continue
                        except Exception:
                            # Corrupted JSON evidence: fail closed
                            return self.max_depth + 1, False
                    break
                trigger_depth = max(trigger_depth, causal_trig_steps)

            # 3. Causal chain of affected task
            visited_aff: set[str] = set()
            aff_curr = task_id
            causal_aff_steps = 0
            while aff_curr:
                if aff_curr in visited_aff:
                    # Cycle detected in affected task chain -> unsafe
                    return self.max_depth + 1, False
                visited_aff.add(aff_curr)
                if aff_curr not in ledger:
                    break
                a_obj = ledger.get(aff_curr)
                if not a_obj or not a_obj.evidence:
                    break
                ev_text = a_obj.evidence.strip()
                if ev_text.startswith("{"):
                    try:
                        ev_data = json.loads(ev_text)
                        if isinstance(ev_data, dict) and "trigger_task_id" in ev_data:
                            parent_trig = ev_data.get("trigger_task_id")
                            if parent_trig:
                                if parent_trig in visited_aff or parent_trig == aff_curr:
                                    # Self-reference or cycle in affected causal evidence: unsafe
                                    return self.max_depth + 1, False
                                causal_aff_steps += 1
                                aff_curr = parent_trig
                                if causal_aff_steps > self.max_depth + 2:
                                    break
                                continue
                    except Exception:
                        return self.max_depth + 1, False
                break

            # Branch depth combining trigger propagation and affected task depth
            branch_from_trigger = (trigger_depth + 1) if trigger_task_id else 0
            total_depth = max(affected_prefix, causal_aff_steps, branch_from_trigger)
            return total_depth, True
        except Exception:
            return self.max_depth + 1, False

    def process_change_event(
        self,
        changed_files: list[str],
        trigger_task_id: str | None = None,
        trigger_type: str = "INTEGRATION",
        patch_ast_details: dict[str, dict] | None = None,
        lock_held: bool = False,
    ) -> RevalidationSummary:
        """
        Evaluate impact of changed files, invalidate stale evidence/approvals through CoordinatorAPI,
        and schedule bounded revalidation tasks in topological dependency order.
        """
        summary = RevalidationSummary(trigger_task_id=trigger_task_id)

        # 1. Load latest ledger
        ledger = self.store.load_ledger()
        hub = CoordinatorAPI(self.store)

        # 2. Register all tasks in dependency registry
        for task in ledger:
            self.registry.register_task(task)

        # 3. Assess impact
        assessments = self.registry.assess_impact(
            changed_files=changed_files,
            trigger_task_id=trigger_task_id,
            candidate_tasks=list(ledger),
            patch_ast_details=patch_ast_details,
        )

        # 4. Filter affected vs unaffected
        to_invalidate: list[tuple[TaskDefinition, ImpactAssessment]] = []
        for assessment in assessments:
            t_id = assessment.affected_task_id
            task = ledger.get(t_id)
            if not task:
                continue

            if assessment.classification == ImpactClassification.UNAFFECTED_WITH_EVIDENCE:
                summary.skipped_unaffected.append(t_id)
                continue

            if assessment.classification in (
                ImpactClassification.AFFECTED_CONFIRMED,
                ImpactClassification.POTENTIALLY_AFFECTED,
                ImpactClassification.IMPACT_UNKNOWN,
            ):
                to_invalidate.append((task, assessment))

        # 5. Order invalidations: direct dependencies first, then downstream dependents
        # Topological sorting
        downstream_counts: dict[str, int] = {}
        for t, _ in to_invalidate:
            downstream_counts[t.task_id] = sum(
                1 for other in ledger if t.task_id in other.dependencies
            )
        to_invalidate.sort(
            key=lambda item: (
                0 if item[1].classification == ImpactClassification.AFFECTED_CONFIRMED else 1,
                -downstream_counts.get(item[0].task_id, 0),
                item[0].task_id,
            )
        )

        # 6. Execute invalidations and revalidations through CoordinatorAPI
        trigger_change_info = {
            "trigger_task_id": trigger_task_id,
            "trigger_type": trigger_type,
            "changed_files": list(changed_files),
        }

        for task, assessment in to_invalidate:
            # Enforce affected-task cascade depth limit and fail closed if unsafe
            depth, is_safe = self.calculate_cascade_depth(task.task_id, trigger_task_id, ledger)
            if depth >= self.max_depth or not is_safe:
                # Invalidate existing verification evidence if needed
                if task.status in (TaskStatus.VERIFIED, TaskStatus.READY_TO_INTEGRATE):
                    hub.invalidate_verification_evidence(
                        ledger,
                        task.task_id,
                        reason=f"Revalidation cascade depth limit reached or unsafe: {assessment.reason}",
                        trigger_change=trigger_change_info,
                        lock_held=lock_held,
                    )
                reason_msg = (
                    f"Revalidation cascade depth limit ({self.max_depth}) reached (depth={depth}); operator review required."
                    if is_safe
                    else "Revalidation cascade depth cannot be established safely; operator review required."
                )
                hub.transition_task(
                    ledger,
                    task.task_id,
                    TaskStatus.NEEDS_APPROVAL,
                    actor="coordinator",
                    blocking_reason=reason_msg,
                    checkpoint_stage="REVALIDATION_DEPTH_EXCEEDED",
                    checkpoint_details={"depth": depth, "is_safe": is_safe, "assessment": assessment.to_dict()},
                    event_type="REVALIDATION_CASCADE_BLOCKED",
                    idempotency_key=f"depth-exceeded:{task.task_id}:{depth}:{is_safe}",
                    lock_held=lock_held,
                )
                summary.human_review_required.append(task.task_id)
                continue

            # Invalidate verification evidence if task was verified or ready to integrate
            if task.status in (TaskStatus.VERIFIED, TaskStatus.READY_TO_INTEGRATE, TaskStatus.NEEDS_APPROVAL):
                inv_res = hub.invalidate_verification_evidence(
                    ledger,
                    task.task_id,
                    reason=assessment.reason,
                    trigger_change=trigger_change_info,
                    lock_held=lock_held,
                )
                if inv_res.accepted:
                    summary.invalidated_tasks.append(task.task_id)

            # If impact unknown or high risk, escalate to human review
            if assessment.classification == ImpactClassification.IMPACT_UNKNOWN:
                hub.transition_task(
                    ledger,
                    task.task_id,
                    TaskStatus.NEEDS_APPROVAL,
                    actor="coordinator",
                    blocking_reason=f"Impact unknown due to missing dependency evidence: {assessment.reason}. Human review required.",
                    checkpoint_stage="IMPACT_UNKNOWN_REVIEW",
                    checkpoint_details={"assessment": assessment.to_dict()},
                    event_type="IMPACT_UNKNOWN_REVIEW_REQUIRED",
                    idempotency_key=f"impact-unknown:{task.task_id}:{trigger_task_id or 'none'}",
                    lock_held=lock_held,
                )
                summary.human_review_required.append(task.task_id)
                continue

            # Schedule revalidation task
            # Prevent revalidating a revalidation task (avoid REVAL-REVAL-...)
            orig_id = task.task_id
            if orig_id.startswith("REVAL-"):
                base_orig = orig_id[len("REVAL-"):]
                if base_orig in ledger:
                    orig_id = base_orig

            reval_id = f"REVAL-{orig_id}"
            is_new = reval_id not in ledger

            reval_task, res = hub.create_or_update_revalidation_task(
                ledger,
                original_task_id=orig_id,
                reason=assessment.reason,
                trigger_task_id=trigger_task_id,
                trigger_change_details=trigger_change_info,
                lock_held=lock_held,
            )
            if res.accepted:
                if is_new:
                    summary.revalidation_tasks_created.append(reval_task.task_id)
                else:
                    summary.revalidation_tasks_updated.append(reval_task.task_id)

        return summary
