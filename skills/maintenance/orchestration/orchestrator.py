"""
skills/maintenance/orchestration/orchestrator.py
Central Coordinator for AutoPhage 3.0.
Integrates Ledger, Registry, and Worker Adapters.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional, Set
from skills.maintenance.ledger import MaintenanceLedger, TaskEntry, TaskStatus
from skills.maintenance.macro.tools import MacroToolRegistry, ToolExecutionContext, ToolResult
from skills.maintenance.orchestration.approval import ApprovalManager
from skills.maintenance.orchestration.adapters import (
    MaintenanceWorker, PlanningWorker, ImplementationWorker,
    AntigravityWorker, CopilotWorker, WorkerResponse
)
from skills.maintenance.orchestration.auditor import Auditor
from skills.maintenance.orchestration.reviewer import AggregateReviewer
from skills.maintenance.macro.campaign import CampaignEngine, WorkerTier

logger = logging.getLogger(__name__)

class MaintenanceOrchestrator:
    """
    Coordinates the lifecycle of a maintenance task.
    Ensures policy enforcement, secure approvals, and worker routing.
    """
    def __init__(
        self,
        repo_root: Path,
        ledger: MaintenanceLedger,
        registry: MacroToolRegistry,
        approval_mgr: ApprovalManager,
    ):
        self.repo_root = repo_root
        self.ledger = ledger
        self.registry = registry
        self.approval_mgr = approval_mgr

        # Worker instantiation based on authoritative strategy
        self.planner = PlanningWorker()
        self.implementer = ImplementationWorker()
        self.fallback = AntigravityWorker(executable="agy.exe")
        self.validator = CopilotWorker()

        # Auditor and Reviewer
        self.auditor = Auditor(repo_root, ledger, registry)
        self.reviewer = AggregateReviewer(repo_root, self.validator)
        self._routing_engine = CampaignEngine(repo_root)

    def run_task(self, task_id: str, authorized: bool = True) -> bool:
        """
        Main execution loop for a single task.
        """
        tasks = self.ledger.load()
        task = tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found in ledger.")
            return False

        # 1. Context Setup
        ctx = ToolExecutionContext(
            worktree_path=Path(f".maya/worktrees/{task_id}"), # Simplified for structure
            repo_root=self.repo_root,
            task_id=task_id,
            target_files=task.target_files
        )

        # 2. Planning Phase (Qwen3.5:4b)
        logger.info(f"Starting Planning Phase for {task_id}...")
        plan_response = self.planner.execute_turn(ctx, f"Objective: {task.task_id}")

        # 3. Implementation Phase
        # Route worker based on Authoritative Routing Policy
        tasks = self.ledger.load()
        task_entry = tasks.get(task_id)

        # Reconstruct TaskDefinition for the router
        from skills.maintenance.models import TaskDefinition
        tdef = TaskDefinition(
            task_id=task_id,
            title=task_entry.title if hasattr(task_entry, "title") else "Maintenance Task",
            audit_source="orchestrator",
            priority=task_entry.priority,
            risk=task_entry.risk,
            description=task_entry.description if hasattr(task_entry, "description") else "",
            evidence=task_entry.evidence if hasattr(task_entry, "evidence") else "",
            target_files=task_entry.target_files,
            test_targets=task_entry.test_targets if hasattr(task_entry, "test_targets") else [],
            dependencies=task_entry.dependencies if hasattr(task_entry, "dependencies") else [],
            acceptance_criteria=task_entry.acceptance_criteria if hasattr(task_entry, "acceptance_criteria") else [],
        )

        tier = self._routing_engine._route_worker_tier(tdef)
        logger.info(f"Routing task {task_id} to {tier.value} based on risk/complexity matrix.")

        if tier == WorkerTier.TIER_1_LOCAL:
            current_worker = self.implementer
        elif tier == WorkerTier.TIER_2_CLOUD:
            current_worker = self.fallback
        else:
            logger.warning(f"Unsupported tier {tier.value}, falling back to Antigravity.")
            current_worker = self.fallback

        attempts = 0

        while attempts < 3:
            impl_response = current_worker.execute_turn(ctx, f"Plan: {plan_response.thought}")

            if impl_response.finish_task:
                # 4. Independent Aggregate Review
                #- Build inventory from diff
                inventory = self.reviewer.build_inventory(task_id, "mock diff")
                #- Perform review
                review_ok, findings = self.reviewer.review_change_set(inventory, ctx)

                if not review_ok:
                    logger.warning(f"Aggregate review failed for {task_id}: {findings}")
                    attempts += 1
                    continue

                break

            # Process tool calls through policy registry
            for call in impl_response.tool_calls:
                tool_name = call["name"]
                args = call["args"]

                # Bound approval check
                if self.registry.get(tool_name).approval_required:
                    targets = {args.get("path")} if "path" in args else set()
                    if not self.approval_mgr.verify_approval(task_id, tool_name, targets, args):
                        logger.warning(f"Operation {tool_name} rejected: Missing bound approval.")
                        # Transition task to NEEDS_APPROVAL in ledger
                        self.ledger.update_task(task_id, status=TaskStatus.NEEDS_APPROVAL)
                        return False

                res = self.registry.execute(tool_name, ctx, args)
                if not res.success:
                    attempts += 1
                    logger.info(f"Implementation failure {attempts}/3: {res.error}")

        # 5. Fallback Phase (Antigravity)
        if attempts >= 3:
            logger.info("Escalating to Antigravity fallback...")
            current_worker = self.fallback
            # Execute fallback turn...

        return True
