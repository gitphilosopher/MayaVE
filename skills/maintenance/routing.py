"""
Complexity-Aware Model Routing for MayaVE Autophage Maintenance Engine (P3).
Deterministic, configurable, model-independent task routing across local,
efficient cloud, advanced cloud, and human-decision tiers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Set, Tuple

try:
    from .adapter import (
        AgentAdapter,
        AgentExecutionResult,
        AntigravityAgentAdapter,
        FakeAgentAdapter,
        LocalModelAgentAdapter,
        resolve_antigravity_executable,
    )
    from .hub import CoordinatorAPI
    from .ledger import TaskLedger
    from .models import TaskDefinition, TaskPriority, TaskRisk, TaskStatus, utc_now_iso
    from .reviewer import ReviewPolicy, ReviewerAdapter
except ImportError:
    from adapter import (
        AgentAdapter,
        AgentExecutionResult,
        AntigravityAgentAdapter,
        FakeAgentAdapter,
        LocalModelAgentAdapter,
        resolve_antigravity_executable,
    )
    from hub import CoordinatorAPI
    from ledger import TaskLedger
    from models import TaskDefinition, TaskPriority, TaskRisk, TaskStatus, utc_now_iso
    from reviewer import ReviewPolicy, ReviewerAdapter

logger = logging.getLogger("maya.maintenance.routing")


class RoutingTier(str, Enum):
    TIER_1_LOCAL = "tier_1_local"              # Local Ollama adapter (e.g. llama3.2)
    TIER_2_EFFICIENT_CLOUD = "tier_2_efficient_cloud"  # Efficient cloud adapter (Antigravity / implementer)
    TIER_3_ADVANCED_CLOUD = "tier_3_advanced_cloud"    # Advanced cloud adapter (Antigravity / architect)
    TIER_4_HUMAN_DECISION = "tier_4_human_decision"    # Independent review + human approval (NEEDS_APPROVAL)


ARCHITECTURAL_SUBSYSTEMS: tuple[str, ...] = (
    "core",
    "brain",
    "services",
    "frontend",
    "config",
    "scripts",
    "skills",
    "audio",
)

SECURITY_SENSITIVE_PATTERNS: tuple[str, ...] = (
    ".git",
    ".env",
    "token",
    "secret",
    "password",
    "credential",
    "key",
    "cert",
    "auth",
    "oauth",
    "crypto",
    "terminal_approval.py",
    "approvals.json",
    "hub.py",
    "coordinator.py",
    "validator.py",
    "store.py",
    "active_session.json",
)


def normalize_target_path(f: str) -> str:
    """Normalize a target file path for consistent security and subsystem checks."""
    if not f:
        return ""
    norm = f.replace("\\", "/").strip()
    while norm.startswith("./") or norm.startswith("/"):
        norm = norm[2:] if norm.startswith("./") else norm[1:]
    return norm.lower()


def detect_subsystems(target_files: Sequence[str]) -> tuple[list[str], bool]:
    """
    Detect which architectural subsystems are touched by declared target files,
    and whether the task crosses subsystem boundaries.
    """
    subsystems: set[str] = set()
    for f in target_files:
        norm = normalize_target_path(f)
        parts = norm.split("/")
        if parts:
            first = parts[0]
            if first in ARCHITECTURAL_SUBSYSTEMS:
                subsystems.add(first)
            elif norm.startswith("main.py"):
                subsystems.add("core")
            else:
                subsystems.add(first or "misc")
    sorted_subs = sorted(subsystems)
    crosses_boundaries = len(subsystems) > 1
    return sorted_subs, crosses_boundaries


def has_security_sensitive_targets(target_files: Sequence[str]) -> bool:
    """Check if any target file matches security-sensitive paths or patterns."""
    for f in target_files:
        norm = normalize_target_path(f)
        if not norm:
            continue
        if any(pat in norm for pat in SECURITY_SENSITIVE_PATTERNS):
            return True
        parts = norm.split("/")
        for part in parts:
            if any(pat in part for pat in SECURITY_SENSITIVE_PATTERNS):
                return True
    return False


@dataclass
class RoutingConfig:
    """Minimal, documented configuration for routing thresholds, budgets, and ceilings."""
    # Scope thresholds
    tier1_max_files: int = 2
    tier1_max_risk: TaskRisk = TaskRisk.LOW
    tier2_max_files: int = 5
    tier2_max_risk: TaskRisk = TaskRisk.MEDIUM
    # Cross-boundary rule: tasks that cross multiple architectural boundaries require Tier 3+
    escalate_cross_boundary_tier: RoutingTier = RoutingTier.TIER_3_ADVANCED_CLOUD
    # Max retries
    tier1_max_retries: int = 1
    tier2_max_retries: int = 2
    tier3_max_retries: int = 2
    # Adapter availability flags
    local_adapter_enabled: bool = True
    cloud_adapter_enabled: bool = True
    # Cloud budget constraints
    max_cloud_tasks_per_session: int = 10
    cloud_tasks_consumed: int = 0
    # Model IDs / roles from existing codebase
    local_model: str = "llama3.2"
    efficient_cloud_agent: str = "implementer"
    advanced_cloud_agent: str = "architect"
    # Local resource ceiling: e.g. percent memory ceiling
    max_local_memory_percent: float = 90.0
    current_memory_percent: Optional[float] = None
    # Fallback policy flags
    allow_fallback_to_cloud: bool = True
    allow_fallback_to_advanced: bool = True
    # Concurrency controls
    max_implementation_workers: int = 1
    allow_concurrent_isolated_review: bool = False

    def has_cloud_budget(self) -> bool:
        return self.cloud_tasks_consumed < self.max_cloud_tasks_per_session

    def consume_cloud_budget(self) -> None:
        self.cloud_tasks_consumed += 1


@dataclass
class ComplexityAssessment:
    """Structured assessment of task complexity and evidence."""
    assessed_tier: RoutingTier
    confidence: float
    uncertainty: float
    subsystems_affected: list[str]
    crosses_boundaries: bool
    reasons: list[str]
    is_ambiguous_or_contradictory: bool = False
    requires_human_approval: bool = False
    raw_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "assessed_tier": self.assessed_tier.value,
            "confidence": round(self.confidence, 2),
            "uncertainty": round(self.uncertainty, 2),
            "subsystems_affected": list(self.subsystems_affected),
            "crosses_boundaries": self.crosses_boundaries,
            "reasons": list(self.reasons),
            "is_ambiguous_or_contradictory": self.is_ambiguous_or_contradictory,
            "requires_human_approval": self.requires_human_approval,
            "raw_metrics": dict(self.raw_metrics),
        }


@dataclass
class RoutingDecision:
    """Structured routing decision returned by the complexity router."""
    task_id: str
    selected_tier: RoutingTier
    adapter_name: str
    adapter_config: dict
    is_available: bool
    reasons: list[str]
    confidence: float
    uncertainty: float
    subsystems_affected: list[str]
    crosses_boundaries: bool
    fallback_eligible: bool
    fallback_tier: Optional[RoutingTier]
    requires_human_approval: bool
    budget_permitted: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "selected_tier": self.selected_tier.value,
            "adapter_name": self.adapter_name,
            "adapter_config": dict(self.adapter_config),
            "is_available": self.is_available,
            "reasons": list(self.reasons),
            "confidence": round(self.confidence, 2),
            "uncertainty": round(self.uncertainty, 2),
            "subsystems_affected": list(self.subsystems_affected),
            "crosses_boundaries": self.crosses_boundaries,
            "fallback_eligible": self.fallback_eligible,
            "fallback_tier": self.fallback_tier.value if self.fallback_tier else None,
            "requires_human_approval": self.requires_human_approval,
            "budget_permitted": self.budget_permitted,
            "metadata": dict(self.metadata),
        }


@dataclass
class FallbackDecision:
    """Decision on whether a failed adapter execution qualifies for fallback."""
    qualifies: bool
    task_id: str
    original_tier: RoutingTier
    fallback_tier: Optional[RoutingTier]
    fallback_adapter: Optional[str]
    reason: str
    non_qualifying_reason: Optional[str] = None
    requires_human_approval: bool = False

    def to_dict(self) -> dict:
        return {
            "qualifies": self.qualifies,
            "task_id": self.task_id,
            "original_tier": self.original_tier.value if self.original_tier else None,
            "fallback_tier": self.fallback_tier.value if self.fallback_tier else None,
            "fallback_adapter": self.fallback_adapter,
            "reason": self.reason,
            "non_qualifying_reason": self.non_qualifying_reason,
            "requires_human_approval": self.requires_human_approval,
        }


class ComplexityAssessor:
    """
    Evaluates task signals and evidence to determine complexity tier, uncertainty,
    and required review/approval policies.
    """

    def __init__(self, config: Optional[RoutingConfig] = None) -> None:
        self.config = config or RoutingConfig()

    def assess(self, task: TaskDefinition) -> ComplexityAssessment:
        reasons: list[str] = []
        raw_metrics: dict[str, Any] = {}

        # 1. Missing or unverified scope evidence check
        target_files = task.target_files or []
        raw_metrics["file_count"] = len(target_files)
        raw_metrics["attempts"] = task.attempts
        raw_metrics["risk"] = task.risk.value if hasattr(task.risk, "value") else str(task.risk)

        if not target_files:
            reasons.append("Missing declared target files: scope cannot be safely bounded.")
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                confidence=0.1,
                uncertainty=1.0,
                subsystems_affected=[],
                crosses_boundaries=False,
                reasons=reasons,
                is_ambiguous_or_contradictory=True,
                requires_human_approval=True,
                raw_metrics=raw_metrics,
            )

        # 2. Risk validation & fail closed: None, missing, or invalid risk metadata (F-07)
        valid_risks = (TaskRisk.LOW, TaskRisk.MEDIUM, TaskRisk.HIGH, TaskRisk.VERY_HIGH)
        if task.risk not in valid_risks:
            reasons.append(
                f"Missing, unknown, or invalid risk classification ('{task.risk}'); fails closed to Tier 4 human authorization."
            )
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                confidence=0.1,
                uncertainty=1.0,
                subsystems_affected=detect_subsystems(target_files)[0] if target_files else [],
                crosses_boundaries=detect_subsystems(target_files)[1] if target_files else False,
                reasons=reasons,
                is_ambiguous_or_contradictory=True,
                requires_human_approval=True,
                raw_metrics=raw_metrics,
            )

        # 3. Security-sensitive targets: MUST route to Tier 4 regardless of declared risk level (F-01 / R-01)
        is_security_sensitive = has_security_sensitive_targets(target_files)
        raw_metrics["is_security_sensitive"] = is_security_sensitive

        if is_security_sensitive:
            is_contradictory = (task.risk == TaskRisk.LOW)
            reasons.append(
                f"Task touches security-sensitive repository paths; human authorization required (declared risk: {task.risk.value})."
            )
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                confidence=0.3 if is_contradictory else 1.0,
                uncertainty=0.9 if is_contradictory else 0.0,
                subsystems_affected=detect_subsystems(target_files)[0],
                crosses_boundaries=detect_subsystems(target_files)[1],
                reasons=reasons,
                is_ambiguous_or_contradictory=is_contradictory,
                requires_human_approval=True,
                raw_metrics=raw_metrics,
            )

        # 4. Explicit HIGH or VERY_HIGH risk policy check
        if task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH):
            reasons.append(f"Task risk is {task.risk.value}: high-impact policy requires human authorization.")
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                confidence=1.0,
                uncertainty=0.0,
                subsystems_affected=detect_subsystems(target_files)[0],
                crosses_boundaries=detect_subsystems(target_files)[1],
                reasons=reasons,
                is_ambiguous_or_contradictory=False,
                requires_human_approval=True,
                raw_metrics=raw_metrics,
            )

        # 5. Prior attempt security denial or non-qualifying failure check (F-02 / R-02)
        prev_failure = (task.failure_reason or "").lower()
        has_security_fail = any(
            sec_term in prev_failure
            for sec_term in (
                "permission denied",
                "security",
                "sandbox",
                "non-qualifying",
                "unauthorized",
                "path traversal",
            )
        )
        checkpoint_val = getattr(task, "checkpoint", None)
        if checkpoint_val is not None:
            details = getattr(checkpoint_val, "details", {})
            if isinstance(details, dict):
                if (
                    details.get("last_non_qualifying_reason")
                    or details.get("non_qualifying_reason")
                    or details.get("permission_denied")
                ):
                    has_security_fail = True
            elif isinstance(checkpoint_val, dict):
                d = checkpoint_val.get("details", {})
                if isinstance(d, dict) and (
                    d.get("last_non_qualifying_reason")
                    or d.get("non_qualifying_reason")
                    or d.get("permission_denied")
                ):
                    has_security_fail = True

        if has_security_fail:
            reasons.append(
                "Previous attempt failed with non-qualifying error or security violation; cannot escalate to cloud execution. Human authorization required."
            )
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                confidence=1.0,
                uncertainty=0.0,
                subsystems_affected=detect_subsystems(target_files)[0],
                crosses_boundaries=detect_subsystems(target_files)[1],
                reasons=reasons,
                is_ambiguous_or_contradictory=False,
                requires_human_approval=True,
                raw_metrics=raw_metrics,
            )

        # 6. Max attempts / retry exhaustion check
        if task.attempts >= task.max_attempts:
            reasons.append(f"Max execution attempts ({task.max_attempts}) reached; requires human intervention.")
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                confidence=1.0,
                uncertainty=0.0,
                subsystems_affected=detect_subsystems(target_files)[0],
                crosses_boundaries=detect_subsystems(target_files)[1],
                reasons=reasons,
                is_ambiguous_or_contradictory=False,
                requires_human_approval=True,
                raw_metrics=raw_metrics,
            )

        # 7. Subsystem detection and boundary crossing
        subsystems, crosses_boundaries = detect_subsystems(target_files)
        raw_metrics["subsystems"] = subsystems
        raw_metrics["crosses_boundaries"] = crosses_boundaries

        if crosses_boundaries:
            reasons.append(
                f"Task crosses architectural boundaries across multiple subsystems: {', '.join(subsystems)}."
            )
            # Boundary crossing tasks require at least Tier 3 (Advanced Cloud)
            return ComplexityAssessment(
                assessed_tier=self.config.escalate_cross_boundary_tier,
                confidence=0.85,
                uncertainty=0.2,
                subsystems_affected=subsystems,
                crosses_boundaries=True,
                reasons=reasons,
                is_ambiguous_or_contradictory=False,
                requires_human_approval=False,
                raw_metrics=raw_metrics,
            )

        # 6. Scope and risk tier evaluation
        file_count = len(target_files)

        # Tier 1 eligibility: LOW risk, <= tier1_max_files, attempt 0 or 1
        if (
            task.risk == TaskRisk.LOW
            and file_count <= self.config.tier1_max_files
            and task.attempts < self.config.tier1_max_retries
        ):
            reasons.append(
                f"Small bounded scope ({file_count} files) and LOW risk qualifies for Tier 1 local model."
            )
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_1_LOCAL,
                confidence=0.9,
                uncertainty=0.1,
                subsystems_affected=subsystems,
                crosses_boundaries=False,
                reasons=reasons,
                is_ambiguous_or_contradictory=False,
                requires_human_approval=False,
                raw_metrics=raw_metrics,
            )

        # Escalation from prior attempts on Tier 1
        if task.attempts >= self.config.tier1_max_retries and file_count <= self.config.tier1_max_files:
            reasons.append(
                f"Previous local model attempt ({task.attempts}) failed or reached retry limit; escalating to Tier 2 efficient cloud."
            )
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_2_EFFICIENT_CLOUD,
                confidence=0.85,
                uncertainty=0.2,
                subsystems_affected=subsystems,
                crosses_boundaries=False,
                reasons=reasons,
                is_ambiguous_or_contradictory=False,
                requires_human_approval=False,
                raw_metrics=raw_metrics,
            )

        # Tier 2 eligibility: MEDIUM risk or files <= tier2_max_files
        if (
            task.risk in (TaskRisk.LOW, TaskRisk.MEDIUM)
            and file_count <= self.config.tier2_max_files
            and task.attempts < (self.config.tier1_max_retries + self.config.tier2_max_retries)
        ):
            reasons.append(
                f"Medium complexity ({file_count} files, {task.risk.value} risk) routes to Tier 2 efficient cloud."
            )
            return ComplexityAssessment(
                assessed_tier=RoutingTier.TIER_2_EFFICIENT_CLOUD,
                confidence=0.85,
                uncertainty=0.15,
                subsystems_affected=subsystems,
                crosses_boundaries=False,
                reasons=reasons,
                is_ambiguous_or_contradictory=False,
                requires_human_approval=False,
                raw_metrics=raw_metrics,
            )

        # Tier 3: Complex multi-file or higher retry count
        reasons.append(
            f"High complexity ({file_count} files or multiple previous attempts) routes to Tier 3 advanced cloud."
        )
        return ComplexityAssessment(
            assessed_tier=RoutingTier.TIER_3_ADVANCED_CLOUD,
            confidence=0.8,
            uncertainty=0.2,
            subsystems_affected=subsystems,
            crosses_boundaries=False,
            reasons=reasons,
            is_ambiguous_or_contradictory=False,
            requires_human_approval=False,
            raw_metrics=raw_metrics,
        )


class ComplexityRouter:
    """
    Deterministic, configurable routing policy orchestrating task tier selection,
    adapter availability verification, resource ceilings, cloud budget tracking,
    and safe fallback.
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        assessor: Optional[ComplexityAssessor] = None,
        local_probe: Optional[Callable[[], bool]] = None,
        cloud_probe: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.config = config or RoutingConfig()
        self.assessor = assessor or ComplexityAssessor(self.config)
        self.local_probe = local_probe
        self.cloud_probe = cloud_probe
        self.routing_history: list[RoutingDecision] = []
        self.fallback_history: list[FallbackDecision] = []

    def check_local_available(self) -> tuple[bool, str]:
        """Verify local adapter availability (enabled, resource pressure, endpoint)."""
        if not self.config.local_adapter_enabled:
            return False, "Local adapter is disabled by configuration."

        # Resource ceiling check
        if self.config.current_memory_percent is not None:
            if self.config.current_memory_percent > self.config.max_local_memory_percent:
                return False, (
                    f"Local memory pressure ({self.config.current_memory_percent:.1f}%) exceeds "
                    f"configured ceiling ({self.config.max_local_memory_percent:.1f}%)."
                )

        if self.local_probe is not None:
            if not self.local_probe():
                return False, "Local model endpoint probe returned unreachable."

        return True, "Local model endpoint is available."

    def check_cloud_available(self, authorized: bool) -> tuple[bool, str]:
        """Verify cloud adapter availability (enabled, authorization, budget, binary)."""
        if not self.config.cloud_adapter_enabled:
            return False, "Cloud adapter is disabled by configuration."

        if not authorized:
            return False, "Cloud adapter execution requires explicit operator authorization (--authorize)."

        if not self.config.has_cloud_budget():
            return False, (
                f"Cloud task session budget exhausted ({self.config.cloud_tasks_consumed}/"
                f"{self.config.max_cloud_tasks_per_session})."
            )

        if self.cloud_probe is not None:
            if not self.cloud_probe():
                return False, "Cloud adapter probe returned unavailable."

        resolved_exe = resolve_antigravity_executable()
        if not (shutil.which(resolved_exe) or Path(resolved_exe).is_file()):
            return False, f"Antigravity CLI binary '{resolved_exe}' not found."

        return True, "Cloud adapter is verified and available."

    def route_task(
        self,
        task: TaskDefinition,
        authorized: bool = False,
    ) -> RoutingDecision:
        """
        Evaluate task evidence and return a structured routing decision with conservative
        thresholds and fallback eligibility.
        """
        assessment = self.assessor.assess(task)
        reasons = list(assessment.reasons)
        metadata: dict[str, Any] = {
            "assessment": assessment.to_dict(),
            "authorized": authorized,
        }

        # Case 1: Assessment determined Tier 4 (Human decision / NEEDS_APPROVAL)
        if assessment.assessed_tier == RoutingTier.TIER_4_HUMAN_DECISION or assessment.requires_human_approval:
            decision = RoutingDecision(
                task_id=task.task_id,
                selected_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                adapter_name="human_approval",
                adapter_config={"action": "NEEDS_APPROVAL"},
                is_available=True,
                reasons=reasons,
                confidence=assessment.confidence,
                uncertainty=assessment.uncertainty,
                subsystems_affected=assessment.subsystems_affected,
                crosses_boundaries=assessment.crosses_boundaries,
                fallback_eligible=False,
                fallback_tier=None,
                requires_human_approval=True,
                budget_permitted=True,
                metadata=metadata,
            )
            self.routing_history.append(decision)
            return decision

        # Case 2: Assessment determined Tier 1 (Local)
        if assessment.assessed_tier == RoutingTier.TIER_1_LOCAL:
            local_ok, local_reason = self.check_local_available()
            if local_ok:
                decision = RoutingDecision(
                    task_id=task.task_id,
                    selected_tier=RoutingTier.TIER_1_LOCAL,
                    adapter_name="local",
                    adapter_config={"model": self.config.local_model, "base_url": "http://127.0.0.1:11434"},
                    is_available=True,
                    reasons=reasons + [local_reason],
                    confidence=assessment.confidence,
                    uncertainty=assessment.uncertainty,
                    subsystems_affected=assessment.subsystems_affected,
                    crosses_boundaries=assessment.crosses_boundaries,
                    fallback_eligible=self.config.allow_fallback_to_cloud,
                    fallback_tier=RoutingTier.TIER_2_EFFICIENT_CLOUD if self.config.allow_fallback_to_cloud else None,
                    requires_human_approval=False,
                    budget_permitted=True,
                    metadata=metadata,
                )
                self.routing_history.append(decision)
                return decision

            # Local is unavailable: evaluate safe escalation to Tier 2
            reasons.append(f"Tier 1 (local) unavailable: {local_reason}")
            cloud_ok, cloud_reason = self.check_cloud_available(authorized)
            if cloud_ok and self.config.allow_fallback_to_cloud:
                reasons.append(f"Safely escalated from unavailable Tier 1 to Tier 2 efficient cloud: {cloud_reason}")
                decision = RoutingDecision(
                    task_id=task.task_id,
                    selected_tier=RoutingTier.TIER_2_EFFICIENT_CLOUD,
                    adapter_name="antigravity",
                    adapter_config={"mode": "accept-edits", "agent": self.config.efficient_cloud_agent},
                    is_available=True,
                    reasons=reasons,
                    confidence=max(0.1, assessment.confidence - 0.1),
                    uncertainty=min(1.0, assessment.uncertainty + 0.1),
                    subsystems_affected=assessment.subsystems_affected,
                    crosses_boundaries=assessment.crosses_boundaries,
                    fallback_eligible=self.config.allow_fallback_to_advanced,
                    fallback_tier=RoutingTier.TIER_3_ADVANCED_CLOUD,
                    requires_human_approval=False,
                    budget_permitted=True,
                    metadata=metadata,
                )
                self.routing_history.append(decision)
                return decision

            # Neither local nor cloud available for execution: fail closed to Tier 4
            reasons.append(f"Cannot escalate to cloud: {cloud_reason}. Routing safely to NEEDS_APPROVAL.")
            decision = RoutingDecision(
                task_id=task.task_id,
                selected_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                adapter_name="human_approval",
                adapter_config={"action": "NEEDS_APPROVAL"},
                is_available=False,
                reasons=reasons,
                confidence=assessment.confidence,
                uncertainty=1.0,
                subsystems_affected=assessment.subsystems_affected,
                crosses_boundaries=assessment.crosses_boundaries,
                fallback_eligible=False,
                fallback_tier=None,
                requires_human_approval=True,
                budget_permitted=False,
                metadata=metadata,
            )
            self.routing_history.append(decision)
            return decision

        # Case 3: Assessment determined Tier 2 (Efficient Cloud)
        if assessment.assessed_tier == RoutingTier.TIER_2_EFFICIENT_CLOUD:
            cloud_ok, cloud_reason = self.check_cloud_available(authorized)
            if cloud_ok:
                decision = RoutingDecision(
                    task_id=task.task_id,
                    selected_tier=RoutingTier.TIER_2_EFFICIENT_CLOUD,
                    adapter_name="antigravity",
                    adapter_config={"mode": "accept-edits", "agent": self.config.efficient_cloud_agent},
                    is_available=True,
                    reasons=reasons + [cloud_reason],
                    confidence=assessment.confidence,
                    uncertainty=assessment.uncertainty,
                    subsystems_affected=assessment.subsystems_affected,
                    crosses_boundaries=assessment.crosses_boundaries,
                    fallback_eligible=self.config.allow_fallback_to_advanced,
                    fallback_tier=RoutingTier.TIER_3_ADVANCED_CLOUD,
                    requires_human_approval=False,
                    budget_permitted=True,
                    metadata=metadata,
                )
                self.routing_history.append(decision)
                return decision

            # Cloud unavailable for Tier 2: NEVER silently downgrade a complex task to local!
            reasons.append(f"Tier 2 (efficient cloud) unavailable: {cloud_reason}. Failing closed to human approval.")
            decision = RoutingDecision(
                task_id=task.task_id,
                selected_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                adapter_name="human_approval",
                adapter_config={"action": "NEEDS_APPROVAL"},
                is_available=False,
                reasons=reasons,
                confidence=assessment.confidence,
                uncertainty=1.0,
                subsystems_affected=assessment.subsystems_affected,
                crosses_boundaries=assessment.crosses_boundaries,
                fallback_eligible=False,
                fallback_tier=None,
                requires_human_approval=True,
                budget_permitted=self.config.has_cloud_budget(),
                metadata=metadata,
            )
            self.routing_history.append(decision)
            return decision

        # Case 4: Assessment determined Tier 3 (Advanced Cloud)
        if assessment.assessed_tier == RoutingTier.TIER_3_ADVANCED_CLOUD:
            cloud_ok, cloud_reason = self.check_cloud_available(authorized)
            if cloud_ok:
                decision = RoutingDecision(
                    task_id=task.task_id,
                    selected_tier=RoutingTier.TIER_3_ADVANCED_CLOUD,
                    adapter_name="antigravity",
                    adapter_config={"mode": "accept-edits", "agent": self.config.advanced_cloud_agent},
                    is_available=True,
                    reasons=reasons + [cloud_reason],
                    confidence=assessment.confidence,
                    uncertainty=assessment.uncertainty,
                    subsystems_affected=assessment.subsystems_affected,
                    crosses_boundaries=assessment.crosses_boundaries,
                    fallback_eligible=False,
                    fallback_tier=None,
                    requires_human_approval=False,
                    budget_permitted=True,
                    metadata=metadata,
                )
                self.routing_history.append(decision)
                return decision

            reasons.append(f"Tier 3 (advanced cloud) unavailable: {cloud_reason}. Routing safely to human approval.")
            decision = RoutingDecision(
                task_id=task.task_id,
                selected_tier=RoutingTier.TIER_4_HUMAN_DECISION,
                adapter_name="human_approval",
                adapter_config={"action": "NEEDS_APPROVAL"},
                is_available=False,
                reasons=reasons,
                confidence=assessment.confidence,
                uncertainty=1.0,
                subsystems_affected=assessment.subsystems_affected,
                crosses_boundaries=assessment.crosses_boundaries,
                fallback_eligible=False,
                fallback_tier=None,
                requires_human_approval=True,
                budget_permitted=self.config.has_cloud_budget(),
                metadata=metadata,
            )
            self.routing_history.append(decision)
            return decision

        # Default fallback: safe fail-closed
        decision = RoutingDecision(
            task_id=task.task_id,
            selected_tier=RoutingTier.TIER_4_HUMAN_DECISION,
            adapter_name="human_approval",
            adapter_config={"action": "NEEDS_APPROVAL"},
            is_available=False,
            reasons=reasons + ["Unknown tier state: failed closed."],
            confidence=0.0,
            uncertainty=1.0,
            subsystems_affected=[],
            crosses_boundaries=False,
            fallback_eligible=False,
            fallback_tier=None,
            requires_human_approval=True,
            budget_permitted=False,
            metadata=metadata,
        )
        self.routing_history.append(decision)
        return decision

    def evaluate_fallback(
        self,
        task: TaskDefinition,
        current_decision: RoutingDecision,
        agent_result: AgentExecutionResult,
        authorized: bool = False,
    ) -> FallbackDecision:
        """
        Evaluate if an adapter execution failure qualifies for bounded automatic fallback.
        Strictly forbids fallback on security, path traversal, authorization, or attempt limit violations.
        """
        task_id = task.task_id
        original_tier = current_decision.selected_tier

        # Non-qualifying failure 1: Permission denial or unauthorized execution attempt
        if agent_result.permission_denied:
            fb = FallbackDecision(
                qualifies=False,
                task_id=task_id,
                original_tier=original_tier,
                fallback_tier=None,
                fallback_adapter=None,
                reason="Permission denied or sandbox violation during execution.",
                non_qualifying_reason="SECURITY_VIOLATION_PERMISSION_DENIED",
                requires_human_approval=True,
            )
            self.fallback_history.append(fb)
            return fb

        # Non-qualifying failure 2: Max attempt budget reached
        if task.attempts >= task.max_attempts:
            fb = FallbackDecision(
                qualifies=False,
                task_id=task_id,
                original_tier=original_tier,
                fallback_tier=None,
                fallback_adapter=None,
                reason=f"Task reached maximum attempts limit ({task.max_attempts}).",
                non_qualifying_reason="MAX_ATTEMPTS_EXHAUSTED",
                requires_human_approval=True,
            )
            self.fallback_history.append(fb)
            return fb

        # Non-qualifying failure 3: Fallback disabled by decision or tier limit
        if not current_decision.fallback_eligible or not current_decision.fallback_tier:
            fb = FallbackDecision(
                qualifies=False,
                task_id=task_id,
                original_tier=original_tier,
                fallback_tier=None,
                fallback_adapter=None,
                reason="Current tier is not eligible for automatic fallback.",
                non_qualifying_reason="TIER_NOT_FALLBACK_ELIGIBLE",
                requires_human_approval=True,
            )
            self.fallback_history.append(fb)
            return fb

        # Non-qualifying failure 4: High or Very High risk tasks never auto-fallback
        if task.risk in (TaskRisk.HIGH, TaskRisk.VERY_HIGH):
            fb = FallbackDecision(
                qualifies=False,
                task_id=task_id,
                original_tier=original_tier,
                fallback_tier=None,
                fallback_adapter=None,
                reason=f"{task.risk.value}-risk task cannot auto-fallback; requires human review.",
                non_qualifying_reason="HIGH_RISK_POLICY",
                requires_human_approval=True,
            )
            self.fallback_history.append(fb)
            return fb

        # Check target fallback tier
        target_tier = current_decision.fallback_tier

        # Qualifying check: Tier 1 -> Tier 2 fallback
        if original_tier == RoutingTier.TIER_1_LOCAL and target_tier == RoutingTier.TIER_2_EFFICIENT_CLOUD:
            # Qualifying conditions: syntax parsing failure, output formatting failure, or recoverable timeout
            err = (agent_result.error_message or "").lower()
            is_parse_error = "failed to parse" in err or "parseable file blocks" in err
            is_timeout = agent_result.timed_out
            is_generic_fail = (not agent_result.success) and (agent_result.exit_code != 0)

            if is_parse_error or is_timeout or is_generic_fail:
                # Check cloud availability and authorization
                cloud_ok, cloud_reason = self.check_cloud_available(authorized)
                if cloud_ok:
                    fb = FallbackDecision(
                        qualifies=True,
                        task_id=task_id,
                        original_tier=original_tier,
                        fallback_tier=RoutingTier.TIER_2_EFFICIENT_CLOUD,
                        fallback_adapter="antigravity",
                        reason=f"Qualifying local model failure ({err or 'execution failure'}); falling back to Tier 2.",
                    )
                    self.fallback_history.append(fb)
                    return fb
                else:
                    fb = FallbackDecision(
                        qualifies=False,
                        task_id=task_id,
                        original_tier=original_tier,
                        fallback_tier=None,
                        fallback_adapter=None,
                        reason=f"Local failure qualified for fallback, but cloud adapter is unavailable: {cloud_reason}",
                        non_qualifying_reason="CLOUD_UNAVAILABLE",
                        requires_human_approval=True,
                    )
                    self.fallback_history.append(fb)
                    return fb

        # Qualifying check: Tier 2 -> Tier 3 fallback
        if original_tier == RoutingTier.TIER_2_EFFICIENT_CLOUD and target_tier == RoutingTier.TIER_3_ADVANCED_CLOUD:
            cloud_ok, cloud_reason = self.check_cloud_available(authorized)
            if cloud_ok and self.config.allow_fallback_to_advanced:
                fb = FallbackDecision(
                    qualifies=True,
                    task_id=task_id,
                    original_tier=original_tier,
                    fallback_tier=RoutingTier.TIER_3_ADVANCED_CLOUD,
                    fallback_adapter="antigravity",
                    reason=f"Qualifying Tier 2 failure; falling back to Tier 3 advanced reasoning.",
                )
                self.fallback_history.append(fb)
                return fb
            else:
                fb = FallbackDecision(
                    qualifies=False,
                    task_id=task_id,
                    original_tier=original_tier,
                    fallback_tier=None,
                    fallback_adapter=None,
                    reason=f"Tier 2 failure qualified for fallback, but Tier 3 is unavailable: {cloud_reason}",
                    non_qualifying_reason="ADVANCED_TIER_UNAVAILABLE",
                    requires_human_approval=True,
                )
                self.fallback_history.append(fb)
                return fb

        # Default non-qualifying
        fb = FallbackDecision(
            qualifies=False,
            task_id=task_id,
            original_tier=original_tier,
            fallback_tier=None,
            fallback_adapter=None,
            reason="Failure mode does not qualify for automatic fallback.",
            non_qualifying_reason="NON_QUALIFYING_ERROR",
            requires_human_approval=True,
        )
        self.fallback_history.append(fb)
        return fb


class WorkerConcurrencyGuard:
    """
    Enforces in-process single-implementation-worker concurrency and safe isolated
    review concurrency invariant.

    Note on Concurrency Scope:
    This guard enforces in-process mutual exclusion across threads and tasks within
    a single coordinator/orchestrator runtime session. It is not an OS-level or
    cross-process distributed lock, nor a security sandbox. Cross-process isolation
    is maintained by the coordinator's single supervisory runtime with atomic state locks.
    """

    def __init__(self, max_implementation_workers: int = 1) -> None:
        self.max_implementation_workers = max_implementation_workers
        self.active_implementation_tasks: set[str] = set()
        self.active_review_tasks: set[str] = set()

    def acquire_implementation_slot(self, task_id: str) -> bool:
        if len(self.active_implementation_tasks) >= self.max_implementation_workers:
            return False
        # Do not allow implementation writer if an active review is reading this worktree
        if task_id in self.active_review_tasks:
            return False
        self.active_implementation_tasks.add(task_id)
        return True

    def release_implementation_slot(self, task_id: str) -> None:
        self.active_implementation_tasks.discard(task_id)

    def acquire_review_slot(self, task_id: str, allow_concurrent: bool = False) -> bool:
        if not allow_concurrent and len(self.active_implementation_tasks) > 0:
            # When concurrent review is disabled, cannot run review while implementation is active
            return False
        # Reviewer cannot inspect a task currently undergoing implementation edits in the same worktree
        if task_id in self.active_implementation_tasks:
            return False
        self.active_review_tasks.add(task_id)
        return True

    def release_review_slot(self, task_id: str) -> None:
        self.active_review_tasks.discard(task_id)
