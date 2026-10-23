"""
Provider-Neutral Model Registry & Preflight Health Probing for MayaVE Autophage (P4).
Defines typed model/provider descriptors, capabilities, availability statuses,
pricing metadata, and preflight health probes for local and external runtimes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    from .adapter import resolve_antigravity_executable
    from .routing import RoutingTier
except ImportError:
    from adapter import resolve_antigravity_executable
    from routing import RoutingTier

logger = logging.getLogger("maya.maintenance.registry")


class ProviderType(str, Enum):
    OLLAMA = "ollama"
    ANTIGRAVITY = "antigravity"
    CLOUD_REST = "cloud_rest"
    MOCK = "mock"


class ModelCapability(str, Enum):
    UNIFIED_DIFF = "unified_diff"
    MULTI_FILE_BLOCK = "multi_file_block"
    TOOL_CALLING = "tool_calling"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelDescriptor:
    """Descriptor defining model identity, provider runtime, limits, and capabilities."""
    model_id: str
    provider: ProviderType
    tier: RoutingTier
    context_window: int
    is_local: bool
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    capabilities: tuple[ModelCapability, ...]
    requires_auth: bool
    status: ModelStatus = ModelStatus.UNKNOWN
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider.value,
            "tier": self.tier.value,
            "context_window": self.context_window,
            "is_local": self.is_local,
            "cost_per_1k_input_tokens": self.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": self.cost_per_1k_output_tokens,
            "capabilities": [c.value for c in self.capabilities],
            "requires_auth": self.requires_auth,
            "status": self.status.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class PreflightProbeResult:
    """Result of an active runtime health and availability probe."""
    provider: ProviderType
    endpoint_or_target: str
    reachable: bool
    available_models: tuple[str, ...] = ()
    latency_ms: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "endpoint_or_target": self.endpoint_or_target,
            "reachable": self.reachable,
            "available_models": list(self.available_models),
            "latency_ms": round(self.latency_ms, 2),
            "error_message": self.error_message,
        }


class ModelRegistry:
    """
    Central repository of verified model descriptors, pricing rules,
    and runtime health probes. Fail-closed: missing models or unreachable
    endpoints are marked UNAVAILABLE or NOT_CONFIGURED.
    """

    def __init__(self, populate_defaults: bool = True) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}
        if populate_defaults:
            self._load_default_descriptors()

    def _load_default_descriptors(self) -> None:
        defaults = [
            # Tier 1: Local verified coder model (Ollama)
            ModelDescriptor(
                model_id="qwen2.5-coder:3b",
                provider=ProviderType.OLLAMA,
                tier=RoutingTier.TIER_1_LOCAL,
                context_window=32768,
                is_local=True,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                capabilities=(
                    ModelCapability.UNIFIED_DIFF,
                    ModelCapability.MULTI_FILE_BLOCK,
                    ModelCapability.STRUCTURED_OUTPUT,
                ),
                requires_auth=False,
                status=ModelStatus.UNKNOWN,
                description="Default Tier 1 local model for low-risk code repair.",
            ),
            # Legacy Tier 1 fallback (Ollama)
            ModelDescriptor(
                model_id="llama3.2",
                provider=ProviderType.OLLAMA,
                tier=RoutingTier.TIER_1_LOCAL,
                context_window=8192,
                is_local=True,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                capabilities=(ModelCapability.MULTI_FILE_BLOCK,),
                requires_auth=False,
                status=ModelStatus.UNKNOWN,
                description="Legacy local model preserved for backwards compatibility.",
            ),
            # Planner target (Metadata & contract only; no runtime loop in P4)
            ModelDescriptor(
                model_id="qwen3.5:4b",
                provider=ProviderType.OLLAMA,
                tier=RoutingTier.TIER_1_LOCAL,
                context_window=32768,
                is_local=True,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                capabilities=(
                    ModelCapability.REASONING,
                    ModelCapability.STRUCTURED_OUTPUT,
                ),
                requires_auth=False,
                status=ModelStatus.UNKNOWN,
                description="Target Planner model for task intake and contract generation (deferred to P5).",
            ),
            # Tier 2 Cloud Candidate A (Requires benchmark evaluation)
            ModelDescriptor(
                model_id="glm-5.3-flash",
                provider=ProviderType.CLOUD_REST,
                tier=RoutingTier.TIER_2_EFFICIENT_CLOUD,
                context_window=128000,
                is_local=False,
                cost_per_1k_input_tokens=0.0001,
                cost_per_1k_output_tokens=0.0002,
                capabilities=(
                    ModelCapability.UNIFIED_DIFF,
                    ModelCapability.MULTI_FILE_BLOCK,
                    ModelCapability.STRUCTURED_OUTPUT,
                ),
                requires_auth=True,
                status=ModelStatus.NOT_CONFIGURED,
                description="Tier 2 candidate: High-throughput cloud model evaluated via offline benchmark.",
            ),
            # Tier 2 Cloud Candidate B (Requires benchmark evaluation)
            ModelDescriptor(
                model_id="gemma-4-31b",
                provider=ProviderType.CLOUD_REST,
                tier=RoutingTier.TIER_2_EFFICIENT_CLOUD,
                context_window=128000,
                is_local=False,
                cost_per_1k_input_tokens=0.0002,
                cost_per_1k_output_tokens=0.0004,
                capabilities=(
                    ModelCapability.UNIFIED_DIFF,
                    ModelCapability.MULTI_FILE_BLOCK,
                    ModelCapability.STRUCTURED_OUTPUT,
                ),
                requires_auth=True,
                status=ModelStatus.NOT_CONFIGURED,
                description="Tier 2 candidate: Large cloud model evaluated via offline benchmark.",
            ),
            # Tier 3 Advanced Cloud: Antigravity CLI (Google Flash 3.8)
            ModelDescriptor(
                model_id="google-flash-3.8",
                provider=ProviderType.ANTIGRAVITY,
                tier=RoutingTier.TIER_3_ADVANCED_CLOUD,
                context_window=1048576,
                is_local=False,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                capabilities=(
                    ModelCapability.REASONING,
                    ModelCapability.TOOL_CALLING,
                    ModelCapability.UNIFIED_DIFF,
                ),
                requires_auth=True,
                status=ModelStatus.UNKNOWN,
                description="Tier 3 advanced reasoning via authorized Antigravity CLI supervisor.",
            ),
        ]
        for d in defaults:
            self.register(d)

    def register(self, descriptor: ModelDescriptor) -> None:
        """Register or update a model descriptor."""
        self._descriptors[descriptor.model_id] = descriptor

    def get(self, model_id: str) -> Optional[ModelDescriptor]:
        """Retrieve descriptor by model_id."""
        return self._descriptors.get(model_id)

    def list_models(
        self,
        tier: Optional[RoutingTier] = None,
        provider: Optional[ProviderType] = None,
    ) -> list[ModelDescriptor]:
        """List registered models filtered by tier and/or provider."""
        results = list(self._descriptors.values())
        if tier is not None:
            results = [m for m in results if m.tier == tier]
        if provider is not None:
            results = [m for m in results if m.provider == provider]
        return results

    def probe_ollama(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 1.5,
    ) -> PreflightProbeResult:
        """
        Probe local Ollama daemon for endpoint reachability and installed models.
        Updates status of matching OLLAMA descriptors in the registry.
        Does not download models or throw unhandled network exceptions.
        """
        url = f"{base_url.rstrip('/')}/api/tags"
        start = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": "MayaVE-Autophage-Preflight/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency = (time.monotonic() - start) * 1000.0
                raw_models = data.get("models", [])
                installed_names: list[str] = []
                for m in raw_models:
                    name = m.get("name", "")
                    if name:
                        installed_names.append(name)
                        # Also register short tag without ':latest'
                        if name.endswith(":latest"):
                            installed_names.append(name[:-7])

                installed_tuple = tuple(sorted(set(installed_names)))

                # Update registry statuses
                for desc in list(self._descriptors.values()):
                    if desc.provider == ProviderType.OLLAMA:
                        is_installed = any(
                            desc.model_id == name or name.startswith(f"{desc.model_id}:")
                            for name in installed_tuple
                        )
                        new_status = ModelStatus.AVAILABLE if is_installed else ModelStatus.UNAVAILABLE
                        self._descriptors[desc.model_id] = ModelDescriptor(
                            model_id=desc.model_id,
                            provider=desc.provider,
                            tier=desc.tier,
                            context_window=desc.context_window,
                            is_local=desc.is_local,
                            cost_per_1k_input_tokens=desc.cost_per_1k_input_tokens,
                            cost_per_1k_output_tokens=desc.cost_per_1k_output_tokens,
                            capabilities=desc.capabilities,
                            requires_auth=desc.requires_auth,
                            status=new_status,
                            description=desc.description,
                        )

                return PreflightProbeResult(
                    provider=ProviderType.OLLAMA,
                    endpoint_or_target=base_url,
                    reachable=True,
                    available_models=installed_tuple,
                    latency_ms=latency,
                )

        except urllib.error.URLError as e:
            latency = (time.monotonic() - start) * 1000.0
            err_msg = f"Ollama connection error at {url}: {e.reason}"
            self._mark_ollama_unavailable()
            return PreflightProbeResult(
                provider=ProviderType.OLLAMA,
                endpoint_or_target=base_url,
                reachable=False,
                latency_ms=latency,
                error_message=err_msg,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000.0
            err_msg = f"Ollama probe failed: {e}"
            self._mark_ollama_unavailable()
            return PreflightProbeResult(
                provider=ProviderType.OLLAMA,
                endpoint_or_target=base_url,
                reachable=False,
                latency_ms=latency,
                error_message=err_msg,
            )

    def _mark_ollama_unavailable(self) -> None:
        """Mark all Ollama descriptors as UNAVAILABLE."""
        for desc in list(self._descriptors.values()):
            if desc.provider == ProviderType.OLLAMA:
                self._descriptors[desc.model_id] = ModelDescriptor(
                    model_id=desc.model_id,
                    provider=desc.provider,
                    tier=desc.tier,
                    context_window=desc.context_window,
                    is_local=desc.is_local,
                    cost_per_1k_input_tokens=desc.cost_per_1k_input_tokens,
                    cost_per_1k_output_tokens=desc.cost_per_1k_output_tokens,
                    capabilities=desc.capabilities,
                    requires_auth=desc.requires_auth,
                    status=ModelStatus.UNAVAILABLE,
                    description=desc.description,
                )

    def probe_antigravity(self, executable: str | Path | None = None) -> PreflightProbeResult:
        """Probe presence of Antigravity CLI binary."""
        start = time.monotonic()
        try:
            resolved = resolve_antigravity_executable(executable)
            is_present = bool(shutil.which(str(resolved)) or Path(resolved).is_file())
            latency = (time.monotonic() - start) * 1000.0

            status = ModelStatus.AVAILABLE if is_present else ModelStatus.UNAVAILABLE
            if "google-flash-3.8" in self._descriptors:
                d = self._descriptors["google-flash-3.8"]
                self._descriptors["google-flash-3.8"] = ModelDescriptor(
                    model_id=d.model_id,
                    provider=d.provider,
                    tier=d.tier,
                    context_window=d.context_window,
                    is_local=d.is_local,
                    cost_per_1k_input_tokens=d.cost_per_1k_input_tokens,
                    cost_per_1k_output_tokens=d.cost_per_1k_output_tokens,
                    capabilities=d.capabilities,
                    requires_auth=d.requires_auth,
                    status=status,
                    description=d.description,
                )

            return PreflightProbeResult(
                provider=ProviderType.ANTIGRAVITY,
                endpoint_or_target=str(resolved),
                reachable=is_present,
                available_models=("google-flash-3.8",) if is_present else (),
                latency_ms=latency,
                error_message=None if is_present else f"Antigravity executable '{resolved}' not found.",
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000.0
            return PreflightProbeResult(
                provider=ProviderType.ANTIGRAVITY,
                endpoint_or_target=str(executable or "agy.exe"),
                reachable=False,
                latency_ms=latency,
                error_message=f"Antigravity probe failed: {e}",
            )

    def probe_all(self, ollama_url: str = "http://127.0.0.1:11434") -> dict[ProviderType, PreflightProbeResult]:
        """Run all preflight probes and return consolidated summary."""
        return {
            ProviderType.OLLAMA: self.probe_ollama(base_url=ollama_url),
            ProviderType.ANTIGRAVITY: self.probe_antigravity(),
        }
