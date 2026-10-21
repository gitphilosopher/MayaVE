"""
brain/router/registry.py
Loads config/command_domains.json into CommandSpec objects and exposes
the domain/operation registry used by validate.py and the LLM fallback
prompt (validate.py trusts only what's registered here — never anything
the LLM returns).

Semantic prototypes live in each operation's `seeds` list. Seeds are
validated and de-duplicated here; the same seed text under two different
operations is rejected — it would make the two commands indistinguishable.

BATCH 1: each operation may declare `"requires_confirmation": true`
(bool, default false). Previously the key was never read, so every spec —
including system.shutdown/restart — loaded as False. It is metadata for
the IR; the confirmation itself is still enforced by the skills
(skills/system/power.py, skills/utilities/notepad.py).
"""

import json
import logging
from pathlib import Path

from brain.router.schemas import CommandSpec

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "config" / "command_domains.json"

LEGACY_BUILTIN_INTENTS: frozenset[str] = frozenset({"greet", "farewell", "thanks", "help"})
LEGACY_SKILL_INTENTS: frozenset[str] = frozenset({"mute", "perform_action"})
ALLOWED_LEGACY_INTENTS: frozenset[str] = LEGACY_BUILTIN_INTENTS | LEGACY_SKILL_INTENTS


class CommandRegistryError(ValueError):
    """Raised for a malformed config/command_domains.json."""


def _normalize_seed(seed: str) -> str:
    return " ".join(seed.lower().split())


def load_specs(path: Path = _DEFAULT_PATH) -> list[CommandSpec]:
    """Load and structurally validate the command-domain registry.
    Raises CommandRegistryError on malformed input rather than degrading
    silently — a broken taxonomy must fail loudly at startup."""
    if not path.exists():
        raise CommandRegistryError(f"{path} does not exist.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CommandRegistryError(f"{path} is not valid JSON: {e}") from e

    domains = data.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise CommandRegistryError(f"{path}: 'domains' must be a non-empty object.")

    specs: list[CommandSpec] = []
    seed_owner: dict[str, str] = {}   # normalized seed -> owning domain.operation
    legacy_owner: dict[str, str] = {} # legacy_intent -> owning domain.operation
    for domain, dspec in domains.items():
        operations = dspec.get("operations") if isinstance(dspec, dict) else None
        if not isinstance(operations, dict) or not operations:
            raise CommandRegistryError(f"{path}: domain '{domain}' has no operations.")
        for operation, ospec in operations.items():
            key = f"{domain}.{operation}"
            if not isinstance(ospec, dict):
                raise CommandRegistryError(f"{path}: {key} must be an object.")
            legacy_intent = ospec.get("legacy_intent")
            if not isinstance(legacy_intent, str) or not legacy_intent:
                raise CommandRegistryError(f"{path}: {key} missing 'legacy_intent'.")
            if legacy_intent in legacy_owner:
                raise CommandRegistryError(
                    f"{path}: legacy_intent '{legacy_intent}' appears under both {legacy_owner[legacy_intent]} and {key}."
                )
            legacy_owner[legacy_intent] = key
            entities = ospec.get("entities", {})
            if not isinstance(entities, dict):
                raise CommandRegistryError(f"{path}: {key}.entities must be an object.")
            target_mode = ospec.get("target_mode", "raw")
            requires_confirmation = ospec.get("requires_confirmation", False)
            if not isinstance(requires_confirmation, bool):
                raise CommandRegistryError(f"{path}: {key}.requires_confirmation must be a boolean.")

            raw_seeds = ospec.get("seeds", [])
            if not isinstance(raw_seeds, list) or not all(isinstance(s, str) and s.strip() for s in raw_seeds):
                raise CommandRegistryError(f"{path}: {key}.seeds must be a list of non-empty strings.")
            seeds: list[str] = []
            for s in raw_seeds:
                norm = _normalize_seed(s)
                owner = seed_owner.get(norm)
                if owner == key:
                    continue   # exact repeat within one operation — dropped, order preserved
                if owner is not None:
                    raise CommandRegistryError(
                        f"{path}: seed '{s}' appears under both {owner} and {key}."
                    )
                seed_owner[norm] = key
                seeds.append(s.strip())

            specs.append(CommandSpec(
                domain=domain, operation=operation, legacy_intent=legacy_intent,
                entities=entities, target_mode=target_mode, seeds=tuple(seeds),
                requires_confirmation=requires_confirmation,
            ))

    logger.info(f"Command domain registry loaded — {len(specs)} operation(s) across {len(domains)} domain(s).")
    return specs


class CommandRegistry:
    """In-memory lookup used by validate.py and the LLM fallback prompt builder."""

    def __init__(self, specs: list[CommandSpec]):
        self._specs = specs
        self._by_key: dict[str, CommandSpec] = {s.key: s for s in specs}
        self._by_legacy: dict[str, CommandSpec] = {s.legacy_intent: s for s in specs}
        self._domains: dict[str, set[str]] = {}
        for s in specs:
            self._domains.setdefault(s.domain, set()).add(s.operation)

    @property
    def specs(self) -> list[CommandSpec]:
        return list(self._specs)

    @property
    def by_key(self) -> dict[str, CommandSpec]:
        return dict(self._by_key)

    @property
    def by_legacy(self) -> dict[str, CommandSpec]:
        return dict(self._by_legacy)

    def get(self, domain: str, operation: str) -> CommandSpec | None:
        return self._by_key.get(f"{domain}.{operation}")

    def get_by_legacy(self, legacy_intent: str) -> CommandSpec | None:
        return self._by_legacy.get(legacy_intent)

    def domains(self) -> list[str]:
        return sorted(self._domains.keys())

    def operations_for(self, domain: str) -> list[str]:
        return sorted(self._domains.get(domain, set()))

    def is_authorized(
        self,
        domain: str | None,
        operation: str | None,
        legacy_intent: str | None,
        requires_confirmation: bool = False,
    ) -> bool:
        """
        Check whether an operation and legacy intent are authorized by the registry
        or explicit compatibility policy.
        """
        if not legacy_intent:
            return False

        # Case 1: Canonical registered operation
        spec = self._by_legacy.get(legacy_intent)
        if spec is not None:
            if domain != spec.domain or operation != spec.operation:
                return False
            # Prevent confirmation downgrade: if spec requires confirmation,
            # an unconfirmed command is unauthorized.
            if spec.requires_confirmation and not requires_confirmation:
                return False
            return True

        # Case 2: Explicitly approved legacy compatibility route
        if domain == "legacy" and operation in ALLOWED_LEGACY_INTENTS:
            return legacy_intent == operation and not requires_confirmation

        return False


_default_registry: CommandRegistry | None = None


def get_default_registry() -> CommandRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = CommandRegistry(load_specs())
    return _default_registry