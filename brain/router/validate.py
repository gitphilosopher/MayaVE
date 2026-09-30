"""
brain/router/validate.py
Mandatory validation boundary between (semantic retrieval | LLM
fallback) output and the adapter/legacy router.

Every Command — regardless of source — must pass through validate()
before it can reach adapter.py. Malformed JSON, a hallucinated
domain/operation, or an invalid entity degrades to a safe outcome, never
raises, and never reaches a skill unvalidated.

BATCH 2: validate() takes `enforce_required` (default True). Semantic
retrieval selects a command BEFORE any entity is extracted, so it validates
the command's identity with enforce_required=False and extracts entities
afterwards (see hybrid_engine.py). LLM output is always validated with the
default, since the LLM is expected to supply what it claims.
"""

import logging

from brain.router.registry import CommandRegistry
from brain.router.schemas import Command, ValidationResult

logger = logging.getLogger(__name__)

_MAX_ENTITY_STR_LEN = 500


def validate_raw_llm_output(data: dict) -> Command | None:
    """
    Structural-only validation of the LLM's raw JSON (before it's
    checked against the domain registry). Returns None on any structural
    problem rather than raising.
    """
    if not isinstance(data, dict):
        return None
    domain = data.get("domain")
    operation = data.get("operation")
    entities = data.get("entities", {})
    confidence = data.get("confidence", 0.0)
    needs_clarification = data.get("needs_clarification", False)

    if not isinstance(domain, str) or not domain:
        return None
    if not isinstance(operation, str) or not operation:
        return None
    if not isinstance(entities, dict):
        entities = {}
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if not isinstance(needs_clarification, bool):
        needs_clarification = False

    return Command(
        domain=domain, operation=operation, entities=entities,
        confidence=max(0.0, min(1.0, confidence)),
        needs_clarification=needs_clarification,
    )


def validate(command: Command, registry: CommandRegistry, *, enforce_required: bool = True) -> ValidationResult:
    """
    Full validation against the real command registry:
      - domain exists
      - operation exists for that domain
      - every entity key is one the operation declares
      - every required entity is present (skipped when enforce_required=False)
      - every entity value is a plain string within a sane length

    A domain='unknown'/operation='unknown' Command validates as ok=False
    with error "not_a_command" so the caller can route it to the
    conversational LLM path.
    """
    if command.domain == "unknown" or command.operation == "unknown":
        return ValidationResult(ok=False, error="not_a_command")

    spec = registry.get(command.domain, command.operation)
    if spec is None:
        return ValidationResult(ok=False, error=f"unknown domain/operation: {command.domain}.{command.operation}")

    if command.needs_clarification:
        return ValidationResult(ok=False, error="needs_clarification")

    allowed_keys = set(spec.entities.keys())
    for key, value in command.entities.items():
        if key not in allowed_keys:
            return ValidationResult(ok=False, error=f"unexpected entity '{key}' for {spec.key}")
        if not isinstance(value, str):
            return ValidationResult(ok=False, error=f"entity '{key}' must be a string, got {type(value).__name__}")
        if len(value) > _MAX_ENTITY_STR_LEN:
            return ValidationResult(ok=False, error=f"entity '{key}' exceeds {_MAX_ENTITY_STR_LEN} characters")

    if enforce_required:
        for name, meta in spec.entities.items():
            if meta.get("required") and not (command.entities.get(name) or "").strip():
                return ValidationResult(ok=False, error=f"missing required entity '{name}' for {spec.key}")

    return ValidationResult(ok=True, spec=spec)