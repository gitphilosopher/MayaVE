"""
brain/router/adapter.py
Translates a VALIDATED Command into the legacy intent dictionary shape
that brain/router/dispatch.py::Router.dispatch() and every skill already expect.

This is the interpretation layer described in the migration spec: no
skill is rewritten to understand {domain, operation, entities} — the
adapter's only job is picking the right legacy_intent (from the
CommandSpec, already resolved in validate.py) and the right
intent["target"] string.

target_mode per operation (see config/command_domains.json):
  "raw"          — pass the original utterance through unchanged. Used
                   by every skill that re-parses the raw text itself
                   rather than trusting a pre-extracted target — most
                   notably skills/utilities/timer.py, which extracts
                   its own duration/reminder message from `text` and
                   ignores intent["target"] entirely (see its
                   docstring), and skills/utilities/notepad.py, which
                   does the same for note content/name extraction.
  "entity:<name>" — use the named entity as intent["target"] verbatim
                   when present; otherwise fall back to "raw" so a
                   command the LLM under-extracted still reaches the
                   skill's own fallback clarification prompt (e.g.
                   open_target's "What would you like me to open?")
                   instead of being silently dropped.
"""

import logging

from brain.router.schemas import Command, CommandSpec

logger = logging.getLogger(__name__)


def to_legacy_intent(command: Command, spec: CommandSpec, raw_text: str, *,
                      confidence: float, model_source: str) -> dict:
    """Build the legacy-shaped dict. Confidence/model_source are passed
    in separately from `command.confidence` because the caller (the
    hybrid engine) may want to report the retrieval similarity or a
    fixed 1.0 for a validated LLM decision rather than the LLM's own
    self-reported confidence, which validate.py never trusted anyway."""
    if spec.target_mode == "raw":
        target = raw_text
    elif spec.target_mode.startswith("entity:"):
        entity_name = spec.target_mode.split(":", 1)[1]
        value = (command.entities.get(entity_name) or "").strip()
        target = value if value else raw_text
    else:
        logger.warning(f"Unknown target_mode '{spec.target_mode}' for {spec.key} — falling back to raw text.")
        target = raw_text

    return {
        "intent": spec.legacy_intent,
        "target": target,
        "confidence": round(float(confidence), 3),
        "raw": raw_text,
        "model": model_source,
        "response_mode": "skill",
        "_command": {
            "domain": command.domain,
            "operation": command.operation,
            "entities": dict(command.entities),
            "source": model_source,
        },
    }
