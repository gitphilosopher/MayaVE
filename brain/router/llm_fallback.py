"""
brain/router/llm_fallback.py
==========================
Structured Ollama fallback for command routing.

This module is a small, dedicated router-side LLM client. It intentionally does
not reuse the conversational TTS-oriented service layer because routing needs a
single blocking JSON response with strict validation semantics, not streaming
chat behavior, persona handling, or audio generation.

Its responsibility is narrow and defensive: build a command-routing prompt from
registry metadata, call the local Ollama model, parse the response as JSON, and
return a plain dict or None on any failure. The returned payload is never trusted
as-is; callers must validate it through ``brain.router.validate`` before turning it
into a command or dispatch decision.

The module is a safety net for uncertain or unresolved commands, not the primary
routing mechanism. It is designed to fail closed: malformed responses, timeouts,
connection issues, and invalid JSON all degrade to ``None`` rather than raising an
exception back to the caller.
"""

import functools
import json
import logging
import time

import httpx

from config.settings import config
from services.llm.ollama_lifecycle import chat_keep_alive

logger = logging.getLogger(__name__)

_ROUTE_TIMEOUT_S = 8.0

_RESPONSE_SCHEMA_HINT = (
    '{"domain": "string", "operation": "string", "entities": {}, '
    '"confidence": 0.0, "needs_clarification": false}'
)

_OP_DESCRIPTIONS: dict[str, str] = {
    "datetime.time": "Check or report the current clock time.",
    "datetime.date": "Check or report the current calendar date, day, month, or year.",
    "timer.create": "Set a new active countdown timer or reminder to alert the user.",
    "timer.cancel": "Cancel or stop an active countdown timer or reminder.",
    "timer.status": "Check remaining time or status of currently running timers.",
    "notes.create": "Create and save a new written note or memo in the notepad.",
    "notes.view": "View, open, or list saved notes in the notepad.",
    "notes.delete": "Delete or discard a saved note from the notepad.",
    "app_or_web.open": "Launch, open, or navigate to a specific desktop application or website URL.",
    "web.search": "Search the web or Google for a new search query requested by the user right now.",
    "weather.current": (
        "Check current weather conditions, outdoor conditions, or forecasts "
        "(covers both explicitly specified locations and local outdoor weather; location is optional). "
        "Does NOT cover non-weather physical activity, events, or vision (e.g. 'what is happening outside', "
        "'look outside', 'who is outside', or 'what is going on outside')."
    ),
    "media.play": "Start or resume local media or music playback.",
    "media.pause": "Pause or stop currently playing media or music.",
    "media.next": "Skip to the next media track.",
    "media.previous": "Return to the previous media track.",
    "media.volume_up": "Increase computer audio volume.",
    "media.volume_down": "Decrease computer audio volume.",
    "system.info": "Report computer hardware specs, CPU, RAM, battery, or disk status.",
    "system.screenshot": "Capture a screenshot of the computer display.",
    "system.lock": "Lock the host computer workstation or screen only (NOT physical doors, premises, or locks).",
    "system.shutdown": "Shut down or power off the host computer/PC only (NOT lights, appliances, TVs, or external devices).",
    "system.restart": "Reboot or restart the host computer/PC only.",
    "clipboard.read": "Read or display text currently stored on the computer clipboard.",
    "clipboard.copy": "Copy specified text to the computer clipboard.",
    "clipboard.clear": "Wipe or clear the computer clipboard contents.",
}


@functools.lru_cache(maxsize=1)
def _entity_keys_by_operation() -> dict[str, tuple[str, ...]]:
    """'domain.operation' -> allowed entity keys, read from the same
    registry (config/command_domains.json) validate.py checks against.
    Returns {} if the registry can't be loaded, in which case the prompt
    simply omits the entity schema section (previous behavior); never raises."""
    try:
        from brain.router.registry import load_specs
        return {spec.key: tuple(spec.entities.keys()) for spec in load_specs()}
    except Exception as e:
        logger.warning(f"Router LLM fallback: entity schema unavailable for prompt: {e}")
        return {}


def _build_prompt(utterance: str, domains: list[str], operations_by_domain: dict[str, list[str]],
                   top_candidates: list[str]) -> str:
    """Construct the strict routing instruction prompt sent to the local Ollama model."""
    domain_lines = "\n".join(
        f"- {d}: {', '.join(operations_by_domain.get(d, []))}" for d in domains
    )

    op_desc_lines = "\n".join(
        f"- {d}.{op}: {_OP_DESCRIPTIONS.get(f'{d}.{op}', 'Supported operation.')}"
        for d in domains
        for op in operations_by_domain.get(d, [])
    )

    entity_keys = _entity_keys_by_operation()
    entity_lines = "\n".join(
        f"- {d}.{op}: entities {{{', '.join(entity_keys[f'{d}.{op}'])}}}"
        for d in domains
        for op in operations_by_domain.get(d, [])
        if f"{d}.{op}" in entity_keys
    )
    entity_section = (
        "Allowed entity keys per operation (an empty {} means the operation takes no entities):\n"
        f"{entity_lines}\n\n"
        "Entity keys are schema-defined. Never invent entity keys from words in the user's "
        "request. Use only the entity keys allowed for the selected operation, and place the "
        "extracted value under the appropriate key. If the operation allows no entities, "
        "return \"entities\": {}.\n\n"
        if entity_lines else ""
    )

    candidate_hint = (
        f"\nThe most similar known commands (for reference only, do not assume one is correct): "
        f"{', '.join(top_candidates)}\n" if top_candidates else ""
    )

    routing_rules = (
        "STRICT ROUTING RULES:\n"
        "1. ACTIVE COMMAND REQUIREMENT: An utterance is a command ONLY if the user is currently "
        "and directly instructing the assistant to execute an action right now. Respond with "
        "domain='unknown', operation='unknown' if the utterance is:\n"
        "   - A past event, completed narrative, or story (e.g. describing what happened previously).\n"
        "   - A critique, evaluation, or complaint about past results or tools (e.g. 'my Google search results were useless', 'the results were terrible').\n"
        "   - An internal thought, deliberation, or personal reminder to self (e.g. 'I should...', 'I need to...').\n"
        "   - A casual mention or discussion of a tool or capability without requesting its execution.\n\n"
        "2. COMPUTER SCOPE VS PHYSICAL WORLD: Supported operations apply strictly to the host computer / desktop environment. "
        "The assistant CANNOT control external physical devices, appliances, lights, televisions, doors, or physical premises. "
        "Commands regarding physical fixtures (e.g. turning off lights or TVs, locking doors or shops, unlocking premises) are OUT-OF-SCOPE. "
        "NEVER map physical-world actions to system.shutdown or system.lock. "
        "Weather queries about outdoor conditions/forecasts are supported computer queries (weather.current). "
        "Respond with domain='unknown', operation='unknown' for non-weather physical tasks.\n\n"
        "3. NO NEAREST-NEIGHBOR GUESSING: If a request is not one of the supported computer operations, "
        "do NOT guess the closest registered command. Respond with domain='unknown', operation='unknown'.\n\n"
        "4. PRESERVE LEGITIMATE COMMANDS: When the user directly commands the assistant to perform a supported "
        "computer action (such as setting a timer, checking weather or outdoor conditions, launching apps, "
        "taking notes, locking the computer screen, shutting down the computer, or searching the web), "
        "classify it into the correct domain and operation with high confidence (e.g. 0.8 to 1.0).\n\n"
    )

    return (
        "You are a strict command router for a voice assistant running on a personal computer (PC). "
        "Classify the user's utterance into EXACTLY ONE domain and operation from the list below, or "
        "respond with domain='unknown', operation='unknown' if it is not an actionable command for this PC assistant.\n\n"
        f"Domains and valid operations:\n{domain_lines}\n\n"
        f"Supported operations and descriptions:\n{op_desc_lines}\n\n"
        f"{entity_section}"
        f"{routing_rules}"
        f"{candidate_hint}\n"
        f'User utterance: "{utterance}"\n\n'
        "If the utterance is not an actionable command from the list above (e.g. smalltalk, conversation, "
        "past narrative, physical world task, or unsupported request), respond with domain='unknown', operation='unknown'.\n"
        "If it IS one of the listed commands but you cannot confidently determine the operation or a required entity, "
        "set needs_clarification=true and confidence<=0.3. For weather.current, location is optional (defaults to local weather) "
        "and never requires clarification.\n"
        "confidence is REQUIRED: a JSON number from 0.0 to 1.0 reflecting how certain "
        "you are that the utterance matches the selected domain and operation. "
        "Never omit it.\n\n"
        "Respond with ONLY this exact JSON shape (all fields required), no other "
        f"text:\n{_RESPONSE_SCHEMA_HINT}"
    )


async def route(utterance: str, domains: list[str], operations_by_domain: dict[str, list[str]],
                 top_candidates: list[str] | None = None) -> dict | None:
    """
    Ask the local Ollama model for a structured routing decision.
    Returns a plain dict (unvalidated!) or None on any failure
    (connection error, timeout, malformed JSON, non-2xx). Never raises.
    """
    prompt = _build_prompt(utterance, domains, operations_by_domain, top_candidates or [])
    url = f"{config.llm.base_url.rstrip('/')}/api/chat"
    payload = {
        "model": config.llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "keep_alive": chat_keep_alive(),
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=_ROUTE_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as e:
        logger.warning(f"Router LLM fallback: connection error: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Router LLM fallback: timeout after {time.perf_counter()-t0:.1f}s "
                       f"(limit {_ROUTE_TIMEOUT_S}s): {e!r}")
        return None
    except httpx.HTTPError as e:
        logger.warning(f"Router LLM fallback: HTTP error: {e}")
        return None
    except ValueError as e:
        logger.warning(f"Router LLM fallback: unparseable response envelope: {e}")
        return None

    content = data.get("message", {}).get("content", "")
    if not content:
        logger.warning("Router LLM fallback: empty content in response.")
        return None
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Router LLM fallback: model did not return valid JSON: {e}  raw={content[:200]!r}")
        return None
    if not isinstance(parsed, dict):
        logger.warning(f"Router LLM fallback: JSON was not an object: {type(parsed).__name__}")
        return None
    ns = 1e9
    logger.info("[router] llm route ok %.2fs load=%.2fs prompt_tokens=%s prompt_eval=%.2fs gen_tokens=%s gen=%.2fs",
                time.perf_counter()-t0, (data.get("load_duration") or 0)/ns, data.get("prompt_eval_count"),
                (data.get("prompt_eval_duration") or 0)/ns, data.get("eval_count"),
                (data.get("eval_duration") or 0)/ns)

    return parsed