"""
brain/router/llm_fallback.py
Structured, non-streaming Ollama routing fallback.

Deliberately does NOT reuse services/llm/llm_service.py — that module
is built for streaming conversational replies straight into Kokoro TTS
(phrase-boundary splitting, expression tags, prosody). None of that
applies here: this is a single blocking-shaped request/response JSON
call with no persona, no streaming, no audio. Reusing it would mean
either dragging the TTS pipeline along for a routing decision or
forking half of its internals — both worse than a small dedicated
client.

The LLM is NEVER trusted directly: callers MUST run the returned dict
through brain/router/validate.py before using it for anything. This
module's only job is "ask Ollama for JSON, parse it defensively,
return a plain dict or None" — never raises out to the caller.
"""

import json
import logging

import httpx

from config.settings import config

logger = logging.getLogger(__name__)

_ROUTE_TIMEOUT_S = 8.0

_RESPONSE_SCHEMA_HINT = (
    '{"domain": "string", "operation": "string", "entities": {}, '
    '"confidence": 0.0, "needs_clarification": false}'
)


def _build_prompt(utterance: str, domains: list[str], operations_by_domain: dict[str, list[str]],
                   top_candidates: list[str]) -> str:
    domain_lines = "\n".join(
        f"- {d}: {', '.join(operations_by_domain.get(d, []))}" for d in domains
    )
    candidate_hint = (
        f"\nThe most similar known commands (for reference only, do not assume one is correct): "
        f"{', '.join(top_candidates)}\n" if top_candidates else ""
    )
    return (
        "You are a strict command router for a voice assistant. Classify the user's "
        "utterance into EXACTLY ONE domain and operation from the list below. Do not "
        "invent a domain or operation that isn't listed. Extract only entities that are "
        "explicitly present in the utterance.\n\n"
        f"Allowed domains and operations:\n{domain_lines}\n"
        f"{candidate_hint}\n"
        f'User utterance: "{utterance}"\n\n'
        "If the utterance is not an actionable command from the list above (e.g. it's "
        "smalltalk, a factual question, or a conversation), respond with domain='unknown', "
        "operation='unknown'.\n"
        "If it IS one of the listed commands but you cannot confidently determine the "
        "operation or a required entity, set needs_clarification=true and confidence<=0.3.\n\n"
        f"Respond with ONLY this exact JSON shape, no other text:\n{_RESPONSE_SCHEMA_HINT}"
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
        "keep_alive": getattr(config.llm, "keep_alive", "60m"),
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    try:
        async with httpx.AsyncClient(timeout=_ROUTE_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as e:
        logger.warning(f"Router LLM fallback: connection error: {e}")
        return None
    except httpx.TimeoutException as e:
        logger.warning(f"Router LLM fallback: timeout: {e}")
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
    return parsed