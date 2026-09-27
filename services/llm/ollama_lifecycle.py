"""
services/llm/ollama_lifecycle.py
Lifecycle helpers for the Ollama-backed chat model.

This module centralizes the model-residency behavior used by the LLM pipeline and
startup warmup. It owns the `keep_alive` policy, extracts per-turn timing and
throughput metrics from Ollama's final response object, and emits a best-effort
post-turn residency snapshot so cold starts and unexpected eviction are visible
without interrupting normal runtime.

It intentionally stays dependency-light and imports only config and stdlib
helpers so it can be used safely from the LLM service, warmup code, and
diagnostic tooling without depending on the full speech/TTS stack.
"""

import asyncio
import logging
import time

from config.settings import config

logger = logging.getLogger(__name__)

_DEFAULT_KEEP_ALIVE = "60m"
_COLD_LOAD_SECONDS  = 1.0     # load_duration at/above this = model wasn't resident

_last_chat_done_at: float | None = None
_bg_tasks: set = set()


def chat_keep_alive():
    """Return the per-request keep_alive value for the active chat model."""
    raw = getattr(config.llm, "keep_alive", _DEFAULT_KEEP_ALIVE)
    try:
        return int(str(raw).strip())   # "-1"/"3600" must be sent as numbers, not strings
    except ValueError:
        return raw


def placement(size: float, size_vram: float) -> str:
    """Classify a model's placement as GPU-, CPU-, or mixed-backed from VRAM share."""
    pct = (size_vram / size * 100) if size else 0.0
    return "GPU" if pct >= 99 else "CPU" if pct <= 1 else f"MIXED({pct:.0f}% GPU)"


def summarize_metrics(data: dict) -> dict:
    """Extract load, prompt, and generation timings from Ollama's final stream payload."""
    ns = 1e9
    load = (data.get("load_duration") or 0) / ns
    pe_n, pe_s = data.get("prompt_eval_count") or 0, (data.get("prompt_eval_duration") or 0) / ns
    ev_n, ev_s = data.get("eval_count") or 0, (data.get("eval_duration") or 0) / ns
    return {
        "load_s":        load,
        "cold":          load >= _COLD_LOAD_SECONDS,
        "prompt_tokens": pe_n,
        "prompt_s":      pe_s,
        "prompt_tps":    pe_n / pe_s if pe_s > 0 else None,
        "gen_tokens":    ev_n,
        "gen_s":         ev_s,
        "gen_tps":       ev_n / ev_s if ev_s > 0 else None,
    }


def _log_gpu_memory() -> None:
    """Emit a best-effort VRAM snapshot for the whole device and this Python process."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        logger.info(
            f"[TIMING] GPU memory: device used={(total-free)/1e6:.0f}MB of {total/1e6:.0f}MB "
            f"(all processes) | this process (Kokoro) allocated={torch.cuda.memory_allocated()/1e6:.0f}MB "
            f"reserved={torch.cuda.memory_reserved()/1e6:.0f}MB"
        )
    except Exception:
        pass


async def log_residency() -> None:
    """Query live model residency and GPU usage after a turn without raising runtime errors."""
    try:
        from brain.embeddings import describe_ollama_models
        await describe_ollama_models()
        await asyncio.get_running_loop().run_in_executor(None, _log_gpu_memory)
    except Exception as e:
        logger.debug(f"log_residency failed (non-fatal): {e}")


def log_chat_turn(data: dict) -> None:
    """Log the turn's cold/warm status and schedule a non-blocking residency snapshot."""
    global _last_chat_done_at
    m = summarize_metrics(data)
    now = time.time()
    # Gap to this request's START (completion-to-completion would include the turn itself).
    total_s = (data.get("total_duration") or 0) / 1e9
    gap = None if _last_chat_done_at is None else max(0.0, now - _last_chat_done_at - total_s)
    _last_chat_done_at = now

    f = lambda v: "n/a" if v is None else f"{v:.1f}"
    logger.info(
        f"[TIMING] chat turn: {'COLD' if m['cold'] else 'warm'} load={m['load_s']:.2f}s "
        f"prompt={m['prompt_tokens']}tok@{f(m['prompt_tps'])}tok/s "
        f"gen={m['gen_tokens']}tok@{f(m['gen_tps'])}tok/s "
        f"gap_since_last_chat={'first' if gap is None else '%.0fs' % gap} "
        f"keep_alive={chat_keep_alive()}"
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:   # no running loop
        return
    task = loop.create_task(log_residency())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)