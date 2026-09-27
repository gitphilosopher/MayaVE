"""
brain/embeddings.py
Ollama-backed embedding service for Maya's semantic memory.

This module provides the optional text-embedding layer used by higher-level
semantic-memory features. `OllamaEmbedder` implements the small `Embedder`
interface used across the project by calling Ollama's `/api/embeddings`
endpoint, caching exact-text results, reusing an HTTP client, and logging slow
requests for debugging.

The embedder is intentionally resilient: absent models, failed requests, and
empty vectors all return `None` so callers can degrade gracefully to recent-turn
context rather than interrupt an active conversation. The abstraction is isolated
so another provider can replace Ollama without changing the rest of the memory
stack.
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import httpx

from config.settings import config

logger = logging.getLogger(__name__)

_CACHE_MAX = 64
_SLOW_EMBED_SECONDS = 2.0
_KEEPALIVE_EXPIRY = 300.0

_bg_tasks: set[asyncio.Task] = set()


def _resolve_embedding_device() -> str:
    """Resolve the embedding device preference from env override or config fallback."""
    env_override = os.environ.get("MAYA_EMBEDDING_DEVICE")
    if env_override:
        return env_override.strip().lower()
    return getattr(config.context, "embedding_device", "auto")


def _embedding_gpu_options() -> dict | None:
    """Return `{"num_gpu": 0}` only when the embedder must be forced onto CPU."""
    if _resolve_embedding_device() == "cpu":
        return {"num_gpu": 0}
    return None


async def describe_ollama_models() -> None:
    """Log the currently loaded Ollama models for runtime placement and keep-alive diagnostics."""
    url = f"{config.llm.base_url.rstrip('/')}/api/ps"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if not models:
                logger.info("[TIMING] Ollama /api/ps: no models currently loaded")
                return
            for m in models:
                name = m.get("name", "?")
                size = m.get("size", 0)
                size_vram = m.get("size_vram", 0)
                pct_gpu = (size_vram / size * 100) if size else 0.0
                placement = "GPU" if pct_gpu >= 99 else "CPU" if pct_gpu <= 1 else f"MIXED({pct_gpu:.0f}% GPU)"
                logger.info(
                    f"[TIMING] Ollama model '{name}': {placement}  "
                    f"size={size/1e6:.0f}MB size_vram={size_vram/1e6:.0f}MB "
                    f"expires_at={m.get('expires_at', '?')}"
                )
    except Exception as e:
        logger.debug(f"describe_ollama_models diagnostic failed (non-fatal): {e}")


class Embedder(ABC):
    """Common interface for text embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float] | None:
        """Return a numeric embedding for the input text, or `None` if unavailable."""
        ...


class OllamaEmbedder(Embedder):
    """Ollama-backed embedding client with a bounded exact-text cache and pooled HTTP usage."""

    def __init__(self, model: str | None = None, timeout: float = 20.0):
        self._model = model or config.context.embedding_model
        self._timeout = timeout
        self._url = f"{config.llm.base_url.rstrip('/')}/api/embeddings"
        self._device = _resolve_embedding_device()
        logger.info(f"[TIMING] Embedding device mode: '{self._device}' (model='{self._model}')")
        self._unavailable = False
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._in_flight = 0
        self._last_call_at: float | None = None
        self._logged_first_request = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and reuse a single HTTP client for embedding requests."""
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=self._timeout,
                        limits=httpx.Limits(keepalive_expiry=_KEEPALIVE_EXPIRY),
                    )
        return self._client

    async def aclose(self) -> None:
        """Close the shared client when the app is shutting down."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _remember(self, text: str, emb: list[float]) -> None:
        """Cache an exact-text embedding and evict the oldest item when the limit is exceeded."""
        self._cache[text] = emb
        self._cache.move_to_end(text)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)

    def _report_slow(self, elapsed: float, gap: float | None, overlap: int) -> None:
        """Log timing and model-placement diagnostics when an embedding request is unusually slow."""
        gap_s = f"{gap:.0f}s" if gap is not None else "first call"
        logger.warning(
            f"Slow embedding ({elapsed:.2f}s): model='{self._model}' device='{self._device}' "
            f"gap_since_last={gap_s} overlapping_requests={overlap} — Ollama state follows"
        )
        task = asyncio.create_task(describe_ollama_models())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    async def embed(self, text: str) -> list[float] | None:
        """Return an embedding for `text`, or `None` if the model is unavailable or the request fails."""
        if self._unavailable or not text.strip():
            return None
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            logger.info("[TIMING]       embed cache hit")
            return cached

        overlap = self._in_flight
        self._in_flight += 1
        try:
            t0 = time.perf_counter()
            gap = (t0 - self._last_call_at) if self._last_call_at is not None else None
            client = await self._get_client()
            payload = {
                "model": self._model,
                "prompt": text,
                "keep_alive": "30m",
            }
            gpu_opts = _embedding_gpu_options()
            if gpu_opts:
                payload["options"] = gpu_opts
            if not self._logged_first_request:
                self._logged_first_request = True
                logger.info(
                    f"[TIMING]       first embed request: keep_alive={payload['keep_alive']} "
                    f"options={payload.get('options')}"
                )
            resp = await client.post(self._url, json=payload)
            elapsed = time.perf_counter() - t0
            self._last_call_at = time.perf_counter()
            logger.info(
                f"[TIMING]       httpx POST /api/embeddings: {elapsed:.3f}s "
                f"(chars={len(text)} gap={'%.0fs' % gap if gap is not None else 'first'} "
                f"in_flight_at_start={overlap})"
            )
            resp.raise_for_status()
            emb = resp.json().get("embedding")
            if not emb:
                logger.warning(f"Ollama embeddings returned no vector for model '{self._model}'")
                return None
            self._remember(text, emb)
            if elapsed >= _SLOW_EMBED_SECONDS:
                self._report_slow(elapsed, gap, overlap)
            return emb
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(
                    f"Embedding model '{self._model}' not found in Ollama — "
                    f"semantic memory disabled for this session. Run: ollama pull {self._model}"
                )
                self._unavailable = True
            else:
                logger.warning(f"Embedding request failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"Embedding error (non-fatal): {e}")
            return None
        finally:
            self._in_flight -= 1