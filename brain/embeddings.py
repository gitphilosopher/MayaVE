"""
brain/embeddings.py
Embedding provider abstraction for Maya's long-term semantic memory.

Isolates the embedding backend behind Embedder so it can be swapped
later without touching vector_store.py or conversation.py.

Default backend: Ollama's /api/embeddings endpoint. Reuses the local
Ollama server Maya already requires for chat — no new pip dependency,
no separate model-serving process. Requires an embedding model to be
pulled once:
    ollama pull nomic-embed-text
If that model isn't available, embed() returns None and every caller
in the Stage 2 pipeline is required to degrade gracefully rather than
fail (see ContextManager in conversation.py).

Latency fix: embed() used to open a brand-new httpx.AsyncClient on
every call. It now reuses one lazily-created, process-wide client
(_get_client()) so repeated embedding calls don't pay fresh
connection-setup cost each turn. Timeout raised 8.0s -> 20.0s so a
genuine cold model load (first embedding call after Ollama/Maya start)
isn't cut off mid-load — this should only matter once per session now
that OLLAMA_MAX_LOADED_MODELS=2 keeps both the chat and embedding
models resident simultaneously (see Handoff) instead of Ollama
evicting one to load the other on every turn.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod

import httpx
import time

from config.settings import config

logger = logging.getLogger(__name__)


def _resolve_embedding_device() -> str:
    """
    "cpu" forces the embedding model off the GPU via Ollama's num_gpu
    runtime option (see _embedding_gpu_options). "auto" (default) sends
    no override — today's existing behavior, whatever Ollama's own
    default placement is.

    Checked via env var first (MAYA_EMBEDDING_DEVICE=cpu) so the two
    configurations can be A/B tested with a restart and no code/config
    file edits; falls back to config.context.embedding_device via
    getattr() so this also works if that field doesn't exist yet.
    """
    env_override = os.environ.get("MAYA_EMBEDDING_DEVICE")
    if env_override:
        return env_override.strip().lower()
    return getattr(config.context, "embedding_device", "auto")


def _embedding_gpu_options() -> dict | None:
    """{'num_gpu': 0} to force CPU-only, or None to send no override."""
    if _resolve_embedding_device() == "cpu":
        return {"num_gpu": 0}
    return None


async def describe_ollama_models() -> None:
    """
    Diagnostic only — logs each currently-loaded Ollama model's GPU/CPU
    placement via GET /api/ps (size_vram / size = % resident in VRAM),
    so a device-placement change can be verified from Maya's own log
    instead of external tooling. Best-effort; never raises.
    """
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
                name      = m.get("name", "?")
                size      = m.get("size", 0)
                size_vram = m.get("size_vram", 0)
                pct_gpu   = (size_vram / size * 100) if size else 0.0
                placement = "GPU" if pct_gpu >= 99 else "CPU" if pct_gpu <= 1 else f"MIXED({pct_gpu:.0f}% GPU)"
                logger.info(
                    f"[TIMING] Ollama model '{name}': {placement}  "
                    f"size={size/1e6:.0f}MB size_vram={size_vram/1e6:.0f}MB"
                )
    except Exception as e:
        logger.debug(f"describe_ollama_models diagnostic failed (non-fatal): {e}")


class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float] | None:
        ...


class OllamaEmbedder(Embedder):
    def __init__(self, model: str | None = None, timeout: float = 20.0):
        self._model = model or config.context.embedding_model
        self._timeout = timeout
        self._url = f"{config.llm.base_url.rstrip('/')}/api/embeddings"
        self._device = _resolve_embedding_device()
        logger.info(f"[TIMING] Embedding device mode: '{self._device}' (model='{self._model}')")
        # Sticky flag — once we confirm the model is missing, stop retrying
        # every turn and just skip straight to "no embedding available".
        self._unavailable = False
        # Lazily-created, reused across every embed() call instead of a
        # fresh httpx.AsyncClient per request (see module docstring).
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Release the pooled connection — call on app shutdown if one exists."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def embed(self, text: str) -> list[float] | None:
        if self._unavailable or not text.strip():
            return None
        try:
            t0 = time.perf_counter()
            client = await self._get_client()
            payload = {
                "model": self._model,
                "prompt": text,
                # Keep the embedding model resident well past a
                # normal gap between conversational turns so it
                # isn't evicted and forced to cold-load again on
                # the next semantic-memory lookup (cold loads run
                # several seconds — see Handoff debugging notes).
                "keep_alive": "30m",
            }
            gpu_opts = _embedding_gpu_options()
            if gpu_opts:
                payload["options"] = gpu_opts
            resp = await client.post(self._url, json=payload)
            logger.info(f"[TIMING]       httpx POST /api/embeddings: {time.perf_counter()-t0:.3f}s")
            resp.raise_for_status()
            emb = resp.json().get("embedding")
            if not emb:
                logger.warning(f"Ollama embeddings returned no vector for model '{self._model}'")
                return None
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