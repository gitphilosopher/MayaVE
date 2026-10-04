"""
config/settings.py
Runtime configuration for MayaVE.

This module defines the single shared configuration object used by the app, its
services, and any helper modules that need stable runtime defaults. The values in
these dataclasses are intentionally centralized so environment-specific tuning is
kept in one place instead of being duplicated across modules.

The settings are organized into a nested structure:
- Audio and transcription settings drive microphone capture and VAD.
- TTS settings configure Kokoro voice output and routing.
- LLM settings point the app at the local Ollama endpoint.
- Context settings tune memory and recent-turn retrieval.
- Node settings support optional MayaNode discovery/sync integration.
- Router settings control the VE11 hybrid semantic command router
  (brain/router/); default backend is "hybrid".
- MayaConfig wires all of the above into the singleton `config` object imported
  across the project.

This file is the source of truth for runtime defaults. Most modules consume the
`config` singleton directly rather than re-defining their own tuning values.
"""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """Microphone capture and VAD settings used by the live listener pipeline."""
    sample_rate: int = 16_000
    channels: int = 1
    chunk_ms: int = 30
    silence_ms: int = 800
    pre_roll_ms: int = 200
    device_index: int = None


@dataclass
class STTConfig:
    """Transcription defaults for the speech-to-text layer."""
    language: str = "en"


@dataclass
class TTSConfig:
    """Kokoro voice synthesis settings and playback routing."""
    voice: str = "af_sky"
    voice_blend: str = "jf_alpha"
    blend_ratio: float = 0.92
    lang_code: str = "a"
    speed: float = 1
    output: str = "avatar"
    device: str = "auto"
    repo_id: str = "hexgrad/Kokoro-82M"


@dataclass
class LLMConfig:
    """Settings for the local LLM backend used by conversation and routing logic."""
    model: str = "llama3.2"
    base_url: str = "http://127.0.0.1:11434"
    max_tokens: int = 150
    temperature: float = 0.7
    num_gpu: int | None = None
    num_ctx: int = 2048


@dataclass
class ContextConfig:
    """Tuning for recent-memory retrieval, deduplication, and semantic context windows."""
    recent_turns: int = 6
    max_open_loops: int = 3
    max_semantic_memories: int = 3
    similarity_threshold: float = 0.75
    dedup_threshold: float = 0.92
    semantic_recency_guard_seconds: float = 120.0
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    memory_dir: str = None


@dataclass
class RouterConfig:
    """Hybrid semantic command router (brain/router/). Rollback is
    `backend = "legacy"` and nothing else."""
    backend: str = "hybrid"                    # "legacy" | "hybrid"
    hybrid_domains: list[str] = field(default_factory=list)   # empty = all domains when backend == "hybrid"
    shadow_mode: bool = False                  # log hybrid-vs-legacy comparison; never dispatches hybrid
    command_vector_db_path: str | None = None  # default ~/Maya/Router/command_vectors_bge.sqlite3 (local) or command_vectors.sqlite3 (ollama)
    embedding_provider: str = "local"          # "local" | "ollama"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Placeholders pending brain/router/eval_router.py measurements on the real dataset.
    min_similarity: float = 0.80
    min_margin: float = 0.08
    low_similarity_floor: float = 0.55


@dataclass
class NodeConfig:
    """Optional MayaNode discovery and sync settings for infrastructure-level coordination."""
    enabled: bool = False
    base_url: str | None = None
    discovery_candidates: list[str] = field(default_factory=lambda: [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ])
    discovery_timeout: float = 2.0
    request_timeout: float = 8.0
    sync_interval_s: float = 60.0
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 300.0
    auth_token: str | None = None
    state_dir: str | None = None
    # Step 3 — per-sync-round upload caps so one round never tries to
    # push an unbounded outbox in a single request; anything past the
    # cap simply stays pending for the next round (see
    # services/node/sync_manager.py's _sync_once).
    max_events_per_sync: int = 200
    max_memory_per_sync: int = 200

@dataclass
class MayaConfig:
    """Top-level runtime configuration object that aggregates all runtime subsystems."""
    name: str = "Maya"
    user_name: str = "senpai"
    wake_word: str = "wake up Maya"
    log_level: str = "INFO"
    log_dir: str = "logs"
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    ws_host: str = "localhost"
    ws_port: int = 8765
    ws_allowed_origins: list[str | None] | None = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        None,
    ])


config = MayaConfig()