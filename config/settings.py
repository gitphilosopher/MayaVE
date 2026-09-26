"""
config/settings.py
Central configuration — edit here, nowhere else.
"""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int   = 16_000   # Hz — required by Silero VAD & Whisper
    channels: int      = 1        # mono
    chunk_ms: int      = 30       # VAD frame size (10 / 20 / 30 ms)
    silence_ms: int    = 800      # ms of silence that ends an utterance
    pre_roll_ms: int   = 200      # ms of audio kept before speech starts
    device_index: int  = None     # None = system default mic


@dataclass
class STTConfig:
    language: str      = "en"


@dataclass
class TTSConfig:
    # Kokoro voice IDs: af_sky (soft youthful American female), jf_alpha (Japanese female),
    # if_sara (Italian female). Blend ratio: 0.0 = pure primary, 1.0 = pure blend.
    voice:       str   = "af_sky"      # primary voice (cute + soft)
    voice_blend: str   = "jf_alpha"    # accent blend — swap to "if_sara" for Italian
    blend_ratio: float = 0.92          # 35% Japanese lilt over English baseline
    lang_code:   str   = "a"           # a=American EN, b=British EN, j=JP, i=IT
    speed:       float = 1           # slightly faster feels more alive
    # Output mode: 'local' = sounddevice only, 'avatar' = WebSocket only, 'both' = both
    output:      str   = "avatar"
    # Kokoro device: "cpu" | "cuda" | "auto". CPU keeps the GPU free for Ollama.
    device:      str   = "cpu"


# Replace your LLMConfig in config/settings.py with this:

@dataclass
class LLMConfig:
    model: str         = "llama3.2"          # 3B params — 2x faster than llama3.1 8B
    base_url: str      = "http://localhost:11434"
    max_tokens: int    = 150                 # short answers = faster response
    temperature: float = 0.7


@dataclass
class ContextConfig:
    """Stage 2 — Context Intelligence tuning (see brain/conversation.py)."""
    recent_turns:          int   = 6      # size of the recent-context window sent to Ollama
    max_open_loops:        int   = 3      # open loops surfaced per turn, relevance-ranked
    max_semantic_memories: int   = 3      # semantic memories surfaced per turn
    similarity_threshold:  float = 0.75   # min cosine similarity to surface a memory
    dedup_threshold:       float = 0.92   # min similarity to treat as the same memory (update, not duplicate)
    semantic_recency_guard_seconds: float = 120.0  # skip semantic hits newer than this — already covered by recent context
    embedding_model:       str   = "nomic-embed-text"   # Ollama embedding model — pull separately: ollama pull nomic-embed-text
    memory_dir:            str   = None                 # None -> ~/Maya/Memory


@dataclass
class NodeConfig:
    """
    MayaVE -> MayaNode integration (infrastructure only; see
    docs/CONTRIBUTING.md and services/node/sync_manager.py's module
    docstring). No application data (events/memory/intents) is wired
    through this yet — this config only supports discovery, connection,
    and an empty-payload sync loop that persists/advances a cursor.

    Disabled by default: this is new, opt-in background network activity
    that nothing else in MayaVE depends on yet. Flip `enabled` to True
    once a MayaNode instance is actually running to talk to.
    """
    enabled: bool = False
    # Tried first, ahead of discovery_candidates, if set (e.g. a fixed
    # LAN address for a MayaNode instance that isn't on localhost).
    base_url: str | None = None
    # Tried in order — the first to answer GET /status wins.
    discovery_candidates: list[str] = field(default_factory=lambda: [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ])
    discovery_timeout: float = 2.0     # seconds, per candidate probe
    request_timeout: float   = 8.0     # seconds, per heartbeat/sync call
    sync_interval_s: float   = 60.0    # gap between successful sync passes
    initial_backoff_s: float = 2.0     # first retry delay after a failure
    max_backoff_s: float     = 300.0   # retry delay ceiling (5 minutes)
    # Auth-ready: sent as `Authorization: Bearer <token>` when set.
    # MayaNode does not validate this yet (see its api/sync.py docstring)
    # — the slot exists so enabling real auth later is a config change on
    # both sides, not a protocol change.
    auth_token: str | None = None
    # None -> ~/Maya/Node (device_id + sync cursor persisted here)
    state_dir: str | None = None


@dataclass
class MayaConfig:
    name: str          = "Maya"
    user_name: str     = "senpai"                   # for more natural conversations
    wake_word: str     = "wake up Maya"           # optional future wake-word
    log_level: str     = "INFO"
    log_dir: str       = "logs"
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig     = field(default_factory=STTConfig)
    tts: TTSConfig     = field(default_factory=TTSConfig)
    llm: LLMConfig     = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    node: NodeConfig   = field(default_factory=NodeConfig)
    # WebSocket server for browser avatar
    ws_host: str       = "localhost"
    ws_port: int       = 8765
    # Origins allowed to open the avatar WebSocket (services/ws_server.py).
    # Exact Origin strings; a None entry also accepts clients that send no Origin
    # header (non-browser tools). Vite moves to another port if 5173 is busy —
    # add that origin here. Set the whole value to None to disable the check.
    ws_allowed_origins: list[str | None] | None = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        None,
    ])


# Singleton — import this everywhere
config = MayaConfig()