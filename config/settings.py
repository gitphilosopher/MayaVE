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
    model_size: str    = "small"   # tiny | base | small | medium | large
    language: str      = "en"
    device: str        = "cpu"    # "cpu" or "cuda"
    compute_type: str  = "int8"   # int8 | float16 | float32


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


# Replace your LLMConfig in config/settings.py with this:

@dataclass
class LLMConfig:
    provider: str      = "ollama"
    model: str         = "llama3.2"          # 3B params — 2x faster than llama3.1 8B
    api_key: str       = ""
    base_url: str      = "http://localhost:11434"
    max_tokens: int    = 150                 # short answers = faster response
    temperature: float = 0.7
    system_prompt: str = (
        "You are Maya, a concise and intelligent voice assistant inspired by "
        "Iron Man's FRIDAY. Address the user as 'senpai'. "
        "Keep every answer under 3 sentences. "
        "Speak naturally — your response will be read aloud."
    )


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
    # WebSocket server for browser avatar
    ws_host: str       = "localhost"
    ws_port: int       = 8765


# Singleton — import this everywhere
config = MayaConfig()