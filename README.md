# MayaVI — Maya (MARK 10)

Maya is a **local-first, Windows 11 desktop AI voice assistant** with a live 3D VRM avatar — a FRIDAY-style companion (Iron Man) that listens, thinks, speaks, and emotes, running entirely on local infrastructure (Ollama for LLM inference, Kokoro for TTS, Silero for VAD) with no cloud AI dependency for the core assistant loop. Google's free STT API is currently used for transcription (requires internet).

## Goals

- **Local-first**: LLM (Ollama), TTS (Kokoro), VAD (Silero), and intent classification (PyTorch + TensorFlow) all run on-device.
- **Emotionally expressive**: every reply carries an emotion tag that drives both TTS prosody and a blended VRM facial/body performance.
- **Low-latency, interruptible**: phrase-level streaming TTS and barge-in support so Maya can be cut off mid-sentence by saying her name.
- **Context-aware**: short-term conversation memory, conversational state tracking (topic/goal/task/decisions), and long-term semantic memory via a local vector store.

---

## Key Features

- **Wake-word activation** ("hey maya" / "maya") with name-gated barge-in interruption.
- **Voice pipeline**: Silero VAD → Google STT → dual-model ML intent classification → skill dispatch or Ollama LLM.
- **20+ voice skills**: web search, open website/app, weather, clipboard, notes, timers/reminders, system info (battery/CPU/RAM/disk/screenshot), media/volume control, date/time, physical action animations (nod/giggle/sigh/shrug/wink).
- **Streaming LLM responses** with phrase-level TTS pipelining (synthesis starts before the full sentence has streamed).
- **Emotion & nuance tags**: `[expression]`, optional `[attitude:word]`, `[intensity:word]`, and `*action*` stage-direction tags, parsed and enforced against closed vocabularies.
- **Persistent mood engine**: Maya's anger/sadness persists and decays across turns instead of resetting every response.
- **Behavioral Engine**: composes emotion + mood + attitude + personality into a blended VRM expression (six-knob legacy blend or a calibrated fine-grained morph "recipe").
- **3D avatar (VRM)**: idle fidgets, gaze/screen-attention system, lip-sync, life-motion (breathing/posture), bone-ownership arbitration between concurrent animation systems.
- **Stage 2/3 conversational intelligence**: recent-turn memory, topic reconciliation (continuation/subtopic/digression/return/switch), open-loop tracking, and SQLite + Ollama-embedding semantic long-term memory.
- **Expression Lab**: standalone browser tool for calibrating the VRM morph-target "recipes" behind each emotion/attitude/intensity combination.

---

## Architecture

```
 ┌────────────────────────────── Python Backend (asyncio) ──────────────────────────────────┐
 │                                                                                          │
 │  mic ──▶ Listener (Silero VAD +      ──▶ QueueManager ──▶ Processor ──▶ Router         │
 │          WakeWordDetector)               (asyncio.Queue,      │            │             │
 │              │                            serial FIFO)        │            ├─▶ Skills   │
 │              ▼                                                │            │  (20+)      │
 │        Transcriber (Google STT)                               │            │             │
 │                                                               ▼            ▼             │
 │                                                          IntentEngine   llm_service      │
 │                                                        (PyTorch BiLSTM +   (Ollama       │
 │                                                         TF CNN ensemble +  streaming +   │
 │                                                         keyword fallback)  Kokoro TTS)   │
 │                                                                                          │
 │  ContextManager (brain/conversation.py): recent window, conversation state,              │
 │  open loops, SQLiteVectorStore + Ollama embeddings (semantic memory)                     │
 │                                                                                          │
 │  MoodManager (persistent anger/sad) ──▶ BehaviorEngine (compose expression packet)      │
 │                                                                                          │
 │  Speaker / llm_service TTS worker ──▶ ws_server (WebSocket, :8765) ──▶ browser avatar   │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
                                            │  WebSocket (audio / state / behavior /
                                            ▼  transcript / animation)
 ┌───────────────────────────── Frontend (Three.js + @pixiv/three-vrm) ─────────────────────┐
 │  websocket.js → avatar.js (VRM load, lip-sync, idle fidgets, animations)                 │
 │                → expression-composer.js (behavior packet → VRM weights)                  │
 │  animation-controller.js (bone ownership) · expression-controller.js (layered exprs)     │
 │  life-motion-controller.js (breathing/posture) · gaze-controller.js (screen attention)   │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Backend component responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | App entrypoint; warms up Ollama/Kokoro/embeddings, wires WS handlers, wake/sleep/interrupt logic, launches listener + queue worker as concurrent tasks. |
| `core/state.py` | Global FSM (`SLEEPING`/`IDLE`/`LISTENING`/`PROCESSING`/`SPEAKING`/`INTERRUPTED`) and barge-in interrupt plumbing (`run_interruptible`, `interrupt()`). |
| `core/listener.py` | Mic capture via `sounddevice`, Silero VAD utterance segmentation, feeds `WakeWordDetector`. |
| `core/wake_word.py` | 2-second rolling-window Google STT wake-word spotting; shared trigger set also gates barge-in. |
| `core/transcriber.py` | Google Speech Recognition (`recognize_google`), executor-wrapped. |
| `core/queue_manager.py` | Single `asyncio.Queue` — serializes all commands so responses never overlap. |
| `core/processor.py` | Orchestrates one command: intent classify → context update → route → speak. |
| `core/speaker.py` | Kokoro TTS playback for skill/greeting/alert lines; dual output (`local` sounddevice / `avatar` WebSocket / `both`). |
| `core/mood.py` | Event-driven persistent mood (angry/sad) with decay, apology forgiveness, ragebait/teasing detection. |
| `core/behavior_engine.py` | Composes tag + mood + personality + attitude/intensity into a communicative-intent packet (+ expression "recipe"). |
| `core/expression_library.py` | Deterministic VRoid morph-weight recipe cache, persisted to `frontend/assets/expressions.json`. |
| `brain/intent_engine.py` | Dual-model ML classifier (PyTorch BiLSTM+Attention, TF 1-D CNN) with keyword-rule fallback and several regex guards (dismissal, presence, action-request). Auto-retrains on `TRAINING_DATA` hash change. |
| `brain/router.py` | Maps classified intent → skill function or `llm_service.query`. |
| `brain/conversation.py` | `ConversationManager` (recent-turn window) + `ContextManager` (topic/state/open-loop tracking, semantic memory retrieval/persistence). |
| `brain/vector_store.py` | SQLite-backed brute-force cosine-similarity vector store (`~/Maya/Memory/semantic_memory.sqlite3`). |
| `brain/embeddings.py` | `OllamaEmbedder` — calls Ollama's `/api/embeddings` (`nomic-embed-text`). |
| `brain/memory.py` | Bounded in-process recent-turn list with an eviction callback for compaction. |
| `services/llm/llm_service.py` | Ollama streaming chat, phrase-level TTS pipelining (streamer → synth_worker → play_worker), expression/attitude/intensity/action tag parsing, prosody enhancement, filler phrases. |
| `services/ws_server.py` | WebSocket server (`:8765`) — broadcasts `audio`/`state`/`behavior`/`transcript`/`animation`/`stop_audio`; receives `interrupt`/`audio_done`. |
| `skills/*` | Individual voice skills (see table below). |
| `config/settings.py` | Single `MayaConfig` dataclass — audio, STT, TTS, LLM, context tuning, WS host/port. |

### Skills (`skills/`)

| Skill | File |
|---|---|
| Open website | `skills/web/open_website.py` |
| Google search | `skills/web/google_search.py` |
| Weather (Open-Meteo + ip-api.com) | `skills/web/weather.py` |
| Open app | `skills/system/open_app.py` |
| System info / screenshot | `skills/system/system_info.py` |
| Clipboard read/write/clear | `skills/system/clipboard.py` |
| Physical action animations | `skills/system/perform_action.py` |
| Media/volume control (media keys) | `skills/media/play_music.py` |
| Date/time | `skills/utilities/datetime_skill.py` |
| Reminder (fire-and-print) | `skills/utilities/reminder.py` |
| Countdown timers (spoken alert) | `skills/utilities/timer.py` |
| Notepad (create/read/list/append/delete/open) | `skills/utilities/notepad.py` |

### Frontend component responsibilities (`frontend/js/`)

| File | Responsibility |
|---|---|
| `main.js` | Three.js scene/camera/renderer bootstrap, render loop. |
| `avatar.js` | VRM load, VRMA animation playback, lip-sync (AnalyserNode-driven), idle-fidget scheduler, sleep/wake, head/eye movement. |
| `websocket.js` | WS client; routes `audio`/`state`/`behavior`/`animation`/`stop_audio` messages. |
| `expression-composer.js` | Turns a backend behavior packet into VRM expression weights — either a verified fine-grained morph "recipe" or the legacy six-knob (neutral/joy/fun/angry/sorrow/surprised) blend. |
| `animation-controller.js` | Bone-ownership arbiter (BASE < FIDGET < ACTION priority) with smooth handoff easing. |
| `expression-controller.js` | Layered `expressionManager.setValue()` arbiter (BASE < EMOTION < ACTION < LIPSYNC < BLINK). |
| `life-motion-controller.js` | Continuous breathing/posture/shoulder micro-motion (BASE tier). |
| `gaze-controller.js` | Screen-attention session tracking → eye/head gaze offsets, boredom ramp. |
| `expression-lab.js` / `expression-lab.html` | Standalone calibration tool for `expressions.json` (localStorage-backed, Import/Export). |

---

## Backend ↔ Frontend Communication

WebSocket server at `ws://localhost:8765` (see `services/ws_server.py` / `frontend/js/websocket.js`).

**Server → client messages** (`type` field):
| Type | Payload | Purpose |
|---|---|---|
| `audio` | base64 WAV | Speech to play |
| `state` | `listening`\|`processing`\|`speaking`\|`idle` | Drives idle-fidget gating |
| `behavior` | `{primary, secondary, intensity, attitude, gaze, actions, recipe}` | Composed expression, consumed by `expression-composer.js` |
| `transcript` | `{text, role}` | User/Maya transcript (overlay reserved, not yet wired) |
| `animation` | `{name}` | Fire one-shot VRMA animation (wave/nod/giggle/sigh/shrug/wink) |
| `stop_audio` | — | Barge-in: hard-stop client playback immediately |

**Client → server messages**:
| Type | Purpose |
|---|---|
| `interrupt` | Manual "stop talking" (routed to `core.state.interrupt()`) |
| `audio_done` | Signals current sentence finished playing — gates the next queued phrase |

---

## Tech Stack

**Backend (Python, asyncio)**
- Ollama (LLM inference, default `llama3.2`, `http://localhost:11434`) + Ollama embeddings (`nomic-embed-text`)
- Kokoro TTS (offline neural TTS, voice blend `af_sky` + `jf_alpha`)
- Silero VAD (`torch.hub`) for speech segmentation
- Google Speech Recognition (`SpeechRecognition`) for STT + wake-word (online, free, no key)
- PyTorch (BiLSTM+Attention) and TensorFlow/Keras (1-D CNN) — dual-model intent ensemble
- SQLite (stdlib) + NumPy — local semantic vector store
- `websockets`, `httpx`, `sounddevice`, `pyperclip`, `psutil`, `keyboard`, `pyautogui`, `wikipedia`

**Frontend (browser/Electron renderer)**
- Three.js, `@pixiv/three-vrm`, `@pixiv/three-vrm-animation`
- Native WebSocket + WebAudio (AnalyserNode-driven lip-sync)

---

## Repository Structure

```
config/
  settings.py            # MayaConfig — single source of runtime config
core/
  state.py listener.py transcriber.py speaker.py mood.py
  behavior_engine.py expression_library.py queue_manager.py
  processor.py wake_word.py
brain/
  intent_engine.py router.py conversation.py vector_store.py
  embeddings.py memory.py train_intent.py
services/
  ws_server.py
  llm/llm_service.py
skills/
  web/ system/ media/ utilities/
frontend/
  js/  (avatar.js, main.js, websocket.js, expression-composer.js,
        animation-controller.js, expression-controller.js,
        life-motion-controller.js, gaze-controller.js,
        expression-lab.js)
  assets/ (mayaaa.vrm, vrmas/*.vrma, expressions.json)
  index.html              # main avatar window
  expression-lab.html      # standalone calibration tool
models/                   # generated: pytorch_intent.pt, tf_intent.keras, vocab.json, labels.json, training_hash.txt
main.py
requirements.txt
```

External runtime data (created on demand, not in-repo): `~/Maya/Notes/` (notepad), `~/Maya/Memory/semantic_memory.sqlite3` (long-term memory), `logs/maya.log`.

---

## Prerequisites

- **Windows 11** (skills use `os.startfile`; media-key control via `keyboard`).
- **Python 3.10+** with PyTorch and TensorFlow support.
- **[Ollama](https://ollama.com)** running locally with:
  - `ollama pull llama3.2` (or whichever model is set in `config.llm.model`)
  - `ollama pull nomic-embed-text` (semantic memory embeddings)
- **PortAudio** (`sounddevice` dependency) — bundled on Windows wheels.
- **Kokoro TTS** weights (installed via `pip install kokoro`).
- A **VRM avatar model** at `frontend/assets/mayaaa.vrm` and matching `.vrma` animation clips at `frontend/assets/vrmas/` (not included in this repo — user-supplied).
- Internet connectivity (Google STT for transcription + wake word are online APIs).
- A Node/static file server (e.g. Vite) to serve `frontend/` — `ws_server.py` explicitly allows `http://localhost:5173` as an origin.

---

## Installation

```bash
# Backend
pip install -r requirements.txt

# Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text

# Frontend (from frontend/, assuming a Vite-based setup)
npm install
```

Place your VRM model at `frontend/assets/mayaaa.vrm` and animation clips under `frontend/assets/vrmas/` (see `_VRMA_ASSETS` in `avatar.js` for the expected filenames: `wave.vrma`, `hard_nod.vrma`, `excited.vrma`, `sigh.vrma`, `shrug.vrma`, `weight_shift.vrma`, `idle_lalala.vrma`, `yawn.vrma`, plus several registered-but-unwired clips).

---

## Running Maya

1. Start Ollama (`ollama serve`, or it auto-runs as a service).
2. Start the backend:
   ```bash
   python main.py
   ```
   This pre-warms Ollama, the embedding model, and two Kokoro pipelines concurrently, opens the WebSocket server on `:8765`, and begins listening on the default mic — starting **SLEEPING**.
3. Serve/open the frontend (`frontend/index.html`) — on connect it plays the wake/greeting animation and Maya enters **IDLE**.
4. Say **"hey maya"** (configurable in `config.wake_word`) to wake her; say **"go to sleep"**, **"sleep"**, or **"goodbye"** to put her back to sleep.
5. Interrupt her mid-sentence by calling her name again (`contains_wake_word` gate on barge-in).

`frontend/expression-lab.html` can be opened independently to calibrate expression morph recipes; it writes to browser `localStorage` and exports `expressions.json`.

---

## Configuration

All runtime tuning lives in `config/settings.py` (`MayaConfig`), notably:

| Section | Key fields |
|---|---|
| `AudioConfig` | `sample_rate=16000`, `chunk_ms=30`, `silence_ms=800`, `pre_roll_ms=200` |
| `STTConfig` | `model_size`, `language="en"`, `device`, `compute_type` (currently unused directly — STT is Google, not Whisper) |
| `TTSConfig` | `voice="af_sky"`, `voice_blend="jf_alpha"`, `blend_ratio=0.92`, `speed=1`, `output="avatar"` |
| `LLMConfig` | `provider="ollama"`, `model="llama3.2"`, `base_url="http://localhost:11434"`, `max_tokens=150`, `temperature=0.7`, `system_prompt` |
| `ContextConfig` | `recent_turns=6`, `max_open_loops=3`, `max_semantic_memories=3`, `similarity_threshold=0.75`, `dedup_threshold=0.92`, `embedding_model="nomic-embed-text"` |
| top-level | `wake_word="wake up Maya"` *(effective default trigger set derived from this — see `core/wake_word.py`)*, `ws_host="localhost"`, `ws_port=8765`, `log_dir="logs"` |

Optional env var: `MAYA_EMBEDDING_DEVICE=cpu` forces the embedding model off GPU (see `brain/embeddings.py`). `HF_HUB_OFFLINE=1` is set automatically to avoid cold-start network calls.

Intent training data lives inline in `brain/intent_engine.py::TRAINING_DATA`; models auto-retrain on startup when its content hash changes (or run `python -m brain.train_intent` for a full manual retrain + test report).

---

## Development Workflow

- **Single-developer, iterative delivery**: complete files are exchanged and merged whole (no inline diffs), each carrying forward prior session fixes (barge-in wiring, hash-based retrain logic, etc.).
- **Backend**: all commands route through the single `asyncio.Queue` (`core/queue_manager.py`) — never bypass it. Skills that speak their own response must return `ALREADY_SPOKEN` (see `services/llm/llm_service.py`) so `processor.py` doesn't double-speak.
- **Blocking calls** (Kokoro synthesis, psutil, file I/O) must go through `run_in_executor`.
- **Retraining the intent engine**: edit `TRAINING_DATA`, restart (auto-retrain) or run `python -m brain.train_intent` for a full wipe + classification test report.
- **Tuning expressions**: use `expression-lab.html` to calibrate `Fcl_BRW_*`/`Fcl_EYE_*`/`Fcl_MTH_*` morph recipes per `emotion|attitude|intensity` key, then Export to `frontend/assets/expressions.json`.
- **Kokoro output mode** (`config.tts.output`) must be `"avatar"`, not `"both"`, when the Electron/browser avatar is running (audio would otherwise double-play).

---

## Current Limitations / Known Gaps

- **Wave/speech sequencing latency**: Kokoro synthesis currently stacks sequentially after a lead-in delay; a concurrent `asyncio.gather()` fix was attempted and rolled back — unresolved.
- **Idle fidget system** is still being actively tuned (cooldowns, gaze-boredom modulation).
- **Transcript overlay** message type exists on the wire (`type: "transcript"`) but has no frontend UI consumer yet.
- **STT/wake-word require internet** (Google Speech API) — no fully offline fallback currently wired in.
- **Windows-only** skill implementations (`os.startfile`, `keyboard` media-key sends); partial Darwin fallback exists in `open_app.py` only.
- Reminder skill (`skills/utilities/reminder.py`) only prints to console on expiry — it does not speak, unlike `timer.py`'s alert path.
- No automated test suite; validation is manual functional testing of routing/trigger-matching/hash-detection before delivery.
- VRM model and `.vrma` animation assets are not included in the repository — must be user-supplied.

## Roadmap

- SQLite-backed persistence hardening for long-term memory (currently functional but young).
- Transcript overlay UI.
- Further interrupt-handling refinement.
- Strategic TTS pivot under consideration: Edge-TTS as a short-term step, Coqui XTTS v2 long-term for genuine emotion-conditioned synthesis (moving off Kokoro).
- End-to-end skill test coverage.
