# MayaVE — Maya Virtual Entity

**Maya VE11**, a local-first Windows desktop AI voice assistant with a live 3D VRM avatar.

Inspired by AI companions such as FRIDAY from *Iron Man*, Maya is designed to be conversational, expressive, and context-aware rather than simply acting as a voice-controlled utility.

She can **listen, understand, think, speak, remember, perform actions, and express herself** through voice, facial expressions, animations, gaze, and mood.

The system runs primarily on local AI infrastructure, with internet connectivity currently required for speech transcription.

---

## ✨ Features

### 🧠 Intelligence

* Local LLM inference through **Ollama** (`llama3.2`, optimized 2048 context with 100% GPU residency)
* Hybrid semantic command router (default runtime path): deterministic guards, classifier margin gating, semantic retrieval, and validated LLM fallback
* ML-based intent classification using **PyTorch + TensorFlow**
* Context-aware conversation handling
* Topic and conversational-state tracking
* Short-term conversation memory
* Long-term semantic memory using **SQLite + local BGE embeddings**

### 🎙️ Voice

* Wake-word activation
* Name-gated barge-in interruption
* Silero VAD for speech detection
* Google Speech Recognition for transcription
* Kokoro neural TTS with automatic CUDA GPU acceleration (`device="auto"`, CPU fallback)
* Streaming LLM responses
* Phrase-level TTS pipelining

### 🎭 Personality & Behavior

* Emotion-aware responses
* Attitude and intensity control
* Persistent mood with decay
* Behavior composition from emotion, mood, personality, attitude, and intensity
* Structured action cues for physical avatar behavior

### 🧍 3D Avatar

* Live **VRM** avatar powered by Three.js, shown in a transparent, always-on-top **Electron** window
* Facial expressions
* Lip-sync
* Eye and head gaze
* Idle fidgets
* Breathing and life motion
* VRMA animations
* Layered expression control
* Animation and bone-ownership arbitration

### 🛠️ Voice Skills

Maya can interact with the system and external services through dedicated skills, including:

* Web search
* Website and application launching
* Weather
* Clipboard operations
* Notes
* Timers and reminders
* System information
* Screenshots
* Lock screen
* Shutdown and restart (with spoken confirmation)
* Media and volume control
* Date and time
* Avatar actions and animations

### 🎨 Expression Lab

MayaVE includes a standalone **Expression Lab** for calibrating VRM facial-expression recipes.

```text
frontend/expression-lab.html
```

The resulting expression configuration is stored in:

```text
frontend/assets/expressions.json
```

---

## 🏗️ Architecture

At a high level, MayaVE follows this pipeline:

```text
Microphone
    │
    ▼
Silero VAD / Wake Word
    │
    ▼
Speech Recognition (STT)
    │
    ▼
Processor (_classify)
    │
    ▼
CommandUnderstander (VE11 Hybrid Router)
[Guards ──► Classifier Gate ──► Semantic Retrieval ──► LLM Fallback]
    │
    ▼
CommandIR (READY / NEEDS_CLARIFICATION / UNKNOWN / REJECTED)
    │
    ▼
Router.dispatch
    ├──────────────► Voice Skills
    ├──────────────► Clarification
    └──────────────► Ollama LLM
                         │
                         ▼
                 Context + Memory
                         │
                         ▼
                  Mood + Behavior
                    ┌────┴────┐
                    ▼         ▼
                 Kokoro     VRM Avatar
                CUDA TTS    + Animation
```

The Python backend communicates with the browser-based avatar through a local WebSocket connection.

For the complete architecture and internal component breakdown, see **[`docs/architecture.md`](docs/architecture.md)**.

---

## 🛠️ Tech Stack

### Backend

* Python 3.11+
* asyncio
* Ollama (`llama3.2`, `nomic-embed-text`)
* Kokoro TTS (GPU-accelerated via PyTorch CUDA)
* Silero VAD
* Google Speech Recognition
* PyTorch (CUDA build)
* TensorFlow / Keras
* SQLite
* NumPy
* WebSockets
* WebSockets

### Frontend

* JavaScript
* Electron + Vite
* Three.js
* `@pixiv/three-vrm`
* `@pixiv/three-vrm-animation`
* Web Audio API
* Native WebSocket

---

## 📁 Repository Structure

```text
MayaVE/
│
├── config/
│   ├── settings.py
│   └── requirements.txt
│
├── core/
│
├── brain/
│
├── services/
│   └── llm/
│
├── skills/
│   ├── web/
│   ├── system/
│   ├── media/
│   └── utilities/
│
├── frontend/
│   ├── electron/
│   ├── js/
│   ├── assets/
│   ├── index.html
│   └── expression-lab.html
│
├── datasets/
│   ├── training/
│   ├── intent_model/
│   ├── intents.json
│
├── main.py
│
├── docs/
│   ├── architecture.md
│   ├── CHANGELOG.md
│   └── CONTRIBUTING.md
│
└── README.md
```

Runtime data such as logs, notes, and semantic memory is generated outside the repository.

---

## 💻 Requirements

* **Windows 11**
* **Python 3.11+**
* **Node.js / npm** (version required by Vite 8: `^20.19` or `>=22.12`)
* **Git LFS** (VRM/VRMA assets are stored with LFS)
* **Ollama**
* Microphone and audio output
* Internet connection for Google Speech Recognition
* User-supplied VRM avatar and VRMA animations

### Ollama Models

The default setup requires:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

The configured LLM can be changed through MayaVE's settings.

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/gitphilosopher/MayaVE.git
cd MayaVE
```

### Install backend dependencies

```bash
pip install -r config/requirements.txt
```

### Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Install Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

If the VRM/VRMA assets are tracked with Git LFS, fetch them:

```bash
git lfs pull
```

Otherwise place the required VRM and VRMA assets inside:

```text
frontend/assets/
```

---

## ▶️ Running Maya

Start the backend:

```bash
python main.py
```

Then start the frontend (Vite dev server + Electron shell):

```bash
cd frontend
npm run dev
```

Only dev-server mode is currently supported; `vite build` packaging of assets is not yet set up.

The backend WebSocket server runs locally on:

```text
ws://localhost:8765
```

Once the frontend connects, Maya's avatar becomes active and the assistant begins listening.

Maya can be put to sleep using commands such as:

```text
sleep
go to sleep
goodbye
```

She can be awakened again using the configured wake word.

---

## 🔭 Future Features

The following capabilities are still pending or materially incomplete in the current codebase.

### 📓 Diary

A separate chronological diary for Maya's own observations and experiences, distinct from the semantic memory used for retrieval and live context.

### 🌙 Dream Mode

A scheduled autonomous reflection cycle that runs while the user is away, consolidating recent interactions and feeding higher-level insight back into memory and diary flows.

### 🔎 Autonomous Research

Independent research initiation without an immediate user prompt, including source gathering, synthesis, and follow-up knowledge acquisition beyond direct request-driven web actions.

### 📚 Learning Notes

A dedicated store for research-derived knowledge that remains distinct from personal memory, user notes, and raw interaction history.

### 🧭 Autonomous Curiosity

A self-directed prioritization system for choosing what Maya should investigate next based on unresolved questions, knowledge gaps, novelty, and recent context.

### 🧠 Cognitive States

Explicit runtime cognition modes such as focused, exploratory, or reflective behavior, separate from emotional mood and avatar animation state.

### 🔁 Advanced Retrieval Reranking

A second-stage ranking step for semantic retrieval and command candidates so the router can refine top matches before dispatch or fallback decisions.

### 🧬 Metacognitive Memory Curation

Autonomous memory synthesis beyond simple deduplication: detecting redundancy, contradiction, temporal drift, and fragmented knowledge while preserving provenance and source traces.

---

## 📚 Project Documentation

| File                                   | Purpose                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| [`docs/architecture.md`](docs/architecture.md)   | Detailed system architecture, invariants, known issues, and component design                |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md)         | Chronological record of features, fixes, changes, and breaking changes                      |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)   | Development rules and persistent project context for handing work between sessions          |

The README intentionally contains only the information needed to understand, install, and run MayaVE. Implementation-level details are maintained in the project documentation above; recent changes are tracked in the changelog.

---

## 🚧 Status

MayaVE is an **actively developed personal AI assistant**.

The core voice pipeline, local LLM integration, intent engine, memory system, mood and behavior system, VRM avatar, TTS pipeline, and voice skills are currently under active development and refinement.

The project is evolving toward a more natural, expressive, and autonomous desktop AI companion.

---

## 📜 License

MayaVE is currently maintained as a personal development project.

Third-party libraries, AI models, VRM avatars, animations, and other assets remain subject to their respective licenses.