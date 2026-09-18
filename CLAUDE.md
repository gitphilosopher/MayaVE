# MayaVI Development Context

MayaVI is a locally hosted AI companion with:
- Python backend
- Ollama LLM
- semantic memory
- mood/behavior systems
- Kokoro TTS
- Three.js + VRM frontend
- Electron desktop shell

## Rules

Read existing architecture before modifying it.

Do not replace working systems unnecessarily.

Keep backend and frontend contracts synchronized.

Do not modify unrelated files.

Prefer incremental changes over rewrites.

Preserve existing behavior unless the task explicitly changes it.

After changes, report:
- files changed
- what changed
- important architectural implications
- testing performed