# MayaVE — Development Rules

MayaVE is a local-first AI companion with a Python backend, Ollama, semantic memory, mood/behavior systems, Kokoro TTS, and a Three.js/VRM frontend running in an Electron desktop shell.

## Rules

* Read only files relevant to the current task.
* Read `architecture.md` before making architectural changes.
* Read `HANDOFF.md` when current state, known issues, or pending work is relevant.
* Preserve existing behavior unless explicitly asked to change it.
* Do not rewrite or replace working systems unnecessarily.
* Keep backend/frontend contracts synchronized.
* Do not modify unrelated files.
* Prefer small, targeted changes over broad refactors.
* Follow existing patterns and naming conventions.

## After Changes

Report briefly:

* Files changed as artifacts 
* What changed
* Architectural implications, if any
* Testing performed
* Any remaining issues
