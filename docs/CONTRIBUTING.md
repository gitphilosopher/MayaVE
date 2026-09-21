# MayaVE — Contributing & Session Context

Persistent rules and context for taking over MayaVE development between sessions. This file is not a log and does not describe architecture:

| Need | Read |
|---|---|
| How the system works, invariants | `docs/architecture.md` |
| What changed recently, how a bug was fixed, known issues | `docs/CHANGELOG.md` (`## [Unreleased]`) |
| What the project is, install, run | `README.md` |

## Project Context

MayaVE is a local-first, Windows-first AI voice companion: Python `asyncio` backend (`main.py`, Ollama, semantic memory, mood/behavior systems, Kokoro TTS) ⇄ WebSocket `ws://localhost:8765` ⇄ Electron/Vite/Three.js VRM avatar (pure WS client). Persona: FRIDAY-like, addresses the user as `config.user_name` ("senpai").

**Current focus:** expression realism/calibration, latency (`[TIMING]`/TTFA logging), animation/fidget tuning, context/memory quality.

## Rules

* Read only files relevant to the current task.
* Read `docs/architecture.md` before making architectural changes; check its **Invariants** section before touching queue/state/audio/WS/expression code.
* Read `### Issues` and recent entries in `docs/CHANGELOG.md` when current state, known issues, or pending work is relevant.
* Preserve existing behavior unless explicitly asked to change it.
* Do not rewrite or replace working systems unnecessarily.
* Keep backend/frontend contracts synchronized (WS messages, closed vocabularies).
* Do not modify unrelated files.
* Prefer small, targeted changes over broad refactors.
* Follow existing patterns and naming conventions.
* Blocking work goes to an executor (`run_in_executor`), never on the event loop.
* Code is the source of truth. If code and any document disagree, trust the code and fix the document.

## Verification Basis

All project knowledge so far comes from **static inspection; nothing has been runtime-tested**. Treat "confirmed" as "traced in source". VRM/VRMA assets were Git LFS pointer stubs, so model-dependent behavior (morph names, animation tracks) is unverified. Do not claim runtime behavior you have not observed.

## Decisions Already Made (do not revisit unless asked)

* Extra `expressions.json` entries stay Lab-only; no `default` fallback recipe is added (runtime lookup stays deterministic).
* Closed emotion/attitude/action vocabularies are not expanded to reach stored recipes.
* `lock_screen` is immediate (no confirmation); `shutdown`/`restart` require spoken confirmation.
* Unused config fields are kept but marked `UNUSED`; removal (e.g. `LLMConfig.api_key`) belongs in a dedicated config cleanup, not behavioral work.
* Interrupted LLM turns record only the phrases that were played; barge-in cancels PROCESSING/filler turns.
* Reminder text is kept and the timer alert is routed through the injected `Speaker` under `run_interruptible`.
* `skills/utilities/reminder.py` is dead code (`set_reminder` routes to `timer.py`). Decide delete-vs-keep explicitly; do not "fix" it silently.

## Start Here

**Understand first:** (1) `Processor.handle` → `Router.dispatch` → `llm_service.query` and its `synth_q`/`play_q` pipeline; (2) the `audio_done` handshake and single-queue serialization; (3) the state sequence `listening → processing → speaking → idle` and which function owns each transition; (4) tag → `BehaviorEngine.compose` → `expression-composer.js`; (5) frontend arbitration (`animation-controller`, `expression-controller`).

**Inspect first:** `main.py`, `core/processor.py`, `core/speaker.py`, `services/llm/llm_service.py`, `services/llm/ollama_lifecycle.py`, `core/state.py`, `services/ws_server.py`, `brain/router.py`, `skills/system/power.py`, `skills/utilities/timer.py`, `brain/intent_engine.py` (`_predict`), `brain/conversation.py`, `core/behavior_engine.py`, `frontend/js/{websocket,avatar,expression-composer}.js`, `config/settings.py`.

**Fragile areas:** Kokoro concurrency/timeouts/rebuild; the single `_current_task` in `state`; fidget/animation bone ownership; `expressions.json` dual writers; closed vocabularies duplicated across files; one-shot power confirmation vs. mic echo.

## Runtime Verification Checklist

Untested behavior worth checking on a real setup before or while changing related code:

* Sleep/wake: "go to sleep" holds SLEEPING; "hey maya" wakes her.
* Power: "shut down"/"restart" asks first; only "yes" schedules the action (abort with `shutdown /a`); "no", an unrelated utterance, or >30 s cancels without acting.
* State sequence `listening → processing → speaking → idle` in WS/log (no `listening` mid-reply); brief surprised face on `listening` returns to baseline after noise/empty STT; a late-connecting frontend receives `state` right after connect.
* Barge-in by name (mid-reply, during filler, before first phrase, during a skill/greeting line): she stops, no 30 s stall, history/transcript contains only played phrases (log line "LLM turn interrupted…" when none).
* Timers/reminders: "set a timer for 5 minutes" fires at 300 s; "remind me to drink water in 15 minutes" speaks the reminder; the alert waits for a mid-reply turn; re-set named timers stay cancellable; fidgets resume afterwards.
* "can you giggle/wink/nod" plays in sync with the reply; an ACTION right after a `headTilt`/VRMA fidget returns the neck to rest.
* Ollama-down error line is spoken but absent from later LLM context; skill replies appear in history without `[tag]` prefixes.
* "Well, yes." stays intact while a bare "Yes!" expands; "what is pi" speaks "3.14" as one number.
* Skills: screenshot writes a file; "lock my screen" locks Windows; "take a note buy milk and write the report" keeps the whole content, and "note read the report" is saved rather than read back.
* Expression Lab: browsing combinations without touching sliders then Export leaves untouched ones out.
* Restart the backend with the avatar open: blinks/eye motion stay normal speed after reconnect.
* Real VRM: which morphs/expression names exist (`vrm.expressionManager.expressions`, `morphTargetDictionary`; `Fcl_*`, `joy/fun/sorrow` vs `happy/relaxed`) and whether a recipe face holds after its 220 ms transition.
* `ollama ps` and `[TIMING] chat turn` logs: COLD vs warm, `keep_alive` applied, GPU/VRAM split (Kokoro on CPU).
* Backend starts with no frontend connected without a 30 s greeting stall; WS server stays up over long sessions.
* Echo with headphones vs. speakers (self-transcription, self-barge-in, dropped power confirmation).
* Environment: Python ≥3.11; `git lfs pull` done for assets.

## After Changes

Report briefly:

* Files changed as artifacts
* What changed
* Architectural implications, if any
* Testing performed
* Any remaining issues

Also, in the same change:

* Add each notable change to `docs/CHANGELOG.md` under `## [Unreleased]`.
* Update `docs/architecture.md` if behavior, contracts, or invariants changed; update `README.md` only for user-facing/install/run changes. Never copy content between documents — link instead.
* Known issues live only in `### Issues` of `docs/CHANGELOG.md`; do not record them in `docs/architecture.md`.

## Changelog Format

Machine-readable; follow exactly. Only `## [Unreleased]` is scanned automatically.

* Structure: `# Changelog` → intro line → `## [Unreleased]` → `### Features | Improvements | Architecture | Fixes | Issues` → older `## [x.y.z] — YYYY-MM-DD` sections, newest first.
* Categories (in this order):
  * `Features` — New capabilities.
  * `Improvements` — Enhancements to existing behavior.
  * `Architecture` — Significant architectural/structural changes.
  * `Fixes` — Resolved bugs/issues.
  * `Issues` — Known unresolved bugs, limitations, or required work. Start each description with a class tag: `[Confirmed]` (deterministic from source), `[Potential]` (plausible edge case), `[Limitation]` (known constraint or missing capability), `[Unverified]` (needs a runtime/asset check), `[Debt]` (hygiene/cleanup). Order items by class in that sequence.
* Lifecycle: An issue appears in `Issues`, stays there until resolved, then moves to `Fixes`. Never keep the same resolved issue in both categories.
* Architecture: `Architecture` records the change; `docs/architecture.md` remains the source of truth for the current design.
* Items: Every item is one line: `- **Title** — Description`. No nested bullets. Titles must be concise and unique..
* History: No session logs, implementation conversations, or debugging narration.
* Accuracy: Don't rewrite historical facts except to correct an obvious error.