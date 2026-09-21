# Changelog

All notable changes to MayaVE are documented here.

## [Unreleased]

### Features
- **Shutdown and restart skill** — `skills/system/power.py` now handles the `shutdown`/`restart` intents (previously answered by the LLM, nothing happened) by asking "say yes to confirm"; `Router.dispatch` calls `resolve_pending` first so a one-shot request (30 s TTL) is confirmed, declined or dropped by the next utterance, and confirm runs `shutdown /s` or `/r /t 10` (no `/f`) in the executor.
- **Lock screen skill** — `lock_screen` intent routed to `skills/system/lock_screen.py` (`LockWorkStation` via ctypes; apology on non-Windows), immediate by design with no confirmation because it is reversible.
- **Spoken timer reminder text** — `timer._extract_reminder` captures "remind me to X" (duration stripped, max 120 chars) and the alert speaks "Time's up, senpai! Reminder: X."

### Improvements
- **Ollama residency diagnostics** — `services/llm/ollama_lifecycle.py` logs per-turn COLD/warm status, load/prompt/generation tok/s and gap since the last chat, plus a background `/api/ps` and CUDA-memory snapshot; diagnostics only, never raises.
- **TTS device and LLM model defaults** — `config.tts.device` is now defined as `"cpu"` (`_resolve_tts_device`; `KPipeline(device=...)` is passed only if the installed kokoro accepts it) and `config.llm.model` is `llama3.2`.
- **Unused config fields marked** — `STTConfig.model_size/device/compute_type`, `LLMConfig.provider/api_key/system_prompt` and `TTSConfig.cpu_threads` are kept as config surface and commented `UNUSED` in `settings.py`.
- **Barge-in history** — interrupted LLM turns now record only the phrases whose playback started (`_spoken_phrases`) while finished turns record the full reply (`_reply_finished`).
- **Error strings and tags in history** — `query()` error replies set `intent["_no_history"]` (spoken, not stored) and `Processor` stores and broadcasts tag-stripped text for both skill and LLM replies.
- **Log volume** — `main.py` writes `logs/maya.log` through a `RotatingFileHandler` (5 MB × 3 backups) and sets the `httpx`/`httpcore`/`websockets`/`urllib3`/`tensorflow` loggers to WARNING to drop per-request noise.

### Architecture
- **Listening state broadcast** — `main.on_speech` broadcasts `state:listening` only when Maya is not PROCESSING/SPEAKING, empty STT resets to IDLE with baseline behavior, and the FSM no longer overwrites an in-flight turn, giving the sequence `listening → processing → speaking → idle`.
- **WebSocket last-state replay** — `ws_server` records `_last_state` on every `broadcast_state` (even with no clients, default `idle`) and `_handler` sends it to each new client, so the frontend's `_currentBackendState` is set at connect instead of staying `null` until the first broadcast.
- **Barge-in during processing and filler** — `state.can_interrupt()` allows PROCESSING with a live registered speech task, `on_speech` snapshots it instead of `is_speaking()`, and `query()` re-sets `_filler_done` in a `finally` so a cancelled filler never blocks the first phrase.
- **Timer alert routing** — the alert speaks through the `Speaker` injected by `Router` under `run_interruptible`, is queued on the command worker via `queue_manager.put_job` so it runs in order with commands, and the expired timer is removed from `_timers` before the alert.
- **Unreferenced background tasks** — `main.py` keeps `ws_task` with `_on_ws_done` failure logging and `ContextManager` keeps `_bg_tasks`.
- **Duplicate elongation helper removed** — `llm_service.py` keeps only the non-recursive `_elongation_re_sub` definition.

### Fixes
- **Open website unknown sites** — `open_website` adds amazon/linkedin/stackoverflow to `_SITES` and resolves other targets through `_resolve_url` (explicit URL, dotted host, spoken "dot", or bare label → `https://www.<label>.com`); multi-word or junk targets now get a clarification reply instead of an invalid `https://…` URL.
- **HF offline flag import order** — `main.py` sets `HF_HUB_OFFLINE` (`setdefault "1"`) as its first statement, before any import that pulls in kokoro/huggingface_hub, so it now takes effect (huggingface_hub reads it at import time); the later `setdefault` in `llm_service.py` is redundant but harmless.
- **Blocking calls on the event loop** — `Processor` runs `IntentEngine.classify` via `run_in_executor`, `ContextManager` runs every semantic-store call (`has_records`, `search`, `find_similar`, `update`, `add`) via `run_in_executor`, `system_info` builds its reply (CPU sample, screenshot) in the executor and reports mood events back on the loop, and `expression_library.save_recipe` updates the cache immediately but writes the file on a single-thread writer (submission order preserved, last save wins); only startup work (`IntentEngine()` load/train, first `expressions.json` read) stays synchronous.
- **Kokoro rebuild blocks loop and resets wrong pipeline** — `Speaker._synthesise_guarded` rebuilds its pipeline in the executor and passes `on_timeout=lambda: None` to `_run_kokoro`, so a Speaker timeout no longer resets llm_service's pipeline.
- **Timer alert not serialized** — `timer._alert` queues the alert with `queue_manager.put_job`, so it runs on the command worker in order with commands and never overlaps a turn.
- **Reminder and timer parsing gaps** — `_words_to_digits` parses number words ("one hour"), "remind me to X" without a duration keeps X pending and `resolve_pending` takes the next utterance as the duration, and the label lookahead uses word boundaries so labels like "apple timer" are kept.
- **Residual audio_done stall** — `wait_for_audio_done` clears the shared Event and then compares `_stop_gen` with `_audio_sent_gen`, so a stop before entry returns immediately and a stop after entry sets the Event.
- **Skill-exception line stored in history** — the router's error fallback sets `intent["_no_history"]`, so it is spoken but not stored as an assistant turn.
- **Degenerate context topic and entity data** — `Processor` blanks `intent["target"]` when it equals the whole utterance before `observe_user_turn`, so `TOPIC`, `ENTITIES` and `LIKELY REFERRING TO` no longer echo the current utterance.
- **Decimals and versions split mid-token** — `_next_boundary` now waits when a `.` ends the buffer right after an alphanumeric, so "3.14" and "google.com" are no longer split into separate phrases.
- **Spurious Kokoro pipeline rebuild** — `Speaker._synthesise_guarded` rebuilds its pipeline only on a real timeout, using a `_NO_AUDIO` sentinel for "no audio produced".
- **expressions.json wipe on corrupt read** — `expression_library._load` records `_load_failed` and `_save_all` then refuses to overwrite the file; writes go through a `.tmp` file with atomic replace.
- **WebSocket broadcast set mutation** — `ws_server._broadcast` iterates a copy of `_clients` instead of the live set.
- **Stale utterance after sleep** — `Listener._process_frame` clears the in-progress utterance while SLEEPING.
- **Audio-done stalls** — the frontend sends `audio_done` when `vrm` is missing or decoding fails, `broadcast_stop_audio` sets the shared Event, and `_audioGen` drops audio whose decode finished after a stop.
- **Short exclamation over-expansion** — `_expand_short_exclamation` is skipped for `continuation` chunks so "Well, yes." keeps "yes." while a bare "Yes!" still expands.
- **Stale fidget timers and headTilt fighting actions** — the VRMA cleanup timer is guarded by mixer identity and `headTilt` is cancellable via `_headTiltStop`, halted when an ACTION animation starts.
- **Expression Lab default pinning** — only edited (`_dirty`) combinations are persisted on switch or Export, so browsing no longer pins generated defaults.
- **Timer duration double-count** — `timer._parse_duration` uses one pattern per unit (`hours?|hrs?`, `minutes?|mins?`, `seconds?|secs?`) so "5 minutes" no longer counts twice.
- **Timer name race and reminder hijack** — `_countdown`'s `finally` removes only its own `_timers` entry and `timer.execute` skips cancel/status word checks for `set_reminder`, so "remind me to stop by…" no longer cancels a timer.
- **Perform action animation** — `Processor` now broadcasts `intent["action"]` (set by `perform_action`) via `on_audio_start` (nod/giggle/sigh/shrug/wink).
- **Dismissal guard misroutes** — the dismissal guard is now an exact-phrase match (trailing `.!?,` stripped) instead of `startswith`, fixing misroutes of "note…", "nod…", "stop the timer" and similar.
- **Screenshot skill** — saves a timestamped file, auto-creates the folder, no longer calls the broken `logging.log`, and also triggers on intent `screenshot` ("capture my screen").
- **Sleep not persisting** — `Speaker.speak()` restores SLEEPING when it was sleeping on entry and `main.py` sets IDLE before the startup greeting, so Maya still starts awake and the wake word is now actually active while asleep.
- **Chat model idle unload** — `keep_alive` (`ollama_lifecycle.chat_keep_alive()`, default `"60m"`) is sent on every chat request and the chat warmup instead of Ollama's 5-minute default.
- **Startup greeting stall without a client** — `wait_for_audio_done` returns False immediately when no frontend client is connected instead of waiting 30 s.
- **Notepad content handling** — a `note_*` intent dispatches directly to its handler and `_extract_content` strips only the leading command phrase, so "take a note buy milk and write the report" keeps the whole content.
- **Missing dependencies and logs directory** — `websockets` and `pyperclip` added to `config/requirements.txt` and `main.py` creates `logs/` before logging setup.
- **Frontend reconnect leak** — `startBlinking()` clears `_blinkTimeout` and `startEyeMovement()` starts once per page (`_eyeLoopStarted`), so WS reconnects no longer double the blink/eye loops.

### Issues
- **Empty-target fallback never triggers** — [Confirmed] `_extract_target` returns the whole utterance (or the untouched trigger phrase, e.g. "search") when nothing strips, so `google_search`'s empty-target prompt rarely fires and it searches for the trigger text itself.
- **Weather location and code gaps** — [Confirmed] The location regex runs to end of string ("in London today" → "London today") and `_WMO` lacks codes 56/57/66/67/77/85/86, yielding "unknown conditions" with a neutral tag.
- **System info substring match** — [Confirmed] "ram" is matched as a substring, so "program" triggers the RAM report.
- **Perform action over-triggering** — [Confirmed] The action-word guard fires on any whole-word occurrence ("what does nod mean"), and the animation only plays via the `Speaker.speak()` avatar/both path.
- **Energy fragmentation truncation** — [Confirmed] `_fragment_for_energy` truncates to 4 comma-fragments on excited/happy/angry final chunks, and the Speaker path passes whole multi-sentence text.
- **Listening state edge cases** — [Potential] A dropped command (queue full) or a manual `interrupt` with no follow-up leaves the FSM in LISTENING with the frontend stuck on `listening` (no fidgets), noise-triggered `listening → idle` resets the frontend's 30-minute idle timer, and a late-connecting frontend briefly shows the surprised-0.3 face if the replayed state is `listening`.
- **Interrupted turn leaves unanswered user turn** — [Potential] After a barge-in the user turn stays in history with no (or a partial) assistant reply, and a barge-in before the first phrase records nothing for the assistant.
- **Sleep race and startup log** — [Potential] A command already queued or in flight when "go to sleep" is spoken can overwrite SLEEPING (`_play_worker` sets IDLE at `_DONE`, `Processor` sets PROCESSING), and the Listener still logs "Sleeping — say …" at startup though Maya starts awake.
- **Echo and self-transcription** — [Potential] VAD and STT run while Maya speaks with no echo cancellation, so her own speech (e.g. "I'm Maya") can queue as a command, trigger self-barge-in or drop a pending power confirmation (setup-dependent).
- **Power confirmation fragility** — [Potential] The confirmation is one-shot so an STT mishearing of "yes" or Maya's own prompt picked up by the mic drops it (safe direction), the 10 s OS countdown has no voice abort, and graceful shutdown without `/f` can be blocked by apps with unsaved work.
- **Wake window boundary** — [Potential] The wake detector uses 2 s non-overlapping windows, so a wake word straddling a boundary is missed.
- **Substring and prefix matching** — [Potential] Sleep/wake triggers ("asleep", "I didn't sleep well"), greeting prefixes ("yo" → "you…"), `open_website`, `timer` and `system_info` use substring or prefix tests, and phrases like "no thanks please" miss the exact-match dismissal guard and fall to keyword/ML routing.
- **Kokoro concurrency and lazy init** — [Potential] Filler and first phrase synthesize concurrently on one pipeline (plus Speaker overlap), `_get_kokoro` lazy init is unlocked, a stuck native call cannot be killed (daemon thread orphaned), and Ollama/Kokoro/embeddings may contend for GPU if `tts.device` leaves `"cpu"`.
- **Latency on the critical path** — [Potential] The embedding call sits before the first LLM token when memories exist, the per-phrase WAV plus `audio_done` round trip creates inter-phrase gaps, and the chat model still unloads if idle beyond `keep_alive` (default 60 m) or evicted by Ollama (check `[TIMING] chat turn` COLD lines).
- **Clipboard import crash** — [Potential] `skills/system/clipboard.py` imports `pyperclip` at top level, so a missing package crashes startup through the Router import.
- **Skill-path barge-in** — [Limitation] `Speaker.speak()` is not wrapped in `run_interruptible`, so a barge-in during a skill, greeting, wake or sleep line stops audio but does not cancel the task, and skill turns in PROCESSING are not interruptible.
- **Avatar never visually sleeps** — [Limitation] Backend SLEEPING is never sent to the frontend; sleep/wake follows the WebSocket connection only.
- **Open app has no name mapping** — [Limitation] `open_app` passes the target straight to `os.startfile` and replies untagged.
- **Notepad delete without confirmation** — [Limitation] `_delete` removes the latest note immediately.
- **Cloud-dependent speech recognition** — [Limitation] Google STT is required for both utterances and the wake word, so Maya cannot run offline despite an otherwise local-first design.
- **Recent window not persisted** — [Limitation] The 50-entry conversation window is lost on restart; only semantic memories (narrow write policy) survive, and `VALID_MEM_TYPES` values goal/decision/relationship/project are never produced.
- **Duplicate Kokoro pipelines** — [Limitation] `Speaker` and `llm_service` each hold a `KPipeline` (deliberate, avoids cold reloads), duplicating RAM, or VRAM if the device changes.
- **Unrestricted WebSocket origin** — [Limitation] `ws_server` uses `origins=None` and accepts any local page (which could send `interrupt`), and `audio_done` from multiple clients would confuse the gate; acceptable for localhost only.
- **Production build unsupported** — [Limitation] `vite build` does not copy `frontend/assets/` into `dist/` (no `publicDir`), so only dev-server mode is evidenced.
- **Idle fidget tuning unfinished** — [Limitation] Cooldown and threshold constants in `avatar.js` are still in flux and not final.
- **Wave and speech out of sync** — [Unverified] Per project notes the `wave` fires in `on_audio_start` after synthesis; a concurrent `asyncio.gather` fix was attempted and rolled back, and the current status cannot be re-verified from source.
- **Expression naming split** — [Unverified] The composer uses `joy/fun/sorrow` while `avatar.js` uses `happy/relaxed`, and which set the real VRM exposes (plus which `Fcl_*` morphs exist) is unknown, so one path may silently no-op.
- **Recipe persistence past 220 ms** — [Unverified] `_animateRecipeTo` stops writing raw morphs after 220 ms; if `expressionManager.update()` resets morphs bound to expressions, recipe faces would fade.
- **Ollama residency and GPU placement** — [Unverified] Actual keep-alive behavior, model availability and VRAM contention between Ollama and embeddings (Kokoro on CPU) are unchecked; use `ollama ps` and the COLD log lines.
- **Windows skills and Electron launch** — [Unverified] `shutdown.exe` invocation, lock screen, screenshots and an actual `npm run dev` Electron launch have not been run.
- **Nothing runtime-tested** — [Unverified] Every fix in this changelog and all speech-recognition accuracy/latency, Kokoro thread-safety and echo behavior were checked by static inspection only; VRM/VRMA assets were Git LFS stubs.
- **Dead code and stray files** — [Debt] `skills/utilities/reminder.py` is unreferenced, `wikipedia` is an unused dependency, unused config fields remain (`STTConfig`, `LLMConfig.provider/api_key/system_prompt`, `TTSConfig.cpu_threads`), and `.gitignore` has a BOM plus stray `0.9.0`/`12.0` lines; older docs also claimed a `_strip_markdown` that does not exist.
- **Unpinned websockets import path** — [Debt] `websockets>=12.0` is unpinned and the `websockets.server.WebSocketServerProtocol` import is a legacy path on newer releases.