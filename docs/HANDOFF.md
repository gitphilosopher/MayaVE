# MayaVE — Technical Handoff

**Basis:** static inspection of the repository snapshot. **No runtime was available** — "Confirmed" below means deterministic from source (code trace), not observed at runtime. Binary assets (`.vrm`, `.vrma`, `.vroid`) were Git LFS pointer stubs (text with `oid`/`size`) in the inspected snapshot, not real binaries; their contents are **UNVERIFIED**. If your checkout has the real assets, they were still not inspected. **`README.md` and `docs/architecture.md` are current, relevant project documentation. When documentation conflicts with the actual implementation, the code is the source of truth.** Known doc conflict: README/architecture describe `set_reminder` as routed to `skills/utilities/reminder.py` (silent); the code routes it to `timer.py` (§9/§11). The README's Python-version and requirements-path statements were corrected in a later audit — re-check the README rather than trusting older notes.

**Fix log:** consolidated from three Handoff revisions; a fix reported in any revision is treated as applied. Batch 1–4 fixes (timer/reminder routing and race, `perform_action` animation, `keep_alive` + lifecycle logging, `lock_screen`, `shutdown`/`restart`, sleep persistence, `listening` broadcast + state replay, screenshot, notepad, dismissal guard, requirements/`logs/`, reconnect-loop leak) are applied in source — statically traced, **not runtime-tested**. Bugs found by the later static audit are listed in §11 Confirmed. See "Resolved" in §11.
---

# 1. MAYAVE CURRENT STATE

- **What:** Windows-first desktop voice assistant "Maya" with a transparent always-on-top 3D VRM avatar. Persona: FRIDAY-like, addresses user as `config.user_name` = "senpai".
- **Two processes:** Python `asyncio` backend (`main.py`) ⇄ WebSocket `ws://localhost:8765` ⇄ Electron/Vite/Three.js frontend (`frontend/`). Frontend is a pure WS client.
- **Models/services:** Ollama chat (`config.llm.model`) + Ollama embeddings (`nomic-embed-text`); Kokoro TTS (local, 24 kHz, CPU by default); Silero VAD (`torch.hub`); Google STT via `SpeechRecognition` (**online**, used for utterances *and* wake word); intent classifier = PyTorch BiLSTM + TF CNN ensemble; SQLite semantic memory.
- **Capabilities present in code (several have known runtime/behavioral problems — skill-path barge-in, timer alert; see §11):** wake word/VAD/STT; serial command queue; ML intent routing + regex guards; ~13 routed skills (incl. lock_screen and confirmed shutdown/restart; `reminder.py` is dead code); streaming LLM with phrase-level TTS pipelining and emotion/attitude/intensity/action tags; persistent mood; behavior engine → Fcl_* morph recipes; VRMA animation system; idle fidgets; gaze/life-motion layers; barge-in; recent-window + semantic long-term memory; chat/embedding model residency diagnostics; Expression Lab (dev tool).
- **Development focus (from code state):** expression realism/calibration, latency (`[TIMING]`/TTFA logging is everywhere), animation/fidget tuning, context/memory quality.

---

# 2. ACTUAL ARCHITECTURE

**Startup (`main.py::main`)**
1. `os.makedirs(config.log_dir)` before logging setup. `asyncio.create_task(ws_server.serve())` (**reference not kept**, §11 #11); construct `Speaker()`, `Transcriber()`, `Processor(speaker)` (→ `IntentEngine()` load/auto-train — *synchronous, blocks loop*; `ConversationManager`; `Router(speaker)`).
2. `state.register_stop_callback(_hard_stop_audio)`.
3. Warmups (gathered): `_warmup_ollama` (1-token chat, sends `keep_alive=chat_keep_alive()`), `_warmup_embeddings` (`keep_alive:"30m"`), `llm_service.warmup` (Kokoro, executor), `speaker.warmup` (executor). Then `describe_ollama_models()` diagnostic.
4. `state.set(IDLE)` (Maya starts awake), then startup greeting: `ws_server.broadcast_animation("wave")` + `await speaker.speak(...)` — **before** the Listener exists.
5. `asyncio.TaskGroup`: `Listener.start()` + `queue_manager.run()` (**requires Python ≥3.11**).

**Runtime flow**
```
sounddevice callback thread → Listener._process_frame
   ├─ WakeWordDetector.feed_frame (every frame; active only while SLEEPING)
   └─ Silero VAD → utterance → run_coroutine_threadsafe(main.on_speech)
on_speech → [if not busy: FSM LISTENING + WS "listening"] → Transcriber.transcribe (Google STT, executor)
   → empty: back to IDLE (+WS idle, baseline behavior) | barge-in check (was_speaking snapshot + contains_wake_word) / sleep-trigger check
   → queue_manager.put(text)                       [core/queue_manager.py, maxsize=10, drops when full]
QueueManager.run → Processor.handle (core/processor.py)   [FSM → PROCESSING + WS "processing"]
   → IntentEngine.classify (sync) → context_manager.observe_user_turn
   → Router.dispatch (brain/router.py)   [pending power confirmation resolved FIRST]
        ├─ skill → returns "[tag] text" → Speaker.speak (Kokoro → WS; FSM SPEAKING → IDLE)
        └─ llm_service.query → returns ALREADY_SPOKEN (speaks itself)
   → ws: state idle + mood-baseline behavior
ws_server → frontend/js/websocket.js → avatar.js (audio/lip-sync), expression-composer.js, animations
```

| Component | File / symbol | Responsibility & key interactions |
|---|---|---|
| FSM | `core/state.py` `StateManager`, singleton `state` | States SLEEPING/IDLE/LISTENING/PROCESSING/SPEAKING/INTERRUPTED; `run_interruptible`, `interrupt()`, stop callback. |
| Capture | `core/listener.py` `Listener` | 512-sample frames @16 kHz; VAD prob>0.5; 25 silence frames end utterance; 6-frame pre-roll. Audio-callback thread → loop via `run_coroutine_threadsafe`. |
| Wake | `core/wake_word.py` | 2 s non-overlapping windows → Google STT; `compute_wake_triggers/contains_wake_word` shared with barge-in gate. |
| STT | `core/transcriber.py` | `recognize_google(language=config.stt.language)` in default executor; returns None on failure. |
| Intent | `brain/intent_engine.py` `IntentEngine` | Guards → ML ensemble → keyword fallback (see §3). |
| Router | `brain/router.py` `Router` | Static dict intent→coroutine; unknown/unrouted → `llm_query`. `dispatch` first calls `skills.system.power.resolve_pending` (pending shutdown/restart confirmation). |
| LLM | `services/llm/llm_service.py` | Ollama stream → phrase splitter/tag parser → `synth_q` → Kokoro → `play_q` → WS. |
| Model lifecycle | `services/llm/ollama_lifecycle.py` | `chat_keep_alive()`, per-turn `log_chat_turn()` (COLD/warm, load/prompt/gen tok/s, gap since last chat), background `log_residency()` (`/api/ps` + CUDA memory). Diagnostics only; never raises. |
| Context | `brain/conversation.py` `ContextManager` (singleton `context_manager`) | Topic/phase/goal state, open loops, semantic memory. |
| Mood | `core/mood.py` `MoodManager` (`mood_manager`) | Persistent angry/sad + transient tease. |
| Behavior | `core/behavior_engine.py` `BehaviorEngine.compose` | Tag+mood → packet incl. Fcl_* recipe via `core/expression_library.py`. |
| WS | `services/ws_server.py` `MayaWebSocketServer` (`ws_server`) | Broadcast + `audio_done` gate + interrupt handler; remembers last broadcast state and replays it to each new client. |
| Frontend | `frontend/js/{main,avatar,websocket,expression-composer,expression-controller,animation-controller,gaze-controller,life-motion-controller}.js` | See §8. |

---

# 3. BACKEND

**Input handling (`core/listener.py`, `main.py::on_speech`)**
- Listener runs VAD **even while SPEAKING** (no echo cancellation/gating in code). Each utterance spawns an independent `on_speech` task; queue order = STT *completion* order.
- `on_speech` (called at utterance **end**, so `listening` covers the STT round trip, not speech onset): snapshots `was_speaking = state.is_speaking()` and `began_listening = not state.is_busy()` before STT. Only when `began_listening`: FSM→LISTENING + WS `state:listening` (never overwrites PROCESSING/SPEAKING). Empty STT → `_end_listening()`: if FSM is still LISTENING → IDLE + WS `idle` + baseline behavior (clears the frontend's surprised-0.3 listening face). Non-empty: if `was_speaking` and `contains_wake_word(text)` → `on_interrupt()` (→ `state.interrupt()`); otherwise queued normally. Then `_SLEEP_TRIGGERS` (`go to sleep, sleep, goodbye, bye, stop listening`, **substring** match) → SLEEPING + goodbye line (`Speaker.speak` restores SLEEPING afterwards). Else `queue_manager.put`; the FSM is **left in LISTENING** and `Processor.handle` moves it to PROCESSING (no forced IDLE after `put`).
- Effective wake triggers: `config.wake_word="wake up Maya"` → set `{"wake up maya","maya","hey/hello/hi/yo maya"}`; because `"maya"` is in the set (substring match), **any text containing "maya" wakes/interrupts**.

**Queue/processor:** `QueueManager` single serial worker; `priority` field unused. `Processor.handle`: PROCESSING + WS state/transcript → `ConversationManager.add_user` → classify → `context_manager.observe_user_turn` → dispatch → if response ≠ `ALREADY_SPOKEN`: `add_assistant` (tagged text), transcript, `speaker.speak` (`greet` intent → `on_audio_start` broadcasts `wave`; a skill-set `intent["action"]` (from `perform_action`) → `on_audio_start` broadcasts that animation). Always ends with WS `idle` + baseline behavior; exceptions → IDLE. State sequence per command: `listening → processing → speaking → idle` (skill/LLM paths set FSM IDLE themselves).

**Intent engine (`brain/intent_engine.py`)** — `_predict` order:
1. Dismissal guard: **exact** match (trailing `.!?,` stripped) against `_DISMISSAL_PHRASES` → `dismissal`. No prefix/token-count matching.
2. `_PRESENCE_RE` → `smalltalk`. 3. `_ACTION_WORD_RE` (nod/giggl*/sigh/shrug/wink/wynk, whole word anywhere in the utterance) → `perform_action`.
4. Keyword-first if ≤3 tokens or `startswith` greeting prefix (and not comparison) → else `general_query` "short_input_fallback".
5. Ensemble avg softmax (BiLSTM `_build_pytorch_model`, CNN `_build_tf_model`), `_CONF_THRESH=0.65` else `_KEYWORD_RULES` (substring).
- `TRAINING_DATA` sha256 → `models/training_hash.txt`; mismatch/missing → auto-retrain on startup. `python -m brain.train_intent` = wipe + retrain + test print. 45 intent labels. `result["model"]` values (`pytorch+tensorflow`, `keyword_fallback`, `short_input_fallback`, `negation_guard`, `presence_guard`, `action_guard`, `keyword_short_input`) are consumed by `_should_play_filler`.
- `_extract_target` **returns the full utterance when no trigger strips** (i.e., `target` is almost never empty) — matters for context (§5).

**Router coverage:** every labelled intent is routed. `set_reminder` and `set_timer`/`cancel_timer`/`timer_status` → `skills/utilities/timer.py` (`reminder.py` is imported nowhere). `lock_screen` → `skills/system/lock_screen.py` (immediate, deliberately no confirmation — reversible). `shutdown`/`restart` → `skills/system/power.py` (confirmation flow, §9). Skill exceptions → `"[sad] Sorry senpai, I ran into a problem with that."`. `farewell` intent (e.g. "see you later") only replies; it does not sleep.

**Pending-confirmation intercept (`Router.dispatch`):** before intent routing, `await resolve_pending(raw_text)`. If a power request is pending it is **consumed by the next utterance** (one-shot, 30 s TTL): confirm phrase → OS action + spoken reply; deny phrase → "cancelled"; anything else → request dropped and the utterance routes normally.

**Barge-in (`core/state.py`, `main.py`, `ws_server.py`, `avatar.js`)** — `interrupt()` only when SPEAKING: state→INTERRUPTED, stop callback (`sd.stop()` + `broadcast_stop_audio`), cancel `_current_task`, state→LISTENING. Only `llm_service.query()` registers a task via `run_interruptible`; **`Speaker.speak()` is never wrapped** (processor/main call it directly). The filler and the PROCESSING window before the first phrase are not SPEAKING, so they cannot be barged into. After a voice barge-in the interrupting utterance is queued, so the FSM proceeds LISTENING → PROCESSING normally.

**Mood:** see §7. **Skills:** §9. **Error handling:** most paths log and degrade (memory/embedding `try/except` → recent-context-only; Kokoro timeout → phrase dropped; `ws_server._on_message` swallows all exceptions).

**Blocking-on-loop calls (violates project rule "blocking → executor"):** `IntentEngine.classify`; `SQLiteVectorStore.search/has_records`; `expression_library.save_recipe` (file write); `system_info` CPU branch (`psutil.cpu_percent(interval=1)`); screenshot branch (`pyautogui.screenshot`); `Speaker._synthesise_guarded` pipeline rebuild (`_build_kokoro_pipeline`, model load). (`power._run` correctly uses the executor.)

---

# 4. LLM + RESPONSE PIPELINE

- **Config:** `config.llm`: `provider="ollama"` (unused), `model="llama3.2"`, `base_url=http://localhost:11434`, `max_tokens=150`→`num_predict`, `temperature=0.7`. `LLMConfig.system_prompt`, `provider` and `api_key` are **unused** (marked in `settings.py`); the prompt is the module constant `llm_service._SYSTEM_PROMPT`. `config.llm.keep_alive` is not a dataclass field (read via `getattr`, see §10).
- **Request:** `/api/chat` via `httpx.AsyncClient(timeout=60)`, `stream=True`, `aiter_lines`. **`keep_alive` is sent on every chat request and on the chat warmup** via `ollama_lifecycle.chat_keep_alive()` (default `"60m"`; `config.llm.keep_alive` override, numeric strings like `"-1"` are sent as numbers). Embeddings use `keep_alive:"30m"`; `brain/embeddings.py` docstring assumes `OLLAMA_MAX_LOADED_MODELS=2` (server-side, not in repo).
- **Per-turn diagnostics:** on the final stream object `log_chat_turn(data)` logs `[TIMING] chat turn: COLD|warm load=… prompt=…tok@…tok/s gen=…tok@…tok/s gap_since_last_chat=… keep_alive=…` (COLD = `load_duration ≥ 1.0 s`; a cold load with a gap shorter than keep_alive implies eviction, not expiry) and schedules a background `/api/ps` + CUDA-memory snapshot.
- **Prompt assembly (`_stream_and_speak`):** `_SYSTEM_PROMPT + mood_manager.system_prompt_note() + context_package.as_system_note()` + `context_package.recent` (last `config.context.recent_turns=6` turns from global `memory`, already containing the current user turn — **never re-append the user turn**).
- **Prompt content:** persona ("under 3 sentences"), `[emotion]` tag per sentence, optional `[attitude:x][intensity:x]`, one optional leading `*action*` (closed: nod/giggle/sigh/shrug/wink), CAPS word-stress rules.
- **Streaming (`_ollama_streamer`):** tokens → `_next_boundary(buffer)` (split at `.!?,;—` followed by whitespace/buffer-end, or after 12 words; comma before the vocative `config.user_name` is skipped; waits when next word not yet streamed) → `_emit` → `_parse_expression` (`*action*` extracted first; then consecutive leading `[..]` tags; `_ANY_BRACKET_RE` strips other brackets) → `synth_q` item `(text, expression, actions, is_final, attitude, intensity)`. Expression/attitude/intensity carry forward across phrases of one sentence until `is_final`. `*...*` spans are removed entirely (so markdown emphasis words are dropped, not just unstyled). A `.!?` at the very end of the buffer counts as a boundary (§11 #5: decimals split).
- **Workers:** `_synth_worker` (serial; `_enhance_prosody` → `_run_kokoro(_synthesise_blocking)`), `_play_worker` (serial; per phrase: `broadcast_animation`(actions) → `broadcast_behavior(compose(...))` → state `speaking` → `broadcast_audio` → `wait_for_audio_done` → baseline behavior). At `_DONE`: `state.set(IDLE)` + WS idle.
- **Filler:** `_should_play_filler` (intent ∈ {general_query, unknown}; model source lacks "fallback/keyword/guard"; not `_NO_FILLER_RE`; ≥3 words) → `asyncio.gather(_play_filler(), _stream_and_speak())`; `_filler_done` event gates only the first real phrase.
- **History:** after reply, `_last_response` (module-level list) → `_conv.add_assistant(clean text)`, transcript broadcast, `context_manager.record_assistant_turn`. Empty (cancelled before the streamer finished) → nothing recorded. Generation normally finishes long before playback, so a barge-in during playback still records the **full** reply (§11 #7).
- **Mood hooks:** `observe_user_text` at top of `query()` (LLM-routed turns only); `observe_turn(turn_expressions)` once at end of streamer (skipped on cancel).
- **Latency chain (logged `[TIMING][TTFA]`):** classify → `build_context_package` (may await an embedding call) → Ollama first token → first phrase boundary → Kokoro synth → WS → browser decode/play.
- **Not implemented:** Edge-TTS/XTTS, cloud LLM providers, tool-calling, offline STT.

---

# 5. MEMORY

| Layer | Implementation | Persistence |
|---|---|---|
| Recent window | `brain/memory.py` `Memory` singleton `memory` (max 50 entries); wrapped by 3 `ConversationManager` instances (processor, llm_service `_conv`, ContextManager) — all share `memory`. `get_history(last_n)`; LLM gets last 6. | In-process only |
| Conversation state | `ContextManager._state` (`ConversationState`: topic, topic_history≤5, goal, task, constraints, decisions, entities≤10, phase, last_intent) | In-process |
| Open loops | `_open_loops`; created on topic switch/return with pending unanswered question, or `_DEFERRAL_CUE_RE`; resolved by cues/dismissal/`concluding` | In-process |
| Semantic long-term | `brain/vector_store.py` `SQLiteVectorStore` at `~/Maya/Memory/semantic_memory.sqlite3` (`config.context.memory_dir=None`) | SQLite |
| Notes | `skills/utilities/notepad.py` `.txt` in `~/Maya/Notes` | Files |

**Flow:** `Processor` → `observe_user_turn` (every command, state only) → LLM path: `build_context_package(question)` (recent, state, `_relevant_open_loops` keyword overlap, `_retrieve_semantic`, `_resolve_reference`) → `as_system_note()` (empty sections omitted) → after reply `record_assistant_turn` (LLM turns only): resolve loop; `_memory_candidate` → `_persist_memory`.
- **Classification/write policy:** only when `_REMEMBER_CUE_RE` matches the *user* text and intent ∉ `_NOISE_INTENTS`; `mem_type`="preference" (if `_PREFERENCE_RE`) else "fact", importance 0.8, content = raw user question. Type set `VALID_MEM_TYPES` also lists goal/decision/relationship/project but **nothing produces them**. No episodic memory. History also stores skill replies with `[tag]` prefixes (LLM replies are stored tag-stripped) and `query()` error strings as assistant turns (§11 #17).
- **Embeddings:** `OllamaEmbedder.embed` → `/api/embeddings` (`nomic-embed-text`), pooled client (`keepalive_expiry` 300 s), 20 s timeout, 64-entry exact-text LRU, 404 → sticky `_unavailable`. Calls ≥2 s log a warning + `/api/ps` snapshot; every call logs gap/in-flight diagnostics. `MAYA_EMBEDDING_DEVICE=cpu` → `options.num_gpu=0`.
- **Retrieval (`_retrieve_semantic`):** skipped if store None, short followup/confirm/dismissal turn (≤6 tokens), or store empty (`has_records()`); else embed → brute-force cosine over all rows → top `max_semantic_memories=3` ≥ `similarity_threshold=0.75` → drop hits newer than `semantic_recency_guard_seconds=120`.
- **Dedup:** `find_similar(threshold=0.92)` takes only the single best match (`top_k=1`) and requires same `mem_type` **and same `topic`** (when topic non-empty) → update in place, else insert.
- **Compaction:** `Memory.on_evict` → `_on_memory_evict`; every 10 evicted turns → one `conversation_summary` record: `"Earlier discussion touched on: "` + alphabetically-first 15 keywords (`asyncio.create_task`, unreferenced).
- **Degraded behavior:** any failure → empty semantic list / recent-only; store init failure → `_store=None`.
- **Known data-quality issue (Confirmed):** because `intent["target"]` ≈ full utterance, `_infer_topic` = first 40 chars of the utterance, `_extract_entities` appends the whole utterance as an "entity", and `_resolve_reference` returns `entities[-1]` = *the current utterance* — so `TOPIC`, `ENTITIES`, `LIKELY REFERRING TO` in the system note are largely degenerate, and topic-scoped dedup rarely matches.

---

# 6. TTS + AUDIO

- **Config (`config.tts`):** `voice="af_sky"`, `voice_blend="jf_alpha"`, `blend_ratio=0.92` (→ 8% af_sky / 92% jf_alpha; the "35%" comment is wrong), `lang_code="a"`, `speed=1`, `output="avatar"`, **`device="cpu"`** (keeps GPU free for Ollama), `cpu_threads=1` (**unused**, marked in `settings.py`). `_resolve_tts_device()` returns `config.tts.device` unless it is `"auto"`/empty (then cuda if available else cpu); `KPipeline(device=...)` is only passed if the installed kokoro accepts it (else warning + library default). `_log_kokoro_device`/`_log_cuda_memory` log placement at init/warmup.
- **Two independent `KPipeline`s:** `Speaker._pipeline` (skills/greeting/wake/sleep lines) and `llm_service._kokoro_pipeline` (LLM stream, filler, timer alert), both built via `_build_kokoro_pipeline`. Both warmed at startup. With `device="cpu"` the duplication is RAM only; with `cuda`/`auto` it is VRAM too.
- **Synthesis:** all Kokoro calls via `llm_service._run_kokoro` (new **daemon thread** per call, 15 s timeout, `_reset_kokoro_pipeline()` on timeout → returns None; `Speaker._synthesise_guarded` rebuilds its own pipeline on any None result). Output float32 24 kHz → `_numpy_to_wav` (executor) → base64 in JSON.
- **Prosody (`_enhance_prosody(text, expression, is_final)`):** `_elongation_re_sub` (single definition) → `_fix_caps` (+`_CAPS_WHITELIST`) → partial chunk: strip trailing punct only; final: `_expand_short_exclamation` → per-expression punctuation/fragmentation (`_fragment_for_energy` caps at 4 fragments). `EXPRESSION_SPEED` (0.84–1.13) applied **only** in `llm_service._synthesise_blocking`, not in `Speaker._synthesise`.
- **Playback:** one WAV per phrase; browser `speakFromBytes` (WebAudio `AudioBufferSourceNode`), `source.onended` → `_onAudioDone` → WS `{"type":"audio_done"}` → `ws_server._audio_done_event`. `wait_for_audio_done(timeout=30)` **returns False immediately when no client is connected**; otherwise it **clears the single shared Event on entry**, warns and continues on timeout. `output="local"` uses `sounddevice`; `"both"` double-plays with the avatar (must stay `"avatar"` when frontend runs).
- **Speaker path (`core/speaker.py::speak`):** first valid `[tag]` only (`_strip_tags`), whole text one synthesis, `mood_manager.observe_turn([expr])`, snapshots `was_sleeping`, sets SPEAKING, broadcasts behavior/state/audio (calls `on_audio_start` just before audio), waits `audio_done`, `finally`: state **SLEEPING if it was sleeping on entry, else IDLE** + WS `idle` + baseline behavior.
- **Barge-in audio:** `stopCurrentAudio()` sets `onended=null` (no `audio_done` is sent after a stop) and stops lip-sync.
- **Concurrency:** filler + first phrase synthesize concurrently on the same module pipeline (thread-safety UNVERIFIED); `_get_kokoro()` lazy init has no lock; Ollama/Kokoro/embedding model may share GPU if `tts.device` is changed from `"cpu"`.
- **Known problems:** see §11 (30 s stall residuals; wave-vs-speech sync — wave fires in `on_audio_start` *after* synthesis).

---

# 7. MAYA BEHAVIOR / MOOD / EXPRESSIONS

**Vocabularies (closed):** emotion tags `happy|sad|angry|surprised|relaxed|neutral|excited`; attitude `sincere|playful|teasing|mock`; intensity `low|medium|high`; actions `nod|giggle|sigh|shrug|wink`; gaze `direct|soft|away`. Duplicated in `llm_service.py`, `speaker.py`, `behavior_engine.py`, `expression_library.py`, prompt text. **Deliberately not expanded** to make stored recipes reachable.

**Mood (`core/mood.py`):** tracks only `angry`/`sad` (`_STICKY_MOODS`) as `(mood, intensity)`; transient tease (`temp_mood`, 90 s). Events: USER (`observe_user_text`: provocation, ragebait = provocation+`_TEASING_MARKERS_RE` → transient, sadness, frustration, excitement eases, apology forgives), SKILL/SYSTEM (`report_event`, e.g. `system_info` battery/CPU/RAM), MAYA tags (`observe_turn`: only *confirm* an event raised this turn, +0.15). Decay: `_DISTRACTION_DECAY` per unrelated turn, `_TIME_DECAY_PER_MIN`, 20-min hard forget. `baseline_expression()` = resting face (used after every phrase/skill/idle). `system_prompt_note()` injected each turn. `is_teasing()` read by behavior engine. **`observe_user_text` is called only from `llm_service.query()`**, so skill/built-in-routed utterances never trigger mood events. Events left in `_pending_events` by a cancelled LLM turn (no `observe_turn`) are consumed by the next `observe_turn`, including a skill line's.

**Behavior engine (`BehaviorEngine.compose(expression, actions, source, attitude, intensity)`)** → packet `{primary, secondary, intensity, attitude, gaze, actions, recipe}`. Attitude: valid Ollama attitude else `"teasing"` if `is_teasing()` else `"sincere"`. Secondary: personality bias (`_SECONDARY_BIAS`), `happy` for mock/teasing angry, or active mood bleed (>0.3). Intensity: `0.45 + mood*0.25 + 0.65*0.25 + jitter`, blended 70/30 with Ollama word (0.3/0.6/0.9); bucketed low(<0.4)/medium(<0.7)/high. Recipe: `get_recipe(emotion, attitude, band)` else `compose_default` **and `save_recipe` (writes `frontend/assets/expressions.json`)**; bounded jitter (±0.04·chaos) applied per call, never persisted. `PERSONALITY` dict duplicated client-side in `expression-composer.js`.

**`expressions.json`:** flat `{"emotion|attitude|intensity": {Fcl_*: weight}}`, 35 entries in this snapshot (grows as the backend persists generated recipes). The backend only requests the 7 emotions × attitudes sincere/playful/teasing/mock × bands low/medium/high, so **22/35 recipes are unreachable at runtime — intentional (decision):** they are Lab-only calibration/experimental entries (attitudes like `default`/`gloating`, emotion `scared`, band `extreme`). No `default` fallback is added; runtime lookup stays deterministic. `expression_library._load` returns `{}` on any parse error and `save_recipe` then overwrites the whole file (non-atomic `write_text`) — §11 #10. Inspect the file when needed.

**Expression keys referenced by code**
- *Recipe morph keys referenced by the code (raw mesh morph targets; `expression-composer.js` is written to drop any key not found on the loaded VRM):* `Fcl_BRW_{Angry,Joy,Sorrow,Surprised,Fun}`, `Fcl_EYE_{Angry,Joy,Joy_L,Fun,Sorrow,Surprised,Spread,Natural,Close_L,Close_R}`, `Fcl_MTH_{Joy,Large,Angry,Sorrow,Surprised,Down,Neutral,Fun,Up}`. `Fcl_MTH_{A,I,U,E,O}` are excluded (lip-sync). Existence on the actual VRM is UNVERIFIED.
- *Legacy six-knob (`RENDERABLE`, expression-manager keys):* `neutral, joy, fun, angry, sorrow, surprised`.
- *Other expression-manager keys used in `avatar.js`:* `blink`, `blinkLeft` (wink), `aa ee ih oh ou` (lip-sync), `happy` (0.08 during speech), `relaxed` (BASE 0.4/0.6), `surprised` (0.3 on `listening`). Note the **naming inconsistency**: composer uses VRM0-style `joy/fun/sorrow`; avatar.js uses VRM1-style `happy/relaxed`. Which set the model exposes is UNVERIFIED (`expressionManager.setValue` on an unknown name silently no-ops).

**LLM → face:** Ollama `[happy][attitude:playful][intensity:high]` → `_parse_expression` → `_play_worker` → `behavior_engine.compose` (+ mood/recipe) → `ws_server.broadcast_behavior` `{"type":"behavior",...}` **before** the phrase audio → `websocket.js` → `expression-composer.js::applyBehavior`: if ≥1 recipe key exists on a mesh → `_animateRecipeTo` (writes `morphTargetInfluences` directly, 220 ms smoothstep, then stops writing) and legacy knobs eased to 0; else `_composeWeights` six-knob path via `expressionController` EMOTION layer. `gaze` → `applyBehavioralGaze` (eye offset for 0.9–1.4 s, lower priority than speaking/screen gaze — mostly visible only between phrases). Actions go separately as `animation` messages.

**Expression Lab (`frontend/expression-lab.html` + `js/expression-lab.js`):** dev-server-only page (not a Vite build input). Edits persist to `localStorage["maya.expressionLab.recipes"]` (schema v1); Import merges an `expressions.json`; **Export writes only the localStorage set** (drops backend-generated/never-imported entries) — Import first. Switching combinations saves the outgoing combination's recipe even if never edited, so merely browsing pins the generated default into the export (§11 #16). Its `composeDefault` mirrors `core/expression_library.py` (duplicated intentionally). Attitude list is wider than the backend's.

---

# 8. FRONTEND / AVATAR

- **Runtime:** Electron `^31.7.7` + Vite `^8.0.16` (lock 8.0.16, rolldown; needs Node `^20.19 || >=22.12`) + `vite-plugin-electron ^0.29.0` (`vite.config.js`, `base:"./"`, entry `electron/main.js`), `three ^0.184.0`, `@pixiv/three-vrm ^3.5.3`, `@pixiv/three-vrm-animation ^3.5.5` (lock nests its own `three-vrm-core` 3.5.5 beside top-level 3.5.3). Scripts: `dev`, `build`, `preview` only. `frontend/package_electron.json` is an unreferenced older manifest.
- **Window (`electron/main.js`):** 820×440, x=-260 (bottom-left), transparent/frameless/always-on-top `"screen-saver"` with a 500 ms `keepOnTop` interval, `skipTaskbar`, `setIgnoreMouseEvents(true,{forward:true})` (click-through; injected drag CSS is inert). Loads `VITE_DEV_SERVER_URL` or `../dist/index.html`.
- **Scene (`main.js`):** FOV 18 camera (-0.05,1.35,2.5), transparent renderer, 3 lights; animate loop: render → `vrm.update` → `updateVrmaAnimations(delta)` → `updateGaze(delta)`.
- **VRM load (`avatar.js::loadAvatar`):** `assets/mayaaa.vrm`; `expressionController.attach`; eyes closed, arms posed, `startHeadMovement()` always running (also calls `lifeMotionController.update`). `window.vrm` and `window.maya.*` console helpers exposed.
- **Sleep/wake:** `wakeAvatar()` on WS open (blink loop, eye loop, idle fidgets), `sleepAvatar()` on WS close. **Tied to socket connection only — backend SLEEPING state is never sent to the frontend** (the avatar stays visually awake while Maya is voice-slept). Re-wake is meant to be idempotent: `startBlinking()` clears `_blinkTimeout` before starting; `startEyeMovement()` runs once per page (`_eyeLoopStarted`) (reconnect leak fixed).
- **WS (`websocket.js`):** reconnect every 2 s; handles `audio, state, behavior, stop_audio, transcript(no-op), animation(wave|nod|giggle|sigh|shrug|wink)`; sends `audio_done` only. No `interrupt` sender exists in the frontend. URL hard-coded `ws://localhost:8765`. `handleState` forwards every value to `setAvatarState` (fidget gate, resets/starts `_idleSince`) and, for `listening`, sets `surprised` 0.3 via `setExpression`; the next `behavior` packet overwrites it.
- **Layering:** `expression-controller.js` (BASE<EMOTION<ACTION<LIPSYNC<BLINK per key; unresolved key → 0); `animation-controller.js` (bone owner tiers BASE<FIDGET<ACTION, `canWrite/claim/release`, 350 ms `handoffProgress`); `life-motion-controller.js` (BASE-tier breathing/posture/shoulders/hips, scaled down while speaking/observing); `gaze-controller.js` (attention session/boredom; `observeScreenActivity` has **no caller except `window.maya`** — no screen observer exists).
- **Lip-sync:** AnalyserNode (fft 256, 5 bands) → `aa/ee/ih/oh/ou` + `happy` on LIPSYNC layer; token-guarded loop.
- **VRMA (`playVrmaAnimation(name)`):** `_VRMA_ASSETS` maps animation names to `.vrma` files (filenames often differ from names — read the map; all mapped files exist as LFS pointers); URL-cached loads; `createVRMAnimationClip` → keep only `.quaternion` (minus shoulders via `_PROTECTED_BONE_KEYS`) and `.weight` tracks; fade in/out; claims bones via `animation-controller` (FIDGET tier for `_FIDGET_VRMA_NAMES`, else ACTION); cleanup `setTimeout` does `action.stop(); mixer.update(0)`; a non-fidget play halts active VRMA fidgets first (procedural `headTilt`/`shoulderRoll` are not halted). `wink`, `headTilt`, `shoulderRoll` are procedural (not VRMA).
- **Wired automatically:** server `animation` messages (wave/nod/giggle/sigh/shrug/wink) and the fidget pool. Other registered clips are console-only via `window.maya`.
- **Fidgets:** randomized scheduler (7–18 s) gated by `_canFidgetNow` (awake, not speaking, backend state `"idle"`, nothing else playing, past a 5-min post-launch calm period). Per-fidget cooldowns/min-idle (`_FIDGET_COOLDOWN_CONFIG`, `_FIDGET_MIN_IDLE_MS`), a forced `waving` after 30 min idle, screen-attention suppression (`_gazeFidgetFactor`) and a stuck-state clear are all in `avatar.js`; tuning is still in flux.
- **UI that exists:** only `#ws-status` (💤 when disconnected) in `index.html`, plus the dev-only Expression Lab panel. No transcript overlay/cards.

---

# 9. SKILLS / TOOLS

Convention: `async execute(intent, text) -> str` returning `"[tag] text"`; `perform_action` also mutates `intent` (`intent["action"]`, consumed by `Processor`).

| Skill (file) | Trigger | Behavior / output | Notes & failures |
|---|---|---|---|
| Open website (`skills/web/open_website.py`) | `open_website` | `webbrowser.open` from 8-site `_SITES` (substring match on the utterance); else `https://{target}` | Unknown sites (amazon/linkedin/stackoverflow are in TRAINING_DATA) → invalid `https://amazon`. |
| Google search (`web/google_search.py`) | `search_web` | opens Google query from `intent["target"]` | Prompts if empty (target is rarely empty, §3). |
| Weather (`web/weather.py`) | `get_weather` | Open-Meteo + geocoding, ip-api.com auto-location (executor, 8 s) | Emits 3 tags; Speaker keeps only the first. Location regex runs to end of string ("in London today" → "London today"). `_WMO` lacks codes 56/57/66/67/77/85/86 → "unknown conditions" + neutral. |
| Open app (`system/open_app.py`) | `open_app` | `os.startfile(target)` (Win) | No name mapping; untagged reply; failure → "couldn't open". |
| Lock screen (`system/lock_screen.py`) | `lock_screen` | `ctypes.windll.user32.LockWorkStation()`; non-Windows → "[sad] I can only lock the screen on Windows" | **Immediate by design** (no confirmation — reversible). Windows-only; not runtime-tested. |
| Power (`system/power.py`) | `shutdown`, `restart` | `execute` only **asks** ("Say yes to confirm") and stores `(intent, expiry)`; `resolve_pending(text)` (called first by `Router.dispatch`) confirms/declines. Confirm → `shutdown /s` or `/r /t 10` (executor, no `/f`) + goodbye line. Non-Windows → apology, nothing pending | Constants `_DELAY_S=10`, `_CONFIRM_TTL_S=30`. Confirm regex: yes/yeah/yep/yup/confirm(ed)/affirmative/do it/go ahead/proceed (+ optional "please", "maya"/"senpai"); deny: no/nope/nah/cancel/abort/never mind/don't. No voice abort — use `shutdown /a` during the 10 s countdown. Not runtime-tested. |
| System info (`system/system_info.py`) | `system_info`, `screenshot` | psutil battery/cpu/ram/disk; battery/CPU/RAM extremes call `mood_manager.report_event(source="skill", "angry")`; screenshot (word "screenshot" **or** intent `screenshot`) saves `screenshot_YYYYMMDD_HHMMSS.png` to `%OneDrive%/Pictures/Screenshots` (else `~/Pictures/Screenshots`, folder auto-created) | CPU branch blocks loop 1 s; "ram" substring matches "program". |
| Clipboard (`system/clipboard.py`) | `clipboard_*` | pyperclip read(200-char)/write/clear | Top-level `import pyperclip` — missing package crashes startup (Router import); now listed in requirements. |
| Media (`media/play_music.py`) | play/pause/next/prev/volume/mute | `keyboard.send` media keys (volume ×5) | Guarded import; message if missing. |
| Date/time (`utilities/datetime_skill.py`) | `get_time`, `get_date` | formatted local time/date | — |
| Timer (`utilities/timer.py`) | `set_timer`, `cancel_timer`, `timer_status`, **and `set_reminder`** (router maps it here) | named/numbered asyncio timers; `_parse_duration` uses one pattern per unit (`hours?\|hrs?`, `minutes?\|mins?`, `seconds?\|secs?`; digits only — "one hour" is not parsed); cancel/status word checks are skipped for `set_reminder`; `_countdown`'s `finally` removes only its own `_timers` entry; `_alert` speaks via llm_service Kokoro and broadcasts `idle` afterwards | Alert always says "Your {timer name} is done" — reminder text is discarded; "remind me to X" without a duration → "How long should I set the timer for". Label regex's lookahead is prefix-based (labels starting a/an/me/the/for/set/min/sec/hour dropped). Alert skips Speaker/queue/state, is not interruptible, shares the single `_audio_done_event`. |
| Reminder (`utilities/reminder.py`) | **none — dead code** | `asyncio.sleep` then `print` only | Not imported by `brain/router.py`; README/architecture wrongly describe it as the `set_reminder` handler. Do not "fix" it without deciding whether to delete it. |
| Notepad (`utilities/notepad.py`) | `note_*` | `.txt` in `~/Maya/Notes`; `_TIMESTAMP_RE` stripped on read; a `note_*` intent dispatches directly to its handler (word matching is only a fallback for other intents); `_extract_content` strips only the **leading** command phrase (verb + optional "a/the/my… note"/"down"), so later verbs in the content survive | `_delete` removes latest note without confirmation. |
| Perform action (`system/perform_action.py`) | guard-routed | picks nod/giggle/sigh/shrug/wink (+`wynk`), sets `intent["action"]`, returns confirmation; `Processor` fires `broadcast_animation(action)` via `on_audio_start` | Fires only via the `Speaker.speak()` avatar/both path. The guard fires on any whole-word occurrence (e.g. "what does nod mean"). |
| Built-ins (`brain/router.py`) | greet/farewell/thanks/help | canned strings; greet triggers wave | — |

---

# 10. CONFIGURATION + RUNTIME REQUIREMENTS

- **Config:** `config/settings.py` singleton `config` (`MayaConfig`): `name="Maya"`, `user_name="senpai"`, `wake_word="wake up Maya"`, `log_level="INFO"`, `log_dir="logs"`, `ws_host="localhost"`, `ws_port=8765`. `audio`: 16000 Hz, mono, `chunk_ms=30` (clamped up to 512 samples), `silence_ms=800`, `pre_roll_ms=200`, `device_index=None`. `stt`: only `language="en"` is used. `tts`: see §6. `llm`: `model="llama3.2"`, see §4. `context`: `recent_turns=6, max_open_loops=3, max_semantic_memories=3, similarity_threshold=0.75, dedup_threshold=0.92, semantic_recency_guard_seconds=120, embedding_model="nomic-embed-text", memory_dir=None`.
- **Unused fields (kept as config/API surface, commented `UNUSED` in `settings.py`):** `STTConfig.model_size/device/compute_type`, `LLMConfig.provider/api_key/system_prompt`, `TTSConfig.cpu_threads`. `api_key` is reserved (Ollama needs no key) — candidate for removal in a dedicated config cleanup, not during behavioral work.
- **`getattr`-only settings (not defined in dataclasses):** `config.context.embedding_device`, `config.notes_dir`, `config.llm.keep_alive` (default `"60m"`).
- **Env vars:** `MAYA_EMBEDDING_DEVICE` (`cpu` → embeddings `num_gpu=0`), `HF_HUB_OFFLINE` (`setdefault "1"` in `llm_service.py`), `TF_CPP_MIN_LOG_LEVEL` (`setdefault "3"`), `VITE_DEV_SERVER_URL` (Electron), `OneDrive` (screenshot path). Secrets: none in repo (`LLMConfig.api_key=""` unused) → nothing to `[REDACTED]`.
- **Ports/services:** WS 8765; Ollama 11434 with `llama3.2` and `nomic-embed-text` pulled; Vite dev server default 5173 (not set in config); external: Google STT, Open-Meteo (+geocoding), `ip-api.com` (HTTP), `torch.hub` `snakers4/silero-vad`, HF cache for Kokoro voices.
- **Runtime:** **Python ≥3.11** (`asyncio.TaskGroup`); Node per Vite 8; Windows 11 (`os.startfile`, `keyboard`, `ctypes.windll`, `shutdown.exe`, `OneDrive`). GPU optional: used by Ollama; Kokoro runs on CPU unless `config.tts.device` is changed.
- **Dependencies:** `config/requirements.txt` includes `websockets>=12.0` (unpinned; the `websockets.server.WebSocketServerProtocol` import is a legacy path on newer releases) and `pyperclip>=1.8.2`; still lists `wikipedia` (unused).
- **Paths:** `models/` (auto-created, gitignored), `logs/maya.log` (`logs/` auto-created by `main.py`), `~/Maya/Notes`, `~/Maya/Memory/semantic_memory.sqlite3`, `~/Pictures/Screenshots` or `%OneDrive%/Pictures/Screenshots`, `frontend/assets/{mayaaa.vrm,expressions.json,vrmas/*.vrma}` (LFS via `.gitattributes`; `mmodel.vroid` unused by code). `.gitignore` ends with stray `0.9.0`/`12.0` lines (pip `>=` redirect artifacts, likely actually named `=0.9.0`/`=12.0`, so they may not match) and starts with a BOM.
- **Invariant:** `config.tts.output` must be `"avatar"` while the frontend runs.

---

# 11. CURRENT PROBLEMS / FRAGILE AREAS

### Resolved (Batch 1–4; statically traced, not runtime-tested)
- Timer "N minutes/seconds" double-count → one pattern per unit (`timer._parse_duration`).
- `perform_action` animation → `Processor` broadcasts `intent["action"]` via `on_audio_start`.
- `websockets`/`pyperclip` added to `config/requirements.txt`; `logs/` auto-created in `main.py`.
- Dismissal guard → exact-phrase match (no more `startswith` misroutes of "note…", "nod…", "stop the timer", etc.).
- Screenshot skill → timestamped file path, folder auto-created, broken `logging.log` removed; also triggers on intent `screenshot` ("capture my screen").
- Sleep persistence → `Speaker.speak()` restores SLEEPING when it was sleeping on entry; `main.py` sets IDLE before the startup greeting so Maya still starts awake. Wake word is now actually active while asleep.
- Chat model idle unload → `keep_alive` sent on every chat request + warmup (`ollama_lifecycle.chat_keep_alive()`); per-turn COLD/warm + residency logging added.
- `lock_screen` intent → routed to `skills/system/lock_screen.py`; `shutdown`/`restart` implemented with spoken confirmation (`power.py`), no longer answered by the LLM.
- `wait_for_audio_done` no longer waits 30 s when no frontend client is connected (fixes the startup-greeting stall).
- Notepad: a `note_*` intent dispatches directly to its handler; `_extract_content` no longer cuts content after a later verb.
- `config.tts.device` defined (`"cpu"`); `config.llm.model` = `llama3.2`.
- `ws_server` stores `_last_state` (updated on every `broadcast_state`, even with no clients; default `"idle"`) and sends it to each new client → frontend `_currentBackendState` is no longer `null` until the first broadcast (fidgets can run after connect).
- `listening` is broadcast from `main.on_speech` (only when not PROCESSING/SPEAKING); empty STT resets to IDLE; FSM no longer force-set to LISTENING/IDLE over an in-flight PROCESSING turn; sequence is `listening → processing → speaking → idle`.
- `set_reminder` vs `set_timer` "nondeterministic skill": **not a bug** — router maps both to `timer`; only `reminder.py` is dead code. The real defect is dropped reminder text (§11 open).
- Duplicate dead `_elongation_re_sub` removed from `llm_service.py`.
- Timer name race → `_countdown`'s `finally` removes only its own `_timers` entry; `_alert` now broadcasts `idle` afterwards; `timer.execute` skips cancel/status word checks for `set_reminder` (no more "remind me to stop by…" hijack).
- Frontend reconnect leak → `startBlinking()` clears `_blinkTimeout`; `startEyeMovement()` starts once per page (`_eyeLoopStarted`).

### Confirmed (deterministic from source; not runtime-tested) — still open
1. **Timer alert:** `_alert` is not interruptible, bypasses Speaker/queue/state, and shares the single `_audio_done_event` with any concurrent speech. Reminder-wording gaps: #18.
2. **30 s `audio_done` stalls (residual):** `wait_for_audio_done` still waits the full timeout when (a) a client is connected but browser decode fails or `vrm` not loaded (no `audio_done`); (b) barge-in during any `Speaker.speak` path (`stopCurrentAudio` nulls `onended`, task not registered, so it isn't cancelled). Queue worker is blocked meanwhile. (The developer reported the stall fixed; in the inspected source only the no-client case is evidenced.)
3. **Code-hygiene traps:** `_strip_markdown` does not exist (older docs claim it); unused config fields (§10); `skills/utilities/reminder.py` unreferenced.
4. **Data-quality:** degenerate context topic/entity data (§5); 22/35 `expressions.json` entries Lab-only (§7, decision).
5. **Decimal/version numbers split mid-token:** `_next_boundary` only skips a `.` not followed by whitespace when more buffer exists; a `.` that is the last char of the buffer is accepted. Ollama streams "3", ".", "14" as separate tokens, so "3.14" becomes phrases "…3." and "14…" (spoken as two sentences). Same for "v1.2", "google.com".
6. **Spurious Kokoro pipeline rebuild on the event loop:** `Speaker._synthesise_guarded` rebuilds its pipeline (`_build_kokoro_pipeline`, model load, synchronous) whenever `_run_kokoro` returns `None` — including the non-timeout "Kokoro returned no audio chunks" case (e.g. a punctuation/emoji-only skill line). `_run_kokoro`'s timeout branch also resets the *llm_service* pipeline even when Speaker's was the stuck one.
7. **Barge-in leaves a full reply in history:** generation finishes long before playback, so `out_text` is already appended when a barge-in cancels playback; `query()` then records the complete reply (and the semantic-memory write policy sees it) although Maya only said part of it.
8. **Barge-in impossible during PROCESSING/filler:** `interrupt()` requires SPEAKING; the filler (`_play_filler`) and the pre-first-phrase window never set SPEAKING, so calling her name then just queues a command.
9. **`_expand_short_exclamation` alters spoken words mid-sentence:** it runs on any *final* phrase chunk that is one word, so "Well, yes." → chunk "yes." → "Oh yes, absolutely!".
10. **`expression_library` can wipe calibrated recipes:** `_load` swallows any read/parse error and caches `{}`; the next `save_recipe` (any newly generated key) rewrites `expressions.json` with only that entry. Writes are non-atomic (`write_text`), so a crash mid-write produces exactly that state. Back up the file before editing it outside the Lab.
11. **Unreferenced tasks:** `asyncio.create_task(ws_server.serve())` (`main.py`) can be garbage-collected (it only awaits a private Future), which would close the server; a startup failure (e.g. port 8765 in use) is never surfaced. `ContextManager._flush_evicted_buffer` has the same pattern (`ollama_lifecycle`/`embeddings` correctly keep `_bg_tasks`).
12. **`ws_server._broadcast` iterates `self._clients` while awaiting `ws.send`:** a client connecting/disconnecting during a broadcast raises `RuntimeError: Set changed size during iteration` (Potential; iterate over a copy).
13. **Listener stale utterance after sleep:** `_process_frame` returns early while SLEEPING without touching `_in_speech/_speech_buffer/_silence_count`; if sleep began mid-utterance, the old buffer is delivered to STT after the next wake.
14. **Stale fidget timeouts (Potential):** halting a VRMA fidget for an ACTION animation leaves its fade/cleanup `setTimeout`s pending; if the same fidget replays before they fire they delete the new run's `_activeMixers`/`_activeNames` entries and release its bone claims (frozen pose). Procedural `headTilt`/`shoulderRoll` tweens are never halted by actions and write bones without re-checking ownership, so an overlapping action fights them and headTilt restores a stale `origZ`.
15. **Barge-in/audio race (Potential):** `speakFromBytes` awaits `decodeAudioData` before creating the source; a `stop_audio` arriving during decode is lost and the audio still plays. A second `audio` message overwrites `_currentSource` (both play).
16. **Expression Lab pinning:** browsing a combination persists its (default) recipe to localStorage and Export then writes it as if calibrated (§7).
17. **Query error strings enter history:** `query()`'s "can't reach my AI core"/timeout/error strings are returned (≠ `ALREADY_SPOKEN`) and `Processor` stores them as assistant turns; weather/skill replies are stored with `[tag]`s while LLM replies are tag-stripped; transcript broadcasts carry the raw tags.
18. **Reminders lose their content (timer skill):** "remind me to drink water in 15 minutes" becomes an anonymous timer; the alert says "Your timer 1 is done." "remind me to X" without a duration → "How long should I set the timer for". Numerals as words ("one hour timer", in TRAINING_DATA) are not parsed. The label regex's negative lookahead is prefix-based, so labels starting with a/an/me/the/for/set/min/sec/hour (e.g. "apple timer") are dropped.

### Potential
- **Listening/FSM edges:** if `queue_manager.put` drops a command (queue full) the FSM stays LISTENING and the frontend's last state is `listening` (no fidgets) until the next state broadcast; a manual browser `interrupt` with no follow-up utterance leaves FSM at LISTENING (voice barge-in is fine — the interrupting utterance is queued); noise-triggered `listening → idle` resets the frontend's 30-min idle timer.
- **Power confirmation:** one-shot — Maya's own spoken prompt picked up by the mic (no AEC) is a non-confirm utterance and drops the request (safe direction, but she may need to re-ask); STT mishearing "yes" drops it too. The 10 s OS countdown has no voice abort. Graceful `shutdown` without `/f` can be blocked by apps with unsaved work.
- **Sleep edge cases:** a command already queued/in flight when "go to sleep" is spoken can still overwrite SLEEPING (`llm_service._play_worker` sets IDLE at `_DONE`; `Processor` sets PROCESSING); Listener still logs "Sleeping — say …" at startup though Maya starts awake; the avatar never visually sleeps (§8).
- **Dismissal exact match:** phrases with extra words ("no thanks please") no longer hit the guard and fall to keyword/ML routing (`keyword_fallback` still has a `dismissal` rule).
- **Echo/self-transcription:** VAD+STT run while speaking with no AEC; her own speech (e.g., "I'm Maya") could queue as a command or trigger self-barge-in (setup-dependent).
- **Wake window** is 2 s non-overlapping (word straddling a boundary missed); sleep/wake triggers are substring matches ("asleep", "I didn't sleep well").
- **Kokoro:** concurrent synthesis on one pipeline (filler + first phrase; 2 pipelines; Speaker/timer overlap); `_get_kokoro` lazy init unlocked; stuck native call can't be killed (daemon thread orphaned); `HF_HUB_OFFLINE` is set *after* `from kokoro import KPipeline` in `llm_service.py` (and `core/speaker.py` imports kokoro first) so it may not take effect.
- **Latency:** embedding call sits on the LLM critical path when memories exist; sync `IntentEngine.classify`/SQLite scan on the event loop; per-phrase WAV + `audio_done` round trip gives inter-phrase gaps. Chat model still unloads if idle beyond `keep_alive` (default 60 m) or if Ollama evicts it (check `[TIMING] chat turn` COLD lines).
- **`_fragment_for_energy`** truncates to 4 comma-fragments (excited/happy/angry final chunks; Speaker path passes whole multi-sentence text).
- **Expression naming split** (`joy/fun/sorrow` vs `happy/relaxed`) — one path may no-op on the real model.
- **Recipe persistence (UNVERIFIED):** `_animateRecipeTo` stops writing raw morphs after 220 ms. If three-vrm's `expressionManager.update()` (run each frame in `vrm.update`) resets morphs bound to expressions, recipe faces on those morphs would fade after the transition. Check on the real VRM.
- **Production build:** assets are loaded by runtime string path from `frontend/assets/`; `vite build` (no `publicDir` config) won't copy them into `dist/` — only dev-server mode is evidenced.
- **`ws_server` `origins=None`** accepts any origin (any local web page can connect/send `interrupt`); `audio_done` from multiple clients would confuse the gate.
- Greeting prefix `startswith` ("yo"→"you…", "hi"→"history…") routes to keyword/LLM fallback instead of ML.
- **Skill substring matching:** `open_website` (`name in text`), `timer`, `system_info`, `_SLEEP_TRIGGERS` all use substring tests.

---

# 12. IMPORTANT DO-NOT-BREAK DETAILS

**Ordering/async**
- All commands go through `queue_manager` (serial). Startup/wake/sleep lines and timer alerts already bypass it — don't add more.
- Per phrase order in `_play_worker`: actions → **behavior (before audio)** → state `speaking` → audio → `wait_for_audio_done` → baseline behavior. `Speaker.speak` mirrors it.
- `Speaker.speak()` restores the state it entered with only for SLEEPING (else IDLE): the go-to-sleep line depends on `main.on_speech` setting SLEEPING *before* calling it, and the startup greeting depends on `main.py` setting IDLE *before* calling it.
- `audio_done` is sent only from `source.onended`; `stopCurrentAudio` nulls it deliberately. `wait_for_audio_done` clears a single shared Event — keep speech serialized.
- Audio-callback thread → loop only via `run_coroutine_threadsafe`; `state.set_sync` exists for that thread.
- All Kokoro calls through `_run_kokoro` (daemon thread + timeout + reset), never the default executor. Blocking work → `run_in_executor` (`power._run` follows this).
- `query()` reads/clears module-level `_last_response`; `_filler_done` starts **set**; `run_interruptible` swallows `CancelledError`.
- **`on_speech` must only take LISTENING when `not state.is_busy()`** and must not force IDLE after `queue_manager.put` — `Processor.handle` owns LISTENING → PROCESSING. `_end_listening()` only resets if the FSM is still LISTENING.
- **`Router.dispatch` must call `resolve_pending` before intent routing** — otherwise the confirming "yes" is classified as general_query and sent to the LLM. Pending requests are one-shot with a TTL; never let a stale one execute.

**Contracts**
- WS server→client: `audio{data}`, `state{value: listening|processing|speaking|idle}` (an initial `state` with the last known value is sent on connect), `behavior{primary,secondary,intensity,attitude,gaze,actions,recipe}`, `stop_audio`, `transcript{text,role}`, `animation{name}`; client→server: `audio_done`, `interrupt`. `expression` message is legacy/unused and the frontend doesn't handle it.
- `ws_server._last_state` must be written by `broadcast_state` even when no clients are connected.
- Skills return `"[tag] text"`; LLM path returns `ALREADY_SPOKEN`. `Speaker._strip_tags` only strips `\[\w+\]` — other brackets (e.g. `[2026-06-15 12:00]`) reach TTS, which is why `notepad.py` strips timestamps itself; LLM text relies on `_ANY_BRACKET_RE`.
- Closed vocabularies must be edited together: emotions (llm_service/speaker/behavior_engine + prompt), attitudes, actions (`_ACTION_VOCABULARY`, `perform_action._ACTION_WORDS`, `_ACTION_WORD_RE`, prompt, `websocket.js` switch, `_VRMA_ASSETS`).
- `_next_boundary` vocative handling depends on `config.user_name`.
- Every request that touches the chat model (chat + warmup) must send `keep_alive` via `chat_keep_alive()`; omitting it resets expiry to Ollama's 5-minute default.
- `timer.execute` must keep skipping its cancel/status word checks for the `set_reminder` intent (reminders are routed to it).

**Memory/mood**
- `processor` is the only place that adds the user turn; `observe_user_turn` must run before `build_context_package`; `ContextManager` registers `memory.set_evict_callback` at import.
- `observe_user_text` before routing/LLM; `observe_turn` once per turn; `baseline_expression()` (not "neutral") for every rest reset; `report_event` events are consumed by the next `observe_turn`.
- Editing `TRAINING_DATA` triggers a full retrain on next start (don't hand-edit `models/`).
- `OllamaEmbedder` keep-alive/warmup and pooled client are latency workarounds — keep.

**Expressions/frontend**
- `expressions.json` is written by the backend and by Lab Export (flat schema; viseme keys excluded). Import before Export; keep a backup (§11 #10).
- BASE-tier writers must check `animationController.canWrite` (and use `handoffProgress`); VRMA clips keep only `.quaternion`/`.weight` tracks and skip shoulder bones; `mixer.update(0)` after `action.stop()`; `updateVrmaAnimations` must run every frame from `main.js`.
- Recipe path bypasses `expressionController` (raw morphs); lip-sync/blink stay on `expressionController`.
- `wakeAvatar()` runs on every WS reconnect: the blink/eye loops must stay idempotent (`_blinkTimeout`, `_eyeLoopStarted`).
- Fidget calm period, cooldown ratchets and `_isAnyFidgetPlaying` stuck-timeout are deliberate.
- `config.tts.output` must stay `"avatar"` with the frontend; assets are LFS-tracked.

---

# 13. CURRENT IMPLEMENTATION STATUS

*(Static inspection only. "Implemented / internally consistent" means the code paths trace correctly end-to-end; it does NOT establish that mic/VAD/STT, Kokoro, Ollama or the avatar actually work at runtime.)*

### Implemented / internally consistent
Mic/VAD/STT pipeline; serial queue; intent ensemble + auto-retrain; Ollama streaming with phrase splitting (except the decimal split, §11 #5), tag parsing, filler gate, `keep_alive` + turn diagnostics; Kokoro synth path with timeout/reset (CPU by default); WS protocol (incl. state replay on connect and `listening` broadcast); behavior engine + recipe generation/persistence; mood logic; recent-window + semantic memory pipeline (structure); VRMA player/fidget scheduler; lip-sync; datetime, clipboard, media, weather, open_website (listed sites), google_search, lock_screen (Windows, immediate), power (shutdown/restart with confirmation, Windows), notepad, screenshot (word or intent), timers' duration parsing, reminders via the timer skill, `perform_action` skill animation, LLM `*action*` animations via the LLM playback path, voice sleep/wake state handling.

### Present in code, with known limitations
Barge-in (LLM playback path only; skill/greeting path stalls; not during PROCESSING/filler); sleep/wake (state holds and wake word is active; frontend never sees sleep; in-flight command can overwrite SLEEPING); timers/reminders (alert not interruptible, shares `audio_done` event, reminder text dropped); context state quality (§5); expression library (22/35 entries Lab-only by decision; naming split; wipe-on-corrupt); gaze (no screen observer); open_app (no name mapping); Expression Lab (dev only); listening/FSM edge cases (§11 Potential).

### Known broken
None confirmed beyond the open items in §11 (§11 Confirmed); `reminder.py` is unused dead code.

### Unverified
VRM morph/expression names and VRMA track contents (LFS pointers); whether recipe morphs persist past `expressionManager.update()`; Ollama model availability and actual keep-alive behavior (check COLD logs / `ollama ps`); STT accuracy/latency; GPU placement and VRAM contention (Kokoro is CPU by default); Kokoro thread-safety; audio echo behavior; production (`vite build`) packaging; actual `npm run dev` Electron launch; runtime behavior of all Batch 1–4 fixes (incl. `shutdown.exe` invocation) and all audit findings.

---

# RUNTIME VERIFICATION CHECKLIST

Things to actually test (not architectural requirements) before or while changing code:
- Does "go to sleep" hold SLEEPING, and does "hey maya" wake her?
- Say "shut down": does Maya ask for confirmation, and does only "yes" schedule `shutdown /s /t 10` (abort with `shutdown /a`)? Does an unrelated utterance, "no", or waiting >30 s cancel it without acting? Same for "restart".
- Does the avatar show the brief surprised face on `listening`, and return to baseline after noise/empty STT? Does `listening → processing → speaking → idle` appear in the WS/log sequence, including while she is mid-reply (no `listening` broadcast then)?
- Does a frontend connecting late get an initial `state` (fidgets start after the calm period)?
- Does "can you giggle/wink/nod" play the animation in sync with the reply? Does "set a timer for 5 minutes" fire at 300 s? Does "remind me to drink water in 15 minutes" speak an alert (and what does it say — §11 #18)? Re-setting a named timer: is the new one still cancellable? Does the avatar resume fidgeting after a timer alert (`idle` broadcast)?
- Does "capture my screen"/"take a screenshot" write a file? Does "lock my screen" lock Windows? Does "take a note buy milk and write the report" keep the whole content, and "note read the report" get saved rather than read back?
- Restart the backend while the avatar is open: after the WS reconnect, do blinks and eye motion stay at normal speed (no doubled loops)?
- Ask "what is pi" style questions: is "3.14" spoken as one number or split (§11 #5)?
- Which expression names/morphs the real VRM exposes (`vrm.expressionManager.expressions`, mesh `morphTargetDictionary`), including which `Fcl_*` keys exist, whether `joy/fun/sorrow` or `happy/relaxed` resolve, and whether a recipe face holds after its 220 ms transition.
- `ollama ps` and `[TIMING] chat turn` logs: chat/embedding model residency, COLD vs warm, and whether `keep_alive` is applied (`expires_at`); GPU/VRAM split across Ollama and embeddings (Kokoro on CPU).
- Does starting the backend with no frontend connected proceed without a 30 s greeting stall? Does the WS server stay up for long sessions (§11 #11)?
- Echo behavior with headphones vs speakers (self-transcription/self-barge-in; also whether it drops a pending power confirmation).
- Environment: Python ≥3.11; `git lfs pull` done for assets.

---

# NEXT CLAUDE CONTEXT

**What it is:** a local-first Windows voice assistant. Python asyncio backend does mic→VAD→Google STT→intent→skill/Ollama→Kokoro, and pushes audio + a composed "behavior" packet over WS to an Electron/Three.js VRM avatar that lip-syncs and animates.

**Understand first:** (1) the `Processor.handle` → `Router.dispatch` → `llm_service.query` path and its three-stage `synth_q`/`play_q` pipeline; (2) the `audio_done` handshake and single-queue serialization; (3) the state sequence `listening → processing → speaking → idle` and who owns each transition (`on_speech`, `Processor.handle`, `speak()`/`_play_worker`); (4) tag → `BehaviorEngine.compose` → `expression-composer.js` (recipe path vs six-knob fallback); (5) frontend arbitration (`animation-controller`, `expression-controller`).

**Present in code:** everything under §13 (the "Implemented / internally consistent" group has no additional known source-level defect from static inspection beyond §11, but none of it is runtime-verified). Edge-TTS/XTTS, transcript overlay, SQLite for the recent window, offline wake word, screen observation are **not** in code.

**Top issues to know (remaining):** 30 s `audio_done` stalls (decode failure / barge-in on skill path); decimal numbers split by `_next_boundary`; unreferenced `ws_server.serve()` task; timer/reminder alert (not interruptible, reminder text dropped); `expressions.json` wipe-on-corrupt; degenerate context topic/entity data; barge-in leaves full reply in history and is impossible during PROCESSING/filler; unreferenced/unlocked async patterns in §11 audit. Decisions already made: extra `expressions.json` entries stay Lab-only; `lock_screen` immediate; unused config fields kept but marked. Batch 1–4 fixes are applied but not runtime-tested.

**Fragile:** Kokoro concurrency/timeouts/rebuild; barge-in only on the LLM task during playback; fidget/animation ownership and stale timeouts; `expressions.json` dual writers; closed vocabularies duplicated across files; one-shot power confirmation vs echo.

**Do not break:** §12 (queue order, per-phrase WS order, `audio_done` semantics, `Speaker.speak()` SLEEPING/IDLE contract, `on_speech` LISTENING rules, `resolve_pending` before routing, vocab sync, mood hook placement, `_run_kokoro`, chat `keep_alive`, TRAINING_DATA auto-retrain, `.quaternion/.weight` filtering, idempotent wake, `timer.execute` reminder bypass).

**Runtime checks:** see RUNTIME VERIFICATION CHECKLIST above.

**Inspect first:** `main.py`, `core/processor.py`, `core/speaker.py`, `services/llm/llm_service.py`, `services/llm/ollama_lifecycle.py`, `core/state.py`, `services/ws_server.py`, `brain/router.py`, `skills/system/power.py`, `skills/utilities/timer.py`, `brain/intent_engine.py` (`_predict`), `brain/conversation.py`, `core/behavior_engine.py`, `frontend/js/{websocket,avatar,expression-composer}.js`, `config/settings.py`.