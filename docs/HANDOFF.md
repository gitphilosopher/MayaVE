# MayaVI — Technical Handoff

**Basis:** static inspection of the repository snapshot. **No runtime was available** — "Confirmed" below means deterministic from source (code trace), not observed at runtime. Binary assets (`.vrm`, `.vrma`, `.vroid`) were Git LFS pointer stubs (text with `oid`/`size`) in the inspected snapshot, not real binaries; their contents are **UNVERIFIED**. If your checkout has the real assets, they were still not inspected. **`README.md` and `docs/architecture.md` are current, relevant project documentation. When documentation conflicts with the actual implementation, the code is the source of truth.**

---

# 1. MAYAVI CURRENT STATE

- **What:** Windows-first desktop voice assistant "Maya" with a transparent always-on-top 3D VRM avatar. Persona: FRIDAY-like, addresses user as `config.user_name` = "senpai".
- **Two processes:** Python `asyncio` backend (`main.py`) ⇄ WebSocket `ws://localhost:8765` ⇄ Electron/Vite/Three.js frontend (`frontend/`). Frontend is a pure WS client.
- **Models/services:** Ollama chat (`config.llm.model`) + Ollama embeddings (`nomic-embed-text`); Kokoro TTS (local, 24 kHz); Silero VAD (`torch.hub`); Google STT via `SpeechRecognition` (**online**, used for utterances *and* wake word); intent classifier = PyTorch BiLSTM + TF CNN ensemble; SQLite semantic memory.
- **Capabilities present in code (several have known runtime/behavioral problems — sleep, skill-path barge-in, `perform_action` animation, timers; see §11):** wake word/VAD/STT; serial command queue; ML intent routing + regex guards; ~12 skills; streaming LLM with phrase-level TTS pipelining and emotion/attitude/intensity/action tags; persistent mood; behavior engine → Fcl_* morph recipes; VRMA animation system; idle fidgets; gaze/life-motion layers; barge-in; recent-window + semantic long-term memory; Expression Lab (dev tool).
- **Development focus (from code state):** expression realism/calibration, latency (`[TIMING]`/TTFA logging is everywhere), animation/fidget tuning, context/memory quality.

---

# 2. ACTUAL ARCHITECTURE

**Startup (`main.py::main`)**
1. `asyncio.create_task(ws_server.serve())`; construct `Speaker()`, `Transcriber()`, `Processor(speaker)` (→ `IntentEngine()` load/auto-train — *synchronous, blocks loop*; `ConversationManager`; `Router(speaker)`).
2. `state.register_stop_callback(_hard_stop_audio)`.
3. Warmups (gathered): `_warmup_ollama` (1-token chat), `_warmup_embeddings`, `llm_service.warmup` (Kokoro, executor), `speaker.warmup` (executor). Then `describe_ollama_models()` diagnostic.
4. Startup greeting: `ws_server.broadcast_animation("wave")` + `await speaker.speak(...)` — **before** the Listener exists.
5. `asyncio.TaskGroup`: `Listener.start()` + `queue_manager.run()` (**requires Python ≥3.11**).

**Runtime flow**
```
sounddevice callback thread → Listener._process_frame
   ├─ WakeWordDetector.feed_frame (every frame; active only while SLEEPING)
   └─ Silero VAD → utterance → run_coroutine_threadsafe(main.on_speech)
on_speech → Transcriber.transcribe (Google STT, executor)
   → barge-in check (was_speaking snapshot + contains_wake_word) / sleep-trigger check
   → queue_manager.put(text)                       [core/queue_manager.py, maxsize=10, drops when full]
QueueManager.run → Processor.handle (core/processor.py)
   → IntentEngine.classify (sync) → context_manager.observe_user_turn
   → Router.dispatch (brain/router.py)
        ├─ skill → returns "[tag] text" → Speaker.speak (Kokoro → WS)
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
| Router | `brain/router.py` `Router` | Static dict intent→coroutine; unknown/unrouted → `llm_query`. |
| LLM | `services/llm/llm_service.py` | Ollama stream → phrase splitter/tag parser → `synth_q` → Kokoro → `play_q` → WS. |
| Context | `brain/conversation.py` `ContextManager` (singleton `context_manager`) | Topic/phase/goal state, open loops, semantic memory. |
| Mood | `core/mood.py` `MoodManager` (`mood_manager`) | Persistent angry/sad + transient tease. |
| Behavior | `core/behavior_engine.py` `BehaviorEngine.compose` | Tag+mood → packet incl. Fcl_* recipe via `core/expression_library.py`. |
| WS | `services/ws_server.py` `MayaWebSocketServer` (`ws_server`) | Broadcast + `audio_done` gate + interrupt handler. |
| Frontend | `frontend/js/{main,avatar,websocket,expression-composer,expression-controller,animation-controller,gaze-controller,life-motion-controller}.js` | See §8. |

---

# 3. BACKEND

**Input handling (`core/listener.py`, `main.py::on_speech`)**
- Listener runs VAD **even while SPEAKING** (no echo cancellation/gating in code). Each utterance spawns an independent `on_speech` task; queue order = STT *completion* order.
- `on_speech`: `was_speaking = state.is_speaking()` snapshot before STT. If speaking and `contains_wake_word(text)` → `on_interrupt()` (→ `state.interrupt()`); otherwise queued normally. Then `_SLEEP_TRIGGERS` (`go to sleep, sleep, goodbye, bye, stop listening`, **substring** match) → SLEEPING + goodbye line. Else `queue_manager.put`.
- Effective wake triggers: `config.wake_word="wake up Maya"` → set `{"wake up maya","maya","hey/hello/hi/yo maya"}`; because `"maya"` is in the set (substring match), **any text containing "maya" wakes/interrupts**.

**Queue/processor:** `QueueManager` single serial worker; `priority` field unused. `Processor.handle`: PROCESSING + WS state/transcript → `ConversationManager.add_user` → classify → `context_manager.observe_user_turn` → dispatch → if response ≠ `ALREADY_SPOKEN`: `add_assistant` (tagged text), transcript, `speaker.speak` (greet intent → `on_audio_start` broadcasts `wave`). Always ends with WS `idle` + baseline behavior; exceptions → IDLE.

**Intent engine (`brain/intent_engine.py`)** — `_predict` order:
1. Dismissal guard (≤4 tokens; `t in _DISMISSAL_PHRASES or startswith(p)`) → `dismissal`.
2. `_PRESENCE_RE` → `smalltalk`. 3. `_ACTION_WORD_RE` (nod/giggl*/sigh/shrug/wink/wynk) → `perform_action`.
4. Keyword-first if ≤3 tokens or `startswith` greeting prefix (and not comparison) → else `general_query` "short_input_fallback".
5. Ensemble avg softmax (BiLSTM `_build_pytorch_model`, CNN `_build_tf_model`), `_CONF_THRESH=0.65` else `_KEYWORD_RULES` (substring).
- `TRAINING_DATA` sha256 → `models/training_hash.txt`; mismatch/missing → auto-retrain on startup. `python -m brain.train_intent` = wipe + retrain + test print. 45 intent labels. `result["model"]` values (`pytorch+tensorflow`, `keyword_fallback`, `short_input_fallback`, `negation_guard`, `presence_guard`, `action_guard`, `keyword_short_input`) are consumed by `_should_play_filler`.
- `_extract_target` **returns the full utterance when no trigger strips** (i.e., `target` is almost never empty) — matters for context (§5).

**Router coverage:** no route for `shutdown`, `restart`, `lock_screen` (in TRAINING_DATA) → falls to LLM with a warning log. Skill exceptions → `"[sad] Sorry senpai, I ran into a problem with that."`. `farewell` intent (e.g. "see you later") only replies; it does not sleep.

**Barge-in (`core/state.py`, `main.py`, `ws_server.py`, `avatar.js`)** — `interrupt()` only when SPEAKING: state→INTERRUPTED, stop callback (`sd.stop()` + `broadcast_stop_audio`), cancel `_current_task`, state→LISTENING. Only `llm_service.query()` registers a task via `run_interruptible`; **`Speaker.speak()` is never wrapped** (processor/main call it directly).

**Mood:** see §7. **Skills:** §9. **Error handling:** most paths log and degrade (memory/embedding `try/except` → recent-context-only; Kokoro timeout → phrase dropped; `ws_server._on_message` swallows all exceptions).

**Blocking-on-loop calls (violates project rule "blocking → executor"):** `IntentEngine.classify`; `SQLiteVectorStore.search/has_records`; `expression_library.save_recipe` (file write); `system_info` CPU branch (`psutil.cpu_percent(interval=1)`).

---

# 4. LLM + RESPONSE PIPELINE

- **Config:** `config.llm`: `provider="ollama"` (not read anywhere), `model="llama3.1"`, `base_url=http://localhost:11434`, `max_tokens=150`→`num_predict`, `temperature=0.7`. `LLMConfig.system_prompt` and `api_key` are **unused**; the prompt is the module constant `llm_service._SYSTEM_PROMPT`. (The comment beside `model` says "3B params… llama3.1 8B" but the value is `llama3.1`; README says `llama3.2`.)
- **Request:** `/api/chat` via `httpx.AsyncClient(timeout=60)`, `stream=True`, `aiter_lines`. **No `keep_alive` in chat or chat-warmup payloads** → Ollama server default applies (UNVERIFIED whether `OLLAMA_KEEP_ALIVE` is set). Embeddings use `keep_alive:"30m"`; `brain/embeddings.py` docstring assumes `OLLAMA_MAX_LOADED_MODELS=2` (server-side, not in repo).
- **Prompt assembly (`_stream_and_speak`):** `_SYSTEM_PROMPT + mood_manager.system_prompt_note() + context_package.as_system_note()` + `context_package.recent` (last `config.context.recent_turns=6` turns from global `memory`, already containing the current user turn — **never re-append the user turn**).
- **Prompt content:** persona ("under 3 sentences"), `[emotion]` tag per sentence, optional `[attitude:x][intensity:x]`, one optional leading `*action*` (closed: nod/giggle/sigh/shrug/wink), CAPS word-stress rules.
- **Streaming (`_ollama_streamer`):** tokens → `_next_boundary(buffer)` (split at `.!?,;—` followed by whitespace/buffer-end, or after 12 words; comma before the vocative `config.user_name` is skipped; waits when next word not yet streamed) → `_emit` → `_parse_expression` (`*action*` extracted first; then consecutive leading `[..]` tags; `_ANY_BRACKET_RE` strips other brackets) → `synth_q` item `(text, expression, actions, is_final, attitude, intensity)`. Expression/attitude/intensity carry forward across phrases of one sentence until `is_final`. `*...*` spans are removed entirely (so markdown emphasis words are dropped, not just unstyled).
- **Workers:** `_synth_worker` (serial; `_enhance_prosody` → `_run_kokoro(_synthesise_blocking)`), `_play_worker` (serial; per phrase: `broadcast_animation`(actions) → `broadcast_behavior(compose(...))` → state `speaking` → `broadcast_audio` → `wait_for_audio_done` → baseline behavior). At `_DONE`: `state.set(IDLE)` + WS idle.
- **Filler:** `_should_play_filler` (intent ∈ {general_query, unknown}; model source lacks "fallback/keyword/guard"; not `_NO_FILLER_RE`; ≥3 words) → `asyncio.gather(_play_filler(), _stream_and_speak())`; `_filler_done` event gates only the first real phrase.
- **History:** after reply, `_last_response` (module-level list) → `_conv.add_assistant(clean text)`, transcript broadcast, `context_manager.record_assistant_turn`. Empty (cancelled) → nothing recorded.
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
- **Classification/write policy:** only when `_REMEMBER_CUE_RE` matches the *user* text and intent ∉ `_NOISE_INTENTS`; `mem_type`="preference" (if `_PREFERENCE_RE`) else "fact", importance 0.8, content = raw user question. Type set `VALID_MEM_TYPES` also lists goal/decision/relationship/project but **nothing produces them**. No episodic memory.
- **Embeddings:** `OllamaEmbedder.embed` → `/api/embeddings` (`nomic-embed-text`), pooled client, 20 s timeout, 64-entry LRU, 404 → sticky `_unavailable`. `MAYA_EMBEDDING_DEVICE=cpu` → `options.num_gpu=0`.
- **Retrieval (`_retrieve_semantic`):** skipped if store None, short followup/confirm/dismissal turn (≤6 tokens), or store empty; else embed → brute-force cosine over all rows → top `max_semantic_memories=3` ≥ `similarity_threshold=0.75` → drop hits newer than `semantic_recency_guard_seconds=120`.
- **Dedup:** `find_similar(threshold=0.92)` requires same `mem_type` **and same `topic`** (when topic non-empty) → update in place, else insert.
- **Compaction:** `Memory.on_evict` → `_on_memory_evict`; every 10 evicted turns → one `conversation_summary` record: `"Earlier discussion touched on: "` + alphabetically-first 15 keywords (`asyncio.create_task`, unreferenced).
- **Degraded behavior:** any failure → empty semantic list / recent-only; store init failure → `_store=None`.
- **Known data-quality issue (Confirmed):** because `intent["target"]` ≈ full utterance, `_infer_topic` = first 40 chars of the utterance, `_extract_entities` appends the whole utterance as an "entity", and `_resolve_reference` returns `entities[-1]` = *the current utterance* — so `TOPIC`, `ENTITIES`, `LIKELY REFERRING TO` in the system note are largely degenerate, and topic-scoped dedup rarely matches.

---

# 6. TTS + AUDIO

- **Config (`config.tts`):** `voice="af_sky"`, `voice_blend="jf_alpha"`, `blend_ratio=0.92` (→ 8% af_sky / 92% jf_alpha; the "35%" comment is wrong), `lang_code="a"`, `speed=1`, `output="avatar"`. `config.tts.device` doesn't exist → `_resolve_tts_device()` = cuda if available else cpu; `KPipeline(device=...)` only if the installed kokoro accepts it.
- **Two independent `KPipeline`s:** `Speaker._pipeline` (skills/greeting/wake/sleep lines) and `llm_service._kokoro_pipeline` (LLM stream, filler, timer alert). Both warmed at startup; VRAM duplicated.
- **Synthesis:** all Kokoro calls via `llm_service._run_kokoro` (new **daemon thread** per call, 15 s timeout, `_reset_kokoro_pipeline()` on timeout → returns None; `Speaker._synthesise_guarded` rebuilds its own pipeline). Output float32 24 kHz → `_numpy_to_wav` (executor) → base64 in JSON.
- **Prosody (`_enhance_prosody(text, expression, is_final)`):** `_elongation_re_sub` → `_fix_caps` (+`_CAPS_WHITELIST`) → partial chunk: strip trailing punct only; final: `_expand_short_exclamation` → per-expression punctuation/fragmentation (`_fragment_for_energy` caps at 4 fragments). `EXPRESSION_SPEED` (0.84–1.13) applied **only** in `llm_service._synthesise_blocking`, not in `Speaker._synthesise`.
- **Playback:** one WAV per phrase; browser `speakFromBytes` (WebAudio `AudioBufferSourceNode`), `source.onended` → `_onAudioDone` → WS `{"type":"audio_done"}` → `ws_server._audio_done_event`. `wait_for_audio_done(timeout=30)` **clears the single shared Event on entry**, warns and continues on timeout. `output="local"` uses `sounddevice`; `"both"` double-plays with the avatar (must stay `"avatar"` when frontend runs).
- **Speaker path (`core/speaker.py::speak`):** first valid `[tag]` only (`_strip_tags`), whole text one synthesis, `mood_manager.observe_turn([expr])`, sets SPEAKING, broadcasts behavior/state/audio, waits `audio_done`, `finally`: state **IDLE** + WS `idle` + baseline behavior.
- **Barge-in audio:** `stopCurrentAudio()` sets `onended=null` (no `audio_done` is sent after a stop) and stops lip-sync.
- **Concurrency:** filler + first phrase synthesize concurrently on the same module pipeline (thread-safety UNVERIFIED); Ollama/Kokoro/embedding model may share GPU.
- **Known problems:** see §11 (30 s stalls; wave-vs-speech sync — wave fires in `on_audio_start` *after* synthesis).

---

# 7. MAYA BEHAVIOR / MOOD / EXPRESSIONS

**Vocabularies (closed):** emotion tags `happy|sad|angry|surprised|relaxed|neutral|excited`; attitude `sincere|playful|teasing|mock`; intensity `low|medium|high`; actions `nod|giggle|sigh|shrug|wink`; gaze `direct|soft|away`. Duplicated in `llm_service.py`, `speaker.py`, `behavior_engine.py`, `expression_library.py`, prompt text.

**Mood (`core/mood.py`):** tracks only `angry`/`sad` (`_STICKY_MOODS`) as `(mood, intensity)`; transient tease (`temp_mood`, 90 s). Events: USER (`observe_user_text`: provocation, ragebait = provocation+`_TEASING_MARKERS_RE` → transient, sadness, frustration, excitement eases, apology forgives), SKILL/SYSTEM (`report_event`, e.g. `system_info` battery/CPU/RAM), MAYA tags (`observe_turn`: only *confirm* an event raised this turn, +0.15). Decay: `_DISTRACTION_DECAY` per unrelated turn, `_TIME_DECAY_PER_MIN`, 20-min hard forget. `baseline_expression()` = resting face (used after every phrase/skill/idle). `system_prompt_note()` injected each turn. `is_teasing()` read by behavior engine. **`observe_user_text` is called only from `llm_service.query()`**, so skill/built-in-routed utterances never trigger mood events.

**Behavior engine (`BehaviorEngine.compose(expression, actions, source, attitude, intensity)`)** → packet `{primary, secondary, intensity, attitude, gaze, actions, recipe}`. Attitude: valid Ollama attitude else `"teasing"` if `is_teasing()` else `"sincere"`. Secondary: personality bias (`_SECONDARY_BIAS`), `happy` for mock/teasing angry, or active mood bleed (>0.3). Intensity: `0.45 + mood*0.25 + 0.65*0.25 + jitter`, blended 70/30 with Ollama word (0.3/0.6/0.9); bucketed low(<0.4)/medium(<0.7)/high. Recipe: `get_recipe(emotion, attitude, band)` else `compose_default` **and `save_recipe` (writes `frontend/assets/expressions.json`)**; bounded jitter (±0.04·chaos) applied per call, never persisted. `PERSONALITY` dict duplicated client-side in `expression-composer.js`.

**`expressions.json`:** flat `{"emotion|attitude|intensity": {Fcl_*: weight}}`, 33 entries. The backend only requests the 7 emotions × attitudes sincere/playful/teasing/mock × bands low/medium/high, so **22/33 recipes are currently unreachable at runtime** (they exist only via Expression Lab: attitudes like `default`/`gloating`, emotion `scared`, band `extreme`). Inspect the file when needed.

**Expression keys referenced by code**
- *Recipe morph keys referenced by the code (raw mesh morph targets; `expression-composer.js` is written to drop any key not found on the loaded VRM):* `Fcl_BRW_{Angry,Joy,Sorrow,Surprised,Fun}`, `Fcl_EYE_{Angry,Joy,Joy_L,Fun,Sorrow,Surprised,Spread,Natural,Close_L,Close_R}`, `Fcl_MTH_{Joy,Large,Angry,Sorrow,Surprised,Down,Neutral,Fun,Up}`. `Fcl_MTH_{A,I,U,E,O}` are excluded (lip-sync). Existence on the actual VRM is UNVERIFIED.
- *Legacy six-knob (`RENDERABLE`, expression-manager keys):* `neutral, joy, fun, angry, sorrow, surprised`.
- *Other expression-manager keys used in `avatar.js`:* `blink`, `blinkLeft` (wink), `aa ee ih oh ou` (lip-sync), `happy` (0.08 during speech), `relaxed` (BASE 0.4/0.6), `surprised` (0.3 on `listening`). Note the **naming inconsistency**: composer uses VRM0-style `joy/fun/sorrow`; avatar.js uses VRM1-style `happy/relaxed`. Which set the model exposes is UNVERIFIED (`expressionManager.setValue` on an unknown name silently no-ops).

**LLM → face:** Ollama `[happy][attitude:playful][intensity:high]` → `_parse_expression` → `_play_worker` → `behavior_engine.compose` (+ mood/recipe) → `ws_server.broadcast_behavior` `{"type":"behavior",...}` **before** the phrase audio → `websocket.js` → `expression-composer.js::applyBehavior`: if ≥1 recipe key exists on a mesh → `_animateRecipeTo` (writes `morphTargetInfluences` directly, 220 ms smoothstep) and legacy knobs eased to 0; else `_composeWeights` six-knob path via `expressionController` EMOTION layer. `gaze` → `applyBehavioralGaze` (eye offset for 0.9–1.4 s, lower priority than speaking/screen gaze). Actions go separately as `animation` messages.

**Expression Lab (`frontend/expression-lab.html` + `js/expression-lab.js`):** dev-server-only page (not a Vite build input). Edits persist to `localStorage["maya.expressionLab.recipes"]` (schema v1); Import merges an `expressions.json`; **Export writes only the localStorage set** (drops backend-generated/never-imported entries) — Import first. Its `composeDefault` mirrors `core/expression_library.py` (duplicated intentionally). Attitude list is wider than the backend's.

---

# 8. FRONTEND / AVATAR

- **Runtime:** Electron `^31.7.7` + Vite `^8.0.16` (lock 8.0.16, rolldown; needs Node `^20.19 || >=22.12`) + `vite-plugin-electron ^0.29.0` (`vite.config.js`, `base:"./"`, entry `electron/main.js`), `three ^0.184.0`, `@pixiv/three-vrm ^3.5.3`, `@pixiv/three-vrm-animation ^3.5.5` (lock nests its own `three-vrm-core` 3.5.5 beside top-level 3.5.3). Scripts: `dev`, `build`, `preview` only. `frontend/package_electron.json` is an unreferenced older manifest.
- **Window (`electron/main.js`):** 820×440, x=-260 (bottom-left), transparent/frameless/always-on-top `"screen-saver"` with a 500 ms `keepOnTop` interval, `skipTaskbar`, `setIgnoreMouseEvents(true,{forward:true})` (click-through; injected drag CSS is inert). Loads `VITE_DEV_SERVER_URL` or `../dist/index.html`.
- **Scene (`main.js`):** FOV 18 camera (-0.05,1.35,2.5), transparent renderer, 3 lights; animate loop: render → `vrm.update` → `updateVrmaAnimations(delta)` → `updateGaze(delta)`.
- **VRM load (`avatar.js::loadAvatar`):** `assets/mayaaa.vrm`; `expressionController.attach`; eyes closed, arms posed, `startHeadMovement()` always running (also calls `lifeMotionController.update`). `window.vrm` and `window.maya.*` console helpers exposed.
- **Sleep/wake:** `wakeAvatar()` on WS open (blink loop, eye loop, idle fidgets), `sleepAvatar()` on WS close. **Tied to socket connection only — backend SLEEPING state is never sent to the frontend.**
- **WS (`websocket.js`):** reconnect every 2 s; handles `audio, state, behavior, stop_audio, transcript(no-op), animation(wave|nod|giggle|sigh|shrug|wink)`; sends `audio_done` only. No `interrupt` sender exists in the frontend. URL hard-coded `ws://localhost:8765`.
- **Layering:** `expression-controller.js` (BASE<EMOTION<ACTION<LIPSYNC<BLINK per key; unresolved key → 0); `animation-controller.js` (bone owner tiers BASE<FIDGET<ACTION, `canWrite/claim/release`, 350 ms `handoffProgress`); `life-motion-controller.js` (BASE-tier breathing/posture/shoulders/hips, scaled down while speaking/observing); `gaze-controller.js` (attention session/boredom; `observeScreenActivity` has **no caller except `window.maya`** — no screen observer exists).
- **Lip-sync:** AnalyserNode (fft 256, 5 bands) → `aa/ee/ih/oh/ou` + `happy` on LIPSYNC layer; token-guarded loop.
- **VRMA (`playVrmaAnimation(name)`):** `_VRMA_ASSETS` maps animation names to `.vrma` files (filenames often differ from names — read the map); URL-cached loads; `createVRMAnimationClip` → keep only `.quaternion` (minus shoulders via `_PROTECTED_BONE_KEYS`) and `.weight` tracks; fade in/out; claims bones via `animation-controller` (FIDGET tier for `_FIDGET_VRMA_NAMES`, else ACTION); cleanup `setTimeout` does `action.stop(); mixer.update(0)`; a non-fidget play halts active fidgets first. `wink`, `headTilt`, `shoulderRoll` are procedural (not VRMA).
- **Wired automatically:** server `animation` messages (wave/nod/giggle/sigh/shrug/wink) and the fidget pool. Other registered clips are console-only via `window.maya`.
- **Fidgets:** randomized scheduler (7–18 s) gated by `_canFidgetNow` (awake, not speaking, backend state `"idle"`, nothing else playing, past a 5-min post-launch calm period). Per-fidget cooldowns/min-idle (`_FIDGET_COOLDOWN_CONFIG`, `_FIDGET_MIN_IDLE_MS`), a forced `waving` after 30 min idle, screen-attention suppression (`_gazeFidgetFactor`) and a stuck-state clear are all in `avatar.js`; tuning is still in flux.
- **UI that exists:** only `#ws-status` (💤 when disconnected) in `index.html`, plus the dev-only Expression Lab panel. No transcript overlay/cards.

---

# 9. SKILLS / TOOLS

Convention: `async execute(intent, text) -> str` returning `"[tag] text"`; `perform_action` also mutates `intent`.

| Skill (file) | Trigger | Behavior / output | Notes & failures |
|---|---|---|---|
| Open website (`skills/web/open_website.py`) | `open_website` | `webbrowser.open` from 8-site `_SITES`; else `https://{target}` | Unknown sites (amazon/linkedin/stackoverflow are in TRAINING_DATA) → invalid `https://amazon`. |
| Google search (`web/google_search.py`) | `search_web` | opens Google query from `intent["target"]` | Prompts if empty. |
| Weather (`web/weather.py`) | `get_weather` | Open-Meteo + geocoding, ip-api.com auto-location (executor, 8 s) | Emits 3 tags; Speaker keeps only the first. Location regex runs to end of string ("in London today" → "London today"). |
| Open app (`system/open_app.py`) | `open_app` | `os.startfile(target)` (Win) | No name mapping; untagged reply; failure → "couldn't open". |
| System info (`system/system_info.py`) | `system_info`, `screenshot` | psutil battery/cpu/ram/disk; battery/CPU/RAM extremes call `mood_manager.report_event(source="skill", "angry")` | **Screenshot broken** (§11). CPU branch blocks loop 1 s. |
| Clipboard (`system/clipboard.py`) | `clipboard_*` | pyperclip read(200-char)/write/clear | Top-level `import pyperclip` — missing package crashes startup (Router import). |
| Media (`media/play_music.py`) | play/pause/next/prev/volume/mute | `keyboard.send` media keys (volume ×5) | Guarded import; message if missing. |
| Date/time (`utilities/datetime_skill.py`) | `get_time`, `get_date` | formatted local time/date | — |
| Timer (`utilities/timer.py`) | `set_timer`, `cancel_timer`, `timer_status` | named/numbered asyncio timers; `_alert` speaks via llm_service Kokoro | Bugs in §11. Skips Speaker/queue/state. |
| Reminder (`utilities/reminder.py`) | `set_reminder` | `asyncio.sleep` then **`print` only** | No speech/WS on expiry. |
| Notepad (`utilities/notepad.py`) | `note_*` | `.txt` in `~/Maya/Notes`; `_TIMESTAMP_RE` stripped on read | Bugs in §11. `_delete` removes latest note without confirmation. |
| Perform action (`system/perform_action.py`) | guard-routed | picks nod/giggle/sigh/shrug/wink (+`wynk`), sets `intent["action"]`, returns confirmation | **Animation never fired** (§11). |
| Built-ins (`brain/router.py`) | greet/farewell/thanks/help | canned strings; greet triggers wave | — |

---

# 10. CONFIGURATION + RUNTIME REQUIREMENTS

- **Config:** `config/settings.py` singleton `config` (`MayaConfig`): `name="Maya"`, `user_name="senpai"`, `wake_word="wake up Maya"`, `log_level="INFO"`, `log_dir="logs"`, `ws_host="localhost"`, `ws_port=8765`. `audio`: 16000 Hz, mono, `chunk_ms=30` (clamped up to 512 samples), `silence_ms=800`, `pre_roll_ms=200`, `device_index=None`. `stt`: only `language="en"` is used (`model_size/device/compute_type` unused). `tts`/`llm` as above. `context`: `recent_turns=6, max_open_loops=3, max_semantic_memories=3, similarity_threshold=0.75, dedup_threshold=0.92, semantic_recency_guard_seconds=120, embedding_model="nomic-embed-text", memory_dir=None`.
- **`getattr`-only settings (not defined in dataclasses):** `config.tts.device`, `config.context.embedding_device`, `config.notes_dir`.
- **Env vars:** `MAYA_EMBEDDING_DEVICE` (`cpu` → embeddings `num_gpu=0`), `HF_HUB_OFFLINE` (`setdefault "1"` in `llm_service.py`), `TF_CPP_MIN_LOG_LEVEL` (`setdefault "3"`), `VITE_DEV_SERVER_URL` (Electron), `OneDrive` (screenshot path). Secrets: none in repo (`LLMConfig.api_key=""` unused) → nothing to `[REDACTED]`.
- **Ports/services:** WS 8765; Ollama 11434 with `llama3.1` and `nomic-embed-text` pulled; Vite dev server default 5173 (not set in config); external: Google STT, Open-Meteo (+geocoding), `ip-api.com` (HTTP), `torch.hub` `snakers4/silero-vad`, HF cache for Kokoro voices.
- **Runtime:** **Python ≥3.11** (`asyncio.TaskGroup`; README says 3.10+, wrong); Node per Vite 8; Windows 11 (`os.startfile`, `keyboard`, `OneDrive`). GPU optional but used by Ollama and both Kokoro pipelines when present.
- **Dependencies:** `config/requirements.txt` (README references root `requirements.txt`) **omits `websockets` and `pyperclip`** (both hard imports); lists `wikipedia` (unused).
- **Paths:** `models/` (auto-created, gitignored), `logs/maya.log` (**directory not auto-created**), `~/Maya/Notes`, `~/Maya/Memory/semantic_memory.sqlite3`, `frontend/assets/{mayaaa.vrm,expressions.json,vrmas/*.vrma}` (LFS via `.gitattributes`; `mmodel.vroid` unused by code).
- **Invariant:** `config.tts.output` must be `"avatar"` while the frontend runs.

---

# 11. CURRENT PROBLEMS / FRAGILE AREAS

### Confirmed (deterministic from source; not runtime-tested)
1. **Voice sleep doesn't stick.** `main.on_speech` sets SLEEPING then `await speaker.speak(goodbye)`; `Speaker.speak`'s `finally` sets IDLE. Same effect at startup: initial SLEEPING → IDLE after the greeting (Listener log says "Sleeping"). WakeWordDetector is therefore effectively never active. Frontend never learns about sleep either.
2. **`perform_action` animation never plays.** `core/processor.py` only special-cases `greet`; nothing reads `intent["action"]` (skill docstring says processor does). LLM `*action*` animations work through the LLM playback path (`_play_worker` → `broadcast_animation`); the `perform_action` skill's `intent["action"]` is never consumed by `Processor`, so skill-triggered animations do not fire.
3. **Dismissal guard uses `startswith` over `{"no","stop","cancel",...}`** (`intent_engine._predict`): "note …", "nod …", "now what", "nothing", "stop the timer/song", "cancel the timer" (≤4 tokens) → `dismissal` → LLM, bypassing note/timer/action/presence guards and ML.
4. **Timer duration double-counted** (`timer._parse_duration`): patterns `minute` and `min` (and `second`/`sec`) both match "5 minutes" → 600 s. "1 hour" is fine.
5. **Timer misc:** re-setting a *named* timer: old task's `finally` pops the new entry (untracked, uncancellable); `_alert` broadcasts `speaking` but never `idle` (frontend `_currentBackendState` stuck → fidgets stop), is not interruptible, and shares the single `_audio_done_event` with any concurrent speech.
6. **Screenshot skill broken** (`system_info`): `pyautogui.screenshot(path)` is given a directory; `logging.log("…")` (missing level) raises when `OneDrive` is unset; exceptions surface as the generic router apology.
7. **30 s stalls:** `wait_for_audio_done` waits the full timeout when (a) no frontend client connected (`broadcast_audio` returns early but the wait doesn't) — including the **startup greeting, which runs before the Listener starts**; (b) browser decode fails or `vrm` not loaded (no `audio_done`); (c) barge-in during any `Speaker.speak` path (`stopCurrentAudio` nulls `onended`, task not registered, so it isn't cancelled). Queue worker is blocked meanwhile.
8. **No `listening` state is ever broadcast** (only processing/speaking/idle) → `handleState("listening")` branch is dead; server pushes no state on WS connect → frontend `_currentBackendState` stays `null` (no fidgets) until the first state broadcast.
9. **Frontend reconnect leaks:** each `wakeAvatar()` starts another `startEyeMovement()` rAF loop + `setInterval` (never stopped) and may duplicate the blink loop (`_blinkingActive` re-armed before old timeout fires).
10. **Config/router gaps:** no route for `shutdown/restart/lock_screen`; `set_reminder` vs `set_timer` share identical training utterances ("set a timer for 5 minutes", "remind me in …") → nondeterministic skill (reminder never speaks).
11. **Notepad:** `_handle` checks `("read","show",…)` words in text regardless of intent (notes containing "read/show" are read, not saved); `_extract_content` strips verbs sequentially, so content after a later verb ("…write the report") is cut.
12. **Code-hygiene traps:** `llm_service._elongation_re_sub` defined twice (first is self-recursive dead code; second wins); `_strip_markdown` does not exist (older docs claim it); `LLMConfig.system_prompt`, `provider`, `STTConfig` model fields unused; `logs/` must pre-exist.
13. **Expression data reachability:** 22 of 33 `expressions.json` entries unreachable (§7).

### Potential
- **Echo/self-transcription:** VAD+STT run while speaking with no AEC; her own speech (e.g., "I'm Maya") could queue as a command or trigger self-barge-in (setup-dependent).
- **FSM overwrite:** `on_speech` sets LISTENING/IDLE regardless of PROCESSING; after a barge-in the FSM rests at LISTENING until the next command.
- **Wake window** is 2 s non-overlapping (word straddling a boundary missed); sleep triggers/wake triggers are substring matches ("asleep").
- **Kokoro:** concurrent synthesis on one pipeline (filler + first phrase; 2 pipelines; Speaker/timer overlap); stuck native call can't be killed (daemon thread orphaned); `HF_HUB_OFFLINE` is set *after* `from kokoro import KPipeline` in `llm_service.py` (and `core/speaker.py` imports kokoro first) so it may not take effect.
- **Latency:** chat model may unload after Ollama's default idle keep-alive (no `keep_alive` sent); embedding call sits on the LLM critical path when memories exist; sync `IntentEngine.classify`/SQLite scan on the event loop; per-phrase WAV + `audio_done` round trip gives inter-phrase gaps.
- **`_fragment_for_energy`** truncates to 4 comma-fragments (excited/happy/angry final chunks; Speaker path passes whole multi-sentence text).
- **Expression naming split** (`joy/fun/sorrow` vs `happy/relaxed`) — one path may no-op on the real model.
- **Production build:** assets are loaded by runtime string path from `frontend/assets/`; `vite build` (no `publicDir` config) won't copy them into `dist/` — only dev-server mode is evidenced.
- **`ws_server` `origins=None`** accepts any origin (any local web page can connect/send `interrupt`); `audio_done` from multiple clients would confuse the gate.
- Greeting prefix `startswith` ("yo"→"you…", "hi"→"history…") routes to keyword/LLM fallback instead of ML.

---

# 12. IMPORTANT DO-NOT-BREAK DETAILS

**Ordering/async**
- All commands go through `queue_manager` (serial). Startup/wake/sleep lines and timer alerts already bypass it — don't add more.
- Per phrase order in `_play_worker`: actions → **behavior (before audio)** → state `speaking` → audio → `wait_for_audio_done` → baseline behavior. `Speaker.speak` mirrors it.
- `audio_done` is sent only from `source.onended`; `stopCurrentAudio` nulls it deliberately. `wait_for_audio_done` clears a single shared Event — keep speech serialized.
- Audio-callback thread → loop only via `run_coroutine_threadsafe`; `state.set_sync` exists for that thread.
- All Kokoro calls through `_run_kokoro` (daemon thread + timeout + reset), never the default executor. Blocking work → `run_in_executor`.
- `query()` reads/clears module-level `_last_response`; `_filler_done` starts **set**; `run_interruptible` swallows `CancelledError`.

**Contracts**
- WS server→client: `audio{data}`, `state{value}`, `behavior{primary,secondary,intensity,attitude,gaze,actions,recipe}`, `stop_audio`, `transcript{text,role}`, `animation{name}`; client→server: `audio_done`, `interrupt`. `expression` message is legacy/unused and the frontend doesn't handle it.
- Skills return `"[tag] text"`; LLM path returns `ALREADY_SPOKEN`. `Speaker._strip_tags` only strips `\[\w+\]` — other brackets (e.g. `[2026-06-15 12:00]`) reach TTS, which is why `notepad.py` strips timestamps itself; LLM text relies on `_ANY_BRACKET_RE`.
- Closed vocabularies must be edited together: emotions (llm_service/speaker/behavior_engine + prompt), attitudes, actions (`_ACTION_VOCABULARY`, `perform_action._ACTION_WORDS`, `_ACTION_WORD_RE`, prompt, `websocket.js` switch, `_VRMA_ASSETS`).
- `_next_boundary` vocative handling depends on `config.user_name`.

**Memory/mood**
- `processor` is the only place that adds the user turn; `observe_user_turn` must run before `build_context_package`; `ContextManager` registers `memory.set_evict_callback` at import.
- `observe_user_text` before routing/LLM; `observe_turn` once per turn; `baseline_expression()` (not "neutral") for every rest reset; `report_event` events are consumed by the next `observe_turn`.
- Editing `TRAINING_DATA` triggers a full retrain on next start (don't hand-edit `models/`).
- `OllamaEmbedder` keep-alive/warmup and pooled client are latency workarounds — keep.

**Expressions/frontend**
- `expressions.json` is written by the backend and by Lab Export (flat schema; viseme keys excluded). Import before Export.
- BASE-tier writers must check `animationController.canWrite` (and use `handoffProgress`); VRMA clips keep only `.quaternion`/`.weight` tracks and skip shoulder bones; `mixer.update(0)` after `action.stop()`; `updateVrmaAnimations` must run every frame from `main.js`.
- Recipe path bypasses `expressionController` (raw morphs); lip-sync/blink stay on `expressionController`.
- Fidget calm period, cooldown ratchets and `_isAnyFidgetPlaying` stuck-timeout are deliberate.
- `config.tts.output` must stay `"avatar"` with the frontend; assets are LFS-tracked.

---

# 13. CURRENT IMPLEMENTATION STATUS

*(Static inspection only. "Implemented / internally consistent" means the code paths trace correctly end-to-end; it does NOT establish that mic/VAD/STT, Kokoro, Ollama or the avatar actually work at runtime.)*

### Implemented / internally consistent
Mic/VAD/STT pipeline; serial queue; intent ensemble + auto-retrain; Ollama streaming with phrase splitting, tag parsing, filler gate; Kokoro synth path with timeout/reset; WS protocol; behavior engine + recipe generation/persistence; mood logic; recent-window + semantic memory pipeline (structure); VRMA player/fidget scheduler; lip-sync; datetime, clipboard, media, weather, open_website (listed sites), google_search, notepad (basic), LLM `*action*` animations via the LLM playback path (not the `perform_action` skill).

### Present in code, with known limitations
Barge-in (LLM path only; skill/greeting path stalls); sleep/wake (wake word exists, sleep doesn't hold); timers (double-count, name race, no idle broadcast); context state quality (§5); expression library (22/33 entries unreachable; naming split); gaze (no screen observer); `set_reminder` (silent); open_app (no name mapping); Expression Lab (dev only).

### Known broken
`perform_action` animation; screenshot; dismissal-guard misroutes; timer "N minutes" duration; shutdown/restart/lock_screen skills (absent); `logs/` bootstrap; missing `websockets`/`pyperclip` in requirements.

### Unverified
VRM morph/expression names and VRMA track contents (LFS pointers); Ollama model availability/keep-alive env; STT accuracy/latency; GPU placement and VRAM contention; Kokoro thread-safety; audio echo behavior; production (`vite build`) packaging; actual `npm run dev` Electron launch.

---

# RUNTIME VERIFICATION CHECKLIST

Things to actually test (not architectural requirements) before or while changing code:
- Does "go to sleep" leave Maya IDLE (§11.1), and does the wake word ever fire afterward?
- Which expression names/morphs the real VRM exposes (`vrm.expressionManager.expressions`, mesh `morphTargetDictionary`), including which `Fcl_*` keys exist and whether `joy/fun/sorrow` or `happy/relaxed` resolve.
- `ollama ps`: chat/embedding model residency and keep-alive; GPU/VRAM split across Ollama, embeddings and the two Kokoro pipelines.
- Echo behavior with headphones vs speakers (self-transcription/self-barge-in).
- Environment: `logs/` exists; `websockets` and `pyperclip` installed; Python ≥3.11; `git lfs pull` done for assets.

---

# NEXT CLAUDE CONTEXT

**What it is:** a local-first Windows voice assistant. Python asyncio backend does mic→VAD→Google STT→intent→skill/Ollama→Kokoro, and pushes audio + a composed "behavior" packet over WS to an Electron/Three.js VRM avatar that lip-syncs and animates.

**Understand first:** (1) the `Processor.handle` → `Router.dispatch` → `llm_service.query` path and its three-stage `synth_q`/`play_q` pipeline; (2) the `audio_done` handshake and single-queue serialization; (3) tag → `BehaviorEngine.compose` → `expression-composer.js` (recipe path vs six-knob fallback); (4) frontend arbitration (`animation-controller`, `expression-controller`).

**Present in code:** everything under §13 (the "Implemented / internally consistent" group has no additional known source-level defect from static inspection, but none of it is runtime-verified). Edge-TTS/XTTS, transcript overlay, SQLite for the recent window, offline wake word, screen observation are **not** in code.

**Top issues to know:** sleep doesn't stick; `perform_action` doesn't animate; dismissal `startswith` bug; timer minute double-count; 30 s `audio_done` stalls (no frontend / barge-in on skill path / greeting before Listener); degenerate context topic/entity data; unreachable calibrated expressions; requirements/Python-version/`logs/` bootstrap gaps.

**Fragile:** Kokoro concurrency/timeouts; barge-in only on the LLM task; frontend loop duplication on reconnect; fidget/animation ownership; `expressions.json` dual writers; closed vocabularies duplicated across files.

**Do not break:** §12 (queue order, per-phrase WS order, `audio_done` semantics, vocab sync, mood hook placement, `_run_kokoro`, TRAINING_DATA auto-retrain, `.quaternion/.weight` filtering).

**Runtime checks:** see FIRST RUNTIME CHECKS above.

**Inspect first:** `main.py`, `core/processor.py`, `core/speaker.py`, `services/llm/llm_service.py`, `core/state.py`, `services/ws_server.py`, `brain/intent_engine.py` (`_predict`), `brain/conversation.py`, `core/behavior_engine.py`, `frontend/js/{websocket,avatar,expression-composer}.js`, `config/settings.py`.
