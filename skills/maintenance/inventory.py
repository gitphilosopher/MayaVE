"""
Initial task catalog for MayaVE Maintenance Coordinator.
Directly ingested from Claude's architectural audit and the VE11 maintenance design report.
"""

from __future__ import annotations

try:
    from .models import (
        PreValidationPattern,
        TaskDefinition,
        TaskPriority,
        TaskRisk,
        TaskStatus,
    )
except ImportError:
    from models import (
        PreValidationPattern,
        TaskDefinition,
        TaskPriority,
        TaskRisk,
        TaskStatus,
    )


def get_default_audit_inventory() -> list[TaskDefinition]:
    """Return the 14 verified architectural audit tasks in dependency order."""
    return [
        # --- Tier 1: Safe Foundations & Quick Wins ---
        TaskDefinition(
            task_id="T1-01",
            title="Resolve tracer.js clean checkout import defect",
            audit_source="F2 / ARCH-01",
            priority=TaskPriority.P0,
            risk=TaskRisk.LOW,
            description=(
                "Frontend modules (main.js, avatar.js, websocket.js) unconditionally import ./tracer.js. "
                "Since tracer.* is in .gitignore and untracked, clean checkouts fail Vite build and avatar load. "
                "Provide a safe fallback/stub so clean checkouts run reliably."
            ),
            evidence=(
                "frontend/js/main.js:13 imports './tracer.js'. git ls-files frontend/js/tracer.js returns empty. "
                ".gitignore line 19 excludes tracer.*."
            ),
            target_files=[
                "frontend/js/main.js",
                "frontend/js/avatar.js",
                "frontend/js/websocket.js",
                "frontend/js/tracer_stub.js",
            ],
            test_targets=[".maya/tests/frontend/test_clean_clone_imports.py"],
            dependencies=[],
            acceptance_criteria=[
                "Frontend builds and runs cleanly when tracer.js is absent.",
                "Tracer calls degrade gracefully to no-ops when tracer module is missing.",
                "docs/CHANGELOG.md updated with entry under Fixes.",
            ],
            verification_instructions="Run frontend import test with tracer.js temporarily omitted.",
            pre_validation_pattern=PreValidationPattern(
                file_path="frontend/js/main.js",
                pattern=r'from\s+["\']\./tracer\.js["\']',
                description="Unconditional import of ./tracer.js in frontend/js/main.js",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T1-02",
            title="Wire dormant WebGL render throttle into animation frame loop",
            audit_source="F4 / ARCH-11",
            priority=TaskPriority.P1,
            risk=TaskRisk.LOW,
            description=(
                "avatar.js exports isAvatarDormant() and documents SLEEP_FPS, but main.js never imports or "
                "checks dormancy. The WebGL loop runs at full framerate continuously while asleep."
            ),
            evidence=(
                "frontend/js/avatar.js:1026 exports isAvatarDormant(). frontend/js/main.js:120 animate() has no "
                "dormancy check, keeping GPU active during sleep."
            ),
            target_files=["frontend/js/main.js", "frontend/js/avatar.js"],
            test_targets=[".maya/tests/frontend/test_avatar_dormancy.py"],
            dependencies=[],
            acceptance_criteria=[
                "main.js imports and queries isAvatarDormant().",
                "Render loop throttles to sleep FPS (or pauses) when avatar is dormant.",
                "Awake rendering rate is preserved at 60 FPS.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Verify render loop throttling when avatar state is sleeping/dormant.",
            pre_validation_pattern=PreValidationPattern(
                file_path="frontend/js/avatar.js",
                pattern=r"export\s+function\s+isAvatarDormant",
                description="isAvatarDormant() export present in avatar.js",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T1-03",
            title="Safe cross-thread async error handling & loop resolution",
            audit_source="F5 / F7 / ARCH-03",
            priority=TaskPriority.P1,
            risk=TaskRisk.LOW,
            description=(
                "asyncio.run_coroutine_threadsafe results are discarded in Listener._process_frame and "
                "WakeWordDetector.feed_frame. _on_state_change calls deprecated get_event_loop() outside try, "
                "failing if called from worker threads. Add done-callbacks and safe loop resolution."
            ),
            evidence=(
                "core/listener.py:167 and core/wake_word.py:113 discard Future. "
                "main.py:102 calls asyncio.get_event_loop_policy().get_event_loop() outside try block."
            ),
            target_files=["core/listener.py", "core/wake_word.py", "main.py"],
            test_targets=[".maya/tests/core/test_cross_thread_async.py"],
            dependencies=[],
            acceptance_criteria=[
                "All cross-thread submitted coroutines attach logging done-callbacks.",
                "State change observer resolves running loop safely without raising from worker threads.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Run unit tests invoking state changes and speech frames from separate threads.",
            pre_validation_pattern=PreValidationPattern(
                file_path="main.py",
                pattern=r"loop\s*=\s*asyncio\.get_event_loop_policy\(\)\.get_event_loop\(\)",
                description="Unprotected get_event_loop() call in main.py _on_state_change",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T1-04",
            title="WebSocket server task fail-fast & lifecycle supervision",
            audit_source="F6 / ARCH-03",
            priority=TaskPriority.P1,
            risk=TaskRisk.LOW,
            description=(
                "ws_task and node_task sit outside asyncio.TaskGroup. _on_ws_done only logs an error. "
                "If port 8765 is in use, the backend continues running headlessly with no avatar link. "
                "Surface failure fatally or trigger graceful restart."
            ),
            evidence="main.py:215 _on_ws_done only logs error; main.py:241 ws_task created outside TaskGroup.",
            target_files=["main.py"],
            test_targets=[".maya/tests/services/test_ws_task_lifecycle.py"],
            dependencies=[],
            acceptance_criteria=[
                "WebSocket server failure (e.g. port taken) cancels or terminates the main application.",
                "Shutdown cleans up server tasks predictably.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Simulate port collision and verify main task exits cleanly.",
            pre_validation_pattern=PreValidationPattern(
                file_path="main.py",
                pattern=r"def\s+_on_ws_done\(task:\s*asyncio\.Task\)\s*->\s*None:",
                description="_on_ws_done callback definition in main.py",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T1-05",
            title="Disentangle runtime expression generation from tracked assets",
            audit_source="F9",
            priority=TaskPriority.P2,
            risk=TaskRisk.LOW,
            description=(
                "BehaviorEngine._resolve_recipe calls save_recipe on every cache miss, persisting generated "
                "default formulas into tracked frontend/assets/expressions.json. Ephemeral defaults should remain "
                "in memory or in an untracked cache."
            ),
            evidence="core/behavior_engine.py:153 calls save_recipe(emotion, attitude, intensity_word, base) on cache miss.",
            target_files=["core/behavior_engine.py", "core/expression_library.py"],
            test_targets=[".maya/tests/core/test_expression_recipe_caching.py"],
            dependencies=[],
            acceptance_criteria=[
                "Generated recipes are cached in memory without mutating expressions.json on disk.",
                "User calibrations in expressions.json remain intact.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Verify that resolving uncalibrated recipes leaves expressions.json unmodified.",
            pre_validation_pattern=PreValidationPattern(
                file_path="core/behavior_engine.py",
                pattern=r"save_recipe\(emotion,\s*attitude,\s*intensity_word,\s*base\)",
                description="Automatic save_recipe call in BehaviorEngine._resolve_recipe",
            ),
            status=TaskStatus.PENDING,
        ),

        # --- Tier 2: Voice & Skill Safety Boundaries ---
        TaskDefinition(
            task_id="T2-01",
            title="Silence/energy VAD gating for sleep-mode wake detector",
            audit_source="F3 / ARCH-06",
            priority=TaskPriority.P1,
            risk=TaskRisk.LOW,
            description=(
                "WakeWordDetector.feed_frame fills 2s audio buffers and unconditionally posts them to "
                "recognize_google even when the room is silent while sleeping. Introduce energy/VAD gating "
                "so ambient silence is never transmitted."
            ),
            evidence="core/wake_word.py:100 feed_frame concatenates window and runs recognize_google without energy threshold.",
            target_files=["core/wake_word.py"],
            test_targets=[".maya/tests/core/test_wake_word_vad_gate.py"],
            dependencies=["T1-03"],
            acceptance_criteria=[
                "Audio buffers with RMS energy below threshold are discarded before STT transmission.",
                "Wake words spoken at normal volumes continue to wake Maya reliably.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Run unit tests with silent vs vocal audio buffers against feed_frame.",
            pre_validation_pattern=PreValidationPattern(
                file_path="core/wake_word.py",
                pattern=r"audio\s*=\s*np\.concatenate\(list\(self\._window\)\)\s*\n\s*self\._window\.clear\(\)\s*\n\s*self\._pending\s*=\s*True",
                description="Unconditional window scheduling without RMS energy threshold in WakeWordDetector.feed_frame",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T2-02",
            title="Restrict open_target launch passthrough to allowlisted executables",
            audit_source="F10 / ARCH-05",
            priority=TaskPriority.P1,
            risk=TaskRisk.MEDIUM,
            description=(
                "skills/system/open_target.py lines 179-181 call os.startfile on arbitrary non-table cleaned strings. "
                "Enforce an explicit allowlist of applications and URLs; require spoken confirmation for unrecognized targets."
            ),
            evidence="skills/system/open_target.py:179 calls _launch_app(cleaned, (cleaned,)) -> os.startfile(candidate).",
            target_files=["skills/system/open_target.py"],
            test_targets=[".maya/tests/skills/test_open_target_safety.py"],
            dependencies=[],
            acceptance_criteria=[
                "Recognized apps (_APP_TABLE) and URLs launch directly.",
                "Arbitrary system file paths and unknown protocol handlers require confirmation.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Test open_target with known apps vs arbitrary local paths.",
            pre_validation_pattern=PreValidationPattern(
                file_path="skills/system/open_target.py",
                pattern=r"_launch_app\(cleaned,\s*\(cleaned,\)\)",
                description="Fallback open-ended _launch_app call in skills/system/open_target.py",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T2-03",
            title="TtsEngine result type definition and timeout decoupling",
            audit_source="F1 / ARCH-02 Step 1",
            priority=TaskPriority.P1,
            risk=TaskRisk.MEDIUM,
            description=(
                "Establish a typed AudioSynthesisResult (distinguishing SUCCESS, TIMEOUT, POLICY_REJECTED, NO_AUDIO) "
                "to replace the fragile 'None means timeout' sentinel that causes spurious pipeline resets."
            ),
            evidence="services/llm/llm_service.py:2233 _run_kokoro returns None on timeout and rejection.",
            target_files=["services/llm/llm_service.py", "core/speaker.py"],
            test_targets=[".maya/tests/services/llm/test_tts_result_types.py"],
            dependencies=[],
            acceptance_criteria=[
                "Synthesis outcomes are strongly typed.",
                "Policy rejections and no-audio outcomes do not trigger Kokoro pipeline rebuilds.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Run synthesis failure tests verifying pipeline is only rebuilt on real timeouts.",
            pre_validation_pattern=PreValidationPattern(
                file_path="services/llm/llm_service.py",
                pattern=r"async\s+def\s+_run_kokoro\(",
                description="_run_kokoro definition in services/llm/llm_service.py",
            ),
            status=TaskStatus.PENDING,
        ),

        # --- Tier 3: Subsystem Architecture Extraction ---
        TaskDefinition(
            task_id="T3-01",
            title="TurnController turn-scoped state & single WS projector",
            audit_source="ARCH-03",
            priority=TaskPriority.P1,
            risk=TaskRisk.HIGH,
            description=(
                "Unify turn lifecycle management into a dedicated TurnController with turn-scoped context dataclass, "
                "one cancel scope per turn, and a single state->WS broadcast projector, closing skill cancellation gaps."
            ),
            evidence="Global turn state in llm_service.py (_last_response, _spoken_phrases); six separate state broadcast sites.",
            target_files=["core/turn_lifecycle.py", "core/processor.py", "main.py"],
            test_targets=[".maya/tests/core/test_turn_controller.py"],
            dependencies=["T1-03", "T1-04"],
            acceptance_criteria=[
                "Turn state encapsulated in TurnContext rather than module globals.",
                "Mid-dispatch cancellation reliably interrupts in-flight skill work.",
                "Barge-in and sleep race suites pass without regression.",
                "docs/CHANGELOG.md updated under Improvements.",
            ],
            verification_instructions="Run full turn lifecycle and cancellation tests.",
            pre_validation_pattern=PreValidationPattern(
                file_path="core/processor.py",
                pattern=r"class\s+Processor",
                description="Processor class in core/processor.py",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T3-02",
            title="Extract speech subsystem out of llm_service god module",
            audit_source="ARCH-02 Step 2",
            priority=TaskPriority.P1,
            risk=TaskRisk.HIGH,
            description=(
                "Extract neural synthesis, prosody enhancement, segmentation, and playback sequencing from "
                "llm_service.py into a clean speech/ subsystem, leaving llm_service as a focused dialogue orchestrator."
            ),
            evidence="services/llm/llm_service.py exceeds 2500 lines mixing prompting, Kokoro, GPU scheduler, and playback.",
            target_files=["speech/", "services/llm/llm_service.py"],
            test_targets=[".maya/tests/speech/test_speech_subsystem.py"],
            dependencies=["T2-03", "T3-01"],
            acceptance_criteria=[
                "speech/ package created with backward-compatible re-export shims in llm_service.",
                "No circular imports between core.speaker and services.llm.",
                "Median conversational TTFA remains within established baseline.",
                "docs/CHANGELOG.md updated under Improvements.",
            ],
            verification_instructions="Run speech synthesis benchmark and conversational integration tests.",
            pre_validation_pattern=PreValidationPattern(
                file_path="services/llm/llm_service.py",
                pattern=r"def\s+_enhance_prosody\(",
                description="Prosody helper embedded inside services/llm/llm_service.py",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T3-03",
            title="Versioned WebSocket protocol handshake & renderer negotiation",
            audit_source="ARCH-04",
            priority=TaskPriority.P1,
            risk=TaskRisk.MEDIUM,
            description=(
                "Implement versioned client hello handshake {protocol_version, role: renderer|observer, audio_unlocked}. "
                "Ensure only active renderers receive audio frames, preventing browser tabs from hijacking audio authority."
            ),
            evidence="services/ws_server.py first-connect authority; unhandled exceptions in _on_message.",
            target_files=["services/ws_server.py", "contracts/ws_protocol.md"],
            test_targets=[".maya/tests/services/test_ws_versioned_handshake.py"],
            dependencies=["T3-01"],
            acceptance_criteria=[
                "Handshake negotiates renderer authority safely with backward compatibility window.",
                "Non-renderer observer clients receive avatar state deltas without consuming audio transactions.",
                "docs/CHANGELOG.md updated under Improvements.",
            ],
            verification_instructions="Execute dual-client connection tests (browser + Electron).",
            pre_validation_pattern=PreValidationPattern(
                file_path="services/ws_server.py",
                pattern=r"class\s+MayaWebSocketServer",
                description="MayaWebSocketServer definition in services/ws_server.py",
            ),
            status=TaskStatus.PENDING,
        ),

        # --- Tier 4: Storage, Bootstrap & Documentation ---
        TaskDefinition(
            task_id="T4-01",
            title="Vector store schema versioning & legacy memory re-embedding",
            audit_source="F8 / ARCH-08",
            priority=TaskPriority.P2,
            risk=TaskRisk.MEDIUM,
            description=(
                "Add PRAGMA user_version schema tracking to SQLiteVectorStore. Automatically re-embed legacy rows "
                "whose vector dimension does not match active model (e.g. 768 nomic -> 384 BGE) using stored content."
            ),
            evidence="brain/vector_store.py:209 skips mismatched dimensions with warning; no migration logic.",
            target_files=["brain/vector_store.py"],
            test_targets=[".maya/tests/brain/test_vector_store_migration.py"],
            dependencies=[],
            acceptance_criteria=[
                "user_version pragmas track vector database schema state.",
                "Mismatched dimension memories are re-embedded on startup from stored text content.",
                "docs/CHANGELOG.md updated under Fixes.",
            ],
            verification_instructions="Test database opening with 768-dim legacy rows; assert rows become 384-dim BGE.",
            pre_validation_pattern=PreValidationPattern(
                file_path="brain/vector_store.py",
                pattern=r"Skipping\s+vector\s+id=.*with\s+mismatched\s+dimension",
                description="Vector dimension skipping warning in brain/vector_store.py",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T4-02",
            title="Reproducible model prefetch script & dependency constraints",
            audit_source="F11 / ARCH-01",
            priority=TaskPriority.P2,
            risk=TaskRisk.MEDIUM,
            description=(
                "Create scripts/fetch_models.py to prefetch BGE and Kokoro weights before HF_HUB_OFFLINE=1 takes effect. "
                "Document CUDA torch wheel index and add requirements constraints."
            ),
            evidence="config/requirements.txt resolves torch to CPU on Windows; HF_HUB_OFFLINE=1 causes offline failure on fresh install.",
            target_files=["scripts/fetch_models.py", "config/requirements.txt"],
            test_targets=[".maya/tests/config/test_bootstrap_reproducibility.py"],
            dependencies=["T1-01"],
            acceptance_criteria=[
                "scripts/fetch_models.py downloads weights with checksum validation.",
                "Requirements file documents explicit CUDA installation commands.",
                "docs/CHANGELOG.md updated under Improvements.",
            ],
            verification_instructions="Run fetch_models.py in dry-run/mock mode.",
            pre_validation_pattern=PreValidationPattern(
                file_path="config/requirements.txt",
                pattern=r"torch>=2\.0\.0",
                description="Unpinned torch entry in config/requirements.txt",
            ),
            status=TaskStatus.PENDING,
        ),
        TaskDefinition(
            task_id="T4-03",
            title="Synchronize documentation with active codebase architecture",
            audit_source="F12",
            priority=TaskPriority.P3,
            risk=TaskRisk.LOW,
            description=(
                "Correct doc drift in README.md (duplicate WebSockets, remove nomic-embed-text requirement), "
                "docs/architecture.md (§7 timeout, §9 BGE embedding defaults, WebSocket message inventory), "
                "and docs/CHANGELOG.md."
            ),
            evidence="README.md:150 lists WebSockets twice; architecture §9 refers to nomic and command_vectors.sqlite3.",
            target_files=["README.md", "docs/architecture.md", "docs/CHANGELOG.md"],
            test_targets=[".maya/tests/docs/test_doc_links.py"],
            dependencies=[],
            acceptance_criteria=[
                "All documented embedding models, timeouts, and ports match active config defaults.",
                "Duplicate list items in README.md removed.",
                "docs/CHANGELOG.md updated under Documentation.",
            ],
            verification_instructions="Run markdown documentation link and keyword consistency check.",
            pre_validation_pattern=PreValidationPattern(
                file_path="README.md",
                pattern=r"ollama\s+pull\s+nomic-embed-text",
                description="Stale nomic-embed-text instruction in README.md",
            ),
            status=TaskStatus.PENDING,
        ),
    ]
