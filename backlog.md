# Backlog: MAYAVE-MAINTENANCE-BACKLOG

## BUG-F10-LAUNCH-PASSTHROUGH: Restrict open-ended application launch passthrough
- **Priority:** P1
- **Risk:** MEDIUM
- **Target Files:**
  - `skills/system/open_target.py`
- **Test Targets:**
  - `.maya/tests/skills/test_open_target_boundary.py`
- **Dependencies:** None
- **Actual Behavior:**
  `open_target._launch_app(cleaned, (cleaned,))` unconditionally passes any unrecognized target string to `os.startfile(candidate)` on Windows. This permits arbitrary protocol handlers, file paths, UNC network paths, or local executable launches without verification or user confirmation.
- **Expected Behavior:**
  Desktop application launches should be strictly validated against the known applications table, known safe protocol handlers (e.g. `http://`, `https://` via `webbrowser`), or an explicit allowlist. Any unrecognized executable or path should require user confirmation or be rejected cleanly.
- **Reproduction:**
  Inspecting `skills/system/open_target.py` lines 125-136 and 178-180 reveals that after site and known-app table lookups fail, the raw cleaned string is dispatched to `os.startfile`.
- **Acceptance Criteria:**
  - `open_target` rejects or gates arbitrary non-whitelisted executable strings
  - Known website shortcuts and table-defined application shortcuts continue launching normally
  - Safe fallbacks return a polite voice apology when an unrecognized application cannot be safely launched

## BUG-F8-LEGACY-MEMORIES: Automatic re-embedding and recovery for mismatched vector store dimensions
- **Priority:** P2
- **Risk:** LOW
- **Target Files:**
  - `brain/vector_store.py`
- **Test Targets:**
  - `.maya/tests/brain/test_context_bge_migration.py`
- **Dependencies:** None
- **Actual Behavior:**
  When querying `SQLiteVectorStore.search`, vectors stored with differing dimensionality (such as 768-dim vectors from nomic prior to the 384-dim BGE migration) trigger a warning and are skipped. As a result, historical memory records remain in the database but become permanently unretrievable.
- **Expected Behavior:**
  When a dimensionality mismatch is encountered during search or vector store initialization, records containing raw text content should be flagged for background re-embedding using the active model, or cleanly upgraded so legacy memories remain accessible.
- **Reproduction:**
  `brain/vector_store.py` lines 207-211 log `Skipping vector id=... with mismatched dimension` and `continue`, dropping legacy records from candidate results.
- **Acceptance Criteria:**
  - `SQLiteVectorStore` handles mismatched vector dimensions gracefully without throwing runtime errors
  - Stored records with valid text content can be re-embedded into the active vector dimension
  - Active search returns valid similarity scores for both newly created and re-embedded memory records

## BUG-ARCH04-WS-AUTHORITY: Resolve first-connect-wins playback authority race in WebSocket bridge
- **Priority:** P2
- **Risk:** MEDIUM
- **Target Files:**
  - `services/ws_server.py`
- **Test Targets:**
  - `.maya/tests/architecture/test_ws_playback_contracts.py`
- **Dependencies:** None
- **Actual Behavior:**
  `ws_server._handler` assigns playback authority to whichever client connects first (`if self._playback_authority is None: self._playback_authority = ws`). If a Vite browser tab or external client connects prior to the primary Electron renderer, it claims authority with a suspended AudioContext, stalling audio playback until timeouts expire.
- **Expected Behavior:**
  Playback authority should be negotiated explicitly based on client role and active renderer readiness, ensuring the active Electron window with an unlocked AudioContext is recognized as the authoritative playback destination.
- **Reproduction:**
  `services/ws_server.py` lines 179-180 unconditionally set `self._playback_authority = ws` on the first inbound connection without checking renderer role or AudioContext unlock status.
- **Acceptance Criteria:**
  - WebSocket authority is bound to an active renderer client rather than an arbitrary first connection
  - Multiple connected clients do not hijack or stall audio playback acknowledgement
  - Disconnection of a non-authority client does not disrupt the primary playback pipeline

## BUG-ARCH09-CONFIG-DRIFT: Declare and validate dynamic config attributes in Settings schema
- **Priority:** P3
- **Risk:** LOW
- **Target Files:**
  - `config/settings.py`
- **Test Targets:**
  - `.maya/tests/config/test_bootstrap_reproducibility.py`
- **Dependencies:** None
- **Actual Behavior:**
  Subsystems access undeclared configuration properties using `getattr(config.xxx, "attr", default)`, such as `config.context.embedding_device`, `config.context.ollama_embedding_model`, and `config.llm.keep_alive`. Typos in environment overrides or user config fail silently back to internal defaults without schema validation.
- **Expected Behavior:**
  All recognized configuration options should be explicitly declared on the dataclass schemas in `config/settings.py` with type annotations and documented defaults, eliminating unvalidated `getattr` fallbacks.
- **Reproduction:**
  `brain/embeddings.py` lines 44 and 96 query `getattr(config.context, "embedding_device", "auto")`, and `services/llm/ollama_lifecycle.py` queries `getattr(config.llm, "keep_alive", ...)`, none of which are declared on the respective dataclasses in `config/settings.py`.
- **Acceptance Criteria:**
  - `ContextConfig`, `LLMConfig`, and root `Config` declare all supported options with type annotations
  - Default values match runtime expectations
  - Unknown configuration keys or invalid types trigger clear validation diagnostics

## DOC-F12-ARCHITECTURE-DRIFT: Harmonize architecture specification with codebase reality
- **Priority:** P3
- **Risk:** LOW
- **Target Files:**
  - `docs/architecture.md`
- **Test Targets:**
  - `.maya/tests/docs/test_doc_links.py`
- **Dependencies:** None
- **Actual Behavior:**
  `docs/architecture.md` contains outdated references from earlier iterations: Section 7 states a 15-second Kokoro watchdog timeout (code uses 3.0 seconds), and Section 9 references `nomic-embed-text` and `command_vectors.sqlite3` instead of the resident BGE model.
- **Expected Behavior:**
  Documentation should accurately reflect active timeouts (3.0s CUDA watchdog), default model identifiers (BGE small en v1.5), and active SQLite database filenames.
- **Reproduction:**
  Inspect `docs/architecture.md` Section 7 and compare against `KOKORO_CUDA_WATCHDOG_TIMEOUT_SEC = 3.0` in `services/llm/llm_service.py`.
- **Acceptance Criteria:**
  - Section 7 records the accurate 3.0s Kokoro watchdog timeout
  - Model references in Section 9 align with current default BGE embeddings
  - All internal documentation links remain valid
