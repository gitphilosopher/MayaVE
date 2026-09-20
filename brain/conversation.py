"""
brain/conversation.py
Thin wrapper around Memory that adds conversation-level logic
(ConversationManager — Stage 1, unchanged), Stage 2 context intelligence
(ContextManager), and Stage 3 conversational state.

ContextManager layers on top of the recent-window memory:
  - recent context        — reuses ConversationManager/Memory as-is
  - conversation state     — topic, goal, task, constraints, entities,
                              decisions, last intent, phase (Stage 3)
  - open loops             — unresolved topics, retrieved by relevance
  - long-term semantic memory — SQLiteVectorStore + OllamaEmbedder

Integration (unchanged from Stage 2):
  - processor.py calls context_manager.observe_user_turn() AFTER its own
    conversation.add_user() call — state tracking only, no history write.
  - llm_service.py calls context_manager.build_context_package() to get a
    ready-to-use ContextPackage and record_assistant_turn() after a reply
    is produced. It does not implement retrieval/persistence logic itself.
  - Memory's on_evict hook feeds turns about to fall out of the recent
    window into a compacted semantic-memory summary before they're lost.

Stage 3 — conversational state:
  - ConversationState gained current_task, entities, phase, last_intent,
    and topic_history (bounded stack, enables "return to previous topic").
  - _reconcile_topic() classifies each new topic against the active one
    (continuation / subtopic / digression / return / switch) instead of
    the old binary "topic changed?" check, so temporary digressions and
    genuine returns no longer look like a fresh topic loss.
  - Reference resolution (_resolve_reference) is computed fresh per turn
    in build_context_package() from current entities/decisions/topic —
    no extra state, no second LLM call, and it only resolves when the
    utterance is sparse enough to be confident (see _is_referential).
  - ContextPackage now exposes state fields directly (topic/phase/goal/
    task/constraints/decisions/entities) instead of one flattened string,
    so as_system_note() can emit the compact, section-per-line format
    with empty sections omitted.

Every retrieval/persistence path is wrapped so a missing or broken
embedding backend / vector store degrades to recent-context-only
behaviour — Maya must keep working without them.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

from brain.memory import memory, MemoryEntry
from brain.embeddings import OllamaEmbedder
from brain.vector_store import SQLiteVectorStore, MemoryRecord, VALID_MEM_TYPES
from config.settings import config

logger = logging.getLogger(__name__)


class ConversationManager:
    def add_user(self, text: str) -> None:
        memory.add("user", text)

    def add_assistant(self, text: str) -> None:
        memory.add("assistant", text)

    def get_context(self, last_n: int = 6) -> list[dict]:
        """Recent turns for LLM context window."""
        return memory.get_history(last_n)

    def reset(self) -> None:
        memory.clear()


# ══════════════════════════════════════════════════════════════════════════
# Stage 2/3 — Context Intelligence + Conversational State
# ══════════════════════════════════════════════════════════════════════════

# Bounded set — conversation phase is a compact label, not a state machine.
_PHASES = frozenset({
    "casual", "discussing", "planning", "deciding",
    "executing", "troubleshooting", "concluding",
})


@dataclass
class ConversationState:
    active_topic:  str | None = None
    topic_history: list[str] = field(default_factory=list)   # bounded stack — enables "return to X"
    goal:          str | None = None
    current_task:  str | None = None
    constraints:   list[str] = field(default_factory=list)
    decisions:     list[str] = field(default_factory=list)
    entities:      list[str] = field(default_factory=list)   # important nouns/names, most-recent last
    phase:         str = "casual"
    last_intent:   str | None = None
    last_updated:  float = field(default_factory=time.time)


@dataclass
class OpenLoop:
    description: str
    topic:       str
    created_at:  float = field(default_factory=time.time)
    resolved:    bool = False


@dataclass
class ContextPackage:
    """Everything llm_service needs for one turn — pre-assembled, ranked,
    and already trimmed to size. llm_service just consumes this."""
    recent:            list[dict]
    topic:             str | None
    phase:             str | None
    goal:              str | None
    task:              str | None
    constraints:       list[str]
    decisions:         list[str]
    entities:          list[str]
    open_loops:        list[str]
    semantic_memories: list[str]
    reference_hint:    str | None = None

    def as_system_note(self) -> str:
        """Compact, section-per-line block appended to the Ollama system
        prompt. Empty sections are omitted so a quiet conversation doesn't
        pad every prompt with nothing."""
        lines = []
        if self.topic:
            lines.append(f"CURRENT TOPIC: {self.topic}")
        if self.phase and self.phase != "casual":
            lines.append(f"PHASE: {self.phase}")
        if self.goal:
            lines.append(f"GOAL: {self.goal}")
        if self.task:
            lines.append(f"TASK: {self.task}")
        if self.constraints:
            lines.append("CONSTRAINTS: " + "; ".join(self.constraints[-3:]))
        if self.decisions:
            lines.append("DECISIONS: " + "; ".join(self.decisions[-3:]))
        if self.entities:
            lines.append("ENTITIES: " + ", ".join(self.entities[-4:]))
        if self.reference_hint:
            lines.append(f"LIKELY REFERRING TO: {self.reference_hint}")
        if self.open_loops:
            lines.append("OPEN LOOPS: " + "; ".join(self.open_loops))
        if self.semantic_memories:
            lines.append("RELEVANT MEMORY: " + "; ".join(self.semantic_memories))
        return ("\n\n" + "\n".join(lines)) if lines else ""


# Conversational-noise intents — never durable, never a genuine topic
# departure by themselves. Shared by memory-persist gating, last_intent
# tracking, and topic-reconciliation's digression check.
_NOISE_INTENTS = frozenset({
    "smalltalk", "greet", "farewell", "thanks", "joke",
    "followup", "confirm", "dismissal",
})

# Intents that represent Maya actually doing something concrete.
_TASK_INTENTS = frozenset({
    "open_app", "search_web", "open_website", "play_music", "pause_music",
    "next_track", "prev_track", "volume_up", "volume_down", "mute",
    "get_time", "get_date", "set_reminder", "get_weather",
    "clipboard_read", "clipboard_write", "clipboard_clear",
    "set_timer", "cancel_timer", "timer_status",
    "note_create", "note_append", "note_read", "note_list", "note_delete", "note_open",
    "perform_action", "system_info", "screenshot",
})
_CASUAL_PHASE_INTENTS = frozenset({"smalltalk", "greet", "joke", "motivate", "identity"})
_DISCUSSING_INTENTS   = frozenset({"general_query", "unknown", "followup", "opinion"})

_TROUBLESHOOT_RE = re.compile(
    r"\b(error|doesn'?t work|not working|broken|issue|problem|bug|"
    r"fails?|failing|crash(?:ed|ing)?|stuck|help me fix|can'?t get)\b",
    re.IGNORECASE,
)
_CONCLUDE_RE = re.compile(
    r"\b(that'?s all|sounds good|that'?s it|we'?re done|all set|"
    r"that (?:should|will) do|perfect,? thanks)\b",
    re.IGNORECASE,
)
_PLANNING_RE = re.compile(
    r"\b(let'?s plan|planning to|should we|what if we)\b", re.IGNORECASE,
)
_RESOLUTION_CUE_RE = re.compile(
    r"\b(that'?s (?:done|fixed|resolved|solved|sorted)|"
    r"(?:i|we) (?:fixed|solved|sorted|figured (?:it|that) out|got it working)|"
    r"(?:it'?s|that'?s) working now|problem solved|answered my question)\b",
    re.IGNORECASE,
)

# Explicit "remember this" style cues — the clearest, lowest-risk signal
# that something belongs in long-term memory rather than just the recent
# window. Deliberately narrow; see MEMORY POLICY in ContextManager.
_REMEMBER_CUE_RE = re.compile(
    r"\b(remember that|don'?t forget|my favorite|i prefer|i always|i never|"
    r"from now on|i'?m working on|my goal is|i decided|we decided|"
    r"i'?ve got a meeting|i'?m planning)\b", re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(r"\b(prefer|favorite|always|never)\b", re.IGNORECASE)

# Narrower cues that update ConversationState directly (current-turn info,
# not long-term memory). Kept separate from _REMEMBER_CUE_RE so a single
# "i'm planning a trip" doesn't get dumped verbatim into state.decisions —
# only the specific goal/decision/constraint/task content is captured,
# and only short.
_GOAL_CUE_RE = re.compile(
    r"\b(?:my goal is|i'?m trying to|i'?m working on|i'?m planning to)\s+(.+)",
    re.IGNORECASE,
)
_DECISION_CUE_RE = re.compile(
    r"\b(?:i'?ve decided|i decided|we decided|let'?s go with)\b\s*(.*)",
    re.IGNORECASE,
)
_CONSTRAINT_CUE_RE = re.compile(
    r"\b(?:it (?:must|needs to|has to) be|must be|needs to be|has to be|"
    r"should be|can'?t be|cannot be|no more than|at least|requires)\b\s*(.*)",
    re.IGNORECASE,
)
_TASK_CUE_RE = re.compile(
    r"\b(?:help me|can you help me|i need to|i want to|let'?s)\b\s*(.+)",
    re.IGNORECASE,
)
# Explicit "unfinished business" cues — these create an open loop directly,
# regardless of topic. A bare topic change never does (see observe_user_turn).
_DEFERRAL_CUE_RE = re.compile(
    r"\b(?:remind me to|let'?s come back to|circle back to|we still need to|"
    r"i'?ll deal with that later|pick this up later|follow up on)\b\s*(.*)",
    re.IGNORECASE,
)
_STATE_TEXT_MAX     = 80  # keep state entries short — this is a pointer, not a transcript
_TOPIC_HISTORY_MAX  = 5
_ENTITIES_MAX       = 10
_TOPIC_OVERLAP_MIN  = 0.34  # jaccard threshold for "same topic, different wording"
_OPEN_LOOPS_MAX     = 20    # unresolved loops kept (oldest dropped)

# Short referential turns ("tell me more", "are you sure", "never mind") have
# no content of their own to match memories against — retrieval skips the
# embedding round-trip for them (see ContextManager._skips_semantic).
_SEMANTIC_SKIP_INTENTS    = frozenset({"followup", "confirm", "dismissal"})
_SEMANTIC_SKIP_MAX_TOKENS = 6

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "and", "or", "but", "it", "that", "this", "i", "you", "me",
    "my", "your", "what", "how", "do", "does", "did", "can", "could",
    "please", "senpai", "maya", "with", "at", "as", "be", "so",
    "about", "again", "just", "really", "also",
}

# Reference-resolution — a turn is only treated as referential (see
# _is_referential) when it's dominated by these words, i.e. genuinely
# sparse of its own content. Deliberately conservative: better to miss a
# reference than to guess wrong.
_REFERENCE_TRIGGER = {"it", "that", "this", "one", "thing", "same", "other"}
_REFERENCE_FILLER  = {"use", "do", "go", "try", "want", "like", "get", "take", "about", "lets", "let's"}

_ENTITY_SKIP = {"I", "Maya", "Senpai"}


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _is_referential(text: str) -> bool:
    """True only when the utterance is dominated by a reference word
    ('it'/'that'/...) with essentially no other content — e.g. 'what
    about it', 'let's use that', 'same thing'. A sentence with real
    content alongside the pronoun ('can you fix it') is left alone
    rather than guessed at."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if not any(t in _REFERENCE_TRIGGER for t in tokens):
        return False
    content = [
        t for t in tokens
        if t not in _STOPWORDS and t not in _REFERENCE_TRIGGER
        and t not in _REFERENCE_FILLER and len(t) > 2
    ]
    return len(content) == 0


class ContextManager:
    """
    Owns conversation state, open loops, and long-term semantic memory.
    Wraps a ConversationManager (read-only, for the recent window) rather
    than duplicating its history-write responsibility.
    """

    # How many evicted turns to buffer before compacting them into one
    # conversation_summary memory (see _on_memory_evict).
    _EVICT_FLUSH_SIZE = 10

    def __init__(self):
        self._conv = ConversationManager()
        self._state = ConversationState()
        self._open_loops: list[OpenLoop] = []
        self._evicted_buffer: list[MemoryEntry] = []
        # Tracks whether the most recent user turn was an unanswered
        # question, so leaving its topic without a reply can become a
        # genuine open loop (see observe_user_turn).
        self._pending_question: str | None = None
        self._pending_question_topic: str | None = None
        # (stripped text, intent name) of the latest observed turn — lets
        # retrieval gate on intent without changing llm_service's signatures.
        self._turn: tuple[str, str] | None = None

        self._embedder = OllamaEmbedder()
        try:
            self._store = SQLiteVectorStore()
        except Exception as e:
            logger.error(f"Semantic memory store unavailable — running recent-context-only: {e}", exc_info=True)
            self._store = None

        # Feed turns about to drop out of the recent window into
        # compaction instead of losing them outright.
        memory.set_evict_callback(self._on_memory_evict)

    # ── Observation (called once per user turn; no history side-effects) ──

    def observe_user_turn(self, text: str, intent: dict | None = None) -> None:
        """Updates topic/state/open-loop tracking. Does NOT write to
        conversation memory — processor.py already owns that via
        ConversationManager.add_user(). Best-effort: a failure here must
        never block skill/LLM dispatch for the turn."""
        try:
            self._observe_user_turn(text, intent)
        except Exception as e:
            logger.debug(f"Context state update failed (non-fatal): {e}")

    def _observe_user_turn(self, text: str, intent: dict | None) -> None:
        stripped    = text.strip()
        intent_name = (intent or {}).get("intent", "")
        self._turn  = (stripped, intent_name)
        new_topic   = self._infer_topic(text, intent)
        prev_topic  = self._state.active_topic

        relation = self._reconcile_topic(new_topic, intent_name)

        # Only a genuine departure (not a continuation/subtopic/digression)
        # can turn a left-behind question into an open loop.
        if relation in ("switch", "return") and prev_topic:
            if self._pending_question and self._pending_question_topic == prev_topic:
                self._add_open_loop(f"Unanswered: {self._pending_question[:_STATE_TEXT_MAX]}", prev_topic)

        is_question = stripped.endswith("?")
        self._pending_question = stripped if is_question else None
        self._pending_question_topic = self._state.active_topic if is_question else None

        # Explicit "I'll come back to this" style cues create an open loop
        # directly, independent of topic tracking.
        deferral = _DEFERRAL_CUE_RE.search(stripped)
        if deferral:
            desc = (deferral.group(1) or stripped).strip()[:_STATE_TEXT_MAX] or stripped[:_STATE_TEXT_MAX]
            self._add_open_loop(desc, self._state.active_topic or "general")

        # Explicit completion/abandonment cues resolve a loop on the
        # current thread — an open loop is only closed when the item was
        # actually answered, completed, cancelled, or abandoned.
        if _RESOLUTION_CUE_RE.search(stripped):
            self._resolve_open_loop(self._state.active_topic)
        if intent_name == "dismissal":
            self._resolve_open_loop(prev_topic or self._state.active_topic)

        goal_match = _GOAL_CUE_RE.search(stripped)
        if goal_match:
            self._state.goal = goal_match.group(1).strip()[:_STATE_TEXT_MAX]

        decision_match = _DECISION_CUE_RE.search(stripped)
        if decision_match:
            decision_text = (decision_match.group(1) or stripped).strip()[:_STATE_TEXT_MAX]
            if decision_text and decision_text not in self._state.decisions:
                self._state.decisions.append(decision_text)
                self._state.decisions = self._state.decisions[-5:]

        constraint_match = _CONSTRAINT_CUE_RE.search(stripped)
        if constraint_match:
            constraint_text = (constraint_match.group(1) or stripped).strip()[:_STATE_TEXT_MAX]
            if constraint_text and constraint_text not in self._state.constraints:
                self._state.constraints.append(constraint_text)
                self._state.constraints = self._state.constraints[-5:]

        if intent_name not in _NOISE_INTENTS:
            task_match = _TASK_CUE_RE.search(stripped)
            if task_match:
                self._state.current_task = (task_match.group(1) or stripped).strip()[:_STATE_TEXT_MAX]

        self._extract_entities(stripped, intent)

        self._state.phase = self._infer_phase(
            stripped, intent_name, has_goal=bool(goal_match), has_decision=bool(decision_match),
        )
        if self._state.phase == "concluding":
            # The current thread reads as wrapped up — resolve it and
            # clear the immediate task. Goal/decisions/constraints are
            # more durable and are left alone.
            self._resolve_open_loop(self._state.active_topic)
            self._state.current_task = None

        if intent_name and intent_name not in _NOISE_INTENTS:
            self._state.last_intent = intent_name

        self._state.last_updated = time.time()

    def _reconcile_topic(self, new_topic: str | None, intent_name: str) -> str:
        """Classifies the new topic against the active one. Returns one
        of: 'none', 'continuation', 'subtopic', 'digression', 'return',
        'switch'. Only 'switch'/'return' push the old topic's unanswered
        question into an open loop or move it onto topic_history —
        continuations, subtopics, and digressions all preserve state."""
        old = self._state.active_topic
        if not new_topic:
            return "none"
        if old is None:
            self._state.active_topic = new_topic
            return "switch"
        if new_topic == old:
            return "continuation"

        # Conversational noise ("thanks", "no thanks", "tell me more"...)
        # never counts as a genuine topic departure by itself.
        if intent_name in _NOISE_INTENTS:
            return "digression"

        old_kw, new_kw = _keywords(old), _keywords(new_topic)
        union = len(old_kw | new_kw) or 1
        if len(old_kw & new_kw) / union >= _TOPIC_OVERLAP_MIN:
            self._state.active_topic = new_topic
            return "continuation"

        for i, past in enumerate(self._state.topic_history):
            past_kw = _keywords(past)
            p_union = len(past_kw | new_kw) or 1
            if len(past_kw & new_kw) / p_union >= _TOPIC_OVERLAP_MIN:
                self._state.topic_history.pop(i)
                self._push_topic_history(old)
                self._state.active_topic = past
                return "return"

        if old_kw & new_kw:
            # Some shared ground but not enough to call it the same topic —
            # a related subtopic. Doesn't push history since the parent
            # topic hasn't really been left.
            self._state.active_topic = new_topic
            return "subtopic"

        self._push_topic_history(old)
        self._state.active_topic = new_topic
        return "switch"

    def _push_topic_history(self, topic: str | None) -> None:
        if not topic:
            return
        if topic in self._state.topic_history:
            self._state.topic_history.remove(topic)
        self._state.topic_history.append(topic)
        self._state.topic_history = self._state.topic_history[-_TOPIC_HISTORY_MAX:]

    def _infer_phase(self, stripped: str, intent_name: str, has_goal: bool, has_decision: bool) -> str:
        if intent_name in ("farewell", "thanks") or _CONCLUDE_RE.search(stripped):
            return "concluding"
        if _TROUBLESHOOT_RE.search(stripped):
            return "troubleshooting"
        if has_decision:
            return "deciding"
        if has_goal or _PLANNING_RE.search(stripped):
            return "planning"
        if intent_name in _TASK_INTENTS:
            return "executing"
        if intent_name in _CASUAL_PHASE_INTENTS:
            return "casual"
        if intent_name in _DISCUSSING_INTENTS:
            return "discussing"
        # Ambiguous/neutral turn (e.g. confirm, dismissal) — keep whatever
        # phase was already active rather than guessing.
        return self._state.phase if self._state.phase in _PHASES else "casual"

    def _extract_entities(self, text: str, intent: dict | None) -> None:
        """Deterministic entity capture: capitalised words (proper nouns),
        skipping the sentence-initial word, plus the intent's target if
        any. Adjacent capitalised words are merged into one entity (e.g.
        'Eiffel Tower') rather than split. Bounded, deduplicated,
        most-recent last."""
        found = []
        words = text.split()
        run: list[str] = []

        def _flush_run():
            if run:
                found.append(" ".join(run))
                run.clear()

        for i, w in enumerate(words):
            core = w.strip(".,!?;:\"'()")
            if i == 0 or not core:
                _flush_run()
                continue
            if core.isalpha() and core.istitle() and len(core) > 2 and core not in _ENTITY_SKIP:
                run.append(core)
            else:
                _flush_run()
        _flush_run()
        if intent:
            target = (intent.get("target") or "").strip()
            if target and len(target) <= _STATE_TEXT_MAX:
                found.append(target)
        for e in found:
            if e in self._state.entities:
                self._state.entities.remove(e)
            self._state.entities.append(e)
        self._state.entities = self._state.entities[-_ENTITIES_MAX:]

    @staticmethod
    def _infer_topic(text: str, intent: dict | None) -> str | None:
        if intent:
            target = (intent.get("target") or "").strip()
            if target:
                return target[:40]
        kws = _keywords(text)
        return " ".join(sorted(kws)[:3]) if kws else None

    # ── Reference resolution (computed per-turn, no extra stored state) ────

    def _resolve_reference(self, question: str) -> str | None:
        """Resolves 'it'/'that'/'same thing'/etc. to the most recent
        entity, decision, or the active topic — only when the utterance
        is sparse enough to be confident (see _is_referential) and a
        referent actually exists. Returns None rather than guessing."""
        if not _is_referential(question):
            return None
        if self._state.entities:
            return self._state.entities[-1]
        if self._state.decisions:
            return self._state.decisions[-1]
        if self._state.active_topic:
            return self._state.active_topic
        return None

    # ── Context assembly (called once per turn, before Ollama) ────────────

    async def build_context_package(self, question: str) -> ContextPackage:
        """Assembles the full package. Each section is independently
        fault-isolated so a failure in state/loop/semantic tracking still
        leaves the LLM with at least the recent-context window."""
        t0 = time.perf_counter()
        try:
            recent = self._conv.get_context(last_n=config.context.recent_turns)
        except Exception as e:
            recent = []
        logger.info(f"[TIMING]   recent_context: {time.perf_counter()-t0:.4f}s")

        t1 = time.perf_counter()
        try:
            s = self._state
            topic, phase, goal, task = s.active_topic, s.phase, s.goal, s.current_task
            constraints, decisions, entities = list(s.constraints), list(s.decisions), list(s.entities)
        except Exception:
            topic = phase = goal = task = None
            constraints = decisions = entities = []
        logger.info(f"[TIMING]   state_snapshot: {time.perf_counter()-t1:.4f}s")

        t2 = time.perf_counter()
        try:
            relevant_loops = self._relevant_open_loops(question)
        except Exception:
            relevant_loops = []
        logger.info(f"[TIMING]   open_loops: {time.perf_counter()-t2:.4f}s")

        t3 = time.perf_counter()
        try:
            semantic = await self._retrieve_semantic(question)
        except Exception:
            semantic = []
        logger.info(f"[TIMING]   retrieve_semantic total: {time.perf_counter()-t3:.3f}s")

        t4 = time.perf_counter()
        try:
            reference_hint = self._resolve_reference(question)
        except Exception:
            reference_hint = None
        logger.info(f"[TIMING]   resolve_reference: {time.perf_counter()-t4:.4f}s")

        return ContextPackage(
            recent=recent, topic=topic, phase=phase, goal=goal, task=task,
            constraints=constraints, decisions=decisions, entities=entities,
            open_loops=relevant_loops, semantic_memories=semantic,
            reference_hint=reference_hint,
        )

    def _relevant_open_loops(self, question: str) -> list[str]:
        if not self._open_loops:
            return []
        q_kw = _keywords(question)
        scored = []
        for loop in self._open_loops:
            if loop.resolved:
                continue
            overlap = len(q_kw & _keywords(loop.topic))
            if overlap > 0:
                scored.append((overlap, loop))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [l.description for _, l in scored[:config.context.max_open_loops]]

    def _skips_semantic(self, question: str) -> bool:
        """True for a short followup/confirm/dismissal turn — only when the
        intent recorded by observe_user_turn is for this exact text."""
        turn = self._turn
        if not turn or turn[0] != question.strip() or turn[1] not in _SEMANTIC_SKIP_INTENTS:
            return False
        return len(re.findall(r"[a-z0-9']+", question.lower())) <= _SEMANTIC_SKIP_MAX_TOKENS

    async def _retrieve_semantic(self, question: str) -> list[str]:
        if self._store is None:
            return []
        try:
            if self._skips_semantic(question):
                logger.info("[TIMING]     semantic retrieval skipped: short referential turn")
                return []
            if not self._store.has_records():
                logger.info("[TIMING]     semantic retrieval skipped: no stored memories")
                return []
            t0 = time.perf_counter()
            embedding = await self._embedder.embed(question)
            logger.info(f"[TIMING]     embed(): {time.perf_counter()-t0:.3f}s")
            if embedding is None:
                return []
            t1 = time.perf_counter()
            results = self._store.search(
                embedding, top_k=config.context.max_semantic_memories,
                min_similarity=config.context.similarity_threshold,
            )
            logger.info(f"[TIMING]     store.search(): {time.perf_counter()-t1:.4f}s")
            cutoff = time.time() - config.context.semantic_recency_guard_seconds
            return [r.content for r, _ in results if r.timestamp < cutoff]
        except Exception as e:
            return []

    # ── Post-reply bookkeeping ─────────────────────────────────────────────

    async def record_assistant_turn(self, question: str, response: str, intent: dict | None = None) -> None:
        """Resolves the just-discussed open loop and, if the turn meets
        the memory policy, persists a semantic memory. Best-effort —
        never lets a memory failure surface to the caller."""
        try:
            self._resolve_open_loop(self._state.active_topic)
            candidate = self._memory_candidate(question, intent)
            if candidate:
                await self._persist_memory(candidate)
        except Exception as e:
            logger.debug(f"Context bookkeeping failed (non-fatal): {e}")

    def _add_open_loop(self, description: str, topic: str) -> None:
        if any(not l.resolved and l.topic == topic and l.description == description
               for l in self._open_loops):
            return
        self._open_loops.append(OpenLoop(description=description, topic=topic))
        # Resolved loops are never read again — prune them and bound the list.
        self._open_loops = [l for l in self._open_loops if not l.resolved][-_OPEN_LOOPS_MAX:]

    def _resolve_open_loop(self, topic: str | None) -> None:
        if not topic:
            return
        for loop in self._open_loops:
            if loop.topic == topic:
                loop.resolved = True

    def _memory_candidate(self, question: str, intent: dict | None) -> MemoryRecord | None:
        """
        MEMORY POLICY — persist only explicit "remember this" / stable-
        preference / decision cues. Never persist smalltalk, greetings,
        jokes, or other transient conversational noise, and never persist
        every task-intent turn just because it's task-shaped — notes and
        reminders already have their own durable storage (notepad.py /
        timer.py); duplicating them here would be redundant. When
        uncertain, prefer not storing.
        """
        intent_name = (intent or {}).get("intent", "")
        if intent_name in _NOISE_INTENTS:
            return None

        if _REMEMBER_CUE_RE.search(question):
            mem_type = "preference" if _PREFERENCE_RE.search(question) else "fact"
            return MemoryRecord(
                content=question.strip(), mem_type=mem_type,
                topic=self._state.active_topic or "", importance=0.8, source="user_stated",
            )

        return None

    async def _persist_memory(self, candidate: MemoryRecord) -> None:
        if self._store is None or candidate.mem_type not in VALID_MEM_TYPES:
            return
        embedding = await self._embedder.embed(candidate.content)
        if embedding is None:
            return
        candidate.embedding = embedding

        existing = self._store.find_similar(
            embedding, candidate.mem_type, candidate.topic,
            threshold=config.context.dedup_threshold,
        )
        if existing and existing.id is not None:
            # Prefer the newer statement over a stale duplicate rather than
            # accumulating near-identical or contradictory entries.
            self._store.update(existing.id, candidate.content, embedding, candidate.timestamp)
            logger.debug(f"Semantic memory updated (id={existing.id}): '{candidate.content[:50]}'")
        else:
            new_id = self._store.add(candidate)
            logger.debug(f"Semantic memory stored (id={new_id}, type={candidate.mem_type}): '{candidate.content[:50]}'")

    # ── Long-conversation compaction ───────────────────────────────────────

    def _on_memory_evict(self, entry: MemoryEntry) -> None:
        """Called by Memory right before it drops its oldest entry. Buffers
        it; once enough have piled up, compacts them into one
        conversation_summary memory instead of letting the detail vanish
        as the recent window slides forward."""
        self._evicted_buffer.append(entry)
        if len(self._evicted_buffer) >= self._EVICT_FLUSH_SIZE:
            self._flush_evicted_buffer()

    def _flush_evicted_buffer(self) -> None:
        if not self._evicted_buffer or self._store is None:
            self._evicted_buffer.clear()
            return
        # Compact into a topic-keyword gist, NOT a verbatim transcript —
        # recent context already owns raw dialogue; semantic memory should
        # only hold the durable gist of what's about to fall out of it.
        keywords: set[str] = set()
        for e in self._evicted_buffer:
            keywords |= _keywords(e.content)
        self._evicted_buffer.clear()
        if not keywords:
            return
        summary = "Earlier discussion touched on: " + ", ".join(sorted(keywords)[:15])
        candidate = MemoryRecord(
            content=summary, mem_type="conversation_summary",
            topic=self._state.active_topic or "", importance=0.3, source="auto_compaction",
        )
        try:
            asyncio.create_task(self._persist_memory(candidate))
        except RuntimeError:
            # No running event loop (e.g. called outside the app's async
            # context, such as in a script) — drop rather than crash.
            logger.debug("No running loop for compaction persist — summary dropped.")


# Singleton — shared by processor.py and llm_service.py so conversation
# state, open loops, and semantic memory stay consistent across both call
# sites. Matches the existing singleton pattern (memory, mood_manager,
# queue_manager, ws_server).
context_manager = ContextManager()