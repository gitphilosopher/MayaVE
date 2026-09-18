"""
core/mood.py
Persistent emotional mood layer for Maya — event-driven design.

Instead of a [tag] independently deciding mood, every mood-relevant
happening is a MoodEvent from one of four sources, each with its own
rules/weight:

  USER            — genuine sadness/frustration/provocation detected in the
                     user's raw text (observe_user_text, before Ollama runs).
                     Ragebait (insult wording + a teasing marker like "lol")
                     produces a short-lived TRANSIENT reaction — playful
                     irritation that expires on its own — rather than a
                     full persistent lock. A bare insult with no teasing
                     marker is genuine provocation and locks mood as before.
  SKILL / SYSTEM  — skills can report an event directly via report_event()
                     (e.g. critical battery -> concern/annoyance). These
                     bypass the user-context requirement entirely, since
                     they reflect something actually happening to Maya, not
                     the conversation.
  MAYA            — Ollama's own [angry]/[sad] tags (observe_turn). A tag
                     no longer creates or reinforces mood by itself. It can
                     only CONFIRM an event already raised earlier this same
                     turn (by USER or SKILL/SYSTEM) for a modest bonus. An
                     unconfirmed tag is expressive only — Maya can still say
                     it, she just doesn't become it.

Mood is only ever evaluated ONCE per turn (observe_turn), never per
sentence — a factual [neutral] line inside an angry reply isn't Maya
calming down.

Two layers of state:
  - Persistent mood (mood, intensity) — decays via distraction ticks,
    per-minute time decay, and a hard 20-minute forget.
  - Transient reaction (temp_mood, temp_expires_at) — a short playful
    blip (e.g. ragebait) that colours the avatar's face for a bit and
    fades on its own, mostly independent of the persistent decay math.

is_teasing() (Behavioral Engine integration — core/behavior_engine.py):
  Exposes whether a transient teasing reaction is currently active so the
  engine can distinguish "genuinely angry" from "mock outrage" when
  composing an [angry]-tagged line into a blended expression. Read-only —
  no new mood logic, just a public accessor for existing state.
"""

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_STICKY_MOODS = {"angry", "sad"}   # the only emotions the persistent layer tracks

# ── Tuning ──────────────────────────────────────────────────────────────────
_TRIGGER_STEP           = 0.35   # persistent intensity added by a genuine event
_TAG_CONFIRM_BONUS      = 0.15   # extra bump when Maya's tag confirms this turn's event
_TEASE_INTENSITY        = 0.20   # transient-reaction strength for ragebait/teasing
_TEASE_DURATION_SEC     = 90     # how long a teasing reaction lasts before fading
_TEASE_PERSISTENT_BLEED = 0.25   # fraction of a tease that still bleeds into persistent mood
_EXCITEMENT_RELIEF      = 0.15   # persistent intensity eased off by genuine user excitement
_DISTRACTION_DECAY      = 0.22   # intensity removed per turn with no valid/confirmed event
_TIME_DECAY_PER_MIN     = 0.03   # gradual intensity removed per minute idle
_APOLOGY_FORGIVENESS    = 0.65   # intensity removed on a detected apology
_FORGET_AFTER_SEC       = 20 * 60  # hard reset if unreinforced this long

# Per-source influence weight applied to an event's declared intensity.
_SOURCE_WEIGHT = {
    "system": 1.0,   # e.g. critical battery — trusted, bypasses user context
    "skill":  1.0,   # same trust level as system
    "user":   1.0,   # genuine user emotion
}

_APOLOGY_RE = re.compile(
    r"\b(sorry|i apologi[sz]e|my bad|my fault|forgive me|"
    r"didn'?t mean (it|that|to)|i mean no (harm|offense)|"
    r"i shouldn'?t have|that was (wrong|rude|out of line) of me)\b",
    re.IGNORECASE,
)

# Direct insults aimed at Maya herself, any stack of intensifiers allowed
# between "you are" and the insult word (including none).
_PROVOCATION_RE = re.compile(
    r"\b(you'?re|you\s+are|you'?ve\s+been|ur|u\s+are)\s+"
    r"(?:(?:so|really|too|very|extremely|super|pretty|quite|incredibly|"
    r"absolutely|totally|kind\s+of|sort\s+of)\s+)*"
    r"(?:such\s+an?\s+)?"
    r"(annoying|useless|stupid|dumb|terrible|awful|bad|broken|garbage|trash|"
    r"pathetic|worthless|slow|lazy|incompetent|horrible)\b"
    r"|\b(you|u)\s+(don'?t|dont|do\s+not)\s+work\s+(properly|right|at\s+all)\b"
    r"|\b(you|u)\s+(just\s+)?(waste|wasted|are\s+wasting|keep\s+wasting)\s+(my\s+)?time\b"
    r"|\b(shut up|screw you|you suck|i hate you|"
    r"you'?re (garbage|trash|a joke)|worst (assistant|ai)|useless (bot|assistant|ai))\b",
    re.IGNORECASE,
)

# Markers indicating the message is playful rather than a genuine attack —
# when these co-occur with _PROVOCATION_RE it's ragebait/teasing, not a
# real insult (see observe_user_text).
_TEASING_MARKERS_RE = re.compile(
    r"\b(lol|lmao|lmfao|rofl|jk|just kidding|just teasing|only teasing|"
    r"just messing(\s+with\s+you)?|kidding|teasing)\b|😂|🤣|😜|😉",
    re.IGNORECASE,
)

_USER_SADNESS_RE = re.compile(
    r"\b(i feel|i'?m feeling|feeling)\s+(so |really |very |extremely )?"
    r"(sad|terrible|awful|down|depressed|miserable|heartbroken|hopeless|horrible|upset|devastated)\b"
    r"|\bi'?m\s+(so |really |very |extremely )?(sad|down|depressed|upset|heartbroken|miserable|devastated)\b"
    r"|\bi\s+(failed|lost|messed up|screwed up|blew it)\b"
    r"|\bi'?m\s+crying\b"
    r"|\bi\s+feel\s+(like\s+)?(a\s+)?(failure|worthless)\b",
    re.IGNORECASE,
)

_USER_FRUSTRATION_RE = re.compile(
    r"\bi'?m\s+(so |really |very |extremely )?(frustrated|annoyed|irritated|mad|pissed|fed up)\b"
    r"|\bthis\s+is\s+(so |really |very )?(frustrating|annoying|ridiculous|infuriating)\b"
    r"|\bi\s+can'?t\s+believe\s+this\b"
    r"|\bugh+\b",
    re.IGNORECASE,
)

_USER_EXCITEMENT_RE = re.compile(
    r"\b(yay+|woohoo|can'?t\s+wait|i'?m\s+(so |really )?excited|this\s+is\s+(so\s+)?(great|awesome|amazing))\b"
    r"|!{2,}",
    re.IGNORECASE,
)


@dataclass
class MoodEvent:
    source:    str                 # "user" | "skill" | "system" | "maya"
    emotion:   str                 # only "angry"/"sad" affect persistent mood
    intensity: float = _TRIGGER_STEP
    duration:  float | None = None  # None = persistent event; seconds = transient reaction
    reason:    str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class _MoodState:
    mood: str = "neutral"
    intensity: float = 0.0
    last_reinforced: float = field(default_factory=time.time)
    unrelated_turns: int = 0
    temp_mood: str | None = None     # transient reaction (e.g. playful irritation)
    temp_expires_at: float = 0.0


class MoodManager:
    def __init__(self):
        self._s = _MoodState()
        self._pending_events: list[MoodEvent] = []   # this turn's events, awaiting tag confirmation

    # ── Event pipeline ───────────────────────────────────────────────────

    def _apply_event(self, event: MoodEvent) -> None:
        self._apply_time_decay()
        weight = _SOURCE_WEIGHT.get(event.source, 1.0)
        strength = event.intensity * weight

        if event.duration:
            self._s.temp_mood = event.emotion
            self._s.temp_expires_at = time.time() + event.duration
            self._reinforce_persistent(event.emotion, strength * _TEASE_PERSISTENT_BLEED)
            logger.info(
                f"Transient mood [{event.source}] -> {event.emotion} for "
                f"{event.duration:.0f}s ({event.reason})"
            )
        else:
            self._reinforce_persistent(event.emotion, strength)
            logger.info(
                f"Mood event [{event.source}] -> {self._s.mood} "
                f"({self._s.intensity:.2f}) — {event.reason}"
            )

    def _reinforce_persistent(self, emotion: str, intensity_add: float) -> None:
        if intensity_add <= 0 or emotion not in _STICKY_MOODS:
            return
        if self._s.mood == emotion:
            self._s.intensity = min(1.0, self._s.intensity + intensity_add)
        elif self._s.intensity < intensity_add:
            self._s.mood = emotion
            self._s.intensity = intensity_add
        # else: a stronger opposite mood resists being overwritten by a weaker event
        self._s.last_reinforced = time.time()
        self._s.unrelated_turns = 0

    def report_event(self, source: str, emotion: str, intensity: float = _TRIGGER_STEP,
                      duration: float | None = None, reason: str = "") -> None:
        """
        Public hook for non-conversational sources (skills, system checks)
        to report a mood-relevant event directly — e.g. a critical-battery
        skill reporting concern/annoyance. Bypasses the user-context
        requirement that gates Maya's own [tag] confirmations. Only
        "angry"/"sad" are tracked; anything else is ignored.
        """
        if emotion not in _STICKY_MOODS:
            logger.debug(f"report_event ignored — '{emotion}' isn't a tracked mood.")
            return
        event = MoodEvent(source=source, emotion=emotion, intensity=intensity,
                           duration=duration, reason=reason)
        self._apply_event(event)
        self._pending_events.append(event)

    # ── Observation hooks (public API — unchanged signatures) ───────────

    def observe_turn(self, expressions: list[str]) -> None:
        """
        Call ONCE per conversational turn with every expression tag used
        in it — not per sentence. A sticky tag only reinforces persistent
        mood if it CONFIRMS an event already raised this turn (by the user
        or a skill/system report); otherwise it's expressive only, and is
        treated like having no sticky tag at all for decay purposes.
        """
        self._apply_time_decay()

        sticky_present = [e for e in expressions if e in _STICKY_MOODS]
        pending = self._pending_events
        self._pending_events = []   # consume — one-shot per turn

        if sticky_present:
            chosen = next((e for e in sticky_present if e == self._s.mood), sticky_present[0])
            confirmed = any(ev.emotion == chosen for ev in pending)
            if confirmed:
                self._reinforce_persistent(chosen, _TAG_CONFIRM_BONUS)
                logger.info(f"Tag [{chosen}] confirmed this turn's event -> {self._s.mood} ({self._s.intensity:.2f})")
            else:
                logger.debug(f"Tag [{chosen}] has no backing event this turn — expressive only.")
                if self._s.mood != "neutral":
                    self._register_unrelated_turn()
        elif self._s.mood != "neutral":
            self._register_unrelated_turn()

    def observe_user_text(self, text: str) -> None:
        """
        Check incoming user text BEFORE the turn is routed. Classifies it
        into (at most) one USER-sourced event — genuine provocation,
        ragebait/teasing, sadness, or frustration — applies it immediately,
        and stashes it so this turn's Ollama tag can confirm it. Excitement
        eases an active mood; an apology forgives it.
        """
        self._apply_time_decay()

        is_provocation = bool(_PROVOCATION_RE.search(text))
        is_teasing     = bool(_TEASING_MARKERS_RE.search(text))

        if is_provocation and not is_teasing:
            event = MoodEvent(source="user", emotion="angry",
                               intensity=_TRIGGER_STEP, reason="direct insult")
            self._apply_event(event)
            self._pending_events.append(event)
            return

        if is_provocation and is_teasing:
            event = MoodEvent(source="user", emotion="angry", intensity=_TEASE_INTENSITY,
                               duration=_TEASE_DURATION_SEC, reason="ragebait/teasing")
            self._apply_event(event)
            self._pending_events.append(event)
            return

        if _USER_SADNESS_RE.search(text):
            event = MoodEvent(source="user", emotion="sad",
                               intensity=_TRIGGER_STEP, reason="user expressed sadness")
            self._apply_event(event)
            self._pending_events.append(event)
            return

        if _USER_FRUSTRATION_RE.search(text):
            event = MoodEvent(source="user", emotion="angry", intensity=_TRIGGER_STEP * 0.8,
                               reason="user expressed frustration")
            self._apply_event(event)
            self._pending_events.append(event)
            return

        if _USER_EXCITEMENT_RE.search(text) and self._s.mood != "neutral":
            self._s.intensity = max(0.0, self._s.intensity - _EXCITEMENT_RELIEF)
            logger.debug(f"User excitement -> mood eased to {self._s.intensity:.2f}")
            if self._s.intensity <= 0.05:
                self._reset()
            return

        if self._s.mood == "neutral":
            return

        if _APOLOGY_RE.search(text):
            self._s.intensity = max(0.0, self._s.intensity - _APOLOGY_FORGIVENESS)
            logger.info(f"Apology detected -> mood intensity now {self._s.intensity:.2f}")
            if self._s.intensity <= 0.05:
                self._reset()

    def _register_unrelated_turn(self) -> None:
        self._s.unrelated_turns += 1
        self._s.intensity = max(0.0, self._s.intensity - _DISTRACTION_DECAY)
        logger.debug(f"Distraction tick {self._s.unrelated_turns} -> {self._s.intensity:.2f}")
        if self._s.intensity <= 0.05:
            self._reset()

    def _apply_time_decay(self) -> None:
        if self._s.mood == "neutral":
            return
        elapsed = time.time() - self._s.last_reinforced
        if elapsed >= _FORGET_AFTER_SEC:
            logger.info("Mood expired — Maya forgot what she was upset about.")
            self._reset()
            return
        decay = (elapsed / 60.0) * _TIME_DECAY_PER_MIN
        if decay > 0:
            self._s.intensity = max(0.0, self._s.intensity - decay)
            self._s.last_reinforced = time.time()
            if self._s.intensity <= 0.05:
                self._reset()

    def _reset(self) -> None:
        if self._s.mood != "neutral":
            logger.info(f"Mood reset: {self._s.mood} -> neutral")
        temp_mood, temp_expires = self._s.temp_mood, self._s.temp_expires_at
        self._s = _MoodState()
        # A live transient reaction survives a persistent reset — it's an
        # independent, self-expiring blip, not part of the forgotten grudge.
        if temp_mood and time.time() < temp_expires:
            self._s.temp_mood = temp_mood
            self._s.temp_expires_at = temp_expires

    # ── Read ─────────────────────────────────────────────────────────────

    def current(self) -> tuple[str, float]:
        self._apply_time_decay()
        return self._s.mood, self._s.intensity

    def is_teasing(self) -> bool:
        """True while a transient playful/teasing reaction is active (see
        observe_user_text's ragebait branch) — used by BehaviorEngine
        (core/behavior_engine.py) to tell mock outrage apart from a
        genuine angry tag."""
        return self._s.temp_mood == "angry" and time.time() < self._s.temp_expires_at

    def baseline_expression(self) -> str:
        """What Maya's face should rest at between sentences / at idle."""
        if self._s.temp_mood:
            if time.time() < self._s.temp_expires_at:
                return self._s.temp_mood
            self._s.temp_mood = None   # expired — stop overriding

        mood, intensity = self.current()
        return mood if intensity > 0.15 else "neutral"

    def system_prompt_note(self) -> str:
        """Injected into the LLM system prompt each turn so Ollama's own
        generation stays in character with Maya's current mood."""
        mood, intensity = self.current()
        note = ""

        if intensity > 0.15:
            strength = "mildly" if intensity < 0.4 else "noticeably" if intensity < 0.7 else "very"
            if mood == "angry":
                note = (
                    f"\n\nCURRENT MOOD: You are {strength} angry right now because of "
                    "something earlier in this conversation — likely senpai insulted or "
                    "put you down. Stay short, clipped, and irritated, and tag your lines "
                    "[angry], not [sad] — you're not hurt, you're annoyed. Do not reset to "
                    "cheerful on your own — only warm back up if senpai apologises, the "
                    "subject changes enough to distract you, or you've had time to cool off."
                )
            elif mood == "sad":
                note = (
                    f"\n\nCURRENT MOOD: You are {strength} sad right now because of "
                    "something earlier in this conversation. Let your tone stay quieter "
                    "and more subdued until something genuinely cheers you up, distracts "
                    "you, or enough time passes."
                )

        # Only add the teasing note when it isn't already covered by a
        # genuine persistent mood — this is a light, short-lived reaction.
        if (self._s.temp_mood == "angry" and time.time() < self._s.temp_expires_at
                and mood != "angry"):
            note += (
                "\n\nSENPAI IS TEASING YOU: playful ragebait, not a real insult. React with "
                "mock offense or sass — [angry] is fine for a line or two — but this is "
                "banter, not a real grudge. Don't stay mad."
            )

        return note


# Singleton
mood_manager = MoodManager()