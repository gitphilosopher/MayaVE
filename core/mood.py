"""
core/mood.py
Persistent mood layer for Maya's emotional state.

This module is responsible for the agent's ongoing emotional baseline and its
short-lived reaction overlays. Mood is not derived from a single tag in
isolation; instead, user text, skill/system events, and the current turn's
explicit expressive tags are converted into `MoodEvent` records and evaluated
through a shared policy.

The system has two interacting state layers:
- persistent mood: the long-lived `mood` and `intensity` values used to guide
  reply tone and avatar resting expression
- transient reaction: a short-lived `temp_mood` that fades on its own and can
  temporarily override the baseline without becoming a persistent grudge

The public flow is intentionally narrow:
- `observe_user_text()` classifies incoming user messages before routing
- `report_event()` lets non-user sources (skills/system checks) inject
  mood-relevant events directly
- `observe_turn()` evaluates the current turn's explicit tags once and uses
  them as confirmation or as a distraction signal, not as a standalone mood
  trigger
- `baseline_expression()` and `system_prompt_note()` provide the state needed by
  the avatar and the LLM prompt layer

Important behavior:
- a direct insult is a genuine angry event; teasing plus insult is treated as a
  short-lived mocking reaction instead of a full persistent lock
- a sticky emotion only reinforces the persistent mood when it confirms an event
  already raised this turn; otherwise it is expressive-only and should not be
  treated as a new emotional state
- the mood decays with distraction, time, and a hard forget threshold, while
  transient teasing can persist independently for a brief window
- `is_teasing()` exposes the transient teasing state to the behavior engine so it
  can distinguish mock outrage from a truly sustained angry baseline
"""

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_STICKY_MOODS = {"angry", "sad"}   # the only emotions the persistent layer tracks

_TRIGGER_STEP = 0.35
_TAG_CONFIRM_BONUS = 0.15
_TEASE_INTENSITY = 0.20
_TEASE_DURATION_SEC = 90
_TEASE_PERSISTENT_BLEED = 0.25
_EXCITEMENT_RELIEF = 0.15
_DISTRACTION_DECAY = 0.22
_TIME_DECAY_PER_MIN = 0.03
_APOLOGY_FORGIVENESS = 0.65
_FORGET_AFTER_SEC = 20 * 60

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
    """One mood-relevant signal from a given source and turn."""

    source: str
    emotion: str
    intensity: float = _TRIGGER_STEP
    duration: float | None = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class _MoodState:
    """Internal persistent + transient mood snapshot used by the manager."""

    mood: str = "neutral"
    intensity: float = 0.0
    last_reinforced: float = field(default_factory=time.time)
    unrelated_turns: int = 0
    temp_mood: str | None = None
    temp_expires_at: float = 0.0


class MoodManager:
    """Manage Maya's persistent mood and transient emotional reactions."""

    def __init__(self):
        self._s = _MoodState()
        self._pending_events: list[MoodEvent] = []

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
        """Record a direct mood event from a skill, system check, or other non-user source."""
        if emotion not in _STICKY_MOODS:
            logger.debug(f"report_event ignored — '{emotion}' isn't a tracked mood.")
            return
        event = MoodEvent(source=source, emotion=emotion, intensity=intensity,
                           duration=duration, reason=reason)
        self._apply_event(event)
        self._pending_events.append(event)

    # ── Observation hooks (public API — unchanged signatures) ───────────

    def observe_turn(self, expressions: list[str]) -> None:
        """Evaluate the turn's explicit expression tags once, using them as confirmation only."""
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
        """Classify user text into the next mood event and stash it for this-turn confirmation."""
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
        # A live transient reaction survives a persistent reset because it is an
        # independent self-expiring blip, not a forgotten grudge.
        if temp_mood and time.time() < temp_expires:
            self._s.temp_mood = temp_mood
            self._s.temp_expires_at = temp_expires

    # ── Read ─────────────────────────────────────────────────────────────

    def current(self) -> tuple[str, float]:
        self._apply_time_decay()
        return self._s.mood, self._s.intensity

    def is_teasing(self) -> bool:
        """Return whether a transient teasing reaction is still currently active."""
        return self._s.temp_mood == "angry" and time.time() < self._s.temp_expires_at

    def baseline_expression(self) -> str:
        """Return the expression that should be shown when Maya is idle or between lines."""
        if self._s.temp_mood:
            if time.time() < self._s.temp_expires_at:
                return self._s.temp_mood
            self._s.temp_mood = None   # expired — stop overriding

        mood, intensity = self.current()
        return mood if intensity > 0.15 else "neutral"

    def system_prompt_note(self) -> str:
        """Build the mood guidance sent to the LLM prompt for the current turn."""
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

        # Only add the teasing note when a real persistent mood is not already
        # dominating the turn; this is a light, short-lived reaction.
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