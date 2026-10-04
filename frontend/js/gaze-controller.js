/**
 * frontend/js/gaze-controller.js
 * Singleton attention-session and gaze-offset controller for the avatar.
 * avatar.js forwards awake/speaking state and externally observed screen
 * activity, and calls update() through main.js's per-frame loop. This module
 * does not capture the screen or inspect its contents.
 *
 * A new bucketed x/y target or type starts a fresh session at full attention.
 * Same-target reports refresh the inactivity clock but do not restart boredom
 * after a randomized ~3-minute sustain window. Continued activity then fades
 * over four minutes; silence fades over six seconds. Speaking masks the gaze
 * visually without discarding its active session.
 *
 * While observing, eyes hold a fixed screen-facing horizontal pose and track
 * target y; small head offsets track target x/y. avatar.js consumes these
 * offsets in its eye/head loops and yields head motion to higher-priority
 * animations.
 */

export const ATTENTION_STATE = Object.freeze({
    IDLE:      "IDLE",
    OBSERVING: "OBSERVING",
    SPEAKING:  "SPEAKING",
    SLEEPING:  "SLEEPING",
});

const _TARGET_TYPES = new Set(["code", "text", "window", "notification", "general", "unknown"]);

// ── Timing / tuning constants ────────────────────────────────────────────────
const GAZE_SUSTAIN_BASE_MS   = 3 * 60_000;  // baseline hold before boredom can begin
const GAZE_SUSTAIN_JITTER_MS = 45_000;      // vary session duration to avoid a fixed expiry
const GAZE_FADE_MS           = 6000;        // fade after activity reports stop
const BOREDOM_RAMP_MS        = 4 * 60_000;  // gradual fade while the same activity continues
const ATTENTION_MIN_ACTIVE   = 0.02;

const SAME_TARGET_BUCKET = 0.03;            // normalized-distance bucket for "same spot" detection

// Fixed horizontal eye pose; sign must match the rig's screen-facing direction.
const EYE_GAZE_LOCK_X = 0.3;
const EYE_PITCH_RANGE = 0.14;               // vertical eye tracking still follows the target's y

const HEAD_YAW_RANGE   = 0.04;              // reduced neck contribution
const HEAD_PITCH_RANGE = 0.02;

const GAZE_EYE_LERP  = 0.08;
const GAZE_HEAD_LERP = 0.04;

function _clamp01(v) {
    return Math.max(0, Math.min(1, typeof v === "number" ? v : 0));
}

// Smooth monotonic ramp avoids an abrupt boredom transition.
function _smoothstep(p) {
    return p * p * (3 - 2 * p);
}

/** Maintains one attention session and its smoothed eye/head offsets. */
class GazeController {
    constructor() {
        this._awake    = false;
        this._speaking = false;
        this._state    = ATTENTION_STATE.SLEEPING;

        this._target           = null;   // last accepted {x,y,intensity,type}
        this._lastEventAt      = 0;      // last time ANY activity was reported
        this._lastSignature    = null;
        this._unchangedSinceAt = 0;      // last time the reported activity actually changed
        this._sustainThreshold = this._rollSustainDuration();
        this._boredomRampStartAt = null; // set once the sustain threshold is first crossed; null = not yet bored

        this._attentionLevel = 0;

        this._eyeOffset  = { x: 0, y: 0 };
        this._headOffset = { x: 0, y: 0 };
    }

    /** Choose a slightly varied sustain window for a new activity session. */
    _rollSustainDuration() {
        return GAZE_SUSTAIN_BASE_MS + (Math.random() * 2 - 1) * GAZE_SUSTAIN_JITTER_MS;
    }

    // ── External state hooks (called from avatar.js) ────────────────────────

    /** Set wake state; sleeping clears the active target and attention session. */
    setAwake(awake) {
        this._awake = awake;
        if (!awake) {
            this._state = ATTENTION_STATE.SLEEPING;
            this._target = null;
            this._lastSignature = null;
            this._attentionLevel = 0;
            this._boredomRampStartAt = null;
        } else if (this._state === ATTENTION_STATE.SLEEPING) {
            this._state = ATTENTION_STATE.IDLE;
        }
    }

    /** Override visible gaze while preserving the current attention session. */
    setSpeaking(speaking) {
        this._speaking = speaking;
    }

    // ── Input API ─────────────────────────────────────────────────────────

    /**
    * Record activity with normalized x/y, intensity in 0..1 (default 0.5),
    * and type code|text|window|notification|general|unknown (other values
    * become unknown). A new bucketed x/y or type starts a full-attention
    * session; same-target pings refresh inactivity but do not restart
    * boredom after the sustain window.
     */
    observeScreenActivity(target) {
        if (!this._awake || !target) return;

        const x = _clamp01(target.x);
        const y = _clamp01(target.y);
        const intensity = target.intensity == null ? 0.5 : Math.max(0, Math.min(1, target.intensity));
        const type = _TARGET_TYPES.has(target.type) ? target.type : "unknown";

        const now = performance.now();
        const signature = `${Math.round(x / SAME_TARGET_BUCKET)}:${Math.round(y / SAME_TARGET_BUCKET)}:${type}`;
        const isNewActivity = signature !== this._lastSignature;

        if (isNewActivity) {
            this._unchangedSinceAt = now;
            this._sustainThreshold = this._rollSustainDuration();
            this._attentionLevel = 1;
            this._boredomRampStartAt = null;   // genuinely different target — fresh session
        } else if (now - this._unchangedSinceAt <= this._sustainThreshold) {
            this._attentionLevel = 1;
        }
        // Same-target pings after the sustain window must not restore attention
        // or restart the boredom ramp.

        this._lastSignature = signature;
        this._lastEventAt   = now;
        this._target = { x, y, intensity, type };
    }

    // ── Frame update ───────────────────────────────────────────────────────

    /** Advance session timers (delta in seconds), state, and smoothed offsets. */
    update(delta) {
        if (!this._awake) {
            this._state = ATTENTION_STATE.SLEEPING;
            return;
        }

        if (this._target) {
            // The clock is only needed while a target session exists; reading it
            // here skips a performance.now() call on every idle frame.
            const now = performance.now();
            const sinceLastEvent = now - this._lastEventAt;
            const unchangedFor   = now - this._unchangedSinceAt;

            // No reports for the sustain window ends the session and uses the
            // faster fade to idle.
            const sessionStopped = sinceLastEvent > this._sustainThreshold;

            // Ongoing same-target reports past the sustain window cause a
            // gradual boredom fade. Session expiration takes precedence above.
            const boredomActive = !sessionStopped && unchangedFor > this._sustainThreshold;

            if (sessionStopped) {
                this._boredomRampStartAt = null;
                this._attentionLevel = Math.max(0, this._attentionLevel - (delta * 1000) / GAZE_FADE_MS);
            } else if (boredomActive) {
                if (this._boredomRampStartAt === null) {
                    this._boredomRampStartAt = now;
                }
                const p = Math.min(1, (now - this._boredomRampStartAt) / BOREDOM_RAMP_MS);
                this._attentionLevel = 1 - _smoothstep(p);
            }
            // Within the sustain window, attention stays at the level set by
            // the latest new-target event.

            if (this._attentionLevel <= ATTENTION_MIN_ACTIVE) {
                this._target = null;
                this._lastSignature = null;
                this._boredomRampStartAt = null;
            }
        }

        if (this._speaking) {
            this._state = ATTENTION_STATE.SPEAKING;
        } else if (this._target && this._attentionLevel > ATTENTION_MIN_ACTIVE) {
            this._state = ATTENTION_STATE.OBSERVING;
        } else {
            this._state = ATTENTION_STATE.IDLE;
        }

        this._updateOffsets();
    }

    _updateOffsets() {
        let desiredEyeX = 0, desiredEyeY = 0, desiredHeadX = 0, desiredHeadY = 0;

        if (this._target && this._state === ATTENTION_STATE.OBSERVING) {
            const nx = (this._target.x - 0.5) * 2;   // -1..1
            const ny = (this._target.y - 0.5) * 2;

            // Keep eyes screen-facing horizontally; only track vertical target position.
            desiredEyeX = EYE_GAZE_LOCK_X * this._attentionLevel;
            desiredEyeY = ny * EYE_PITCH_RANGE * this._attentionLevel;

            desiredHeadY = nx * HEAD_YAW_RANGE * this._attentionLevel;
            desiredHeadX = ny * HEAD_PITCH_RANGE * this._attentionLevel;
        }

        this._eyeOffset.x  += (desiredEyeX  - this._eyeOffset.x)  * GAZE_EYE_LERP;
        this._eyeOffset.y  += (desiredEyeY  - this._eyeOffset.y)  * GAZE_EYE_LERP;
        this._headOffset.x += (desiredHeadX - this._headOffset.x) * GAZE_HEAD_LERP;
        this._headOffset.y += (desiredHeadY - this._headOffset.y) * GAZE_HEAD_LERP;
    }

    // ── Output for avatar.js's eye/head loops ────────────────────────────────

    /** Whether gaze should currently override idle eye/head motion. */
    hasActiveTarget() {
        return this._awake && !this._speaking && this._state === ATTENTION_STATE.OBSERVING;
    }

    getEyeOffset()  { return this._eyeOffset; }
    getHeadOffset() { return this._headOffset; }

    // ── Status API ─────────────────────────────────────────────────────────

    getAttentionState() { return this._state; }
    getAttentionLevel()  { return this._attentionLevel; }

    /** Whether an active target has exceeded its sustain window. */
    isBored() {
        if (!this._target) return false;
        const now = performance.now();
        return (now - this._lastEventAt > this._sustainThreshold) ||
               (now - this._unchangedSinceAt > this._sustainThreshold);
    }
}

// Shared so screen input, avatar motion, and fidget scheduling use one session.
export const gazeController = new GazeController();