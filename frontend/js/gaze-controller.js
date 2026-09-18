/**
 * frontend/js/gaze-controller.js
 * Phase 3 — screen attention & gaze for Maya's avatar.
 *
 * Maintains an attention SESSION (not a per-frame reset): any reported
 * screen-activity event (typing, window focus, etc.) holds full attention
 * for a "few minutes" sustain window, refreshed by further activity at the
 * same spot. Once that window elapses — either because activity stopped
 * entirely, or because the same activity kept the SAME spot/type for the
 * whole window despite continuing — attention fades out and idle/fidget
 * behavior becomes eligible again. A genuinely new/different target (e.g.
 * switching windows) always overrides immediately, resetting both clocks.
 *
 * While OBSERVING, eyes hold a fixed "looking toward the screen" pose with
 * no idle drift (only vertical tracking of the target's y) — deliberately
 * NOT the small ambient jitter used elsewhere in the avatar, since a still
 * gaze reads as more attentive here. Head/neck contribution is kept small
 * (lower-priority/base tier — see avatar.js's startHeadMovement()).
 *
 * No screen capture / OCR / CV lives here — observeScreenActivity() is a
 * pure input API a future observation module can call.
 */

export const ATTENTION_STATE = Object.freeze({
    IDLE:      "IDLE",
    OBSERVING: "OBSERVING",
    SPEAKING:  "SPEAKING",
    SLEEPING:  "SLEEPING",
});

const _TARGET_TYPES = new Set(["code", "text", "window", "notification", "general", "unknown"]);

// ── Timing / tuning constants ────────────────────────────────────────────────
const GAZE_SUSTAIN_BASE_MS   = 3 * 60_000;  // "a few minutes" a gaze session holds before boredom is eligible
const GAZE_SUSTAIN_JITTER_MS = 45_000;      // +/- variation so it never expires on the dot
const GAZE_FADE_MS           = 6000;        // time to ease attention 1 -> 0 once a session truly ends (no more activity at all)
const BOREDOM_RAMP_MS        = 4 * 60_000;  // time to gradually ease attention 1 -> 0 while the SAME activity keeps pinging past the sustain threshold
const ATTENTION_MIN_ACTIVE   = 0.02;

const SAME_TARGET_BUCKET = 0.03;            // normalized-distance bucket for "same spot" detection

// Eyes hold this fixed yaw while OBSERVING instead of tracking left/right —
// sign is whichever direction reads as "toward the screen" on this rig;
// flip if it turns out to be mirrored once tested in-engine.
const EYE_GAZE_LOCK_X = 0.3;
const EYE_PITCH_RANGE = 0.14;               // vertical eye tracking still follows the target's y

const HEAD_YAW_RANGE   = 0.04;              // reduced neck contribution
const HEAD_PITCH_RANGE = 0.02;

const GAZE_EYE_LERP  = 0.08;
const GAZE_HEAD_LERP = 0.04;

function _clamp01(v) {
    return Math.max(0, Math.min(1, typeof v === "number" ? v : 0));
}

// Smooth, monotonic 0->1 ease used for the boredom ramp — no linear "cliff".
function _smoothstep(p) {
    return p * p * (3 - 2 * p);
}

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

    _rollSustainDuration() {
        return GAZE_SUSTAIN_BASE_MS + (Math.random() * 2 - 1) * GAZE_SUSTAIN_JITTER_MS;
    }

    // ── External state hooks (called from avatar.js) ────────────────────────

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

    // Speaking overrides gaze visually but leaves the attention session
    // (target/level/timers) untouched so gaze resumes where it left off.
    setSpeaking(speaking) {
        this._speaking = speaking;
    }

    // ── Input API ─────────────────────────────────────────────────────────

    /**
     * observeScreenActivity({ x, y, intensity, type })
     * x/y: normalized 0..1 screen coordinates.
     * intensity: 0..1, defaults to 0.5.
     * type: one of code|text|window|notification|general|unknown.
     *
     * Any call refreshes the "still active" clock. A call whose x/y/type
     * differs enough from the last one (e.g. a window switch) is treated
     * as a fresh gazing event: it resets the boredom clock too and jumps
     * attention straight to full, overriding wherever gaze currently is.
     * Repeated calls at the same spot (e.g. continued typing) keep
     * attention up only until that same-activity session's own sustain
     * window elapses — after that they no longer re-energize it, so she
     * still gets bored of the same thing eventually.
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
        // else: this same activity already ran past its sustain window —
        // boredom has already started ramping (see update()); further pings
        // at the same spot must NOT reset attention back to 1 or restart
        // the ramp — she still gets bored of the same thing eventually.

        this._lastSignature = signature;
        this._lastEventAt   = now;
        this._target = { x, y, intensity, type };
    }

    // ── Frame update — call every frame (e.g. from main.js's animate loop) ──

    update(delta) {
        if (!this._awake) {
            this._state = ATTENTION_STATE.SLEEPING;
            return;
        }

        const now = performance.now();

        if (this._target) {
            const sinceLastEvent = now - this._lastEventAt;
            const unchangedFor   = now - this._unchangedSinceAt;

            // Activity has genuinely stopped (no pings at all for the whole
            // sustain window) — this is real session expiration, so keep the
            // original fast GAZE_FADE_MS ease down to idle.
            const sessionStopped = sinceLastEvent > this._sustainThreshold;

            // Same activity is still pinging, but has run past its sustain
            // window — boredom, not expiration. Ease down gradually over
            // BOREDOM_RAMP_MS instead of the fast fade. sessionStopped
            // implies this too (unchangedFor >= sinceLastEvent always), so
            // it's only reached here when pings are still arriving.
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
            // else: still within the sustain window — attentionLevel stays
            // at whatever observeScreenActivity() last set it to (1).

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

            // Eyes hold a fixed "looking at the screen" pose — no drift,
            // no left/right tracking off the target's x.
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

    hasActiveTarget() {
        return this._awake && !this._speaking && this._state === ATTENTION_STATE.OBSERVING;
    }

    getEyeOffset()  { return this._eyeOffset; }
    getHeadOffset() { return this._headOffset; }

    // ── Public status API — for the existing idle system / future use ───────

    getAttentionState() { return this._state; }
    getAttentionLevel()  { return this._attentionLevel; }

    isBored() {
        if (!this._target) return false;
        const now = performance.now();
        return (now - this._lastEventAt > this._sustainThreshold) ||
               (now - this._unchangedSinceAt > this._sustainThreshold);
    }
}

// Singleton — one avatar, one attention session.
export const gazeController = new GazeController();