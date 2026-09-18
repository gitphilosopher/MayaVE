/**
 * frontend/js/life-motion-controller.js
 * Phase 4 — subtle continuous "alive" motion: breathing, slow posture
 * variation, asymmetric shoulder micro-motion, and occasional posture
 * adjustments.
 *
 * BASE-tier only (see animation-controller.js): every bone write is
 * gated on animationController.canWrite(boneName, PRIORITY.BASE) and
 * never calls claim() — a FIDGET/ACTION owner always wins outright, and
 * this controller yields the bone immediately when that happens. On
 * release, animationController.handoffProgress() eases the bone back
 * under this controller's control instead of snapping.
 *
 * Call update(vrm, state) once per frame from avatar.js's existing
 * head-sway rAF loop — no separate loop is created here.
 */

import { animationController, PRIORITY } from "./animation-controller.js";

// Tunable amplitudes (radians) and periods (ms). Deliberately mismatched
// periods/phases across signals so they never read as synchronized sines.
export const LIFE_MOTION_CONFIG = {
    breathing: { period: 4200, amplitude: 0.012 },
    posture:   { period: 9000, amplitude: 0.01, phase: 0.8 },
    shoulderL: { period: 5300, amplitude: 0.008, phase: 0 },
    shoulderR: { period: 6100, amplitude: 0.007, phase: 1.7 },
    hipsAdjust: {
        amplitude: 0.015,
        durationMs: 2500,
        minDelayMs: 18000,
        maxDelayMs: 32000,
    },
    // Amplitude multipliers per avatar state.
    scaleSpeaking:  0.35,
    scaleObserving: 0.6,
};

function _smoothstep(p) { return p * p * (3 - 2 * p); }

class LifeMotionController {
    constructor() {
        this._t0 = performance.now();

        this._hipsValue    = 0;
        this._hipsFromZ     = 0;
        this._hipsTargetZ   = 0;
        this._hipsAdjustAt  = null;   // start time of current tween, or null when settled
        this._hipsLastAt    = performance.now();
        this._hipsNextDelay = this._rollHipsDelay();
    }

    _rollHipsDelay() {
        const { minDelayMs, maxDelayMs } = LIFE_MOTION_CONFIG.hipsAdjust;
        return minDelayMs + Math.random() * (maxDelayMs - minDelayMs);
    }

    /** Writes `target` (already scaled) to bone.rotation[axis], respecting
     *  BASE-tier ownership and easing back in via handoffProgress(). */
    _writeBlended(bone, axis, target) {
        if (!bone) return;
        if (!animationController.canWrite(bone.name, PRIORITY.BASE)) return;
        const progress = animationController.handoffProgress(bone.name);
        const current  = bone.rotation[axis];
        bone.rotation[axis] = current + (target - current) * progress;
    }

    /**
     * state: { awake, speaking, attentionState }
     * Call once per frame. No-op entirely while asleep.
     */
    update(vrm, state) {
        if (!vrm?.humanoid || !state?.awake) return;

        const now = performance.now();
        const t   = now - this._t0;

        const speaking  = !!state.speaking;
        const observing = state.attentionState === "OBSERVING";
        const scale     = speaking ? LIFE_MOTION_CONFIG.scaleSpeaking
                        : observing ? LIFE_MOTION_CONFIG.scaleObserving
                        : 1;
        // Attentive/still while observing or mid-speech — no new random
        // posture shifts, though breathing/posture/shoulders continue.
        const allowHipsAdjust = !speaking && !observing;

        this._applyBreathing(vrm, t, scale);
        this._applyPosture(vrm, t, scale);
        this._applyShoulders(vrm, t, scale);
        this._applyHipsAdjustment(vrm, now, scale, allowHipsAdjust);
    }

    _applyBreathing(vrm, t, scale) {
        const chest = vrm.humanoid.getNormalizedBoneNode("chest");
        if (!chest) return;
        const { period, amplitude } = LIFE_MOTION_CONFIG.breathing;
        const target = Math.sin((t / period) * Math.PI * 2) * amplitude * scale;
        this._writeBlended(chest, "x", target);
    }

    _applyPosture(vrm, t, scale) {
        const spine = vrm.humanoid.getNormalizedBoneNode("spine");
        if (!spine) return;
        const { period, amplitude, phase } = LIFE_MOTION_CONFIG.posture;
        const target = Math.sin((t / period) * Math.PI * 2 + phase) * amplitude * scale;
        this._writeBlended(spine, "z", target);
    }

    _applyShoulders(vrm, t, scale) {
        const left  = vrm.humanoid.getNormalizedBoneNode("leftShoulder");
        const right = vrm.humanoid.getNormalizedBoneNode("rightShoulder");

        if (left) {
            const { period, amplitude, phase } = LIFE_MOTION_CONFIG.shoulderL;
            const target = Math.sin((t / period) * Math.PI * 2 + phase) * amplitude * scale;
            this._writeBlended(left, "x", target);
        }
        if (right) {
            const { period, amplitude, phase } = LIFE_MOTION_CONFIG.shoulderR;
            const target = Math.sin((t / period) * Math.PI * 2 + phase) * amplitude * scale;
            this._writeBlended(right, "x", target);
        }
    }

    _applyHipsAdjustment(vrm, now, scale, allowAdjust) {
        const hips = vrm.humanoid.getNormalizedBoneNode("hips");
        if (!hips) return;

        if (allowAdjust) {
            if (this._hipsAdjustAt === null && now - this._hipsLastAt >= this._hipsNextDelay) {
                this._hipsAdjustAt = now;
                this._hipsFromZ    = this._hipsTargetZ;
                this._hipsTargetZ  = (Math.random() - 0.5) * 2 * LIFE_MOTION_CONFIG.hipsAdjust.amplitude;
            }
            if (this._hipsAdjustAt !== null) {
                const p = Math.min(1, (now - this._hipsAdjustAt) / LIFE_MOTION_CONFIG.hipsAdjust.durationMs);
                this._hipsValue = this._hipsFromZ + (this._hipsTargetZ - this._hipsFromZ) * _smoothstep(p);
                if (p >= 1) {
                    this._hipsAdjustAt = null;
                    this._hipsLastAt   = now;
                    this._hipsNextDelay = this._rollHipsDelay();
                }
            }
        }
        // else: hold at the last settled value — no new shifts while
        // speaking/observing, but keep asserting it (see _writeBlended)
        // so a released bone still eases back to the right resting spot.

        this._writeBlended(hips, "z", this._hipsValue * scale);
    }
}

// Singleton — one avatar, one life-motion layer.
export const lifeMotionController = new LifeMotionController();
