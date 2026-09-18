/**
 * frontend/js/animation-controller.js
 * Phase 1 — bone-ownership arbiter for Maya's avatar animation systems.
 * Phase 2 — adds smooth pose handoff when a bone is released.
 *
 * Three systems can want to write the same bone in the same frame:
 * VRMA mixers, procedural tweens (headTilt/shoulderRoll), and continuous
 * base idle motion (startHeadMovement). This tracks which tier currently
 * owns each bone so a lower-priority writer can skip that bone for a
 * frame instead of fighting a higher-priority one for it.
 *
 * Ownership is advisory: callers must call canWrite() before writing,
 * claim() when they start driving a bone, and release() when done.
 *
 * Smooth handoff: releasing a bone starts a short handoff window rather
 * than freeing it outright. canWrite() still flips true immediately (the
 * lower-priority system may resume writing right away — priority rules
 * are unchanged), but handoffProgress() ramps 0→1 over that window so a
 * continuous writer (e.g. startHeadMovement) can ease its own value in
 * from the bone's current pose instead of snapping straight to it.
 * Procedural one-shot tweens (headTilt/shoulderRoll) already read the
 * bone's live rotation as their own starting point, so they're smooth by
 * construction and don't need to consult this.
 */

export const PRIORITY = Object.freeze({
    BASE:   1,   // continuous idle motion (head sway, etc.)
    FIDGET: 2,   // idle fidgets — VRMA-backed or procedural
    ACTION: 3,   // deliberate/action animations (wave, nod, giggle, ...)
});

// How long a released bone eases back under the lower-priority writer's
// control before that writer resumes driving it at full strength.
const HANDOFF_MS = 350;

class AnimationController {
    constructor() {
        this._owners   = new Map(); // boneName -> { tier, ownerId }
        this._handoffs = new Map(); // boneName -> releasedAt (performance.now())
    }

    /**
     * Claim boneNames for ownerId at tier. A bone already held by a
     * strictly higher tier under a different owner is left untouched.
     * Returns the bones actually claimed.
     */
    claim(boneNames, tier, ownerId) {
        const claimed = [];
        for (const name of boneNames) {
            const cur = this._owners.get(name);
            if (cur && cur.tier > tier && cur.ownerId !== ownerId) continue;
            this._owners.set(name, { tier, ownerId });
            // A fresh claim takes over outright — any in-progress handoff
            // from a prior release no longer applies.
            this._handoffs.delete(name);
            claimed.push(name);
        }
        return claimed;
    }

    /** Release every bone currently held by ownerId, starting a handoff. */
    release(ownerId) {
        const now = performance.now();
        for (const [name, owner] of this._owners) {
            if (owner.ownerId === ownerId) {
                this._owners.delete(name);
                this._handoffs.set(name, now);
            }
        }
    }

    /** Can `tier` write `boneName` right now? */
    canWrite(boneName, tier) {
        const cur = this._owners.get(boneName);
        return !cur || cur.tier <= tier;
    }

    /**
     * Blend factor for easing back into a bone just released by a
     * higher-priority owner: 0 right after release (hold at the current
     * pose), ramping linearly to 1 over HANDOFF_MS (back to normal direct
     * control). Returns 1 if the bone was never released or the window
     * has already elapsed.
     */
    handoffProgress(boneName) {
        const releasedAt = this._handoffs.get(boneName);
        if (releasedAt === undefined) return 1;
        const elapsed = performance.now() - releasedAt;
        if (elapsed >= HANDOFF_MS) {
            this._handoffs.delete(boneName);
            return 1;
        }
        return elapsed / HANDOFF_MS;
    }
}

// Singleton — one skeleton, one controller.
export const animationController = new AnimationController();