/**
 * frontend/js/animation-controller.js
 * Coordinates write ownership for the avatar skeleton across VRMA actions,
 * procedural fidgets, and continuous idle motion. Exported priorities let
 * higher-tier animations take precedence while lower-tier writers skip bones
 * they do not currently own.
 *
 * Ownership is advisory: an animation claims its affected bones when it
 * starts and releases them when it ends; continuous BASE writers check
 * canWrite() before each write but do not claim bones. The singleton is shared
 * by these systems for the avatar's one skeleton.
 *
 * Releasing ownership makes the bone immediately writable, while
 * handoffProgress() ramps from 0 to 1 over HANDOFF_MS. Continuous writers
 * can use that factor to blend from the bone's current pose; one-shot
 * procedural tweens typically start from the live pose directly.
 */

export const PRIORITY = Object.freeze({
    BASE: 1,   // continuous idle motion (head sway, etc.)
    FIDGET: 2,   // idle fidgets — VRMA-backed or procedural
    ACTION: 3,   // deliberate/action animations (wave, nod, giggle, ...)
});

// How long a released bone eases back under the lower-priority writer's
// control before that writer resumes driving it at full strength.
const HANDOFF_MS = 350;

class AnimationController {
    /** Arbitrates bone writers and tracks release times for pose handoff. */
    constructor() {
        this._owners = new Map(); // boneName -> { tier, ownerId }
        this._handoffs = new Map(); // boneName -> releasedAt (performance.now())
    }

    /**
        * Assign bones to an owner unless another owner has a strictly higher
        * tier. Equal-tier claims may replace one another; returns claimed bones.
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

    /** Free this owner's bones immediately and start their handoff windows. */
    release(ownerId) {
        const now = performance.now();
        for (const [name, owner] of this._owners) {
            if (owner.ownerId === ownerId) {
                this._owners.delete(name);
                this._handoffs.set(name, now);
            }
        }
    }

    /** Whether a writer at `tier` may write `boneName` under current ownership. */
    canWrite(boneName, tier) {
        const cur = this._owners.get(boneName);
        return !cur || cur.tier <= tier;
    }

    /**
        * Return the linear blend factor for a recently released bone: 0 at
        * release, rising to 1 over HANDOFF_MS. Returns 1 when no handoff is
        * active; elapsed handoffs are removed from the tracking map.
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