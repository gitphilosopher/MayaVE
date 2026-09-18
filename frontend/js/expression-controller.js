/**
 * frontend/js/expression-controller.js
 * Phase 2 — centralized, layered expressionManager.setValue() arbiter.
 *
 * Several systems want to drive VRM expression keys at once: mood/skill
 * responses (emotion), short one-off overlays (wink's blinkLeft), the
 * mouth shapes during speech (lip-sync), and the blink cycle. Each layer
 * tracks its own key->value writes; for any given key, the highest-
 * priority layer that has set a value wins and is applied.
 *
 * This does NOT replace setExpression()'s public signature — it's the
 * funnel every expression write in avatar.js now goes through.
 */

// Higher number = higher priority.
export const EXPRESSION_LAYER = Object.freeze({
    BASE:    0,   // initial/default resting look
    EMOTION: 1,   // mood + skill-response tags via setExpression()
    ACTION:  2,   // short temporary overlays (e.g. wink's blinkLeft)
    LIPSYNC: 3,   // mouth shapes while speaking
    BLINK:   4,   // eyelid open/close
});

// Checked highest-to-lowest when resolving a key's final value.
const _TIERS_DESC = Object.values(EXPRESSION_LAYER).sort((a, b) => b - a);

class ExpressionController {
    constructor() {
        this._manager = null;
        this._layers = new Map(); // layer -> Map(key -> value)
    }

    /** Called once the VRM's expressionManager exists (see loadAvatar()). */
    attach(expressionManager) {
        this._manager = expressionManager;
    }

    /** Set `key` to `value` on `layer`, then re-resolve and apply that key. */
    setValue(layer, key, value) {
        if (!this._layers.has(layer)) this._layers.set(layer, new Map());
        this._layers.get(layer).set(key, value);
        this._apply(key);
    }

    /** Remove `key` from `layer`, letting a lower layer show through. */
    clearValue(layer, key) {
        this._layers.get(layer)?.delete(key);
        this._apply(key);
    }

    _apply(key) {
        if (!this._manager) return;
        for (const tier of _TIERS_DESC) {
            const map = this._layers.get(tier);
            if (map && map.has(key)) {
                this._manager.setValue(key, map.get(key));
                return;
            }
        }
        // No layer holds this key — resolve to baseline 0 rather than
        // leaving whatever raw value was last written (e.g. a stale
        // open-mouth shape after lip-sync's clearValue()).
        this._manager.setValue(key, 0);
    }
}

// Singleton — one avatar, one expression arbiter.
export const expressionController = new ExpressionController();