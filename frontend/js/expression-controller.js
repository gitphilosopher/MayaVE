/**
 * frontend/js/expression-controller.js
 * Centralized per-key arbiter for VRMExpressionManager values. Writers store
 * values in priority layers; the highest layer holding a key is applied to
 * the attached manager.
 *
 * avatar.js attaches the manager when the VRM loads and routes its expression
 * writes here; expression-composer.js uses the EMOTION layer. EXPRESSION_LAYER
 * defines the shared ordering for resting values, emotion, actions, lip-sync,
 * and blinking. Clearing a key reveals the next layer; when none holds it,
 * the manager is explicitly set to zero to avoid stale values.
 *
 * This layers registered manager keys behind avatar.js's setExpression API.
 * Direct mesh morph-target recipe writes bypass this controller.
 */

// Higher number = higher priority.
export const EXPRESSION_LAYER = Object.freeze({
    BASE:    0,   // initial/default resting look
    EMOTION: 1,   // mood + skill-response tags via setExpression()
    ACTION:  2,   // short temporary overlays (e.g. wink's blinkLeft)
    LIPSYNC: 3,   // mouth shapes while speaking
    BLINK:   4,   // eyelid open/close
});

// Resolve strongest-to-weakest so the first layer holding a key wins.
const _TIERS_DESC = Object.values(EXPRESSION_LAYER).sort((a, b) => b - a);

/** Resolves per-key expression layers into the attached VRM manager. */
class ExpressionController {
    constructor() {
        this._manager = null;
        this._layers = new Map(); // layer -> Map(key -> value)
        // this._warned = new Set(); // unknown manager keys, warned once
        this._checked = new Set(); // keys already verified against the manager (warn at most once)
    }

    /** Called once the VRM's expressionManager exists (see loadAvatar()). */
    attach(expressionManager) {
        this._manager = expressionManager;
        this._checked.clear();     // a new model may expose different expressions
    }
    // setValue / clearValue unchanged

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
        if (!this._checked.has(key)) {
            this._checked.add(key);
            if (typeof this._manager.getExpression === "function" && !this._manager.getExpression(key)) {
                console.warn(`[Maya][expr] expression '${key}' does not exist on this VRM — writes are no-ops.`);
            }
        }
        for (const tier of _TIERS_DESC) {
            const map = this._layers.get(tier);
            if (map && map.has(key)) {
                this._manager.setValue(key, map.get(key));
                return;
            }
        }
        this._manager.setValue(key, 0);
    }
}

export const expressionController = new ExpressionController();