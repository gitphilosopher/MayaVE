/**
 * frontend/js/expression-composer.js
 * Expression Composer — turns a communicative-intent packet ({primary,
 * secondary, intensity, attitude, gaze, actions, recipe}) from the backend
 * Behavioral Engine (core/behavior_engine.py) into VRM expression weights.
 *
 * CRITICAL VRM CONSTRAINT: the six-knob legacy path below assumes exactly
 * six emotional blendshapes — neutral, joy, fun, angry, sorrow, surprised.
 * There is no "happy"/"sad"/"relaxed" blendshape; those words only ever
 * appear as semantic tags (core/behavior_engine.py's _VALID_EXPRESSIONS)
 * and get translated into the six knobs here.
 *
 * Semantic recipe path (additive, backward compatible): the backend may
 * also send `intent.recipe` — a dict of fine-grained, verified VRoid
 * `Fcl_BRW_*`/`Fcl_EYE_*`/`Fcl_MTH_*` weights composed from emotion +
 * attitude + intensity (see core/expression_library.py). These are RAW
 * mesh morph targets, not VRMExpressionManager presets/customs — they
 * are written directly via `mesh.morphTargetInfluences[index]` (found
 * through each mesh's `morphTargetDictionary`), the same technique VRoid
 * tooling uses, and bypass expressionController's layer system entirely
 * since that system only mediates VRMExpressionManager-registered keys.
 * A recipe key is only ever written after confirming some mesh actually
 * has that morph target; an unverified or missing key is dropped, never
 * invented. If no verified recipe key is found (older models, or a
 * packet with no recipe), the legacy six-knob path below runs exactly
 * as before.
 *
 * `Fcl_MTH_A/I/U/E/O` (viseme vowel shapes) are never part of a recipe —
 * those belong to lip-sync (see avatar.js's _startLipSync) — so there is
 * no ownership conflict to resolve here.
 */

import { expressionController, EXPRESSION_LAYER } from "./expression-controller.js";
import { applyBehavioralGaze, vrm } from "./avatar.js";

// The ONLY six valid legacy VRM emotional keys. Never emit anything outside this set.
const RENDERABLE = ["neutral", "joy", "fun", "angry", "sorrow", "surprised"];

// Intrinsic character per semantic tag: which knob carries it (primary),
// and which knob colours it with Maya's personality (accent). This is
// the compact base a handful of tags are built from — not a table of
// named expressions like "mock irritation" or "dramatic disbelief";
// those emerge from composing primary + secondary + attitude below.
const BASE = {
    happy:     { primary: "joy",       accent: "fun" },
    excited:   { primary: "joy",       accent: "surprised" },
    surprised: { primary: "surprised", accent: "joy" },
    angry:     { primary: "angry",     accent: null },
    sad:       { primary: "sorrow",    accent: "neutral" },
    relaxed:   { primary: "neutral",   accent: "joy" },
    neutral:   { primary: "neutral",   accent: null },
};

// Client-side mirror of core/behavior_engine.py's PERSONALITY — shapes
// HOW the six knobs combine, not WHICH emotion is intended.
const PERSONALITY = {
    dramaticity:    0.65,
    playfulness:    0.7,
    wit:            0.6,
    expressiveness: 0.75,
    chaos:          0.4,
    subtlety:       0.3,
};

// Base transition duration; per-knob rate multipliers below let some
// components (e.g. surprised) snap in faster than others (e.g. neutral
// settling back) instead of everything moving in lockstep.
const TRANSITION_MS_BASE = 220;
const KEY_RATE = { neutral: 1.3, joy: 1.0, fun: 0.9, angry: 0.75, sorrow: 1.25, surprised: 0.7 };

let _current  = new Map(RENDERABLE.map((k) => [k, 0]));
let _rafToken = 0;

function _clamp01(v) {
    return Math.max(0, Math.min(1, v));
}

/**
 * Composes a communicative-intent packet into six continuous VRM
 * weights. Every step is a function of (tag, intensity, attitude,
 * personality) — no giant static expression table.
 */
function _composeWeights(intent) {
    const primaryTag = BASE[intent.primary] ? intent.primary : "neutral";
    const base = BASE[primaryTag];
    const secondaryTag = intent.secondary && BASE[intent.secondary] && intent.secondary !== primaryTag
        ? intent.secondary
        : null;

    const intensity = _clamp01(intent.intensity ?? 0.6);
    const { dramaticity, playfulness, wit, expressiveness, chaos, subtlety } = PERSONALITY;

    const weights = new Map(RENDERABLE.map((k) => [k, 0]));
    const add = (key, w) => { if (weights.has(key)) weights.set(key, weights.get(key) + w); };

    // Overall amplitude — more expressive personality reads slightly
    // stronger at the same intensity, without forcing every expression
    // to the ceiling.
    const amplitude = 0.75 + expressiveness * 0.35;

    // Primary emotion — nonlinear curve (not intensity * 1.0): higher
    // dramaticity makes Maya commit to a read faster as intensity rises;
    // low-intensity turns stay visibly restrained instead of scaling
    // linearly toward full weight.
    const primaryExponent = 0.75 - dramaticity * 0.25;
    const primaryStrength = Math.pow(intensity, primaryExponent) * amplitude;
    add(base.primary, primaryStrength);

    // Accent — Maya's own flavour of the primary emotion (e.g. joy
    // carries a playful "fun" undertone). Grows much more slowly than
    // intensity so it stays a subtle undertone except at real peaks;
    // playfulness controls how much of it bleeds through at all.
    if (base.accent) {
        const accentStrength = Math.pow(intensity, 1.6) * (0.3 + playfulness * 0.4) * amplitude;
        add(base.accent, accentStrength);
    }

    // Secondary tag from the Behavioral Engine (mood bleed-through, or a
    // personality-biased pairing like angry+happy for mock outrage) rides
    // along at a reduced, personality-shaped weight on ITS OWN primary
    // knob — this is what lets "angry" (genuine) and "angry" (teasing,
    // secondary=happy -> extra joy/fun) look meaningfully different.
    if (secondaryTag) {
        const secondaryRatio = 0.28 + playfulness * 0.22 + dramaticity * 0.1;
        add(BASE[secondaryTag].primary, primaryStrength * secondaryRatio);
    }

    // Attitude — teasing pushes weight into "fun" (mischief), wit-scaled,
    // plus a small "surprised" touch (raised-eyebrow quality), independent
    // of whichever primary tag is active.
    if (intent.attitude === "teasing") {
        add("fun", (0.2 + wit * 0.2) * intensity * amplitude);
        add("surprised", 0.1 * intensity);
    }

    // Neutral is a real, deliberately-used knob — not a fallback. It
    // grounds the face: more prominent when Maya's personality favours
    // subtlety and/or intensity is low, so restrained moments read as
    // genuinely restrained rather than just "everything else scaled down".
    add("neutral", subtlety * (1 - intensity) * 0.6 + 0.04);

    // Controlled variation — small, bounded, and applied everywhere
    // except the primary knob, so the intended emotion never flips turn
    // to turn even though the exact composition does.
    for (const key of RENDERABLE) {
        if (key === base.primary) continue;
        weights.set(key, weights.get(key) + (Math.random() * 2 - 1) * 0.05 * chaos);
    }

    for (const key of RENDERABLE) weights.set(key, _clamp01(weights.get(key)));
    return weights;
}

function _durationFor(key) {
    return TRANSITION_MS_BASE * (KEY_RATE[key] ?? 1.0) * (1 - PERSONALITY.dramaticity * 0.3);
}

function _animateTo(target) {
    const token = ++_rafToken;
    const start = new Map(_current);
    const t0 = performance.now();
    const durations = new Map(RENDERABLE.map((k) => [k, _durationFor(k)]));

    function tick(now) {
        if (token !== _rafToken) return;   // superseded by a newer behavior
        const elapsed = now - t0;
        let stillAnimating = false;

        for (const key of RENDERABLE) {
            const p = Math.min(1, elapsed / durations.get(key));
            if (p < 1) stillAnimating = true;
            const eased = p * p * (3 - 2 * p);   // smoothstep — no robotic snap
            const from  = start.get(key);
            const to    = target.get(key);
            const value = from + (to - from) * eased;
            _current.set(key, value);
            expressionController.setValue(EXPRESSION_LAYER.EMOTION, key, value);
        }

        if (stillAnimating) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// ── Semantic recipe path — raw verified Fcl_* mesh morph targets ───────────

// name -> Array<{mesh, index}> | null (null = confirmed absent on this VRM).
// Populated lazily, one scene traversal per never-before-seen name — not
// re-traversed on every write/frame.
const _morphCache = new Map();

function _resolveMorphTargets(name) {
    if (_morphCache.has(name)) return _morphCache.get(name);
    const targets = [];
    if (vrm?.scene) {
        vrm.scene.traverse((object) => {
            if (object.isMesh && object.morphTargetDictionary) {
                const index = object.morphTargetDictionary[name];
                if (index !== undefined) targets.push({ mesh: object, index });
            }
        });
    }
    const result = targets.length > 0 ? targets : null;
    _morphCache.set(name, result);
    return result;
}

function _setRawMorph(name, value) {
    const targets = _resolveMorphTargets(name);
    if (!targets) return false;
    for (const { mesh, index } of targets) {
        mesh.morphTargetInfluences[index] = value;
    }
    return true;
}

// Tracks whichever recipe keys are currently being driven, so a key no
// longer present in a new recipe eases back to 0 instead of sticking.
let _recipeCurrent  = new Map();
let _recipeRafToken = 0;

function _animateRecipeTo(targetMap) {
    const allKeys = new Set([..._recipeCurrent.keys(), ...targetMap.keys()]);
    if (allKeys.size === 0) return;

    const token = ++_recipeRafToken;
    const start  = new Map(_recipeCurrent);
    const t0     = performance.now();

    function tick(now) {
        if (token !== _recipeRafToken) return;
        const p = Math.min(1, (now - t0) / TRANSITION_MS_BASE);
        const eased = p * p * (3 - 2 * p);
        for (const key of allKeys) {
            const from  = start.get(key) ?? 0;
            const to    = targetMap.get(key) ?? 0;
            const value = from + (to - from) * eased;
            _recipeCurrent.set(key, value);
            _setRawMorph(key, value);
        }
        if (p < 1) {
            requestAnimationFrame(tick);
        } else {
            for (const key of allKeys) {
                if ((targetMap.get(key) ?? 0) === 0) _recipeCurrent.delete(key);
            }
        }
    }
    requestAnimationFrame(tick);
}

function _logDebug(intent, weights) {
    const parts = RENDERABLE
        .map((k) => [k, weights.get(k)])
        .filter(([, v]) => v > 0.02)
        .map(([k, v]) => `${k}=${v.toFixed(2)}`)
        .join(" ");
    console.debug(
        `[Maya][expr] legacy ${intent.primary}${intent.secondary ? "+" + intent.secondary : ""} `
        + `(${intent.attitude}, i=${(intent.intensity ?? 0).toFixed(2)}) -> ${parts || "neutral=0.00"}`
    );
}

/**
 * Applies a composed behavior packet from the backend:
 * {primary, secondary, intensity, attitude, gaze, actions, recipe}.
 * Body actions (nod/giggle/...) are dispatched separately over the
 * existing "animation" WS message/channel — this only handles emotional
 * expression weights + gaze.
 */
export function applyBehavior(intent) {
    if (!intent || !intent.primary) return;

    const rawRecipe = intent.recipe && typeof intent.recipe === "object" ? intent.recipe : null;

    let verifiedRecipe = null;
    if (rawRecipe && vrm) {
        verifiedRecipe = {};
        for (const [key, value] of Object.entries(rawRecipe)) {
            if (_resolveMorphTargets(key)) verifiedRecipe[key] = value;
        }
        if (Object.keys(verifiedRecipe).length === 0) verifiedRecipe = null;
    }

    if (verifiedRecipe) {
        // Fine-grained Fcl_* composition — ease the legacy six-knob path
        // to zero so it doesn't fight the recipe on the same VRM.
        _animateTo(new Map(RENDERABLE.map((k) => [k, 0])));
        _animateRecipeTo(new Map(Object.entries(verifiedRecipe)));
        console.debug(
            `[Maya][expr] semantic=${intent.primary}|${intent.attitude} recipe_keys=`
            + Object.entries(verifiedRecipe).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(" ")
        );
    } else {
        // No verified recipe key available (older/plain VRM, or a packet
        // with no recipe) — legacy six-knob composition, unchanged.
        _animateRecipeTo(new Map());   // ease out any stale recipe keys
        const weights = _composeWeights(intent);
        _logDebug(intent, weights);
        _animateTo(weights);
    }

    if (intent.gaze) applyBehavioralGaze(intent.gaze);
}