/**
 * frontend/js/avatar.js
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { VRMLoaderPlugin } from "@pixiv/three-vrm";

import {
    VRMAnimationLoaderPlugin,
    createVRMAnimationClip
} from "@pixiv/three-vrm-animation";

import { animationController, PRIORITY } from "./animation-controller.js";
import { gazeController, ATTENTION_STATE } from "./gaze-controller.js";
import { lifeMotionController } from "./life-motion-controller.js";
import { expressionController, EXPRESSION_LAYER } from "./expression-controller.js";

export let vrm;
export let isSpeaking = false;

// ── Post-launch calm period (Phase 9) ──────────────────────────────────────
// Maya intentionally does not start automatic idle/fidget behavior for a
// short window after avatar init, so she doesn't appear to fidget the
// instant she's summoned. Timestamp is captured at module load (this file
// is evaluated once, synchronously, when main.js starts) — not on wake or
// first message. Only gates the scheduler's own initiation in
// _canFidgetNow(); every other animation path (actions, window.maya.*,
// gaze, life motion, speaking/lip-sync, blink/wink, sleep/wake) is untouched.
const _CALM_PERIOD_MS = 5 * 60_000;
const _avatarInitAt   = performance.now();
function _isCalmPeriodActive() {
    return performance.now() - _avatarInitAt < _CALM_PERIOD_MS;
}

// Tracks the in-flight audio source so stopCurrentAudio() can halt a
// barge-in immediately — see Handoff §3.12 for the full backend chain.
let _currentSource  = null;   // AudioBufferSourceNode from speakFromBytes
let _audioGen = 0;   // bumped by stopCurrentAudio(); drops audio whose decode finished after a stop
let _currentAudioEl = null;   // HTMLAudioElement from speak()

// Callback set by websocket.js to send audio_done signal to backend
let _onAudioDone = null;
export function setAudioDoneCallback(fn) { _onAudioDone = fn; }

// Called whenever new audio playback begins. The idle-fidget scheduler
// (see further down) runs on its own free-running 7–18s timer that isn't
// otherwise synced to speech — without this, a tick already due can land
// right as audio ends and fire a fidget immediately instead of waiting a
// natural pause. Only restarts the countdown — must NOT touch active
// animations, since e.g. the greet wave is dispatched right as speech
// starts and would otherwise get cut off before it plays.
function _onSpeakingStart() {
    _scheduleNextFidget();
    gazeController.setSpeaking(true);
}

// Called whenever audio playback ends — restarts the countdown again so
// the next fidget is a full 7–18s away from the moment speech actually
// stopped, not from whenever it happened to last tick.
function _onSpeakingEnd() {
    _scheduleNextFidget();
    gazeController.setSpeaking(false);
}

// Single AudioContext reused for the lifetime of the page
let _audioCtx = null;
function getAudioContext() {
    if (!_audioCtx || _audioCtx.state === "closed") {
        _audioCtx = new AudioContext();
    }
    return _audioCtx;
}

// ── Behavioral gaze hook ────────────────────────────────────────────────────
// Execution-only: WHICH mode and WHEN is decided by the Behavioral Engine
// (core/behavior_engine.py) and forwarded via expression-composer.js —
// this just applies the resulting eye offset, at lower priority than
// speaking and screen-attention gaze (see startEyeMovement() below).
let _behGazeX = 0, _behGazeY = 0, _behGazeUntil = 0;

export function applyBehavioralGaze(mode) {
    if (!_awake) return;
    const now = performance.now();
    switch (mode) {
        case "away":
            _behGazeX = (Math.random() < 0.5 ? -1 : 1) * 0.22;
            _behGazeY = 0.06;
            _behGazeUntil = now + 1400;
            break;
        case "soft":
            _behGazeX = 0;
            _behGazeY = 0.03;
            _behGazeUntil = now + 900;
            break;
        default: // "direct" — no override, normal gaze/idle-drift resolves
            _behGazeUntil = 0;
            break;
    }
}

// ── Sleep / wake state ────────────────────────────────────────────────────────

let _awake          = false;
let _blinkingActive = false;  // true while the blink loop is scheduled
let _blinkTimeout   = null;   // pending blink tick — cleared on re-wake to avoid duplicate loops
let _eyeLoopStarted = false;  // eye-movement loop is started once per page, not per wake

/**
 * Called by websocket.js when WS connects.
 * Opens eyes and starts normal idle animations.
 */
export function wakeAvatar() {
    if (_awake) return;
    _awake = true;
    gazeController.setAwake(true);

    startIdleFidgets();

    if (!vrm?.expressionManager) return;

    expressionController.setValue(EXPRESSION_LAYER.BLINK, "blink", 0);
    startBlinking();
    startEyeMovement();
}

/**
 * Called by websocket.js when WS disconnects.
 * Closes eyes and stops the blink loop.
 */
export function sleepAvatar() {
    if (!_awake) return;
    _awake          = false;
    _blinkingActive = false;   // blink loop checks this and exits on next tick
    gazeController.setAwake(false);

    stopIdleFidgets();

    if (!vrm?.expressionManager) return;

    // Close eyes and reset to neutral resting state. Written directly on
    // their layers (not via setExpression()) since _awake is already false
    // here and setExpression() would no-op.
    expressionController.setValue(EXPRESSION_LAYER.BLINK,   "blink",     1);
    expressionController.setValue(EXPRESSION_LAYER.EMOTION, "relaxed",   0);
    expressionController.setValue(EXPRESSION_LAYER.EMOTION, "happy",     0);
    expressionController.setValue(EXPRESSION_LAYER.EMOTION, "surprised", 0);
    expressionController.setValue(EXPRESSION_LAYER.EMOTION, "neutral",   0);
}

// ── Lip-sync ──────────────────────────────────────────────────────────────────
// Driven purely by the AnalyserNode attached to whichever audio is actually
// playing (see speakFromBytes/speak below) — not by any LLM/TTS response
// event. Mouth shapes + the speaking "happy" boost go through the LIPSYNC
// tier of expressionController so they layer over (and never permanently
// clobber) mood/skill-response expressions written on the EMOTION tier —
// e.g. an active [happy] mood tag is no longer stomped to a flat 0.08 by
// the per-frame mouth loop while Maya is speaking that same line.
//
// _lipSyncToken guards against overlapping loops: each _startLipSync() call
// gets its own token, and a loop whose token no longer matches the current
// one (because a newer start or a stop bumped it) exits on its very next
// frame instead of continuing to run alongside the newer loop.

let _lipSyncToken  = 0;
let _lipSyncActive = false;

function _startLipSync(analyser) {
    const token = ++_lipSyncToken;
    _lipSyncActive = true;

    const data    = new Uint8Array(analyser.frequencyBinCount);
    const bands   = Math.floor(data.length / 5);
    const bandAvg = (i) =>
        data.slice(i * bands, (i + 1) * bands)
            .reduce((a, b) => a + b, 0) / bands / 128;

    let smoothed = 0;

    function animateMouth() {
        if (token !== _lipSyncToken || !vrm?.expressionManager) return;

        analyser.getByteFrequencyData(data);

        const volume = data.reduce((a, b) => a + b, 0) / data.length;

        const target = Math.min(volume / 28, 1);
        smoothed += target > smoothed
            ? (target - smoothed) * 0.4
            : (target - smoothed) * 0.15;

        expressionController.setValue(EXPRESSION_LAYER.LIPSYNC, "aa", smoothed * Math.max(bandAvg(0), 0.15));
        expressionController.setValue(EXPRESSION_LAYER.LIPSYNC, "ee", smoothed * bandAvg(1) * 0.8);
        expressionController.setValue(EXPRESSION_LAYER.LIPSYNC, "ih", smoothed * bandAvg(2) * 0.6);
        expressionController.setValue(EXPRESSION_LAYER.LIPSYNC, "oh", smoothed * bandAvg(3) * 0.7);
        expressionController.setValue(EXPRESSION_LAYER.LIPSYNC, "ou", smoothed * bandAvg(4) * 0.5);
        expressionController.setValue(EXPRESSION_LAYER.LIPSYNC, "happy", 0.08);

        requestAnimationFrame(animateMouth);
    }

    requestAnimationFrame(animateMouth);
}

function _stopLipSync() {
    _lipSyncToken++;   // invalidates any in-flight animateMouth loop immediately
    _lipSyncActive = false;
    isSpeaking     = false;
    _onSpeakingEnd();
    if (!vrm?.expressionManager) return;
    ["aa", "ee", "ih", "ou", "oh", "happy"].forEach(k =>
        expressionController.clearValue(EXPRESSION_LAYER.LIPSYNC, k)
    );
    // Resting look on the BASE tier — never fights an active EMOTION-tier
    // mood value, but shows through the instant lip-sync releases its keys.
    expressionController.setValue(EXPRESSION_LAYER.BASE, "relaxed", 0.4);
}

// ── speakFromBytes — called by websocket.js ───────────────────────────────────

export async function speakFromBytes(arrayBuffer) {
    if (!vrm) {
        if (typeof _onAudioDone === "function") _onAudioDone();   // don't leave the backend waiting 30 s
        return;
    }

    const ctx = getAudioContext();
    const gen = _audioGen;

    let decoded;
    try {
        decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
    } catch (e) {
        console.error("[Maya] decodeAudioData failed:", e);
        if (gen === _audioGen && typeof _onAudioDone === "function") _onAudioDone();
        return;
    }
    if (gen !== _audioGen) return;   // stop_audio arrived during decode

    const source   = ctx.createBufferSource();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;

    source.buffer = decoded;
    source.connect(analyser);
    analyser.connect(ctx.destination);

    isSpeaking      = true;
    _onSpeakingStart();
    _currentSource  = source;
    source.onended = () => {
        if (_currentSource === source) _currentSource = null;
        _stopLipSync();
        if (typeof _onAudioDone === "function") _onAudioDone();
    };

    source.start(0);
    _startLipSync(analyser);
}

// ── speak — local file (testing only) ────────────────────────────────────────

export function speak(audioFile) {
    if (!vrm) return;

    const audio    = new Audio(audioFile);
    const ctx      = getAudioContext();
    const source   = ctx.createMediaElementSource(audio);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;

    source.connect(analyser);
    analyser.connect(ctx.destination);

    isSpeaking      = true;
    _onSpeakingStart();
    _currentAudioEl = audio;
    audio.onended   = () => {
        if (_currentAudioEl === audio) _currentAudioEl = null;
        _stopLipSync();
    };

    audio.play();
    _startLipSync(analyser);
}

// ── Barge-in: stop whatever is currently playing ─────────────────────────────

/** Halts whichever audio path is active + stops lip-sync. Safe no-op if idle. */
export function stopCurrentAudio() {
    _audioGen++;
    if (_currentSource) {
        const s = _currentSource;
        _currentSource = null;
        s.onended = null;   // avoid a duplicate audio_done from the native 'ended' event
        try { s.stop(0); } catch (e) { /* already stopped/ended */ }
    }
    if (_currentAudioEl) {
        const a = _currentAudioEl;
        _currentAudioEl = null;
        a.onended = null;
        try { a.pause(); a.currentTime = 0; } catch (e) { /* no-op */ }
    }
    _stopLipSync();
}

// ── Expression helper ─────────────────────────────────────────────────────────

export function setExpression(name, value) {
    if (!vrm?.expressionManager) return;
    // Don't apply state expressions while sleeping — blink must stay at 1
    if (!_awake) return;
    expressionController.setValue(EXPRESSION_LAYER.EMOTION, name, value);
}

// ── VRMA-driven animation system — see Handoff §3.11 / §3.12 for the full
// design rationale (track filtering, fade envelopes, shoulder protection,
// expression-track caveat) ─────────────────────────────────────────────────

// Full .vrma inventory — not all are wired to anything yet, see §3.12.
const _VRMA_ASSETS = {
    wave:         "assets/vrmas/wave.vrma",
    nod:          "assets/vrmas/hard_nod.vrma",
    giggle:       "assets/vrmas/excited.vrma",
    sigh:         "assets/vrmas/sigh.vrma",
    shrug:        "assets/vrmas/shrug.vrma",
    weightShift:  "assets/vrmas/weight_shift.vrma",
    lookAround:   "assets/vrmas/idle_lalala.vrma",
    stretch:      "assets/vrmas/yawn.vrma",

    // Registered, not yet wired to anything.
    idle:         "assets/vrmas/idle.vrma",
    angry:        "assets/vrmas/angry.vrma",
    bashful:      "assets/vrmas/bashful.vrma",
    catwalk:      "assets/vrmas/catwalk.vrma",
    clapping:     "assets/vrmas/clapping.vrma",
    crying:       "assets/vrmas/crying.vrma",
    cuteThink:    "assets/vrmas/cute_think.vrma",
    greeting:     "assets/vrmas/greeting.vrma",
    handRaise:    "assets/vrmas/hand_raise.vrma",
    headShake:    "assets/vrmas/head_shake.vrma",
    macarenaDance:"assets/vrmas/macarena_dance.vrma",
    rejected:     "assets/vrmas/rejected.vrma",
    thankful:     "assets/vrmas/thankful.vrma",
    waving:       "assets/vrmas/waving.vrma",
};

// Per-animation fade timing (ms). Falls back to _DEFAULT_FADE below for
// any name not listed here. Slow, settling fidgets (weightShift, stretch)
// fade a bit slower than the default.
const _FADE_OVERRIDES = {
    weightShift:  { fadeInMs: 400, fadeOutMs: 400 },
    stretch:      { fadeInMs: 400, fadeOutMs: 400 },
};
const _DEFAULT_FADE = { fadeInMs: 350, fadeOutMs: 1500 };

// Bones no .vrma may ever rotate — preserves custom shoulder pose (§3.11).
const _PROTECTED_BONE_KEYS = ["leftShoulder", "rightShoulder"];

function _protectedBoneNames() {
    return new Set(
        _PROTECTED_BONE_KEYS
            .map((key) => vrm.humanoid.getNormalizedBoneNode(key))
            .filter(Boolean)
            .map((bone) => bone.name)
    );
}

// url -> Promise<VRMAnimation>, so concurrent/repeat plays of the same
// animation only ever load and parse the file once.
const _vrmaCache = new Map();

async function _loadVrmaAnimation(name, url) {
    if (_vrmaCache.has(url)) return _vrmaCache.get(url);

    const promise = (async () => {
        const loader = new GLTFLoader();
        loader.register((parser) => new VRMAnimationLoaderPlugin(parser));

        const gltf = await loader.loadAsync(url);
        const animations = gltf.userData.vrmAnimations;

        if (!animations || animations.length === 0) {
            throw new Error(`No VRMA animation found in ${url}`);
        }

        return animations[0];
    })();

    _vrmaCache.set(url, promise);
    return promise;
}

/** Retargets + filters a VRMAnimation to quaternion/weight tracks only (§3.11). */
function _buildFilteredClip(vrma, name) {
    const rawClip = createVRMAnimationClip(vrma, vrm);
    const excludedBones = _protectedBoneNames();

    const clip = rawClip.clone();
    clip.tracks = clip.tracks.filter((track) => {
        if (track.name.endsWith(".quaternion")) {
            const boneName = track.name.slice(0, track.name.lastIndexOf("."));
            return !excludedBones.has(boneName);
        }
        if (track.name.endsWith(".weight")) {
            return true; // expression tracks, e.g. blink for the wink
        }
        return false; // drop position/scale and anything else unrecognised
    });

    // Empty after filtering usually means root-motion, not bone rotation (§3.11).
    if (clip.tracks.length === 0) {
        console.warn(
            `[Maya] '${name}' has NO playable tracks after filtering — its ` +
            `source motion is likely on a position/scale track (e.g. hips root ` +
            `motion) that this system intentionally strips. Re-export it as a ` +
            `pure bone-rotation animation, or bake root motion into rotations.`
        );
    }
    return clip;
}

// name -> { mixer, action } for every animation currently mid-playback.
// Multiple different named animations CAN run concurrently (e.g. a fidget
// plus a wink); a single name cannot re-enter itself until it finishes
// (see the `_activeNames` guard in playVrmaAnimation).
const _activeMixers = new Map();
const _activeNames  = new Set();

// The fidget-pool entries that are VRMA-backed (drive a real mixer on the
// skeleton) rather than procedural (headTilt/shoulderRoll, which don't use
// this mixer system). Kept as its own literal here — not derived from
// _FIDGET_POOL further down the file — so it's usable from playVrmaAnimation
// without depending on declaration order.
const _FIDGET_VRMA_NAMES = new Set([
    "weightShift", "lookAround", "stretch", "idle", "cuteThink", "waving",
]);

/** Ticks every active VRMA mixer — call every frame from main.js's animate loop. */
export function updateVrmaAnimations(delta) {
    for (const { mixer } of _activeMixers.values()) {
        mixer.update(delta);
    }
}

/** Ticks the gaze/attention system — call every frame from main.js's animate loop. */
export function updateGaze(delta) {
    gazeController.update(delta);
}

/** Generic VRMA player: load (cached) → filter → fade in → play once → fade out (§3.11).
 *  Claims the clip's bone tracks in animation-controller.js (FIDGET tier for
 *  the idle-fidget pool, ACTION tier otherwise) so base idle motion (head
 *  sway, gaze) yields those bones for the duration of playback. */
async function playVrmaAnimation(name) {
    if (!vrm || _activeNames.has(name)) return;

    // A deliberately-triggered animation (wave/nod/giggle/sigh/shrug, or a
    // manually-tested one) must not run at the same time as a randomised
    // idle fidget. Two independent THREE.AnimationMixer instances driving
    // the same bone fight every frame, and when the newer one later stops
    // and restores, it restores to whatever it captured as "original" at
    // its own start — the fidget's mid-motion pose, not the fidget's real
    // rest pose — leaving the fidget's bone frozen there even after it's
    // long gone. Cleanly stop any active fidget first (same stop + forced
    // mixer.update(0) restore used for a normal fidget's own end-of-clip
    // cleanup, see below) so only one bone-driving mixer is ever live.
    if (!_FIDGET_VRMA_NAMES.has(name)) {
        for (const fidgetName of _FIDGET_VRMA_NAMES) {
            const entry = _activeMixers.get(fidgetName);
            if (entry) {
                entry.action.stop();
                entry.mixer.update(0);
                _activeMixers.delete(fidgetName);
                _activeNames.delete(fidgetName);
                animationController.release(fidgetName);
            }
        }
        // headTilt drives the neck too — stop it and restore its bone now.
        _headTiltStop?.();
    }

    const url = _VRMA_ASSETS[name];
    if (!url) {
        console.error(`[Maya] No .vrma asset registered for animation '${name}'`);
        return;
    }

    _activeNames.add(name);

    try {
        const vrma = await _loadVrmaAnimation(name, url);

        // The load above is async — on an uncached .vrma (first play this
        // session) it can take long enough for speech to start in the
        // meantime. The scheduler only checks _canFidgetNow() BEFORE this
        // await; without a re-check here, a fidget requested while idle
        // would still start playing the moment it finishes loading, even
        // over/right after speech that began in between. Actions are
        // exempt — they must always play regardless of timing.
        if (_FIDGET_VRMA_NAMES.has(name) && isSpeaking) {
            _activeNames.delete(name);
            return;
        }

        const clip = _buildFilteredClip(vrma, name);

        const boneNames = clip.tracks
            .filter((t) => t.name.endsWith(".quaternion"))
            .map((t) => t.name.slice(0, t.name.lastIndexOf(".")));
        const tier = _FIDGET_VRMA_NAMES.has(name) ? PRIORITY.FIDGET : PRIORITY.ACTION;
        animationController.claim(boneNames, tier, name);

        const mixer  = new THREE.AnimationMixer(vrm.scene);
        const action = mixer.clipAction(clip);

        action.reset();
        action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = true;   // we drive our own fade-out below

        const { fadeInMs, fadeOutMs } = _FADE_OVERRIDES[name] ?? _DEFAULT_FADE;
        const clipDurationMs = clip.duration * 1000;

        action.fadeIn(fadeInMs / 1000);
        action.play();

        _activeMixers.set(name, { mixer, action });

        const fadeOutDelay = Math.max(0, clipDurationMs - fadeOutMs);
        setTimeout(() => {
            action.fadeOut(fadeOutMs / 1000);
        }, fadeOutDelay);

        setTimeout(() => {
            // Stale timer: this run was halted and a newer run of the same
            // name now owns the entry/bones — don't touch them.
            const current = _activeMixers.get(name);
            if (current && current.mixer !== mixer) return;
            action.stop();
            mixer.update(0);   // forces THREE's pose restore before we stop ticking it (§3.11)
            _activeMixers.delete(name);
            _activeNames.delete(name);
            animationController.release(name);
        }, clipDurationMs + 50); // small buffer past the clip's natural end

    } catch (error) {
        console.error(`[Maya] Failed to load/play '${name}' VRMA (${url}):`, error);
        _activeMixers.delete(name);
        _activeNames.delete(name);
        animationController.release(name);
    }
}

// ── Public animation entry points — same names/signatures as before (§3.11) ─

export function playWaveAnimation()         { return playVrmaAnimation("wave"); }
export function playNodAnimation()          { return playVrmaAnimation("nod"); }
export function playGiggleAnimation()       { return playVrmaAnimation("giggle"); }
export function playSighAnimation()         { return playVrmaAnimation("sigh"); }
export function playShrugAnimation()        { return playVrmaAnimation("shrug"); }
export function playWeightShiftAnimation()  { return playVrmaAnimation("weightShift"); }
export function playLookAroundAnimation()   { return playVrmaAnimation("lookAround"); }
export function playStretchAnimation()      { return playVrmaAnimation("stretch"); }

// Registered but not yet wired up — manual testing only, see §3.12.
export function playIdleAnimation()          { return playVrmaAnimation("idle"); }
export function playAngryAnimation()         { return playVrmaAnimation("angry"); }
export function playBashfulAnimation()       { return playVrmaAnimation("bashful"); }
export function playCatwalkAnimation()       { return playVrmaAnimation("catwalk"); }
export function playClappingAnimation()      { return playVrmaAnimation("clapping"); }
export function playCryingAnimation()        { return playVrmaAnimation("crying"); }
export function playCuteThinkAnimation()     { return playVrmaAnimation("cuteThink"); }
export function playGreetingAnimation()      { return playVrmaAnimation("greeting"); }
export function playHandRaiseAnimation()     { return playVrmaAnimation("handRaise"); }
export function playHeadShakeAnimation()     { return playVrmaAnimation("headShake"); }
export function playMacarenaDanceAnimation() { return playVrmaAnimation("macarenaDance"); }
export function playRejectedAnimation()      { return playVrmaAnimation("rejected"); }
export function playThankfulAnimation()      { return playVrmaAnimation("thankful"); }
export function playWavingAnimation()        { return playVrmaAnimation("waving"); }

// ── Manual (procedural) animations — deliberately not VRMA-backed (§3.11) ──

let _winkActive = false;

export function playWinkAnimation() {
    if (!vrm?.expressionManager || _winkActive) return;
    _winkActive = true;

    expressionController.setValue(EXPRESSION_LAYER.ACTION, "blinkLeft", 1);
    setTimeout(() => {
        if (vrm?.expressionManager) expressionController.clearValue(EXPRESSION_LAYER.ACTION, "blinkLeft");
        _winkActive = false;
    }, 280);
}

let _headTiltActive = false;
let _headTiltStop   = null;   // cancels the running tween and restores the neck

export function playHeadTiltAnimation() {
    if (!vrm || _headTiltActive) return;
    const neck = vrm.humanoid.getNormalizedBoneNode("neck");
    if (!neck) return;

    _headTiltActive = true;
    animationController.claim([neck.name], PRIORITY.FIDGET, "headTilt");

    const origZ = neck.rotation.z;
    const direction   = Math.random() < 0.5 ? 1 : -1;
    const TILT_ANGLE  = 0.16 * direction;

    const TILT_MS = 400, HOLD_MS = 900, RETURN_MS = 450;
    const TOTAL_MS = TILT_MS + HOLD_MS + RETURN_MS;
    const start = performance.now();
    function smoothstep(t) { return t * t * (3 - 2 * t); }

    const stop = () => {
        if (_headTiltStop !== stop) return;
        _headTiltStop = null;
        neck.rotation.z = origZ;
        _headTiltActive = false;
        animationController.release("headTilt");
    };
    _headTiltStop = stop;

    function tick(now) {
        if (_headTiltStop !== stop) return;   // halted by an action
        const elapsed = now - start;
        let p;
        if (elapsed < TILT_MS) {
            p = smoothstep(elapsed / TILT_MS);
        } else if (elapsed < TILT_MS + HOLD_MS) {
            p = 1;
        } else if (elapsed < TOTAL_MS) {
            p = 1 - smoothstep((elapsed - TILT_MS - HOLD_MS) / RETURN_MS);
        } else {
            stop();
            return;
        }
        neck.rotation.z = origZ + TILT_ANGLE * p;
        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

let _shoulderRollActive = false;

export function playShoulderRollAnimation() {
    if (!vrm || _shoulderRollActive) return;
    const shoulder = Math.random() < 0.5
        ? vrm.humanoid.getNormalizedBoneNode("leftShoulder")
        : vrm.humanoid.getNormalizedBoneNode("rightShoulder");
    if (!shoulder) return;

    _shoulderRollActive = true;
    animationController.claim([shoulder.name], PRIORITY.FIDGET, "shoulderRoll");

    const origX = shoulder.rotation.x;
    const DURATION_MS = 1300;
    const start = performance.now();

    function tick(now) {
        const elapsed = now - start;
        if (elapsed < DURATION_MS) {
            const t    = elapsed / DURATION_MS;
            const roll = Math.sin(t * Math.PI * 2) * 0.06 * (1 - t);
            shoulder.rotation.x = origX + roll;
            requestAnimationFrame(tick);
        } else {
            shoulder.rotation.x = origX;
            _shoulderRollActive = false;
            animationController.release("shoulderRoll");
        }
    }
    requestAnimationFrame(tick);
}

// ── Idle fidgets — gated on backend state + isSpeaking (§3.12), modulated
// by screen-attention/boredom (Phase 5, see _gazeFidgetFactor below) ─────

let _currentBackendState = null;   // mirrors "idle" | "listening" | "processing" | "speaking"
let _idleFidgetTimeout   = null;

// Wall-clock timestamp of when the backend last became continuously idle,
// or null when it isn't. Drives the 30-minute "grab attention" wave below.
let _idleSince = null;

/** Called by websocket.js on every "state" message (drives _canFidgetNow). */
export function setAvatarState(stateValue) {
    _currentBackendState = stateValue;
    if (stateValue === "idle") {
        if (_idleSince === null) _idleSince = Date.now();
    } else {
        _idleSince = null;
    }
}

function _canFidgetNow() {
    return _awake && !isSpeaking && _currentBackendState === "idle"
        && !_isAnyFidgetPlaying() && !_isActionAnimationPlaying()
        && !_isCalmPeriodActive();
}

// True while a deliberately-triggered animation (wave/nod/giggle/sigh/shrug,
// or any manually-tested one) is mid-playback — i.e. any currently-active
// name that isn't in the fidget pool. Idle fidgets must never interrupt one
// of these, only the reverse (see the halt loop in playVrmaAnimation).
function _isActionAnimationPlaying() {
    for (const name of _activeNames) {
        if (!_FIDGET_VRMA_NAMES.has(name)) return true;
    }
    return false;
}

// Pool of fidgets the scheduler picks from at random. Order doesn't matter.
const _FIDGET_POOL = [
    { name: "weightShift",  fn: playWeightShiftAnimation },
    { name: "lookAround",   fn: playLookAroundAnimation },
    { name: "headTilt",     fn: playHeadTiltAnimation },
    { name: "shoulderRoll", fn: playShoulderRollAnimation },
    { name: "stretch",      fn: playStretchAnimation },
    { name: "idle",         fn: playIdleAnimation },
    { name: "cuteThink",    fn: playCuteThinkAnimation },
    { name: "waving",       fn: playWavingAnimation },
];

// True while any fidget (VRMA-backed or manual/procedural) is still mid-
// playback. Without this, two fidgets sharing a bone (e.g. lookAround and
// headTilt both drive the neck) could overlap: a manual tween like headTilt
// captures its "rest" rotation at start, and if that capture happens while
// a VRMA fidget is still mid-motion, headTilt writes that mid-motion pose
// back as the neck's resting rotation when it finishes — visibly "stuck".
function _isAnyFidgetPlaying() {
    return _FIDGET_POOL.some(({ name }) => _activeNames.has(name))
        || _headTiltActive || _shoulderRollActive;
}

// Longest any real fidget should ever take. If _isAnyFidgetPlaying() is
// still true after this long, one of its flags failed to reset (e.g. an
// rAF-driven tween like headTilt/shoulderRoll stalled because this window
// never holds focus and got background-throttled) — force-clear rather
// than let it block every future fidget forever.
const _FIDGET_STUCK_TIMEOUT_MS = 15_000;
let _fidgetStartedAt = 0;

function _clearStuckFidgetState() {
    console.warn("[Maya] Idle fidget appears stuck — force-clearing state.");
    _headTiltActive     = false;
    _shoulderRollActive = false;
    for (const { name } of _FIDGET_POOL) {
        _activeMixers.delete(name);
        _activeNames.delete(name);
        animationController.release(name);
    }
    animationController.release("headTilt");
    animationController.release("shoulderRoll");
}

// Extra per-animation min spacing on top of the 7–18s pool interval (§3.12).
// Cooldowns ratchet up each time that fidget plays (base -> +step per play,
// capped at max) so a repeated gesture gradually becomes rarer instead of
// firing at a fixed interval forever.
const _FIDGET_COOLDOWN_CONFIG = {
    lookAround: { base: 2 * 60_000,  step: 30_000,       max: 5 * 60_000 },
    idle:       { base: 2 * 60_000,  step: 30_000,       max: 5 * 60_000 },
    cuteThink:  { base: 2 * 60_000,  step: 30_000,       max: 5 * 60_000 },
    stretch:    { base: 15 * 60_000, step: 5 * 60_000,   max: 30 * 60_000 },
};

// name -> current cooldown (ms), seeded from each config's base and bumped
// by _bumpFidgetCooldown() after every play of that name.
const _fidgetCooldownMs = new Map(
    Object.entries(_FIDGET_COOLDOWN_CONFIG).map(([name, cfg]) => [name, cfg.base])
);

function _bumpFidgetCooldown(name) {
    const cfg = _FIDGET_COOLDOWN_CONFIG[name];
    if (!cfg) return;
    const current = _fidgetCooldownMs.get(name) ?? cfg.base;
    _fidgetCooldownMs.set(name, Math.min(cfg.max, current + cfg.step));
}

// name -> last-played timestamp (ms), checked against real elapsed time.
const _fidgetLastPlayedAt = new Map();

// True once any fidget has played this wake session — stretch is excluded
// from the very first pick (see _scheduleNextFidget) so Maya doesn't open
// with a stretch the moment she becomes idle.
let _hasPlayedFirstFidget = false;

// If the backend stays continuously idle this long, force a "waving"
// animation to try to grab the user's attention back, then restart the
// countdown so it can fire again every 30 minutes of continued idling.
const _ATTENTION_IDLE_MS = 30 * 60_000;

// Hard floor on how long the CURRENT idle streak (_idleSince) must have run
// before a fidget is even eligible for its first appearance that streak —
// separate from _fidgetCooldownMs, which only spaces out repeats after a
// fidget has already played once. stretch stays off the table for the
// first 10 minutes of idle; waving stays off the random pool entirely
// until the same 30-minute mark that triggers its forced play below, so
// it never shows up early via the normal random pick.
const _FIDGET_MIN_IDLE_MS = {
    stretch: 10 * 60_000,
    waving:  _ATTENTION_IDLE_MS,
};

// ── Phase 5 — screen-attention-aware fidget modulation ──────────────────
// Reuses gazeController's existing attention/boredom signal (no second
// timer). Full attention (new/changing screen activity) suppresses fidgets
// almost entirely; as that attention decays toward boredom the factor rises
// back to 1, restoring normal probability. Outside OBSERVING (plain idle,
// speaking, sleeping) this is always 1 — unchanged pre-Phase-5 behaviour.
function _gazeFidgetFactor() {
    if (gazeController.getAttentionState() !== ATTENTION_STATE.OBSERVING) return 1;
    return Math.max(0, Math.min(1, 1 - gazeController.getAttentionLevel()));
}

// Fidgets subtle enough to allow while boredom is still building during
// OBSERVING — larger/attention-grabbing gestures (stretch, waving,
// lookAround, cuteThink) stay reserved for genuine idle (factor === 1).
const _SUBTLE_FIDGET_NAMES = new Set(["weightShift", "headTilt", "shoulderRoll"]);

function _scheduleNextFidget() {
    clearTimeout(_idleFidgetTimeout);
    const delay = 7000 + Math.random() * 11000;   // 7–18s between fidgets
    _idleFidgetTimeout = setTimeout(() => {
        console.log(
            `[Maya][fidget] tick — awake=${_awake} isSpeaking=${isSpeaking} ` +
            `backendState=${_currentBackendState} isAnyFidgetPlaying=${_isAnyFidgetPlaying()} ` +
            `activeNames=[${[..._activeNames].join(",")}] headTiltActive=${_headTiltActive} ` +
            `shoulderRollActive=${_shoulderRollActive} idleForMs=${_idleSince !== null ? Date.now() - _idleSince : "n/a"} ` +
            `calmPeriodActive=${_isCalmPeriodActive()}`
        );

        if (_isAnyFidgetPlaying() && Date.now() - _fidgetStartedAt > _FIDGET_STUCK_TIMEOUT_MS) {
            _clearStuckFidgetState();
        }

        if (_canFidgetNow()) {
            const now = Date.now();

            if (_idleSince !== null && now - _idleSince >= _ATTENTION_IDLE_MS) {
                console.log("[Maya][fidget] forcing waving (30min idle)");
                _fidgetLastPlayedAt.set("waving", now);
                _fidgetStartedAt = now;
                _hasPlayedFirstFidget = true;
                playWavingAnimation();
                _idleSince = now;   // restart the 30-minute countdown
            } else {
                const gazeFactor = _gazeFidgetFactor();

                if (Math.random() >= gazeFactor) {
                    console.log(
                        `[Maya][fidget] suppressed by screen attention ` +
                        `(state=${gazeController.getAttentionState()} factor=${gazeFactor.toFixed(2)})`
                    );
                } else {
                    const idleFor = _idleSince !== null ? now - _idleSince : 0;
                    let eligible = _FIDGET_POOL.filter(({ name }) => {
                        if (!_hasPlayedFirstFidget && name === "stretch") return false;
                        const minIdle = _FIDGET_MIN_IDLE_MS[name] ?? 0;
                        if (idleFor < minIdle) return false;
                        const cooldown   = _fidgetCooldownMs.get(name) ?? 0;
                        const lastPlayed = _fidgetLastPlayedAt.get(name) ?? -Infinity;
                        return now - lastPlayed >= cooldown;
                    });

                    // Boredom is still building (screen activity ongoing,
                    // attention only partially faded) — keep it to subtle
                    // fidgets rather than a big attention-grabbing gesture.
                    if (gazeController.getAttentionState() === ATTENTION_STATE.OBSERVING) {
                        const subtle = eligible.filter(({ name }) => _SUBTLE_FIDGET_NAMES.has(name));
                        if (subtle.length > 0) eligible = subtle;
                    }

                    const pool   = eligible.length > 0 ? eligible : _FIDGET_POOL;   // never stall
                    const choice = pool[Math.floor(Math.random() * pool.length)];

                    console.log(
                        `[Maya][fidget] idleForMs=${idleFor} gazeFactor=${gazeFactor.toFixed(2)} ` +
                        `eligible=[${eligible.map(e => e.name).join(",")}] choice=${choice?.name}`
                    );

                    _fidgetLastPlayedAt.set(choice.name, now);
                    _fidgetStartedAt = now;
                    _hasPlayedFirstFidget = true;
                    choice.fn();
                    _bumpFidgetCooldown(choice.name);
                }
            }
        } else {
            console.log("[Maya][fidget] skipped — _canFidgetNow() was false");
        }
        _scheduleNextFidget();
    }, delay);
}

/** Started from wakeAvatar() — begins the randomised idle-fidget loop. */
export function startIdleFidgets() {
    _hasPlayedFirstFidget = false;
    _scheduleNextFidget();
}

/** Stopped from sleepAvatar() — no fidgeting while Maya's asleep. */
export function stopIdleFidgets() {
    clearTimeout(_idleFidgetTimeout);
    _idleFidgetTimeout = null;
}

// ── Screen attention / gaze — Phase 3 (see gaze-controller.js) ─────────────
// Pure input API for a future screen-observation module; no capture/OCR/CV
// happens here. Consumed by startEyeMovement()/startHeadMovement() below.

export function observeScreenActivity(target) {
    gazeController.observeScreenActivity(target);
}
export function getAttentionState() { return gazeController.getAttentionState(); }
export function getAttentionLevel()  { return gazeController.getAttentionLevel(); }
export function isAttentionBored()   { return gazeController.isBored(); }

// ── Avatar loader ─────────────────────────────────────────────────────────────

export function loadAvatar(scene) {
    const loader = new GLTFLoader();
    loader.register(parser => new VRMLoaderPlugin(parser));

    loader.load("assets/mayaaa.vrm", (gltf) => {
        vrm = gltf.userData.vrm;
        scene.add(vrm.scene);
        window.vrm = vrm;
        expressionController.attach(vrm.expressionManager);

        // Start with eyes closed — sleeps until WS connects
        expressionController.setValue(EXPRESSION_LAYER.BASE, "blink", 1);
        expressionController.setValue(EXPRESSION_LAYER.BASE, "relaxed", 0.6);

        // Arm resting pose
        const leftUpperArm  = vrm.humanoid.getNormalizedBoneNode("leftUpperArm");
        const rightUpperArm = vrm.humanoid.getNormalizedBoneNode("rightUpperArm");
        if (leftUpperArm)  { leftUpperArm.rotation.z  = -Math.PI / 2.5; leftUpperArm.rotation.x  = 0.2; }
        if (rightUpperArm) { rightUpperArm.rotation.z =  Math.PI / 2.5; rightUpperArm.rotation.x = 0.2; }

        // Head sway always runs — looks fine even with closed eyes
        startHeadMovement();

        // Handle race: WS connected before VRM finished loading
        if (_awake) {
            expressionController.setValue(EXPRESSION_LAYER.BLINK, "blink", 0);
            startBlinking();
            startEyeMovement();
        }
    });
}

// ── Internal animations ───────────────────────────────────────────────────────

function startBlinking() {
    _blinkingActive = true;
    clearTimeout(_blinkTimeout);   // a pending tick from before a reconnect would double the loop

    function blink() {
        if (!_blinkingActive || !vrm) return;  // exits loop when sleeping

        expressionController.setValue(EXPRESSION_LAYER.BLINK, "blink", 1);
        setTimeout(() => {
            if (!vrm) return;
            expressionController.setValue(EXPRESSION_LAYER.BLINK, "blink", 0);
        }, 120);

        _blinkTimeout = setTimeout(blink, 2000 + Math.random() * 5000);
    }
    blink();
}

// Base-tier (see animation-controller.js) continuous head sway. Layers a
// gaze-driven offset (Phase 3) on top when GazeController is actively
// observing something, and yields the neck/spine bones entirely whenever a
// higher-priority fidget or action animation currently owns them. Also
// drives Phase 4's LifeMotionController (breathing/posture/shoulders) each
// frame — reuses this loop rather than starting a second one.
function startHeadMovement() {
    let t = 0;
    function update() {
        requestAnimationFrame(update);
        if (!vrm) return;
        t += 0.01;
        const neck  = vrm.humanoid.getNormalizedBoneNode("neck");
        const spine = vrm.humanoid.getNormalizedBoneNode("spine");

        const gaze = (gazeController.hasActiveTarget() && !isSpeaking)
            ? gazeController.getHeadOffset()
            : { x: 0, y: 0 };

        if (neck && (!neck.name || animationController.canWrite(neck.name, PRIORITY.BASE))) {
            neck.rotation.y = Math.sin(t) * 0.05 + gaze.y;
            neck.rotation.x = gaze.x;
        }
        if (spine && (!spine.name || animationController.canWrite(spine.name, PRIORITY.BASE))) {
            spine.rotation.x = Math.sin(t * 2) * 0.011;
        }

        lifeMotionController.update(vrm, {
            awake: _awake,
            speaking: isSpeaking,
            attentionState: gazeController.getAttentionState(),
        });
    }
    update();
}

// Idle-drift eye loop. When GazeController has an active screen-attention
// target it takes over the look direction (Phase 3); otherwise this keeps
// its original random idle-drift behaviour. isSpeaking still wins either way.
function startEyeMovement() {
    const leftEye  = vrm.humanoid.getNormalizedBoneNode("leftEye");
    const rightEye = vrm.humanoid.getNormalizedBoneNode("rightEye");
    if (!leftEye || !rightEye) return;

    // Started once per page — the loops below never stop, so a re-wake must not add another.
    if (_eyeLoopStarted) return;
    _eyeLoopStarted = true;

    let idleTargetX = 0, idleTargetY = 0;

    setInterval(() => {
        if (!isSpeaking && _awake && !gazeController.hasActiveTarget()) {
            idleTargetX = (Math.random() - 0.5) * 0.15;
            idleTargetY = (Math.random() - 0.5) * 0.08;
        }
    }, 2000);

    function update() {
        requestAnimationFrame(update);

        let targetX = idleTargetX;
        let targetY = idleTargetY;

        if (isSpeaking) {
            targetX = 0; targetY = -0.02;
        } else if (gazeController.hasActiveTarget()) {
            const gaze = gazeController.getEyeOffset();
            targetX = gaze.x;
            targetY = gaze.y;
        } else if (performance.now() < _behGazeUntil) {
            targetX = _behGazeX;
            targetY = _behGazeY;
        }

        leftEye.rotation.y  += (targetX - leftEye.rotation.y)  * 0.05;
        rightEye.rotation.y += (targetX - rightEye.rotation.y) * 0.05;
        leftEye.rotation.x  += (targetY - leftEye.rotation.x)  * 0.05;
        rightEye.rotation.x += (targetY - rightEye.rotation.x) * 0.05;
    }
    update();
}

window.maya = {
    wave: playWaveAnimation,
    nod: playNodAnimation,
    giggle: playGiggleAnimation,
    sigh: playSighAnimation,
    shrug: playShrugAnimation,
    wink: playWinkAnimation,
    // Fidget animations — same manual-testing convenience as the set above.
    weightShift: playWeightShiftAnimation,
    lookAround: playLookAroundAnimation,
    headTilt: playHeadTiltAnimation,
    shoulderRoll: playShoulderRollAnimation,
    stretch: playStretchAnimation,
    // Registered but not yet wired to anything automatic — manual testing only.
    idle: playIdleAnimation,
    angry: playAngryAnimation,
    bashful: playBashfulAnimation,
    catwalk: playCatwalkAnimation,
    clapping: playClappingAnimation,
    crying: playCryingAnimation,
    cuteThink: playCuteThinkAnimation,
    greeting: playGreetingAnimation,
    handRaise: playHandRaiseAnimation,
    headShake: playHeadShakeAnimation,
    macarenaDance: playMacarenaDanceAnimation,
    rejected: playRejectedAnimation,
    thankful: playThankfulAnimation,
    waving: playWavingAnimation,
    // Phase 3 — screen attention / gaze manual testing.
    observeScreenActivity: observeScreenActivity,
    getAttentionState: getAttentionState,
    getAttentionLevel: getAttentionLevel,
    isBored: isAttentionBored,
    // Behavioral Engine gaze hook — manual testing only.
    behavioralGaze: applyBehavioralGaze,
};