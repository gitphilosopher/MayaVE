/**
 * frontend/js/main.js
 * Browser entry point loaded by frontend/index.html. Creates and exports the
 * Three.js scene, camera, and renderer; loads the VRM avatar; and imports
 * websocket.js to establish backend communication and event handling.
 *
 * A single animation-frame loop renders the scene, updates the VRM, advances
 * active VRMA mixers, and updates gaze. The transparent scene/renderer let the
 * avatar appear over the desktop window; resize events keep the camera and
 * renderer dimensions in sync with the viewport.
 */

import * as THREE from "three";
import { loadAvatar, vrm, updateVrmaAnimations, updateGaze, warmUpAudioContext, isSpeaking } from "./avatar.js";
import "./websocket.js";
import { recordFrameGap, recordTrace, recordRafHeartbeat } from "./tracer.js";

export const scene = new THREE.Scene();
// Transparent background for the overlay window.
scene.background = null;

const clock = new THREE.Clock();

export const camera = new THREE.PerspectiveCamera(
    18,                                  // tight FOV — passport/bust crop
    window.innerWidth / window.innerHeight,
    0.1,
    100
);

// Bust framing keeps the face and shoulders in view while cropping below the waist.
camera.position.set(-0.05, 1.35, 2.5);
// camera.position.set(0.25, 1.35, 10);
camera.lookAt(0, 1.32, 0);

export const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha:     true,          // transparent canvas
    premultipliedAlpha: false,
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0);   // fully transparent clear
document.body.appendChild(renderer.domElement);

const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(1, 2, 3);
scene.add(key);

const rim = new THREE.DirectionalLight(0xb0c8ff, 0.8);
rim.position.set(-2, 1, -2);
scene.add(rim);

const fill = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(fill);

loadAvatar(scene);
warmUpAudioContext(true);

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

export let AVATAR_FPS = 60;
window.setAvatarFps = function(fps) {
    AVATAR_FPS = Number(fps) || 0;
};
window.getAvatarFps = function() {
    return AVATAR_FPS;
};

const _PERF_CAP = 3600;
const _perfBuf = new Float64Array(_PERF_CAP);
let _perfHead = 0, _perfCount = 0;
let _perfLastTime = performance.now();
let _lastRenderTime = performance.now();

window.getPerfMetrics = function(reset = true) {
    if (_perfCount === 0) {
        return {
            fps: 0, avg_ms: 0, p95_ms: 0, p99_ms: 0, max_ms: 0, count: 0,
            stalls_16ms: 0, stalls_33ms: 0, stalls_100ms: 0
        };
    }
    const sorted = Array.from(_perfBuf.subarray(0, _perfCount)).sort((a, b) => a - b);
    const count = sorted.length;
    const sum = sorted.reduce((a, b) => a + b, 0);
    const avg = sum / count;
    const fps = 1000 / avg;
    const p95 = sorted[Math.floor(count * 0.95)];
    const p99 = sorted[Math.floor(count * 0.99)];
    const max = sorted[count - 1];

    let s16 = 0, s33 = 0, s100 = 0;
    for (let i = 0; i < count; i++) {
        const val = sorted[i];
        if (val > 16.7) s16++;
        if (val > 33.3) s33++;
        if (val > 100.0) s100++;
    }

    if (reset) { _perfCount = 0; _perfHead = 0; }
    return {
        fps: Number(fps.toFixed(1)),
        avg_ms: Number(avg.toFixed(2)),
        p95_ms: Number((p95 || 0).toFixed(2)),
        p99_ms: Number((p99 || 0).toFixed(2)),
        max_ms: Number((max || 0).toFixed(2)),
        count,
        stalls_16ms: s16,
        stalls_33ms: s33,
        stalls_100ms: s100
    };
};

/** Render the scene and advance avatar systems once per animation frame. */
function animate(timestamp) {
    requestAnimationFrame(animate);

    const now = (typeof timestamp === "number" && timestamp > 0) ? timestamp : performance.now();

    if (AVATAR_FPS > 0) {
        const targetInterval = 1000 / AVATAR_FPS;
        const elapsed = now - _lastRenderTime;
        if (elapsed < targetInterval) {
            return;
        }
        _lastRenderTime = now - (elapsed % targetInterval);
    } else {
        _lastRenderTime = now;
    }

    const dt = now - _perfLastTime;
    const prevTime = _perfLastTime;
    _perfLastTime = now;
    recordRafHeartbeat(now, dt);
    if (dt > 0 && dt < 1000) {
        _perfBuf[_perfHead] = dt;
        _perfHead = (_perfHead + 1) % _PERF_CAP;
        if (_perfCount < _PERF_CAP) _perfCount++;
    }

    const tRender0 = performance.now();
    renderer.render(scene, camera);
    const tRender1 = performance.now();

    const rawDelta = clock.getDelta();
    const delta = Math.min(rawDelta, 0.1);

    const tVrm0 = performance.now();
    if (vrm) vrm.update(delta);
    updateVrmaAnimations(delta);
    updateGaze(delta);
    const tVrm1 = performance.now();

    if (dt >= 33.3) {
        recordFrameGap(prevTime, now, dt, {
            is_speaking: !!isSpeaking,
            vrm_loaded: !!vrm,
            render_ms: Number((tRender1 - tRender0).toFixed(3)),
            vrm_ms: Number((tVrm1 - tVrm0).toFixed(3))
        });
    }
}

animate();
