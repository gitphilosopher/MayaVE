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
import { loadAvatar, vrm, updateVrmaAnimations, updateGaze } from "./avatar.js";
import "./websocket.js";

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

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

/** Render the scene and advance avatar systems once per animation frame. */
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    const delta = clock.getDelta();
    if (vrm) vrm.update(delta);
    updateVrmaAnimations(delta);
    updateGaze(delta);
}

animate();
