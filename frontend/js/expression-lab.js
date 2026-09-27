/**
 * frontend/js/expression-lab.js
 * Standalone browser calibration tool for the recipes used by
 * core/expression_library.py. Opened through frontend/expression-lab.html,
 * it loads mayaaa.vrm and previews recipes by writing verified raw morph
 * targets directly to mesh influences; it is not part of the runtime avatar.
 *
 * Each emotion|attitude|intensity combination is a versioned localStorage
 * draft. Slider edits save immediately; Import merges a selected JSON library;
 * Export downloads all saved combinations without clearing local drafts. The
 * exported file belongs at frontend/assets/expressions.json.
 *
 * Default recipes mirror the backend's emotion bases, attitude modifiers,
 * and intensity exponents so unedited combinations preview its generated
 * values. This logic is intentionally duplicated for the standalone tool.
 * File import/export uses browser APIs rather than Electron Node integration.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin } from "@pixiv/three-vrm";

const VRM_PATH = "assets/mayaaa.vrm";

// Excluded from generated defaults; vowel/viseme shapes belong to lip-sync.
const MOUTH_VISEME_KEYS = new Set(["Fcl_MTH_A", "Fcl_MTH_I", "Fcl_MTH_U", "Fcl_MTH_E", "Fcl_MTH_O"]);

// Mirrors core/expression_library.py's _EMOTION_BASE.
const EMOTION_BASE = {
    happy:     { Fcl_BRW_Joy: 0.6,  Fcl_EYE_Joy: 0.55, Fcl_MTH_Joy: 0.7 },
    excited:   { Fcl_BRW_Joy: 0.7,  Fcl_EYE_Surprised: 0.35, Fcl_EYE_Joy: 0.5,
                 Fcl_MTH_Joy: 0.85, Fcl_MTH_Large: 0.25 },
    surprised: { Fcl_BRW_Surprised: 0.75, Fcl_EYE_Surprised: 0.75, Fcl_MTH_Surprised: 0.6 },
    angry:     { Fcl_BRW_Angry: 0.75, Fcl_EYE_Angry: 0.65, Fcl_MTH_Angry: 0.55 },
    sad:       { Fcl_BRW_Sorrow: 0.6, Fcl_EYE_Sorrow: 0.6, Fcl_MTH_Sorrow: 0.55,
                 Fcl_EYE_Close_L: 0.15, Fcl_EYE_Close_R: 0.15 },
    scared:    { Fcl_BRW_Surprised: 0.9, Fcl_EYE_Surprised: 0.85, Fcl_EYE_Spread: 0.7,
                 Fcl_MTH_Surprised: 0.6, Fcl_MTH_Down: 0.35 },
    relaxed:   { Fcl_EYE_Natural: 0.4, Fcl_MTH_Neutral: 0.3 },
    neutral:   { Fcl_EYE_Natural: 0.25, Fcl_MTH_Neutral: 0.15 },
};

// Mirrors core/expression_library.py's _ATTITUDE_MODIFIERS.
const ATTITUDE_MODIFIERS = {
    playful: { Fcl_BRW_Fun: 0.3,  Fcl_EYE_Fun: 0.25,  Fcl_MTH_Fun: 0.3 },
    teasing:  { Fcl_BRW_Fun: 0.25, Fcl_EYE_Joy_L: 0.2, Fcl_MTH_Fun: 0.2 },
    mock:     { Fcl_MTH_Fun: 0.25 },
    sincere:  {},
};

const INTENSITY_BANDS = { low: 0.3, medium: 0.6, high: 0.9 };

const INTENSITY_EXPONENT = { Fcl_BRW: 0.85, Fcl_EYE: 0.9, Fcl_MTH: 0.75 };
/** Return the intensity exponent assigned to a VRoid morph family. */
function exponentFor(key) {
    for (const [prefix, exp] of Object.entries(INTENSITY_EXPONENT)) {
        if (key.startsWith(prefix)) return exp;
    }
    return 0.85;
}

function semanticKey(emotion, attitude, intensityWord) {
    return `${emotion}|${attitude}|${intensityWord}`;
}

/** Generate an unpersisted preview recipe using the backend's base formula. */
function composeDefault(emotion, attitude, intensityWord) {
    const base = EMOTION_BASE[emotion] || EMOTION_BASE.neutral;
    const modifier = ATTITUDE_MODIFIERS[attitude] || {};
    const intensity = INTENSITY_BANDS[intensityWord] ?? INTENSITY_BANDS.medium;

    const recipe = {};
    for (const [key, weight] of Object.entries(base)) {
        if (MOUTH_VISEME_KEYS.has(key)) continue;
        recipe[key] = Math.round(Math.pow(intensity, exponentFor(key)) * weight * 1000) / 1000;
    }
    for (const [key, weight] of Object.entries(modifier)) {
        if (MOUTH_VISEME_KEYS.has(key)) continue;
        const add = Math.pow(intensity, exponentFor(key)) * weight;
        recipe[key] = Math.round(((recipe[key] || 0) + add) * 1000) / 1000;
    }
    return recipe;
}

// ── Versioned local drafts in localStorage ─────────────────────────────────

const STORAGE_KEY     = "maya.expressionLab.recipes";
const SCHEMA_VERSION  = 1;

/** Load local drafts, falling back to an empty store for unsupported data. */
function _loadStore() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return { version: SCHEMA_VERSION, recipes: {} };
        const parsed = JSON.parse(raw);
        if (parsed.version !== SCHEMA_VERSION || typeof parsed.recipes !== "object" || parsed.recipes === null) {
            // Unknown version or missing/non-object recipes field: start fresh.
            return { version: SCHEMA_VERSION, recipes: {} };
        }
        return parsed;
    } catch (e) {
        console.warn("[Maya][expr-lab] localStorage read failed, starting fresh:", e);
        return { version: SCHEMA_VERSION, recipes: {} };
    }
}

function _saveStore() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    } catch (e) {
        console.warn("[Maya][expr-lab] localStorage write failed:", e);
    }
}

let store = _loadStore();         // { version, recipes: { semanticKey: recipe } }
let _activeKey = null;            // the combination currently shown/edited
let _dirty = false;               // true once a slider was moved on the open combination

/** Save one semantic-key draft to this browser's localStorage. */
function persistRecipe(key, recipe) {
    if (!key) return;
    store.recipes[key] = recipe;
    _saveStore();
}

function setStatus(msg) {
    document.getElementById("status").textContent = msg;
}

document.getElementById("importLib").addEventListener("click", () => {
    document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
        const imported = JSON.parse(await file.text());
        let count = 0;
        for (const [key, recipe] of Object.entries(imported)) {
            store.recipes[key] = recipe;
            count++;
        }
        _saveStore();
        setStatus(`Imported ${count} recipe(s) from ${file.name} into local storage.`);
        // Refresh the open combination in case it was among the imported keys.
        if (_activeKey && store.recipes[_activeKey]) {
            const recipe = { ...store.recipes[_activeKey] };
            buildSliders(recipe);
            applyRecipeToVrm(recipe);
            _dirty = false;
        }
    } catch (err) {
        setStatus(`Failed to parse ${file.name}: ${err}`);
    }
    e.target.value = "";
});

document.getElementById("exportLib").addEventListener("click", () => {
    // Only an edited combination needs saving here; an untouched one must
    // not be pinned into the export as if it were calibrated.
    if (_activeKey && _dirty) persistRecipe(_activeKey, currentRecipe());

    const blob = new Blob([JSON.stringify(store.recipes, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "expressions.json";
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`Exported ${Object.keys(store.recipes).length} recipe(s) to expressions.json. `
        + `Place it at frontend/assets/expressions.json. (Local storage unchanged.)`);
});

// ── VRM scene ────────────────────────────────────────────────────────────────

const viewport = document.getElementById("viewport");
const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(30, viewport.clientWidth / viewport.clientHeight, 0.1, 20);
camera.position.set(0, 1.3, 1.6);
camera.lookAt(0, 1.2, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
viewport.appendChild(renderer.domElement);

const light = new THREE.DirectionalLight(0xffffff, 1.2);
light.position.set(1, 2, 2);
scene.add(light);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));

let vrm = null;

// Cache raw mesh morph-target matches (not VRMExpressionManager presets).
const morphCache = new Map();

/** Find all meshes exposing a morph target, caching both matches and misses. */
function resolveMorphTargets(name) {
    if (morphCache.has(name)) return morphCache.get(name);
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
    morphCache.set(name, result);
    return result;
}

// Verified morphs written by previews, tracked so old values can be cleared.
const seenVerified = new Set();

const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));
loader.load(
    VRM_PATH,
    (gltf) => {
        vrm = gltf.userData.vrm;
        scene.add(vrm.scene);
        setStatus("VRM loaded.");
        switchCombination();
    },
    undefined,
    (err) => setStatus(`Failed to load ${VRM_PATH}: ${err}`)
);

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
    camera.aspect = viewport.clientWidth / viewport.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewport.clientWidth, viewport.clientHeight);
});

// ── Panel wiring ─────────────────────────────────────────────────────────────

const emotionSel   = document.getElementById("emotion");
const attitudeSel   = document.getElementById("attitude");
const intensitySel  = document.getElementById("intensity");
const slidersDiv    = document.getElementById("sliders");

function currentKey() {
    return semanticKey(emotionSel.value, attitudeSel.value, intensitySel.value);
}

function currentRecipe() {
    const recipe = {};
    slidersDiv.querySelectorAll("input[type=range]").forEach((input) => {
        recipe[input.dataset.key] = parseFloat(input.value);
    });
    return recipe;
}

/** Clear previous preview weights, then apply the recipe's verified morphs. */
function applyRecipeToVrm(recipe) {
    if (!vrm) return;
    // Clear every morph target seen so far (across any prior recipe) so
    // switching selections doesn't leave a stale weight behind.
    for (const name of seenVerified) {
        const targets = resolveMorphTargets(name);
        if (targets) for (const { mesh, index } of targets) mesh.morphTargetInfluences[index] = 0;
    }
    for (const [key, value] of Object.entries(recipe)) {
        const targets = resolveMorphTargets(key);
        if (targets) {
            seenVerified.add(key);
            for (const { mesh, index } of targets) mesh.morphTargetInfluences[index] = value;
        }
    }
}

function buildSliders(recipe) {
    slidersDiv.innerHTML = "";
    const keys = Object.keys(recipe).sort();
    if (keys.length === 0) {
        slidersDiv.innerHTML = "<p style='font-size:11px;color:#888'>No morphs in this recipe.</p>";
        return;
    }
    for (const key of keys) {
        const verified = !vrm || !!resolveMorphTargets(key);
        const row = document.createElement("div");
        row.className = "slider-row";
        row.innerHTML = `
            <div class="row-head">
                <span>${key}${verified ? "" : " (unverified)"}</span>
                <span class="val">${recipe[key].toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="1" step="0.01" value="${recipe[key]}" data-key="${key}" />
        `;
        const input = row.querySelector("input");
        const valLabel = row.querySelector(".val");
        input.addEventListener("input", () => {
            valLabel.textContent = parseFloat(input.value).toFixed(2);
            const recipe = currentRecipe();
            applyRecipeToVrm(recipe);
            _dirty = true;
            persistRecipe(_activeKey, recipe);   // immediate local persistence — never expressions.json here
        });
        slidersDiv.appendChild(row);
    }
}

/** Save a dirty outgoing draft, then show the saved or generated target recipe. */
function switchCombination() {
    if (_activeKey && _dirty) {
        persistRecipe(_activeKey, currentRecipe());
    }

    _activeKey = currentKey();
    _dirty = false;
    const saved = store.recipes[_activeKey];
    const recipe = saved ? { ...saved } : composeDefault(emotionSel.value, attitudeSel.value, intensitySel.value);

    buildSliders(recipe);
    applyRecipeToVrm(recipe);
    setStatus((saved ? "Restored saved recipe for " : "Generated default recipe for ") + `'${_activeKey}'.`);
}

[emotionSel, attitudeSel, intensitySel].forEach((el) =>
    el.addEventListener("change", switchCombination)
);