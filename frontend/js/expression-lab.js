/**
 * frontend/js/expression-lab.js
 * Standalone calibration tool for core/expression_library.py's
 * expressions.json (canonical location: frontend/assets/expressions.json)
 * — NOT wired into the runtime avatar.js/expression-composer.js pipeline.
 *
 * Every emotion|attitude|intensity combination is edited and persisted
 * independently in this browser's localStorage (versioned schema, see
 * STORAGE_KEY/SCHEMA_VERSION below) — slider edits save immediately and
 * survive a page reload. expressions.json itself is only ever written by
 * the explicit Export action, which dumps every locally-saved combination
 * at once (not just the one currently open) without clearing localStorage.
 * Import seeds/merges an existing expressions.json into localStorage.
 *
 * Deliberately browser-file-based (File API + localStorage) rather than
 * assuming Electron Node-integration is enabled in this renderer — keeps
 * the tool dependency-free and safe to run anywhere the dev server runs.
 *
 * The default composition mirrors core/expression_library.py's
 * _EMOTION_BASE / _ATTITUDE_MODIFIERS / intensity exponents so a
 * never-edited combination previews the same base the backend would
 * generate. Duplicated intentionally — this is a dev tool, not a shared
 * runtime module.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin } from "@pixiv/three-vrm";

const VRM_PATH = "assets/mayaaa.vrm";

// Never composed/shown here — vowel/viseme mouth shapes belong to lip-sync.
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
function exponentFor(key) {
    for (const [prefix, exp] of Object.entries(INTENSITY_EXPONENT)) {
        if (key.startsWith(prefix)) return exp;
    }
    return 0.85;
}

function semanticKey(emotion, attitude, intensityWord) {
    return `${emotion}|${attitude}|${intensityWord}`;
}

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

// ── Persistence — versioned localStorage, one entry per semantic key ───────
//
// Slider edits persist immediately to localStorage (never to a file while
// editing — see task requirements). expressions.json is only ever written
// by the explicit Export action, which dumps every locally-saved recipe at
// once, in the same flat {semanticKey: recipe} schema
// core/expression_library.py reads.

const STORAGE_KEY     = "maya.expressionLab.recipes";
const SCHEMA_VERSION  = 1;

function _loadStore() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return { version: SCHEMA_VERSION, recipes: {} };
        const parsed = JSON.parse(raw);
        if (parsed.version !== SCHEMA_VERSION || typeof parsed.recipes !== "object" || parsed.recipes === null) {
            // Unknown/legacy shape — start fresh rather than risk corrupt data.
            // Future schema bumps migrate here instead of dropping data.
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
        }
    } catch (err) {
        setStatus(`Failed to parse ${file.name}: ${err}`);
    }
    e.target.value = "";
});

document.getElementById("exportLib").addEventListener("click", () => {
    // Include the currently-open combination's live edits even if the
    // last slider tweak hasn't otherwise triggered a save.
    if (_activeKey) persistRecipe(_activeKey, currentRecipe());

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

// name -> Array<{mesh, index}> | null. Raw mesh morph targets — NOT
// VRMExpressionManager presets/customs — same lookup as
// frontend/js/expression-composer.js's _resolveMorphTargets.
const morphCache = new Map();

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

// All morph names actually collected so far (built up as recipes are
// checked) — used only for the slider's "(unverified)" label.
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
            persistRecipe(_activeKey, recipe);   // immediate local persistence — never expressions.json here
        });
        slidersDiv.appendChild(row);
    }
}

/**
 * Saves the outgoing combination's current recipe (if any was active),
 * then loads the target combination — from localStorage if it's been
 * edited before, otherwise a freshly generated default. Never touches
 * expressions.json (see Export).
 */
function switchCombination() {
    if (_activeKey) {
        persistRecipe(_activeKey, currentRecipe());
    }

    _activeKey = currentKey();
    const saved = store.recipes[_activeKey];
    const recipe = saved ? { ...saved } : composeDefault(emotionSel.value, attitudeSel.value, intensitySel.value);

    buildSliders(recipe);
    applyRecipeToVrm(recipe);
    setStatus((saved ? "Restored saved recipe for " : "Generated default recipe for ") + `'${_activeKey}'.`);
}

[emotionSel, attitudeSel, intensitySel].forEach((el) =>
    el.addEventListener("change", switchCombination)
);