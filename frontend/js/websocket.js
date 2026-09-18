/**
 * frontend/js/websocket.js
 *
 * Mood note: handleState() used to force an expression on "processing"
 * and "idle" (neutral / relaxed), which fought with Maya's persistent
 * mood (core/mood.py on the backend). The backend now sends an explicit
 * `behavior` message — composed by core/behavior_engine.py, already
 * mood-aware — right alongside every state change, so handleState()
 * only owns "listening" here.
 *
 * Behavior note: `behavior` messages (composed communicative intent —
 * primary/secondary emotion, intensity, attitude, gaze) are handed to
 * expression-composer.js's applyBehavior(), which blends them into VRM
 * weights instead of the old flat tag -> single blendshape map.
 *
 * Idle-fidget note: handleState() also forwards every state value to
 * avatar.js's setAvatarState() so the idle-fidget scheduler (see
 * avatar.js's "Idle fidget animations" section) knows when it's actually
 * safe to play one — only while the backend reports "idle", never during
 * listening/processing/speaking.
 */

import {
    speakFromBytes, setExpression, playWaveAnimation, setAudioDoneCallback,
    wakeAvatar, sleepAvatar, stopCurrentAudio, setAvatarState,
    playNodAnimation, playGiggleAnimation, playSighAnimation,
    playShrugAnimation, playWinkAnimation,
} from "./avatar.js";
import { applyBehavior } from "./expression-composer.js";

const WS_URL       = "ws://localhost:8765";
const RECONNECT_MS = 2000;

export let ws = null;

// Tell avatar.js to notify us when a sentence finishes — we send audio_done to backend
setAudioDoneCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "audio_done" }));
    }
});

function connect() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log("[Maya WS] Connected");
        setStatus("connected");
        wakeAvatar();
    };

    ws.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

        switch (data.type) {
            case "audio":
                speakFromBytes(base64ToArrayBuffer(data.data));
                break;
            case "state":
                handleState(data.value);
                break;
            case "behavior":
                applyBehavior(data);
                break;
            case "stop_audio":
                // Barge-in: backend cancelled Maya's current speech task
                // (core/state.py's interrupt()) — halt playback here too
                // instead of letting an already-queued sentence finish.
                stopCurrentAudio();
                break;
            case "transcript":
                // Reserved for transcript overlay — no-op for now
                break;
            case "animation":
                // Fire-and-forget: several of these (wave, nod, giggle,
                // sigh, shrug) are now async in avatar.js since they load
                // a .vrma file before playing. They're intentionally never
                // awaited here — playback of the incoming "audio" message
                // (handled above, also fire-and-forget) must never wait on
                // an animation, so Maya can speak while an animation is
                // still loading or mid-playback. .catch() just prevents an
                // unhandled-rejection warning if a .vrma fails to load;
                // it does not delay anything.
                switch (data.name) {
                    case "wave":   playWaveAnimation()?.catch(_logAnimationError);   break;
                    case "nod":    playNodAnimation()?.catch(_logAnimationError);    break;
                    case "giggle": playGiggleAnimation()?.catch(_logAnimationError); break;
                    case "sigh":   playSighAnimation()?.catch(_logAnimationError);   break;
                    case "shrug":  playShrugAnimation()?.catch(_logAnimationError);  break;
                    case "wink":   playWinkAnimation();   break;
                }
                break;
        }
    };

    ws.onclose = () => {
        console.warn("[Maya WS] Disconnected — retrying in", RECONNECT_MS, "ms");
        setStatus("disconnected");
        sleepAvatar();
        setTimeout(connect, RECONNECT_MS);
    };

    ws.onerror = (err) => console.error("[Maya WS] Error:", err);
}

function handleState(value) {
    // Let the idle-fidget scheduler know the backend's current state
    // regardless of which branch (if any) below fires for it.
    setAvatarState(value);

    switch (value) {
        case "listening":
            setExpression("surprised", 0.3);
            break;
        // "processing" and "idle" no longer force an expression here —
        // the backend sends an explicit, mood-aware `behavior` message
        // alongside every state change (see core/behavior_engine.py).
    }
}

function setStatus(status) {
    const el = document.getElementById("ws-status");
    if (el) el.textContent = status === "connected" ? "" : "💤";
}

function base64ToArrayBuffer(b64) {
    const binary = atob(b64);
    const buf    = new ArrayBuffer(binary.length);
    const view   = new Uint8Array(buf);
    for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
    return buf;
}

function _logAnimationError(err) {
    console.error("[Maya] Animation playback error:", err);
}

connect();