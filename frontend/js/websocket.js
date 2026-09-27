/**
 * frontend/js/websocket.js
 *
 * Browser client for the backend avatar protocol. main.js imports this module
 * for its connection side effects; the current WebSocket is also exported as
 * `ws`.
 *
 * Routes audio and state messages to avatar.js, behavior packets to
 * expression-composer.js, and animation events to avatar actions. Playback
 * completion is reported as `audio_done` while the socket is open; `stop_audio`
 * cancels local playback. Async VRMA actions are not awaited, so asset loading
 * cannot delay speech. Transcript messages currently have no UI handler.
 *
 * Every backend state reaches the fidget gate, which requires `idle` for
 * automatic fidgets. `sleeping` selects the sleep path; `listening` also
 * applies its expression, and all other states wake the avatar. The server
 * replays its last state on connect, so socket open only updates transport
 * status; socket close sleeps the avatar and schedules a reconnect.
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

// Report streamed-audio completion to the backend while the socket is open.
setAudioDoneCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "audio_done" }));
    }
});

/** Open the backend socket and install handlers; closure schedules a retry. */
function connect() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log("[Maya WS] Connected");
        setStatus("connected");
        // The replayed backend state, not transport connection, sets sleep/wake.
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
                // Stop current playback immediately when the backend interrupts speech.
                stopCurrentAudio();
                break;
            case "transcript":
                // No transcript UI is currently connected.
                break;
            case "animation":
                // Fire-and-forget so animation loading never delays concurrent audio.
                // Log promise rejections at the WebSocket event boundary.
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

/** Sync fidget eligibility and the avatar's visual awake/sleep state. */
function handleState(value) {
    // The scheduler needs every backend state, including "sleeping".
    setAvatarState(value);

    switch (value) {
        case "listening":
            setExpression("surprised", 0.3);
            wakeAvatar();
            break;
        case "sleeping":
            // Voice-triggered sleep is a backend state; the socket remains open.
            sleepAvatar();
            setStatus("sleeping");
            break;
        default:
            wakeAvatar();
            setStatus("connected");
            break;
    }
}

function setStatus(status) {
    const el = document.getElementById("ws-status");
    if (el) el.textContent = status === "connected" ? "" : "💤";
}

/** Decode an incoming base64 audio payload for avatar playback. */
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