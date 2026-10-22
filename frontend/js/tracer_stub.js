/**
 * frontend/js/tracer_stub.js
 * Fallback stub providing safe no-op implementations of the tracer interface.
 * Ensures clean checkouts build and run cleanly when tracer.js is omitted.
 */

let _currentTraceId = null;
let _rafSeq = 0;

/**
 * Record a trace event. Degrades to a no-op when full tracer is omitted.
 * @param {string|null} [traceId]
 * @param {string} [event]
 * @param {Object} [data]
 */
export function recordTrace(traceId = null, event = "", data = {}) {
    if (typeof window !== "undefined" && window.__maya_tracer?.recordTrace) {
        return window.__maya_tracer.recordTrace(traceId, event, data);
    }
}

/**
 * Record a frame gap. Degrades to a no-op when full tracer is omitted.
 * @param {...*} _args
 */
export function recordFrameGap(..._args) {
    if (typeof window !== "undefined" && window.__maya_tracer?.recordFrameGap) {
        return window.__maya_tracer.recordFrameGap(..._args);
    }
}

/**
 * Record a requestAnimationFrame heartbeat.
 * @param {number} [_now]
 * @param {number} [_dt]
 * @returns {number} Sequence number
 */
export function recordRafHeartbeat(_now = 0, _dt = 0) {
    if (typeof window !== "undefined" && window.__maya_tracer?.recordRafHeartbeat) {
        return window.__maya_tracer.recordRafHeartbeat(_now, _dt);
    }
    return ++_rafSeq;
}

/**
 * Set the current active trace identifier.
 * @param {string|null} [traceId]
 */
export function setCurrentTraceId(traceId = null) {
    _currentTraceId = traceId;
    if (typeof window !== "undefined" && window.__maya_tracer?.setCurrentTraceId) {
        window.__maya_tracer.setCurrentTraceId(traceId);
    }
}

/**
 * Retrieve collected trace data.
 * @param {boolean} [_reset=true]
 * @returns {Object} Trace payload
 */
export function getTraceData(_reset = true) {
    if (typeof window !== "undefined" && window.__maya_tracer?.getTraceData) {
        return window.__maya_tracer.getTraceData(_reset);
    }
    return {
        traces: {},
        frame_gaps: [],
        heartbeats: [],
    };
}
