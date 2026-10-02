/**
 * electron/main.js
 * Maya avatar window — upper body, transparent, bottom-left of screen.
 */

import pkg from "electron";
const { app, BrowserWindow, screen } = pkg;
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const W = 820;   // portrait width
const H = 440;   // tall enough for face + shoulders, waist clipped

// The taskbar re-raises itself on click without focusing/blurring this
// click-through window, so no window event signals it. A slow watchdog stays
// as the only way to re-assert z-order; event hooks cover everything else.
const KEEP_ON_TOP_WATCHDOG_MS = 2000;

let win = null;
let keepOnTopTimer = null;

/** Same placement as before: bottom-left, flush to the taskbar. */
function computeBounds() {
    const { height: sh } = screen.getPrimaryDisplay().workAreaSize;
    return { x: -260, y: sh - H + 136, width: W, height: H };
}

function keepOnTop() {
    if (!win || win.isDestroyed()) return;
    win.setAlwaysOnTop(true, "screen-saver");   // highest topmost band
    win.moveTop();
}

function reposition() {
    if (!win || win.isDestroyed()) return;
    win.setBounds(computeBounds());
    keepOnTop();
}

function createWindow() {
    win = new BrowserWindow({
        ...computeBounds(),
        transparent: true,
        frame:       false,
        alwaysOnTop: true,
        resizable:   false,
        hasShadow:   false,
        skipTaskbar: true,
        webPreferences: {
            contextIsolation: true,
            spellcheck: false,        // no text-input UI
        },
    });

    keepOnTop();

    win.on("always-on-top-changed", (_e, isOnTop) => { if (!isOnTop) keepOnTop(); });
    win.on("blur", keepOnTop);
    win.on("show", keepOnTop);
    keepOnTopTimer = setInterval(keepOnTop, KEEP_ON_TOP_WATCHDOG_MS);

    screen.on("display-metrics-changed", reposition);
    screen.on("display-added", reposition);
    screen.on("display-removed", reposition);

    win.on("closed", () => {
        clearInterval(keepOnTopTimer);
        keepOnTopTimer = null;
        screen.removeListener("display-metrics-changed", reposition);
        screen.removeListener("display-added", reposition);
        screen.removeListener("display-removed", reposition);
        win = null;
    });

    if (process.env.VITE_DEV_SERVER_URL) {
        win.loadURL(process.env.VITE_DEV_SERVER_URL);
    } else {
        win.loadFile(path.join(__dirname, "../dist/index.html"));
    }

    // Fully click-through. No renderer code listens for mouse events, so
    // forwarding moves ({ forward: true }) was pure overhead.
    win.setIgnoreMouseEvents(true);
}

if (!app.requestSingleInstanceLock()) {
    app.quit();
} else {
    app.on("second-instance", keepOnTop);
    app.whenReady().then(createWindow);
    app.on("window-all-closed", () => {
        if (process.platform !== "darwin") app.quit();
    });
}