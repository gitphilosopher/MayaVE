/**
 * electron/main.js
 * Maya avatar window — upper body, transparent, bottom-left of screen.
 */

import { app, BrowserWindow, screen } from "electron";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let win;

function createWindow() {
    const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;

    const W = 820;   // portrait width
    const H = 440;   // tall enough for face + shoulders, waist clipped

    win = new BrowserWindow({
        width:       W,
        height:      H,
        x:           -260,               // mirror of the old right-side offset
        y:           sh - H + 136,        // flush to taskbar
        transparent: true,
        frame:       false,
        alwaysOnTop: true,
        resizable:   false,
        hasShadow:   false,
        skipTaskbar: true,           // don't show in taskbar
        webPreferences: {
            contextIsolation: true,
        },
    });

    // Highest topmost band so she stays above the taskbar.
    win.setAlwaysOnTop(true, "screen-saver");

    // The taskbar re-raises itself on click; re-assert Maya's z-order.
    const keepOnTop = () => {
        if (!win || win.isDestroyed()) return;
        win.setAlwaysOnTop(true, "screen-saver");
        win.moveTop();
    };
    win.on("always-on-top-changed", (_e, isOnTop) => { if (!isOnTop) keepOnTop(); });
    setInterval(keepOnTop, 500);

    if (process.env.VITE_DEV_SERVER_URL) {
        win.loadURL(process.env.VITE_DEV_SERVER_URL);
    } else {
        win.loadFile(path.join(__dirname, "../dist/index.html"));
    }

    // Make window draggable
    win.webContents.on("did-finish-load", () => {
        win.webContents.insertCSS(`
            body  { -webkit-app-region: drag; }
            canvas { -webkit-app-region: no-drag; }
        `);
    });

    win.setIgnoreMouseEvents(true, { forward: true });
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
});