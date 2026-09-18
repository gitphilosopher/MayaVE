/**
 * electron/main.js
 * Maya avatar window — upper body, transparent, bottom-right of screen.
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
        x:           sw - W + 240,   // 16px from right edge
        y:           sh - H + 48,        // flush to taskbar
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