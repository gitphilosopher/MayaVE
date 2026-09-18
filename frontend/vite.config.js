/**
 * frontend/vite.config.js
 * Adds vite-plugin-electron to the existing Vite setup.
 */

import { defineConfig } from "vite";
import electron from "vite-plugin-electron/simple";

export default defineConfig({
    plugins: [
        electron({
            main: {
                // Electron main process entry
                entry: "electron/main.js",
            },
        }),
    ],
    // Needed so asset paths resolve correctly when Electron loads built files
    base: "./",
});
