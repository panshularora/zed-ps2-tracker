/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { readdirSync, readFileSync } from "node:fs";
import type { Plugin } from "vite";

// Committed benchmark replays (viewer/data/*.json.gz) are served at /offline/viewer-clips/ so the
// app works without the backend (src/lib/offline.ts). Copied at build, served from disk in dev.
const CLIPS = fileURLToPath(new URL("../viewer/data", import.meta.url));
function offlineClips(): Plugin {
  const files = () => readdirSync(CLIPS).filter((f) => f.endsWith(".json.gz"));
  return {
    name: "offline-viewer-clips",
    configureServer(server) {
      server.middlewares.use("/offline/viewer-clips/", (req, res, next) => {
        const f = (req.url ?? "").replace(/^\//, "").split("?")[0];
        if (!files().includes(f)) return next();
        res.setHeader("content-type", "application/gzip");
        res.end(readFileSync(`${CLIPS}/${f}`));
      });
    },
    generateBundle() {
      for (const f of files()) this.emitFile({ type: "asset", fileName: `offline/viewer-clips/${f}`, source: readFileSync(`${CLIPS}/${f}`) });
    },
  };
}

// The web app talks to the FastAPI service (serve/) under /api. In dev, Vite proxies /api to
// VITE_API_TARGET (default http://127.0.0.1:8000); in production the service serves web/dist
// itself, or nginx proxies /api (docker-compose).
const target = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss(), offlineClips()],
  resolve: {
    // theme tokens and the frame-clock maths are shared with the static viewer (one product look)
    alias: { "@viewer": fileURLToPath(new URL("../viewer/src", import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: { "/api": { target, changeOrigin: true } },
    fs: { allow: [".."] },
  },
  preview: { port: 4173, proxy: { "/api": { target, changeOrigin: true } } },
  build: { outDir: "dist", sourcemap: true },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
    globals: true,
  },
});
