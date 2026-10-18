/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import type { Plugin } from "vite";

// Committed repo data served under /offline/ so the app works without the backend
// (src/lib/offline.ts): benchmark replays from viewer/data and the result plots from docs/results.
// Copied at build, served from disk in dev; nothing is duplicated in git.
const repo = (p: string) => fileURLToPath(new URL(`../${p}`, import.meta.url));
function offlineFiles(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of readdirSync(repo("viewer/data")).filter((x) => x.endsWith(".json.gz"))) out[`viewer-clips/${f}`] = repo(`viewer/data/${f}`);
  const plots = { stress_visibility: "ablation/stress_visibility.png", stress_density: "ablation/stress_density.png", go_no_go_curve: "go_no_go/curve.png" };
  for (const [k, v] of Object.entries(plots)) if (existsSync(repo(`docs/results/${v}`))) out[`plots/${k}.png`] = repo(`docs/results/${v}`);
  // GPU run plots (docs/results/phase3/<condition>/<plot>.png) as plots/gpu-<condition>-<plot>.png, like the API
  const p3 = repo("docs/results/phase3");
  if (existsSync(p3))
    for (const c of readdirSync(p3, { withFileTypes: true }).filter((d) => d.isDirectory()))
      for (const k of ["stress_visibility", "stress_density"]) if (existsSync(`${p3}/${c.name}/${k}.png`)) out[`plots/gpu-${c.name}-${k}.png`] = `${p3}/${c.name}/${k}.png`;
  return out;
}
function offlineData(): Plugin {
  return {
    name: "offline-repo-data",
    configureServer(server) {
      server.middlewares.use("/offline/", (req, res, next) => {
        const src = offlineFiles()[(req.url ?? "").replace(/^\//, "").split("?")[0]];
        if (!src) return next();
        res.setHeader("content-type", src.endsWith(".png") ? "image/png" : "application/octet-stream");
        res.end(readFileSync(src));
      });
    },
    generateBundle() {
      for (const [name, src] of Object.entries(offlineFiles())) this.emitFile({ type: "asset", fileName: `offline/${name}`, source: readFileSync(src) });
    },
  };
}

// The web app talks to the FastAPI service (serve/) under /api. In dev, Vite proxies /api to
// VITE_API_TARGET (default http://127.0.0.1:8000); in production the service serves web/dist
// itself, or nginx proxies /api (docker-compose).
const target = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss(), offlineData()],
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
