/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The web app talks to the FastAPI service (serve/) under /api. In dev, Vite proxies /api to
// VITE_API_TARGET (default http://127.0.0.1:8000); in production the service serves web/dist
// itself, or nginx proxies /api (docker-compose).
const target = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
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
