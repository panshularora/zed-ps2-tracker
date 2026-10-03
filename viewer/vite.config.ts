/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// One self-contained dist/index.html (JS, CSS and the embedded clips inlined), so it opens from
// file:// with no server.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  base: "./",
  build: { outDir: "dist", assetsInlineLimit: 100_000_000 },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
