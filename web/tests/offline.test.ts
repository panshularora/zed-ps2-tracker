import { describe, expect, it, vi } from "vitest";
import { offlinePath, withOfflineFallback, isOffline } from "../src/lib/offline";

describe("offline fallback", () => {
  it("maps API routes to committed snapshots", () => {
    expect(offlinePath("/api/benchmarks")).toBe("/offline/benchmarks.json");
    expect(offlinePath("/api/results/go-no-go")).toBe("/offline/results__go-no-go.json");
    expect(offlinePath("/api/config/baselines/bytetrack")).toBe("/offline/config__baselines__bytetrack.json");
    expect(offlinePath("/api/viewer-clips/MOT17-09.gt.json.gz")).toBe("/offline/viewer-clips/MOT17-09.gt.json.gz");
    expect(offlinePath("/api/clips/subway_turnstile/video")).toBe("/demo/subway_turnstile.mp4");
    expect(offlinePath("/api/jobs/abc")).toBeNull();
    expect(offlinePath("/other")).toBeNull();
  });
  it("falls back on network error and 5xx, not on 4xx or POST", async () => {
    const base = vi.fn(async (u: RequestInfo | URL) => {
      const url = String(u);
      if (url.startsWith("/offline/")) return new Response('{"ok":1}', { headers: { "content-type": "application/json" } });
      if (url === "/api/a") throw new TypeError("Failed to fetch");
      if (url === "/api/b") return new Response("x", { status: 502 });
      return new Response("nf", { status: 404 });
    }) as unknown as typeof fetch;
    const f = withOfflineFallback(base);
    expect(await (await f("/api/a")).json()).toEqual({ ok: 1 });
    expect(isOffline()).toBe(true);
    expect(await (await f("/api/b")).json()).toEqual({ ok: 1 });
    expect((await f("/api/c")).status).toBe(404);
    await expect(f("/api/a", { method: "POST" })).rejects.toThrow();
  });
  it("passes a live service's own 5xx error through (model_missing), no snapshot", async () => {
    const body = { error: { code: "model_missing", message: "file is missing", fix: "python -m serve.detector fetch --out data/models" } };
    const base = vi.fn(async (u: RequestInfo | URL) =>
      String(u).startsWith("/offline/")
        ? new Response("<html>", { headers: { "content-type": "text/html" } })
        : new Response(JSON.stringify(body), { status: 503, headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
    const r = await withOfflineFallback(base)("/api/clips/upload-1/first-frame.jpg");
    expect(r.status).toBe(503);
    expect((await r.json()).error.code).toBe("model_missing");
    expect(base).toHaveBeenCalledTimes(1);
    expect(isOffline()).toBe(false);
  });
  it("falls back when a static host returns the SPA html for /api (200)", async () => {
    const base = vi.fn(async (u: RequestInfo | URL) =>
      String(u).startsWith("/offline/")
        ? new Response('{"status":"ok","clips":6}', { headers: { "content-type": "application/json" } })
        : new Response("<!doctype html><title>Gate tracker</title>", {
            status: 200,
            headers: { "content-type": "text/html; charset=utf-8" },
          })) as unknown as typeof fetch;
    const r = await withOfflineFallback(base)("/api/health");
    expect(await r.json()).toEqual({ status: "ok", clips: 6 });
    expect(isOffline()).toBe(true);
    expect(base).toHaveBeenCalledWith("/offline/health.json");
  });
  it("reports a clear 503 when there is no snapshot (SPA html)", async () => {
    const base = (async (u: RequestInfo | URL) =>
      String(u).startsWith("/offline/") ? new Response("<html>", { headers: { "content-type": "text/html" } }) : new Response("", { status: 500 })) as typeof fetch;
    const r = await withOfflineFallback(base)("/api/nothing");
    expect(r.status).toBe(503);
    expect((await r.json()).error.code).toBe("offline");
  });
});
