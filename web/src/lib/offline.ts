// Offline fallback: when the FastAPI service is unreachable (fresh clone, static hosting) GET /api/*
// requests are answered from snapshots of the same routes over committed repo data
// (scripts/web/export_offline.py → public/offline/, viewer/data → offline/viewer-clips/).
// Nothing is invented: a route without a snapshot keeps its original error.

const ROOT = `${import.meta.env?.BASE_URL ?? "/"}offline/`;

/** Map an /api GET path to its committed snapshot, or null when there is none. */
export function offlinePath(path: string): string | null {
  const p = path.split("?")[0];
  if (!p.startsWith("/api/")) return null;
  const clip = /^\/api\/viewer-clips\/([\w.-]+\.json\.gz)$/.exec(p);
  if (clip) return `${ROOT}viewer-clips/${clip[1]}`;
  const video = /^\/api\/clips\/([\w-]+)\/video$/.exec(p);
  if (video) return `${import.meta.env?.BASE_URL ?? "/"}demo/${video[1]}.mp4`;
  const poster = /^\/api\/clips\/([\w-]+)\/poster\.jpg$/.exec(p);
  if (poster) return `${import.meta.env?.BASE_URL ?? "/"}thumbs/${poster[1]}.jpg`;
  const plot = /^\/api\/benchmarks\/plots\/([\w-]+\.png)$/.exec(p);
  if (plot) return `${ROOT}plots/${plot[1]}`;
  if (p.startsWith("/api/jobs/")) return null; // a job only exists on a running service
  return `${ROOT}${p.slice(5).replace(/\//g, "__")}.json`;
}

type Listener = (offline: boolean) => void;
let offline = false;
const listeners = new Set<Listener>();
export const isOffline = () => offline;
export function onOffline(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
function setOffline(v: boolean) {
  if (v === offline) return;
  offline = v;
  listeners.forEach((l) => l(v));
}

const urlOf = (input: RequestInfo | URL) => (typeof input === "string" ? input : input instanceof URL ? input.href : input.url);

/** True when a 5xx carries the service's own JSON error envelope ({error: {code}}). */
async function isServiceError(res: Response): Promise<boolean> {
  if (!(res.headers.get("content-type") ?? "").includes("application/json")) return false;
  try {
    const b = (await res.clone().json()) as { error?: { code?: unknown } };
    return typeof b?.error?.code === "string";
  } catch {
    return false;
  }
}

/** Wrap fetch so failed (network error, 5xx, 502 proxy) GET /api calls fall back to snapshots. */
export function withOfflineFallback(base: typeof fetch): typeof fetch {
  return async (input, init) => {
    const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
    const raw = urlOf(input);
    const path = raw.startsWith("http") ? new URL(raw).pathname + new URL(raw).search : raw;
    const alt = method === "GET" ? offlinePath(path) : null;
    if (!alt) return base(input, init);
    try {
      const res = await base(input, init);
      if (res.status < 500) {
        if (res.ok) setOffline(false);
        return res;
      }
      // a running service answering with its own error (e.g. 503 model_missing) is not "offline"
      if (await isServiceError(res)) {
        setOffline(false);
        return res;
      }
    } catch {
      /* unreachable: fall through */
    }
    const snap = await base(alt);
    const type = snap.headers.get("content-type") ?? "";
    // SPA hosts answer unknown paths with index.html; treat that as "no snapshot"
    if (!snap.ok || (alt.endsWith(".json") && type.includes("text/html"))) {
      return new Response(JSON.stringify({ error: { code: "offline", message: "The service is offline and this view has no committed snapshot." } }), {
        status: 503,
        headers: { "content-type": "application/json" },
      });
    }
    setOffline(true);
    return snap;
  };
}

export function installOfflineFallback() {
  if (typeof window === "undefined" || (window.fetch as { __offline?: boolean }).__offline) return;
  const wrapped = withOfflineFallback(window.fetch.bind(window)) as typeof fetch & { __offline?: boolean };
  wrapped.__offline = true;
  window.fetch = wrapped;
}
