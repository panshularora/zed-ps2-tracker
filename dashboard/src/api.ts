import { loadClipGz, parseClip, type IndexedClip } from "@viewer/data";

export type Screen = "watch" | "compare" | "evidence" | "try";

const LEGACY: Record<string, Screen> = {
  studio: "watch",
  analytics: "evidence",
  experiments: "try",
};

export interface ClipRef {
  file: string;
  clip: string;
  tracker: string;
}

export interface AblationStressBin {
  bin: string;
  id_recall?: number;
  idsw_per_1k_gt?: number;
  idf1?: number;
  recall?: number;
}

export interface AblationRow {
  name: string;
  label: string;
  group: string;
  HOTA: number;
  IDF1: number;
  MOTA: number;
  IDSW: number;
  AssA: number;
  DetA: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
  stress?: {
    visibility?: AblationStressBin[];
    density?: AblationStressBin[];
  };
}

export interface GateRow {
  gt_crossings: number;
  trk_crossings: number;
  matched: number;
  count_error: number;
  precision: number;
  recall: number;
  f1: number;
  attribution_error: number;
  gt_tailgates: number;
  trk_tailgates: number;
  tailgate_matched: number;
  tailgate_precision: number;
  tailgate_recall: number;
}

export interface GoNoGoRow {
  condition: string;
  K: number;
  K_ms_mean: number;
  HOTA: number;
  IDF1: number;
  MOTA: number;
  IDSW_trackeval: number;
  p95_ms: number;
  max_ms: number;
  changed_decisions: number;
}

export interface OutputRun {
  split: string;
  tracker: string;
  sequences: string[];
  combined: Record<string, number> | null;
}

export interface TrackResponse {
  sequence: string;
  config: string;
  events: Record<string, unknown>[];
  latency: { frames: number; p50_ms: number | null; p95_ms: number | null; machine: string };
  clip: unknown;
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json() as Promise<T>;
}

export async function fetchHealth() {
  return getJson<{
    status: string;
    version: string;
    configs: number;
    viewer: boolean;
    dashboard: boolean;
  }>("/api/health");
}

export async function fetchDashboard() {
  return getJson<{
    status: string;
    version: string;
    configs: number;
    clips: number;
    results: { ablation: boolean; gate_events: boolean; go_no_go: boolean };
  }>("/api/dashboard");
}

export async function fetchConfigs(): Promise<string[]> {
  return getJson("/api/configs");
}

export async function fetchConfig(name: string) {
  return getJson<{ name: string; path: string; config: Record<string, unknown> }>(
    `/api/config/${name}`,
  );
}

export async function fetchClips(): Promise<ClipRef[]> {
  const body = await getJson<{ clips: ClipRef[] }>("/api/clips");
  return body.clips;
}

export async function fetchClip(file: string): Promise<IndexedClip> {
  const r = await fetch(`/api/clips/${encodeURIComponent(file)}`);
  if (!r.ok) throw new Error(`clip ${file} → ${r.status}`);
  return loadClipGz(new Uint8Array(await r.arrayBuffer()));
}

export async function fetchAblation(): Promise<AblationRow[]> {
  return getJson("/api/results/ablation");
}

export async function fetchGateEvents(): Promise<Record<string, GateRow>> {
  return getJson("/api/results/gate-events");
}

export async function fetchGoNoGo(): Promise<GoNoGoRow[]> {
  return getJson("/api/results/go-no-go");
}

export async function fetchOutputs(): Promise<OutputRun[]> {
  const body = await getJson<{ runs: OutputRun[] }>("/api/outputs");
  return body.runs;
}

export async function runTrack(
  cache: File,
  config: string,
  gate?: string,
  startTime?: string,
): Promise<{ body: TrackResponse; clip: IndexedClip }> {
  const fd = new FormData();
  fd.append("cache", cache);
  fd.append("config", config);
  if (gate) fd.append("gate", gate);
  if (startTime) fd.append("start_time", startTime);
  const r = await fetch("/api/track", { method: "POST", body: fd });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      const err = (await r.json()) as { detail?: string };
      if (err.detail) detail = String(err.detail);
    } catch {
      /* keep status */
    }
    throw new Error(detail);
  }
  const body = (await r.json()) as TrackResponse;
  return { body, clip: parseClip(body.clip) };
}

export function parseHash(): { screen: Screen; params: URLSearchParams } {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  const mapped = LEGACY[path] ?? path;
  const screen = (["watch", "compare", "evidence", "try"].includes(mapped)
    ? mapped
    : "watch") as Screen;
  return { screen, params: new URLSearchParams(query ?? "") };
}

export function writeHash(screen: Screen, params?: URLSearchParams) {
  const q = params?.toString();
  const next = `#/${screen}${q ? `?${q}` : ""}`;
  if (window.location.hash !== next) window.location.hash = next;
}
