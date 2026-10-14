// Client for the synthetic gate scene (serve/gate_sim.py): NDJSON stream of real tracker output.
export type PaneKey = "bytetrack" | "ours";
export const PANES: PaneKey[] = ["bytetrack", "ours"];

export interface SimMeta {
  type: "meta";
  width: number;
  height: number;
  gate: [[number, number], [number, number]];
  fps: number;
  params: { seed: number; density: number; occlusion_s: number; noise: number; speed: number; seconds: number; fps: number };
  trackers: Record<PaneKey, { label: string; config: string }>;
  max_people: number;
}
/** [person id, from tracker id, to tracker id, x, y] */
export type IdEvent = [number, number, number, number, number];
export interface PaneFrame { ids: [number, number][]; extra: [number, number, number][]; events: IdEvent[]; counted: number; id_changes: number }
export interface SimFrame {
  type: "frame";
  f: number;
  /** [person id, x, y, visible 0/1] (exact ground truth) */
  people: [number, number, number, number][];
  dets: number[][];
  true_count: number;
  bytetrack: PaneFrame;
  ours: PaneFrame;
}
export interface SimEnd { type: "end"; frames: number; true_count: number; summary: Record<PaneKey, { id_changes: number; counted: number; truth: number }> }
export type SimLine = SimMeta | SimFrame | SimEnd;

export interface SimControls { seed: number; density: number; occlusion: number; noise: number }
export const DEFAULTS: SimControls = { seed: 7, density: 0.5, occlusion: 1.0, noise: 0.3 };
export const RECORDED_SECONDS = 30;
export const isDefault = (c: SimControls) => c.seed === DEFAULTS.seed && c.density === DEFAULTS.density && c.occlusion === DEFAULTS.occlusion && c.noise === DEFAULTS.noise;

export function streamUrl(c: SimControls, seconds = 60): string {
  const q = new URLSearchParams({ seed: String(c.seed), density: String(c.density), occlusion: String(c.occlusion), noise: String(c.noise), seconds: String(seconds) });
  return `/api/gate-sim/stream?${q}`;
}
export const RECORDING_URL = `${import.meta.env?.BASE_URL ?? "/"}offline/gate-sim.ndjson`;

/** Feed text chunks; returns complete parsed lines, keeps the partial tail. */
export function lineSplitter(): (chunk: string, flush?: boolean) => SimLine[] {
  let rest = "";
  return (chunk, flush = false) => {
    rest += chunk;
    const parts = rest.split("\n");
    rest = flush ? "" : parts.pop() ?? "";
    return parts.filter((x) => x.trim()).map((x) => JSON.parse(x) as SimLine);
  };
}

/** Read an NDJSON response incrementally, calling onLine for each parsed line. */
export async function readStream(res: Response, onLine: (l: SimLine) => void, signal?: AbortSignal): Promise<void> {
  const split = lineSplitter();
  const body = res.body;
  if (!body || typeof body.getReader !== "function") {
    for (const l of split(await res.text(), true)) onLine(l);
    return;
  }
  const reader = body.getReader();
  const dec = new TextDecoder();
  for (;;) {
    if (signal?.aborted) { await reader.cancel().catch(() => undefined); return; }
    const { done, value } = await reader.read();
    if (done) break;
    for (const l of split(dec.decode(value, { stream: true }))) onLine(l);
  }
  for (const l of split(dec.decode(), true)) onLine(l);
}

/** Person ids involved in an ID change in either pane between frames [from, to]. */
export function involvedBetween(frames: SimFrame[], from: number, to: number): Set<number> {
  const s = new Set<number>();
  for (const fr of frames) if (fr.f >= from && fr.f <= to) for (const k of PANES) for (const e of fr[k].events) s.add(e[0]);
  return s;
}

/** Latest ID-change event in a pane within `windowF` frames before f (inclusive), with its age. */
export function recentEvents(frames: SimFrame[], idx: number, pane: PaneKey, windowF: number): { e: IdEvent; age: number }[] {
  const out: { e: IdEvent; age: number }[] = [];
  for (let i = Math.max(0, idx - windowF); i <= idx && i < frames.length; i++) for (const e of frames[i][pane].events) out.push({ e, age: idx - i });
  return out;
}
