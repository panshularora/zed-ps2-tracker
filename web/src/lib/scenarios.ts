// Pre-computed Scenarios overlays (scripts/demo/precompute_scenarios.py → web/public/scenarios).
export type Box = [id: number, x: number, y: number, w: number, h: number];
export type RunKey = "baseline" | "ours";

export interface ScenarioMeta {
  id: string;
  title: string;
  problem: string;
  caption: string;
  source: string;
  licence: string;
  detector: string;
  kind: "illustrative" | "measured";
  width: number;
  height: number;
  src_fps: number;
  stride: number;
  frames_tracked: number;
  covered_s: number;
  id_switches: Record<RunKey, number>;
  tracks: Record<RunKey, number>;
  disagreements: number;
}

export interface Disagreement {
  frame: number;
  kept: RunKey;
  box: [number, number, number, number];
  ours_id?: number;
  baseline_from?: number;
  baseline_to?: number;
  baseline_id?: number;
  ours_from?: number;
  ours_to?: number;
}

export interface ScenarioData {
  meta: ScenarioMeta;
  runs: Record<RunKey, { label: string; config: string; id_switches: number; tracks: number; frames: Record<string, Box[]> }>;
  disagreements: Disagreement[];
}

/** Benchmark slots: MOT17/MOT20 val sequences with human labels, data pending. */
export interface BenchmarkSlot {
  id: string;
  title: string;
  problem: string;
  licence: string;
}
export const BENCHMARK_SLOTS: BenchmarkSlot[] = [
  { id: "mot17_val", title: "MOT17 val sequence", problem: "Measured ID swaps, street crowd", licence: "MOTChallenge: non-commercial use, attribution (Milan et al. 2016)" },
  { id: "mot20_val", title: "MOT20 val sequence", problem: "Measured ID swaps, very dense crowd", licence: "MOTChallenge: non-commercial use, attribution (Dendorfer et al. 2020)" },
];

export const BASE = `${import.meta.env?.BASE_URL ?? "/"}scenarios/`;

export async function fetchIndex(): Promise<ScenarioMeta[]> {
  const r = await fetch(`${BASE}index.json`);
  if (!r.ok) throw new Error(`scenarios index: HTTP ${r.status}`);
  return r.json();
}
export async function fetchScenario(id: string): Promise<ScenarioData> {
  const r = await fetch(`${BASE}${id}.json`);
  if (!r.ok) throw new Error(`scenario ${id}: HTTP ${r.status}`);
  return r.json();
}

export const videoUrl = (id: string) => `/api/clips/${id}/video`;
export const posterUrl = (id: string) => `/api/clips/${id}/poster.jpg`;

/** Tracked frame (1-based) shown at video time t; null outside the tracked range. */
export function frameAt(meta: ScenarioMeta, t: number): number | null {
  const k = Math.floor((t * meta.src_fps) / meta.stride + 1e-6) + 1;
  return k >= 1 && k <= meta.frames_tracked ? k : null;
}
export function timeOf(meta: ScenarioMeta, frame: number): number {
  return ((frame - 1) * meta.stride) / meta.src_fps;
}

/** Seek target for a marker: 2 s before it, clamped at 0. */
export function seekBefore(meta: ScenarioMeta, frame: number): number {
  return Math.max(0, timeOf(meta, frame) - 2);
}

/** Merge disagreements closer than `gapS` seconds into one marker. */
export function markers(data: ScenarioData, gapS = 1): Disagreement[] {
  const out: Disagreement[] = [];
  for (const d of data.disagreements) {
    const last = out[out.length - 1];
    if (last && timeOf(data.meta, d.frame) - timeOf(data.meta, last.frame) < gapS) continue;
    out.push(d);
  }
  return out;
}

/** Label shown at a marker: neutral for unlabeled Pexels clips; benchmark clips may claim a swap. */
export function markerLabel(kind: ScenarioMeta["kind"], d: Disagreement): string {
  if (kind === "illustrative") return "Trackers disagree here";
  if (d.kept === "ours") return `ByteTrack swapped ${d.baseline_from} → ${d.baseline_to} · ours kept ${d.baseline_from}`;
  return "Trackers disagree here";
}

// Neutral ID palette for Pexels footage: no reds/oranges (no error connotation on strangers).
export const NEUTRAL_IDS = ["#56B4E9", "#9AD1F0", "#009E73", "#8DD3C7", "#F0E442", "#BEBADA", "#80B1D3", "#B3DE69", "#CCEBC5", "#A6CEE3", "#E6E8EB", "#CC79A7"];
export const neutralColor = (id: number) => NEUTRAL_IDS[((id % NEUTRAL_IDS.length) + NEUTRAL_IDS.length) % NEUTRAL_IDS.length];

export function drawBoxes(ctx: CanvasRenderingContext2D, boxes: Box[], sx: number, sy: number, ring?: [number, number, number, number] | null) {
  ctx.lineWidth = 2;
  ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
  for (const [id, x, y, w, h] of boxes) {
    const c = neutralColor(id);
    ctx.strokeStyle = c;
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    const label = String(id);
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = c;
    ctx.fillRect(x * sx - 1, y * sy - 16, tw, 16);
    ctx.fillStyle = "#0E1013";
    ctx.fillText(label, x * sx + 3, y * sy - 4);
  }
  if (ring) {
    const [x, y, w, h] = ring;
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse((x + w / 2) * sx, (y + h / 2) * sy, (w * sx) / 2 + 10, (h * sy) / 2 + 10, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
}
