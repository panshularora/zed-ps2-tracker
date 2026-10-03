// Clip format (Web Experience Analyst spec): one <clip>.<tracker>.json.gz per clip per tracker.
export type Box = [number, number, number, number, number, number]; // frame, x, y, w, h, conf

export interface Pending {
  f0: number;
  f1: number;
  candidates: number[];
  chosen: number;
}

export type ClipEvent =
  | { f: number; type: "id_switch"; gt: number; from: number; to: number }
  | { f: number; type: "cross"; id: number; dir: "in" | "out"; correct: boolean }
  | { f: number; type: "tailgate"; ids: [number, number] };

export interface Clip {
  clip: string;
  tracker: string;
  fps: number;
  frames: number;
  width: number;
  height: number;
  gate: { a: [number, number]; b: [number, number] } | null;
  tracks: Record<string, Box[]>;
  pending: Pending[];
  events: ClipEvent[];
}

export interface FrameBox {
  id: number;
  x: number;
  y: number;
  w: number;
  h: number;
  conf: number;
}

export interface IndexedClip extends Clip {
  byFrame: Map<number, FrameBox[]>;
}

function fail(msg: string): never {
  throw new Error(`invalid clip: ${msg}`);
}

/** Validate a parsed JSON object against the clip format and index boxes by frame. */
export function parseClip(raw: unknown): IndexedClip {
  if (typeof raw !== "object" || raw === null) fail("not an object");
  const o = raw as Record<string, unknown>;
  for (const k of ["clip", "tracker"]) if (typeof o[k] !== "string") fail(`${k} must be a string`);
  for (const k of ["fps", "frames", "width", "height"])
    if (typeof o[k] !== "number" || !((o[k] as number) > 0)) fail(`${k} must be a positive number`);
  if (typeof o.tracks !== "object" || o.tracks === null) fail("tracks must be an object");
  if (!Array.isArray(o.pending)) fail("pending must be an array");
  if (!Array.isArray(o.events)) fail("events must be an array");
  const clip = o as unknown as Clip;
  const byFrame = new Map<number, FrameBox[]>();
  for (const [id, rows] of Object.entries(clip.tracks)) {
    if (!Array.isArray(rows)) fail(`track ${id} is not an array`);
    for (const r of rows) {
      if (!Array.isArray(r) || r.length !== 6) fail(`track ${id} row must be [frame,x,y,w,h,conf]`);
      const f = r[0];
      if (!Number.isInteger(f) || f < 1 || f > clip.frames) fail(`track ${id} frame ${f} out of range`);
      const list = byFrame.get(f) ?? [];
      list.push({ id: Number(id), x: r[1], y: r[2], w: r[3], h: r[4], conf: r[5] });
      byFrame.set(f, list);
    }
  }
  for (const e of clip.events) {
    if (!["id_switch", "cross", "tailgate"].includes((e as ClipEvent).type)) fail("unknown event type");
  }
  return { ...clip, gate: clip.gate ?? null, byFrame };
}

/** gunzip with the browser's DecompressionStream (also in Node >= 18). */
export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function loadClipGz(bytes: Uint8Array): Promise<IndexedClip> {
  const text = new TextDecoder().decode(await gunzip(bytes));
  return parseClip(JSON.parse(text));
}

/** Clicking a timeline mark jumps to 2 s before the event (spec), clamped to the clip. */
export function seekFrameForEvent(f: number, fps: number, frames: number): number {
  return Math.min(frames, Math.max(1, f - Math.round(2 * fps)));
}

export function clampFrame(f: number, frames: number): number {
  return Math.min(frames, Math.max(1, Math.round(f)));
}

/** In/out gate counts over events up to and including frame f. */
export function gateCounts(events: ClipEvent[], f: number): { in: number; out: number } {
  let i = 0;
  let o = 0;
  for (const e of events) {
    if (e.type === "cross" && e.f <= f) {
      if (e.dir === "in") i++;
      else o++;
    }
  }
  return { in: i, out: o };
}

/** Pending decisions open at frame f (f0 <= f <= f1). */
export function openPending(p: Pending[], f: number): Pending[] {
  return p.filter((d) => d.f0 <= f && f <= d.f1);
}
