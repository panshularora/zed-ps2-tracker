// Replay data from /api/stress/replay and /api/break/run, and the verdict logic shown on it.
export type PaneKey = "bytetrack" | "ours";
export const PANE_ORDER: PaneKey[] = ["bytetrack", "ours"]; // ByteTrack left, ours right

export interface IdChange { frame: number; pid: number; from: number; to: number; reason: string; line: string; why: string }
export interface Pane {
  label: string;
  config: string;
  /** per frame: [track id, matched person or -1, foot x, foot y] */
  tracks: [number, number, number, number][][];
  matched: Record<string, number>[];
  events: IdChange[];
  crossings: number[];
  count: number;
  true_count: number;
}
export interface TruthFrame { f: number; p: [number, number, number, number][]; hidden: number[] }
export interface SceneData {
  seed: number;
  settings: Record<string, unknown>;
  width: number;
  height: number;
  gate_y: number;
  fps: number;
  truth: TruthFrame[];
  true_crossings: number[];
  panes: Record<PaneKey, Pane>;
  pillars?: [number, number, number, number][];
  people?: { pid: number; kind: string }[];
}

export const personHeight = (footY: number) => 46 + 0.16 * Math.max(0, footY);

/** Counts up to and including frame f (1-based). */
export function countsAt(d: SceneData, f: number): { truth: number } & Record<PaneKey, number> {
  const upto = (xs: number[]) => xs.filter((x) => x <= f).length;
  return { truth: upto(d.true_crossings), bytetrack: upto(d.panes.bytetrack.crossings), ours: upto(d.panes.ours.crossings) };
}

export type Verdict =
  | { kind: "kept"; id: number; frame: number; text: string }
  | { kind: "changed"; from: number; to: number; frame: number; text: string }
  | { kind: "lost"; from: number; frame: number; text: string };

/** Hidden window [first, last] (1-based frames) of person pid, or null. */
export function hideWindow(d: SceneData, pid: number): [number, number] | null {
  const fs = d.truth.filter((t) => t.hidden.includes(pid)).map((t) => t.f);
  return fs.length ? [fs[0], fs[fs.length - 1]] : null;
}

/**
 * What a tracker did with person pid across their hide: the number they had just before it, the
 * number they get when they come back. Kept = same number; changed = '#a → #b'; lost = no number
 * within `lookS` seconds of coming back on screen. Null when there is no hide or no number before.
 */
export function verdict(d: SceneData, pane: PaneKey, pid: number, lookS = 3): Verdict | null {
  const w = hideWindow(d, pid);
  if (!w) return null;
  const m = d.panes[pane].matched;
  const key = String(pid);
  let before: number | null = null;
  for (let k = w[0] - 2; k >= 0; k--) if (m[k]?.[key] != null) { before = m[k][key]; break; }
  if (before == null) return null;
  const lastF = Math.min(m.length, w[1] + Math.round(lookS * d.fps));
  for (let k = w[1]; k < lastF; k++) {
    const t = m[k]?.[key];
    if (t == null) continue;
    const frame = k + 1;
    return t === before
      ? { kind: "kept", id: t, frame, text: `Kept #${t}` }
      : { kind: "changed", from: before, to: t, frame, text: `#${before} \u2192 #${t}` };
  }
  return { kind: "lost", from: before, frame: lastF, text: `#${before} \u2192 no number` };
}

/** Latest ID change in a pane at or before frame f (within `windowS`), optionally for one person. */
export function latestChange(d: SceneData, pane: PaneKey, f: number, windowS = 4, pid?: number): IdChange | null {
  const lo = f - windowS * d.fps;
  const ev = d.panes[pane].events.filter((e) => e.frame <= f && e.frame >= lo && (pid == null || e.pid === pid));
  return ev.length ? ev[ev.length - 1] : null;
}

export const whyQuestion = (e: IdChange) => `Why did #${e.from} become #${e.to}?`;

/** The part of the scene a pane shows, in scene pixels, always the scene's aspect ratio. */
export interface Crop { x: number; y: number; w: number; h: number }

const clampCrop = (d: Pick<SceneData, "width" | "height">, cx: number, cy: number, w: number): Crop => {
  const ww = Math.min(d.width, Math.max(40, w));
  const hh = (ww * d.height) / d.width;
  return {
    x: Math.min(d.width - ww, Math.max(0, cx - ww / 2)),
    y: Math.min(d.height - hh, Math.max(0, cy - hh / 2)),
    w: ww,
    h: hh,
  };
};

/** Where person pid is at frame f (1-based), or at the nearest frame they are in the scene. */
export function personAt(d: SceneData, pid: number, f: number): [number, number] | null {
  for (let k = 0; k < d.truth.length; k++) {
    for (const s of [-1, 1]) {
      const p = d.truth[f - 1 + s * k]?.p.find((q) => q[0] === pid);
      if (p) return [p[1], p[2]];
    }
  }
  return null;
}

/** Zoom `zoom`× and follow person pid (centre of their body), clamped to the scene. */
export function cropFollow(d: SceneData, pid: number | null, f: number, zoom = 2): Crop {
  const p = pid == null ? null : personAt(d, pid, f);
  const [cx, cy] = p ? [p[0], p[1] - personHeight(p[1]) / 2] : [d.width / 2, d.gate_y];
  return clampCrop(d, cx, cy, d.width / zoom);
}

/**
 * Break-it pane crop: the walking area (every person's path, the pillars, the gate line), cut down
 * just enough that a person standing on the gate line is at least `minFigPx` CSS pixels tall in a
 * pane `paneCssW` wide, centred on the hidden person's hide point.
 */
export function cropWalkingArea(d: SceneData, focus: number | null, paneCssW: number, minFigPx = 90): Crop {
  let x0 = Infinity, x1 = -Infinity, y0 = d.gate_y, y1 = d.gate_y;
  for (const t of d.truth) for (const [, x, y] of t.p) {
    x0 = Math.min(x0, x - 30); x1 = Math.max(x1, x + 30); y0 = Math.min(y0, y - personHeight(y)); y1 = Math.max(y1, y + 8);
  }
  for (const [px, py, pw, ph] of d.pillars ?? []) { x0 = Math.min(x0, px - 16); x1 = Math.max(x1, px + pw + 16); y0 = Math.min(y0, py - 16); y1 = Math.max(y1, py + ph + 16); }
  if (!Number.isFinite(x0)) return { x: 0, y: 0, w: d.width, h: d.height };
  const aspect = d.width / d.height;
  const fitW = Math.max(x1 - x0, (y1 - y0) * aspect);
  const maxW = (paneCssW * personHeight(d.gate_y)) / minFigPx;
  const w = Math.min(fitW, maxW);
  let cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (w < fitW && focus != null) {
    const win = hideWindow(d, focus);
    const p = win ? personAt(d, focus, Math.round((win[0] + win[1]) / 2)) : null;
    if (p) { cx = p[0]; cy = d.gate_y; }
  }
  return clampCrop(d, cx, cy, w);
}

/** One plain sentence for a replay: who was hidden and which tracker gave them a new number. */
export function replaySentence(people: number, pid: number, hideS: number | null, v: Partial<Record<PaneKey, Verdict | null>>, firstChanged?: string): string {
  const head = hideS != null ? `${people} people, person ${pid} hidden ${hideS} s.` : `${people} people, following person ${pid}.`;
  const did = (x: Verdict) => (x.kind === "kept" ? "kept their number" : x.kind === "changed" ? "gave them a new number" : "lost them");
  const b = v.bytetrack, o = v.ours;
  if (b && o) {
    if (b.kind === o.kind) return `${head} Both trackers ${did(b)}.`;
    if (o.kind === "kept") return `${head} ByteTrack ${did(b)}, ours didn't.`;
    if (b.kind === "kept") return `${head} Ours ${did(o)}, ByteTrack didn't.`;
    return `${head} ByteTrack ${did(b)}, ours ${did(o)}.`;
  }
  if (firstChanged) return `${head} ${firstChanged === "ours" ? "Ours" : "ByteTrack"} changed their number first.`;
  return head;
}
