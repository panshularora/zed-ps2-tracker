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
  /** true crossings this tracker did not count: [frame, person, direction] */
  missed?: [number, number, number][];
  /** counted crossings that match no person's real crossing */
  extra?: number;
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
  true_crossings_by?: [number, number, number][];
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

/** Crop of fixed zoom centred on person pid at frame f, NOT clamped: past the scene edge the
 * pane shows padding, so the camera never moves off the person and their neighbours stay in view. */
export function cropAt(d: SceneData, pid: number | null, f: number, zoom = 1.5): Crop {
  const p = pid == null ? null : personAt(d, pid, f);
  const [cx, cy] = p ? [p[0], p[1] - personHeight(p[1]) / 2] : [d.width / 2, d.gate_y];
  const w = d.width / zoom, h = d.height / zoom;
  return { x: cx - w / 2, y: cy - h / 2, w, h };
}

/** Break-it pane crop: a box around every walking path (whole bodies), the pillars and the gate
 * line, plus 10 % padding, widened to the scene's aspect ratio and capped at the whole scene. */
export function cropWalkingArea(d: SceneData, pad = 0.1): Crop {
  let x0 = Infinity, x1 = -Infinity, y0 = d.gate_y, y1 = d.gate_y;
  for (const t of d.truth) for (const [, x, y] of t.p) {
    const h = personHeight(y);
    x0 = Math.min(x0, x - h * 0.3); x1 = Math.max(x1, x + h * 0.3); y0 = Math.min(y0, y - h); y1 = Math.max(y1, y);
  }
  for (const [px, py, pw, ph] of d.pillars ?? []) { x0 = Math.min(x0, px); x1 = Math.max(x1, px + pw); y0 = Math.min(y0, py); y1 = Math.max(y1, py + ph); }
  if (!Number.isFinite(x0)) return { x: 0, y: 0, w: d.width, h: d.height };
  const pw = (x1 - x0) * (1 + 2 * pad), ph = (y1 - y0) * (1 + 2 * pad);
  const aspect = d.width / d.height;
  // never wider than the scene itself: a crop past that would only shrink the figures
  const w = Math.min(d.width, Math.max(pw, ph * aspect)), h = w / aspect;
  if (w >= d.width) return { x: 0, y: 0, w: d.width, h: d.height };
  return { x: (x0 + x1) / 2 - w / 2, y: (y0 + y1) / 2 - h / 2, w, h };
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

const ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"];
const PANE_NAME: Record<PaneKey, string> = { bytetrack: "ByteTrack", ours: "Ours" };

/** 'Whole scene: counted N of M'. */
export const wholeSceneText = (p: Pick<Pane, "count" | "true_count">) => `Whole scene: counted ${p.count} of ${p.true_count}`;

/**
 * One line under a pane's verdict: how many real crossings this tracker lost over the whole scene,
 * and where the focus person's crossing (the one nearest frame `at`) sits among them.
 */
export function lostLine(d: SceneData, pane: PaneKey, focus: number | null, at: number): string | null {
  const p = d.panes[pane];
  if (!p.missed) return null;
  const name = PANE_NAME[pane];
  const missed = [...p.missed].sort((a, b) => a[0] - b[0]);
  const extra = p.extra ?? 0;
  const tail = extra ? ` ${extra} of its counts ${extra === 1 ? "matches" : "match"} no one's real crossing.` : "";
  const mine = focus == null ? [] : (d.true_crossings_by ?? []).filter((c) => c[1] === focus);
  const fc = mine.length ? mine.reduce((a, b) => (Math.abs(b[0] - at) < Math.abs(a[0] - at) ? b : a)) : null;
  if (!missed.length) return `${name} counted every real crossing in the scene.${tail}`;
  const n = missed.length;
  let where = ".";
  if (fc) {
    const k = missed.findIndex((m) => m[0] === fc[0] && m[1] === fc[1]);
    where = k >= 0 ? `; this is the ${ORD[k] ?? `#${k + 1}`}.` : "; it counted this one.";
  }
  return `${name} lost ${n} crossing${n === 1 ? "" : "s"} across the scene${where}${tail}`;
}

/** People other than `focus` whose figure overlaps the crop at frame f. */
export function othersInView(d: Pick<SceneData, "truth">, f: number, crop: { x: number; y: number; w: number; h: number }, focus: number): number {
  const t = d.truth[f - 1];
  if (!t) return 0;
  return t.p.filter(([pid, x, y]) => {
    if (pid === focus) return false;
    const h = personHeight(y), w = h * 0.4;
    return x + w / 2 >= crop.x && x - w / 2 <= crop.x + crop.w && y >= crop.y && y - h <= crop.y + crop.h;
  }).length;
}

/** Caption under the paused cell replay, so the sentence matches what is drawn. */
export const pausedNote = (focus: number, others: number) =>
  others === 0 ? `Paused: person ${focus} ringed; nobody else is in this view.` : `Paused: person ${focus} ringed; the other ${others} ${others === 1 ? "person in view is" : "people in view are"} dimmed.`;
