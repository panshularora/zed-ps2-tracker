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
