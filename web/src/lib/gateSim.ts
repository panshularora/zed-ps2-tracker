// Client for the synthetic gate scene (serve/gate_sim.py): NDJSON stream of real tracker output.
export type PaneKey = "bytetrack" | "ours";
export const PANES: PaneKey[] = ["bytetrack", "ours"];

export interface SimMeta {
  type: "meta";
  width: number;
  height: number;
  gate: [[number, number], [number, number]];
  fps: number;
  params: { seed: number; density: number; occlusion_s: number; noise: number; speed: number; seconds: number; fps: number; play?: string };
  trackers: Record<PaneKey, { label: string; config: string }>;
  max_people: number;
  play?: string;
  /** keep play: the simulated (ground-truth) person the scene is about */
  focus?: number | null;
}
/** [person id, from tracker id, to tracker id, x, y] */
export type IdEvent = [number, number, number, number, number];
export interface PaneFrame { ids: [number, number][]; extra: [number, number, number][]; held?: [number, number, number][]; events: IdEvent[]; counted: number; id_changes: number }
export type Beat = "hide" | "back";
export interface SimFrame {
  type: "frame";
  f: number;
  /** [person id, x, y, visible 0/1] (exact ground truth) */
  people: [number, number, number, number][];
  dets: number[][];
  true_count: number;
  beat?: Beat | null;
  focus?: number | null;
  bytetrack: PaneFrame;
  ours: PaneFrame;
}
export interface SimEnd { type: "end"; frames: number; true_count: number; summary: Record<PaneKey, { id_changes: number; counted: number; truth: number }> }
export type SimLine = SimMeta | SimFrame | SimEnd;

export type PlayKind = "keep" | "crowd";
export interface SimControls { seed: number; density: number; occlusion: number; noise: number; play: PlayKind }
export const DEFAULTS: SimControls = { seed: 7, density: 0.5, occlusion: 1.5, noise: 0.15, play: "keep" };
export const RECORDED_SECONDS = 12;
export const isDefault = (c: SimControls) => c.seed === DEFAULTS.seed && c.density === DEFAULTS.density && c.occlusion === DEFAULTS.occlusion && c.noise === DEFAULTS.noise && c.play === DEFAULTS.play;

export function streamUrl(c: SimControls, seconds?: number): string {
  const sec = seconds ?? (c.play === "keep" ? 12 : 60);
  const q = new URLSearchParams({ seed: String(c.seed), density: String(c.density), occlusion: String(c.occlusion), noise: String(c.noise), seconds: String(sec), play: c.play });
  return `/api/gate-sim/stream?${q}`;
}
export function recordingUrl(play: PlayKind = "keep"): string {
  const file = play === "keep" ? "gate-keep.ndjson" : "gate-sim.ndjson";
  return `${import.meta.env?.BASE_URL ?? "/"}offline/${file}`;
}
export const RECORDING_URL = recordingUrl("keep");

export function beatCopy(beat: Beat, pid: number): { tag: string; body: string } {
  if (beat === "hide") {
    return {
      tag: `Look here · person ${pid} walked behind the other`,
      body: "The detector dropped their box. Amber HELD means the tracker still has their number. Watch both chips — they should stay the same number.",
    };
  }
  return {
    tag: `Look here · person ${pid} came back`,
    body: "Same body. ID with the first number = kept; NEW with a different number = the tracker forgot them.",
  };
}

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

/** First tracker number this person was given in `pane` — the ID that should be kept. */
export function firstKeepId(frames: SimFrame[], idx: number, pane: PaneKey, pid: number): number | null {
  for (let i = 0; i <= idx && i < frames.length; i++) {
    const tid = new Map(frames[i][pane].ids).get(pid);
    if (tid != null) return tid;
  }
  return null;
}

export type IdKind = "ok" | "held" | "fail" | "none";

/** On-canvas badge: ID #N (visible, same number), HELD #N (hidden, remembering), NEW #M was #N (forgot). */
export function personIdLabel(vis: boolean, tid: number | null | undefined, keep: number | null, switchedTo?: number | null): { text: string; kind: IdKind } {
  const now = switchedTo ?? tid ?? null;
  if (now != null && keep != null && now !== keep) return { text: `NEW #${now}  was #${keep}`, kind: "fail" };
  if (!vis) return keep != null ? { text: `HELD #${keep}`, kind: "held" } : { text: "hidden · no ID", kind: "held" };
  if (now != null) return { text: `ID #${now}`, kind: "ok" };
  return { text: "no ID", kind: "none" };
}

/**
 * How long person ``pid`` was hidden, measured from the frames themselves: the longest completed
 * run of frames with the ground-truth "visible" flag at 0, divided by fps. ``null`` until that
 * person has been hidden and seen again (so the number is never a guess).
 */
export function measuredHideS(frames: SimFrame[], pid: number, fps: number): number | null {
  let run = 0;
  let best = 0;
  for (const fr of frames) {
    const p = fr.people.find((q) => q[0] === pid);
    if (p && p[3] === 0) run += 1;
    else {
      if (p && run > best) best = run;
      run = 0;
    }
  }
  return best > 0 && fps > 0 ? best / fps : null;
}
