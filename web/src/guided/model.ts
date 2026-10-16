// Guided replay: pure logic shared by the Scenarios player and the Studio player.
// Everything here is derived from precomputed tracker output (scripts/demo/guide.py or a GT
// viewer export). Wording rule: on "illustrative" clips (no ground truth) we only say what each
// tracker did and that they disagree; only "measured" moments (human labels) may say a number
// changed wrongly, and a tracker comparison only when the data shows the other one kept it.
import type { Box, RunKey, ScenarioData } from "../lib/scenarios";
import { collapseRenumbers, oursEvents } from "../identity/keepId";

export type Kind = "illustrative" | "measured";

export type GEvent =
  | { t: "cross"; f: number; id: number; dir: "in" | "out" }
  | { t: "lost"; f: number; id: number }
  | { t: "back"; f: number; id: number; gap_s: number }
  | { t: "renumber"; f: number; from: number; to: number };

export interface PersonIds { before: number | null; after: number | null }
export interface Moment {
  f: number;
  kind: "overlap" | "occlusion" | "gate";
  /** which tracker(s) changed this person's number here (the moment rule: at least one did) */
  changed: Record<RunKey, boolean>;
  people: { box: number[]; ids: Partial<Record<RunKey, PersonIds>> }[];
  focus: number[]; // x, y, w, h (video px)
  /** old and new number of that tracker both crossed the gate the same way */
  counted_twice: Record<RunKey, boolean>;
  near_gate?: boolean;
  /** measured clips only: the change was checked against human labels */
  verified?: boolean;
}

export interface ReplayData {
  id: string;
  title: string;
  kind: Kind;
  width: number;
  height: number;
  fps: number; // tracked frames per second of video
  frames: number; // tracked frames
  duration: number; // seconds of video covered by tracking
  videoUrl: string | null;
  /** offline demo cut (web/public/demo) tried when the full clip is not downloaded */
  demoUrl?: string | null;
  /** video time of tracked frame 1 (benchmark val-half clips start mid-sequence) */
  videoOffsetS?: number;
  runs: Record<RunKey, { label: string; short: string; frames: Record<string, Box[]> }>;
  gate: { a: [number, number]; b: [number, number] } | null;
  events: Record<RunKey, GEvent[]>;
  moments: Moment[];
  credit?: string;
  /** one-line framing shown under the title */
  caption?: string;
  /** Studio / Scenarios: ours IDs collapsed so a minted number keeps the first one. */
  keepId?: boolean;
}

export const SHORT: Record<RunKey, string> = { baseline: "ByteTrack", ours: "ours" };
export const NAMES: Record<RunKey, string> = { baseline: "ByteTrack", ours: "ours" };

interface Guide {
  gate: { a: [number, number]; b: [number, number] };
  events: Record<RunKey, GEvent[]>;
  moments: Moment[];
}

export const demoUrlFor = (id: string) => `${import.meta.env?.BASE_URL ?? "/"}demo/${id}.mp4`;
export const thumbUrlFor = (id: string) => `${import.meta.env?.BASE_URL ?? "/"}thumbs/${id}.jpg`;

export function fromScenario(
  d: ScenarioData & { guide?: Guide },
  videoUrl: string | null,
  demoUrl: string | null = demoUrlFor(d.meta.id),
  opts?: { keepId?: boolean },
): ReplayData {
  const m = d.meta;
  const fps = m.src_fps / m.stride;
  // Never relabel in the display: replays show the tracker's own numbers (Keep-ID collapse is opt-in only).
  const applyKeep = opts?.keepId ?? false;
  const rawOurs = d.guide?.events?.ours ?? [];
  let oursFrames = d.runs.ours.frames;
  let oursEv = rawOurs;
  let oursLabel = d.runs.ours.label;
  let oursShort = SHORT.ours;
  if (applyKeep) {
    const { frames, remap, idOf } = collapseRenumbers(oursFrames, rawOurs);
    oursFrames = frames;
    oursEv = oursEvents(rawOurs, remap, frames, fps, idOf);
    oursLabel = "Keep-ID";
    oursShort = "Keep-ID";
  }
  return {
    id: m.id,
    title: m.title,
    caption: m.caption,
    kind: m.kind,
    width: m.width,
    height: m.height,
    fps,
    frames: m.frames_tracked,
    duration: m.covered_s,
    videoUrl,
    demoUrl,
    keepId: applyKeep,
    runs: {
      baseline: { label: d.runs.baseline.label, short: SHORT.baseline, frames: d.runs.baseline.frames },
      ours: { label: oursLabel, short: oursShort, frames: oursFrames },
    },
    gate: d.guide?.gate ?? null,
    events: { baseline: d.guide?.events?.baseline ?? [], ours: oursEv },
    moments: d.guide?.moments ?? [],
  };
}

// ---- time ---------------------------------------------------------------------------------
export const timeOfFrame = (d: Pick<ReplayData, "fps">, f: number) => (f - 1) / d.fps;
export function frameAtTime(d: Pick<ReplayData, "fps" | "frames">, t: number): number | null {
  const k = Math.floor(t * d.fps + 1e-6) + 1;
  return k >= 1 && k <= d.frames ? k : null;
}

// ---- key-moment schedule ------------------------------------------------------------------
export const SLOW_RATE = 0.25;
export const SLOW_LEAD_S = 1.2; // slow down this long before the question
export const ASK_LEAD_S = 0.35; // ask just before the moment
export const ANSWER_AFTER_S = 0.6; // show the answer once the change is on screen
export const ZOOM_S = 0.6;

export interface Stop { kind: "ask" | "answer"; t: number; index: number }

export function schedule(d: Pick<ReplayData, "fps" | "moments">): Stop[] {
  const out: Stop[] = [];
  d.moments.forEach((m, index) => {
    const tm = timeOfFrame(d, m.f);
    out.push({ kind: "ask", t: Math.max(0, tm - ASK_LEAD_S), index });
    out.push({ kind: "answer", t: tm + ANSWER_AFTER_S, index });
  });
  return out.sort((a, b) => a.t - b.t);
}

/** Playback rate at time t in guided mode: slow from SLOW_LEAD_S before a question to its answer. */
export function rateAt(stops: Stop[], t: number, guided: boolean): number {
  if (!guided) return 1;
  for (const s of stops) {
    if (s.kind !== "ask") continue;
    const ans = stops.find((x) => x.kind === "answer" && x.index === s.index);
    if (t >= s.t - SLOW_LEAD_S && t <= (ans?.t ?? s.t)) return SLOW_RATE;
  }
  return 1;
}

/** First not-yet-done stop passed when time moves forward from prev to t (seeks back excluded). */
export function stopCrossed(stops: Stop[], prev: number, t: number, done: Set<string>): Stop | null {
  if (t < prev || t - prev > 0.5) return null; // a seek, not playback
  for (const s of stops) if (s.t > prev && s.t <= t && !done.has(stopKey(s))) return s;
  return null;
}
export const stopKey = (s: Stop) => `${s.kind}:${s.index}`;

/** Moment whose spotlight window contains t: from the slow-down to 1.5 s after the answer. */
export function activeMoment(stops: Stop[], t: number): number | null {
  let out: number | null = null; // the latest one whose window has started (moments can overlap)
  for (const s of stops) {
    if (s.kind !== "ask") continue;
    const ans = stops.find((x) => x.kind === "answer" && x.index === s.index)!;
    if (t >= s.t - SLOW_LEAD_S && t <= ans.t + 1.5) out = s.index;
  }
  return out;
}

const ease = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/** Zoom amount 0..1: ease in over ZOOM_S up to the question, hold to the answer, ease out after. */
export function zoomAt(stops: Stop[], t: number): { k: number; index: number | null } {
  for (const s of stops) {
    if (s.kind !== "ask") continue;
    const ans = stops.find((x) => x.kind === "answer" && x.index === s.index)!;
    if (t >= s.t - ZOOM_S && t < s.t) return { k: ease((t - (s.t - ZOOM_S)) / ZOOM_S), index: s.index };
    if (t >= s.t && t <= ans.t) return { k: 1, index: s.index };
    if (t > ans.t && t < ans.t + ZOOM_S) return { k: ease(1 - (t - ans.t) / ZOOM_S), index: s.index };
  }
  return { k: 0, index: null };
}

/** View rectangle (video px) for zoom k toward a 2x crop centred on `focus`, clamped to the frame. */
export function viewRect(W: number, H: number, focus: number[] | null, k: number, maxZoom = 2): [number, number, number, number] {
  if (!focus || k <= 0) return [0, 0, W, H];
  // up to 2x, but never so far that the people involved no longer fit (with some margin)
  const zoom = Math.max(1, Math.min(maxZoom, W / (focus[2] * 2.2 || 1), H / (focus[3] * 1.4 || 1)));
  const z = 1 + (zoom - 1) * k;
  const w = W / z, h = H / z;
  const cx = focus[0] + focus[2] / 2, cy = focus[1] + focus[3] / 2;
  const x = Math.min(Math.max(0, cx - w / 2), W - w);
  const y = Math.min(Math.max(0, cy - h / 2), H - h);
  return [x, y, w, h];
}

// ---- counters -----------------------------------------------------------------------------
export function countsAt(events: GEvent[], f: number | null): { in: number; out: number } {
  let i = 0, o = 0;
  if (f == null) return { in: 0, out: 0 };
  for (const e of events) {
    if (e.t !== "cross" || e.f > f) continue;
    if (e.dir === "in") i++;
    else o++;
  }
  return { in: i, out: o };
}

/** Extra counts on `run` so far: moments where its new number crossed again (counted twice). */
export function extraCountsAt(d: Pick<ReplayData, "moments" | "fps">, run: RunKey, t: number): number {
  return d.moments.filter((m) => m.counted_twice[run] && m.changed[run] && t >= timeOfFrame(d, m.f) + ANSWER_AFTER_S).length;
}

// ---- words --------------------------------------------------------------------------------
export const person = (id: number | null | undefined) => (id == null ? "someone" : `Person ${id}`);

export function eventText(e: GEvent): string | null {
  switch (e.t) {
    case "cross":
      return `${person(e.id)} crossed the line going ${e.dir === "in" ? "IN" : "OUT"}.`;
    case "back":
      return `${person(e.id)} was out of sight for ${e.gap_s.toFixed(1)} s and came back with the same number.`;
    case "renumber":
      return `Person ${e.from} was given a new number: Person ${e.to}.`;
    case "lost":
      return null; // folded into the matching "back" line
  }
}

export interface LogLine { f: number; t: number; text: string; kind: GEvent["t"]; moment?: number }

export function eventLog(d: Pick<ReplayData, "events" | "fps" | "moments">, run: RunKey): LogLine[] {
  const out: LogLine[] = [];
  for (const e of d.events[run]) {
    const text = eventText(e);
    if (!text) continue;
    const mi = e.t === "renumber" ? d.moments.findIndex((m) => m.f === e.f) : -1;
    out.push({ f: e.f, t: timeOfFrame(d, e.f), text, kind: e.t, ...(mi >= 0 ? { moment: mi } : {}) });
  }
  return out.sort((a, b) => a.f - b.f);
}

/** Log lines already on screen: only events at or before the frame shown (none before frame 1). */
export function visibleLog(log: LogLine[], f: number | null): LogLine[] {
  return f == null ? [] : log.filter((l) => l.f <= f);
}

/** Video sources to try in order: full clip, then the committed demo cut. */
export function videoSources(d: Pick<ReplayData, "videoUrl" | "demoUrl">): string[] {
  return [d.videoUrl, d.demoUrl].filter((x): x is string => !!x);
}

export const MISSING_CLIP_TEXT = "Clip not downloaded on this machine. In the repo folder run make clips blur, then reload.";
export const NO_EXPORT_TEXT = "Video not exported yet — showing boxes only";
/** Banner for the schematic fallback, or null when video frames are showing. */
export function fallbackText(d: Pick<ReplayData, "kind" | "videoUrl" | "demoUrl">, failed: boolean): string | null {
  if (videoSources(d).length === 0) return d.kind === "measured" ? NO_EXPORT_TEXT : MISSING_CLIP_TEXT;
  return failed ? (d.kind === "measured" ? NO_EXPORT_TEXT : MISSING_CLIP_TEXT) : null;
}

function idsOf(m: Moment, run: RunKey): PersonIds | undefined {
  return m.people[0]?.ids[run];
}

/** One-line answer for one tracker at a moment, e.g. "#2 became #10" / "kept #2". */
export function answerFor(m: Moment, run: RunKey): string {
  const ids = idsOf(m, run);
  if (!ids || ids.after == null) return "lost them";
  if (ids.before != null && ids.before !== ids.after) return `#${ids.before} became #${ids.after}`;
  return `kept #${ids.after}`;
}

export function askText(m: Moment): string {
  if (m.kind === "overlap") return "These two people overlap. Watch: does the highlighted person keep their number?";
  if (m.kind === "gate") return "This person is at the line. Watch: do they keep their number as they cross?";
  return "This person is partly hidden. Watch: do they keep their number?";
}

export function resultText(m: Moment, kind: Kind, names: Record<RunKey, string> = NAMES): string {
  const b = answerFor(m, "baseline"), o = answerFor(m, "ours");
  if (kind === "measured" && m.verified) return `Checked against human labels: ${names.baseline} ${b}; ${names.ours} ${o}.`;
  if (m.changed.baseline && m.changed.ours) return `Both trackers changed this person's number here: ${names.baseline} ${b}, ${names.ours} ${o}.`;
  return `The two trackers disagree here: ${names.baseline} ${b}, ${names.ours} ${o}.`;
}

export const STAKE_TEXT = "A new number means this person can be counted twice at the gate.";

/** Why the guided player paused, and which person to look at on the full frame. */
export function pauseWhy(
  m: Moment, run: RunKey, phase: "ask" | "answer", kind: Kind, names: Record<RunKey, string> = NAMES,
): { title: string; body: string; tag: string } {
  const ids = idsOf(m, run);
  const n = ids?.before ?? ids?.after;
  const who = n != null ? `person #${n}` : "the person in the bright box";
  const tag = n != null ? `Look here · #${n}` : "Look here";
  if (phase === "ask") {
    if (m.kind === "overlap") {
      return { title: `Paused · look at ${who}`, body: `Two people overlap here. Watch ${who} (gold box) — does that number stay the same when they separate?`, tag: n != null ? `Look here · #${n} overlapping` : "Look here · overlap" };
    }
    if (m.kind === "gate") {
      return { title: `Paused · look at ${who}`, body: `${who} is at the gold gate. Watch whether they keep that number as they cross.`, tag };
    }
    return { title: `Paused · look at ${who}`, body: `${who} is blocked or partly hidden. Watch whether they come back with the same number.`, tag };
  }
  return { title: `Look at ${who} · what happened`, body: resultText(m, kind, names), tag };
}

/** Video-px box to ring on the full frame: union of spotlight boxes, else the moment focus. */
export function lookBox(
  frames: Record<string, Box[]>, f: number | null, spot: Set<number>, focus: number[] | null, W: number, H: number,
): [number, number, number, number] | null {
  const boxes = f != null ? (frames[String(f)] ?? []).filter((b) => spot.has(b[0])) : [];
  let rect: [number, number, number, number] | null = null;
  if (boxes.length) {
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const [, x, y, w, h] of boxes) {
      x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x + w); y2 = Math.max(y2, y + h);
    }
    rect = [x1, y1, x2 - x1, y2 - y1];
  } else if (focus && focus.length >= 4) {
    rect = [focus[0], focus[1], focus[2], focus[3]];
  }
  if (!rect) return null;
  if (rect[2] * rect[3] > 0.4 * W * H && focus && focus.length >= 4) {
    rect = [focus[0], focus[1], focus[2], focus[3]];
  }
  const minW = 0.07 * W, minH = 0.16 * H;
  let [x, y, w, h] = rect;
  if (w < minW) {
    x = Math.max(0, x - (minW - w)); // grow toward the centre when the person is on the right edge
    if (x + minW > W) x = Math.max(0, W - minW);
    w = Math.min(minW, W);
  }
  if (h < minH) {
    y = Math.max(0, y - (minH - h) / 2);
    if (y + minH > H) y = Math.max(0, H - minH);
    h = Math.min(minH, H);
  }
  return [x, y, w, h];
}

/** Chip text for an ID that just changed: "#2 → #10" for 1 s after the change, else "Person 10". */
export function chipText(id: number, f: number, events: GEvent[], fps: number): { text: string; pulse: number } {
  for (const e of events) {
    if (e.t !== "renumber" || e.to !== id) continue;
    const dt = (f - e.f) / fps;
    if (dt >= 0 && dt < 1) return { text: `#${e.from} → #${e.to}`, pulse: dt < 0.3 ? 1 - dt / 0.3 : 0 };
  }
  return { text: person(id), pulse: 0 };
}

/** Ids to spotlight on `run` at a moment (both numbers if it changed). */
export function spotlightIds(m: Moment, run: RunKey): Set<number> {
  const s = new Set<number>();
  for (const p of m.people) {
    const ids = p.ids[run];
    if (ids?.before != null) s.add(ids.before);
    if (ids?.after != null) s.add(ids.after);
  }
  return s;
}

/** Ids that crossed the gate within the last `windowS` seconds (spotlit outside key moments). */
export function recentCrossers(events: GEvent[], f: number, fps: number, windowS = 1.5): Set<number> {
  const s = new Set<number>();
  for (const e of events) if (e.t === "cross" && e.f <= f && (f - e.f) / fps <= windowS) s.add(e.id);
  return s;
}

/** Bottom-centre trail of one id over the last `spanS` seconds: [x, y, alpha 0..1]. Per-ID, so it
 * breaks and restarts in the new colour when the number changes. */
export function trailOf(frames: Record<string, Box[]>, id: number, f: number, fps: number, spanS = 2): [number, number, number][] {
  const n = Math.round(spanS * fps);
  const pts: [number, number, number][] = [];
  for (let k = Math.max(1, f - n); k <= f; k++) {
    const b = frames[String(k)]?.find((r) => r[0] === id);
    if (b) pts.push([b[1] + b[3] / 2, b[2] + b[4], 1 - (f - k) / n]);
  }
  return pts;
}

/** Last box of `id` within `spanS` before f, for a "last seen here" ghost while hidden. */
export function lastSeen(frames: Record<string, Box[]>, id: number, f: number, fps: number, spanS = 1.5): { box: Box; f: number } | null {
  for (let k = f - 1; k >= Math.max(1, f - Math.round(spanS * fps)); k--) {
    const b = frames[String(k)]?.find((r) => r[0] === id);
    if (b) return { box: b, f: k };
  }
  return null;
}

// ---- tally, moment strip, stepping ---------------------------------------------------------
export const MOMENT_RULE = "every moment where at least one tracker changed a person's number.";

export interface Tally { baseline: number; ours: number; both: number; total: number }
/** Totals per tracker (a "both" moment counts for each) plus how many were both. */
export function tally(moments: Pick<Moment, "changed">[]): Tally {
  let b = 0, o = 0, n = 0;
  for (const m of moments) {
    if (m.changed.baseline) b++;
    if (m.changed.ours) o++;
    if (m.changed.baseline && m.changed.ours) n++;
  }
  return { baseline: b, ours: o, both: n, total: moments.length };
}
export function tallyText(t: Tally, names: Record<RunKey, string> = NAMES): string {
  return `${names.baseline} changed ${t.baseline} · ${names.ours} changed ${t.ours} · both: ${t.both}`;
}

export interface Tick { index: number; x: number; top: boolean; bottom: boolean; label: string }
/** One tick per moment at its time (0..1 of duration): top half = baseline, bottom = ours. */
export function stripTicks(d: Pick<ReplayData, "moments" | "fps" | "duration">, names: Record<RunKey, string> = NAMES): Tick[] {
  return d.moments.map((m, index) => {
    const who = m.changed.baseline && m.changed.ours ? "both trackers" : m.changed.baseline ? names.baseline : names.ours;
    return {
      index,
      x: d.duration > 0 ? Math.min(1, Math.max(0, timeOfFrame(d, m.f) / d.duration)) : 0,
      top: m.changed.baseline,
      bottom: m.changed.ours,
      label: `Moment ${index + 1}: ${who} changed the number`,
    };
  });
}

export const SEEK_LEAD_S = 1.5;
/** Moment to jump to from time t: dir +1 = next one after t, -1 = the one before the current. */
export function stepMoment(d: Pick<ReplayData, "moments" | "fps">, t: number, dir: 1 | -1): number | null {
  const ts = d.moments.map((m) => timeOfFrame(d, m.f));
  if (!ts.length) return null;
  // the moment we are "at" is the one whose lead-in we are inside
  const eps = SEEK_LEAD_S + 0.05;
  if (dir > 0) {
    const i = ts.findIndex((x) => x - eps > t);
    return i >= 0 ? i : null;
  }
  for (let i = ts.length - 1; i >= 0; i--) if (ts[i] - eps < t - 0.3) return i;
  return null;
}
