// Side-by-side live replay: pure helpers (interpolation, focus, chip layout, live caption, stats).
// Wording rule as in guided/model.ts: illustrative clips (no ground truth) only say what each
// tracker did; "counted twice" / "wrong" only where the data says so.
import type { Box, RunKey } from "../lib/scenarios";
import { recentCrossers, spotlightIds, timeOfFrame, type GEvent, type Moment, type ReplayData } from "../guided/model";

/** Boxes at continuous time t, linearly interpolated between tracked frames (same id only). */
export function boxesAt(frames: Record<string, Box[]>, fps: number, nFrames: number, t: number): { f: number | null; boxes: Box[] } {
  const x = t * fps + 1; // fractional tracked frame
  const f0 = Math.floor(x + 1e-6);
  if (f0 < 1 || f0 > nFrames) return { f: null, boxes: [] };
  const a = Math.min(1, Math.max(0, x - f0));
  const cur = frames[String(f0)] ?? [];
  const next = f0 < nFrames ? frames[String(f0 + 1)] ?? [] : [];
  if (a === 0 || next.length === 0) return { f: f0, boxes: cur };
  const byId = new Map(next.map((b) => [b[0], b]));
  return {
    f: f0,
    boxes: cur.map((b) => {
      const n = byId.get(b[0]);
      if (!n) return b;
      return [b[0], b[1] + (n[1] - b[1]) * a, b[2] + (n[2] - b[2]) * a, b[3] + (n[3] - b[3]) * a, b[4] + (n[4] - b[4]) * a];
    }),
  };
}

export const LEAD_S = 2.5; // the caption starts telling the story this long before a moment
export const TAIL_S = 3; // and keeps the outcome on screen this long after

/** Index of the moment whose story window contains t (latest started), or null. */
export function momentAt(d: Pick<ReplayData, "moments" | "fps">, t: number): number | null {
  let out: number | null = null;
  d.moments.forEach((m, i) => {
    const tm = timeOfFrame(d, m.f);
    if (t >= tm - LEAD_S && t <= tm + TAIL_S) out = i;
  });
  return out;
}

/** Playback rate: 0.5x close to a moment so the change is visible, else 1x. Never pauses. */
export function liveRate(d: Pick<ReplayData, "moments" | "fps">, t: number): number {
  return d.moments.some((m) => Math.abs(t - timeOfFrame(d, m.f)) <= 1) ? 0.5 : 1;
}

/** Ids to keep at full strength on `run`: the people in the current moment, else recent gate crossers. */
export function focusIds(d: ReplayData, run: RunKey, t: number, f: number | null): Set<number> {
  const mi = momentAt(d, t);
  if (mi != null) return spotlightIds(d.moments[mi], run);
  return f == null ? new Set() : recentCrossers(d.events[run], f, d.fps);
}

/** Focus ids not on screen right now (hidden/occluded), with where they were last seen. */
export function hiddenNow(frames: Record<string, Box[]>, events: GEvent[], focus: Set<number>, f: number | null, fps: number, spanS = 2): { id: number; box: Box }[] {
  if (f == null) return [];
  const here = new Set((frames[String(f)] ?? []).map((b) => b[0]));
  const out: { id: number; box: Box }[] = [];
  for (const id of focus) {
    if (here.has(id)) continue;
    if (events.some((e) => e.t === "renumber" && e.from === id && e.f <= f)) continue; // replaced, not hidden
    for (let k = f - 1; k >= Math.max(1, f - Math.round(spanS * fps)); k--) {
      const b = frames[String(k)]?.find((r) => r[0] === id);
      if (b) { out.push({ id, box: b }); break; }
    }
  }
  return out;
}

export interface Chip { id: number; text: string; x: number; y: number; w: number; h: number; anchorX: number; anchorY: number }

/** Place chips above their boxes; a chip that would overlap one already placed moves up a row
 *  (or down below the frame top), so labels never sit on top of each other. Units: canvas px. */
export function layoutChips(items: { id: number; text: string; cx: number; top: number; w: number }[], h: number, width: number): Chip[] {
  const placed: Chip[] = [];
  const sorted = [...items].sort((a, b) => a.top - b.top || a.cx - b.cx);
  for (const it of sorted) {
    const x = Math.min(Math.max(0, it.cx - it.w / 2), Math.max(0, width - it.w));
    let y = it.top - h - 4;
    for (let tries = 0; tries < 12; tries++) {
      const hit = placed.some((p) => x < p.x + p.w + 2 && x + it.w + 2 > p.x && y < p.y + p.h + 2 && y + h + 2 > p.y);
      if (!hit && y >= 0) break;
      y = y < 0 ? Math.max(0, y) + h + 4 : y - h - 4;
      if (y < 0 && tries > 5) y = it.top + 4; // no room above: go inside the box top
    }
    placed.push({ id: it.id, text: it.text, x, y: Math.max(0, y), w: it.w, h, anchorX: it.cx, anchorY: it.top });
  }
  return placed;
}

const idsFor = (m: Moment, run: RunKey) => m.people[0]?.ids[run];

/** One plain sentence about what is happening right now in the replay. */
export function liveCaption(d: ReplayData, t: number, peopleInView: number): string {
  const mi = momentAt(d, t);
  if (mi == null) {
    const next = d.moments.find((m) => timeOfFrame(d, m.f) > t);
    const lead = peopleInView > 0 ? `${peopleInView} people in view. Both trackers give each person a number and follow them.` : "Waiting for people to appear.";
    if (!d.moments.length) return `${lead} The two trackers agree on every number in this clip.`;
    return next ? `${lead} Next mix-up in ${Math.max(0, timeOfFrame(d, next.f) - LEAD_S - t).toFixed(0)} s.` : `${lead} No more mix-ups in this clip.`;
  }
  const m = d.moments[mi];
  const tm = timeOfFrame(d, m.f);
  const b = idsFor(m, "baseline"), o = idsFor(m, "ours");
  const who = b?.before ?? o?.before;
  const p = who != null ? `Person ${who}` : "Someone";
  const nb = d.runs.baseline.short, no = d.runs.ours.short;
  if (t < tm) {
    if (m.kind === "overlap") return `${p} walks right past another person, their boxes overlap…`;
    if (m.kind === "gate") return `${p} reaches the counting line…`;
    return `${p} walks behind other people and goes out of sight…`;
  }
  const say = (name: string, ids: typeof b, changed: boolean) => {
    if (!ids || ids.after == null) return `${name} loses them`;
    if (changed && ids.before != null && ids.before !== ids.after) return `${name} calls them #${ids.after} (a new person)`;
    return `${name} keeps #${ids.after}`;
  };
  const back = m.kind === "overlap" ? "They separate" : m.kind === "gate" ? "They cross" : `${p} comes back`;
  let s = `…${back}. ${say(nb, b, m.changed.baseline)}; ${say(no, o, m.changed.ours)}.`;
  const twice = (["baseline", "ours"] as const).filter((r) => m.changed[r] && m.counted_twice[r]).map((r) => d.runs[r].short);
  if (twice.length) s += ` With ${twice.join(" and ")}, the gate would count them twice.`;
  else if (m.changed.baseline || m.changed.ours) s += " A new number means the gate could count them twice.";
  if (d.kind === "illustrative") s += " (No ground truth here: the trackers disagree; we can't say which is right.)";
  else if (m.verified) s += " Checked against human labels.";
  return s;
}

export interface RunStats { idChanges: number; crossIn: number; crossOut: number; tracks: number }
export function runStats(d: ReplayData, run: RunKey, upToFrame?: number | null): RunStats {
  const lim = upToFrame ?? Infinity;
  const ev = d.events[run].filter((e) => e.f <= lim);
  const ids = new Set<number>();
  for (const [k, bs] of Object.entries(d.runs[run].frames)) if (Number(k) <= lim) for (const b of bs) ids.add(b[0]);
  return {
    idChanges: ev.filter((e) => e.t === "renumber").length,
    crossIn: ev.filter((e) => e.t === "cross" && e.dir === "in").length,
    crossOut: ev.filter((e) => e.t === "cross" && e.dir === "out").length,
    tracks: ids.size,
  };
}
