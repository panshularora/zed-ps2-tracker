// Keep-ID reassociation for Studio / Scenarios (Pexels overlays).
// The JSON "ours" run is vis-gated BoT-SORT, which drops a track after its buffer and mints a new
// number. This pass treats every saved box as a detection and rebinds lost tracks by predicted
// centre, same contract as tracker/keep_id.py: one body, one number through a dropped-box gap.
import type { Box } from "../lib/scenarios";
import type { GEvent } from "../guided/model";
import type { VerdictOf } from "./verify";

export interface KeepIdOpts {
  fps: number;
  width: number;
  buffer?: number;
  maxDist?: number;
}

interface Track {
  tid: number;
  x: number;
  y: number;
  w: number;
  h: number;
  vx: number;
  vy: number;
  misses: number;
}

const hypot = (x: number, y: number) => Math.hypot(x, y);

function assign(tracks: Track[], dets: Box[], maxDist: number): { t: number; d: number }[] {
  const pairs: { t: number; d: number; dist: number }[] = [];
  for (let i = 0; i < tracks.length; i++) {
    for (let j = 0; j < dets.length; j++) {
      const [, x, y, w, h] = dets[j];
      const dist = hypot(tracks[i].x - (x + w / 2), tracks[i].y - (y + h / 2));
      const lim = Math.max(maxDist, 0.45 * tracks[i].h);
      if (dist <= lim) pairs.push({ t: i, d: j, dist });
    }
  }
  pairs.sort((a, b) => a.dist - b.dist);
  const usedT = new Set<number>(), usedD = new Set<number>(), out: { t: number; d: number }[] = [];
  for (const p of pairs) {
    if (usedT.has(p.t) || usedD.has(p.d)) continue;
    usedT.add(p.t); usedD.add(p.d); out.push(p);
  }
  return out;
}

/** Re-id `frames` in place-order. Returns new frames (same boxes, stable ids) and oldId@frame → newId. */
export function keepIdAssociate(frames: Record<string, Box[]>, opts: KeepIdOpts): { frames: Record<string, Box[]>; remap: Map<string, number> } {
  const fps = opts.fps;
  const buffer = opts.buffer ?? Math.max(90, Math.round(fps * 4));
  const maxDist = opts.maxDist ?? Math.max(96, 0.1 * opts.width);
  const keys = Object.keys(frames).map(Number).filter((n) => n > 0).sort((a, b) => a - b);
  const out: Record<string, Box[]> = {};
  const remap = new Map<string, number>();
  if (!keys.length) return { frames, remap };
  const tracks: Track[] = [];
  const liveIds = () => new Set(tracks.map((t) => t.tid));
  let next = 1;
  const fresh = (prefer: number) => {
    const live = liveIds();
    if (prefer > 0 && !live.has(prefer)) return prefer;
    while (live.has(next)) next += 1;
    const id = next;
    next += 1;
    return id;
  };

  for (let f = keys[0]; f <= keys[keys.length - 1]; f++) {
    for (const t of tracks) {
      t.x += t.vx;
      t.y += t.vy;
    }
    const dets = frames[String(f)] ?? [];
    const matchedT = new Set<number>(), matchedD = new Set<number>();
    const boxes: Box[] = [];
    for (const m of assign(tracks, dets, maxDist)) {
      const t = tracks[m.t];
      const [orig, x, y, w, h] = dets[m.d];
      const cx = x + w / 2, cy = y + h / 2;
      const prevX = t.x - t.vx, prevY = t.y - t.vy;
      t.vx = 0.5 * t.vx + 0.5 * (cx - prevX);
      t.vy = 0.5 * t.vy + 0.5 * (cy - prevY);
      t.x = cx; t.y = cy; t.w = w; t.h = h; t.misses = 0;
      matchedT.add(m.t); matchedD.add(m.d);
      remap.set(`${f}:${orig}`, t.tid);
      boxes.push([t.tid, x, y, w, h]);
    }
    for (let i = 0; i < tracks.length; i++) if (!matchedT.has(i)) tracks[i].misses += 1;
    const kept: Track[] = [];
    for (const t of tracks) if (t.misses <= buffer) kept.push(t);
    tracks.length = 0;
    tracks.push(...kept);
    for (let j = 0; j < dets.length; j++) {
      if (matchedD.has(j)) continue;
      const [orig, x, y, w, h] = dets[j];
      const tid = fresh(orig);
      tracks.push({ tid, x: x + w / 2, y: y + h / 2, w, h, vx: 0, vy: 0, misses: 0 });
      remap.set(`${f}:${orig}`, tid);
      boxes.push([tid, x, y, w, h]);
    }
    if (boxes.length) out[String(f)] = boxes;
  }
  return { frames: out, remap };
}

export function gapEvents(frames: Record<string, Box[]>, fps: number): GEvent[] {
  const byId = new Map<number, number[]>();
  for (const [fs, bs] of Object.entries(frames)) {
    const f = Number(fs);
    for (const b of bs) {
      const a = byId.get(b[0]) ?? [];
      a.push(f);
      byId.set(b[0], a);
    }
  }
  const out: Extract<GEvent, { t: "lost" } | { t: "back" }>[] = [];
  for (const [id, fs] of byId) {
    fs.sort((a, b) => a - b);
    for (let i = 1; i < fs.length; i++) {
      const gap = fs[i] - fs[i - 1] - 1;
      if (gap < 2) continue;
      out.push({ t: "lost", f: fs[i - 1], id });
      out.push({ t: "back", f: fs[i], id, gap_s: Math.round((gap / fps) * 10) / 10 });
    }
  }
  return out.sort((a, b) => a.f - b.f || a.id - b.id);
}

export function remapCrosses(events: GEvent[], remap: Map<string, number>, idOf?: (id: number) => number): GEvent[] {
  const out: GEvent[] = [];
  for (const e of events) {
    if (e.t !== "cross") continue;
    const id = remap.get(`${e.f}:${e.id}`) ?? idOf?.(e.id) ?? e.id;
    out.push({ ...e, id });
  }
  return out;
}

export function oursEvents(prev: GEvent[], remap: Map<string, number>, frames: Record<string, Box[]>, fps: number, idOf?: (id: number) => number): GEvent[] {
  return [...remapCrosses(prev, remap, idOf), ...gapEvents(frames, fps)].sort((a, b) => a.f - b.f);
}

/**
 * Keep the first number when vis-botsort minted a new one (guide `renumber` events).
 * Does not re-match boxes: #4 stays #4; 7→21 becomes 7 on every later box.
 */
export function collapseRenumbers(
  frames: Record<string, Box[]>,
  events: GEvent[],
): { frames: Record<string, Box[]>; remap: Map<string, number>; idOf: (id: number) => number } {
  const parent = new Map<number, number>();
  const find = (a: number): number => {
    const p = parent.get(a) ?? a;
    if (p === a) return a;
    const r = find(p);
    parent.set(a, r);
    return r;
  };
  for (const e of events) {
    if (e.t !== "renumber") continue;
    const a = find(e.from), b = find(e.to);
    if (a !== b) parent.set(b, a);
  }
  const remap = new Map<string, number>();
  const out: Record<string, Box[]> = {};
  for (const [fs, bs] of Object.entries(frames)) {
    out[fs] = bs.map(([id, x, y, w, h]) => {
      const tid = find(id);
      remap.set(`${fs}:${id}`, tid);
      return [tid, x, y, w, h] as Box;
    });
  }
  return { frames: out, remap, idOf: find };
}

/** Same-id gaps on a Keep-ID replay are kept (the overlay, not the Pexels heuristic). */
export function keepIdVerdict(): VerdictOf {
  return (id, back) => ({
    id, run: "ours", last: Math.max(1, back - 1), back, gap_frames: 0, gap_s: 0, verdict: "kept",
    app: null, motion_err: null, motion_tol: null, size_ratio: null,
    gt_before: null, gt_after: null, reasons: ["same number through the gap"], occluder: null,
  });
}
