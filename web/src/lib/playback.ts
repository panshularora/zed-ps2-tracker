// Pure playback logic for the demo page: video time <-> tracked frame, trails, recovery rings,
// gate counters, event filtering by clip consent. No DOM here, so it is unit-tested directly.
import { frameFromCurrentTime, frameFromMediaTime, seekTime } from "@viewer/media";
import type { JobEvent, JobResult, Styling, TrackRow } from "../api";

export const TRAIL_FRAMES = 30;
export const RING_MS = 300;

/** Tracked frame (1-based) shown at a video frame (1-based); null outside the tracked range. */
export function trackedFrame(videoFrame: number, stride: number, framesTracked: number): number | null {
  if (videoFrame < 1) return null;
  const k = Math.floor((videoFrame - 1) / stride) + 1;
  return k <= framesTracked ? k : null;
}

/** Video frame (1-based) of tracked frame k. */
export function videoFrameOf(k: number, stride: number): number {
  return (k - 1) * stride + 1;
}

/** Seconds into the video at which tracked frame k is shown (frame start). */
export function timeOfTracked(k: number, stride: number, fps: number): number {
  return (videoFrameOf(k, stride) - 1) / fps;
}

/** currentTime to seek to so that tracked frame k is on screen. */
export function seekTimeForTracked(k: number, stride: number, fps: number): number {
  return seekTime(videoFrameOf(k, stride), fps);
}

export function frameAtMediaTime(mediaTime: number, fps: number): number {
  return frameFromMediaTime(mediaTime, fps);
}

export function frameAtCurrentTime(currentTime: number, fps: number): number {
  return frameFromCurrentTime(currentTime, fps);
}

export interface IndexedTrack {
  id: number;
  rows: TrackRow[]; // sorted by frame
}

export interface Indexed {
  tracks: IndexedTrack[];
  byFrame: Map<number, { id: number; row: TrackRow }[]>;
}

export function indexTracks(tracks: Record<string, TrackRow[]>): Indexed {
  const out: IndexedTrack[] = [];
  const byFrame = new Map<number, { id: number; row: TrackRow }[]>();
  for (const [key, rows] of Object.entries(tracks)) {
    const id = Number(key);
    const sorted = [...rows].sort((a, b) => a[0] - b[0]);
    out.push({ id, rows: sorted });
    for (const row of sorted) {
      const list = byFrame.get(row[0]) ?? [];
      list.push({ id, row });
      byFrame.set(row[0], list);
    }
  }
  return { tracks: out, byFrame };
}

/** Index of the last row with frame <= f (binary search), or -1. */
export function lastAtOrBefore(rows: TrackRow[], f: number): number {
  let lo = 0;
  let hi = rows.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid][0] <= f) {
      ans = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return ans;
}

export interface TrailPoint {
  x: number;
  y: number;
  /** 1 at the current frame, falling linearly to 0 at TRAIL_FRAMES frames ago */
  alpha: number;
}

/** Bottom-centre trail of a track over the last `span` tracked frames up to f (the gate point). */
export function trail(rows: TrackRow[], f: number, span = TRAIL_FRAMES): TrailPoint[] {
  const end = lastAtOrBefore(rows, f);
  if (end < 0 || rows[end][0] !== f) return [];
  const pts: TrailPoint[] = [];
  for (let i = end; i >= 0; i--) {
    const r = rows[i];
    const age = f - r[0];
    if (age >= span) break;
    pts.push({ x: r[1] + r[3] / 2, y: r[2] + r[4], alpha: 1 - age / span });
  }
  return pts.reverse();
}

/** Events the clip may show. Neutral clips (no consent): crossings and recoveries only. */
export function visibleEvents(events: JobEvent[], styling: Styling): JobEvent[] {
  if (styling === "alert") return events;
  return events.filter((e) => (e.kind === "gate" && e.type === "crossing") || e.kind === "recovery");
}

/** In/out counts up to and including tracked frame f (crossings + held crossings). */
export function gateCounts(events: JobEvent[], f: number): { in: number; out: number } {
  let i = 0;
  let o = 0;
  for (const e of events) {
    if (e.kind !== "gate" || e.frame > f) continue;
    if (e.type !== "crossing" && e.type !== "ambiguous_hold") continue;
    if (e.direction === "in") i++;
    else o++;
  }
  return { in: i, out: o };
}

/**
 * Recovery rings active at video time t (seconds): {id: strength 1 -> 0 over RING_MS}.
 * A ring starts when the recovered frame is first shown.
 */
export function activeRings(
  recoveries: { frame: number; track_id: number }[],
  t: number,
  stride: number,
  fps: number,
): Map<number, number> {
  const out = new Map<number, number>();
  for (const r of recoveries) {
    const t0 = timeOfTracked(r.frame, stride, fps);
    const dt = (t - t0) * 1000;
    if (dt >= -1e-6 && dt < RING_MS) out.set(r.track_id, Math.max(out.get(r.track_id) ?? 0, 1 - dt / RING_MS));
  }
  return out;
}

/** Gate pulse strength at video time t after the latest crossing (neutral amber flash). */
export function gatePulse(crossFrames: number[], t: number, stride: number, fps: number, ms = 300): number {
  let best = 0;
  for (const f of crossFrames) {
    const dt = (t - timeOfTracked(f, stride, fps)) * 1000;
    if (dt >= 0 && dt < ms) best = Math.max(best, 1 - dt / ms);
  }
  return best;
}

export function fmtClock(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, "0")}:${rest.toFixed(2).padStart(5, "0")}`;
}

/** One log line for an event (mono). Neutral wording: what the tracker did, not what a person did wrong. */
export function describe(e: JobEvent): string {
  switch (e.type) {
    case "crossing":
      return `#${e.track_id} crossed ${e.direction.toUpperCase()}`;
    case "recovery":
      return `#${e.track_id} kept its ID after ${e.reason === "lost" ? "lost" : "occluded"} ${e.gap_s.toFixed(2)} s`;
    case "recovered_crossing":
      return `possible missed ${e.direction.toUpperCase()} crossing: #${e.from_track_id} → #${e.track_id} (for an operator to confirm)`;
    case "tailgate":
      return `#${e.track_id} tailgate behind #${e.lead_track_id} (${e.gap_s.toFixed(2)} s)`;
    case "ambiguous_hold":
      return `#${e.track_id} ${e.direction.toUpperCase()} held: ${e.reasons.join(", ")}`;
    case "suspected_id_switch":
      return `#${e.from_id} -> #${e.to_id} possible ID break (heuristic)`;
  }
}

export function eventTrackId(e: JobEvent): number {
  return e.type === "suspected_id_switch" ? e.to_id : e.track_id;
}

/** Fraction (0..1) along the scrubber of tracked frame k, given the playable duration. */
export function tickPosition(k: number, r: Pick<JobResult, "video">, durationS: number): number {
  if (durationS <= 0) return 0;
  return Math.min(1, Math.max(0, timeOfTracked(k, r.video.stride, r.video.fps) / durationS));
}
