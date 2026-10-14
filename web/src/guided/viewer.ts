// Benchmark clips (eval.export_viewer output, with human labels) -> guided replay.
// id_switch events there are measured against ground truth, so moments are "verified"; the caption
// still only states what each tracker's data shows (no comparison is claimed unless both runs are
// present and say so).
import type { IndexedClip } from "@viewer/data";
import type { Box, RunKey } from "../lib/scenarios";
import type { GEvent, Moment, ReplayData } from "./model";

export const BENCHMARK_CLIPS = ["MOT17-09", "MOT17-04", "MOT20-02"];

function framesOf(c: IndexedClip): Record<string, Box[]> {
  const out: Record<string, Box[]> = {};
  for (const [id, rows] of Object.entries(c.tracks))
    for (const r of rows) (out[String(r[0])] ??= []).push([Number(id), r[1], r[2], r[3], r[4]]);
  return out;
}

function eventsOf(c: IndexedClip): GEvent[] {
  const out: GEvent[] = [];
  for (const e of c.events) {
    if (e.type === "cross") out.push({ t: "cross", f: e.f, id: e.id, dir: e.dir });
    else if (e.type === "id_switch") out.push({ t: "renumber", f: e.f, from: e.from, to: e.to });
  }
  return out.sort((a, b) => a.f - b.f);
}

const iou = (a: number[], b: number[]) => {
  const iw = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const i = iw * ih, u = a[2] * a[3] + b[2] * b[3] - i;
  return u > 0 ? i / u : 0;
};

function boxOf(frames: Record<string, Box[]>, f: number, id: number): number[] | null {
  const b = frames[String(f)]?.find((r) => r[0] === id);
  return b ? b.slice(1) : null;
}
function matchId(frames: Record<string, Box[]>, f: number, box: number[]): number | null {
  let best = 0.3, id: number | null = null;
  for (const b of frames[String(f)] ?? []) { const v = iou(b.slice(1), box); if (v > best) { best = v; id = b[0]; } }
  return id;
}
function lastBefore(frames: Record<string, Box[]>, f: number, id: number): number[] | null {
  for (let k = f - 1; k >= Math.max(1, f - 300); k--) { const b = boxOf(frames, k, id); if (b) return b; }
  return null;
}

export function fromViewerClips(base: IndexedClip, ours: IndexedClip, labels: Record<RunKey, string>): ReplayData {
  const frames = { baseline: framesOf(base), ours: framesOf(ours) };
  const events = { baseline: eventsOf(base), ours: eventsOf(ours) };
  const moments: Moment[] = [];
  for (const run of ["baseline", "ours"] as const) {
    const other: RunKey = run === "baseline" ? "ours" : "baseline";
    const clip = run === "baseline" ? base : ours;
    for (const e of clip.events) {
      if (e.type !== "id_switch") continue;
      if (moments.some((m) => Math.abs(m.f - e.f) < 2 * clip.fps)) continue;
      const box = boxOf(frames[run], e.f, e.to);
      if (!box) continue;
      const otherAfter = matchId(frames[other], e.f, box);
      const prev = lastBefore(frames[run], e.f, e.from);
      const otherBefore = prev ? matchId(frames[other], e.f - 1, prev) ?? otherAfter : otherAfter;
      const crossed = (id: number) => events[run].filter((x) => x.t === "cross" && x.id === id).map((x) => (x as { dir: string }).dir);
      moments.push({
        f: e.f, kind: "occlusion", changed: run, kept: null, verified: true,
        people: [{ box, ids: { [run]: { before: e.from, after: e.to }, [other]: { before: otherBefore, after: otherAfter } } }],
        focus: box,
        counted_twice: crossed(e.from).some((d) => crossed(e.to).includes(d)),
      });
    }
  }
  moments.sort((a, b) => a.f - b.f);
  const media = base.media?.type === "video" ? base.media : null;
  return {
    id: base.clip, title: base.clip, kind: "measured",
    width: base.width, height: base.height, fps: base.fps, frames: base.frames, duration: base.frames / base.fps,
    videoUrl: media?.src ?? null, videoOffsetS: (media?.frame_offset ?? 0) / base.fps,
    runs: { baseline: { label: labels.baseline, short: labels.baseline, frames: frames.baseline }, ours: { label: labels.ours, short: labels.ours, frames: frames.ours } },
    gate: base.gate, events, moments: moments.slice(0, 3),
  };
}
