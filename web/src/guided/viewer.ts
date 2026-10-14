// Benchmark clips (eval.export_viewer output, with human labels) -> guided replay.
// id_switch events there are measured against ground truth, so moments are "verified"; the caption
// still only states what each tracker's data shows (no comparison is claimed unless both runs are
// present and say so).
import type { IndexedClip } from "@viewer/data";
import { fetchClip, fetchClips } from "../workstation/api";
import { prettyTracker } from "../workstation/lib/journey";
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
function lastBefore(frames: Record<string, Box[]>, f: number, id: number): { box: number[]; f: number } | null {
  for (let k = f - 1; k >= Math.max(1, f - 300); k--) { const b = boxOf(frames, k, id); if (b) return { box: b, f: k }; }
  return null;
}

export function fromViewerClips(base: IndexedClip, ours: IndexedClip, labels: Record<RunKey, string>): ReplayData {
  const frames = { baseline: framesOf(base), ours: framesOf(ours) };
  const events = { baseline: eventsOf(base), ours: eventsOf(ours) };
  const moments: Moment[] = [];
  type Sw = { f: number; gt: number; from: number; to: number };
  const sw = (c: IndexedClip): Sw[] => c.events.flatMap((e) => (e.type === "id_switch" ? [{ f: e.f, gt: e.gt, from: e.from, to: e.to }] : []));
  const all = { baseline: sw(base), ours: sw(ours) };
  const used = new Set<Sw>();
  const crossed = (run: RunKey, id: number) => new Set(events[run].flatMap((x) => (x.t === "cross" && x.id === id ? [x.dir] : [])));
  const twice = (run: RunKey, a: number, b: number) => [...crossed(run, a)].some((d) => crossed(run, b).has(d));
  for (const run of ["baseline", "ours"] as const) {
    const other: RunKey = run === "baseline" ? "ours" : "baseline";
    for (const e of all[run]) {
      if (used.has(e)) continue;
      used.add(e);
      // same labelled person switching on the other tracker within 1 s = one "both" moment
      const twin = all[other].find((o) => !used.has(o) && o.gt === e.gt && Math.abs(o.f - e.f) <= base.fps);
      if (twin) used.add(twin);
      const box = boxOf(frames[run], e.f, e.to);
      if (!box) continue;
      const prev = lastBefore(frames[run], e.f, e.from);
      const otherIds = twin
        ? { before: twin.from, after: twin.to }
        : { before: prev ? matchId(frames[other], prev.f, prev.box) : null, after: matchId(frames[other], e.f, box) };
      moments.push({
        f: twin ? Math.min(e.f, twin.f) : e.f, kind: "occlusion", verified: true,
        changed: { [run]: true, [other]: !!twin } as Record<RunKey, boolean>,
        people: [{ box, ids: { [run]: { before: e.from, after: e.to }, [other]: otherIds } }],
        focus: box,
        counted_twice: { [run]: twice(run, e.from, e.to), [other]: !!twin && twice(other, twin.from, twin.to) } as Record<RunKey, boolean>,
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
    gate: base.gate, events, moments,
  };
}

export interface BenchPair { name: string; base: IndexedClip; ours: IndexedClip }

/** Benchmark pairs (ByteTrack-like vs another tracker, never gt) in the exported viewer clips. */
export function benchmarkPairs(clips: IndexedClip[]): BenchPair[] {
  const out: BenchPair[] = [];
  for (const name of BENCHMARK_CLIPS) {
    const cs = clips.filter((c) => c.clip.startsWith(name));
    const base = cs.find((c) => /bytetrack/i.test(c.tracker) && !/reid|vis/i.test(c.tracker));
    const rest = cs.filter((c) => c !== base && c.tracker !== "gt");
    const ours = rest.find((c) => /vis/i.test(c.tracker)) ?? rest.find((c) => /botsort/i.test(c.tracker)) ?? rest[0];
    if (base && ours) out.push({ name, base, ours });
  }
  return out;
}

/** Load only the clip files the benchmark pairs need (Scenarios tab has no workstation state). */
export async function loadBenchmarkPairs(): Promise<BenchPair[]> {
  const refs = await fetchClips();
  const want = refs.filter((r) => BENCHMARK_CLIPS.some((n) => r.clip.startsWith(n)) && r.tracker !== "gt" && /^(bytetrack|botsort|.*vis.*)$/i.test(r.tracker));
  return benchmarkPairs(await Promise.all(want.map((r) => fetchClip(r.file))));
}

/** Neutral framing of the first measured moment: the problem, not a win for either tracker. */
export function problemCaption(d: ReplayData): string {
  const m = d.moments[0];
  if (!m) return "Human-labelled benchmark clip. No number changes were measured on it.";
  const ib = m.people[0].ids.baseline, io = m.people[0].ids.ours;
  const ch = (x?: { before: number | null; after: number | null }) => x?.before != null && x.after != null && x.before !== x.after;
  const both = ch(ib) && ch(io);
  const who = both ? "both trackers give them" : ch(ib) ? `${d.runs.baseline.short} gives them` : `${d.runs.ours.short} gives them`;
  const x = ch(ib) ? ib! : io!;
  return `The problem: a person passes behind others and comes back, and ${who} a new number (#${x.before} → #${x.after}). A gate could then count one person twice. This clip shows the problem, not a win for either tracker.`;
}

export function benchmarkReplay(p: BenchPair): ReplayData {
  const d = fromViewerClips(p.base, p.ours, { baseline: prettyTracker(p.base.tracker).title, ours: prettyTracker(p.ours.tracker).title });
  const split = p.name.startsWith("MOT20") ? "MOT20 val-half" : "MOT17 val-half";
  return { ...d, caption: problemCaption(d), title: `${p.name} (${split})` };
}
