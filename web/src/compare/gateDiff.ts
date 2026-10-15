// Gate scene comparison: ByteTrack vs ours on the same seed, from the real tracker stream.
// The scene is synthetic, so every simulated person has an exact identity (ground truth):
// * ID changes: the tracker's own count (PaneFrame.id_changes, = events where one person's number changed)
// * transfers (false keeps): one tracker number that was on person A is later on person B
// * counted vs true count: as streamed
// A "difference" is an ID change or transfer for one person in one pane with no matching event for
// that person in the other pane within 1 s.
import { PANES, type PaneKey, type SimFrame } from "../lib/gateSim";

export interface PaneStats { idChanges: number; transfers: number; counted: number; truth: number }
export interface Transfer { f: number; idx: number; tid: number; from: number; to: number }
export interface Diff { f: number; idx: number; person: number; pane: PaneKey; kind: "change" | "transfer"; text: string; people: number[] }

export function transfersOf(frames: SimFrame[], pane: PaneKey): Transfer[] {
  const owner = new Map<number, number>();
  const out: Transfer[] = [];
  frames.forEach((fr, idx) => {
    for (const [pid, tid] of fr[pane].ids) {
      const o = owner.get(tid);
      if (o != null && o !== pid) out.push({ f: fr.f, idx, tid, from: o, to: pid });
      owner.set(tid, pid);
    }
  });
  return out;
}

export function paneStats(frames: SimFrame[], pane: PaneKey): PaneStats {
  const last = frames[frames.length - 1];
  return { idChanges: last?.[pane].id_changes ?? 0, transfers: transfersOf(frames, pane).length, counted: last?.[pane].counted ?? 0, truth: last?.true_count ?? 0 };
}

const NAME: Record<PaneKey, string> = { bytetrack: "ByteTrack", ours: "ours" };
const other = (p: PaneKey): PaneKey => (p === "ours" ? "bytetrack" : "ours");

export function differences(frames: SimFrame[], fps: number): Diff[] {
  const win = Math.round(fps);
  type E = { f: number; idx: number; person: number; kind: Diff["kind"]; a: number; b: number };
  const ev: Record<PaneKey, E[]> = { bytetrack: [], ours: [] };
  for (const p of PANES) {
    frames.forEach((fr, idx) => { for (const [pid, from, to] of fr[p].events) ev[p].push({ f: fr.f, idx, person: pid, kind: "change", a: from, b: to }); });
    for (const t of transfersOf(frames, p)) ev[p].push({ f: t.f, idx: t.idx, person: t.to, kind: "transfer", a: t.tid, b: t.from });
  }
  const out: Diff[] = [];
  for (const p of PANES) {
    for (const e of ev[p]) {
      const twin = ev[other(p)].some((o) => o.kind === e.kind && o.person === e.person && Math.abs(o.f - e.f) <= win);
      if (twin) continue;
      const text = e.kind === "change"
        ? `${NAME[p]} changed person P${e.person}'s number #${e.a} → #${e.b}; ${NAME[other(p)]} did not (within 1 s)`
        : `${NAME[p]} gave #${e.a} (was P${e.b}) to person P${e.person}; ${NAME[other(p)]} did not (within 1 s)`;
      out.push({ f: e.f, idx: e.idx, person: e.person, pane: p, kind: e.kind, text, people: e.kind === "transfer" ? [e.person, e.b] : [e.person] });
    }
  }
  return out.sort((a, b) => a.f - b.f || a.person - b.person);
}
