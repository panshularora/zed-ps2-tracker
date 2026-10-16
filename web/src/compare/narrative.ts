// "Same person, two trackers": one moment where a simulated person walks behind another, with
// each tracker's number before and after, and the headline + verdicts built ONLY from those numbers
// so the text cannot contradict the panes. Source: the synthetic gate scene stream (exact ground
// truth: every simulated person has a known identity).
import { PANES, type PaneKey, type SimFrame } from "../lib/gateSim";
import { BASE_SHORT, OURS_SHORT } from "../lib/ours";

export interface HideMoment {
  person: number;
  /** nearest visible person when the hide starts, or null */
  occluder: number | null;
  /** frame index (into frames) of the last frame seen before the hide, and first frame back */
  lastIdx: number;
  backIdx: number;
}
export interface Outcome { before: number | null; after: number | null; verdict: "kept" | "changed" | "unknown" }
export interface MomentStory { m: HideMoment; out: Record<PaneKey, Outcome>; headline: string; verdicts: Record<PaneKey, string> }

const idOf = (fr: SimFrame | undefined, pane: PaneKey, pid: number) => {
  const hit = fr?.[pane].ids.find(([p]) => p === pid);
  return hit ? hit[1] : null;
};

/** Every visible → hidden → visible stretch of one simulated person. */
export function hideMoments(frames: SimFrame[]): HideMoment[] {
  const out: HideMoment[] = [];
  const lastSeen = new Map<number, number>();
  const hiddenFrom = new Map<number, number>();
  frames.forEach((fr, idx) => {
    for (const [pid, x, y, vis] of fr.people) {
      if (vis) {
        const h = hiddenFrom.get(pid);
        const ls = lastSeen.get(pid);
        if (h != null && ls != null) {
          const prev = frames[ls];
          const me = prev.people.find((p) => p[0] === pid);
          let occ: number | null = null, best = Infinity;
          for (const q of frames[h]?.people ?? []) {
            if (q[0] === pid || !q[3] || !me) continue;
            const d = Math.hypot(q[1] - me[1], q[2] - me[2]);
            if (d < best) { best = d; occ = q[0]; }
          }
          out.push({ person: pid, occluder: occ, lastIdx: ls, backIdx: idx });
        }
        hiddenFrom.delete(pid);
        lastSeen.set(pid, idx);
        void x; void y;
      } else if (!hiddenFrom.has(pid) && lastSeen.has(pid)) hiddenFrom.set(pid, idx);
    }
  });
  return out;
}

/** Number before the hide vs the first number given within `look` frames of coming back. */
export function outcome(frames: SimFrame[], m: HideMoment, pane: PaneKey, look = 15): Outcome {
  const before = idOf(frames[m.lastIdx], pane, m.person);
  let after: number | null = null;
  for (let i = m.backIdx; i < Math.min(frames.length, m.backIdx + look); i++) {
    after = idOf(frames[i], pane, m.person);
    if (after != null) break;
  }
  const verdict = before == null || after == null ? "unknown" : before === after ? "kept" : "changed";
  return { before, after, verdict };
}

const NAME: Record<PaneKey, string> = { bytetrack: BASE_SHORT, ours: OURS_SHORT };

export function verdictText(o: Outcome): string {
  if (o.verdict === "kept") return `Kept #${o.before} ✓`;
  if (o.verdict === "changed") return `Changed #${o.before} → #${o.after}`;
  return o.before == null ? "No number before the hide" : "No number yet after coming back";
}

function clause(o: Outcome): string {
  if (o.verdict === "kept") return `comes back as #${o.before} (kept)`;
  if (o.verdict === "changed") return `comes back as #${o.after}, a new number (was #${o.before})`;
  return o.before == null ? "had no number before the hide" : "has no number yet after coming back";
}

export function headlineOf(m: HideMoment, out: Record<PaneKey, Outcome>): string {
  const who = `Person ${m.person} walks behind ${m.occluder != null ? `person ${m.occluder}` : "someone"}.`;
  const same = out.bytetrack.verdict === out.ours.verdict && out.bytetrack.after === out.ours.after && out.bytetrack.before === out.ours.before;
  const tail = same && out.ours.verdict !== "unknown" ? ` Both trackers: ${clause(out.ours)}.` : ` ${NAME.bytetrack}: ${clause(out.bytetrack)}. ${NAME.ours}: ${clause(out.ours)}.`;
  return who + tail;
}

export function storyOf(frames: SimFrame[], m: HideMoment): MomentStory {
  const out = { bytetrack: outcome(frames, m, "bytetrack"), ours: outcome(frames, m, "ours") };
  return { m, out, headline: headlineOf(m, out), verdicts: { bytetrack: verdictText(out.bytetrack), ours: verdictText(out.ours) } };
}

/** Pick the moment to show: ByteTrack changed and ours kept first; else any where they differ; else the first. */
export function pickStory(frames: SimFrame[]): { story: MomentStory | null; all: MomentStory[]; kind: "ours-better" | "differ" | "same" | "none" } {
  const all = hideMoments(frames).map((m) => storyOf(frames, m)).filter((s) => s.out.bytetrack.verdict !== "unknown" && s.out.ours.verdict !== "unknown");
  const better = all.find((s) => s.out.bytetrack.verdict === "changed" && s.out.ours.verdict === "kept");
  if (better) return { story: better, all, kind: "ours-better" };
  const differ = all.find((s) => s.out.bytetrack.verdict !== s.out.ours.verdict);
  if (differ) return { story: differ, all, kind: "differ" };
  return { story: all[0] ?? null, all, kind: all.length ? "same" : "none" };
}

export function tally(all: MomentStory[]) {
  const t = Object.fromEntries(PANES.map((p) => [p, { kept: 0, changed: 0 }])) as Record<PaneKey, { kept: number; changed: number }>;
  for (const s of all) for (const p of PANES) { const v = s.out[p].verdict; if (v === "kept" || v === "changed") t[p][v]++; }
  return t;
}
