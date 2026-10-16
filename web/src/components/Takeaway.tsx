// Bottom-of-page takeaway: one 32px number from committed data, a short line under it, and a
// collapsed "What am I looking at?" note. Nothing is typed in by hand: numbers come from
// public/verify/MOT17-*.json (per labelled person) or the recorded gate scene run.
import { useEffect, useState } from "react";
import { readStream, recordingUrl, type SimFrame } from "../lib/gateSim";
import { pickStory, tally } from "../compare/narrative";
import { fetchVerify } from "../identity/verify";

export interface TakeawayData { big: string; line: string }

export async function motTakeaway(): Promise<TakeawayData | null> {
  const docs = await Promise.all(["MOT17-04", "MOT17-09"].map(fetchVerify));
  let n = 0, o = 0, b = 0;
  for (const d of docs) {
    const po = d?.people?.ours, pb = d?.people?.baseline;
    if (!po || !pb || po.labelled !== pb.labelled) return null;
    n += po.labelled; o += po.kept; b += pb.kept;
  }
  return { big: `${o} of ${n}`, line: `labelled people kept one number with ours on MOT17-04 + MOT17-09 (ByteTrack: ${b} of ${n})` };
}

export async function gateTakeaway(): Promise<TakeawayData | null> {
  const r = await fetch(recordingUrl("crowd"));
  if (!r.ok) return null;
  const fs: SimFrame[] = [];
  await readStream(r, (l) => { if (l.type === "frame") fs.push(l); });
  const p = pickStory(fs);
  if (!p.all.length) return null;
  const t = tally(p.all);
  return { big: `${t.ours.kept} of ${p.all.length}`, line: `hidden people came back with their number with ours in the synthetic run (ByteTrack: ${t.bytetrack.kept} of ${p.all.length})` };
}

export const ABOUT: Record<string, string> = {
  "/": "A tracker gives every person a number. This site shows when that number survives someone being hidden, and when it does not.",
  "/dashboard/what-we-use": "The datasets, detector and trackers behind every number on this site, and whether each one has run yet.",
  "/dashboard/live": "Synthetic people with exact ground truth. Both trackers get the same boxes; chips show the number each tracker gave.",
  "/dashboard/compare": "One person walks behind another. Left is ByteTrack, right is ours. The verdict under each pane is computed from the numbers before and after.",
  "/dashboard/studio": "A real clip with our tracker's own numbers (nothing relabelled). The timeline has one bar per person; markers are ID changes and flagged swaps.",
  "/dashboard/scenarios": "Six short Pexels clips plus the human-labelled MOT17 clips. Pick one to open the large player.",
  "/dashboard/results": "Benchmark scores read from docs/results/summary.json. Provisional until the GPU rerun.",
  "/dashboard/experiments": "Run a tracker config on a clip yourself (needs the backend).",
  "/dashboard/evidence": "Per-clip charts behind the results table.",
  "/dashboard/architecture": "How the pieces connect: detector, tracker, visibility rule, scorer.",
};

export function Takeaway({ path }: { path: string }) {
  const [d, setD] = useState<TakeawayData | null>(null);
  const gate = path === "/dashboard/live" || path === "/dashboard/compare";
  useEffect(() => {
    let live = true;
    (gate ? gateTakeaway() : motTakeaway()).then((x) => live && setD(x)).catch(() => live && setD(null));
    return () => { live = false; };
  }, [gate]);
  const about = ABOUT[path === "/dashboard" ? "/" : path];
  return (
    <section className="takeaway" data-testid="takeaway">
      {d && <div><div className="takeaway-big tabular" data-testid="takeaway-big">{d.big}</div><div className="takeaway-line">{d.line}</div></div>}
      {about && <details className="about" data-testid="about"><summary>What am I looking at?</summary><p>{about}</p></details>}
    </section>
  );
}
