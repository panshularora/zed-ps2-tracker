import { useEffect, useMemo, useState } from "react";
import type { IndexedClip } from "@viewer/data";
import { GuidedPlayer } from "../../guided/GuidedPlayer";
import { fromScenario, type ReplayData } from "../../guided/model";
import { BENCHMARK_CLIPS, fromViewerClips } from "../../guided/viewer";
import { fetchIndex, fetchScenario, type ScenarioMeta, videoUrl } from "../../lib/scenarios";
import { prettyTracker } from "../lib/journey";

const DEFAULT = "subway_turnstile";

/** Benchmark pairs (ByteTrack-like vs another tracker) available in the exported viewer clips. */
export function benchmarkPairs(clips: IndexedClip[]): { name: string; base: IndexedClip; ours: IndexedClip }[] {
  const out = [];
  for (const name of BENCHMARK_CLIPS) {
    const cs = clips.filter((c) => c.clip.startsWith(name));
    const base = cs.find((c) => /bytetrack/i.test(c.tracker) && !/reid|vis/i.test(c.tracker));
    const rest = cs.filter((c) => c !== base && c.tracker !== "gt");
    const ours = rest.find((c) => /vis/i.test(c.tracker)) ?? rest[0];
    if (base && ours) out.push({ name, base, ours });
  }
  return out;
}

export function GuidedStudio({ clips, onNext }: { clips: IndexedClip[]; onNext: () => void }) {
  const [index, setIndex] = useState<ScenarioMeta[]>([]);
  const [sel, setSel] = useState<string>(DEFAULT);
  const [scen, setScen] = useState<ReplayData | null>(null);
  const pairs = useMemo(() => benchmarkPairs(clips), [clips]);
  useEffect(() => { fetchIndex().then(setIndex).catch(() => setIndex([])); }, []);
  const bench = pairs.find((p) => `bench:${p.name}` === sel);
  useEffect(() => {
    if (sel.startsWith("bench:")) return;
    let live = true;
    fetchScenario(sel).then((d) => live && setScen(fromScenario(d, videoUrl(d.meta.id)))).catch(() => live && setScen(null));
    return () => { live = false; };
  }, [sel]);
  const data = useMemo(
    () => (bench ? fromViewerClips(bench.base, bench.ours, { baseline: prettyTracker(bench.base.tracker).title, ours: prettyTracker(bench.ours.tracker).title }) : scen),
    [bench, scen],
  );
  return (
    <div className="h-full overflow-auto p-4" data-testid="guided-studio">
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <h2 className="text-[16px] font-semibold text-ink m-0">Watch a replay</h2>
        <label className="flex items-center gap-2 text-[13px] text-mute">
          Clip
          <select value={sel} onChange={(e) => setSel(e.target.value)} className="bg-raised border border-line rounded-sm px-2 py-1 text-ink" data-testid="studio-clip">
            {BENCHMARK_CLIPS.map((n) => {
              const has = pairs.some((p) => p.name === n);
              return <option key={n} value={`bench:${n}`} disabled={!has}>{n} · human labels{has ? "" : " (not exported yet)"}</option>;
            })}
            {index.map((m) => <option key={m.id} value={m.id}>{m.title} · illustrative</option>)}
          </select>
        </label>
        <button type="button" className="btn ml-auto" onClick={onNext}>Next: compare side by side ▸</button>
      </div>
      {data ? <GuidedPlayer key={data.id} data={data} /> : <p className="text-mute">Loading clip…</p>}
      {data && data.kind === "illustrative" && (
        <p className="credit">Illustrative only: no ground truth on this Pexels clip, so the replay shows where the two trackers disagree, not who is right. Both ran on identical detections.</p>
      )}
    </div>
  );
}
