import { useEffect, useMemo, useState } from "react";
import type { IndexedClip } from "@viewer/data";
import { IdentityPlayer } from "../../identity/IdentityPlayer";
import { ScenarioIntro, orderScenarios } from "../../pages/Scenarios";
import { fromScenario, type ReplayData } from "../../guided/model";
import { BENCHMARK_CLIPS, benchmarkPairs, benchmarkReplay } from "../../guided/viewer";
import { OURS_NAME } from "../../lib/ours";
import { fetchIndex, fetchScenario, type ScenarioMeta, videoUrl } from "../../lib/scenarios";

const DEFAULT = "mexico_city_entrance"; // opens on a clip where every checked return kept its number

export { benchmarkPairs };

export function GuidedStudio({ clips, onNext }: { clips: IndexedClip[]; onNext: () => void }) {
  const [index, setIndex] = useState<ScenarioMeta[]>([]);
  const [pick, setSel] = useState<string | null>(null);
  const [scen, setScen] = useState<ReplayData | null>(null);
  const pairs = useMemo(() => benchmarkPairs(clips), [clips]);
  // opener: MOT17-09 (measured) explains the problem; Pexels turnstile if it is not exported
  const sel = pick ?? DEFAULT; // open on a clip that plays; benchmarks are boxes-only here
  useEffect(() => { fetchIndex().then(setIndex).catch(() => setIndex([])); }, []);
  const bench = pairs.find((p) => `bench:${p.name}` === sel);
  useEffect(() => {
    if (sel.startsWith("bench:")) return;
    let live = true;
    fetchScenario(sel).then((d) => live && setScen(fromScenario(d, videoUrl(d.meta.id)))).catch(() => live && setScen(null));
    return () => { live = false; };
  }, [sel]);
  const data = useMemo(
    () => (bench ? benchmarkReplay(bench) : scen),
    [bench, scen],
  );
  return (
    <div className="h-full overflow-auto px-4 pt-0 pb-4" data-testid="guided-studio">
      <div className="flex flex-wrap items-center gap-3 mb-1">
        <h2 className="text-[22px] font-semibold text-ink m-0">Studio: {OURS_NAME}</h2>
        <label className="flex items-center gap-2 text-[13px] text-mute">
          Clip
          <select value={sel} onChange={(e) => setSel(e.target.value)} className="bg-raised border border-line rounded-sm px-2 py-1 text-ink" data-testid="studio-clip">
            {BENCHMARK_CLIPS.map((n) => {
              const has = pairs.some((p) => p.name === n);
              return <option key={n} value={`bench:${n}`} disabled={!has}>{n} · human labels{has ? "" : " (not exported yet)"}</option>;
            })}
            {orderScenarios(index).map((m) => <option key={m.id} value={m.id}>{m.title} · illustrative</option>)}
          </select>
        </label>
        <a href="/dashboard/live" className="ws-btn ml-auto" data-testid="compare-link" onClick={(e) => { e.preventDefault(); onNext(); }}>Compare with ByteTrack ▸</a>
      </div>
      {data ? <IdentityPlayer key={data.id} data={data} /> : <p className="text-mute">Loading clip…</p>}
      {data && <details className="about mt-2"><summary>About this clip</summary><ScenarioIntro id={bench ? bench.name : data.id} /></details>}
      {data && data.kind === "illustrative" && (
        <p className="credit">Illustrative only: the tracker&apos;s own numbers on this Pexels clip, nothing relabelled. No ground truth here; "kept" comes from a heuristic check, not a measured score.</p>
      )}
    </div>
  );
}
