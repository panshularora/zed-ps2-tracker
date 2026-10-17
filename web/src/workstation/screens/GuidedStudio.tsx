import { useEffect, useMemo, useState } from "react";
import type { IndexedClip } from "@viewer/data";
import { IdentityPlayer } from "../../identity/IdentityPlayer";
import { ScenarioIntro, orderScenarios } from "../../pages/Scenarios";
import { fromScenario, type ReplayData } from "../../guided/model";
import { BENCHMARK_CLIPS, benchmarkPairs, benchmarkReplay } from "../../guided/viewer";
import { fetchIndex, fetchScenario, type ScenarioMeta, videoUrl } from "../../lib/scenarios";

const DEFAULT = "subway_turnstile"; // a clip where someone is hidden and comes back

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
    <div className="h-full overflow-auto p-4" data-testid="guided-studio">
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <h2 className="text-[16px] font-semibold text-ink m-0">Our tracker: one person, one number</h2>
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
        <a href="/dashboard/live" className="ws-btn ml-auto" data-testid="compare-link" onClick={(e) => { e.preventDefault(); onNext(); }}>Compare with baseline in the gate scene ▸</a>
      </div>
      {data && <div className="mb-2"><ScenarioIntro id={bench ? bench.name : data.id} /></div>}
      {data ? <IdentityPlayer key={data.id} data={data} /> : <p className="text-mute">Loading clip…</p>}
      {data && data.kind === "illustrative" && (
        <p className="credit">Illustrative only: our tracker (BoT-SORT + visibility) exactly as it ran on this Pexels clip; numbers are not relabelled. No ground truth, not a measured score. Goal we measure against: one body, one number, from approach to exit.</p>
      )}
    </div>
  );
}
