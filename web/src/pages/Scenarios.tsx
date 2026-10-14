import { useEffect, useMemo, useState } from "react";
import { GuidedPlayer } from "../guided/GuidedPlayer";
import { fromScenario } from "../guided/model";
import { Tag } from "../guided/Tag";
import { benchmarkReplay, loadBenchmarkPairs, type BenchPair } from "../guided/viewer";
import {
  fetchIndex, fetchScenario, posterUrl, type ScenarioData, type ScenarioMeta, videoUrl,
} from "../lib/scenarios";
import { useAsync } from "../lib/useAsync";

export { Tag };

const MOT17_LIC = "MOTChallenge: non-commercial use, attribution (Milan et al. 2016)";
export const BENCH_CARDS = [
  { id: "MOT17-09", problem: "The problem: a person comes back with a new number", licence: MOT17_LIC },
  { id: "MOT17-04", problem: "Measured number changes, busy street", licence: MOT17_LIC },
  { id: "MOT20-02", problem: "Measured number changes, very dense crowd", licence: "MOTChallenge: non-commercial use, attribution (Dendorfer et al. 2020)" },
];
export const benchKey = (name: string) => `bench:${name}`;

export function ScenarioRail({ items, bench = [], selected, onSelect }: { items: ScenarioMeta[]; bench?: string[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <nav className="rail" aria-label="Scenarios">
      {BENCH_CARDS.map((b) => bench.includes(b.id) ? (
        <button key={b.id} type="button" className={`scard${selected === benchKey(b.id) ? " active" : ""}`} onClick={() => onSelect(benchKey(b.id))} data-testid="bench-card">
          <div className="thumb pending-thumb">{b.id} · boxes only</div>
          <div className="scard-body">
            <div className="scard-title">{b.problem}</div>
            <Tag kind="measured" />
            <div className="credit">{b.licence}</div>
          </div>
        </button>
      ) : (
        <div key={b.id} className="scard pending" data-testid="bench-slot" aria-disabled>
          <div className="thumb pending-thumb">{b.id} · pending</div>
          <div className="scard-body">
            <div className="scard-title">{b.problem}</div>
            <Tag kind="measured" />
            <div className="credit">Not exported yet · {b.licence}</div>
          </div>
        </div>
      ))}
      {items.map((m) => (
        <button key={m.id} type="button" className={`scard${selected === m.id ? " active" : ""}`} onClick={() => onSelect(m.id)} data-testid="scenario-card">
          <img className="thumb" src={posterUrl(m.id)} alt="" loading="lazy" />
          <div className="scard-body">
            <div className="scard-title">{m.problem}</div>
            <Tag kind={m.kind} />
          </div>
        </button>
      ))}
    </nav>
  );
}

export function ScenarioPlayer({ data }: { data: ScenarioData }) {
  const { meta } = data;
  const replay = useMemo(() => fromScenario(data, videoUrl(meta.id)), [data, meta.id]);
  return (
    <div className="player" data-testid="scenario-player">
      <GuidedPlayer data={replay} />
      <div className="muted" data-testid="switch-counts" style={{ fontSize: 13, marginTop: 8 }}>
        Suspected ID switches in the first {meta.covered_s.toFixed(1)} s: ByteTrack {meta.id_switches.baseline} · ours {meta.id_switches.ours} · {replay.moments.length} key moment{replay.moments.length === 1 ? "" : "s"} shown
      </div>
      <p className="credit">
        Illustrative only: no ground truth, so these are disagreements between the two trackers, not a benchmark result. Both trackers ran on identical detections.
        The counting line is a fixed-fraction line chosen by geometry (the candidate people cross most, both trackers summed).
        Source: <a href={meta.source} target="_blank" rel="noreferrer">Pexels</a> · {meta.licence} · faces blurred · detector {meta.detector} (CPU)
        {meta.stride > 1 ? `, 1 in ${meta.stride} frames tracked` : ""}.
      </p>
    </div>
  );
}

export default function Scenarios() {
  const idx = useAsync(fetchIndex);
  const [pairs, setPairs] = useState<BenchPair[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [data, setData] = useState<ScenarioData | null>(null);
  useEffect(() => { loadBenchmarkPairs().then(setPairs).catch(() => setPairs([])); }, []);
  const items = idx.data ?? [];
  const opener = pairs?.some((p) => p.name === "MOT17-09") ? benchKey("MOT17-09") : (items[0]?.id ?? null);
  const current = sel ?? (pairs == null ? null : opener);
  const pair = pairs?.find((p) => benchKey(p.name) === current);
  const bench = useMemo(() => (pair ? benchmarkReplay(pair) : null), [pair]);
  useEffect(() => {
    if (!current || current.startsWith("bench:")) return;
    let live = true;
    fetchScenario(current).then((d) => live && setData(d)).catch(() => live && setData(null));
    return () => { live = false; };
  }, [current]);
  return (
    <div className="scenarios">
      <ScenarioRail items={items} bench={(pairs ?? []).map((p) => p.name)} selected={current} onSelect={setSel} />
      <div>
        {bench ? (
          <div className="player" data-testid="scenario-player">
            <GuidedPlayer key={current} data={bench} />
            <p className="credit">Measured: number changes come from human labels (MOTChallenge ground truth). Video frames are not on this server, so only boxes are drawn. {pair && `${bench.runs.baseline.label} vs ${bench.runs.ours.label}, exported by eval.export_viewer.`}</p>
          </div>
        ) : data && data.meta.id === current ? <ScenarioPlayer key={current} data={data} /> : <p className="muted">Loading scenario…</p>}
      </div>
    </div>
  );
}
