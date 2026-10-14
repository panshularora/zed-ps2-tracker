import { useEffect, useMemo, useState } from "react";
import { GuidedPlayer } from "../guided/GuidedPlayer";
import { fromScenario, thumbUrlFor } from "../guided/model";
import { Tag } from "../guided/Tag";
import { benchmarkReplay, loadBenchmarkPairs, type BenchPair } from "../guided/viewer";
import {
  fetchIndex, fetchScenario, type ScenarioData, type ScenarioMeta, videoUrl,
} from "../lib/scenarios";
import { useAsync } from "../lib/useAsync";
import LiveRun from "./Demo";

export { Tag };

const MOT17_LIC = "MOTChallenge: non-commercial use, attribution (Milan et al. 2016)";
export const BENCH_CARDS = [
  { id: "MOT17-09", problem: "The problem: a person comes back with a new number", licence: MOT17_LIC },
  { id: "MOT17-04", problem: "Measured number changes, busy street", licence: MOT17_LIC },
  { id: "MOT20-02", problem: "Measured number changes, very dense crowd", licence: "MOTChallenge: non-commercial use, attribution (Dendorfer et al. 2020)" },
];
export const benchKey = (name: string) => `bench:${name}`;

export function Thumb({ id }: { id: string }) {
  const [bad, setBad] = useState(false);
  return (
    <div className="thumb thumb-frame">
      {bad ? <span className="thumb-ph" aria-hidden /> : <img src={thumbUrlFor(id)} alt="" loading="lazy" onError={() => setBad(true)} />}
    </div>
  );
}

export function ScenarioRail({ items, bench = [], selected, onSelect }: { items: ScenarioMeta[]; bench?: string[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <nav className="rail" aria-label="Scenarios">
      {items.map((m) => (
        <button key={m.id} type="button" className={`scard${selected === m.id ? " active" : ""}`} onClick={() => onSelect(m.id)} data-testid="scenario-card">
          <Thumb id={m.id} />
          <div className="scard-body">
            <div className="scard-title">{m.problem}</div>
            <Tag kind={m.kind} />
          </div>
        </button>
      ))}
      {/* benchmark clips have no video here: one line each, below the clips that play */}
      {BENCH_CARDS.map((b) => {
        const has = bench.includes(b.id);
        return (
          <button key={b.id} type="button" disabled={!has} title={b.licence}
            className={`bench-line${selected === benchKey(b.id) ? " active" : ""}`}
            onClick={() => has && onSelect(benchKey(b.id))} data-testid={has ? "bench-card" : "bench-slot"}>
            <b>{b.id}</b> · Measured · {has ? "video not exported yet (boxes only)" : "not exported yet"}
          </button>
        );
      })}
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

/** The old Demo tab's live job flow, mounted only when opened (its keys would clash with the player). */
export function LiveRunPanel() {
  const [open, setOpen] = useState(false);
  return (
    <details className="live-run" data-testid="live-run" style={{ marginTop: 16 }} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Run the tracker yourself on a clip (CPU; needs the backend and <code>make models</code>)</summary>
      {open && <div style={{ marginTop: 12 }}><LiveRun /></div>}
    </details>
  );
}

export const OPENER = "MOT17-09";
export function pickOpener(items: ScenarioMeta[], pairs: BenchPair[] | null): string | null {
  if (pairs === null) return null; // wait for the benchmark list before choosing
  if (pairs.some((p) => p.name === OPENER)) return benchKey(OPENER);
  return items.find((m) => m.id === "subway_turnstile")?.id ?? items[0]?.id ?? null;
}

export default function Scenarios() {
  const idx = useAsync(fetchIndex);
  const [pairs, setPairs] = useState<BenchPair[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [data, setData] = useState<ScenarioData | null>(null);
  useEffect(() => { loadBenchmarkPairs().then(setPairs).catch(() => setPairs([])); }, []);
  const items = idx.data ?? [];
  // open on the measured problem (MOT17-09, human labels, boxes only); turnstile when it is not exported
  const opener = pickOpener(items, pairs);
  const current = sel ?? opener;
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
        <LiveRunPanel />
      </div>
    </div>
  );
}
