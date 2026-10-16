import { useEffect, useMemo, useState, type ReactNode } from "react";
import { IdentityPlayer } from "../identity/IdentityPlayer";
import { INTROS, SCENARIO_ORDER } from "../identity/intros";
import { Link } from "react-router-dom";
import { fromScenario, thumbUrlFor } from "../guided/model";
import { Tag } from "../guided/Tag";
import { benchmarkReplay, loadBenchmarkPairs, type BenchPair } from "../guided/viewer";
import {
  fetchIndex, fetchScenario, type ScenarioData, type ScenarioMeta, videoUrl,
} from "../lib/scenarios";
import { useAsync } from "../lib/useAsync";
import LiveRun from "./Demo";
import type { Truth } from "../compare/ComparePlayer";
import { fetchClip, fetchClips } from "../workstation/api";
import type { GEvent, ReplayData } from "../guided/model";

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
      {orderScenarios(items).map((m) => (
        <button key={m.id} type="button" className={`scard${selected === m.id ? " active" : ""}`} onClick={() => onSelect(m.id)} data-testid="scenario-card">
          <Thumb id={m.id} />
          <div className="scard-body">
            <div className="scard-title">{m.problem}</div>
            <Tag kind={m.kind} />
          </div>
        </button>
      ))}
      {BENCH_CARDS.slice(0, 1).map((b) => {
        const has = bench.includes(b.id);
        return (
          <button key={b.id} type="button" disabled={!has} title={b.licence} className={`scard${selected === benchKey(b.id) ? " active" : ""}`}
            onClick={() => has && onSelect(benchKey(b.id))} data-testid={has ? "bench-card" : "bench-slot"}>
            <Thumb id={b.id} />
            <div className="scard-body">
              <div className="scard-title">{b.id}: real labelled street video (boxes only)</div>
              <Tag kind="measured" />
            </div>
          </button>
        );
      })}
      {BENCH_CARDS.slice(1).map(benchLine)}
    </nav>
  );
  function benchLine(b: (typeof BENCH_CARDS)[number]) {
    const has = bench.includes(b.id);
    return (
      <button key={b.id} type="button" disabled={!has} title={b.licence}
        className={`bench-line${selected === benchKey(b.id) ? " active" : ""}`}
        onClick={() => has && onSelect(benchKey(b.id))} data-testid={has ? "bench-card" : "bench-slot"}>
        <b>{b.id}</b> · Measured · {has ? "video not exported yet (boxes only)" : "not exported yet"}
      </button>
    );
  }
}

export function orderScenarios(items: ScenarioMeta[]): ScenarioMeta[] {
  const rank = (id: string) => { const i = SCENARIO_ORDER.indexOf(id); return i < 0 ? SCENARIO_ORDER.length : i; };
  return [...items].sort((a, b) => rank(a.id) - rank(b.id));
}

export function ScenarioIntro({ id }: { id: string }) {
  const it = INTROS[id] ?? INTROS[id.replace(/ .*/, "")];
  if (!it) return null;
  return (
    <div className="scn-intro" data-testid="scenario-intro">
      <div><b>What you&apos;ll see</b>{it.see}</div>
      <div><b>What to watch for</b>{it.watch}</div>
      <div><b>Takeaway</b>{it.takeaway}</div>
    </div>
  );
}

/** One tracker (ours) only, centred on each person keeping one number; the comparison lives in the gate scene. */
export function ScenarioView({ replay, credit, introId }: { replay: ReplayData; truth?: Truth | null; credit: ReactNode; introId?: string }) {
  return (
    <div className="player" data-testid="scenario-player">
      <div className="g-head" style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <h3 className="g-title" style={{ margin: 0 }}>{replay.title}</h3>
        <Tag kind={replay.kind} />
        <Link to={`/dashboard/scenarios/compare?clip=${encodeURIComponent(replay.kind === "measured" ? `bench:${replay.id}` : replay.id)}`} className="muted" style={{ marginLeft: "auto", fontSize: 13 }} data-testid="real-compare-link">Compare on real clips →</Link>
        <Link to="/dashboard/live" className="muted" style={{ fontSize: 13 }} data-testid="compare-link">Gate scene (synthetic) →</Link>
      </div>
      <ScenarioIntro id={introId ?? replay.id} />
      <IdentityPlayer key={replay.id} data={replay} />
      <p className="credit">{credit}</p>
    </div>
  );
}

export function ScenarioPlayer({ data }: { data: ScenarioData }) {
  const { meta } = data;
  const replay = useMemo(() => fromScenario(data, videoUrl(meta.id)), [data, meta.id]);
  return (
    <ScenarioView replay={replay} credit={<>
      Illustrative only: no ground truth on this clip, so this shows what our tracker did, not a measured score. Every ID change our tracker made is in the timeline.
      The counting line is a fixed-fraction line chosen by geometry (the candidate people cross most, both trackers summed).
      Source: <a href={meta.source} target="_blank" rel="noreferrer">Pexels</a> · {meta.licence} · faces blurred · detector {meta.detector} (CPU)
      {meta.stride > 1 ? `, 1 in ${meta.stride} frames tracked` : ""}.
    </>} />
  );
}

/** Gate crossings from the human-label export (eval.export_viewer *.gt.json.gz), if committed. */
export async function loadTruth(name: string): Promise<Truth | null> {
  const refs = await fetchClips();
  const ref = refs.find((r) => r.clip === name && r.tracker === "gt");
  if (!ref) return null;
  const gt = await fetchClip(ref.file);
  const events: GEvent[] = gt.events.flatMap((e) => (e.type === "cross" ? [{ t: "cross" as const, f: e.f, id: e.id, dir: e.dir }] : []));
  return { events, label: "human labels" };
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
  const first = orderScenarios(items)[0]?.id;
  if (first) return first;
  return pairs.some((p) => p.name === OPENER) ? benchKey(OPENER) : null;
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
  const [truth, setTruth] = useState<{ name: string; t: Truth | null } | null>(null);
  useEffect(() => {
    if (!pair) return;
    let live = true;
    loadTruth(pair.name).then((t) => live && setTruth({ name: pair.name, t })).catch(() => live && setTruth({ name: pair.name, t: null }));
    return () => { live = false; };
  }, [pair]);
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
          <div>
            <ScenarioView key={current} replay={bench} introId={pair?.name} truth={truth && truth.name === pair?.name ? truth.t : null}
              credit={<>Measured: ID changes are checked against human labels (MOTChallenge ground truth). MOTChallenge frames are not licensed for this repo, so only boxes are drawn on an empty frame. Tracker shown: {bench.runs.ours.label} (exported by eval.export_viewer); our visibility variant was not exported for this clip.</>} />
          </div>
        ) : data && data.meta.id === current ? <ScenarioPlayer key={current} data={data} /> : <p className="muted">Loading scenario…</p>}
        <LiveRunPanel />
      </div>
    </div>
  );
}
