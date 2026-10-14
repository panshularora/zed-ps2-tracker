import { useEffect, useMemo, useState } from "react";
import { GuidedPlayer } from "../guided/GuidedPlayer";
import { fromScenario } from "../guided/model";
import {
  BENCHMARK_SLOTS, fetchIndex, fetchScenario, posterUrl, type ScenarioData, type ScenarioMeta, videoUrl,
} from "../lib/scenarios";
import { useAsync } from "../lib/useAsync";

export function Tag({ kind }: { kind: "illustrative" | "measured" }) {
  return kind === "measured"
    ? <span className="tag tag-measured">Measured · human labels</span>
    : <span className="tag tag-illustrative">Illustrative · no ground truth</span>;
}

export function ScenarioRail({ items, selected, onSelect }: { items: ScenarioMeta[]; selected: string | null; onSelect: (id: string) => void }) {
  return (
    <nav className="rail" aria-label="Scenarios">
      {BENCHMARK_SLOTS.map((b) => (
        <div key={b.id} className="scard pending" data-testid="bench-slot" aria-disabled>
          <div className="thumb pending-thumb">pending</div>
          <div className="scard-body">
            <div className="scard-title">{b.problem}</div>
            <Tag kind="measured" />
            <div className="credit">Data pending · {b.licence}</div>
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
      <p className="player-caption"><strong>{meta.title}.</strong> {meta.caption} <Tag kind={meta.kind} /></p>
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
  const [sel, setSel] = useState<string | null>(null);
  const [data, setData] = useState<ScenarioData | null>(null);
  const items = idx.data ?? [];
  const current = sel ?? items[0]?.id ?? null;
  useEffect(() => {
    if (!current) return;
    let live = true;
    fetchScenario(current).then((d) => live && setData(d)).catch(() => live && setData(null));
    return () => { live = false; };
  }, [current]);
  return (
    <div className="scenarios">
      <ScenarioRail items={items} selected={current} onSelect={setSel} />
      <div>
        {data && data.meta.id === current ? <ScenarioPlayer key={current} data={data} /> : <p className="muted">Loading scenario…</p>}
      </div>
    </div>
  );
}
