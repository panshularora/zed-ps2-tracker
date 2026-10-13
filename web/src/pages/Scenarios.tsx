import { useCallback, useEffect, useRef, useState } from "react";
import {
  BENCHMARK_SLOTS, type Disagreement, drawBoxes, fetchIndex, fetchScenario, frameAt, markerLabel, markers,
  posterUrl, type RunKey, type ScenarioData, type ScenarioMeta, seekBefore, timeOf, videoUrl,
} from "../lib/scenarios";
import { useAsync } from "../lib/useAsync";

const RING_S = 0.8;

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
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [run, setRun] = useState<RunKey>("ours");
  const [t, setT] = useState(0);
  const marks = markers(data);
  const differ = meta.disagreements > 0 || meta.id_switches.baseline !== meta.id_switches.ours;
  const frame = frameAt(meta, t);
  const active: Disagreement | null =
    frame == null ? null : (marks.find((d) => Math.abs(timeOf(meta, d.frame) - t) <= RING_S) ?? null);

  const draw = useCallback(() => {
    const v = video.current, c = canvas.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    c.width = c.clientWidth;
    c.height = c.clientHeight;
    ctx.clearRect(0, 0, c.width, c.height);
    const f = frameAt(meta, v.currentTime);
    if (f == null) return;
    const boxes = data.runs[run].frames[String(f)] ?? [];
    drawBoxes(ctx, boxes, c.width / meta.width, c.height / meta.height, active?.box ?? null);
  }, [data, meta, run, active]);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const v = video.current;
      if (v) {
        if (v.currentTime > meta.covered_s) v.currentTime = 0; // loop over the tracked range
        setT(v.currentTime);
      }
      draw();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [draw, meta.covered_s]);

  const seek = (s: number) => {
    if (video.current) video.current.currentTime = s;
    setT(s);
  };

  return (
    <div className="player" data-testid="scenario-player">
      <p className="player-caption"><strong>{meta.title}.</strong> {meta.caption}</p>
      <div className="stage">
        <video ref={video} src={videoUrl(meta.id)} muted loop playsInline autoPlay aria-label={meta.title} />
        <canvas ref={canvas} aria-hidden />
        <span className="badge">{data.runs[run].label}{frame != null ? ` · frame ${frame}` : ""}</span>
        {active && <span className="marker-label" data-testid="marker-label">{markerLabel(meta.kind, active)}</span>}
      </div>
      <div className="seek" aria-label="Seek">
        <input type="range" min={0} max={meta.covered_s} step={0.01} value={Math.min(t, meta.covered_s)} onChange={(e) => seek(Number(e.target.value))} aria-label="Seek position" />
        <div className="seek-track"><div className="seek-fill" style={{ width: `${(100 * Math.min(t, meta.covered_s)) / meta.covered_s}%` }} /></div>
        {marks.map((d) => (
          <button key={d.frame} type="button" className="seek-marker" style={{ left: `${(100 * timeOf(meta, d.frame)) / meta.covered_s}%` }}
            title="Trackers disagree here: jump to 2 s before" aria-label={`Disagreement at ${timeOf(meta, d.frame).toFixed(1)} s`} data-testid="seek-marker"
            onClick={() => seek(seekBefore(meta, d.frame))} />
        ))}
      </div>
      <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
        <div className="toggle" role="group" aria-label="Tracker">
          {(["baseline", "ours"] as const).map((k) => (
            <button key={k} type="button" className={`btn${run === k ? " on-" + k : ""}`} aria-pressed={run === k} disabled={!differ && k !== run} onClick={() => setRun(k)}>
              {k === "baseline" ? "Baseline (ByteTrack)" : "Ours"}
            </button>
          ))}
        </div>
        <div className="muted" data-testid="switch-counts" style={{ fontSize: 13 }}>
          Suspected ID switches in the first {meta.covered_s.toFixed(1)} s: ByteTrack {meta.id_switches.baseline} · ours {meta.id_switches.ours} · {marks.length} disagreement moment{marks.length === 1 ? "" : "s"}
          {!differ && " · trackers agree on this clip"}
        </div>
      </div>
      <p className="credit">
        Illustrative only: no ground truth, so these counts are a heuristic, not a benchmark result. Both trackers ran on identical detections.
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
