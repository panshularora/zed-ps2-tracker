// Side-by-side replay of one simulated scene: ByteTrack left, ours right, the same people in both.
import { useEffect, useMemo, useRef, useState } from "react";
import { AMBER, BLUE } from "./model";
import { PANE_ORDER, countsAt, latestChange, personHeight, verdict, whyQuestion, type PaneKey, type SceneData, type Verdict } from "./scene";

const GREY = "#8A929C";
const FIG = "#C9CED6";
const PANE_COLOUR: Record<PaneKey, string> = { bytetrack: GREY, ours: BLUE };

function drawFigure(ctx: CanvasRenderingContext2D, x: number, y: number, hidden: boolean, vis: number, walk: number) {
  const h = personHeight(y);
  const headR = h * 0.11;
  const legH = h * 0.42;
  const torsoH = h * 0.34;
  const shW = h * 0.13;
  const sw = Math.sin(walk) * h * 0.07;
  ctx.save();
  ctx.globalAlpha = hidden ? 0.35 : 0.45 + 0.55 * Math.min(1, vis);
  ctx.strokeStyle = FIG;
  ctx.fillStyle = FIG;
  ctx.lineCap = "round";
  if (hidden) ctx.setLineDash([4, 3]);
  ctx.lineWidth = Math.max(2, h * 0.07);
  ctx.beginPath();
  ctx.moveTo(x - shW * 0.4, y - legH);
  ctx.lineTo(x - shW * 0.5 - sw, y);
  ctx.moveTo(x + shW * 0.4, y - legH);
  ctx.lineTo(x + shW * 0.5 + sw, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.roundRect(x - shW, y - legH - torsoH, shW * 2, torsoH, 3);
  if (hidden) ctx.stroke(); else ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y - legH - torsoH - headR - 1, headR, 0, Math.PI * 2);
  if (hidden) ctx.stroke(); else ctx.fill();
  ctx.restore();
}

function chip(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, colour: string) {
  ctx.font = "600 13px Inter, system-ui, sans-serif";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = colour;
  ctx.fillRect(x - w / 2, y - 18, w, 18);
  ctx.fillStyle = "#0E1013";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y - 9);
}

function drawPane(c: HTMLCanvasElement, d: SceneData, pane: PaneKey, f: number, focus: number | null, ring: number | null) {
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#161A1F";
  ctx.fillRect(0, 0, d.width, d.height);
  ctx.strokeStyle = AMBER;
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 6]);
  ctx.beginPath();
  ctx.moveTo(0, d.gate_y);
  ctx.lineTo(d.width, d.gate_y);
  ctx.stroke();
  ctx.setLineDash([]);
  const t = d.truth[f - 1];
  const tracks = d.panes[pane].tracks[f - 1] ?? [];
  const pillars = d.pillars ?? [];
  // painter's order: farther (smaller foot y) first; pillars are drawn at their base row
  type Item = { y: number; draw: () => void };
  const items: Item[] = [];
  for (const [px, py, pw, ph] of pillars)
    items.push({ y: py + ph, draw: () => { ctx.fillStyle = "#262B33"; ctx.fillRect(px, py, pw, ph); ctx.strokeStyle = "#3A414C"; ctx.lineWidth = 1; ctx.strokeRect(px + 0.5, py + 0.5, pw - 1, ph - 1); } });
  for (const [pid, x, y, vis] of t?.p ?? []) {
    const hidden = t.hidden.includes(pid);
    items.push({ y, draw: () => drawFigure(ctx, x, y, hidden, vis, f * 0.25 + pid) });
  }
  items.sort((a, b) => a.y - b.y).forEach((it) => it.draw());
  const recent = new Set(d.panes[pane].events.filter((e) => e.frame <= f && e.frame > f - 1.5 * d.fps).map((e) => e.to));
  for (const [tid, pid, x, y] of tracks) {
    const h = personHeight(y);
    chip(ctx, x, y - h - 4, `#${tid}`, recent.has(tid) ? AMBER : PANE_COLOUR[pane]);
    if (focus != null && pid === focus) {
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(x, y - h / 2, h * 0.34, h * 0.62, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  if (ring != null && t) {
    const p = t.p.find((q) => q[0] === ring);
    if (p && !tracks.some((q) => q[1] === ring)) {
      const h = personHeight(p[2]);
      ctx.strokeStyle = "#FFFFFF";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(p[1], p[2] - h / 2, h * 0.34, h * 0.62, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

export interface ScenePlayerProps {
  data: SceneData;
  startFrame?: number;
  /** person to follow (white ring) and whose hide gets a verdict chip */
  focus?: number | null;
  showVerdict?: boolean;
  onPersonClick?: (pid: number) => void;
  onFrame?: (f: number) => void;
}

export function ScenePlayer({ data, startFrame = 1, focus = null, showVerdict = false, onPersonClick, onFrame }: ScenePlayerProps) {
  const n = data.truth.length;
  const [f, setF] = useState(Math.min(Math.max(1, startFrame), n));
  const [playing, setPlaying] = useState(false);
  const refs = { bytetrack: useRef<HTMLCanvasElement>(null), ours: useRef<HTMLCanvasElement>(null) };
  useEffect(() => { setF(Math.min(Math.max(1, startFrame), n)); setPlaying(false); }, [data, startFrame, n]);
  useEffect(() => { onFrame?.(f); }, [f, onFrame]);
  useEffect(() => {
    for (const k of PANE_ORDER) if (refs[k].current) drawPane(refs[k].current!, data, k, f, focus, focus);
  });
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setF((x) => { if (x >= n) { setPlaying(false); return x; } return x + 1; }), 1000 / data.fps);
    return () => window.clearInterval(id);
  }, [playing, n, data.fps]);
  const verdicts = useMemo(() => {
    const out: Partial<Record<PaneKey, Verdict | null>> = {};
    if (focus != null) for (const k of PANE_ORDER) out[k] = verdict(data, k, focus);
    return out;
  }, [data, focus]);
  const counts = countsAt(data, f);
  function click(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onPersonClick) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * data.width;
    const y = ((e.clientY - r.top) / r.height) * data.height;
    let best: [number, number] | null = null;
    for (const [pid, px, py] of data.truth[f - 1]?.p ?? []) {
      const h = personHeight(py);
      if (x > px - h * 0.35 && x < px + h * 0.35 && y > py - h && y < py + 4) {
        const dd = Math.abs(px - x) + Math.abs(py - h / 2 - y);
        if (!best || dd < best[1]) best = [pid, dd];
      }
    }
    if (best) onPersonClick(best[0]);
  }
  return (
    <div className="scene-player" data-testid="scene-player" data-frame={f}>
      <div className="scene-panes">
        {PANE_ORDER.map((k) => {
          const v = verdicts[k];
          const showChip = showVerdict && v && f >= v.frame;
          const last = latestChange(data, k, f);
          return (
            <div key={k} className="scene-pane" data-testid={`pane-${k}`}>
              <div className="scene-pane-head">
                <span style={{ color: PANE_COLOUR[k] }}>{data.panes[k].label}</span>
                <span className="mono muted">count {counts[k]} · true {counts.truth}</span>
              </div>
              <div className="scene-canvas-wrap">
                <canvas ref={refs[k]} width={data.width} height={data.height} onClick={click} style={{ cursor: onPersonClick ? "pointer" : "default" }} />
                {showChip && (
                  <span className={`verdict-chip ${v.kind === "kept" ? "kept" : "broke"}`} data-testid={`verdict-${k}`}>{v.text}</span>
                )}
              </div>
              <p className="why-line" data-testid={`why-${k}`}>
                {last ? <><strong>{whyQuestion(last)}</strong> {last.line}</> : <span className="muted">No ID change in the last 4 s.</span>}
              </p>
            </div>
          );
        })}
      </div>
      <div className="scene-controls">
        <button className="btn" onClick={() => setPlaying((p) => !p)} data-testid="scene-play">{playing ? "Pause" : "Play"}</button>
        <button className="btn" onClick={() => setF((x) => Math.max(1, x - 1))} aria-label="Previous frame">−1</button>
        <button className="btn" onClick={() => setF((x) => Math.min(n, x + 1))} aria-label="Next frame">+1</button>
        <input type="range" min={1} max={n} value={f} onChange={(e) => setF(Number(e.target.value))} aria-label="Frame" style={{ flex: 1 }} />
        <span className="mono muted" data-testid="scene-frame">frame {f} / {n} · {(f / data.fps).toFixed(1)} s</span>
      </div>
    </div>
  );
}
