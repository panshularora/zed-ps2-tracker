// Side-by-side replay of one simulated scene: ByteTrack left, ours right, the same people in both.
// Each pane shows a crop of the scene (zoomed on the person that matters); under each pane, the
// verdict for that person and how many gate crossings the tracker counted.
import { useEffect, useMemo, useRef, useState } from "react";
import { AMBER, BLUE } from "./model";
import {
  PANE_ORDER, cropAt, cropFollow, cropWalkingArea, latestChange, lostLine, wholeSceneText, personHeight, verdict, whyQuestion,
  type Crop, type PaneKey, type SceneData, type Verdict,
} from "./scene";

const GREY = "#8A929C";
const FIG = "#C9CED6";
const HIDDEN_FILL = "#4D4D4D"; // 30 % grey
const PANE_COLOUR: Record<PaneKey, string> = { bytetrack: GREY, ours: BLUE };

function drawFigure(ctx: CanvasRenderingContext2D, x: number, y: number, hidden: boolean, vis: number, walk: number, px: number) {
  const h = personHeight(y);
  const headR = h * 0.11;
  const legH = h * 0.42;
  const torsoH = h * 0.34;
  const shW = h * 0.13;
  const sw = Math.sin(walk) * h * 0.07;
  ctx.save();
  ctx.globalAlpha = hidden ? 1 : 0.45 + 0.55 * Math.min(1, vis);
  const body = hidden ? HIDDEN_FILL : FIG;
  ctx.strokeStyle = body;
  ctx.fillStyle = body;
  ctx.lineCap = "round";
  ctx.lineWidth = Math.max(2 * px, h * 0.07);
  ctx.beginPath();
  ctx.moveTo(x - shW * 0.4, y - legH);
  ctx.lineTo(x - shW * 0.5 - sw, y);
  ctx.moveTo(x + shW * 0.4, y - legH);
  ctx.lineTo(x + shW * 0.5 + sw, y);
  ctx.stroke();
  const torso = new Path2D();
  torso.roundRect(x - shW, y - legH - torsoH, shW * 2, torsoH, 3);
  const head = new Path2D();
  head.arc(x, y - legH - torsoH - headR - 1, headR, 0, Math.PI * 2);
  ctx.fill(torso);
  ctx.fill(head);
  if (hidden) {
    // dashed outline around the whole figure: hidden from the camera, still walking
    ctx.strokeStyle = FIG;
    ctx.lineWidth = 1.5 * px;
    ctx.setLineDash([4 * px, 3 * px]);
    ctx.stroke(torso);
    ctx.stroke(head);
    ctx.beginPath();
    ctx.rect(x - shW * 1.25, y - legH, shW * 2.5, legH);
    ctx.stroke();
  }
  ctx.restore();
}

/** ID chip in canvas pixels: 16 CSS px tall, 12 px semibold text. */
function chip(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, colour: string, r: number) {
  ctx.font = `600 ${12 * r}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 10 * r;
  const hh = 16 * r;
  ctx.fillStyle = colour;
  ctx.fillRect(x - w / 2, y - hh, w, hh);
  ctx.fillStyle = "#0E1013";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y - hh / 2 + 0.5 * r);
}

function drawPane(c: HTMLCanvasElement, d: SceneData, pane: PaneKey, f: number, focus: number | null, crop: Crop) {
  const ctx = c.getContext("2d");
  if (!ctx) return;
  const k = c.width / crop.w; // canvas px per scene px
  const r = c.width / Math.max(1, c.clientWidth || c.width); // canvas px per CSS px
  const sx = (x: number) => (x - crop.x) * k;
  const sy = (y: number) => (y - crop.y) * k;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0E1013"; // padding past the scene edge
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#161A1F";
  ctx.fillRect(sx(0), sy(0), d.width * k, d.height * k);
  ctx.strokeStyle = AMBER;
  ctx.lineWidth = 2 * r;
  ctx.setLineDash([10 * r, 6 * r]);
  ctx.beginPath();
  ctx.moveTo(Math.max(0, sx(0)), sy(d.gate_y));
  ctx.lineTo(Math.min(c.width, sx(d.width)), sy(d.gate_y));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.setTransform(k, 0, 0, k, -crop.x * k, -crop.y * k);
  const t = d.truth[f - 1];
  const tracks = d.panes[pane].tracks[f - 1] ?? [];
  // painter's order: farther (smaller foot y) first; pillars are drawn at their base row
  type Item = { y: number; draw: () => void };
  const items: Item[] = [];
  for (const [px, py, pw, ph] of d.pillars ?? [])
    items.push({ y: py + ph, draw: () => { ctx.fillStyle = "#262B33"; ctx.fillRect(px, py, pw, ph); ctx.strokeStyle = "#3A414C"; ctx.lineWidth = r / k; ctx.strokeRect(px, py, pw, ph); } });
  for (const [pid, x, y, vis] of t?.p ?? []) {
    const hidden = t.hidden.includes(pid);
    items.push({ y, draw: () => drawFigure(ctx, x, y, hidden, vis, f * 0.25 + pid, r / k) });
  }
  items.sort((a, b) => a.y - b.y).forEach((it) => it.draw());
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const ring = (x: number, y: number, dashed: boolean) => {
    const h = personHeight(y);
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2 * r;
    ctx.setLineDash(dashed ? [4 * r, 4 * r] : []);
    ctx.beginPath();
    ctx.ellipse(sx(x), sy(y - h / 2), h * 0.34 * k, h * 0.62 * k, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  const recent = new Set(d.panes[pane].events.filter((e) => e.frame <= f && e.frame > f - 1.5 * d.fps).map((e) => e.to));
  for (const [tid, pid, x, y] of tracks) {
    if (focus != null && pid === focus) ring(x, y, false);
    chip(ctx, sx(x), sy(y - personHeight(y)) - 4 * r, `#${tid}`, recent.has(tid) ? AMBER : PANE_COLOUR[pane], r);
  }
  if (focus != null && t) {
    const p = t.p.find((q) => q[0] === focus);
    if (p && !tracks.some((q) => q[1] === focus)) ring(p[1], p[2], true);
  }
}

export interface ScenePlayerProps {
  data: SceneData;
  startFrame?: number;
  /** person to follow (white ring) and whose hide gets the verdict under each pane */
  focus?: number | null;
  /** "pause": 1.5× on the focus person where the replay opens (cell replay); "follow": 2× and
   * track them; "walking": box around all walking paths + 10 % (Break-it) */
  view?: "pause" | "follow" | "walking" | "full";
  /** "reappearance": the verdict shows once the person is back (Break-it, live);
   * "always": a replay of a finished run, the outcome is shown from the start */
  reveal?: "reappearance" | "always";
  onPersonClick?: (pid: number) => void;
  onFrame?: (f: number) => void;
}

export function ScenePlayer({ data, startFrame = 1, focus = null, view = "full", reveal = "reappearance", onPersonClick, onFrame }: ScenePlayerProps) {
  const n = data.truth.length;
  const [f, setF] = useState(Math.min(Math.max(1, startFrame), n));
  const [playing, setPlaying] = useState(false);
  const refs = { bytetrack: useRef<HTMLCanvasElement>(null), ours: useRef<HTMLCanvasElement>(null) };
  useEffect(() => { setF(Math.min(Math.max(1, startFrame), n)); setPlaying(false); }, [data, startFrame, n]);
  useEffect(() => { onFrame?.(f); }, [f, onFrame]);
  const fixed = useMemo(() => (view === "walking" ? cropWalkingArea(data) : view === "pause" ? cropAt(data, focus, startFrame, 1.5) : null), [view, data, focus, startFrame]);
  const crop: Crop = fixed ?? (view === "follow" ? cropFollow(data, focus, f, 2) : { x: 0, y: 0, w: data.width, h: data.height });
  useEffect(() => {
    for (const k of PANE_ORDER) if (refs[k].current) drawPane(refs[k].current!, data, k, f, focus, crop);
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
  function click(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onPersonClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = crop.x + ((e.clientX - rect.left) / rect.width) * crop.w;
    const y = crop.y + ((e.clientY - rect.top) / rect.height) * crop.h;
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
    <div className="scene-player" data-testid="scene-player" data-frame={f} data-crop={`${Math.round(crop.x)},${Math.round(crop.y)},${Math.round(crop.w)},${Math.round(crop.h)}`}>
      <div className="scene-panes">
        {PANE_ORDER.map((k) => {
          const v = verdicts[k];
          const shown = v && (reveal === "always" || f >= v.frame);
          const last = latestChange(data, k, f);
          const p = data.panes[k];
          return (
            <div key={k} className="scene-pane" data-testid={`pane-${k}`}>
              <div className="scene-pane-head">
                <span style={{ color: PANE_COLOUR[k] }}>{p.label}</span>
              </div>
              <div className="scene-canvas-wrap">
                <canvas ref={refs[k]} width={960} height={Math.round((960 * data.height) / data.width)} onClick={click} style={{ cursor: onPersonClick ? "pointer" : "default" }} />
              </div>
              <div className="pane-foot">
                {v ? (
                  <span className={`verdict-big ${v.kind === "kept" ? "kept" : "broke"}${shown ? "" : " pending"}`} data-testid={`verdict-${k}`} data-shown={shown ? "1" : "0"}>
                    {shown ? v.text : `#${v.kind === "kept" ? v.id : v.from} hidden…`}
                  </span>
                ) : <span className="verdict-big pending">{focus == null ? "Nobody hidden" : "No verdict"}</span>}
                <span className="counted" data-testid={`counted-${k}`} aria-label={wholeSceneText(p)} title="Gate crossings this tracker counted over the whole scene, out of the real number">
                  Whole scene: counted <b>{p.count}</b> of <b>{p.true_count}</b>
                </span>
              </div>
              {(() => { const l = lostLine(data, k, focus, v?.frame ?? startFrame); return l ? <p className="lost-line" data-testid={`lost-${k}`}>{l}</p> : null; })()}
              <p className="why-line" data-testid={`why-${k}`}>
                {last ? <><strong>{whyQuestion(last)}</strong> {last.line}</> : <span className="muted">No ID change in the last 4 s.</span>}
              </p>
            </div>
          );
        })}
      </div>
      <div className="scene-controls">
        <button className="btn" onClick={() => setPlaying((x) => !x)} data-testid="scene-play">{playing ? "Pause" : "Play"}</button>
        <button className="btn" onClick={() => setF((x) => Math.max(1, x - 1))} aria-label="Previous frame">−1</button>
        <button className="btn" onClick={() => setF((x) => Math.min(n, x + 1))} aria-label="Next frame">+1</button>
        <input type="range" min={1} max={n} value={f} onChange={(e) => setF(Number(e.target.value))} aria-label="Frame" style={{ flex: 1 }} />
        <span className="scene-time tabular" data-testid="scene-frame">{(f / data.fps).toFixed(1)} s of {(n / data.fps).toFixed(1)} s</span>
      </div>
    </div>
  );
}
