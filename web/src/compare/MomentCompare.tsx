// "Same person, two trackers": one person walks behind another in the synthetic gate scene (exact
// ground truth), ByteTrack on the left, ours on the right, auto-pause when they come out. Every
// word under and above the panes comes from compare/narrative.ts, computed from the stream.
import { useEffect, useMemo, useRef, useState } from "react";
import { PANES, readStream, recordingUrl, type PaneKey, type SimFrame, type SimMeta } from "../lib/gateSim";
import { neutralColor } from "../lib/scenarios";
import { drawChip, drawFigure, figureGeom, walkStep } from "../lib/figure";
import { BASE_NAME, CHANGED, GREY, KEPT, OURS_NAME } from "../lib/ours";
import { pickStory, tally, type MomentStory } from "./narrative";

const LEAD_S = 2.2, TAIL_F = 10;
type Crop = [number, number, number, number];

/** Walking area around the moment: both people over the window, padded, 16:10, inside the scene. */
export function cropFor(frames: SimFrame[], s: MomentStory, meta: Pick<SimMeta, "width" | "height" | "fps">): Crop {
  const from = Math.max(0, s.m.lastIdx - Math.round(LEAD_S * meta.fps)), to = Math.min(frames.length - 1, s.m.backIdx + TAIL_F);
  const who = new Set([s.m.person, ...(s.m.occluder != null ? [s.m.occluder] : [])]);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = from; i <= to; i++) for (const [pid, x, y] of frames[i].people) if (who.has(pid)) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (!isFinite(x0)) return [0, 0, meta.width, meta.height];
  const pad = 70;
  let w = x1 - x0 + 2 * pad, h = y1 - y0 + 2 * pad + 40;
  if (w / h < 1.6) w = h * 1.6; else h = w / 1.6;
  w = Math.min(w, meta.width); h = Math.min(h, meta.height);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2 - 15;
  return [Math.min(Math.max(0, cx - w / 2), meta.width - w), Math.min(Math.max(0, cy - h / 2), meta.height - h), w, h];
}

export function paintMoment(ctx: CanvasRenderingContext2D, cw: number, ch: number, meta: SimMeta, fr: SimFrame, pane: PaneKey, crop: Crop, focus: number, t: number) {
  const [vx, vy, vw, vh] = crop;
  const s = cw / vw;
  ctx.fillStyle = "#13171C"; ctx.fillRect(0, 0, cw, ch);
  ctx.strokeStyle = "rgba(230,232,235,0.05)"; ctx.lineWidth = 1;
  for (let gx = Math.ceil(vx / 40) * 40; gx < vx + vw; gx += 40) { ctx.beginPath(); ctx.moveTo((gx - vx) * s, 0); ctx.lineTo((gx - vx) * s, ch); ctx.stroke(); }
  for (let gy = Math.ceil(vy / 40) * 40; gy < vy + vh; gy += 40) { ctx.beginPath(); ctx.moveTo(0, (gy - vy) * s); ctx.lineTo(cw, (gy - vy) * s); ctx.stroke(); }
  const gy = meta.gate[0][1];
  if (gy > vy && gy < vy + vh) { ctx.strokeStyle = CHANGED; ctx.lineWidth = 3; ctx.setLineDash([10, 6]); ctx.beginPath(); ctx.moveTo(0, (gy - vy) * s); ctx.lineTo(cw, (gy - vy) * s); ctx.stroke(); ctx.setLineDash([]); }
  const ids = new Map(fr[pane].ids);
  const h = 58 * s;
  const people = [...fr.people].sort((a, b) => a[2] - b[2]);
  for (const [pid, x, y, vis] of people) {
    const px = (x - vx) * s, foot = (y - vy) * s + h * 0.45;
    if (px < -h || px > cw + h || foot < 0 || foot - h > ch) continue;
    const tid = ids.get(pid);
    const color = neutralColor(pid);
    ctx.save();
    if (pid !== focus && focus >= 0) ctx.globalAlpha = 0.85;
    drawFigure(ctx, { cx: px, footY: foot, h, color, step: walkStep(t + pid * 0.37), hidden: !vis });
    ctx.restore();
    const top = figureGeom({ cx: px, footY: foot, h }).top;
    if (!vis) drawChip(ctx, px, top - 4, "hidden", GREY);
    else if (tid != null) drawChip(ctx, px, top - 4, `#${tid}`, neutralColor(tid), pid === focus ? 15 : 13);
  }
}

export default function MomentCompare({ onNext }: { onNext?: () => void }) {
  const [meta, setMeta] = useState<SimMeta | null>(null);
  const [frames, setFrames] = useState<SimFrame[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const ac = new AbortController(), buf: SimFrame[] = [];
    fetch(recordingUrl("crowd"), { signal: ac.signal })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return readStream(r, (l) => { if (l.type === "meta") setMeta(l); else if (l.type === "frame") buf.push(l); }, ac.signal); })
      .then(() => setFrames(buf))
      .catch((e) => { if (!ac.signal.aborted) setErr(String(e)); });
    return () => ac.abort();
  }, []);
  const pick = useMemo(() => (frames.length ? pickStory(frames) : null), [frames]);
  const story = pick?.story ?? null;
  const tl = useMemo(() => (pick ? tally(pick.all) : null), [pick]);
  const crop = useMemo(() => (story && meta ? cropFor(frames, story, meta) : null), [frames, story, meta]);
  const cv = { bytetrack: useRef<HTMLCanvasElement>(null), ours: useRef<HTMLCanvasElement>(null) };
  const [done, setDone] = useState(false);
  const [run, setRun] = useState(0);
  const head = useRef(0);

  useEffect(() => {
    if (!story || !meta || !crop) return;
    const start = Math.max(0, story.m.lastIdx - Math.round(LEAD_S * meta.fps));
    const stop = Math.min(frames.length - 1, story.m.backIdx + TAIL_F);
    head.current = start; setDone(false);
    let raf = 0, last = performance.now(), t = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      if (head.current < stop) { head.current = Math.min(stop, head.current + dt * meta.fps * 0.8); t += dt; if (head.current >= stop) setDone(true); }
      const fr = frames[Math.floor(head.current)];
      for (const p of PANES) {
        const c = cv[p].current; const ctx = c?.getContext?.("2d");
        if (!c || !ctx || !c.clientWidth) continue;
        const dpr = window.devicePixelRatio || 1;
        if (c.width !== Math.round(c.clientWidth * dpr)) { c.width = Math.round(c.clientWidth * dpr); c.height = Math.round(c.clientHeight * dpr); }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        paintMoment(ctx, c.clientWidth, c.clientHeight, meta, fr, p, crop, story.m.person, head.current >= stop ? 0 : t);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [story, meta, crop, run]);

  if (err) return <p className="muted" role="alert">Could not load the recorded gate scene ({err}).</p>;
  if (!story || !meta || !crop) return <p className="muted">{pick && !story ? "No hide-and-return moment in this run." : "Loading the scene…"}</p>;
  const label: Record<PaneKey, string> = { bytetrack: `Left · ${BASE_NAME}`, ours: `Right · ${OURS_NAME}` };
  return (
    <div className="moment-compare" data-testid="moment-compare">
      <h1 className="page-h">Same person, two trackers</h1>
      <p className="page-lead" data-testid="mc-headline">{story.headline}</p>
      <div className="mc-panes">
        {PANES.map((p) => {
          const o = story.out[p];
          return (
            <figure key={p} className="mc-pane" data-testid={`mc-pane-${p}`}>
              <figcaption className={`mc-label ${p === "ours" ? "ours" : "base"}`}>{label[p]}</figcaption>
              <div className="mc-stage" style={{ aspectRatio: `${crop[2]} / ${crop[3]}` }}><canvas ref={cv[p]} aria-label={`${label[p]}: person ${story.m.person}`} /></div>
              <div className="mc-verdict" data-testid={`mc-verdict-${p}`} style={{ color: done ? (o.verdict === "kept" ? KEPT : o.verdict === "changed" ? CHANGED : GREY) : GREY }}>
                {done ? story.verdicts[p] : `Watching person ${story.m.person}…`}
              </div>
            </figure>
          );
        })}
      </div>
      <div className="mc-foot">
        <button type="button" className="btn" onClick={() => setRun((r) => r + 1)} data-testid="mc-replay">⟲ Replay</button>
        <span className="muted" data-testid="mc-tally">
          Synthetic scene · exact ground truth · seed {meta.params.seed}. All {pick!.all.length} hides in this run: ByteTrack kept {tl!.bytetrack.kept}, ours kept {tl!.ours.kept}.
          {pick!.kind !== "ours-better" && " No moment in this run where only ours keeps the number."}
        </span>
        {onNext && <button type="button" className="btn primary next-btn" onClick={onNext}>Next →</button>}
      </div>
    </div>
  );
}
