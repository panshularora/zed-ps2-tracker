import { useEffect, useMemo, useRef, useState } from "react";
import { neutralColor } from "../lib/scenarios";
import { fallbackText, videoSources, type ReplayData } from "../guided/model";
import { PersonTimeline } from "./PersonTimeline";
import { boxesAt, countsAt, people as peopleOf, pickFocus, placeChips, storySteps, tracksOf, watchLine, type Drawn, type XYWH } from "./story";

export const STEP_PAUSE_MS = 1800;
const ZOOM = 1.6, ZOOM_MS = 500, TRAIL_S = 1.5, DIM = 0.35;
const GATE = "#F5A524";
const ease = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/** Counters live in a bar outside the frame: a single row "IN n · OUT n". */
export function CountBar({ inN, outN, note }: { inN: number; outN: number; note?: string | null }) {
  return (
    <div className="count-bar" data-testid="count-bar">
      <span className="count-bar-nums tabular"><b>IN</b> {inN} <span className="muted">·</span> <b>OUT</b> {outN}</span>
      {note && <span className="count-bar-note muted" data-testid="video-hint">{note}</span>}
    </div>
  );
}

export function IdentityLegend() {
  return (
    <div className="id-legend" data-testid="identity-legend">
      <span><i className="lg lg-box" />Box + number = one person</span>
      <span><i className="lg lg-dash" />Dashed = still that number, hidden</span>
      <span><i className="lg lg-kept">kept</i>Came back with the same number</span>
      <span><i className="lg lg-line" />Gate line: crossing it adds to IN or OUT</span>
    </div>
  );
}

export function IdentityPlayer({ data, watch }: { data: ReplayData; watch?: string }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const tracks = useMemo(() => tracksOf(data.runs.ours.frames), [data]);
  const { people } = useMemo(() => peopleOf(data), [data]);
  const focus = useMemo(() => pickFocus(people), [people]);
  const steps = useMemo(() => (focus ? storySteps(data, focus) : []), [data, focus]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [story, setStory] = useState(true);
  const [follow, setFollow] = useState<number | null>(null);
  const [step, setStep] = useState<number | null>(null);
  const sources = useMemo(() => videoSources(data), [data]);
  const [srcIdx, setSrcIdx] = useState(0);
  const src = sources[srcIdx] ?? null;
  const hint = fallbackText(data, sources.length > 0 && src == null);
  const off = data.videoOffsetS ?? 0;
  const clock = useRef(0);
  const prev = useRef(0);
  const holdUntil = useRef(0);
  const zoomK = useRef({ k: 0, at: 0, target: 0 });
  const followRef = useRef(follow); followRef.current = follow;
  const focusIds = useMemo(() => new Set(focus?.ids ?? []), [focus]);

  useEffect(() => {
    zoomK.current = { k: zoomK.current.k, at: performance.now(), target: follow != null ? 1 : 0 };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setFollow(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [follow]);

  useEffect(() => {
    let raf = 0, last = performance.now(), lastUi = 0, from = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      const v = video.current;
      const holding = now < holdUntil.current;
      let cur: number;
      if (v && src) {
        if (playing && !holding && v.paused) v.play().catch(() => undefined);
        if ((!playing || holding) && !v.paused) v.pause();
        if (v.currentTime - off > data.duration) { v.currentTime = off; }
        cur = Math.max(0, v.currentTime - off);
      } else {
        if (playing && !holding) clock.current += dt;
        if (clock.current > data.duration) clock.current = 0;
        cur = clock.current;
      }
      // story: pause briefly on each key moment of the focus person
      if (story && playing && !holding && cur > prev.current && cur - prev.current < 0.5) {
        const i = steps.findIndex((s) => { const ts = (s.f - 1) / data.fps; return ts > prev.current && ts <= cur; });
        if (i >= 0) { holdUntil.current = now + STEP_PAUSE_MS; setStep(i); }
      }
      prev.current = cur;
      // zoom easing
      const z = zoomK.current;
      if (z.k !== z.target) { const p = Math.min(1, (now - z.at) / ZOOM_MS); z.k = z.target ? ease(p) : 1 - ease(p); if (p >= 1) z.k = z.target; }
      draw(cur, z.k, from);
      from = cur;
      if (now - lastUi > 100) { lastUi = now; setT(cur); }
      raf = requestAnimationFrame(tick);
    };
    const draw = (cur: number, zk: number, _p: number) => {
      void _p;
      const c = canvas.current, ctx = c?.getContext("2d");
      if (!c || !ctx) return;
      const dpr = window.devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
      if (!cw || !ch) return;
      if (c.width !== Math.round(cw * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const ff = Math.min(data.frames, Math.max(1, 1 + cur * data.fps));
      const all = boxesAt(tracks, ff, data.fps);
      const fid = followRef.current;
      const fb = fid != null ? all.find((d) => d.id === fid)?.box : null;
      const W = data.width, H = data.height;
      let view: XYWH = [0, 0, W, H];
      if (fb && zk > 0) {
        const z = 1 + (ZOOM - 1) * zk, w = W / z, h = H / z;
        view = [Math.min(Math.max(0, fb[0] + fb[2] / 2 - w / 2), W - w), Math.min(Math.max(0, fb[1] + fb[3] / 2 - h / 2), H - h), w, h];
      }
      paintFrame(ctx, cw, ch, data, v(), view, all, fid, fid == null ? focusIds : new Set([fid]), ff, tracks);
    };
    const v = () => (src && video.current && video.current.readyState >= 2 ? video.current : null);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [data, src, off, playing, story, steps, tracks, focusIds]);

  function seekFrame(f: number) {
    const s = Math.max(0, (f - 1) / data.fps - 0.3);
    clock.current = s; prev.current = s; holdUntil.current = 0;
    if (video.current) video.current.currentTime = s + off;
    setStep(null); setT(s);
  }

  function onClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * data.width, y = ((e.clientY - r.top) / r.height) * data.height;
    const ff = Math.max(1, 1 + t * data.fps);
    const hitBox = boxesAt(tracks, ff, data.fps).find((d) => x >= d.box[0] && x <= d.box[0] + d.box[2] && y >= d.box[1] && y <= d.box[1] + d.box[3]);
    setFollow((cur) => (hitBox && hitBox.id !== cur ? hitBox.id : null));
  }

  const f = Math.min(data.frames, Math.max(1, Math.floor(1 + t * data.fps)));
  const counts = countsAt(data.events.ours, f);
  const cur = step != null ? steps[step] : null;
  const caption = cur && t - (cur.f - 1) / data.fps < 3 ? cur.text
    : follow != null ? `Following #${follow}. Click again or press Esc to stop.`
    : story ? (steps.length ? "Story mode: the replay pauses briefly at each key moment for the highlighted person. Click any box to follow someone." : "Click any box to follow that person.")
    : "Free play. Click any box to follow that person.";
  return (
    <div className="idp" data-testid="identity-player">
      <p className="watch-line" data-testid="watch-line">{watch ?? watchLine(focus, steps)}</p>
      <CountBar inN={counts.in} outN={counts.out} note={hint} />
      <div className="idp-stage" data-testid="identity-stage" style={{ aspectRatio: `${data.width} / ${data.height}` }}>
        {src && <video key={src} ref={video} src={src} muted playsInline autoPlay className="idp-video" aria-label={data.title}
          onError={() => setSrcIdx((i) => i + 1)} onLoadedMetadata={(e) => { if (!(e.currentTarget.duration > 0)) setSrcIdx((i) => i + 1); }} />}
        <canvas ref={canvas} className="idp-canvas" onClick={onClick} aria-label={`${data.title}: ${data.runs.ours.label}. Click a person to follow.`} />
      </div>
      <div className="idp-caption" aria-live="polite" data-testid="story-caption">{caption}</div>
      <div className="idp-controls">
        <button type="button" className="btn" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
        <span className="mono muted tabular">{t.toFixed(1)} / {data.duration.toFixed(1)} s</span>
        <div className="toggle" role="group" aria-label="Mode">
          <button type="button" className={`btn${story ? " on-ours" : ""}`} aria-pressed={story} onClick={() => setStory(true)}>Story</button>
          <button type="button" className={`btn${!story ? " on-ours" : ""}`} aria-pressed={!story} onClick={() => { setStory(false); holdUntil.current = 0; }}>Free play</button>
        </div>
        <span className="muted idp-run">{data.runs.ours.label}</span>
      </div>
      <PersonTimeline people={people} frames={data.frames} f={f} focus={follow ?? focus?.key ?? null} onSeek={seekFrame} />
      <IdentityLegend />
    </div>
  );
}

function paintFrame(ctx: CanvasRenderingContext2D, cw: number, ch: number, d: ReplayData, src: CanvasImageSource | null, view: XYWH, all: Drawn[], follow: number | null, focus: Set<number>, ff: number, tracks: Map<number, Map<number, XYWH>>) {
  const [vx, vy, vw, vh] = view;
  const s = cw / vw;
  ctx.save();
  ctx.fillStyle = "#13171C"; // empty frame when no video
  ctx.fillRect(0, 0, cw, ch);
  ctx.setTransform(ctx.getTransform().scale(s, ch / vh).translate(-vx, -vy));
  if (src) ctx.drawImage(src, 0, 0, d.width, d.height);
  const px = 1 / s;
  if (d.gate) {
    ctx.strokeStyle = GATE; ctx.lineWidth = 3 * px; ctx.setLineDash([12 * px, 6 * px]);
    ctx.beginPath(); ctx.moveTo(...d.gate.a); ctx.lineTo(...d.gate.b); ctx.stroke(); ctx.setLineDash([]);
  }
  const hasFocus = focus.size > 0;
  const isF = (id: number) => !hasFocus || focus.has(id);
  // follow: 1.5 s fading trail in ID colour
  if (follow != null) {
    const m = tracks.get(follow);
    if (m) {
      const n = Math.round(TRAIL_S * d.fps), f = Math.floor(ff);
      ctx.strokeStyle = neutralColor(follow); ctx.lineWidth = 3 * px; ctx.lineCap = "round";
      for (let j = f - n + 1; j <= f; j++) {
        const a = m.get(j - 1), b = m.get(j);
        if (!a || !b) continue;
        ctx.globalAlpha = (j - (f - n)) / n;
        ctx.beginPath(); ctx.moveTo(a[0] + a[2] / 2, a[1] + a[3]); ctx.lineTo(b[0] + b[2] / 2, b[1] + b[3]); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }
  const items: { id: number; box: XYWH; text: string; w: number; h: number; d: Drawn }[] = [];
  for (const b of all) {
    const c = neutralColor(b.id);
    ctx.globalAlpha = isF(b.id) ? 1 : DIM;
    ctx.strokeStyle = c; ctx.lineWidth = 2 * px;
    if (b.state === "hidden") ctx.setLineDash([8 * px, 6 * px]);
    ctx.strokeRect(...b.box);
    ctx.setLineDash([]);
    if (b.id === follow) {
      ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2 * px;
      ctx.strokeRect(b.box[0] - 4 * px, b.box[1] - 4 * px, b.box[2] + 8 * px, b.box[3] + 8 * px);
    }
    const text = b.state === "hidden" ? `still #${b.id} (hidden)` : b.state === "kept" && isF(b.id) ? `kept #${b.id}` : `#${b.id}`;
    const size = b.id === follow ? 18 : 13;
    ctx.font = `600 ${size * px}px Inter, sans-serif`;
    items.push({ id: b.id, box: b.box, text, w: ctx.measureText(text).width + 10 * px, h: (size + 7) * px, d: b });
  }
  for (const c of placeChips(items.filter((i) => i.d.state !== "hidden" || isF(i.id)))) {
    const it = items.find((i) => i.id === c.id)!;
    const size = c.id === follow ? 18 : 13;
    ctx.globalAlpha = isF(c.id) ? (it.d.state === "hidden" ? 0.85 : 1) : DIM;
    ctx.fillStyle = neutralColor(c.id);
    const x = Math.min(Math.max(vx, c.x), vx + vw - c.w), y = Math.max(vy, c.y);
    ctx.beginPath(); ctx.roundRect(x, y, c.w, c.h, 4 * px); ctx.fill();
    ctx.fillStyle = "#0E1013"; ctx.font = `600 ${size * px}px Inter, sans-serif`;
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(c.text, x + 5 * px, y + c.h / 2 + 0.5 * px);
  }
  ctx.globalAlpha = 1;
  ctx.restore();
}
