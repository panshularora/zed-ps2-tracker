// Synced side-by-side live replay: baseline (left) vs ours (right) on one clock, continuous
// autoplay with smooth box interpolation, decluttered overlays and a live plain-words caption.
import { useEffect, useMemo, useRef, useState } from "react";
import type { Box, RunKey } from "../lib/scenarios";
import { neutralColor } from "../lib/scenarios";
import { countsAt, frameAtTime, timeOfFrame, videoSources, fallbackText, type GEvent, type ReplayData } from "../guided/model";
import { boxesAt, focusIds, hiddenNow, layoutChips, liveCaption, liveRate, momentAt, runStats } from "./model";

const GATE = "#F5A524";
const RUNS: RunKey[] = ["baseline", "ours"];

interface PaneState { hover: number | null }

function drawPane(ctx: CanvasRenderingContext2D, cw: number, ch: number, d: ReplayData, run: RunKey, t: number, src: CanvasImageSource | null, st: PaneState) {
  const sx = cw / d.width, sy = ch / d.height;
  ctx.fillStyle = "#13171C";
  ctx.fillRect(0, 0, cw, ch);
  if (src) {
    ctx.drawImage(src, 0, 0, cw, ch);
    ctx.fillStyle = "rgba(14,16,19,0.35)"; // calm the footage so boxes read first
    ctx.fillRect(0, 0, cw, ch);
  }
  const frames = d.runs[run].frames;
  const { f, boxes } = boxesAt(frames, d.fps, d.frames, t);
  const focus = focusIds(d, run, t, f);
  const changing = new Map<number, { from: number; dt: number }>();
  for (const e of d.events[run]) if (e.t === "renumber" && f != null) {
    const dt = (t - timeOfFrame(d, e.f));
    if (dt >= 0 && dt < 2) { changing.set(e.to, { from: e.from, dt }); focus.add(e.to); }
  }
  const mixup = momentAt(d, t) != null;
  if (d.gate) {
    ctx.strokeStyle = GATE; ctx.lineWidth = 2; ctx.setLineDash([10, 6]);
    ctx.beginPath(); ctx.moveTo(d.gate.a[0] * sx, d.gate.a[1] * sy); ctx.lineTo(d.gate.b[0] * sx, d.gate.b[1] * sy); ctx.stroke();
    ctx.setLineDash([]);
  }
  // others first, dimmed, no chip (hover shows it)
  const chips: { id: number; text: string; cx: number; top: number; w: number }[] = [];
  ctx.font = "600 12px Inter, sans-serif";
  const chipW = (s: string) => Math.max(22, ctx.measureText(s).width + 12);
  for (const [id, x, y, w, h] of boxes) {
    const on = focus.has(id) || st.hover === id;
    if (focus.has(id)) continue;
    ctx.globalAlpha = on ? 1 : mixup ? 0.25 : 0.35;
    ctx.strokeStyle = neutralColor(id); ctx.lineWidth = on ? 2 : 1;
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    ctx.globalAlpha = 1;
    if (on) chips.push({ id, text: `#${id}`, cx: (x + w / 2) * sx, top: y * sy, w: chipW(`#${id}`) });
  }
  // focus people: 1 s trail (breaks at an id change: per id), 2 px box, chip
  for (const [id, x, y, w, h] of boxes) {
    if (!focus.has(id)) continue;
    const c = neutralColor(id);
    ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.lineCap = "round";
    const n = Math.round(d.fps);
    let prev: [number, number] | null = null;
    for (let k = Math.max(1, (f ?? 1) - n); k <= (f ?? 0); k++) {
      const b = frames[String(k)]?.find((r) => r[0] === id);
      if (!b) { prev = null; continue; }
      const pt: [number, number] = [(b[1] + b[3] / 2) * sx, (b[2] + b[4]) * sy];
      if (prev) { ctx.globalAlpha = 1 - ((f ?? 0) - k) / n; ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(pt[0], pt[1]); ctx.stroke(); }
      prev = pt;
    }
    ctx.globalAlpha = 1;
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
    const chg = changing.get(id);
    if (chg && chg.dt < 0.6) {
      // 600 ms white ring pulse at the spot of the change
      const k = chg.dt / 0.6;
      ctx.strokeStyle = `rgba(255,255,255,${1 - k})`; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.ellipse((x + w / 2) * sx, (y + h / 2) * sy, (w * sx) / 2 + 6 + 14 * k, (h * sy) / 2 + 6 + 14 * k, 0, 0, Math.PI * 2); ctx.stroke();
    }
    const text = chg ? `#${chg.from} → #${id}` : `#${id}`;
    chips.push({ id, text, cx: (x + w / 2) * sx, top: y * sy, w: chipW(text) });
  }
  // hidden focus people: dashed last-seen box + "hidden" chip
  const hidden = hiddenNow(frames, d.events[run], focus, f, d.fps);
  for (const { id, box } of hidden) {
    const [, x, y, w, h] = box as Box;
    ctx.strokeStyle = neutralColor(id); ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.strokeRect(x * sx, y * sy, w * sx, h * sy); ctx.setLineDash([]);
    const text = `#${id} hidden`;
    chips.push({ id, text, cx: (x + w / 2) * sx, top: y * sy, w: chipW(text) });
  }
  const hiddenIds = new Set(hidden.map((h) => h.id));
  for (const c of layoutChips(chips, 20, cw)) {
    const col = neutralColor(c.id);
    if (Math.abs(c.y + 20 - c.anchorY) > 8) { ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(c.anchorX, c.anchorY); ctx.lineTo(c.x + c.w / 2, c.y + 20); ctx.stroke(); }
    ctx.globalAlpha = hiddenIds.has(c.id) ? 0.7 : 1;
    ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(c.x, c.y, c.w, 20, 4); ctx.fill();
    ctx.fillStyle = "#0E1013"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(c.text, c.x + c.w / 2, c.y + 11);
    ctx.globalAlpha = 1;
  }
  return { f, n: boxes.length, hidden: hidden.map((h) => h.id) };
}

export interface Truth { events: GEvent[]; label: string }

export function ComparePlayer({ data, truth }: { data: ReplayData; truth?: Truth | null }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvases = { baseline: useRef<HTMLCanvasElement>(null), ours: useRef<HTMLCanvasElement>(null) };
  const hover = useRef<Record<RunKey, number | null>>({ baseline: null, ours: null });
  const clock = useRef(0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [info, setInfo] = useState<{ n: number; hidden: Record<RunKey, number[]> }>({ n: 0, hidden: { baseline: [], ours: [] } });
  const sources = useMemo(() => videoSources(data), [data]);
  const [srcIdx, setSrcIdx] = useState(0);
  const src = sources[srcIdx] ?? null;
  const hint = fallbackText(data, sources.length > 0 && src == null);
  const off = data.videoOffsetS ?? 0;

  useEffect(() => {
    let raf = 0, last = performance.now(), lastUi = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      const v = video.current;
      const useVideo = !!(v && src && v.readyState >= 2);
      let cur: number;
      if (useVideo) {
        if (v!.currentTime - off > data.duration || v!.currentTime < off - 0.05) v!.currentTime = off;
        cur = Math.max(0, v!.currentTime - off);
        v!.playbackRate = liveRate(data, cur);
      } else {
        if (playing) clock.current += dt * liveRate(data, clock.current);
        if (clock.current > data.duration) clock.current = 0;
        cur = clock.current;
      }
      clock.current = cur;
      const hid: Record<RunKey, number[]> = { baseline: [], ours: [] };
      let n = 0;
      for (const r of RUNS) {
        const c = canvases[r].current;
        const ctx = c?.getContext("2d");
        if (!c || !ctx) continue;
        const dpr = window.devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
        if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const res = drawPane(ctx, cw, ch, data, r, cur, useVideo ? v! : null, { hover: hover.current[r] });
        hid[r] = res.hidden; n = Math.max(n, res.n);
      }
      if (now - lastUi > 100) { lastUi = now; setT(cur); setInfo({ n, hidden: hid }); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, src, off, playing]);

  useEffect(() => {
    const v = video.current;
    if (!v) return;
    if (playing) v.play()?.catch(() => undefined); else v.pause();
  }, [playing, src]);

  function seek(x: number) {
    const s = Math.max(0, Math.min(data.duration, x));
    clock.current = s;
    if (video.current) video.current.currentTime = s + off;
    setT(s);
  }

  function onMove(r: RunKey, e: React.MouseEvent<HTMLCanvasElement>) {
    const c = e.currentTarget, rect = c.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * data.width, y = ((e.clientY - rect.top) / rect.height) * data.height;
    const { boxes } = boxesAt(data.runs[r].frames, data.fps, data.frames, clock.current);
    const hit = boxes.filter((b) => x >= b[1] && x <= b[1] + b[3] && y >= b[2] && y <= b[2] + b[4]).sort((a, b) => a[3] * a[4] - b[3] * b[4])[0];
    hover.current = { baseline: null, ours: null, [r]: hit ? hit[0] : null };
  }

  const f = frameAtTime(data, t);
  const caption = liveCaption(data, t, info.n);
  const truthCount = truth && f != null ? countsAt(truth.events, f) : null;
  return (
    <div className="cmp" data-testid="compare-player">
      {src && (
        <video key={src} ref={video} src={src} muted playsInline autoPlay loop className="cmp-video" aria-hidden
          onError={() => setSrcIdx((i) => i + 1)} onLoadedMetadata={(e) => { if (!(e.currentTarget.duration > 0)) setSrcIdx((i) => i + 1); }} />
      )}
      <div className="cmp-panes">
        {RUNS.map((r) => {
          const s = runStats(data, r, f);
          return (
            <figure key={r} className={`cmp-pane cmp-${r}`} data-testid={`pane-${r}`}>
              <figcaption className="cmp-head">
                <b className={r === "ours" ? "ours" : "base"}>{data.runs[r].label}</b>
                <span className="muted">{r === "baseline" ? "baseline" : "our tracker"}</span>
              </figcaption>
              <div className="cmp-stage" style={{ aspectRatio: `${data.width} / ${data.height}`, width: `min(100%, calc((100vh - 300px) * ${(data.width / data.height).toFixed(4)}))` }}>
                <canvas ref={canvases[r]} onMouseMove={(e) => onMove(r, e)} onMouseLeave={() => { hover.current = { baseline: null, ours: null }; }}
                  aria-label={`${data.title}, ${data.runs[r].label}`} />
                {hint && <div className="cmp-hint" role="status">{hint}</div>}
                {info.hidden[r].length > 0 && <div className="cmp-hidden" data-testid="hidden-list">Hidden now: {info.hidden[r].map((i) => `#${i}`).join(", ")}</div>}
              </div>
              <div className="cmp-counters tabular" data-testid={`counters-${r}`}>
                {truthCount && <span>True count <b>{truthCount.in + truthCount.out}</b></span>}
                <span>Counted <b className={truthCount && s.crossIn + s.crossOut > truthCount.in + truthCount.out ? "over" : ""}>{s.crossIn + s.crossOut}</b> <small className="muted">({s.crossIn} in · {s.crossOut} out)</small></span>
                <span>ID changes <b>{s.idChanges}</b></span>
              </div>
            </figure>
          );
        })}
      </div>
      <div className="cmp-caption" aria-live="polite" data-testid="live-caption">{caption}</div>
      <div className="cmp-controls">
        <button type="button" className="btn" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
        <div className="cmp-seek">
          <input type="range" min={0} max={data.duration} step={0.01} value={Math.min(t, data.duration)} onChange={(e) => seek(Number(e.target.value))} aria-label="Seek position" />
          {data.moments.map((m, i) => (
            <button key={i} type="button" className="cmp-tick" style={{ left: `${(timeOfFrame(data, m.f) / data.duration) * 100}%` }}
              title={`Mix-up ${i + 1}`} aria-label={`Jump to mix-up ${i + 1}`} onClick={() => seek(timeOfFrame(data, m.f) - 2.5)} data-testid="cmp-tick" />
          ))}
        </div>
        <span className="mono muted tabular">{t.toFixed(1)} / {data.duration.toFixed(1)} s · {liveRate(data, t) < 1 ? "0.5x near a mix-up" : "1x"}</span>
      </div>
    </div>
  );
}
