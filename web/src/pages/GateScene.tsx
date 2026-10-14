// Live gate scene: synthetic people with exact ground truth, a fake detector, and the REAL Python
// trackers (ByteTrack left, BoT-SORT + visibility right) streamed from serve/gate_sim.py.
import { useCallback, useEffect, useRef, useState } from "react";
import { layoutChips } from "../compare/model";
import { neutralColor } from "../lib/scenarios";
import {
  DEFAULTS, PANES, RECORDED_SECONDS, RECORDING_URL, involvedBetween, isDefault, readStream, recentEvents, streamUrl,
  type PaneKey, type SimControls, type SimEnd, type SimFrame, type SimMeta,
} from "../lib/gateSim";

const GATE = "#F5A524";
const RING_S = 0.6, TAG_S = 2, DIM_S = 1.5, TRAIL_S = 1, REPLAY_S = 5;

type Source = "live" | "recorded" | "none";

function paintPane(ctx: CanvasRenderingContext2D, cw: number, ch: number, meta: SimMeta, frames: SimFrame[], idx: number, pane: PaneKey, highlight: Set<number> | null) {
  const s = cw / meta.width, sy = ch / meta.height;
  ctx.fillStyle = "#13171C";
  ctx.fillRect(0, 0, cw, ch);
  const [[ax, ay], [bx, by]] = meta.gate;
  ctx.strokeStyle = GATE; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(ax * s, ay * sy); ctx.lineTo(bx * s, by * sy); ctx.stroke();
  ctx.fillStyle = GATE; ctx.font = "600 11px Inter, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText("GATE", 8, ay * sy - 4);
  const fr = frames[idx];
  if (!fr) return;
  const fps = meta.fps;
  const ids = new Map(fr[pane].ids);
  const ev = recentEvents(frames, idx, pane, Math.round(TAG_S * fps));
  const dimEv = ev.filter((x) => x.age <= DIM_S * fps);
  const bright = highlight ?? (dimEv.length ? new Set(dimEv.map((x) => x.e[0])) : null);
  const alphaOf = (pid: number) => (bright && !bright.has(pid) ? 0.35 : 1);
  // trails: 1 s, per (person, tracker id) so they break at an ID change
  const n = Math.round(TRAIL_S * fps);
  for (const [pid] of fr.people) {
    const tid = ids.get(pid);
    let prev: [number, number] | null = null;
    ctx.strokeStyle = neutralColor(pid); ctx.lineWidth = 2; ctx.lineCap = "round";
    for (let k = Math.max(0, idx - n); k <= idx; k++) {
      const p = frames[k].people.find((q) => q[0] === pid);
      const same = new Map(frames[k][pane].ids).get(pid) === tid && tid != null;
      if (!p || !same) { prev = null; continue; }
      const pt: [number, number] = [p[1] * s, p[2] * sy];
      if (prev) { ctx.globalAlpha = alphaOf(pid) * 0.6 * (1 - (idx - k) / n); ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(pt[0], pt[1]); ctx.stroke(); }
      prev = pt;
    }
  }
  ctx.globalAlpha = 1;
  // unmatched tracker boxes (false positives, merged boxes): faint hollow squares
  for (const [tid, x, y] of fr[pane].extra) {
    ctx.globalAlpha = bright ? 0.2 : 0.5;
    ctx.strokeStyle = "#8A929C"; ctx.lineWidth = 1; ctx.strokeRect(x * s - 8, y * sy - 8, 16, 16);
    ctx.fillStyle = "#8A929C"; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(`#${tid}`, x * s, y * sy + 9);
  }
  ctx.globalAlpha = 1;
  // people: 10 px dots, fill = true identity; hidden ones hollow (the detector cannot see them)
  const chips: { id: number; text: string; cx: number; top: number; w: number }[] = [];
  ctx.font = "600 12px Inter, sans-serif";
  const tagFor = new Map(ev.map((x) => [x.e[0], x]));
  for (const [pid, x, y, vis] of fr.people) {
    const c = neutralColor(pid);
    ctx.globalAlpha = alphaOf(pid);
    ctx.beginPath(); ctx.arc(x * s, y * sy, 5, 0, Math.PI * 2);
    if (vis) { ctx.fillStyle = c; ctx.fill(); } else { ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.setLineDash([2, 2]); ctx.stroke(); ctx.setLineDash([]); }
    ctx.globalAlpha = 1;
    if (alphaOf(pid) < 1) continue;
    const tid = ids.get(pid);
    const tag = tagFor.get(pid);
    const text = tag ? `#${tag.e[1]} → #${tag.e[2]}` : tid != null ? `#${tid}` : vis ? "—" : "hidden";
    chips.push({ id: pid, text, cx: x * s, top: y * sy - 6, w: Math.max(22, ctx.measureText(text).width + 10) });
  }
  for (const c of layoutChips(chips, 22, cw)) {
    const tag = tagFor.get(c.id);
    ctx.fillStyle = tag ? "#FFFFFF" : "#161A1F";
    ctx.strokeStyle = neutralColor(c.id); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(c.x, c.y, c.w, 22, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = tag ? "#0E1013" : "#E6E8EB"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(c.text, c.x + c.w / 2, c.y + 12);
  }
  // 600 ms white ring pulse where an ID changed
  for (const { e, age } of ev) {
    const k = age / (RING_S * fps);
    if (k > 1) continue;
    ctx.strokeStyle = `rgba(255,255,255,${1 - k})`; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(e[3] * s, e[4] * sy, 8 + 16 * k, 0, Math.PI * 2); ctx.stroke();
  }
}

export default function GateScene() {
  const [ctl, setCtl] = useState<SimControls>(DEFAULTS);
  const [playing, setPlaying] = useState(true);
  const [source, setSource] = useState<Source>("live");
  const [meta, setMeta] = useState<SimMeta | null>(null);
  const [end, setEnd] = useState<SimEnd | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ui, setUi] = useState({ idx: 0, slow: false, buffered: 0 });
  const frames = useRef<SimFrame[]>([]);
  const head = useRef(0);
  const replay = useRef<{ until: number; ids: Set<number> } | null>(null);
  const cv = { bytetrack: useRef<HTMLCanvasElement>(null), ours: useRef<HTMLCanvasElement>(null) };

  // (re)start the stream whenever the controls change
  useEffect(() => {
    const ac = new AbortController();
    frames.current = []; head.current = 0; replay.current = null;
    setMeta(null); setEnd(null); setError(null);
    const onLine = (l: SimMeta | SimFrame | SimEnd) => {
      if (l.type === "meta") setMeta(l);
      else if (l.type === "frame") frames.current.push(l);
      else setEnd(l);
    };
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch(streamUrl(ctl), { signal: ac.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setSource("live");
        await readStream(res, onLine, ac.signal);
      } catch (e) {
        if (ac.signal.aborted) return;
        if (!isDefault(ctl)) { setSource("none"); setError("The backend is offline, so the trackers cannot run for these settings. Reset to the defaults to watch the recorded run, or start make serve."); return; }
        try {
          const rec = await fetch(RECORDING_URL, { signal: ac.signal });
          if (!rec.ok) throw new Error(`recording HTTP ${rec.status}`);
          setSource("recorded");
          await readStream(rec, onLine, ac.signal);
        } catch {
          if (!ac.signal.aborted) { setSource("none"); setError(`Could not start the scene (${(e as Error).message}). Start the backend with make serve.`); }
        }
      }
    }, 250);
    return () => { window.clearTimeout(t); ac.abort(); };
  }, [ctl]);

  // render loop at display rate, advancing the playhead at the scene fps (0.5x after a mix-up)
  useEffect(() => {
    if (!meta) return;
    let raf = 0, last = performance.now(), lastUi = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      const fs = frames.current;
      const idx0 = Math.floor(head.current);
      const recent = PANES.some((p) => recentEvents(fs, idx0, p, Math.round(DIM_S * meta.fps)).length > 0);
      const rate = recent ? 0.5 : 1;
      if (playing && fs.length) head.current = Math.min(fs.length - 1, head.current + dt * meta.fps * rate);
      const idx = Math.floor(head.current);
      if (replay.current && idx >= replay.current.until) replay.current = null;
      for (const p of PANES) {
        const c = cv[p].current, ctx = c?.getContext("2d");
        if (!c || !ctx) continue;
        const dpr = window.devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
        if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        paintPane(ctx, cw, ch, meta, fs, idx, p, replay.current?.ids ?? null);
      }
      if (now - lastUi > 120) { lastUi = now; setUi({ idx, slow: rate < 1, buffered: fs.length }); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, playing]);

  const doReplay = useCallback(() => {
    if (!meta) return;
    const idx = Math.floor(head.current);
    const from = Math.max(0, idx - Math.round(REPLAY_S * meta.fps));
    const fs = frames.current;
    const ids = involvedBetween(fs, fs[from]?.f ?? 0, fs[idx]?.f ?? 0);
    replay.current = { until: idx, ids };
    head.current = from;
    setPlaying(true);
  }, [meta]);

  const fr = frames.current[ui.idx];
  const atEnd = !!end && ui.idx >= frames.current.length - 1;
  const set = (k: keyof SimControls) => (e: React.ChangeEvent<HTMLInputElement>) => setCtl((c) => ({ ...c, [k]: Number(e.target.value) }));
  const fixed = source === "recorded";
  return (
    <div className="gate-scene" data-testid="gate-scene">
      <div className="gs-head">
        <h1 style={{ margin: 0, fontSize: 20 }}>Live gate scene</h1>
        <span className="tag" data-testid="synthetic-tag">Synthetic scene · exact ground truth</span>
        <span className="muted gs-sub">
          {source === "recorded" ? `Backend offline: replaying a recorded run of the real trackers (seed ${DEFAULTS.seed}, ${RECORDED_SECONDS} s). Sliders need make serve.`
            : "Simulated people → fake detector (same boxes for both) → the real Python trackers on the server."}
        </span>
      </div>
      {error && <div className="panel muted" role="alert" data-testid="scene-error" style={{ padding: 12 }}>{error}</div>}
      <div className="gs-panes">
        {PANES.map((p) => {
          const pf = fr?.[p];
          const over = pf && fr ? pf.counted > fr.true_count : false;
          return (
            <figure key={p} className="gs-pane" data-testid={`scene-${p}`}>
              <figcaption className="cmp-head">
                <b className={p === "ours" ? "ours" : "base"}>{meta?.trackers[p].label ?? (p === "ours" ? "BoT-SORT + visibility (ours)" : "ByteTrack")}</b>
                <span className="muted">{p === "ours" ? "our tracker" : "baseline"}</span>
              </figcaption>
              <div className="gs-stage">
                <canvas ref={cv[p]} aria-label={`Gate scene, ${p}`} />
                {!meta && !error && <div className="cmp-hint">Starting the trackers…</div>}
              </div>
              <div className="cmp-counters tabular" data-testid={`scene-counters-${p}`}>
                <span>True count <b>{fr?.true_count ?? 0}</b></span>
                <span>Counted <b className={over ? "over" : ""}>{pf?.counted ?? 0}</b></span>
                <span>ID changes <b>{pf?.id_changes ?? 0}</b></span>
              </div>
            </figure>
          );
        })}
      </div>
      <div className="gs-controls" role="group" aria-label="Scene controls">
        <label>Density <input type="range" min={0} max={1} step={0.05} value={ctl.density} onChange={set("density")} disabled={fixed} aria-label="Density" /></label>
        <label>Occlusion <input type="range" min={0} max={3} step={0.25} value={ctl.occlusion} onChange={set("occlusion")} disabled={fixed} aria-label="Occlusion duration" /> <span className="mono tabular">{ctl.occlusion.toFixed(2)} s</span></label>
        <label>Detector noise <input type="range" min={0} max={1} step={0.05} value={ctl.noise} onChange={set("noise")} disabled={fixed} aria-label="Detector noise" /></label>
        <button type="button" className="chip-btn mono" data-testid="seed-chip" title="New random seed (same seed replays identically)" disabled={fixed}
          onClick={() => setCtl((c) => ({ ...c, seed: Math.floor(Math.random() * 100000) }))}>seed {ctl.seed}</button>
        {!isDefault(ctl) && <button type="button" className="btn" onClick={() => setCtl(DEFAULTS)}>Reset</button>}
        <button type="button" className="btn" onClick={doReplay} disabled={!meta} title="Rewind 5 s, only the people involved highlighted">⟲ Replay 5 s</button>
        <button type="button" className="btn" onClick={() => (atEnd ? (head.current = 0, setPlaying(true)) : setPlaying((x) => !x))} aria-label={playing && !atEnd ? "Pause" : "Play"}>
          {atEnd ? "↺ Restart" : playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <span className="mono muted tabular">{meta ? `${(ui.idx / meta.fps).toFixed(1)} s${ui.slow ? " · 0.5x mix-up" : ""}` : ""}</span>
      </div>
    </div>
  );
}
