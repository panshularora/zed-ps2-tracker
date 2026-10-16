// Live gate scene: synthetic people, fake detector, ByteTrack vs Keep-ID (one body, one number).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { layoutChips } from "../compare/model";
import { differences, paneStats, type Diff } from "../compare/gateDiff";
import { neutralColor } from "../lib/scenarios";
import {
  DEFAULTS, PANES, RECORDED_SECONDS, beatCopy, firstKeepId, involvedBetween, isDefault, personIdLabel, readStream, recentEvents, recordingUrl, streamUrl,
  type Beat, type IdKind, type PaneKey, type PlayKind, type SimControls, type SimEnd, type SimFrame, type SimMeta,
} from "../lib/gateSim";

const GATE = "#F5A524";
const RING_S = 0.6, TAG_S = 2, DIM_S = 1.5, REPLAY_S = 5, DIFF_LEAD_S = 2, DIFF_TAIL_S = 1.5;
const CHIP_FILL: Record<IdKind, string> = { ok: "#1f6b45", held: "#b7811a", fail: "#E5484D", none: "#4a525c" };

type Source = "live" | "recorded" | "none";

function mix(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.round(((n >> 16) & 255) * amt);
  const g = Math.round(((n >> 8) & 255) * amt);
  const b = Math.round((n & 255) * amt);
  return `rgb(${r},${g},${b})`;
}

/** Pedestrian icon (head, shirt, swinging arms/legs). Hip is at (x, y). */
function drawPerson(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  visible: boolean,
  walk: number,
) {
  const k = 1.7;
  const headR = 6 * k;
  const shW = 8.5 * k;
  const torsoH = 15 * k;
  const hipW = 6.2 * k;
  const legH = 14 * k;
  const armL = 12 * k;
  const swing = Math.sin(walk) * (3.6 * k);
  const hipY = y;
  const torsoTop = hipY - torsoH;
  const headY = torsoTop - headR - 0.5 * k;
  const shirt = visible ? color : "transparent";
  const pants = visible ? mix(color, 0.62) : "transparent";
  ctx.save();
  ctx.globalAlpha *= visible ? 1 : 0.7;
  ctx.fillStyle = "rgba(0,0,0,0.32)";
  ctx.beginPath();
  ctx.ellipse(x, hipY + legH + 1, 9 * k, 2.6 * k, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (!visible) ctx.setLineDash([4, 3]);
  ctx.lineWidth = 3.2 * k;
  ctx.beginPath();
  ctx.moveTo(x - hipW * 0.35, hipY);
  ctx.lineTo(x - hipW * 0.45 - swing, hipY + legH);
  ctx.moveTo(x + hipW * 0.35, hipY);
  ctx.lineTo(x + hipW * 0.45 + swing, hipY + legH);
  ctx.stroke();
  if (visible) {
    ctx.strokeStyle = pants;
    ctx.lineWidth = 2.6 * k;
    ctx.stroke();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.4 * k;
  ctx.beginPath();
  ctx.moveTo(x - shW * 0.55, torsoTop + 3 * k);
  ctx.lineTo(x - shW * 0.7 - swing * 0.7, torsoTop + armL);
  ctx.moveTo(x + shW * 0.55, torsoTop + 3 * k);
  ctx.lineTo(x + shW * 0.7 + swing * 0.7, torsoTop + armL);
  ctx.stroke();
  ctx.fillStyle = shirt;
  ctx.beginPath();
  ctx.roundRect(x - shW, torsoTop, shW * 2, torsoH, 3.5 * k);
  if (visible) ctx.fill();
  else ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, headY, headR, 0, Math.PI * 2);
  if (visible) ctx.fill();
  else ctx.stroke();
  if (visible) {
    ctx.fillStyle = mix(color, 0.45);
    ctx.beginPath();
    ctx.ellipse(x, headY - headR * 0.25, headR * 0.95, headR * 0.55, 0, Math.PI, 0);
    ctx.fill();
  }
  ctx.setLineDash([]);
  ctx.restore();
}

function drawChestId(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, fill: string) {
  const k = 1.7;
  const torsoTop = y - 15 * k;
  ctx.save();
  ctx.font = `700 ${Math.round(8.5 * k)}px Inter, sans-serif`;
  ctx.fillStyle = fill;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, torsoTop + 8 * k);
  ctx.restore();
}

const FIG_HEAD_ABOVE = 6 * 1.7 * 2 + 15 * 1.7 + 4;

function paintPane(ctx: CanvasRenderingContext2D, cw: number, ch: number, meta: SimMeta, frames: SimFrame[], idx: number, pane: PaneKey, highlight: Set<number> | null, lookPid: number | null) {
  const s = cw / meta.width, sy = ch / meta.height;
  ctx.fillStyle = "#1a1510";
  ctx.fillRect(0, 0, cw, ch);
  const [[ax, ay], [bx, by]] = meta.gate;
  const gy = ay * sy;
  ctx.fillStyle = "#2a241c";
  ctx.fillRect(0, gy, cw, ch - gy);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let x = 40; x < cw; x += 48) {
    ctx.beginPath(); ctx.moveTo(x, gy); ctx.lineTo(x, ch); ctx.stroke();
  }
  ctx.strokeStyle = GATE; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(ax * s, gy); ctx.lineTo(bx * s, by * sy); ctx.stroke();
  ctx.fillStyle = GATE; ctx.font = "600 11px Inter, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText("GATE LINE", 8, gy - 6);
  const fr = frames[idx];
  if (!fr) return;
  const fps = meta.fps;
  const ids = new Map(fr[pane].ids);
  const ev = recentEvents(frames, idx, pane, Math.round(TAG_S * fps));
  const dimEv = ev.filter((x) => x.age <= DIM_S * fps);
  const bright = highlight ?? (lookPid != null ? new Set([lookPid]) : (dimEv.length ? new Set(dimEv.map((x) => x.e[0])) : null));
  const alphaOf = (pid: number) => (bright && !bright.has(pid) ? 0.28 : 1);
  ctx.globalAlpha = 1;
  for (const d of fr.dets) {
    const [dx, dy, dw, dh] = d;
    ctx.strokeStyle = "rgba(245,165,36,0.28)";
    ctx.lineWidth = 1;
    ctx.strokeRect(dx * s, dy * sy, dw * s, dh * sy);
  }
  for (const [tid, x, y] of fr[pane].extra) {
    ctx.globalAlpha = bright ? 0.2 : 0.45;
    ctx.strokeStyle = "#8A929C"; ctx.lineWidth = 1; ctx.strokeRect(x * s - 10, y * sy - 22, 20, 28);
    ctx.fillStyle = "#8A929C"; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(`#${tid}`, x * s, y * sy + 8);
  }
  for (const [tid, x, y] of fr[pane].held ?? []) {
    ctx.save();
    ctx.globalAlpha = 0.95;
    ctx.strokeStyle = "#b7811a";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x * s - 16, y * sy - 40, 32, 56);
    ctx.setLineDash([]);
    ctx.fillStyle = "#b7811a";
    ctx.font = "700 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(`held #${tid}`, x * s, y * sy - 44);
    ctx.restore();
  }
  ctx.globalAlpha = 1;
  const people = [...fr.people].sort((a, b) => a[2] - b[2]);
  for (let i = 0; i < people.length; i++) {
    for (let j = i + 1; j < people.length; j++) {
      const a = people[i], b = people[j];
      const dx = (a[1] - b[1]) * s, dy = (a[2] - b[2]) * sy;
      if (dx * dx + dy * dy < 28 * 28) {
        ctx.fillStyle = "rgba(245,165,36,0.18)";
        ctx.beginPath();
        ctx.ellipse(((a[1] + b[1]) / 2) * s, ((a[2] + b[2]) / 2) * sy, 22, 12, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  const chips: { id: number; text: string; cx: number; top: number; w: number; kind: IdKind }[] = [];
  ctx.font = "600 11px Inter, sans-serif";
  const tagFor = new Map(ev.map((x) => [x.e[0], x]));
  for (const [pid, x, y, vis] of people) {
    const c = neutralColor(pid);
    ctx.globalAlpha = alphaOf(pid);
    drawPerson(ctx, x * s, y * sy, c, !!vis, idx * 0.35 + pid);
    const tid = ids.get(pid);
    const keep = firstKeepId(frames, idx, pane, pid);
    const tag = tagFor.get(pid);
    const badge = personIdLabel(!!vis, tid, keep, tag ? tag.e[2] : null);
    if (vis && (tid != null || keep != null)) {
      drawChestId(ctx, x * s, y * sy, `#${tid ?? keep}`, badge.kind === "fail" ? "#fff" : "#0e1013");
    }
    if (lookPid === pid) {
      const pulse = 0.65 + 0.35 * Math.sin(idx * 0.45);
      ctx.save();
      ctx.globalAlpha = pulse;
      ctx.strokeStyle = GATE;
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.ellipse(x * s, y * sy - 18, 30, 46, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      const tag = vis ? `Look here · #${pid}` : `Look here · hidden #${pid}`;
      ctx.font = "700 12px Inter, sans-serif";
      const tw = ctx.measureText(tag).width + 16;
      const tx = Math.max(4, Math.min(cw - tw - 4, x * s - tw / 2));
      const ty = Math.max(4, y * sy - FIG_HEAD_ABOVE - 44);
      ctx.fillStyle = GATE;
      ctx.beginPath(); ctx.roundRect(tx, ty, tw, 22, 2); ctx.fill();
      ctx.fillStyle = "#0e1013"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(tag, tx + tw / 2, ty + 11);
    }
    ctx.globalAlpha = 1;
    chips.push({ id: pid, text: badge.text, cx: x * s, top: y * sy - FIG_HEAD_ABOVE, w: Math.max(28, ctx.measureText(badge.text).width + 12), kind: badge.kind });
  }
  for (const c of layoutChips(chips, 20, cw)) {
    const kind = chips.find((x) => x.id === c.id)?.kind ?? "none";
    ctx.fillStyle = CHIP_FILL[kind];
    ctx.beginPath(); ctx.roundRect(c.x, c.y, c.w, 20, 4); ctx.fill();
    ctx.fillStyle = "#ffffff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(c.text, c.x + c.w / 2, c.y + 11);
  }
  for (const { e, age } of ev) {
    const k = age / (RING_S * fps);
    if (k > 1) continue;
    ctx.strokeStyle = `rgba(229,72,77,${1 - k})`; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(e[3] * s, e[4] * sy - 18, 10 + 14 * k, 0, Math.PI * 2); ctx.stroke();
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
  const [why, setWhy] = useState<{ beat: Beat; pid: number } | null>(null);
  const frames = useRef<SimFrame[]>([]);
  const head = useRef(0);
  const replay = useRef<{ until: number; ids: Set<number> } | null>(null);
  const [diffAt, setDiffAt] = useState<number | null>(null);
  const heldBeats = useRef<Set<number>>(new Set());
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const cv = { bytetrack: useRef<HTMLCanvasElement>(null), ours: useRef<HTMLCanvasElement>(null) };

  // (re)start the stream whenever the controls change
  useEffect(() => {
    const ac = new AbortController();
    frames.current = []; head.current = 0; replay.current = null; heldBeats.current = new Set();
    setMeta(null); setEnd(null); setError(null); setWhy(null); setDiffAt(null);
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
          const rec = await fetch(recordingUrl(ctl.play), { signal: ac.signal });
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
      if (playingRef.current && fs.length) {
        const next = Math.max(0, Math.min(fs.length - 1, head.current + dt * meta.fps * rate));
        const from = Math.floor(head.current) + 1;
        const to = Math.floor(next);
        let pausedOnBeat = false;
        for (let i = from; i <= to; i++) {
          const b = fs[i];
          if (b?.beat && b.focus != null && !heldBeats.current.has(b.f)) {
            heldBeats.current.add(b.f);
            head.current = i;
            pausedOnBeat = true;
            setPlaying(false);
            setWhy({ beat: b.beat, pid: b.focus });
            break;
          }
        }
        if (!pausedOnBeat) head.current = next;
      }
      const idx = Math.floor(head.current);
      if (replay.current && idx >= replay.current.until) replay.current = null;
      const frNow = fs[idx];
      const hidden = frNow?.people.filter((p) => p[3] === 0).map((p) => p[0]) ?? [];
      const lookPid = (ctl.play === "keep" ? (frNow?.focus ?? hidden[0] ?? null) : (frNow?.beat ? frNow.focus : null)) ?? null;
      for (const p of PANES) {
        const c = cv[p].current;
        let ctx: CanvasRenderingContext2D | null = null;
        try { ctx = c?.getContext("2d") ?? null; } catch { ctx = null; }
        if (!c || !ctx) continue;
        const dpr = window.devicePixelRatio || 1, cw = c.clientWidth, ch = c.clientHeight;
        if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        paintPane(ctx, cw, ch, meta, fs, idx, p, replay.current?.ids ?? null, lookPid);
      }
      if (now - lastUi > 120) { lastUi = now; setUi({ idx, slow: rate < 1, buffered: fs.length }); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, playing, ctl.play]);

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

  const bucket = Math.floor(ui.buffered / 60);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const cmp = useMemo(() => (meta ? { stats: { bytetrack: paneStats(frames.current, "bytetrack"), ours: paneStats(frames.current, "ours") }, diffs: differences(frames.current, meta.fps) } : null), [meta, end, bucket]);
  const playDiff = useCallback((i: number) => {
    if (!meta || !cmp || !cmp.diffs.length) return;
    const k = ((i % cmp.diffs.length) + cmp.diffs.length) % cmp.diffs.length;
    const d: Diff = cmp.diffs[k];
    head.current = Math.max(0, d.idx - Math.round(DIFF_LEAD_S * meta.fps));
    replay.current = { until: Math.min(frames.current.length - 1, d.idx + Math.round(DIFF_TAIL_S * meta.fps)), ids: new Set(d.people) };
    setDiffAt(k);
    setPlaying(true);
  }, [meta, cmp]);
  const fr = frames.current[ui.idx];
  const atEnd = !!end && ui.idx >= frames.current.length - 1;
  const set = (k: "seed" | "density" | "occlusion" | "noise") => (e: React.ChangeEvent<HTMLInputElement>) => setCtl((c) => ({ ...c, [k]: Number(e.target.value) }));
  const setPlay = (play: PlayKind) => setCtl((c) => ({ ...c, play, occlusion: play === "keep" ? 1.5 : c.occlusion, noise: play === "keep" ? 0.15 : c.noise }));
  const fixed = source === "recorded";
  const copy = why ? beatCopy(why.beat, why.pid) : null;
  return (
    <div className="gate-scene" data-testid="gate-scene">
      <div className="gs-head">
        <h1 style={{ margin: 0, fontSize: 20 }}>One body, one number</h1>
        <span className="tag" data-testid="synthetic-tag">Synthetic scene · exact ground truth</span>
        <span className="muted gs-sub">
          {source === "recorded" ? `Backend offline: replaying a recorded run (seed ${DEFAULTS.seed}, ${RECORDED_SECONDS} s). Live trackers need make serve.`
            : "Two people walk to the gate. One goes behind the other. The box disappears. Ours must keep the same number; ByteTrack mints a new one."}
        </span>
      </div>
      <div className="gs-modes" role="tablist" aria-label="Scene">
        <button type="button" role="tab" aria-selected={ctl.play === "keep"} className={`chip-btn${ctl.play === "keep" ? " on" : ""}`} data-testid="play-keep" onClick={() => setPlay("keep")}>One body, one number</button>
        <button type="button" role="tab" aria-selected={ctl.play === "crowd"} className={`chip-btn${ctl.play === "crowd" ? " on" : ""}`} data-testid="play-crowd" onClick={() => setPlay("crowd")}>Random crowd</button>
      </div>
      <ol className="gs-steps" data-testid="how-id-kept">
        <li><b>ID #N (green).</b> In view, first number still on them. Shirt and chip match.</li>
        <li><b>HELD #N (amber).</b> Dashed body plus an amber box: they are behind someone, the detector has no box, the tracker is still remembering N.</li>
        <li><b>NEW #M was #N (red).</b> They came back and the tracker minted a new identity. Ours must not do this.</li>
      </ol>
      {copy && (
        <div className="gs-why" data-testid="gs-why">
          <div>
            <p className="gs-why-tag">{copy.tag}</p>
            <p>{copy.body}</p>
          </div>
          <button type="button" className="btn primary" data-testid="gs-continue" onClick={() => { setWhy(null); setPlaying(true); }}>Continue</button>
        </div>
      )}
      {error && <div className="panel muted" role="alert" data-testid="scene-error" style={{ padding: 12 }}>{error}</div>}
      <p className="gs-caption" data-testid="gate-caption"><b className="base">Left: ByteTrack</b>, the published tracker we start from. <b className="ours">Right: ours</b> = ByteTrack + rules that keep a hidden person&apos;s number.</p>
      {cmp && (
        <div className="cmp-strip tabular" data-testid="compare-strip">
          <div><div className="k">Same seed {meta?.params.seed} · {end ? "whole run" : "so far"}</div><div className="row"><b className="base">ByteTrack (left)</b><b className="ours">ours (right)</b></div></div>
          <div title="Times one simulated person's tracker number changed (the tracker's own count)"><div className="k">ID changes</div><div className="row"><span data-testid="strip-changes-bytetrack">{cmp.stats.bytetrack.idChanges}</span><span data-testid="strip-changes-ours">{cmp.stats.ours.idChanges}</span></div></div>
          <div title="False keeps: a tracker number that was on one simulated person later sits on a different person"><div className="k">Number given to someone else</div><div className="row"><span data-testid="strip-transfers-bytetrack">{cmp.stats.bytetrack.transfers}</span><span data-testid="strip-transfers-ours">{cmp.stats.ours.transfers}</span></div></div>
          <div title="Gate count from the tracker vs the true number of crossings"><div className="k">Gate count / true</div><div className="row"><span data-testid="strip-count-bytetrack">{cmp.stats.bytetrack.counted} / {cmp.stats.bytetrack.truth}</span><span data-testid="strip-count-ours">{cmp.stats.ours.counted} / {cmp.stats.ours.truth}</span></div></div>
          <div><button type="button" className="btn" data-testid="show-diffs" disabled={!cmp.diffs.length} onClick={() => playDiff(diffAt == null ? 0 : diffAt + 1)}>
            {diffAt == null ? `Show me the ${cmp.diffs.length} moments where they differ` : `Next difference ▸ (${diffAt + 1}/${cmp.diffs.length})`}</button></div>
        </div>
      )}
      {cmp && diffAt != null && cmp.diffs[diffAt] && (
        <div className="diff-bar" data-testid="diff-bar">
          <b>Difference {diffAt + 1} of {cmp.diffs.length}</b>
          <span data-testid="diff-text">{cmp.diffs[diffAt].text} (frame {cmp.diffs[diffAt].f}). Both panes highlight the same simulated person{cmp.diffs[diffAt].people.length > 1 ? "s" : ""}.</span>
          <button type="button" className="chip-btn" onClick={() => playDiff(diffAt - 1)}>◂ Prev</button>
          <button type="button" className="chip-btn" onClick={() => playDiff(diffAt)}>⟲ Replay</button>
          <button type="button" className="chip-btn" onClick={() => playDiff(diffAt + 1)}>Next ▸</button>
          <button type="button" className="chip-btn" onClick={() => { setDiffAt(null); replay.current = null; }}>Done</button>
        </div>
      )}
      <div className="gs-panes">
        {PANES.map((p) => {
          const pf = fr?.[p];
          const over = pf && fr ? pf.counted > fr.true_count : false;
          return (
            <figure key={p} className="gs-pane" data-testid={`scene-${p}`}>
              <figcaption className="cmp-head">
                <b className={p === "ours" ? "ours" : "base"}>{meta?.trackers[p].label ?? (p === "ours" ? "Ours · one body, one number" : "ByteTrack")}</b>
                <span className="muted">{p === "ours" ? "Keep-ID · rebind the lost box" : "baseline · 30-frame buffer"}</span>
              </figcaption>
              <div className="gs-stage">
                <canvas ref={cv[p]} aria-label={`Gate scene, ${p}`} />
                {!meta && !error && <div className="cmp-hint">Starting the trackers…</div>}
              </div>
              <div className="cmp-counters tabular" data-testid={`scene-counters-${p}`}>
                <span>True people <b>{fr?.true_count ?? 0}</b></span>
                <span>Counted <b className={over ? "over" : ""}>{pf?.counted ?? 0}</b></span>
                <span>New numbers issued <b>{pf?.id_changes ?? 0}</b></span>
              </div>
              <p className="gs-pane-note muted">
                {p === "ours"
                  ? "Same body keeps the same number from approach to exit. Amber HELD while the box is gone, then the original ID when they step out."
                  : "After the box is gone longer than 30 frames, ByteTrack drops the track and issues a new number when they reappear."}
              </p>
            </figure>
          );
        })}
      </div>
      <div className="gs-legend" aria-hidden>
        <span><i className="gs-swatch keep" /> ID #N — in view, same number</span>
        <span><i className="gs-swatch held" /> HELD #N — behind someone, tracker still remembers N</span>
        <span><i className="gs-swatch fail" /> NEW #M was #N — tracker forgot and issued a new identity</span>
      </div>
      <div className="gs-controls" role="group" aria-label="Scene controls">
        {ctl.play === "crowd" && (
          <>
            <label className="gs-slider">
              Density
              <input type="range" min={0} max={1} step={0.05} value={ctl.density} onChange={set("density")} disabled={fixed} aria-label="Density" />
              <span className="gs-hint">How packed the crowd is. Higher = more people overlapping at the gate.</span>
            </label>
            <label className="gs-slider">
              Occlusion
              <input type="range" min={0} max={3} step={0.25} value={ctl.occlusion} onChange={set("occlusion")} disabled={fixed} aria-label="Occlusion duration" />
              <span className="mono tabular">{ctl.occlusion.toFixed(2)} s</span>
              <span className="gs-hint">How long someone stays fully hidden. The tracker should still call them the same number when they reappear.</span>
            </label>
            <label className="gs-slider">
              Detector noise
              <input type="range" min={0} max={1} step={0.05} value={ctl.noise} onChange={set("noise")} disabled={fixed} aria-label="Detector noise" />
              <span className="gs-hint">How messy the boxes are (jitter, missed boxes, fake boxes).</span>
            </label>
            <button type="button" className="chip-btn mono" data-testid="seed-chip" title="New random seed (same seed replays identically)" disabled={fixed}
              onClick={() => setCtl((c) => ({ ...c, seed: Math.floor(Math.random() * 100000) }))}>seed {ctl.seed}</button>
          </>
        )}
        {ctl.play === "keep" && <span className="gs-hint" data-testid="seed-chip">seed {ctl.seed} · scripted hide</span>}
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
