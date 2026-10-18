import { useEffect, useMemo, useRef, useState } from "react";
import { neutralColor } from "../lib/scenarios";
import { fallbackText, videoSources, type ReplayData } from "../guided/model";
import { PersonTimeline } from "./PersonTimeline";
import { boxesAt, countsAt, eventLines, frameTag, people as peopleOf, pickFocus, placeChips, storySteps, tracksOf, watchLine, type Drawn, type Person, type Step, type XYWH } from "./story";
import { fetchVerify, methodTooltip, verdictOf, type VerdictOf, type VerifyDoc } from "./verify";
import { storyMoments, type StoryMoment } from "./moments";
import { applyContent, letterbox, stageAspect, stageToSource } from "../lib/letterbox";

const SWAP_FILL = "#8A929C";

export const STEP_PAUSE_MS = 1800;
const ZOOM = 1.6, ZOOM_MS = 500, TRAIL_S = 1.5;
export const OTHER_ALPHA = 0.4, OTHER_GREY = "#8A929C";
const GATE = "#F5A524";
const ease = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/** Counters live in a bar outside the frame: a single row "IN n · OUT n". */
/** The focus person gets a solid chip in its ID colour and a 2 px box; everyone else is a thin grey
 *  outline at 40 % with no chip; their events go to the caption bar. */
/** Solid chips; the chip of a hidden person is dimmed (their box is dashed at the last-seen spot). */
export const chipAlpha = (state: Drawn["state"]) => (state === "hidden" ? 0.5 : 1);

/** mm:ss of a clip time in seconds (floored, so it ticks once per second while playing). */
export const clipClock = (s: number) => { const t = Math.max(0, Math.floor(s)); return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`; };

export function CountBar({ inN, outN, note, t = 0 }: { inN: number; outN: number; note?: string | null; t?: number }) {
  return (
    <div className="count-bar" data-testid="count-bar">
      <span className="count-bar-label muted" data-testid="count-bar-label">So far at {clipClock(t)}:</span>
      <span className="count-bar-nums tabular"><b>IN</b> {inN} <span className="muted">·</span> <b>OUT</b> {outN}</span>
      {note && <span className="count-bar-note muted" data-testid="video-hint">{note}</span>}
    </div>
  );
}

export function IdentityLegend({ doc, unchecked }: { doc?: VerifyDoc | null; unchecked?: boolean }) {
  if (unchecked) {
    return (
      <div className="id-legend" data-testid="identity-legend">
        <span><i className="lg lg-box" />Box + number = what the tracker drew</span>
        <span><i className="lg lg-dash" />Dashed = that number, not seen for a moment</span>
        <span data-testid="kept-tooltip">Back with the same number = not checked (no ground truth, no re-identification)</span>
        <span><i className="lg lg-line" />Your gate line: crossing it adds to IN or OUT</span>
      </div>
    );
  }
  const tip = methodTooltip(doc ?? null);
  return (
    <div className="id-legend" data-testid="identity-legend">
      <span><i className="lg lg-box" />Box + number = one person</span>
      <span><i className="lg lg-dash" />Dashed = still that number, hidden</span>
      <span title={tip} data-testid="kept-tooltip"><i className="lg lg-kept">kept</i>Came back and the check says same person <span className="info" aria-label={tip}>ⓘ</span></span>
      <span title={tip}><i className="lg lg-swap">possible swap</i>Number reappeared on someone who does not match</span>
      <span><i className="lg lg-line" />Gate line: crossing it adds to IN or OUT</span>
    </div>
  );
}

/** Steps for the confirmed story moments (hidden → back still #N), in time order. */
export function momentSteps(ms: StoryMoment[]): Step[] {
  return ms.flatMap((m) => [
    { f: m.ev.last + 1, kind: "hidden" as const, text: `Person #${m.ev.id} goes out of sight${m.ev.occluder != null ? ` behind #${m.ev.occluder}` : ""}. Dashed box = "still #${m.ev.id}, hidden".` },
    { f: m.ev.back, kind: "back" as const, text: m.caption },
  ]).sort((a, b) => a.f - b.f);
}

export function useVerify(clip: string, given?: VerifyDoc | null): VerifyDoc | null {
  const [doc, setDoc] = useState<VerifyDoc | null>(given ?? null);
  useEffect(() => {
    if (given !== undefined) { setDoc(given); return; }
    let live = true;
    fetchVerify(clip).then((d) => live && setDoc(d));
    return () => { live = false; };
  }, [clip, given]);
  return doc;
}

export function IdentityPlayer({ data, watch, verify, seek, unchecked }: { data: ReplayData; watch?: string; verify?: VerifyDoc | null; seek?: { f: number; n: number; leadS?: number; playS?: number } | null; unchecked?: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const vdoc = useVerify(data.id, verify);
  const verdict = useMemo<VerdictOf>(() => verdictOf(vdoc), [vdoc]);
  const tracks = useMemo(() => tracksOf(data.runs.ours.frames), [data]);
  const { people } = useMemo(() => peopleOf(data, "ours", verdict), [data, verdict]);
  const moments = useMemo(() => storyMoments(data.id, vdoc), [data.id, vdoc]);
  const focus = useMemo<Person | null>(() => {
    const m = moments[0];
    return (m && people.find((p) => p.ids.includes(m.ev.id))) || pickFocus(people);
  }, [people, moments]);
  const steps = useMemo(() => {
    if (moments.length) return momentSteps(moments);
    return focus ? storySteps(data, focus, verdict) : [];
  }, [data, focus, verdict, moments]);
  const swaps = useMemo(() => people.flatMap((p) => p.swaps.map((s) => ({ ...s, key: p.key }))).sort((a, b) => a.f - b.f), [people]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [story, setStory] = useState(!unchecked);
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
  const stopAt = useRef(0); // seconds; >0 while playing a short window after a seek request
  const zoomK = useRef({ k: 0, at: 0, target: 0 });
  const followRef = useRef(follow); followRef.current = follow;
  const focusIds = useMemo(() => new Set([...(focus?.ids ?? []), ...moments.map((m) => m.ev.id)]), [focus, moments]);

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
        if (playing && !holding && v.paused) void Promise.resolve(v.play()).catch(() => undefined);
        if ((!playing || holding) && !v.paused) v.pause();
        if (v.currentTime - off > data.duration) { v.currentTime = off; }
        cur = Math.max(0, v.currentTime - off);
      } else {
        if (playing && !holding) clock.current += dt;
        if (clock.current > data.duration) clock.current = 0;
        cur = clock.current;
      }
      if (stopAt.current > 0 && cur >= stopAt.current) { stopAt.current = 0; setPlaying(false); }
      // story: pause briefly on each key moment of the focus person
      if (story && stopAt.current === 0 && playing && !holding && cur > prev.current && cur - prev.current < 0.5) {
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
      const all = boxesAt(tracks, ff, data.fps, verdict);
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
  }, [data, src, off, playing, story, steps, tracks, focusIds, verdict]);

  function seekFrame(f: number, lead = 0.3) {
    const s = Math.max(0, (f - 1) / data.fps - lead);
    clock.current = s; prev.current = s; holdUntil.current = 0;
    if (video.current) video.current.currentTime = s + off;
    setStep(null); setT(s);
  }

  useEffect(() => {
    if (!seek) return;
    const lead = seek.leadS ?? 0.3;
    seekFrame(seek.f, lead);
    const start = Math.max(0, (seek.f - 1) / data.fps - lead);
    if (seek.playS && seek.playS > 0) { stopAt.current = start + seek.playS; setPlaying(true); }
    else { stopAt.current = 0; setPlaying(false); }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- seek only when a new request (n) arrives
  }, [seek?.n]);

  function onClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const p = stageToSource(letterbox(r.width, r.height, data.width, data.height), e.clientX - r.left, e.clientY - r.top);
    if (!p) { setFollow(null); return; }
    const [x, y] = p;
    const ff = Math.max(1, 1 + t * data.fps);
    const hitBox = boxesAt(tracks, ff, data.fps, verdict).find((d) => x >= d.box[0] && x <= d.box[0] + d.box[2] && y >= d.box[1] && y <= d.box[1] + d.box[3]);
    setFollow((cur) => (hitBox && hitBox.id !== cur ? hitBox.id : null));
  }

  const f = Math.min(data.frames, Math.max(1, Math.floor(1 + t * data.fps)));
  const counts = countsAt(data.events.ours, f);
  const cur = step != null ? steps[step] : null;
  const liveMoment = moments.filter((m) => { const dt = t - (m.ev.back - 1) / data.fps; return dt >= -0.05 && dt < 2.5; }).sort((a, b) => b.ev.back - a.ev.back)[0];
  const others = eventLines(boxesAt(tracks, Math.max(1, 1 + t * data.fps), data.fps, verdict), follow != null ? new Set([follow]) : focusIds);
  const caption = cur && t - (cur.f - 1) / data.fps < 3 ? cur.text
    : liveMoment ? liveMoment.caption
    : follow != null ? `Following #${follow}. Click again or press Esc to stop.`
    : story ? (steps.length ? "Story mode: the replay pauses briefly at each key moment for the highlighted person. Click any box to follow someone." : "Click any box to follow that person.")
    : "Free play. Click any box to follow that person.";
  return (
    <div className="idp" data-testid="identity-player">
      <p className="watch-line" data-testid="watch-line">{watch ?? (moments.length
        ? `Watch ${moments.map((m) => `#${m.ev.id}`).join(" and ")}: hidden for a moment, back with the same number, and the check agrees it is the same person.`
        : watchLine(focus, steps))}</p>
      <CountBar inN={counts.in} outN={counts.out} note={hint} t={t} />
      <div className="idp-stage" data-testid="identity-stage" data-t={t.toFixed(2)} data-playing={playing ? "1" : "0"} style={{ aspectRatio: String(stageAspect(data.width, data.height)) }}>
        {src && <video key={src} ref={video} src={src} muted playsInline autoPlay className="idp-video" aria-label={data.title}
          onError={() => setSrcIdx((i) => i + 1)} onLoadedMetadata={(e) => { if (!(e.currentTarget.duration > 0)) setSrcIdx((i) => i + 1); }} />}
        <canvas ref={canvas} className="idp-canvas" onClick={onClick} aria-label={`${data.title}: ${data.runs.ours.label}. Click a person to follow.`} />
      </div>
      <div className="idp-caption" aria-live="polite">
        <span data-testid="story-caption">{caption}</span>
        {others.length > 0 && <span className="idp-events muted" data-testid="event-line">Also now: {others.join(" · ")}</span>}
      </div>
      <div className="idp-controls">
        <button type="button" className="btn" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
        <span className="mono muted tabular">{t.toFixed(1)} / {data.duration.toFixed(1)} s</span>
        <div className="toggle" role="group" aria-label="Mode">
          <button type="button" className={`btn${story ? " on-ours" : ""}`} aria-pressed={story} onClick={() => setStory(true)}>Story</button>
          <button type="button" className={`btn${!story ? " on-ours" : ""}`} aria-pressed={!story} onClick={() => { setStory(false); holdUntil.current = 0; }}>Free play</button>
        </div>
        <span className="muted idp-run">{data.runs.ours.label}</span>
      </div>
      <PersonTimeline people={people} frames={data.frames} f={f} focus={follow ?? focus?.key ?? null} onSeek={seekFrame} unchecked={unchecked} />
      {moments.length > 0 && (
        <div className="kept-moments" data-testid="kept-moments">
          <b>Verified moments</b>
          {moments.map((m) => (
            <button key={`${m.ev.id}:${m.ev.back}`} type="button" className="chip-btn" onClick={() => seekFrame(m.ev.last)} data-testid="kept-moment">{m.caption}</button>
          ))}
        </div>
      )}
      {swaps.length > 0 && (
        <details className="swap-list" data-testid="swap-list">
          <summary>{swaps.length} possible {swaps.length === 1 ? "swap" : "swaps"} flagged on this clip (counted as uncertain, not kept)</summary>
          {swaps.map((s) => (
            <button key={`${s.id}:${s.f}`} type="button" className="chip-btn mono" onClick={() => seekFrame(s.last)} data-testid="swap-item">
              #{s.id} · frame {s.last} → {s.f} · {s.why}
            </button>
          ))}
        </details>
      )}
      <IdentityLegend doc={vdoc} unchecked={unchecked} />
    </div>
  );
}

export function paintFrame(ctx: CanvasRenderingContext2D, cw: number, ch: number, d: ReplayData, src: CanvasImageSource | null, view: XYWH, all: Drawn[], follow: number | null, focus: Set<number>, ff: number, tracks: Map<number, Map<number, XYWH>>) {
  const [vx, vy, vw, vh] = view;
  const L = letterbox(cw, ch, vw, vh); // object-fit: contain; overlays share the content rect
  const s = L.s;
  ctx.save();
  ctx.fillStyle = "#0E1013"; // bars
  ctx.fillRect(0, 0, cw, ch);
  ctx.fillStyle = "#13171C"; // empty frame when no video
  ctx.fillRect(L.x, L.y, L.w, L.h);
  applyContent(ctx, L, vx, vy);
  if (src) ctx.drawImage(src, 0, 0, d.width, d.height);
  const px = 1 / s;
  if (d.gate) {
    ctx.strokeStyle = GATE; ctx.lineWidth = 3 * px;
    ctx.beginPath(); ctx.moveTo(...d.gate.a); ctx.lineTo(...d.gate.b); ctx.stroke();
    gateLabels(ctx, d.gate, px);
  }
  const hasFocus = focus.size > 0;
  // outside the focus everyone (a possible swap too) is a thin grey outline; their events are listed
  // in the caption bar under the frame ("Also now: …"), so nothing is hidden
  const isF = (id: number) => !hasFocus || focus.has(id) || id === follow;
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
    if (!isF(b.id)) {
      // not the focus person: thin grey outline at 40 %, no chip
      ctx.globalAlpha = OTHER_ALPHA;
      ctx.strokeStyle = OTHER_GREY; ctx.lineWidth = 1 * px;
      if (b.state === "hidden" || b.state === "swap") ctx.setLineDash([8 * px, 6 * px]);
      ctx.strokeRect(...b.box);
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      continue;
    }
    const c = neutralColor(b.id);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = c; ctx.lineWidth = 2 * px;
    if (b.state === "hidden" || b.state === "swap") ctx.setLineDash([8 * px, 6 * px]);
    ctx.strokeRect(...b.box);
    ctx.setLineDash([]);
    if (b.id === follow) {
      ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2 * px;
      ctx.strokeRect(b.box[0] - 4 * px, b.box[1] - 4 * px, b.box[2] + 8 * px, b.box[3] + 8 * px);
    }
    const text = `#${b.id}`; // the chip is only the number; events never become text on the frame
    const size = b.id === follow ? 18 : 13;
    ctx.font = `600 ${size * px}px Inter, sans-serif`;
    items.push({ id: b.id, box: b.box, text, w: ctx.measureText(text).width + 10 * px, h: (size + 7) * px, d: b });
  }
  // only the focus person's event is drawn on the frame, as one short tag next to their chip
  const tagged = (id: number) => (follow != null ? id === follow : focus.has(id));
  for (const c of placeChips(items)) {
    const it = items.find((i) => i.id === c.id)!;
    const size = c.id === follow ? 18 : 13;
    ctx.globalAlpha = chipAlpha(it.d.state);
    ctx.fillStyle = it.d.state === "swap" ? SWAP_FILL : neutralColor(c.id);
    const x = Math.min(Math.max(vx, c.x), vx + vw - c.w), y = Math.max(vy, c.y);
    ctx.beginPath(); ctx.roundRect(x, y, c.w, c.h, 4 * px); ctx.fill();
    ctx.fillStyle = "#0E1013"; ctx.font = `600 ${size * px}px Inter, sans-serif`;
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(c.text, x + 5 * px, y + c.h / 2 + 0.5 * px);
    ctx.globalAlpha = 1;
    const tag = frameTag(it.d, tagged(c.id));
    if (tag) {
      ctx.font = `600 ${12 * px}px Inter, sans-serif`;
      const tw = ctx.measureText(tag).width + 10 * px, th = 18 * px;
      let tx = x + c.w + 4 * px;
      if (tx + tw > vx + vw) tx = Math.max(vx, x - 4 * px - tw);
      const ty = y + (c.h - th) / 2;
      ctx.fillStyle = "#161A1F"; ctx.beginPath(); ctx.roundRect(tx, ty, tw, th, 4 * px); ctx.fill();
      ctx.strokeStyle = it.d.state === "swap" ? SWAP_FILL : neutralColor(c.id); ctx.lineWidth = 1 * px; ctx.stroke();
      ctx.fillStyle = "#E6E8EB"; ctx.fillText(tag, tx + 5 * px, ty + th / 2 + 0.5 * px);
    }
  }
  ctx.globalAlpha = 1;
  ctx.restore();
}

/** IN / OUT arrows on the gate line (eval/gate_events.py: IN = the line's left-hand normal). */
export function gateNormal(g: { a: [number, number]; b: [number, number] }): [number, number] {
  const dx = g.b[0] - g.a[0], dy = g.b[1] - g.a[1], L = Math.hypot(dx, dy) || 1;
  return [-dy / L, dx / L];
}
function gateLabels(ctx: CanvasRenderingContext2D, g: { a: [number, number]; b: [number, number] }, px: number) {
  const [nx, ny] = gateNormal(g);
  const mx = g.a[0] + (g.b[0] - g.a[0]) * 0.9, my = g.a[1] + (g.b[1] - g.a[1]) * 0.9;
  const k = 26 * px;
  ctx.save();
  ctx.font = `700 ${13 * px}px Inter, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = GATE; ctx.strokeStyle = GATE; ctx.lineWidth = 2 * px;
  for (const [sgn, label] of [[1, "IN"], [-1, "OUT"]] as const) {
    const tx = mx + nx * k * sgn, ty = my + ny * k * sgn;
    ctx.beginPath(); ctx.moveTo(mx + nx * 6 * px * sgn, my + ny * 6 * px * sgn); ctx.lineTo(tx - nx * 9 * px * sgn, ty - ny * 9 * px * sgn); ctx.stroke();
    ctx.fillText(label, tx + nx * 4 * px * sgn, ty + ny * 4 * px * sgn);
  }
  ctx.restore();
}
