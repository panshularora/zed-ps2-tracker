import { useEffect, useMemo, useRef, useState } from "react";
import type { RunKey } from "../lib/scenarios";
import { paint, paintStill } from "./draw";
import {
  activeMoment, answerFor, askText, countsAt, eventLog, extraCountsAt, frameAtTime, recentCrossers,
  resultText, schedule, spotlightIds, STAKE_TEXT, stopCrossed, stopKey, timeOfFrame, viewRect, zoomAt,
  rateAt, fallbackText, videoSources, visibleLog, type ReplayData, type Stop, MOMENT_RULE, SEEK_LEAD_S, stepMoment, stripTicks, tally, tallyText, type Tick,
} from "./model";
import { Tag } from "./Tag";

const AUTO_RESUME_MS = 4000;
/** header + title + legend + caption + tally/strip + controls: keep them on screen at ~768 px */
const CHROME_PX = 400;

export function Legend() {
  return (
    <div className="g-legend" data-testid="guided-legend">
      <span><i className="g-ic g-ic-box" aria-hidden />A box is one person the tracker sees.</span>
      <span><i className="g-ic g-ic-chip" aria-hidden>4</i>The number is that person&apos;s ID. It should never change.</span>
      <span><i className="g-ic g-ic-line" aria-hidden />Crossing the dashed line adds 1 to IN or OUT.</span>
    </div>
  );
}

export function GuidedPlayer({ data, initialRun = "ours" }: { data: ReplayData; initialRun?: RunKey }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const stillB = useRef<HTMLCanvasElement>(null);
  const stillO = useRef<HTMLCanvasElement>(null);
  const logRef = useRef<HTMLOListElement>(null);
  const [run, setRun] = useState<RunKey>(initialRun);
  const [guided, setGuided] = useState(true);
  const [t, setT] = useState(0);
  const [paused, setPaused] = useState<Stop | null>(null);
  const [userPaused, setUserPaused] = useState(false);
  const [stake, setStake] = useState<number | null>(null);
  const stops = useMemo(() => schedule(data), [data]);
  const done = useRef(new Set<string>());
  const prevT = useRef(0);
  const clock = useRef(0); // fallback clock when there is no video
  const tl = useMemo(() => tally(data.moments), [data]);
  const names = { baseline: data.runs.baseline.short, ours: data.runs.ours.short };
  const cap = (x: string) => x.charAt(0).toUpperCase() + x.slice(1);
  const ticks = useMemo(() => stripTicks(data, { baseline: cap(names.baseline), ours: names.ours }), [data, names.baseline, names.ours]);
  const fullLog = useMemo(() => eventLog(data, run), [data, run]);
  const differ = useMemo(() => {
    if (data.moments.length > 0) return true;
    const n = (r: RunKey) => Object.values(data.runs[r].frames).reduce((a, b) => a + b.length, 0);
    const ids = (r: RunKey) => JSON.stringify(Object.values(data.runs[r].frames).map((b) => b.map((x) => x[0])));
    return n("baseline") !== n("ours") || ids("baseline") !== ids("ours");
  }, [data]);
  const f = frameAtTime(data, t);
  const counts = countsAt(data.events[run], f);
  const extra = extraCountsAt(data, run, t);
  const stopped = paused != null || userPaused;
  const log = visibleLog(fullLog, f);
  const off = data.videoOffsetS ?? 0;
  const sources = useMemo(() => videoSources(data), [data]);
  const [srcIdx, setSrcIdx] = useState(0);
  const src = sources[srcIdx] ?? null;
  const failed = sources.length > 0 && src == null;
  const hint = fallbackText(data, failed);
  const onVideoError = () => setSrcIdx((i) => i + 1);

  // main loop: read clock, apply guided rate / pauses, paint
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const v = video.current;
      const dt = (now - last) / 1000;
      last = now;
      let cur: number;
      if (v && src) {
        if (v.currentTime - off > data.duration || v.currentTime < off - 0.05) { v.currentTime = off; done.current.clear(); }
        cur = Math.max(0, v.currentTime - off);
      } else {
        if (!stopped) clock.current += dt * rateAt(stops, clock.current, guided);
        if (clock.current > data.duration) { clock.current = 0; done.current.clear(); }
        cur = clock.current;
      }
      if (guided && !stopped) {
        const s = stopCrossed(stops, prevT.current, cur, done.current);
        if (s) {
          done.current.add(stopKey(s));
          cur = s.t;
          if (v) { v.pause(); v.currentTime = s.t + off; }
          clock.current = s.t;
          setPaused(s);
          if (s.kind === "answer") setStake(null);
        }
      }
      if (v) v.playbackRate = rateAt(stops, cur, guided);
      prevT.current = cur;
      setT(cur);
      const c = canvas.current;
      if (c) {
        const dpr = window.devicePixelRatio || 1;
        const cw = c.clientWidth, ch = c.clientHeight;
        if (c.width !== Math.round(cw * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
        const ctx = c.getContext("2d");
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          const fr = frameAtTime(data, cur);
          const mi = guided ? activeMoment(stops, cur) : null;
          const z = guided ? zoomAt(stops, cur) : { k: 0, index: null };
          const focus = z.index != null ? data.moments[z.index].focus : null;
          const spot = mi != null ? spotlightIds(data.moments[mi], run) : fr != null ? recentCrossers(data.events[run], fr, data.fps) : new Set<number>();
          const pulse = fr == null ? 0 : Math.max(0, ...data.events[run].filter((e) => e.t === "cross" && e.f <= fr).map((e) => 1 - (fr - e.f) / (0.3 * data.fps)));
          paint(ctx, cw, ch, data, v && src && v.readyState >= 2 ? v : null, {
            run, f: fr, view: viewRect(data.width, data.height, focus, z.k), spot, dim: mi != null ? 1 : 0, gatePulse: pulse,
          });
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [data, stops, guided, run, stopped, off, src]);

  // resume video when nothing holds it
  useEffect(() => {
    const v = video.current;
    if (!v) return;
    if (stopped) v.pause();
    else v.play()?.catch(() => undefined);
  }, [stopped]);

  // auto-resume after a pause; after an answer show the stake line
  useEffect(() => {
    if (!paused) return;
    const id = window.setTimeout(() => resume(), AUTO_RESUME_MS);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paused]);

  // frozen stills at an answer pause
  useEffect(() => {
    if (paused?.kind !== "answer") return;
    const m = data.moments[paused.index];
    const fr = frameAtTime(data, paused.t) ?? m.f;
    for (const [ref, r] of [[stillB, "baseline"], [stillO, "ours"]] as const) {
      const c = ref.current;
      const ctx = c?.getContext("2d");
      if (!c || !ctx) continue;
      const dpr = window.devicePixelRatio || 1;
      c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      paintStill(ctx, c.clientWidth, c.clientHeight, data, src && video.current && video.current.readyState >= 2 ? video.current : null, r, fr, m.focus, spotlightIds(m, r));
    }
  }, [paused, data, src]);

  // keep the current log line in view
  useEffect(() => {
    const el = logRef.current?.querySelector<HTMLElement>("[data-current='true']");
    if (el && logRef.current) logRef.current.scrollTop = Math.max(0, el.offsetTop - logRef.current.offsetTop - 60);
  }, [f]);

  function seekMoment(i: number) {
    seek(Math.max(0, timeOfFrame(data, data.moments[i].f) - SEEK_LEAD_S));
  }

  const tRef = useRef(0);
  tRef.current = t;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const el = e.target as HTMLElement | null;
      if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return;
      const i = stepMoment(data, tRef.current, e.key === "ArrowRight" ? 1 : -1);
      if (i == null) return;
      e.preventDefault();
      seekMoment(i);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  function resume() {
    if (paused?.kind === "answer") setStake(paused.index);
    setPaused(null);
  }

  function seek(s: number) {
    const x = Math.max(0, Math.min(data.duration, s));
    // stops at or after the target play again
    // moments whose question is already behind x are skipped whole (no stray answer pause);
    // everything from x on plays again
    const askT = new Map(stops.filter((st) => st.kind === "ask").map((st) => [st.index, st.t]));
    for (const st of stops) {
      if ((askT.get(st.index) ?? st.t) < x - 1e-3) done.current.add(stopKey(st));
      else done.current.delete(stopKey(st));
    }
    prevT.current = x;
    clock.current = x;
    if (video.current) video.current.currentTime = x + off;
    setPaused(null);
    setStake(null);
    setT(x);
  }

  const mIdx = paused?.index ?? activeMoment(stops, t);
  const passed = data.moments.reduce((k, x, i) => (timeOfFrame(data, x.f) <= t + 1e-3 ? i : k), -1);
  const kIdx = mIdx ?? (passed >= 0 ? passed : null);
  const m = mIdx != null ? data.moments[mIdx] : null;
  let caption: string;
  if (paused?.kind === "ask" && m) caption = askText(m);
  else if (paused?.kind === "answer" && m) caption = resultText(m, data.kind, names);
  else if (!guided) caption = "Free play: no pauses. Switch on Guided to stop at the key moments.";
  else if (stake != null && t - timeOfFrame(data, data.moments[stake].f) < 4) caption = STAKE_TEXT;
  else if (data.moments.length === 0) caption = "The two trackers agree on this clip. Watch the numbers stay the same as people cross the line.";
  else {
    const next = data.moments.findIndex((x) => timeOfFrame(data, x.f) > t);
    caption = next >= 0 ? `Keep watching: moment ${next + 1} of ${data.moments.length} is coming up.` : "That was the last key moment. Drag back to replay any of them.";
  }

  return (
    <div className="guided" data-testid="guided-player">
      <div className="g-head">
        <h3 className="g-title">{data.title}</h3>
        <Tag kind={data.kind} />
      </div>
      {data.caption && <p className="g-intro" data-testid="clip-caption">{data.caption}</p>}
      <Legend />
      <div className="g-main">
        <div className="g-left">
          <div className="g-stage" style={{ aspectRatio: `${data.width} / ${data.height}`, width: `min(100%, calc((100vh - ${CHROME_PX}px) * ${(data.width / data.height).toFixed(4)}))` }}>
            {src && (
              <video key={src} ref={video} src={src} muted playsInline autoPlay aria-label={data.title} className="g-video"
                onError={onVideoError} onLoadedMetadata={(e) => { if (!(e.currentTarget.duration > 0)) onVideoError(); }} />
            )}
            <canvas ref={canvas} className="g-canvas" aria-label={`${data.title}, ${data.runs[run].label}`} />
            <div className="g-counters" data-testid="gate-counters">
              <span className="g-count"><b>IN</b> {counts.in}</span>
              <span className="g-count"><b>OUT</b> {counts.out}</span>
              {extra > 0 && <span className="g-extra" data-testid="extra-count">+{extra} extra (counted twice)</span>}
            </div>
            {hint && <div className="g-hint" role="status" data-testid="video-hint">{hint}</div>}
            <span className={`g-runbadge g-run-${run}`}>{data.runs[run].label}</span>
          </div>
          <div className="g-caption" data-testid="guided-caption" aria-live="polite">
            <span className="g-caption-text">{caption}</span>
            {paused && <button type="button" className="btn g-continue" onClick={resume}>Continue ▸</button>}
          </div>
          <div className="g-timeline">
            <div className="g-tally" data-testid="moment-tally">{tallyText(tl, { baseline: cap(names.baseline), ours: names.ours })}</div>
            <div className="g-rule">{MOMENT_RULE}</div>
            <input type="range" className="g-seek" min={0} max={data.duration} step={0.01} value={Math.min(t, data.duration)} onChange={(e) => seek(Number(e.target.value))} aria-label="Seek position" />
            <MomentStrip ticks={ticks} current={kIdx} onPick={(i) => seekMoment(i)} />
            <div className="g-progress muted" data-testid="moment-progress">
              {data.moments.length === 0 ? "No number changes on this clip" : kIdx != null ? `${kIdx + 1} of ${data.moments.length}` : `0 of ${data.moments.length}`}
              {data.moments.length > 0 && <span className="g-keys"> · ← / → step between moments</span>}
            </div>
          </div>
          <div className="g-controls">
            <button type="button" className="btn" onClick={() => (paused ? resume() : setUserPaused((p) => !p))} aria-label={stopped ? "Play" : "Pause"}>{stopped ? "▶ Play" : "❚❚ Pause"}</button>
            <span className="mono muted">{t.toFixed(1)} / {data.duration.toFixed(1)} s{guided && rateAt(stops, t, true) < 1 ? " · 0.25x" : ""}</span>
            <div className="toggle" role="group" aria-label="Mode">
              <button type="button" className={`btn${guided ? " on-ours" : ""}`} aria-pressed={guided} onClick={() => setGuided(true)}>Guided</button>
              <button type="button" className={`btn${!guided ? " on-ours" : ""}`} aria-pressed={!guided} onClick={() => { setGuided(false); setPaused(null); }}>Free play</button>
            </div>
            <div className="toggle" role="group" aria-label="Tracker">
              {(["baseline", "ours"] as const).map((k) => (
                <button key={k} type="button" className={`btn${run === k ? " on-" + k : ""}`} aria-pressed={run === k} disabled={!differ && k !== run} onClick={() => setRun(k)}>
                  {k === "baseline" ? `${cap(names.baseline)} (baseline)` : cap(names.ours)}
                </button>
              ))}
            </div>
          </div>
          {paused?.kind === "answer" && m && (
            <div className="g-stills" data-testid="guided-stills">
              {(["baseline", "ours"] as const).map((r) => (
                <figure key={r} className={`g-still g-run-${r}`}>
                  <canvas ref={r === "baseline" ? stillB : stillO} />
                  <figcaption><b>{cap(names[r])}</b>: {answerFor(m, r)}</figcaption>
                </figure>
              ))}
              <p className="g-stills-note">{data.kind === "illustrative" ? "No ground truth on this clip: the trackers disagree, we don't know which is right." : "Checked against human labels."}</p>
            </div>
          )}
        </div>
        <aside className="g-log" aria-label="What's happening">
          <h3>What&apos;s happening <small className="muted">({names[run]})</small></h3>
          <ol ref={logRef} data-testid="event-log">
            {log.length === 0 && <li className="muted">Nothing yet.</li>}
            {log.map((l, i) => {
              const past = f != null && l.f <= f;
              const current = past && (i === log.length - 1 || log[i + 1].f > (f ?? 0));
              return (
                <li key={`${l.f}-${i}`} className={`g-log-${l.kind}${past ? " past" : ""}${l.moment != null ? " key" : ""}`} data-current={current}>
                  <button type="button" onClick={() => seek(Math.max(0, l.t - 1))}>
                    <span className="mono">{l.t.toFixed(1)}s</span> {l.text}{l.moment != null && data.kind === "illustrative" ? " The two trackers disagree here." : ""}
                  </button>
                </li>
              );
            })}
          </ol>
        </aside>
      </div>
    </div>
  );
}

export function MomentStrip({ ticks, current, onPick }: { ticks: Tick[]; current: number | null; onPick: (i: number) => void }) {
  return (
    <div className="g-strip" role="group" aria-label="Moments where a number changed" data-testid="moment-strip">
      {ticks.map((k) => (
        <button
          key={k.index}
          type="button"
          className={`g-tick${current === k.index ? " on" : ""}`}
          style={{ left: `${k.x * 100}%` }}
          onClick={() => onPick(k.index)}
          title={k.label}
          aria-label={k.label}
          data-testid="moment-tick"
          data-top={k.top}
          data-bottom={k.bottom}
        >
          <span className={`g-tick-top${k.top ? " filled" : ""}`} />
          <span className={`g-tick-bot${k.bottom ? " filled" : ""}`} />
        </button>
      ))}
    </div>
  );
}
