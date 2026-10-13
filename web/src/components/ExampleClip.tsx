import { useEffect, useRef, useState } from "react";
import { drawBoxes, frameAt, type RunKey, type ScenarioData, timeOf, videoUrl } from "../lib/scenarios";

export const FREEZE_MS = 400;

/** The landing page's running example: the moment where ByteTrack changes a person's ID and
 *  ours keeps it. Picks the first such disagreement and a 6 s window around it. */
export function exampleWindow(data: ScenarioData, lenS = 6): { start: number; end: number; swapFrame: number | null } {
  const d = data.disagreements.find((x) => x.kept === "ours") ?? null;
  if (!d) return { start: 0, end: Math.min(lenS, data.meta.covered_s), swapFrame: null };
  const ts = timeOf(data.meta, d.frame);
  const start = Math.max(0, Math.min(ts - lenS / 2, data.meta.covered_s - lenS));
  return { start, end: Math.min(start + lenS, data.meta.covered_s), swapFrame: d.frame };
}

export function ExampleClip({ data, run }: { data: ScenarioData; run: RunKey }) {
  const { meta } = data;
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [frame, setFrame] = useState<number | null>(null);
  const [frozen, setFrozen] = useState(false);
  const win = exampleWindow(data);

  // autoplay only while visible
  useEffect(() => {
    const el = wrap.current, v = video.current;
    if (!el || !v || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) void v.play().catch(() => undefined);
      else v.pause();
    }, { threshold: 0.4 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    let raf = 0;
    let lastF: number | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = () => {
      const v = video.current, c = canvas.current;
      if (v && c) {
        if (v.currentTime < win.start - 0.05 || v.currentTime > win.end) v.currentTime = win.start;
        const f = frameAt(meta, v.currentTime);
        if (run === "baseline" && win.swapFrame != null && f != null && lastF != null && lastF < win.swapFrame && f >= win.swapFrame && !v.paused) {
          v.pause();
          setFrozen(true);
          timer = setTimeout(() => { setFrozen(false); void v.play().catch(() => undefined); }, FREEZE_MS);
        }
        lastF = f;
        setFrame(f);
        const ctx = c.getContext("2d");
        if (ctx) {
          c.width = c.clientWidth;
          c.height = c.clientHeight;
          ctx.clearRect(0, 0, c.width, c.height);
          if (f != null) drawBoxes(ctx, data.runs[run].frames[String(f)] ?? [], c.width / meta.width, c.height / meta.height);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(raf); if (timer) clearTimeout(timer); };
  }, [data, meta, run, win.start, win.end, win.swapFrame]);

  return (
    <div className="stage example" ref={wrap}>
      <video ref={video} src={`${videoUrl(meta.id)}#t=${win.start.toFixed(2)}`} muted loop playsInline preload="metadata" aria-label={`${meta.title}, ${run === "ours" ? "our tracker" : "baseline tracker"}`} />
      <canvas ref={canvas} aria-hidden />
      <span className="badge">{run === "ours" ? "Ours" : "Baseline (ByteTrack)"} · frame {frame ?? "–"}</span>
      {frozen && <span className="marker-label">Same person, new ID</span>}
    </div>
  );
}
