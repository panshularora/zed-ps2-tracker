import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type Clip, type Job, type JobEvent, type JobResult } from "../api";
import { drawOverlay } from "../lib/draw";
import { int, num, seconds } from "../lib/format";
import {
  activeRings,
  describe,
  eventTrackId,
  fmtClock,
  frameAtCurrentTime,
  frameAtMediaTime,
  gateCounts,
  gatePulse,
  indexTracks,
  seekTimeForTracked,
  tickPosition,
  timeOfTracked,
  trackedFrame,
  visibleEvents,
} from "../lib/playback";
import { idColor, theme } from "../theme";

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

const DEFAULT_CLIP = "subway_turnstile";
const DETECTORS = [
  { id: "yolox_s", label: "YOLOX-S · 640" },
  { id: "yolox_tiny", label: "YOLOX-Tiny · 416" },
  { id: "yolox_nano", label: "YOLOX-Nano · 416" },
];

function stageText(job: Job): string {
  const p = `${Math.round(job.progress * 100)} %`;
  if (job.status === "queued") return "Queued";
  if (job.stage === "blur") return `Blurring faces · ${p}`;
  if (job.stage === "detect")
    return job.frames_done != null ? `Detecting · frame ${job.frames_done} / ${job.frames_total} · ${p}` : `Detecting · ${p}`;
  if (job.stage === "track") return `Tracking · ${p}`;
  return `${job.stage} · ${p}`;
}

export default function Demo() {
  const [clips, setClips] = useState<Clip[]>([]);
  const [clipId, setClipId] = useState<string>("");
  const [detector, setDetector] = useState("yolox_s");
  const [stride, setStride] = useState(1);
  const [maxSeconds, setMaxSeconds] = useState<string>("");
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [consent, setConsent] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [t, setT] = useState(0); // video time of the frame on screen
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<VideoWithRvfc>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const clip = clips.find((c) => c.id === clipId) ?? null;

  const loadClips = useCallback(async (select?: string) => {
    try {
      const list = await api.clips();
      setClips(list);
      setClipId((cur) => select ?? (cur || (list.find((c) => c.id === DEFAULT_CLIP)?.id ?? list[0]?.id ?? "")));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void loadClips();
  }, [loadClips]);

  // on clip change: show the latest finished run of that clip, if any
  useEffect(() => {
    if (!clipId) return;
    let alive = true;
    setResult(null);
    setJob(null);
    setT(0);
    if (clipId === "mexico_city_entrance") setMaxSeconds("30");
    else setMaxSeconds("");
    api
      .jobs(clipId)
      .then(async (jobs) => {
        const active = jobs.find((j) => j.status === "running" || j.status === "queued");
        const done = jobs.find((j) => j.status === "succeeded");
        if (!alive) return;
        if (active) setJob(active);
        else if (done) {
          setJob(done);
          const r = await api.result(done.id);
          if (alive) setResult(r);
        }
      })
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [clipId]);

  // poll a running job
  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") return;
    const h = window.setInterval(async () => {
      try {
        const j = await api.job(job.id);
        setJob(j);
        if (j.status === "succeeded") {
          setResult(await api.result(j.id));
          if (j.clip.id.startsWith("upload-")) void loadClips(j.clip.id);
        }
        if (j.status === "failed") setError(j.error?.message ?? "job failed");
      } catch (e) {
        setError((e as Error).message);
      }
    }, 500);
    return () => window.clearInterval(h);
  }, [job, loadClips]);

  const run = async () => {
    if (!clip) return;
    setError(null);
    try {
      const ms = maxSeconds ? Number(maxSeconds) : null;
      const j = await api.createJob({ clip_id: clip.id, detector, stride, max_seconds: ms });
      setJob(j);
      if (j.status === "succeeded") setResult(await api.result(j.id));
      else setResult(null);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (${e.code})` : (e as Error).message);
    }
  };

  const onUpload = async (file: File | undefined) => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const c = await api.upload(file, consent);
      setClips((cs) => [...cs.filter((x) => x.id !== c.id), c]);
      setClipId(c.id);
      const j = await api.createJob({ clip_id: c.id, detector, stride, max_seconds: maxSeconds ? Number(maxSeconds) : null });
      setJob(j);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (${e.code})` : (e as Error).message);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  // ------------------------------------------------------------------ playback + overlay
  const styling = result?.clip.event_styling ?? clip?.event_styling ?? "neutral";
  const idx = useMemo(() => (result ? indexTracks(result.tracks) : null), [result]);
  const events = useMemo(() => (result ? visibleEvents(result.events, styling) : []), [result, styling]);
  const recoveries = useMemo(
    () => events.filter((e): e is Extract<JobEvent, { type: "recovery" }> => e.type === "recovery"),
    [events],
  );
  const crossFrames = useMemo(() => events.filter((e) => e.type === "crossing").map((e) => e.frame), [events]);
  const fps = result?.video.fps ?? clip?.fps ?? 25;
  const stride_ = result?.video.stride ?? 1;
  const tracked = result?.frames_tracked ?? 0;
  const vFrame = frameAtCurrentTime(t, fps);
  const k = result ? trackedFrame(vFrame, stride_, tracked) : null;
  const trackedEnd = result ? timeOfTracked(tracked, stride_, fps) + stride_ / fps : 0;

  const draw = useCallback(
    (time: number, viaRvfc: boolean) => {
      const cv = canvasRef.current;
      if (!cv || !result || !idx) return;
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      const f = viaRvfc ? frameAtMediaTime(time, fps) : frameAtCurrentTime(time, fps);
      const kk = trackedFrame(f, stride_, tracked);
      const alertIds = new Set<number>();
      if (styling === "alert" && kk != null)
        for (const e of events)
          if ((e.type === "tailgate" || e.type === "ambiguous_hold") && Math.abs(e.frame - kk) <= fps / stride_) alertIds.add(e.track_id);
      drawOverlay(ctx, result, idx, {
        k: kk,
        rings: activeRings(recoveries, time, stride_, fps),
        pulse: gatePulse(crossFrames, time, stride_, fps),
        styling,
        alertIds,
      });
    },
    [result, idx, fps, stride_, tracked, styling, events, recoveries, crossFrames],
  );

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let handle: number | null = null;
    let raf: number | null = null;
    const onFrame = (_now: number, meta: { mediaTime: number }) => {
      draw(meta.mediaTime, true);
      setT(meta.mediaTime);
      handle = v.requestVideoFrameCallback!(onFrame);
    };
    const loop = () => {
      draw(v.currentTime, false);
      setT(v.currentTime);
      raf = requestAnimationFrame(loop);
    };
    const start = () => {
      setPlaying(true);
      if (typeof v.requestVideoFrameCallback === "function") handle = v.requestVideoFrameCallback(onFrame);
      else raf = requestAnimationFrame(loop);
    };
    const stop = () => {
      setPlaying(false);
      if (handle != null) v.cancelVideoFrameCallback?.(handle);
      if (raf != null) cancelAnimationFrame(raf);
      handle = raf = null;
      draw(v.currentTime, false);
      setT(v.currentTime);
    };
    const onSeeked = () => {
      draw(v.currentTime, false);
      setT(v.currentTime);
    };
    const onMeta = () => {
      setDuration(v.duration || 0);
      onSeeked();
    };
    v.addEventListener("play", start);
    v.addEventListener("pause", stop);
    v.addEventListener("ended", stop);
    v.addEventListener("seeked", onSeeked);
    v.addEventListener("loadedmetadata", onMeta);
    if (v.readyState >= 1) onMeta();
    if (!v.paused) start();
    return () => {
      v.removeEventListener("play", start);
      v.removeEventListener("pause", stop);
      v.removeEventListener("ended", stop);
      v.removeEventListener("seeked", onSeeked);
      v.removeEventListener("loadedmetadata", onMeta);
      if (handle != null) v.cancelVideoFrameCallback?.(handle);
      if (raf != null) cancelAnimationFrame(raf);
    };
  }, [draw]);

  const seek = (time: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(time, (v.duration || time) - 0.001));
  };
  const step = (dir: 1 | -1) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    const cur = frameAtCurrentTime(v.currentTime, fps);
    const target = Math.max(1, cur + dir * stride_);
    seek((target - 1 + 0.5) / fps);
  };
  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  };

  // keyboard: space = play/pause, arrows = frame step
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT" || (e.target as HTMLElement)?.tagName === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      } else if (e.code === "ArrowRight") step(1);
      else if (e.code === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const counts = gateCounts(events, k ?? (t >= trackedEnd && result ? tracked : 0));
  const pastCount = events.filter((e) => k != null && e.frame <= k).length;
  useEffect(() => {
    const el = logRef.current?.querySelector<HTMLElement>(".log-row.now");
    if (el && logRef.current) logRef.current.scrollTop = el.offsetTop - logRef.current.clientHeight / 2;
  }, [pastCount]);

  const videoUrl = result?.clip.video_url ?? clip?.video_url ?? null;
  const running = job && (job.status === "queued" || job.status === "running");
  const m = result?.metrics;
  const tm = result?.timing;

  return (
    <div>
      <div className="toolbar">
        <label className="field">
          <span>Clip (faces blurred)</span>
          <select value={clipId} onChange={(e) => setClipId(e.target.value)} aria-label="Clip" style={{ minWidth: 240 }}>
            {clips.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title} · {c.duration_s.toFixed(1)} s{c.source === "upload" ? " · upload" : ""}{c.demo_cut ? " · short cut" : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Detector (CPU)</span>
          <select value={detector} onChange={(e) => setDetector(e.target.value)} aria-label="Detector">
            {DETECTORS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Frames</span>
          <select value={stride} onChange={(e) => setStride(Number(e.target.value))} aria-label="Frame stride">
            <option value={1}>every frame</option>
            <option value={2}>every 2nd</option>
            <option value={3}>every 3rd</option>
          </select>
        </label>
        <label className="field">
          <span>Limit (s)</span>
          <input type="number" min={1} max={600} placeholder="whole clip" value={maxSeconds} onChange={(e) => setMaxSeconds(e.target.value)} style={{ width: 110 }} aria-label="Limit seconds" />
        </label>
        <button className="btn primary" onClick={run} disabled={!clip || !!running}>
          {running ? "Running…" : "Run tracking"}
        </button>
        <span style={{ flex: 1 }} />
        <label className="row muted" style={{ gap: 6, fontSize: 12 }}>
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          people in my video consented
        </label>
        <input ref={fileRef} type="file" accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.m4v,.webm,.mkv,.avi" hidden onChange={(e) => onUpload(e.target.files?.[0])} />
        <button className="btn" onClick={() => fileRef.current?.click()} disabled={uploading || !!running}>
          {uploading ? "Uploading…" : "Upload a clip"}
        </button>
      </div>

      {error && (
        <div className="error-box" role="alert" style={{ marginBottom: 12 }}>
          {error}
        </div>
      )}
      {running && job && (
        <div className="panel" style={{ padding: "10px 14px", marginBottom: 12 }} data-testid="progress">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
            <span>{stageText(job)}</span>
            <button className="btn" style={{ height: 28 }} onClick={() => api.cancel(job.id).then(setJob)}>
              Cancel
            </button>
          </div>
          <div className="progress">
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
        </div>
      )}

      <div className="demo">
        <div>
          <div className="stage">
            {videoUrl ? (
              <>
                <video ref={videoRef} src={videoUrl} muted playsInline preload="auto" onClick={toggle} />
                <canvas ref={canvasRef} width={result?.video.width ?? clip?.width ?? 1280} height={result?.video.height ?? clip?.height ?? 720} />
                <span className="badge">
                  {result ? (k != null ? `frame ${vFrame} · tracked ${k}/${tracked}` : `frame ${vFrame} · beyond tracked range`) : "no run yet"}
                </span>
              </>
            ) : (
              <div className="empty">{clip ? "The video is shown after its first job has blurred the faces." : "No clips found. In the repo folder run make clips blur, then reload."}</div>
            )}
          </div>
          <div className="controls">
            <button className="btn icon" onClick={toggle} aria-label={playing ? "Pause" : "Play"} disabled={!videoUrl}>
              {playing ? "❚❚" : "▶"}
            </button>
            <button className="btn icon" onClick={() => step(-1)} aria-label="Previous frame" disabled={!videoUrl}>
              ‹
            </button>
            <button className="btn icon" onClick={() => step(1)} aria-label="Next frame" disabled={!videoUrl}>
              ›
            </button>
            <div className="scrubber">
              <div className="track" />
              {result && duration > 0 && <div className="tracked" style={{ width: `${Math.min(100, (trackedEnd / duration) * 100)}%` }} />}
              <div className="fill" style={{ width: `${duration ? (t / duration) * 100 : 0}%` }} />
              {result &&
                duration > 0 &&
                events.map((e, i) => (
                  <span key={i} className={`tick ${e.type === "crossing" ? "cross" : e.type === "recovery" ? "recovery" : ""}`} style={{ left: `${tickPosition(e.frame, result, duration) * 100}%` }} />
                ))}
              <span className="thumb" style={{ left: `${duration ? (t / duration) * 100 : 0}%` }} />
              <input type="range" min={0} max={duration || 0} step={1 / fps} value={t} onChange={(e) => seek(Number(e.target.value))} aria-label="Scrub" />
            </div>
            <span className="time">
              {fmtClock(t)} / {fmtClock(duration)}
            </span>
          </div>
          {clips.some((c) => c.demo_cut) && (
            <p className="muted" style={{ fontSize: 12, margin: "8px 2px" }} data-testid="demo-cut-note">
              Playing the short demo cuts shipped with the app. Full-length clips are optional: <code>make clips blur</code>.
            </p>
          )}
          <div className="legend">
            <span><i style={{ background: theme.gate }} />gate line and crossings</span>
            <span><i style={{ background: theme.ours }} />ID kept after occlusion (ring on the box)</span>
            <span><i style={{ background: theme.textSecondary }} />dashed box: estimated visibility &lt; 0.5</span>
            {result && tracked < (result.video.frames + stride_ - 1) / stride_ && <span>tracked: first {seconds(trackedEnd)} (lighter bar)</span>}
          </div>

          {result && m && tm && (
            <div className="panel metrics" data-testid="metrics">
              <div className="metric"><div className="label">Tracks</div><div className="v">{int(m.tracks)}</div></div>
              <div className="metric"><div className="label">Avg track length</div><div className="v">{num(m.avg_track_length_s, 2)}<small>s</small></div></div>
              <div className="metric"><div className="label">ID-preserving recoveries</div><div className="v">{int(m.id_preserving_recoveries)}</div></div>
              <div className="metric"><div className="label">Detector p50</div><div className="v">{num(tm.detector_p50_ms, 0)}<small>ms/frame</small></div></div>
              <div className="metric"><div className="label">Tracker p95</div><div className="v">{num(tm.tracker_p95_ms, 2)}<small>ms/frame</small></div></div>
              <div className="metric"><div className="label">CPU job time</div><div className="v">{num(tm.job_s ?? tm.wall_s, 1)}<small>s · {int(m.frames)} fr</small></div></div>
            </div>
          )}
          {result && (
            <p className="caption">
              {result.detector.name} ({result.detector.weights}) → {result.config}. Behaviour on a clip without ground truth, not accuracy:
              see Results for MOT17 numbers.{" "}
              {styling === "neutral" && "Stock footage without consent: only IDs, crossings and counts are shown."}{" "}
              {job && (
                <>
                  Downloads: <a href={`/api/jobs/${job.id}/tracks.txt`}>tracks (MOT)</a> · <a href={`/api/jobs/${job.id}/events.json`}>events</a> ·{" "}
                  <a href={`/api/jobs/${job.id}/result`}>full result</a>
                </>
              )}
            </p>
          )}
        </div>

        <aside className="side">
          <div className="panel counters" data-testid="counters">
            <div className="counter">
              <div className="label">In</div>
              <div className="n">{result ? counts.in : "—"}</div>
            </div>
            <div className="counter">
              <div className="label">Out</div>
              <div className="n">{result ? counts.out : "—"}</div>
            </div>
          </div>
          <div className="panel">
            <div className="log-head">
              <span className="label">Event log</span>
              <span className="muted mono">{events.length} events</span>
            </div>
            <div className="log" ref={logRef} data-testid="event-log">
              {!result && <div className="muted" style={{ padding: "8px 12px" }}>No events yet.</div>}
              {result && events.length === 0 && <div className="muted" style={{ padding: "8px 12px" }}>No events in the tracked range.</div>}
              {events.map((e, i) => {
                const past = k != null && e.frame <= k;
                const isNow = past && (i === events.length - 1 || events[i + 1].frame > (k ?? 0));
                const color = e.type === "crossing" ? theme.gate : idColor(eventTrackId(e));
                return (
                  <div key={i} className={`log-row${past ? " past" : ""}${isNow ? " now" : ""}`} onClick={() => seek(Math.max(0, seekTimeForTracked(e.frame, stride_, fps) - 1))}>
                    <span>{fmtClock(timeOfTracked(e.frame, stride_, fps))}</span>
                    <span className="dot" style={{ background: color }} />
                    <span>{describe(e)}</span>
                  </div>
                );
              })}
            </div>
          </div>
          {clip && (
            <div className="panel" style={{ padding: "10px 12px", fontSize: 12 }}>
              <div className="label" style={{ marginBottom: 4 }}>Clip</div>
              <div>{clip.title}</div>
              <div className="muted">
                {clip.width}×{clip.height} · {num(clip.fps, 2)} fps · {clip.licence}
                {clip.page && (
                  <>
                    {" · "}
                    <a href={clip.page} target="_blank" rel="noreferrer">source</a>
                  </>
                )}
              </div>
              <div className="muted">{clip.consent ? "consented footage: alert styling on" : "no consent: neutral styling"}</div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
