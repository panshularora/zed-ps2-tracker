// Studio step 4 · "Your video": upload a clip, draw the gate on its (face-blurred) first frame, run
// our tracker on the CPU backend, then watch the result in the Studio player. Uploads have no
// ground truth, so the page reports counts, IDs and events only, never accuracy.
import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type Clip, type Job, type JobResult } from "../../api";
import { IdentityPlayer } from "../../identity/IdentityPlayer";
import { adjustedCounts, boxAt, cropRect, crossingsCsv, frameTime, fromJobResult, gateFromClicks, mmss, NO_GT_LABEL, rankSuggestions, REID_OFF_LABEL, rowSeek, suggestions, timeSeek, unsureMoments, uploadSummary, visibleSuggestions, type Decision, type Decisions, type SeekReq, type XYWH } from "../../upload/fromJob";
import { useThumbs, type Grabber, type ThumbReq } from "../../upload/thumbs";
import { neutralColor } from "../../lib/scenarios";

type Phase = "pick" | "uploading" | "gate" | "running" | "done";
type Pt = [number, number];

const DEFAULT_GATE: [number, number, number, number] = [0, 0.6, 1, 0.6];
const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

function errText(e: unknown): string {
  return e instanceof ApiError ? e.message : String(e);
}

export function YourVideo() {
  const [phase, setPhase] = useState<Phase>("pick");
  const [consent, setConsent] = useState(false);
  const [clip, setClip] = useState<Clip | null>(null);
  const [pts, setPts] = useState<Pt[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seek, setSeek] = useState<SeekReq | null>(null);
  const img = useRef<HTMLImageElement>(null);

  async function onFile(f: File | undefined) {
    if (!f) return;
    setError(null); setPhase("uploading"); setPts([]); setResult(null); setJob(null);
    try {
      const c = await api.upload(f, consent);
      setClip(c); setPhase("gate");
    } catch (e) {
      setError(errText(e)); setPhase("pick");
    }
  }

  function onImgClick(e: React.MouseEvent<HTMLImageElement>) {
    const el = e.currentTarget, r = el.getBoundingClientRect();
    const p: Pt = [((e.clientX - r.left) / r.width) * el.naturalWidth, ((e.clientY - r.top) / r.height) * el.naturalHeight];
    setPts((cur) => (cur.length >= 2 ? [p] : [...cur, p]));
  }

  const gate = useMemo<[number, number, number, number] | null>(() => {
    const el = img.current;
    if (pts.length < 2 || !el || !el.naturalWidth) return null;
    return gateFromClicks(pts[0], pts[1], el.naturalWidth, el.naturalHeight);
  }, [pts]);

  async function run(g: [number, number, number, number]) {
    if (!clip) return;
    setError(null); setPhase("running");
    try {
      let j = await api.createJob({ clip_id: clip.id, gate: g });
      setJob(j);
      while (!TERMINAL.has(j.status)) {
        await new Promise((r) => setTimeout(r, 800));
        j = await api.job(j.id);
        setJob(j);
      }
      if (j.status !== "succeeded") throw new Error(j.error?.message ?? `job ${j.status}`);
      setResult(await api.result(j.id));
      setPhase("done");
    } catch (e) {
      setError(errText(e)); setPhase("gate");
    }
  }

  return (
    <section className="bg-raised border border-line rounded-sm p-4 space-y-3" data-testid="your-video">
      {(phase === "pick" || phase === "uploading") && (
        <>
          <label
            className="block rounded-sm border border-dashed border-line bg-paper px-4 py-10 text-center cursor-pointer text-[13px] text-mute"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); void onFile(e.dataTransfer.files[0]); }}
          >
            {phase === "uploading" ? "Uploading…" : "Drop a short MP4 / MOV / WebM here, or click to choose"}
            <input type="file" accept="video/*" className="hidden" data-testid="your-video-input" onChange={(e) => void onFile(e.target.files?.[0])} />
          </label>
          <label className="flex items-center gap-2 text-[12px] text-mute">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            I filmed this at my own gate and the people in it agreed (turns on tailgate / hold alerts)
          </label>
          <p className="text-[11px] text-mute">Faces are blurred before anything is shown or tracked. The original is never served back.</p>
        </>
      )}
      {(phase === "gate" || phase === "running") && clip && (
        <div className="space-y-2">
          <div className="text-[13px] text-ink font-semibold">Click two points on the first frame to draw your gate line</div>
          <div className="text-[11px] text-mute">IN = crossing to the right-hand side of the line as you look from your first click to your second (arrow).</div>
          <div className="relative inline-block max-w-full" data-testid="gate-editor">
            <img ref={img} src={api.firstFrameUrl(clip.id)} alt="First frame of your clip (faces blurred)" onClick={onImgClick}
              className="max-w-full max-h-[420px] cursor-crosshair border border-line" data-testid="first-frame" onLoad={() => setPts((p) => [...p])} />
            {img.current && pts.length > 0 && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${img.current.naturalWidth} ${img.current.naturalHeight}`} preserveAspectRatio="none">
                {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={Math.max(4, img.current!.naturalWidth / 160)} fill="#F5A524" />)}
                {pts.length === 2 && (() => {
                  const [a, b] = pts, mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
                  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1, k = Math.min(60, L / 4);
                  return (
                    <g stroke="#F5A524" strokeWidth={Math.max(2, img.current!.naturalWidth / 400)}>
                      <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} strokeDasharray="12 6" />
                      <line x1={mx} y1={my} x2={mx - (dy / L) * k} y2={my + (dx / L) * k} />
                      <text x={mx - (dy / L) * k * 1.3} y={my + (dx / L) * k * 1.3} fill="#F5A524" stroke="none" fontSize={Math.max(14, img.current!.naturalWidth / 50)}>IN</text>
                    </g>
                  );
                })()}
              </svg>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" disabled={!gate || phase === "running"} onClick={() => gate && void run(gate)} data-testid="run-with-gate"
              className="px-3 py-2 rounded-sm bg-signal text-paper font-semibold text-[12px] disabled:opacity-50">
              {phase === "running" ? "Running our tracker…" : "Run our tracker with this gate"}
            </button>
            <button type="button" disabled={phase === "running"} onClick={() => setPts([])} className="px-3 py-2 rounded-sm border border-line text-[12px]">Clear line</button>
            <button type="button" disabled={phase === "running"} onClick={() => void run(DEFAULT_GATE)} className="px-3 py-2 rounded-sm border border-line text-[12px]">Skip: use a level line at 60% height</button>
            {gate && <span className="font-mono text-[11px] text-mute">gate {gate.join(", ")}</span>}
          </div>
          {phase === "running" && job && (
            <div className="text-[12px] text-mute" data-testid="job-progress">
              {job.stage} · {Math.round(job.progress * 100)}%{job.frames_total ? ` · frame ${job.frames_done ?? 0}/${job.frames_total}` : ""}
            </div>
          )}
        </div>
      )}
      {error && <div className="text-fail text-[12px]" role="alert">{error}</div>}
      {phase === "done" && result && job && <UploadResult result={result} jobId={job.id} seek={seek} onSeek={(f, mode) => setSeek((s) => (mode === "row" ? rowSeek : timeSeek)(f, (s?.n ?? 0) + 1))} onAgain={() => { setPhase("pick"); setClip(null); }} />}
    </section>
  );
}

export type SeekMode = "row" | "time";
const TOP_N = 3;
const TH_W = 64, TH_H = 36, CROP_W = 24, CROP_H = 36;

export function UploadResult({ result, jobId, seek, onSeek, onAgain, grab }: {
  result: JobResult; jobId: string; seek: SeekReq | null; onSeek: (f: number, mode: SeekMode) => void; onAgain?: () => void;
  /** frame grabber for row stills (tests inject one; the default reads the blurred clip) */
  grab?: Grabber;
}) {
  const videoUrl = `/api/clips/${encodeURIComponent(result.clip.id)}/video`;
  const data = useMemo(() => fromJobResult(result, videoUrl), [result, videoUrl]);
  const unsure = useMemo(() => unsureMoments(result), [result]);
  const s = useMemo(() => uploadSummary(result), [result]);
  const sugs = useMemo(() => rankSuggestions(suggestions(result)), [result]);
  const [dec, setDec] = useDecisions(jobId);
  const [more, setMore] = useState(false);
  const shown = visibleSuggestions(sugs, dec);
  const listed = more ? shown : shown.slice(0, TOP_N);
  const adj = adjustedCounts(s, sugs, dec);
  const decide = (k: string, d: Decision | null) => setDec((x) => { const n = { ...x }; if (d) n[k] = d; else delete n[k]; return n; });
  const csvHref = useMemo(() => (typeof URL !== "undefined" && URL.createObjectURL ? URL.createObjectURL(new Blob([crossingsCsv(result, dec)], { type: "text/csv" })) : "#"), [result, dec]);
  useEffect(() => () => { if (csvHref !== "#") URL.revokeObjectURL(csvHref); }, [csvHref]);
  const W = result.video.width, H = result.video.height;
  const reqs = useMemo<ThumbReq[]>(() => {
    const out: ThumbReq[] = [];
    for (const x of sugs) {
      const rects = [{ r: [0, 0, W, H] as XYWH, w: TH_W, h: TH_H }];
      out.push({ key: `s:${x.key}`, t: x.t, srcW: W, srcH: H, rects });
      if (x.fromBox) out.push({ key: `a:${x.key}`, t: frameTime(result, x.fromBox.f), srcW: W, srcH: H, rects: [{ r: cropRect(x.fromBox.box, W, H), w: CROP_W, h: CROP_H }] });
      if (x.toBox) out.push({ key: `b:${x.key}`, t: frameTime(result, x.toBox.f), srcW: W, srcH: H, rects: [{ r: cropRect(x.toBox.box, W, H), w: CROP_W, h: CROP_H }] });
    }
    for (const m of unsure) {
      const b = boxAt(result, m.id, m.f);
      out.push({ key: `u:${m.f}:${m.id}`, t: m.t, srcW: W, srcH: H, rects: [{ r: [0, 0, W, H], w: TH_W, h: TH_H }, ...(b ? [{ r: cropRect(b, W, H), w: CROP_W, h: CROP_H }] : [])] });
    }
    return out;
  }, [sugs, unsure, result, W, H]);
  const thumbs = useThumbs(videoUrl, reqs, grab);
  return (
    <div className="space-y-4" data-testid="upload-result">
      <div className="flex flex-wrap items-center gap-2">
        <span className="px-2 py-0.5 rounded-sm bg-paper border border-line text-[12px] font-semibold" data-testid="no-gt-label">{NO_GT_LABEL}</span>
        <span className="px-2 py-0.5 rounded-sm bg-paper border border-line text-[12px]" data-testid="reid-off-label">{REID_OFF_LABEL}</span>
        <span className="text-[12px] text-mute">{result.clip.title} · {result.frames_tracked} frames · {s.ids} track IDs</span>
      </div>
      <div className="flex flex-wrap gap-4">
        <CountBlock testid="upload-counts" label="Tracker count" inN={s.in} outN={s.out} note="What the tracker counted on its own." />
        <CountBlock testid="adjusted-counts" label="Operator-adjusted" inN={adj.in} outN={adj.out}
          note={`Tracker count + ${adj.confirmed} crossing${adj.confirmed === 1 ? "" : "s"} you confirmed.`} />
      </div>
      <IdentityPlayer data={data} verify={null} seek={seek} unchecked
        watch="Each box and number is what our tracker drew. Click a box to follow a number; the bars below show when each number was on screen." />
      <section className="space-y-2" data-testid="missed-list">
        <h3 className="text-[14px] font-semibold text-ink">Possible missed crossings ({shown.length})</h3>
        <p className="text-[12px] text-mute">One number ended at the line and a new one began just across it. Highest pairing score first (quick, on-the-spot hand-offs). Not counted unless you confirm.</p>
        {shown.length === 0 && <div className="text-[12px] text-mute">None.</div>}
        {listed.map((x) => {
          const d = dec[x.key];
          return (
            <Row key={x.key} testid="suggestion-item" decision={d ?? "pending"} done={d === "confirmed"} onPlay={() => onSeek(x.f, "row")}
              thumb={thumbs[`s:${x.key}`]?.[0]} t={x.t} onTime={() => onSeek(x.f, "time")}
              ids={[x.from, x.to]} crops={[thumbs[`a:${x.key}`]?.[0], thumbs[`b:${x.key}`]?.[0]]}
              detail={`${x.dir.toUpperCase()}${x.score !== null ? ` · pairing score ${x.score.toFixed(2)}` : ""}`}
              actions={d === "confirmed" ? (
                <>
                  <span className="text-[13px] text-ink" aria-label="Confirmed" data-testid="suggestion-confirmed">✓</span>
                  <button type="button" className="h-7 px-2 rounded-sm text-[12px] text-mute hover:text-ink" onClick={() => decide(x.key, null)}>Undo</button>
                </>
              ) : (
                <>
                  <button type="button" className="h-7 px-3 rounded-sm border border-[#56B4E9] text-[#56B4E9] text-[12px] font-semibold" onClick={() => decide(x.key, "confirmed")} data-testid="suggestion-confirm">Confirm</button>
                  <button type="button" className="h-7 px-3 rounded-sm border border-line bg-raised text-mute text-[12px]" onClick={() => decide(x.key, "dismissed")} data-testid="suggestion-dismiss">Dismiss</button>
                </>
              )} />
          );
        })}
        {shown.length > TOP_N && (
          <button type="button" className="h-7 px-3 rounded-sm border border-line text-[12px]" onClick={() => setMore((m) => !m)} aria-expanded={more} data-testid="suggestions-more">
            {more ? "Show top 3 only" : `Show ${shown.length - TOP_N} more`}
          </button>
        )}
      </section>
      <section className="space-y-2" data-testid="unsure-list">
        <h3 className="text-[14px] font-semibold text-ink">Unsure crossings ({unsure.length})</h3>
        <p className="text-[12px] text-mute">Counted crossings where the tracker was unsure (visibility or score under 0.5 in the half-second before the line). Already in the tracker count; worth a look.</p>
        {unsure.length === 0 && <div className="text-[12px] text-mute">None.</div>}
        {unsure.map((m) => {
          const k = `u:${m.f}:${m.id}`;
          return (
            <Row key={k} testid="unsure-item" onPlay={() => onSeek(m.f, "row")} thumb={thumbs[k]?.[0]} t={m.t} onTime={() => onSeek(m.f, "time")}
              ids={[m.id]} crops={[thumbs[k]?.[1]]} detail={m.text.replace(/^#\d+ /, "")} />
          );
        })}
      </section>
      <div className="flex flex-wrap gap-2 text-[12px] pt-2 border-t border-line" data-testid="downloads">
        <a className="h-7 inline-flex items-center px-2 rounded-sm border border-line" href={api.csvUrl(jobId)} download data-testid="dl-csv">Boxes CSV</a>
        <a className="h-7 inline-flex items-center px-2 rounded-sm border border-line" href={csvHref} download={`${result.clip.title}-crossings.csv`} data-testid="dl-crossings-csv">Crossings + your decisions CSV</a>
        <a className="h-7 inline-flex items-center px-2 rounded-sm border border-line" href={api.annotatedUrl(jobId)} download data-testid="dl-annotated">Annotated clip</a>
        <a className="h-7 inline-flex items-center px-2 rounded-sm border border-line" href={api.eventsUrl(jobId)} download>Events JSON</a>
        {onAgain && <button type="button" className="h-7 px-2 rounded-sm border border-line" onClick={onAgain}>Try another clip</button>}
      </div>
    </div>
  );
}

function CountBlock({ testid, label, inN, outN, note }: { testid: string; label: string; inN: number; outN: number; note: string }) {
  return (
    <div className="flex-1 min-w-[240px] px-4 py-3 rounded-sm border border-line bg-paper" data-testid={testid}>
      <div className="text-[12px] text-mute">{label}</div>
      <div className="text-[32px] leading-10 font-semibold tabular text-ink">
        <span className="text-[14px] font-normal text-mute mr-1">IN</span><b data-testid={`${testid}-in`}>{inN}</b>
        <span className="text-mute mx-2">·</span>
        <span className="text-[14px] font-normal text-mute mr-1">OUT</span><b data-testid={`${testid}-out`}>{outN}</b>
      </div>
      <div className="text-[12px] text-mute">{note}</div>
    </div>
  );
}

function IdChip({ id }: { id: number }) {
  return <span className="inline-flex items-center h-5 px-1.5 rounded-sm text-[12px] font-semibold text-[#0E1013] tabular" style={{ background: neutralColor(id) }} data-testid="id-chip">#{id}</span>;
}

function Row({ testid, decision, done, onPlay, thumb, t, onTime, ids, crops, detail, actions }: {
  testid: string; decision?: string; done?: boolean; onPlay: () => void; thumb?: string; t: number; onTime: () => void;
  ids: number[]; crops: (string | undefined)[]; detail: string; actions?: React.ReactNode;
}) {
  return (
    <div className={`flex items-center gap-2 h-12 px-2 rounded-sm border border-line bg-paper cursor-pointer hover:bg-raised ${done ? "opacity-60" : ""}`}
      data-testid={testid} data-decision={decision} onClick={onPlay} role="button" tabIndex={0} aria-label={`Play 2 s around ${mmss(t)}`}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPlay(); } }}>
      {thumb ? <img src={thumb} alt="" width={TH_W} height={TH_H} className="rounded-sm" data-testid="row-thumb" />
        : <span className="inline-block rounded-sm bg-raised" style={{ width: TH_W, height: TH_H }} data-testid="row-thumb-pending" />}
      <button type="button" className="text-[13px] tabular text-[#56B4E9] hover:underline" data-testid="row-time"
        onClick={(e) => { e.stopPropagation(); onTime(); }} title="Jump to 1 s before">{mmss(t)}</button>
      <span className="inline-flex items-center gap-1">
        {ids.map((id, i) => (
          <span key={i} className="inline-flex items-center gap-1">
            {i > 0 && <span className="text-mute">→</span>}
            <IdChip id={id} />
          </span>
        ))}
      </span>
      <span className="inline-flex items-center gap-0.5" data-testid="row-crops">
        {crops.map((c, i) => c ? <img key={i} src={c} alt={`#${ids[i]}`} width={CROP_W} height={CROP_H} className="rounded-sm" />
          : <span key={i} className="inline-block rounded-sm bg-raised" style={{ width: CROP_W, height: CROP_H }} />)}
      </span>
      <span className="text-[12px] text-mute truncate">{detail}</span>
      <span className="flex-1" />
      {actions && <span className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>{actions}</span>}
    </div>
  );
}

/** Operator decisions on suggestions, kept on the client (localStorage per job). */
function useDecisions(jobId: string): [Decisions, (f: (d: Decisions) => Decisions) => void] {
  const key = `zed.decisions.${jobId}`;
  const [d, setD] = useState<Decisions>(() => {
    try { return JSON.parse(window.localStorage.getItem(key) ?? "{}") as Decisions; } catch { return {}; }
  });
  const update = (f: (d: Decisions) => Decisions) => setD((x) => {
    const n = f(x);
    try { window.localStorage.setItem(key, JSON.stringify(n)); } catch { /* storage off: state stays in memory */ }
    return n;
  });
  return [d, update];
}
