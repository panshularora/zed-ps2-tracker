// Studio step 4 · "Your video": upload a clip, draw the gate on its (face-blurred) first frame, run
// our tracker on the CPU backend, then watch the result in the Studio player. Uploads have no
// ground truth, so the page reports counts, IDs and events only, never accuracy.
import { useMemo, useRef, useState } from "react";
import { api, ApiError, type Clip, type Job, type JobResult } from "../../api";
import { IdentityPlayer } from "../../identity/IdentityPlayer";
import { fromJobResult, gateFromClicks, NO_GT_LABEL, REID_OFF_LABEL, unsureMoments, uploadSummary } from "../../upload/fromJob";

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
  const [seek, setSeek] = useState<{ f: number; n: number } | null>(null);
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
      {phase === "done" && result && job && <UploadResult result={result} jobId={job.id} seek={seek} onSeek={(f) => setSeek((s) => ({ f, n: (s?.n ?? 0) + 1 }))} onAgain={() => { setPhase("pick"); setClip(null); }} />}
    </section>
  );
}

export function UploadResult({ result, jobId, seek, onSeek, onAgain }: { result: JobResult; jobId: string; seek: { f: number; n: number } | null; onSeek: (f: number) => void; onAgain?: () => void }) {
  const data = useMemo(() => fromJobResult(result, `/api/clips/${encodeURIComponent(result.clip.id)}/video`), [result]);
  const unsure = useMemo(() => unsureMoments(result), [result]);
  const s = useMemo(() => uploadSummary(result), [result]);
  return (
    <div className="space-y-3" data-testid="upload-result">
      <div className="flex flex-wrap items-center gap-2">
        <span className="px-2 py-0.5 rounded-sm bg-paper border border-line text-[12px] font-semibold" data-testid="no-gt-label">{NO_GT_LABEL}</span>
        <span className="px-2 py-0.5 rounded-sm bg-paper border border-line text-[12px]" data-testid="reid-off-label">{REID_OFF_LABEL}</span>
        <span className="text-[11px] text-mute">{result.clip.title} · {result.config} · {result.detector.name} · {result.frames_tracked} frames</span>
      </div>
      <div className="flex flex-wrap gap-4 text-[13px] tabular" data-testid="upload-counts">
        <span>Track IDs <b>{s.ids}</b></span>
        <span>IN <b>{s.in}</b></span>
        <span>OUT <b>{s.out}</b></span>
        <span>Unsure crossings <b>{s.unsure}</b></span>
        <span title="Heuristic: a new number started next to where another vanished. Not checked against labels.">Possible ID breaks <b>{s.possibleBreaks}</b></span>
      </div>
      <IdentityPlayer data={data} verify={null} seek={seek} unchecked
        watch="Each box and number is what our tracker drew. Click a box to follow a number; the bars below show when each number was on screen." />
      <div className="space-y-1" data-testid="unsure-list">
        <div className="text-[12px] font-semibold text-ink">Unsure moments ({unsure.length})</div>
        {unsure.length === 0 && <div className="text-[12px] text-mute">No crossing where the tracker was unsure (ambiguous_hold rule: visibility under 0.5 or score under 0.5 in the half-second before the line).</div>}
        {unsure.map((m) => (
          <button key={`${m.f}:${m.id}`} type="button" onClick={() => onSeek(m.f)} data-testid="unsure-item"
            className="block text-left text-[12px] font-mono px-2 py-1 rounded-sm border border-line bg-paper hover:bg-raised">{m.text}</button>
        ))}
      </div>
      <div className="flex flex-wrap gap-2 text-[12px]">
        <a className="px-3 py-1.5 rounded-sm border border-line" href={api.csvUrl(jobId)} download data-testid="dl-csv">Download CSV (boxes)</a>
        <a className="px-3 py-1.5 rounded-sm border border-line" href={api.annotatedUrl(jobId)} download data-testid="dl-annotated">Download annotated clip</a>
        <a className="px-3 py-1.5 rounded-sm border border-line" href={api.eventsUrl(jobId)} download>Events (JSON)</a>
        {onAgain && <button type="button" className="px-3 py-1.5 rounded-sm border border-line" onClick={onAgain}>Try another clip</button>}
      </div>
    </div>
  );
}
