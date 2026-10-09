import { useState } from "react";
import { api, type JobResult } from "../api";
import { num } from "../lib/format";
import { useAsync } from "../lib/useAsync";

// Real outputs of ours/visibility on MOT17 val_half public FRCNN boxes, copied from
// docs/integration.md section 2 (the ZIG event contract, serve/events.py).
export const CONTRACT_EXAMPLES = [
  { type: "crossing", track_id: 5, direction: "in", timestamp: "2026-10-05T09:00:07.233+05:30", confidence: 1.0, frame: 218, gate: "h60" },
  { type: "tailgate", track_id: 41, direction: "out", timestamp: "2026-10-05T09:00:13.066+05:30", confidence: 1.0, frame: 393, gate: "h60", lead_track_id: 39, gap_s: 0.233 },
  { type: "ambiguous_hold", track_id: 7, direction: "out", timestamp: "2026-10-05T09:00:06.233+05:30", confidence: 0.093, frame: 188, gate: "h60", reasons: ["occlusion"] },
];

export interface Stage {
  id: string;
  title: string;
  what: string;
  metric: string | null;
  payload: unknown;
  source: string;
}

export function buildStages(r: JobResult | null, jobId: string | null): Stage[] {
  const none = "Run a job on the Demo page to fill this payload from a real run.";
  const det = r?.sample_detections?.[0];
  const mid = r ? Math.max(1, Math.round(r.frames_tracked / 2)) : 0;
  const tracksAt = r
    ? Object.entries(r.tracks)
        .map(([id, rows]) => ({ id: Number(id), row: rows.find((x) => x[0] === mid) }))
        .filter((x) => x.row)
        .slice(0, 4)
        .map(({ id, row }) => ({ id, tlwh: row!.slice(1, 5), score: row![5], visibility: row![6] }))
    : [];
  const crossing = r?.events.find((e) => e.type === "crossing");
  const recovery = r?.events.find((e) => e.type === "recovery");
  const src = jobId ? `job ${jobId}, ${r?.clip.title}` : "";
  return [
    {
      id: "camera",
      title: "Gate camera",
      what: "Frames stay on the edge box; nothing is uploaded or stored.",
      metric: r ? `${r.video.width}×${r.video.height} · ${num(r.video.fps, 2)} fps` : null,
      payload: r ? { clip: r.clip.id, ...r.video } : none,
      source: r ? `${src}: video metadata` : "",
    },
    {
      id: "detector",
      title: "Person detector",
      what: "YOLOX on CPU here (ONNX); YOLOX-X on GPU in the benchmark plan. Persons only, score ≥ 0.1.",
      metric: r ? `p50 ${num(r.timing.detector_p50_ms, 0)} ms / frame` : null,
      payload: det ? { frame: det.frame, detector: r!.detector.name, boxes_tlwh: det.tlwh.slice(0, 4), scores: det.score.slice(0, 4), n: det.score.length } : none,
      source: det ? `${src}: raw detections of frame ${det.frame} (first 4)` : "",
    },
    {
      id: "tracker",
      title: "Tracker",
      what: "BoT-SORT-style online association + box-geometry visibility; occluded tracks are kept longer and re-associated.",
      metric: r ? `p95 ${num(r.timing.tracker_p95_ms, 2)} ms / frame` : null,
      payload: r ? { frame: mid, config: r.config, tracks: tracksAt } : none,
      source: r ? `${src}: tracker output at tracked frame ${mid}` : "",
    },
    {
      id: "gate",
      title: "Gate logic",
      what: "A virtual line; a crossing fires when the box's bottom-centre changes side by more than the hysteresis.",
      metric: r ? `${r.metrics.crossings_in} in · ${r.metrics.crossings_out} out` : null,
      payload: r ? { gate: r.gate, first_crossing: crossing ?? null, first_recovery: recovery ?? null } : none,
      source: r ? `${src}: gate config and first events` : "",
    },
    {
      id: "events",
      title: "Event API",
      what: "Small JSON events to ZIG: crossing, tailgate, and ambiguous_hold (\"tracker unsure, don't guess\").",
      metric: "3 event types",
      payload: CONTRACT_EXAMPLES,
      source: "docs/integration.md §2: real ours/visibility outputs on MOT17 val_half (serve/events.py)",
    },
  ];
}

export function Diagram({ stages, active, onSelect }: { stages: Stage[]; active: string; onSelect: (id: string) => void }) {
  return (
    <div className="diagram" role="tablist" aria-label="Pipeline">
      {stages.map((s, i) => (
        <div key={s.id} style={{ display: "contents" }}>
          {i > 0 && (
            <div className="edge" aria-hidden>
              <svg viewBox="0 0 36 12">
                <line x1="0" y1="6" x2="34" y2="6" stroke="#8A929C" strokeWidth="1" />
                <polyline points="29,2 34,6 29,10" fill="none" stroke="#8A929C" strokeWidth="1" />
              </svg>
            </div>
          )}
          <button role="tab" aria-selected={active === s.id} className={`node${active === s.id ? " active" : ""}`} onClick={() => onSelect(s.id)}>
            <div className="t">{s.title}</div>
            <div className="d">{s.what}</div>
            {s.metric && <div className="m">{s.metric}</div>}
          </button>
        </div>
      ))}
    </div>
  );
}

async function loadLatest(): Promise<{ result: JobResult | null; jobId: string | null }> {
  const jobs = await api.jobs().catch(() => []);
  const done = jobs.find((j) => j.status === "succeeded");
  if (!done) return { result: null, jobId: null };
  return { result: await api.result(done.id), jobId: done.id };
}

export default function Architecture() {
  const { data } = useAsync(loadLatest);
  const [active, setActive] = useState("tracker");
  const stages = buildStages(data?.result ?? null, data?.jobId ?? null);
  const sel = stages.find((s) => s.id === active)!;
  return (
    <div>
      <h1>Architecture</h1>
      <p className="muted" style={{ marginTop: 6, maxWidth: 820 }}>
        One edge box per gate group turns video into a few JSON events per passenger. Click a stage to see its payload; payloads come
        from the latest job on this server, or from the repo where named.
      </p>
      <div className="section" style={{ marginTop: 24 }}>
        <Diagram stages={stages} active={active} onSelect={setActive} />
        <div className="panel payload" data-testid="payload">
          <div className="payload-head">
            <h3>{sel.title}: example payload</h3>
            <span className="muted" style={{ fontSize: 12 }}>{sel.source}</span>
          </div>
          <pre className="mono">{typeof sel.payload === "string" ? sel.payload : JSON.stringify(sel.payload, null, 2)}</pre>
        </div>
      </div>
      <section className="section cols3">
        <div className="panel">
          <h3>Stage A · GPU, once</h3>
          <p>Benchmark detections and re-ID embeddings are computed once on a DGX and cached, so every tracker sees identical boxes.</p>
        </div>
        <div className="panel">
          <h3>Stage B · CPU, many times</h3>
          <p>Tracking, TrackEval scoring, stress bins and ablations run on CPU from the cache. This demo runs a small CPU detector instead.</p>
        </div>
        <div className="panel">
          <h3>Privacy</h3>
          <p>No face recognition and no identity database. Demo footage is face-blurred before it is shown; embeddings, when used, live only as long as the track.</p>
        </div>
      </section>
    </div>
  );
}
