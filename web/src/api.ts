import type { StressMapData } from "./stress/model";
import type { SceneData } from "./stress/scene";
import type { BreakSpec } from "./stress/breakSpec";
// Typed client for the FastAPI service (serve/api.py). Every number the UI shows comes from here.

export type Styling = "neutral" | "alert";

export interface Clip {
  id: string;
  title: string;
  source: "library" | "upload";
  note: string;
  page: string | null;
  licence: string;
  consent: boolean;
  event_styling: Styling;
  blurred: boolean;
  gate: [number, number, number, number] | null;
  width: number;
  height: number;
  fps: number;
  frames: number;
  duration_s: number;
  video_url: string | null;
  poster_url: string | null;
  /** true when the server is serving the committed short demo cut instead of the full blurred clip */
  demo_cut?: boolean;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface JobMetrics {
  frames: number;
  tracks: number;
  avg_track_length_frames: number | null;
  avg_track_length_s: number | null;
  median_track_length_s: number | null;
  boxes: number;
  id_preserving_recoveries: number;
  recoveries_after_lost: number;
  recoveries_after_occluded: number;
  suspected_id_switches: number;
  crossings_in: number;
  crossings_out: number;
  tailgates: number;
  ambiguous_holds: number;
  /** suggestions for an operator to confirm, never part of crossings_in / crossings_out */
  possible_missed_crossings?: number;
}

export interface JobTiming {
  decode_s: number;
  detect_s: number;
  detector_p50_ms: number;
  detector_p95_ms: number;
  tracker_p50_ms: number;
  tracker_p95_ms: number;
  tracker_total_s: number;
  wall_s: number;
  per_frame_ms: number;
  machine: string;
  job_s?: number;
}

export interface JobOptions {
  config: string;
  detector: string;
  stride: number;
  max_frames: number | null;
  gate: [number, number, number, number] | null;
  event_styling: Styling;
  hysteresis_px: number;
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  clip: { id: string; title: string; event_styling: Styling; consent: boolean; fps: number; frames: number };
  options: JobOptions;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: { code: string; message: string; cause?: string; fix?: string } | null;
  metrics: JobMetrics | null;
  timing: JobTiming | null;
  frames_done?: number | null;
  frames_total?: number | null;
}

/** [frame, x, y, w, h, score, visibility]; frame = tracked frame (1-based) */
export type TrackRow = [number, number, number, number, number, number, number];

export type JobEvent =
  | { kind: "gate"; type: "crossing"; frame: number; track_id: number; direction: "in" | "out"; timestamp: string; gate: string; confidence?: number; unsure?: boolean; reasons?: string[] }
  | { kind: "gate"; type: "tailgate"; frame: number; track_id: number; lead_track_id: number; direction: "in" | "out"; timestamp: string; gap_s: number; confidence: number }
  | { kind: "gate"; type: "recovered_crossing"; frame: number; track_id: number; from_track_id: number; direction: "in" | "out"; timestamp: string; recovered: true; gap_s?: number; distance_rel?: number; pair_score?: number }
  | { kind: "gate"; type: "ambiguous_hold"; frame: number; track_id: number; direction: "in" | "out"; timestamp: string; reasons: string[]; confidence: number }
  | { kind: "recovery"; type: "recovery"; frame: number; track_id: number; reason: "lost" | "occluded"; gap_frames: number; gap_s: number; since: number }
  | { kind: "identity"; type: "suspected_id_switch"; frame: number; from_id: number; to_id: number; gap_frames: number; heuristic: true };

export interface JobResult {
  schema: string;
  frames_tracked: number;
  fps_tracked: number;
  gate: { name: string; a: [number, number]; b: [number, number]; norm: number[]; hysteresis_px: number };
  tracks: Record<string, TrackRow[]>;
  events: JobEvent[];
  metrics: JobMetrics;
  timing: JobTiming;
  video: { width: number; height: number; fps: number; frames: number; duration_s: number; stride: number };
  clip: Job["clip"] & { video_url: string };
  config: string;
  detector: { name: string; input: number; weights: string; score_floor: number };
  sample_detections?: { frame: number; tlwh: number[][]; score: number[] }[];
}

export interface AblationRow {
  name: string;
  label: string;
  group: "baseline" | "ours" | "control";
  headline: boolean;
  HOTA: number;
  IDF1: number;
  MOTA: number;
  AssA: number;
  DetA: number;
  IDSW: number;
  p50_ms: number | null;
  p95_ms: number | null;
  max_ms: number | null;
  stress_visibility: { bin: string; gt: number; recall: number; id_recall: number; idsw_per_1k_gt: number }[];
  stress_density: { bin: string; frames: number; gt: number; idf1: number; mota: number; idsw_per_1k_gt: number }[];
}

export interface Benchmarks {
  headline: AblationRow | null;
  ablation: { setting: string; source: string; reproduce: string; rows: AblationRow[]; sequences: string[] } | null;
  gate_events: {
    source: string;
    note: string;
    rows: ({ label: string; check: boolean } & Record<string, number | null | string | boolean>)[];
  } | null;
  visibility: { source: string; rows: { key: string; label: string; ours: boolean; n: number; mae: number; spearman: number; "auroc_vis_lt_0.5": number; "ap_vis_lt_0.5": number }[] } | null;
  go_no_go: { source: string; verdict: string; sanity_check: boolean; summary: string; rows: { condition: string; K: number; K_ms_mean: number; HOTA: number; IDF1: number; IDSW_trackeval: number; p95_ms: number }[] } | null;
  phase3: { status: "pending" | "available"; label?: string; items: { id: string; what: string; source: string }[]; files?: string[] };
  plots: Record<string, string>;
  gpu?: GpuRun | null;
}

export interface GpuRow { name: string; label: string; group: string; HOTA: number | null; IDF1: number | null; MOTA: number | null; AssA: number | null; DetA: number | null; IDSW: number | null; p95_ms: number | null }
export interface GpuRun {
  source: string;
  analysis: string;
  gpu: string | null;
  profile: string | null;
  tuned: Record<string, number> | null;
  datasets: { id: string; cache: string; dataset: string; clean: boolean; detector: string | null; trained_on: string | null; input_size: string | null; source: string; rows: GpuRow[]; plots: Record<string, string> }[];
  excluded: string[];
  reproduction: { verdict: string; source: string; reference: string; rows: { metric: string; paper: number | null; ours: number | null; delta: number | null; tolerance: number | null; within: boolean | null }[] } | null;
  go_no_go_v2: { verdict: string; source: string; rule: string; what: string; criteria: { id: string; what: string; rule: string; got: string | null; passed: boolean }[]; rows: { arm: string; HOTA: number; IDF1: number; MOTA: number; AssA: number; IDSW: number; p95_ms: number; flag_pct: number }[] } | null;
  gate_events: { source: string; rows: { label: string; check: boolean; gt_crossings: number; trk_crossings: number; count_error: number; precision: number; recall: number; attribution_error: number; tailgate_precision: number; tailgate_recall: number }[] } | null;
}

export interface Health {
  status: string;
  version: string;
  detectors: Record<string, boolean>;
  default_detector: string;
  clips: number;
  limits: { max_video_mb: number; max_video_seconds: number; max_job_frames: number };
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly fix?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const BASE = (import.meta.env?.VITE_API_BASE as string | undefined) ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, init);
  } catch (err) {
    throw new ApiError(0, "network", `cannot reach the service (${(err as Error).message})`);
  }
  if (!res.ok) {
    let code = "error";
    let message = `${res.status} ${res.statusText}`;
    let fix: string | undefined;
    try {
      const body = (await res.json()) as { error?: { code?: string; message?: string; fix?: string }; detail?: unknown };
      code = body.error?.code ?? code;
      message = body.error?.message ?? (typeof body.detail === "string" ? body.detail : message);
      fix = body.error?.fix;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message, fix);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  clips: () => request<Clip[]>("/api/clips"),
  jobs: (clipId?: string) => request<Job[]>(`/api/jobs${clipId ? `?clip_id=${encodeURIComponent(clipId)}` : ""}`),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  result: (id: string) => request<JobResult>(`/api/jobs/${id}/result`),
  benchmarks: () => request<Benchmarks>("/api/benchmarks"),
  createJob: (body: { clip_id: string; detector?: string; stride?: number; max_seconds?: number | null; config?: string; reuse?: boolean; gate?: [number, number, number, number] }) =>
    request<Job>("/api/jobs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  firstFrameUrl: (clipId: string) => `/api/clips/${encodeURIComponent(clipId)}/first-frame.jpg`,
  csvUrl: (jobId: string) => `/api/jobs/${jobId}/tracks.csv`,
  annotatedUrl: (jobId: string) => `/api/jobs/${jobId}/annotated.mp4`,
  eventsUrl: (jobId: string) => `/api/jobs/${jobId}/events.json`,
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  stressMap: (config: "default" | "mem6") => request<StressMapData>(`/api/stress/map/${config}`),
  stressReplay: (q: { people: number; hide_s: number; seed: number; config: "default" | "mem6" }) =>
    request<SceneData>(`/api/stress/replay?${new URLSearchParams({ people: String(q.people), hide_s: String(q.hide_s), seed: String(q.seed), config: q.config })}`),
  breakRun: (spec: BreakSpec) =>
    request<SceneData>("/api/break/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(spec) }),
  upload: (file: File, consent: boolean) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("consent", consent ? "true" : "false");
    return request<Clip>("/api/uploads", { method: "POST", body: fd });
  },
};
