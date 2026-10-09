import type { AblationRow, Benchmarks, JobEvent, JobResult, TrackRow } from "../src/api";

export function row(name: string, group: AblationRow["group"], headline = false, over: Partial<AblationRow> = {}): AblationRow {
  return {
    name, label: name, group, headline, HOTA: 50, IDF1: 56, MOTA: 47, AssA: 58, DetA: 43, IDSW: 191,
    p50_ms: 0.9, p95_ms: 1.9, max_ms: 3,
    stress_visibility: [
      { bin: "0.00-0.25", gt: 10, recall: 0.01, id_recall: 0.02, idsw_per_1k_gt: 0.4 },
      { bin: "0.75-1.00", gt: 10, recall: 0.9, id_recall: 0.7, idsw_per_1k_gt: 3 },
    ],
    stress_density: [],
    ...over,
  };
}

export function bench(over: Partial<Benchmarks> = {}): Benchmarks {
  const head = row("ours_vis_botsort", "ours", true, { HOTA: 52.5, IDF1: 59.8, IDSW: 180, p95_ms: 2.27 });
  return {
    headline: head,
    ablation: { setting: "MOT17 val-half", source: "docs/results/ablation/ablation.json", reproduce: "make ablate", rows: [row("botsort", "baseline"), row("buffer90", "control"), head], sequences: [] },
    gate_events: {
      source: "docs/results/gate_events/gate_events.json",
      note: "25 crossings",
      rows: [
        { label: "gt as tracker (check)", check: true, gt_crossings: 25, trk_crossings: 25, recall: 1, precision: 1, attribution_error: 0, count_error: 0, tailgate_precision: 1, tailgate_recall: 1 },
        { label: "BoT-SORT + visibility", check: false, gt_crossings: 25, trk_crossings: 18, recall: 0.6, precision: 0.83, attribution_error: 0.267, count_error: 0.28, tailgate_precision: 0.67, tailgate_recall: 0.44 },
      ],
    },
    visibility: null,
    go_no_go: { source: "docs/results/go_no_go/rows_v2.json", verdict: "no-go", sanity_check: true, summary: "docs/results/go_no_go.md", rows: [{ condition: "MOT17 public FRCNN", K: 0, K_ms_mean: 0, HOTA: 50.1, IDF1: 56.3, IDSW_trackeval: 191, p95_ms: 2.6 }] },
    phase3: { status: "pending", label: "pending DGX run", items: [{ id: "mot20", what: "MOT20 val-half", source: "docs/results/phase3/<id>/" }] },
    plots: {},
    ...over,
  };
}

/** One track walking down through a horizontal gate at y = 100, frames 1..20; bottom y = 60 + 5 f. */
export function result(events: JobEvent[] = [], stride = 1): JobResult {
  const rows: TrackRow[] = [];
  for (let f = 1; f <= 20; f++) if (f < 8 || f > 11) rows.push([f, 10, 10 + 5 * f, 20, 50, 0.9, f === 12 ? 0.4 : 1]);
  return {
    schema: "zed-ps2-job-result/v1",
    frames_tracked: 20,
    fps_tracked: 10 / stride,
    gate: { name: "gate", a: [0, 100], b: [200, 100], norm: [0, 0.5, 1, 0.5], hysteresis_px: 10 },
    tracks: { "7": rows },
    events,
    metrics: { frames: 20, tracks: 1, avg_track_length_frames: 20, avg_track_length_s: 2, median_track_length_s: 2, boxes: 16, id_preserving_recoveries: 1, recoveries_after_lost: 1, recoveries_after_occluded: 0, suspected_id_switches: 0, crossings_in: 1, crossings_out: 0, tailgates: 0, ambiguous_holds: 0 },
    timing: { decode_s: 0.1, detect_s: 1, detector_p50_ms: 58, detector_p95_ms: 70, tracker_p50_ms: 1, tracker_p95_ms: 2, tracker_total_s: 0.02, wall_s: 1.2, per_frame_ms: 60, machine: "x86_64", job_s: 1.3 },
    video: { width: 200, height: 200, fps: 10, frames: 20 * stride, duration_s: 2 * stride, stride },
    clip: { id: "walk", title: "Walk", event_styling: "neutral", consent: false, fps: 10, frames: 20, video_url: "/api/clips/walk/video" },
    config: "ours/visibility_botsort",
    detector: { name: "yolox_s", input: 640, weights: "YOLOX COCO", score_floor: 0.1 },
    sample_detections: [{ frame: 1, tlwh: [[10, 15, 20, 50]], score: [0.9] }],
  };
}

export const EVENTS: JobEvent[] = [
  { kind: "gate", type: "crossing", frame: 6, track_id: 7, direction: "in", timestamp: "t", gate: "gate" },
  { kind: "recovery", type: "recovery", frame: 12, track_id: 7, reason: "lost", gap_frames: 4, gap_s: 0.4, since: 7 },
  { kind: "gate", type: "tailgate", frame: 14, track_id: 9, lead_track_id: 7, direction: "in", timestamp: "t", gap_s: 0.3, confidence: 1 },
  { kind: "gate", type: "ambiguous_hold", frame: 15, track_id: 3, direction: "out", timestamp: "t", reasons: ["occlusion"], confidence: 0.2 },
  { kind: "identity", type: "suspected_id_switch", frame: 16, from_id: 3, to_id: 11, gap_frames: 2, heuristic: true },
];
