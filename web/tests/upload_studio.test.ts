import { describe, expect, it } from "vitest";
import type { JobResult } from "../src/api";
import { fromJobResult, gateFromClicks, NO_GT_LABEL, REID_OFF_LABEL, unsureMoments, uploadSummary } from "../src/upload/fromJob";

const res = {
  frames_tracked: 20,
  video: { width: 640, height: 360, fps: 30, frames: 40, duration_s: 1.33, stride: 2 },
  gate: { name: "user", a: [0, 200], b: [640, 200], norm: [0, 0.556, 1, 0.556], hysteresis_px: 10 },
  clip: { id: "upload-abc", title: "mine", event_styling: "neutral", consent: false, fps: 30, frames: 40, video_url: "/v" },
  tracks: { "3": [[1, 10, 20, 30, 60, 0.9, 1], [2, 12, 22, 30, 60, 0.9, 1]], "7": [[2, 100, 20, 30, 60, 0.8, 0.4]] },
  events: [
    { kind: "gate", type: "crossing", frame: 5, track_id: 3, direction: "in", timestamp: "", gate: "user", unsure: false },
    { kind: "gate", type: "crossing", frame: 11, track_id: 7, direction: "out", timestamp: "", gate: "user", unsure: true, reasons: ["occlusion"] },
    { kind: "gate", type: "ambiguous_hold", frame: 15, track_id: 3, direction: "out", timestamp: "", reasons: ["low_confidence"], confidence: 0.3 },
    { kind: "recovery", type: "recovery", frame: 8, track_id: 3, reason: "lost", gap_frames: 3, gap_s: 0.2, since: 4 },
    { kind: "recovery", type: "recovery", frame: 9, track_id: 3, reason: "lost", gap_frames: 3, gap_s: 0.2, since: 6 },
    { kind: "recovery", type: "recovery", frame: 10, track_id: 3, reason: "occluded", gap_frames: 3, gap_s: 0.2, since: 7 },
    { kind: "identity", type: "suspected_id_switch", frame: 12, from_id: 3, to_id: 7, gap_frames: 2, heuristic: true },
  ],
} as unknown as JobResult;

describe("upload -> Studio player", () => {
  it("builds ReplayData with one ours run, the user gate and every crossing counted", () => {
    const d = fromJobResult(res, "/api/clips/upload-abc/video");
    expect(d.fps).toBe(15);
    expect(d.duration).toBeCloseTo(20 / 15);
    expect(d.runs.ours.frames["2"]).toHaveLength(2);
    expect(d.gate).toEqual({ a: [0, 200], b: [640, 200] });
    expect(d.events.ours.map((e) => e.f)).toEqual([5, 11, 15]);
    expect(d.caption).toContain(NO_GT_LABEL);
    expect(d.caption).toContain(REID_OFF_LABEL);
  });
  it("lists unsure moments (ambiguous_hold and unsure crossings) with their video time", () => {
    const u = unsureMoments(res);
    expect(u.map((m) => [m.f, m.id])).toEqual([[11, 7], [15, 3]]);
    expect(u[0].t).toBeCloseTo((11 - 1) * 2 / 30);
    expect(u[0].text).toMatch(/partly hidden/);
  });
  it("summary is counts and IDs only; never a kept-per-person number that can exceed people", () => {
    const s = uploadSummary(res);
    expect(s).toEqual({ ids: 2, in: 1, out: 2, unsure: 2, possibleBreaks: 1 });
    expect(Object.keys(s)).not.toContain("kept");
    // three recovery events on one track: per-event counts must not be shown as people
    expect(res.events.filter((e) => e.kind === "recovery").length).toBeGreaterThan(s.ids);
  });
  it("gate from two clicks is normalised and clamped", () => {
    expect(gateFromClicks([0, 180], [640, 200], 640, 360)).toEqual([0, 0.5, 1, 0.5556]);
    expect(gateFromClicks([-5, 400], [700, 0], 640, 360)).toEqual([0, 1, 1, 0]);
  });
});
