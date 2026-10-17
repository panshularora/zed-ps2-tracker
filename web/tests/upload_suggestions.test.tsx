import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { JobResult } from "../src/api";
import { adjustedCounts, crossingsCsv, suggestions, uploadSummary, visibleSuggestions, type Decisions } from "../src/upload/fromJob";
import { UploadResult } from "../src/workstation/screens/YourVideo";

const res = {
  frames_tracked: 600,
  video: { width: 640, height: 360, fps: 30, frames: 600, duration_s: 20, stride: 1 },
  gate: { name: "user", a: [0, 200], b: [640, 200], norm: [0, 0.556, 1, 0.556], hysteresis_px: 10 },
  clip: { id: "upload-abc", title: "mine", event_styling: "neutral", consent: false, fps: 30, frames: 600, video_url: "/v" },
  tracks: { "3": [[1, 10, 20, 30, 60, 0.9, 1]], "7": [[2, 100, 20, 30, 60, 0.8, 0.4]], "12": [[430, 100, 220, 30, 60, 0.8, 1]] },
  events: [
    { kind: "gate", type: "crossing", frame: 5, track_id: 3, direction: "in", timestamp: "", gate: "user", unsure: false },
    { kind: "gate", type: "crossing", frame: 11, track_id: 3, direction: "out", timestamp: "", gate: "user", unsure: true, reasons: ["occlusion"] },
    { kind: "gate", type: "recovered_crossing", frame: 421, track_id: 12, from_track_id: 7, direction: "in", timestamp: "", recovered: true },
    { kind: "gate", type: "recovered_crossing", frame: 500, track_id: 14, from_track_id: 9, direction: "out", timestamp: "", recovered: true },
  ],
  metrics: {}, tracker_info: {}, analysis_params: {}, sample_detections: [], timing: {}, schema: "zed-ps2-job-result/v1",
  config: "ours/visibility_botsort", detector: { name: "yolox_s" },
} as unknown as JobResult;

afterEach(() => window.localStorage.clear());

describe("possible missed crossings (pure)", () => {
  const s = suggestions(res);
  it("one suggestion per recovered_crossing, with time and both numbers", () => {
    expect(s.map((x) => x.text)).toEqual(["Possible missed crossing at 00:14 · #7 → #12", "Possible missed crossing at 00:16 · #9 → #14"]);
  });
  it("confirming changes only the operator-adjusted count; dismissing removes the row", () => {
    const plain = uploadSummary(res);
    const d: Decisions = { [s[0].key]: "confirmed" };
    expect(adjustedCounts(plain, s, d)).toEqual({ in: plain.in + 1, out: plain.out, confirmed: 1 });
    expect(uploadSummary(res)).toEqual(plain);
    const d2: Decisions = { ...d, [s[1].key]: "dismissed" };
    expect(visibleSuggestions(s, d2).map((x) => x.key)).toEqual([s[0].key]);
    expect(adjustedCounts(plain, s, d2)).toEqual({ in: plain.in + 1, out: plain.out, confirmed: 1 });
  });
  it("CSV has every crossing, every suggestion and a column for the operator's decision", () => {
    const csv = crossingsCsv(res, { [s[0].key]: "confirmed", [s[1].key]: "dismissed" });
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("frame,time_s,time,kind,track_id,from_track_id,direction,operator_decision,in_plain_count,in_operator_adjusted_count");
    expect(lines).toContain("421,14.00,00:14,possible_missed_crossing,12,7,in,confirmed,no,yes");
    expect(lines).toContain("500,16.63,00:16,possible_missed_crossing,14,9,out,dismissed,no,no");
    expect(lines).toContain("5,0.13,00:00,crossing,3,,in,,yes,yes");
    expect(crossingsCsv(res, {})).toContain(",pending,no,no");
  });
});

describe("upload result: Confirm / Dismiss", () => {
  it("Confirm changes only the operator-adjusted count, never IN/OUT; Dismiss hides the row", () => {
    render(<UploadResult result={res} jobId="j1" seek={null} onSeek={() => {}} />);
    const counts = screen.getByTestId("upload-counts");
    const plain = counts.textContent;
    expect(plain).toContain("IN 1");
    expect(plain).toContain("OUT 1");
    expect(plain).not.toMatch(/Recovered/);
    expect(screen.getByTestId("adjusted-counts").textContent).toMatch(/Operator-adjusted.*0 confirmed.*IN 1 · OUT 1/);
    const rows = screen.getAllByTestId("suggestion-item");
    expect(rows).toHaveLength(2);
    fireEvent.click(within(rows[0]).getByTestId("suggestion-confirm"));
    expect(screen.getByTestId("upload-counts").textContent).toBe(plain);
    expect(screen.getByTestId("adjusted-counts").textContent).toMatch(/1 confirmed.*IN 2 · OUT 1/);
    fireEvent.click(within(screen.getAllByTestId("suggestion-item")[1]).getByTestId("suggestion-dismiss"));
    expect(screen.getAllByTestId("suggestion-item")).toHaveLength(1);
    expect(screen.getByTestId("upload-counts").textContent).toBe(plain);
    expect(screen.getByTestId("adjusted-counts").textContent).toMatch(/IN 2 · OUT 1/);
    expect(JSON.parse(window.localStorage.getItem("zed.decisions.j1")!)).toEqual({ "421:7:12": "confirmed", "500:9:14": "dismissed" });
  });
  it("clicking a suggestion jumps the player to that moment", () => {
    const seen: number[] = [];
    render(<UploadResult result={res} jobId="j2" seek={null} onSeek={(f) => seen.push(f)} />);
    fireEvent.click(screen.getAllByTestId("suggestion-seek")[0]);
    expect(seen).toEqual([421]);
  });
});
