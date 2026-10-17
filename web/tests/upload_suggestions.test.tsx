import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { JobResult } from "../src/api";
import { cropRect, crossingsCsv, rankSuggestions, reviewNotesJson, rowSeek, suggestions, timeSeek, uploadSummary } from "../src/upload/fromJob";
import type { ThumbReq } from "../src/upload/thumbs";
import { neutralColor } from "../src/lib/scenarios";
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
  it("CSV has every crossing and every suggestion, with a review_status column for listed moments", () => {
    const csv = crossingsCsv(res, { [s[0].key]: "confirmed", [s[1].key]: "dismissed", "u:11:3": "dismissed" });
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("frame,time_s,time,kind,track_id,from_track_id,direction,in_tracker_count,unsure,review_status,pair_score");
    expect(lines[0]).not.toMatch(/adjusted/);
    expect(lines).toContain("421,14.00,00:14,possible_missed_crossing,12,7,in,no,,confirmed,");
    expect(lines).toContain("500,16.63,00:16,possible_missed_crossing,14,9,out,no,,dismissed,");
    expect(lines).toContain("5,0.13,00:00,crossing,3,,in,yes,no,,");
    expect(lines).toContain("11,0.33,00:00,crossing,3,,out,yes,yes,dismissed,");
    expect(crossingsCsv(res, {})).toContain("11,0.33,00:00,crossing,3,,out,yes,yes,pending,");
    // notes never change the tracker count
    expect(uploadSummary(res)).toEqual(uploadSummary(res));
  });
  it("review notes JSON lists every moment with its status and derives no count", () => {
    const j = reviewNotesJson(res, "j9", { [s[0].key]: "confirmed" });
    expect(j.moments.map((m) => [m.kind, m.frame, m.status])).toEqual([
      ["unsure_crossing", 11, "pending"], ["possible_missed_crossing", 421, "confirmed"], ["possible_missed_crossing", 500, "pending"],
    ]);
    expect(JSON.stringify(j)).not.toMatch(/adjusted|"in":|"out":/);
  });
});

const fakeGrab = vi.fn(async (_url: string, req: ThumbReq) => req.rects.map((r, i) => `data:${req.key}:${i}:${r.w}x${r.h}`));

// five suggestions with pairing scores, out of frame order, plus one unsure crossing
const ranked = {
  ...res,
  tracks: { ...res.tracks, "20": [[50, 5, 5, 20, 40, 0.9, 1]], "21": [[60, 15, 5, 20, 40, 0.9, 1]] },
  events: [
    res.events[0], res.events[1],
    ...[[100, 0.4], [200, 0.9], [300, 0.1], [400, 0.7], [500, 0.55]].map(([f, sc], k) => ({ kind: "gate", type: "recovered_crossing", frame: f, track_id: 21 + k * 2, from_track_id: 20 + k * 2, direction: k === 0 ? "in" : "out", timestamp: "", recovered: true, pair_score: sc })),
  ],
} as unknown as JobResult;

describe("upload result: Confirm / Dismiss are review notes", () => {
  it("one Tracker count card; Confirm / Dismiss / Undo change no number and no row disappears", () => {
    render(<UploadResult result={res} jobId="j1" seek={null} onSeek={() => {}} grab={fakeGrab} />);
    const tin = () => screen.getByTestId("upload-counts-in").textContent;
    const tout = () => screen.getByTestId("upload-counts-out").textContent;
    const numbers = () => [tin(), tout(), screen.getByTestId("moments-heading").textContent];
    expect(numbers()).toEqual(["1", "1", "Moments worth checking (3)"]);
    expect(screen.getByTestId("upload-counts").textContent).toMatch(/^Tracker count/);
    expect(screen.queryByTestId("adjusted-counts")).toBeNull();
    expect(screen.getByTestId("upload-result").textContent).not.toMatch(/adjusted/i);
    const rows = screen.getAllByTestId("suggestion-item");
    expect(rows).toHaveLength(2);
    fireEvent.click(within(rows[0]).getByTestId("suggestion-confirm"));
    expect(numbers()).toEqual(["1", "1", "Moments worth checking (3)"]);
    expect(within(screen.getAllByTestId("suggestion-item")[0]).getByTestId("review-status").textContent).toBe("Noted: confirmed");
    fireEvent.click(within(screen.getAllByTestId("suggestion-item")[1]).getByTestId("suggestion-dismiss"));
    expect(screen.getAllByTestId("suggestion-item")).toHaveLength(2);
    expect(within(screen.getAllByTestId("suggestion-item")[1]).getByTestId("review-status").textContent).toBe("Noted: dismissed");
    fireEvent.click(within(screen.getByTestId("unsure-item")).getByTestId("suggestion-confirm"));
    expect(numbers()).toEqual(["1", "1", "Moments worth checking (3)"]);
    fireEvent.click(within(screen.getAllByTestId("suggestion-item")[0]).getByTestId("review-undo"));
    expect(within(screen.getAllByTestId("suggestion-item")[0]).getByTestId("suggestion-confirm")).toBeTruthy();
    expect(screen.getAllByTestId("suggestion-item").map((r) => r.dataset.decision)).toEqual(["pending", "dismissed"]);
    expect(JSON.parse(window.localStorage.getItem("zed.decisions.j1")!)).toEqual({ "500:9:14": "dismissed", "u:11:3": "confirmed" });
  });

  it("'Moments worth checking (N)' holds both groups: possible missed crossings first, then unsure crossings, same row style", () => {
    render(<UploadResult result={res} jobId="j3" seek={null} onSeek={() => {}} grab={fakeGrab} />);
    const box = screen.getByTestId("moments-to-check");
    const missed = within(box).getByTestId("missed-list"), unsure = within(box).getByTestId("unsure-list");
    expect(within(missed).getByRole("heading").textContent).toBe("Possible missed crossings (2)");
    expect(within(unsure).getByRole("heading").textContent).toBe("Unsure crossings (1)");
    expect(within(missed).queryAllByTestId("unsure-item")).toHaveLength(0);
    expect(within(unsure).queryAllByTestId("suggestion-item")).toHaveLength(0);
    expect(within(unsure).getAllByTestId("unsure-item")).toHaveLength(1);
    expect(missed.compareDocumentPosition(unsure) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const a = within(missed).getAllByTestId("suggestion-item")[0], b = within(unsure).getAllByTestId("unsure-item")[0];
    expect(a.className).toBe(b.className);
    // downloads sit in one row below both lists, including the review notes
    const dl = screen.getByTestId("downloads");
    expect(unsure.compareDocumentPosition(dl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(dl).getAllByRole("link").length).toBe(5);
    expect(within(dl).getByTestId("dl-review-json").textContent).toBe("Review notes JSON");
  });

  it("each row: 64x36 still of that frame, ID chips in their colours, crops of both IDs; row click plays 2 s, time link seeks 1 s before", async () => {
    const seen: [number, string][] = [];
    render(<UploadResult result={res} jobId="j2" seek={null} onSeek={(f, m) => seen.push([f, m])} grab={fakeGrab} />);
    const row = screen.getAllByTestId("suggestion-item")[0];
    const thumb = await within(row).findByTestId("row-thumb");
    expect(thumb.getAttribute("width")).toBe("64");
    expect(thumb.getAttribute("height")).toBe("36");
    expect(thumb.getAttribute("src")).toBe("data:s:421:7:12:0:64x36");
    const thumbReq = fakeGrab.mock.calls.map((c) => c[1]).find((r) => r.key === "s:421:7:12")!;
    expect(thumbReq.t).toBeCloseTo(14.0);
    expect(thumbReq.rects[0].r).toEqual([0, 0, 640, 360]);
    const chips = within(row).getAllByTestId("id-chip");
    expect(chips.map((c) => c.textContent)).toEqual(["#7", "#12"]);
    expect(chips[0].style.background).toBe(hexToRgb(neutralColor(7)));
    expect(chips[1].style.background).toBe(hexToRgb(neutralColor(12)));
    // crops: last box of #7 (frame 2), first box of #12 (frame 430), with a margin
    const ra = fakeGrab.mock.calls.map((c) => c[1]).find((r) => r.key === "a:421:7:12")!;
    const rb = fakeGrab.mock.calls.map((c) => c[1]).find((r) => r.key === "b:421:7:12")!;
    expect(ra.t).toBeCloseTo(1 / 30);
    expect(rb.t).toBeCloseTo(429 / 30);
    expect(ra.rects[0].r).toEqual(cropRect([100, 20, 30, 60], 640, 360));
    await within(row).findByAltText("#7");
    expect(within(within(row).getByTestId("row-crops")).getAllByRole("img").map((x) => x.getAttribute("alt"))).toEqual(["#7", "#12"]);
    fireEvent.click(row);
    fireEvent.click(within(row).getByTestId("row-time"));
    fireEvent.click(within(row).getByTestId("suggestion-confirm"));
    expect(seen).toEqual([[421, "row"], [421, "time"]]);
    expect(rowSeek(421, 1)).toEqual({ f: 421, n: 1, leadS: 1, playS: 2 });
    expect(timeSeek(421, 2)).toEqual({ f: 421, n: 2, leadS: 1, playS: 0 });
  });

  it("the player starts 1 s before and plays the 2 s window on a row seek, and pauses on a time seek", () => {
    const { rerender } = render(<UploadResult result={res} jobId="j4" seek={rowSeek(421, 1)} onSeek={() => {}} grab={fakeGrab} />);
    const stage = () => screen.getByTestId("identity-stage");
    expect(Number(stage().dataset.t)).toBeCloseTo(13.0, 1);
    expect(stage().dataset.playing).toBe("1");
    rerender(<UploadResult result={res} jobId="j4" seek={timeSeek(421, 2)} onSeek={() => {}} grab={fakeGrab} />);
    expect(Number(stage().dataset.t)).toBeCloseTo(13.0, 1);
    expect(stage().dataset.playing).toBe("0");
  });

  it("every suggestion renders in crossing-time order, no cap or 'Show N more', score as grey text", () => {
    render(<UploadResult result={ranked} jobId="j5" seek={null} onSeek={() => {}} grab={fakeGrab} />);
    const keyOf = (r: HTMLElement) => within(r).getAllByTestId("id-chip").map((c) => c.textContent).join(">");
    expect(screen.queryByTestId("suggestions-more")).toBeNull();
    expect(screen.getByTestId("missed-list").textContent).not.toMatch(/Show \d+ more|Show top/);
    const rows = screen.getAllByTestId("suggestion-item");
    expect(rows.map(keyOf)).toEqual(["#20>#21", "#22>#23", "#24>#25", "#26>#27", "#28>#29"]);
    expect(rows.map((r) => within(r).getByTestId("row-score").textContent)).toEqual(["pairing score 0.40", "pairing score 0.90", "pairing score 0.10", "pairing score 0.70", "pairing score 0.55"]);
    expect(within(rows[0]).getByTestId("row-score").className).toContain("text-mute");
    expect(within(rows[0]).getByTestId("row-score").className).toContain("text-[11px]");
    // the score order is still available for the offline demo, it just isn't the list order
    expect(rankSuggestions(suggestions(ranked)).map((x) => x.score)).toEqual([0.9, 0.7, 0.55, 0.4, 0.1]);
  });

  it("the top count is one 32px 'Tracker count' block; the strip above the player shows the clip time ('So far at mm:ss:')", () => {
    const { rerender } = render(<UploadResult result={res} jobId="j6" seek={null} onSeek={() => {}} grab={fakeGrab} />);
    expect(screen.getByTestId("upload-counts-in").parentElement!.className).toContain("text-[32px]");
    expect(screen.getByTestId("upload-counts").textContent).toMatch(/^Tracker count/);
    expect(screen.queryByTestId("adjusted-counts")).toBeNull();
    expect(screen.getByTestId("count-bar-label").textContent).toBe("So far at 00:00:");
    // follows the player: a seek to 1 s before the crossing at 14.0 s
    rerender(<UploadResult result={res} jobId="j6" seek={timeSeek(421, 1)} onSeek={() => {}} grab={fakeGrab} />);
    expect(screen.getByTestId("count-bar-label").textContent).toBe("So far at 00:13:");
  });
});

function hexToRgb(h: string) {
  const n = parseInt(h.slice(1), 16);
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`;
}
