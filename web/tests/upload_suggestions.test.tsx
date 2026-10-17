import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { JobResult } from "../src/api";
import { adjustedCounts, cropRect, crossingsCsv, rankSuggestions, rowSeek, suggestions, timeSeek, uploadSummary, visibleSuggestions, type Decisions } from "../src/upload/fromJob";
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
    expect(lines[0]).toBe("frame,time_s,time,kind,track_id,from_track_id,direction,operator_decision,in_plain_count,in_operator_adjusted_count,pair_score");
    expect(lines).toContain("421,14.00,00:14,possible_missed_crossing,12,7,in,confirmed,no,yes,");
    expect(lines).toContain("500,16.63,00:16,possible_missed_crossing,14,9,out,dismissed,no,no,");
    expect(lines).toContain("5,0.13,00:00,crossing,3,,in,,yes,yes,");
    expect(crossingsCsv(res, {})).toContain(",pending,no,no,");
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

describe("upload result: Confirm / Dismiss", () => {
  it("Confirm changes only the operator-adjusted count (live), never the tracker count; Dismiss hides the row", () => {
    render(<UploadResult result={res} jobId="j1" seek={null} onSeek={() => {}} grab={fakeGrab} />);
    const tin = () => screen.getByTestId("upload-counts-in").textContent;
    const tout = () => screen.getByTestId("upload-counts-out").textContent;
    const ain = () => screen.getByTestId("adjusted-counts-in").textContent;
    const aout = () => screen.getByTestId("adjusted-counts-out").textContent;
    expect([tin(), tout(), ain(), aout()]).toEqual(["1", "1", "1", "1"]);
    expect(screen.getByTestId("upload-counts").textContent).toMatch(/^Tracker count/);
    expect(screen.getByTestId("adjusted-counts").textContent).toMatch(/^Operator-adjusted.*0 crossings you confirmed/);
    expect(screen.getByTestId("upload-counts").textContent).not.toMatch(/Recovered/);
    const rows = screen.getAllByTestId("suggestion-item");
    expect(rows).toHaveLength(2);
    fireEvent.click(within(rows[0]).getByTestId("suggestion-confirm"));
    expect([tin(), tout(), ain(), aout()]).toEqual(["1", "1", "2", "1"]);
    const done = screen.getAllByTestId("suggestion-item")[0];
    expect(done.className).toContain("opacity-60");
    expect(within(done).getByTestId("suggestion-confirmed").textContent).toBe("✓");
    fireEvent.click(within(screen.getAllByTestId("suggestion-item")[1]).getByTestId("suggestion-dismiss"));
    expect(screen.getAllByTestId("suggestion-item")).toHaveLength(1);
    expect([tin(), tout(), ain(), aout()]).toEqual(["1", "1", "2", "1"]);
    fireEvent.click(within(done).getByText("Undo"));
    expect([ain(), aout()]).toEqual(["1", "1"]);
    expect(JSON.parse(window.localStorage.getItem("zed.decisions.j1")!)).toEqual({ "500:9:14": "dismissed" });
  });

  it("two separate lists: possible missed crossings first, then unsure crossings, same row style", () => {
    render(<UploadResult result={res} jobId="j3" seek={null} onSeek={() => {}} grab={fakeGrab} />);
    const missed = screen.getByTestId("missed-list"), unsure = screen.getByTestId("unsure-list");
    expect(within(missed).getByRole("heading").textContent).toBe("Possible missed crossings (2)");
    expect(within(unsure).getByRole("heading").textContent).toBe("Unsure crossings (1)");
    expect(within(missed).queryAllByTestId("unsure-item")).toHaveLength(0);
    expect(within(unsure).queryAllByTestId("suggestion-item")).toHaveLength(0);
    expect(within(unsure).getAllByTestId("unsure-item")).toHaveLength(1);
    expect(missed.compareDocumentPosition(unsure) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const a = within(missed).getAllByTestId("suggestion-item")[0], b = within(unsure).getAllByTestId("unsure-item")[0];
    expect(a.className).toBe(b.className);
    // downloads sit in one row below both lists
    const dl = screen.getByTestId("downloads");
    expect(unsure.compareDocumentPosition(dl) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(dl).getAllByRole("link").length).toBe(4);
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

  it("suggestions sorted by pairing score; top 3 then 'Show N more'", () => {
    render(<UploadResult result={ranked} jobId="j5" seek={null} onSeek={() => {}} grab={fakeGrab} />);
    const keyOf = (r: HTMLElement) => within(r).getAllByTestId("id-chip").map((c) => c.textContent).join(">");
    expect(screen.getAllByTestId("suggestion-item").map(keyOf)).toEqual(["#22>#23", "#26>#27", "#28>#29"]);
    const more = screen.getByTestId("suggestions-more");
    expect(more.textContent).toBe("Show 2 more");
    fireEvent.click(more);
    expect(screen.getAllByTestId("suggestion-item").map(keyOf)).toEqual(["#22>#23", "#26>#27", "#28>#29", "#20>#21", "#24>#25"]);
    expect(rankSuggestions(suggestions(ranked)).map((x) => x.score)).toEqual([0.9, 0.7, 0.55, 0.4, 0.1]);
  });
});

function hexToRgb(h: string) {
  const n = parseInt(h.slice(1), 16);
  return `rgb(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255})`;
}
