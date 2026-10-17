import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { YourVideo } from "../src/workstation/screens/YourVideo";

const clip = { id: "upload-x1", title: "mine.mp4", source: "upload", note: "", page: null, licence: "", consent: false, event_styling: "neutral", blurred: false, gate: null, width: 640, height: 360, fps: 30, frames: 60, duration_s: 2, video_url: null, poster_url: null };
const job = (status: string) => ({ id: "j1", status, stage: status === "succeeded" ? "done" : "track", progress: status === "succeeded" ? 1 : 0.5, clip: { id: clip.id, title: clip.title, event_styling: "neutral", consent: false, fps: 30, frames: 60 }, options: {}, created_at: "", started_at: null, finished_at: null, error: null, metrics: null, timing: null });
const result = {
  schema: "x", frames_tracked: 30, fps_tracked: 15,
  gate: { name: "user", a: [0, 180], b: [640, 180], norm: [0, 0.5, 1, 0.5], hysteresis_px: 10 },
  tracks: { "1": [[1, 10, 100, 30, 60, 0.9, 1], [2, 10, 110, 30, 60, 0.9, 1]] },
  events: [
    { kind: "gate", type: "crossing", frame: 9, track_id: 1, direction: "in", timestamp: "", gate: "user", unsure: true, reasons: ["occlusion"] },
    { kind: "recovery", type: "recovery", frame: 5, track_id: 1, reason: "lost", gap_frames: 2, gap_s: 0.1, since: 2 },
    { kind: "recovery", type: "recovery", frame: 7, track_id: 1, reason: "lost", gap_frames: 2, gap_s: 0.1, since: 4 },
  ],
  metrics: {}, timing: {}, video: { width: 640, height: 360, fps: 30, frames: 60, duration_s: 2, stride: 2 },
  clip: { ...job("succeeded").clip, video_url: "/api/clips/upload-x1/video" }, config: "ours/visibility_botsort",
  detector: { name: "yolox_s", input: 640, weights: "", score_floor: 0.1 },
};

let calls: { url: string; init?: RequestInit }[] = [];
const ok = (b: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(b), { status, headers: { "content-type": "application/json" } }));

beforeEach(() => {
  calls = [];
  vi.useFakeTimers({ shouldAdvanceTime: true });
  Object.defineProperty(HTMLImageElement.prototype, "naturalWidth", { configurable: true, get: () => 640 });
  Object.defineProperty(HTMLImageElement.prototype, "naturalHeight", { configurable: true, get: () => 360 });
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as never;
  HTMLMediaElement.prototype.play = vi.fn(() => Promise.resolve()) as never;
  HTMLMediaElement.prototype.pause = vi.fn() as never;
  vi.stubGlobal("fetch", vi.fn((url: string, init?: RequestInit) => {
    calls.push({ url, init });
    if (url === "/api/uploads") return ok(clip, 201);
    if (url === "/api/jobs") return ok(job("running"), 202);
    if (url === "/api/jobs/j1") return ok(job("succeeded"));
    if (url === "/api/jobs/j1/result") return ok(result);
    return ok({}, 404);
  }));
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("Studio step 4 · Your video", () => {
  it("upload -> click two points -> job carries the gate -> Studio player with labels and unsure list", async () => {
    render(<YourVideo />);
    const input = screen.getByTestId("your-video-input");
    await act(async () => { fireEvent.change(input, { target: { files: [new File(["x"], "mine.mp4", { type: "video/mp4" })] } }); });
    const imgEl = await screen.findByTestId("first-frame");
    expect(imgEl.getAttribute("src")).toBe("/api/clips/upload-x1/first-frame.jpg");
    imgEl.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 360, right: 640, bottom: 360, x: 0, y: 0, toJSON: () => ({}) });
    fireEvent.click(imgEl, { clientX: 0, clientY: 180 });
    fireEvent.click(imgEl, { clientX: 640, clientY: 180 });
    const run = screen.getByTestId("run-with-gate");
    await waitFor(() => expect((run as HTMLButtonElement).disabled).toBe(false));
    await act(async () => { fireEvent.click(run); });
    await act(async () => { vi.advanceTimersByTime(1000); });
    await screen.findByTestId("upload-result");
    const create = calls.find((c) => c.url === "/api/jobs");
    expect(JSON.parse(String(create?.init?.body))).toEqual({ clip_id: "upload-x1", gate: [0, 0.5, 1, 0.5] });
    expect(screen.getByTestId("no-gt-label").textContent).toBe("Your clip · no ground truth");
    expect(screen.getByTestId("reid-off-label").textContent).toMatch(/re-identification off on CPU/i);
    expect(screen.getByTestId("upload-counts-in").textContent).toBe("1");
    expect(screen.getByTestId("upload-counts").textContent).not.toMatch(/kept|accuracy|IDF1|HOTA/i);
    expect(screen.getAllByTestId("unsure-item")).toHaveLength(1);
    expect(screen.getByTestId("timeline-summary").textContent).toMatch(/track numbers? on screen · no ground truth/);
    expect(screen.getByTestId("upload-result").textContent).not.toMatch(/\bkept\b|accuracy|IDF1|HOTA/i);
    expect(screen.getByTestId("dl-csv").getAttribute("href")).toBe("/api/jobs/j1/tracks.csv");
    expect(screen.getByTestId("dl-annotated").getAttribute("href")).toBe("/api/jobs/j1/annotated.mp4");
  });
});
