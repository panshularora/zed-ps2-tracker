import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../src/api";
import { ErrorCard } from "../src/upload/ErrorCard";
import { explainError, MODELS_FIX, SERVE_FIX } from "../src/upload/explain";
import { YourVideo } from "../src/workstation/screens/YourVideo";

const RAW = "OpenCV(5.0.0) D:\\a\\opencv-python\\onnx_importer2.cpp:341: error: (-5:Bad argument) Can't read ONNX file: C:\\Users\\Panshul\\Desktop\\zed-ps2-tracker-main\\data\\models\\face_detection_yunet_2023mar.onnx in function 'cv::dnn::dnn5_v20260605::ONNXImporter2::parseFile'";

describe("explainError", () => {
  it("turns the raw OpenCV ONNX error into one line + the fetch command", () => {
    const e = explainError(new Error(RAW));
    expect(e.cause).not.toContain("OpenCV");
    expect(e.cause).toMatch(/model file is missing/);
    expect(e.fix).toBe(MODELS_FIX);
    expect(e.raw).toBe(RAW);
  });
  it("uses the server's cause and fix for model_missing job errors", () => {
    const e = explainError({ code: "model_missing", message: "x", cause: "could not download yolox_s.onnx (timed out)", fix: "python -m serve.detector fetch --out data/models" });
    expect(e.cause).toContain("timed out");
    expect(e.fix).toContain("serve.detector fetch");
  });
  it("says the backend is not running on a network error", () => {
    expect(explainError(new ApiError(0, "network", "cannot reach the service (Failed to fetch)")).fix).toBe(SERVE_FIX);
  });
});

describe("ErrorCard", () => {
  it("shows cause, fix and Retry; the raw error stays behind a closed Details toggle", () => {
    const retry = vi.fn();
    render(<ErrorCard err={explainError(new Error(RAW))} onRetry={retry} />);
    const card = screen.getByTestId("error-card");
    expect(card.className).toContain("yv-err");
    expect(card.className).not.toContain("text-fail");
    expect(screen.getByTestId("error-fix").textContent).toBe(MODELS_FIX);
    const det = screen.getByTestId("error-details") as HTMLDetailsElement;
    expect(det.open).toBe(false);
    expect(det.textContent).toContain("Can't read ONNX file");
    fireEvent.click(screen.getByTestId("error-retry"));
    expect(retry).toHaveBeenCalledTimes(1);
  });
});

describe("Your video: failed job and first frame", () => {
  const clip = { id: "upload-x1", title: "mine.mp4", source: "upload", note: "", page: null, licence: "", consent: false, event_styling: "neutral", blurred: false, gate: null, width: 640, height: 360, fps: 30, frames: 60, duration_s: 2, video_url: null, poster_url: null };
  const failed = { id: "j1", status: "failed", stage: "failed", progress: 0, clip: { id: clip.id, title: "", event_styling: "neutral", consent: false, fps: 30, frames: 60 }, options: {}, created_at: "", started_at: null, finished_at: null, error: { code: "job_failed", message: RAW }, metrics: null, timing: null };
  let jobs = 0;
  const ok = (b: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(b), { status, headers: { "content-type": "application/json" } }));
  beforeEach(() => {
    jobs = 0;
    vi.stubGlobal("fetch", vi.fn((url: string) => {
      if (url === "/api/uploads") return ok(clip, 201);
      if (url === "/api/jobs") { jobs += 1; return ok(failed, 202); }
      return ok({}, 404);
    }));
  });
  afterEach(() => vi.unstubAllGlobals());

  it("shows the empty #13171C frame while loading, then a calm card instead of the raw trace; Retry re-runs", async () => {
    render(<YourVideo />);
    await act(async () => { fireEvent.change(screen.getByTestId("your-video-input"), { target: { files: [new File(["x"], "mine.mp4", { type: "video/mp4" })] } }); });
    expect(await screen.findByTestId("first-frame-empty")).toBeTruthy();
    expect(screen.getByTestId("first-frame").className).toContain("hidden");
    await act(async () => { fireEvent.click(screen.getByText("Skip: use a level line at 60% height")); });
    const card = await screen.findByTestId("error-card");
    expect(screen.getByTestId("error-cause").textContent).not.toContain("OpenCV(5.0.0)");
    expect(screen.getByTestId("error-fix").textContent).toBe(MODELS_FIX);
    expect(card.querySelector("details")!.open).toBe(false);
    await act(async () => { fireEvent.click(screen.getByTestId("error-retry")); });
    await waitFor(() => expect(jobs).toBe(2));
  });
});
