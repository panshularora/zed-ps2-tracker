import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, afterEach } from "vitest";
import { beatCopy, firstKeepId, involvedBetween, isDefault, lineSplitter, personIdLabel, readStream, recentEvents, recordingUrl, streamUrl, DEFAULTS, type SimFrame } from "../src/lib/gateSim";
import GateScene from "../src/pages/GateScene";

const pane = (events: [number, number, number, number, number][] = [], counted = 0) => ({ ids: [[1, 4]] as [number, number][], extra: [], held: [] as [number, number, number][], events, counted, id_changes: events.length });
const frame = (f: number, ev: [number, number, number, number, number][] = []): SimFrame => ({ type: "frame", f, people: [[1, 10, 10, 1]], dets: [], true_count: 1, beat: null, focus: null, bytetrack: pane(ev, 2), ours: pane() });
const meta = { type: "meta", width: 800, height: 450, gate: [[0, 225], [800, 225]], fps: 30, params: { seed: 7, play: "keep" }, trackers: { bytetrack: { label: "ByteTrack", config: "c" }, ours: { label: "Ours · one body, one number", config: "keep_id" } }, max_people: 40, play: "keep" };

afterEach(() => vi.unstubAllGlobals());

describe("gate sim client", () => {
  it("splits NDJSON across chunk boundaries", () => {
    const s = lineSplitter();
    expect(s('{"type":"end","fra')).toEqual([]);
    expect(s('mes":1}\n{"type":"me')).toEqual([{ type: "end", frames: 1 }]);
    expect(s('ta"}', true)).toEqual([{ type: "meta" }]);
  });
  it("reads a streamed response", async () => {
    const out: unknown[] = [];
    await readStream(new Response(`${JSON.stringify(meta)}\n${JSON.stringify(frame(1))}\n`), (l) => out.push(l));
    expect(out).toHaveLength(2);
  });
  it("builds the stream url from the controls and knows the defaults", () => {
    expect(streamUrl(DEFAULTS)).toBe("/api/gate-sim/stream?seed=7&density=0.5&occlusion=1.5&noise=0.15&seconds=12&play=keep");
    expect(isDefault({ ...DEFAULTS, seed: 8 })).toBe(false);
    expect(recordingUrl("keep")).toBe("/offline/gate-keep.ndjson");
    expect(beatCopy("hide", 1).tag).toMatch(/person 1 walked behind/);
    expect(beatCopy("back", 1).body).toMatch(/Same body/);
  });
  it("finds the people involved in recent ID changes", () => {
    const fs = [frame(1), frame(2, [[1, 4, 9, 10, 10]]), frame(3)];
    expect([...involvedBetween(fs, 1, 3)]).toEqual([1]);
    expect(recentEvents(fs, 2, "bytetrack", 5).map((x) => x.age)).toEqual([1]);
    expect(recentEvents(fs, 2, "ours", 5)).toEqual([]);
  });
  it("first keep id is the first tracker number that person was given", () => {
    const a = frame(1);
    const b = frame(2, [[1, 4, 9, 10, 10]]);
    b.bytetrack.ids = [[1, 9]];
    expect(firstKeepId([a, b], 1, "bytetrack", 1)).toBe(4);
  });
  it("labels visible keep, hidden hold, and a new number as three different badges", () => {
    expect(personIdLabel(true, 4, 4)).toEqual({ text: "ID #4", kind: "ok" });
    expect(personIdLabel(false, null, 4)).toEqual({ text: "HELD #4", kind: "held" });
    expect(personIdLabel(true, 9, 4)).toEqual({ text: "NEW #9  was #4", kind: "fail" });
  });
});

describe("gate scene page", () => {
  it("streams real tracker output into two panes with counters", async () => {
    const body = [meta, frame(1), frame(2, [[1, 4, 9, 10, 10]]), { type: "end", frames: 2, true_count: 1, summary: {} }].map((x) => JSON.stringify(x)).join("\n") + "\n";
    const f = vi.fn(async (_u: string) => new Response(body, { headers: { "content-type": "application/x-ndjson" } }));
    vi.stubGlobal("fetch", f);
    render(<GateScene />);
    expect(screen.getByTestId("synthetic-tag").textContent).toBe("Synthetic scene · exact ground truth");
    await waitFor(() => expect(f).toHaveBeenCalled());
    expect(String(f.mock.calls[0][0])).toContain("/api/gate-sim/stream?seed=7");
    await waitFor(() => expect(screen.getByText("ByteTrack")).toBeTruthy());
    expect(screen.getByTestId("seed-chip").textContent).toContain("seed 7");
    expect(screen.getByTestId("gate-caption").textContent).toBe("Left: ByteTrack, the published tracker we start from. Right: ours = ByteTrack + rules that keep a hidden person's number.");
    expect(screen.getByTestId("scene-counters-bytetrack").textContent).toContain("True people");
    expect(screen.getByTestId("play-keep").textContent).toMatch(/one body, one number/i);
    expect(screen.getByTestId("how-id-kept").textContent).toContain("HELD #N");
    expect(screen.getByTestId("how-id-kept").textContent).toContain("NEW #M was #N");
    expect(screen.getByText(/Ours · one body, one number/)).toBeTruthy();
  });
  it("falls back to the recorded run when the backend is offline", async () => {
    const body = [meta, frame(1)].map((x) => JSON.stringify(x)).join("\n") + "\n";
    const f = vi.fn(async (u: string) => (u.startsWith("/api/") ? new Response("x", { status: 503 }) : new Response(body)));
    vi.stubGlobal("fetch", f);
    render(<GateScene />);
    await waitFor(() => expect(screen.getByText(/replaying a recorded run/)).toBeTruthy());
    expect(f.mock.calls.map((c) => String(c[0]))).toContain("/offline/gate-keep.ndjson");
  });
  it("pauses on a hide beat and Continue resumes", async () => {
    const hide: SimFrame = { ...frame(2), beat: "hide", focus: 1, people: [[1, 400, 225, 0], [2, 420, 240, 1]] };
    const body = [meta, frame(1), hide, { type: "end", frames: 2, true_count: 1, summary: {} }].map((x) => JSON.stringify(x)).join("\n") + "\n";
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { headers: { "content-type": "application/x-ndjson" } })));
    let now = 0;
    let n = 0;
    vi.stubGlobal("performance", { now: () => now });
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
      n += 1;
      if (n > 24) return 0;
      now += 80;
      queueMicrotask(() => cb(now));
      return n;
    });
    vi.stubGlobal("cancelAnimationFrame", () => undefined);
    render(<GateScene />);
    await waitFor(() => expect(screen.getByTestId("gs-why")).toBeTruthy());
    expect(screen.getByTestId("gs-why").textContent).toMatch(/walked behind/);
    screen.getByTestId("gs-continue").click();
    await waitFor(() => expect(screen.queryByTestId("gs-why")).toBeNull());
  });
});
