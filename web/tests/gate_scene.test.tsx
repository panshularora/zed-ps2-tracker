import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, afterEach } from "vitest";
import { involvedBetween, isDefault, lineSplitter, readStream, recentEvents, streamUrl, DEFAULTS, type SimFrame } from "../src/lib/gateSim";
import GateScene from "../src/pages/GateScene";

const pane = (events: [number, number, number, number, number][] = [], counted = 0) => ({ ids: [[1, 4]] as [number, number][], extra: [], events, counted, id_changes: events.length });
const frame = (f: number, ev: [number, number, number, number, number][] = []): SimFrame => ({ type: "frame", f, people: [[1, 10, 10, 1]], dets: [], true_count: 1, bytetrack: pane(ev, 2), ours: pane() });
const meta = { type: "meta", width: 800, height: 450, gate: [[0, 225], [800, 225]], fps: 30, params: { seed: 7 }, trackers: { bytetrack: { label: "ByteTrack", config: "c" }, ours: { label: "BoT-SORT + visibility (ours)", config: "c" } }, max_people: 40 };

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
    expect(streamUrl(DEFAULTS)).toBe("/api/gate-sim/stream?seed=7&density=0.5&occlusion=1&noise=0.3&seconds=60");
    expect(isDefault({ ...DEFAULTS, seed: 8 })).toBe(false);
  });
  it("finds the people involved in recent ID changes", () => {
    const fs = [frame(1), frame(2, [[1, 4, 9, 10, 10]]), frame(3)];
    expect([...involvedBetween(fs, 1, 3)]).toEqual([1]);
    expect(recentEvents(fs, 2, "bytetrack", 5).map((x) => x.age)).toEqual([1]);
    expect(recentEvents(fs, 2, "ours", 5)).toEqual([]);
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
    expect(screen.getByTestId("seed-chip").textContent).toBe("seed 7");
    expect(screen.getByTestId("gate-caption").textContent).toBe("Left: ByteTrack, the published tracker we start from. Right: ours = ByteTrack + rules that keep a hidden person's number.");
    expect(screen.getByTestId("scene-counters-bytetrack").textContent).toContain("True count");
  });
  it("falls back to the recorded run when the backend is offline", async () => {
    const body = [meta, frame(1)].map((x) => JSON.stringify(x)).join("\n") + "\n";
    const f = vi.fn(async (u: string) => (u.startsWith("/api/") ? new Response("x", { status: 503 }) : new Response(body)));
    vi.stubGlobal("fetch", f);
    render(<GateScene />);
    await waitFor(() => expect(screen.getByText(/replaying a recorded run of the real trackers/)).toBeTruthy());
    expect(f.mock.calls.map((c) => String(c[0]))).toContain("/offline/gate-sim.ndjson");
  });
});
