import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, afterEach } from "vitest";
import { beatCopy, firstKeepId, involvedBetween, isDefault, lineSplitter, personIdLabel, readStream, recentEvents, recordingUrl, streamUrl, DEFAULTS, type SimFrame } from "../src/lib/gateSim";
import GateScene, { hideRangeText, WORLD_FIX_TEXT } from "../src/pages/GateScene";
import { measuredHideS } from "../src/lib/gateSim";

const pane = (events: [number, number, number, number, number][] = [], counted = 0) => ({ ids: [[1, 4]] as [number, number][], extra: [], held: [] as [number, number, number][], events, counted, id_changes: events.length });
const frame = (f: number, ev: [number, number, number, number, number][] = []): SimFrame => ({ type: "frame", f, people: [[1, 10, 10, 1]], dets: [], true_count: 1, beat: null, focus: null, bytetrack: pane(ev, 2), ours: pane() });
const meta = { type: "meta", width: 800, height: 450, gate: [[0, 225], [800, 225]], fps: 30, params: { seed: 7, play: "keep" }, trackers: { bytetrack: { label: "ByteTrack", config: "c" }, ours: { label: "Ours · BoT-SORT + visibility", config: "configs/ours/visibility_botsort.yaml" } }, max_people: 40, play: "keep" };

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

describe("gate scene hide range line", () => {
  it("says whether the hide is inside the Stress Map winning range", () => {
    expect(hideRangeText(1.5, 45)).toBe("#1 is hidden for 1.5 s (45 frames, measured), inside the 1–2.5 s range where our rule wins on the Stress Map.");
    expect(hideRangeText(3.5)).toMatch(/outside the 1–2.5 s range/);
    expect(WORLD_FIX_TEXT).toMatch(/instead of jumping sideways/);
    expect(WORLD_FIX_TEXT).toMatch(/trackers were not changed/);
  });
  it("measures the hide from the frames, only once the person is seen again", () => {
    const at = (f: number, vis: number): SimFrame => ({ ...frame(f), people: [[1, 10, 10, vis]] });
    expect(measuredHideS([at(1, 1), at(2, 0), at(3, 0)], 1, 30)).toBeNull();
    expect(measuredHideS([at(1, 1), at(2, 0), at(3, 0), at(4, 1)], 1, 2)).toBe(1);
    expect(measuredHideS([at(1, 1)], 2, 30)).toBeNull();
  });
  it("the committed keep recording hides #1 for 45 frames = 1.5 s, inside 1–2.5 s", async () => {
    const { readFileSync } = await import("node:fs");
    const { resolve } = await import("node:path");
    const lines = readFileSync(resolve(__dirname, "../public/offline/gate-keep.ndjson"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    const s = measuredHideS(lines.filter((l) => l.type === "frame"), lines[0].focus, lines[0].fps);
    expect(s).toBeCloseTo(1.5, 6);
    expect(hideRangeText(s as number, 45)).toMatch(/inside the 1–2.5 s range/);
  });
});

describe("keep recording (real trackers, no relabelling)", () => {
  it("ours keeps #1 on one number; ByteTrack issues a new one", async () => {
    const { readFileSync } = await import("node:fs");
    const { resolve } = await import("node:path");
    const lines = readFileSync(resolve(__dirname, "../public/offline/gate-keep.ndjson"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
    const ids = (pane: "ours" | "bytetrack") => [...new Set(lines.filter((l) => l.type === "frame").map((l) => new Map(l[pane].ids as [number, number][]).get(1)).filter((v) => v != null))];
    expect(lines[0].focus).toBe(1);
    expect(lines[0].trackers.ours.config).toBe("configs/ours/visibility_botsort.yaml");
    expect(ids("ours")).toHaveLength(1);
    expect(ids("bytetrack").length).toBeGreaterThan(1);
  });
});

describe("gate scene page", () => {
  it("streams real tracker output into two panes with counters", async () => {
    const body = [meta, frame(1), frame(2, [[1, 4, 9, 10, 10]]), { type: "end", frames: 2, true_count: 1, summary: {} }].map((x) => JSON.stringify(x)).join("\n") + "\n";
    const f = vi.fn(async (_u: string) => new Response(body, { headers: { "content-type": "application/x-ndjson" } }));
    vi.stubGlobal("fetch", f);
    render(<GateScene />);
    expect(screen.getByTestId("synthetic-tag").textContent).toBe("Synthetic scene · exact ground truth");
    // the page also hosts Break it yourself (POST /api/break/run); find the gate stream call
    await waitFor(() => expect(f.mock.calls.some((c) => String(c[0]).includes("/api/gate-sim/stream"))).toBe(true));
    expect(f.mock.calls.map((c) => String(c[0])).find((u) => u.includes("/api/gate-sim/stream"))).toContain("/api/gate-sim/stream?seed=7");
    await waitFor(() => expect(screen.getByText("ByteTrack")).toBeTruthy());
    expect(screen.getByTestId("seed-chip").textContent).toContain("seed 7");
    expect(screen.getByTestId("gate-caption").textContent).toBe("Left: ByteTrack, the published tracker we start from. Right: ours = BoT-SORT + a rule that remembers a hidden person longer.");
    expect(screen.getByTestId("scene-counters-bytetrack").textContent).toContain("True people");
    expect(screen.getByTestId("play-keep").textContent).toMatch(/one body, one number/i);
    expect(screen.getByTestId("how-id-kept").textContent).toContain("HELD #N");
    expect(screen.getByTestId("how-id-kept").textContent).toContain("NEW #M was #N");
    expect(screen.getByText(/Ours · BoT-SORT \+ visibility/)).toBeTruthy();
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
