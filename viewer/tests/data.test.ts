import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  base64ToBytes,
  clampFrame,
  gateCounts,
  loadClipGz,
  openPending,
  parseClip,
  seekFrameForEvent,
  type Clip,
} from "../src/data";
import { ID_COLORS, idColor } from "../src/theme";

const clip: Clip = {
  clip: "MOT17-04",
  tracker: "bytetrack",
  fps: 30,
  frames: 100,
  width: 1920,
  height: 1080,
  gate: { a: [0, 648], b: [1920, 648] },
  tracks: { "1": [[1, 10, 20, 30, 80, 0.9], [2, 12, 20, 30, 80, 0.8]], "7": [[2, 500, 20, 30, 80, 1]] },
  pending: [{ f0: 10, f1: 15, candidates: [1, 7], chosen: 7 }],
  events: [
    { f: 40, type: "cross", id: 1, dir: "in", correct: true },
    { f: 60, type: "cross", id: 7, dir: "out", correct: false },
    { f: 50, type: "id_switch", gt: 3, from: 1, to: 7 },
    { f: 61, type: "tailgate", ids: [1, 7] },
  ],
};

describe("parsing", () => {
  it("indexes boxes by frame", () => {
    const c = parseClip(clip);
    expect(c.byFrame.get(1)?.map((b) => b.id)).toEqual([1]);
    expect(c.byFrame.get(2)?.map((b) => b.id).sort()).toEqual([1, 7]);
    expect(c.byFrame.get(3)).toBeUndefined();
  });
  it("rejects malformed clips", () => {
    expect(() => parseClip({ ...clip, fps: 0 })).toThrow(/fps/);
    expect(() => parseClip({ ...clip, tracks: { "1": [[1, 2, 3]] } })).toThrow(/row/);
    expect(() => parseClip({ ...clip, tracks: { "1": [[101, 0, 0, 1, 1, 1]] } })).toThrow(/range/);
    expect(() => parseClip({ ...clip, events: [{ f: 1, type: "x" }] })).toThrow(/event/);
  });
  it("round-trips gzip + base64 through DecompressionStream", async () => {
    const gz = gzipSync(Buffer.from(JSON.stringify(clip)));
    const c = await loadClipGz(base64ToBytes(gz.toString("base64")));
    expect(c.tracker).toBe("bytetrack");
    expect(c.byFrame.size).toBe(2);
  });
});

describe("seeking and overlays", () => {
  it("jumps 2 s before an event, clamped", () => {
    expect(seekFrameForEvent(100, 30, 100)).toBe(40);
    expect(seekFrameForEvent(20, 30, 100)).toBe(1);
    expect(seekFrameForEvent(50, 14, 100)).toBe(22);
    expect(clampFrame(0, 100)).toBe(1);
    expect(clampFrame(250, 100)).toBe(100);
  });
  it("counts gate crossings up to the frame", () => {
    expect(gateCounts(clip.events, 39)).toEqual({ in: 0, out: 0 });
    expect(gateCounts(clip.events, 40)).toEqual({ in: 1, out: 0 });
    expect(gateCounts(clip.events, 100)).toEqual({ in: 1, out: 1 });
  });
  it("shows a held decision only inside its window", () => {
    expect(openPending(clip.pending, 9)).toHaveLength(0);
    expect(openPending(clip.pending, 10)).toHaveLength(1);
    expect(openPending(clip.pending, 15)).toHaveLength(1);
    expect(openPending(clip.pending, 16)).toHaveLength(0);
  });
  it("uses the spec palette by id % 20, with no red", () => {
    expect(ID_COLORS).toHaveLength(20);
    expect(idColor(0)).toBe("#E69F00");
    expect(idColor(21)).toBe("#56B4E9");
    expect(ID_COLORS).not.toContain("#E5484D");
  });
});

describe("media hints", () => {
  it("accepts video / frames media and rejects malformed ones", () => {
    const ok = parseClip({ ...clip, media: { type: "video", src: "MOT17-04.mp4", frame_offset: 526 } });
    expect(ok.media?.frame_offset).toBe(526);
    expect(parseClip(clip).media).toBeNull();
    expect(() => parseClip({ ...clip, media: { type: "gif", src: "x" } })).toThrow(/media/);
    expect(() => parseClip({ ...clip, media: { type: "frames", src: "f/%06d.jpg", frame_offset: 1.5 } })).toThrow(
      /frame_offset/,
    );
  });
});

describe("decision fade window", () => {
  it("lists windows closed within the fade", async () => {
    const { closingPending } = await import("../src/data");
    const p = [{ f0: 10, f1: 15, candidates: [1, 7], chosen: 7 }];
    expect(closingPending(p, 15, 4)).toEqual([]);
    expect(closingPending(p, 16, 4)).toHaveLength(1);
    expect(closingPending(p, 19, 4)).toHaveLength(1);
    expect(closingPending(p, 20, 4)).toEqual([]);
  });
});
