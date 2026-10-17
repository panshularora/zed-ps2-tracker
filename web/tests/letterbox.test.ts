import { describe, expect, it } from "vitest";
import { letterbox, sourceToStage, stageAspect, stageToSource } from "../src/lib/letterbox";

describe("letterbox (object-fit: contain) content rectangle", () => {
  it("portrait 1080x1920 in a 16:9 stage: full height, centred with side bars, uniform scale", () => {
    const r = letterbox(1600, 900, 1080, 1920);
    expect(r.s).toBeCloseTo(900 / 1920);
    expect(r.h).toBeCloseTo(900);
    expect(r.w).toBeCloseTo(1080 * (900 / 1920)); // 506.25
    expect(r.x).toBeCloseTo((1600 - 506.25) / 2);
    expect(r.y).toBe(0);
  });
  it("wide 2.39:1 source in a 16:9 stage: full width, bars top and bottom", () => {
    const r = letterbox(1600, 900, 2390, 1000);
    expect(r.w).toBeCloseTo(1600);
    expect(r.x).toBe(0);
    expect(r.y).toBeCloseTo((900 - 1000 * (1600 / 2390)) / 2);
  });
  it("same aspect: fills the stage exactly", () => {
    expect(letterbox(1280, 720, 1920, 1080)).toEqual({ x: 0, y: 0, w: 1280, h: 720, s: 1280 / 1920 });
  });
  it("boxes and the gate map through the same rect; clicks on the bars hit nothing; round trip", () => {
    const r = letterbox(1600, 900, 1080, 1920);
    // a gate across the full frame width at y = 960 lands exactly on the picture edges
    const [ax, ay] = sourceToStage(r, 0, 960), [bx] = sourceToStage(r, 1080, 960);
    expect(ax).toBeCloseTo(r.x); expect(bx).toBeCloseTo(r.x + r.w); expect(ay).toBeCloseTo(450);
    const box: [number, number] = [540, 1500];
    const p = sourceToStage(r, ...box);
    expect(stageToSource(r, ...p)![0]).toBeCloseTo(540);
    expect(stageToSource(r, ...p)![1]).toBeCloseTo(1500);
    expect(stageToSource(r, 10, 450)).toBeNull(); // left bar
    expect(stageToSource(r, 1590, 450)).toBeNull(); // right bar
    // with a zoom view origin
    expect(stageToSource(r, r.x, 0, 100, 200)).toEqual([100, 200]);
  });
  it("degenerate sizes do not throw or divide by zero", () => {
    expect(letterbox(0, 0, 1080, 1920)).toEqual({ x: 0, y: 0, w: 0, h: 0, s: 1 });
    expect(letterbox(800, 450, 0, 0).s).toBe(1);
  });
  it("stage aspect: landscape keeps its own, portrait and 4:3 get a 16:9 stage with bars", () => {
    expect(stageAspect(1920, 1080)).toBeCloseTo(16 / 9);
    expect(stageAspect(2390, 1000)).toBeCloseTo(2.39);
    expect(stageAspect(1080, 1920)).toBeCloseTo(16 / 9);
    expect(stageAspect(640, 480)).toBeCloseTo(16 / 9);
  });
});
