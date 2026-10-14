import { describe, expect, it } from "vitest";
import { paint } from "../src/guided/draw";
import { neutralColor } from "../src/lib/scenarios";
import type { ReplayData } from "../src/guided/model";

function fakeCtx() {
  const calls: { op: string; args: unknown[]; stroke?: string; fill?: string; alpha: number; lw: number; dash: number[] }[] = [];
  let m = { scale: () => m, translate: () => m };
  const st = { strokeStyle: "", fillStyle: "", globalAlpha: 1, lineWidth: 1, dash: [] as number[] };
  const rec = (op: string) => (...args: unknown[]) => { calls.push({ op, args, stroke: st.strokeStyle, fill: st.fillStyle, alpha: st.globalAlpha, lw: st.lineWidth, dash: st.dash }); };
  const ctx = new Proxy(st as Record<string, unknown>, {
    get(t, k: string) {
      if (k in t) return t[k];
      if (k === "getTransform") return () => m;
      if (k === "setLineDash") return (d: number[]) => { st.dash = d; };
      if (k === "measureText") return (s: string) => ({ width: s.length * 8 });
      return rec(k);
    },
    set(t, k: string, v) { t[k] = v; return true; },
  });
  m = { scale: () => m, translate: () => m };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
}

const data = {
  width: 100, height: 100, fps: 10, gate: null,
  runs: { ours: { frames: { "5": [[1, 10, 10, 20, 40], [3, 50, 10, 20, 40]], "1": [[2, 70, 50, 10, 20]] } }, baseline: { frames: {} } },
  events: { ours: [], baseline: [] },
} as unknown as ReplayData;

describe("guided boxes", () => {
  it("draws every box 2px in its ID colour with an ID chip", () => {
    const { ctx, calls } = fakeCtx();
    paint(ctx, 100, 100, data, null, { run: "ours", f: 5, view: [0, 0, 100, 100], spot: new Set([1]), dim: 0, gatePulse: 0 });
    const rects = calls.filter((c) => c.op === "strokeRect");
    expect(rects.map((r) => [r.stroke, r.lw])).toEqual(expect.arrayContaining([[neutralColor(1), 2], [neutralColor(3), 2]]));
    const texts = calls.filter((c) => c.op === "fillText").map((c) => c.args[0]);
    expect(texts).toEqual(expect.arrayContaining(["Person 1", "3"]));
  });
  it("draws a hidden person as a dashed box in their ID colour, chip at 60 %", () => {
    const { ctx, calls } = fakeCtx();
    paint(ctx, 100, 100, data, null, { run: "ours", f: 5, view: [0, 0, 100, 100], spot: new Set([2]), dim: 0, gatePulse: 0 });
    const ghost = calls.find((c) => c.op === "strokeRect" && c.dash.length > 0)!;
    expect(ghost.stroke).toBe(neutralColor(2));
    const chip = calls.find((c) => c.op === "fillText" && c.args[0] === "2 · hidden")!;
    expect(chip.alpha).toBeCloseTo(0.6);
    expect(neutralColor(2)).not.toBe("#009E73");
  });
});
