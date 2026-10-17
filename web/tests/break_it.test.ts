import { describe, expect, it } from "vitest";
import { DEFAULT_SPEC, fromQuery, harderDone, harderText, heldUntil, simplifyPath, toQuery, type BreakSpec, type HarderStep } from "../src/stress/breakSpec";
import { countsAt, hideWindow, verdict, type SceneData } from "../src/stress/scene";

describe("shareable URL", () => {
  it("round-trips seed, objects, path, hidden person, config and frame exactly", () => {
    const spec: BreakSpec = { seed: 4242, hide_s: 2.5, hide_pid: 3, crowd: 5, pillars: [[470, 340], [120, 500]], groups: [[640, 260]], path: [[10, 20], [300, 310], [700, 520]], config: "mem6" };
    const q = toQuery({ spec, frame: 173 });
    expect(fromQuery(q)).toEqual({ spec, frame: 173 });
    expect(fromQuery(`?${q}`)).toEqual({ spec, frame: 173 });
    expect(toQuery(fromQuery(q)!)).toBe(q);
  });
  it("default scene and 'no hidden person' survive the trip", () => {
    expect(fromQuery(toQuery({ spec: DEFAULT_SPEC, frame: 1 }))).toEqual({ spec: DEFAULT_SPEC, frame: 1 });
    const none = { ...DEFAULT_SPEC, hide_pid: null, pillars: [], groups: [] };
    expect(fromQuery(toQuery({ spec: none, frame: 9 }))).toEqual({ spec: none, frame: 9 });
  });
  it("ignores unrelated or malformed query strings", () => {
    expect(fromQuery("?tab=results")).toBeNull();
    const s = fromQuery("seed=3&path=1,2;x,y;5,6&hide=99&crowd=-4&f=-2")!;
    expect(s.spec.path).toEqual([[1, 2], [5, 6]]);
    expect(s.spec.hide_s).toBe(8);
    expect(s.spec.crowd).toBe(0);
    expect(s.frame).toBe(1);
  });
  it("freehand strokes are thinned but keep their end point", () => {
    const raw: [number, number][] = Array.from({ length: 50 }, (_, i) => [i * 2, i * 2]);
    const p = simplifyPath(raw);
    expect(p[0]).toEqual([0, 0]);
    expect(p[p.length - 1]).toEqual([98, 98]);
    expect(p.length).toBeLessThan(raw.length);
  });
});

/** Minimal scene: person 1 hidden in frames 4-6, ids per frame for both trackers. */
function scene(bt: (number | null)[], ours: (number | null)[]): SceneData {
  const n = bt.length;
  const matched = (ids: (number | null)[]) => ids.map((t) => (t == null ? {} : { "1": t }));
  const pane = (ids: (number | null)[], crossings: number[]) => ({ label: "", config: "", tracks: ids.map(() => []), matched: matched(ids), events: [], crossings, count: crossings.length, true_count: 1 });
  return {
    seed: 1, settings: {}, width: 960, height: 540, gate_y: 300, fps: 1,
    truth: Array.from({ length: n }, (_, k) => ({ f: k + 1, p: [[1, 100, 100, 1]], hidden: k + 1 >= 4 && k + 1 <= 6 ? [1] : [] })),
    true_crossings: [5],
    panes: { bytetrack: pane(bt, []), ours: pane(ours, [7]) },
  } as SceneData;
}

describe("verdict at reappearance", () => {
  it("kept / changed / lost", () => {
    const d = scene([3, 3, 3, null, null, null, 9, 9, 9, 9], [5, 5, 5, null, null, null, null, 5, 5, 5]);
    expect(hideWindow(d, 1)).toEqual([4, 6]);
    expect(verdict(d, "bytetrack", 1)).toEqual({ kind: "changed", from: 3, to: 9, frame: 7, text: "#3 \u2192 #9" });
    expect(verdict(d, "ours", 1)).toEqual({ kind: "kept", id: 5, frame: 8, text: "Kept #5" });
    const lost = scene([3, 3, 3, null, null, null, null, null, null, null], [5, 5, 5, null, null, null, 5, 5, 5, 5]);
    expect(verdict(lost, "bytetrack", 1, 3)?.kind).toBe("lost");
  });
  it("no verdict without a hide or without a number before it", () => {
    const d = scene([null, null, null, null, null, null, 2, 2, 2, 2], [1, 1, 1, null, null, null, 1, 1, 1, 1]);
    expect(verdict(d, "bytetrack", 1)).toBeNull();
    expect(verdict(d, "ours", 2)).toBeNull();
  });
  it("counts so far: truth and each tracker", () => {
    const d = scene([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    expect(countsAt(d, 4)).toEqual({ truth: 0, bytetrack: 0, ours: 0 });
    expect(countsAt(d, 7)).toEqual({ truth: 1, bytetrack: 0, ours: 1 });
  });
});

describe("make it harder", () => {
  const k = (h: number) => ({ kind: "kept" as const, id: 1, frame: h, text: "Kept #1" });
  const x = (h: number) => ({ kind: "changed" as const, from: 1, to: 2, frame: h, text: "#1 \u2192 #2" });
  it("reports the longest hidden time each tracker held", () => {
    const steps: HarderStep[] = [
      { hide_s: 0.5, v: { ours: k(1), bytetrack: k(1) } },
      { hide_s: 1, v: { ours: k(1), bytetrack: x(1) } },
      { hide_s: 1.5, v: { ours: k(1), bytetrack: x(1) } },
      { hide_s: 2, v: { ours: x(1), bytetrack: x(1) } },
    ];
    expect(heldUntil(steps, "ours")).toEqual({ held: 1.5, brokeAt: 2 });
    expect(heldUntil(steps, "bytetrack")).toEqual({ held: 0.5, brokeAt: 1 });
    expect(harderDone(steps)).toBe(false); // both broken only at the last step
    expect(harderDone([...steps, { hide_s: 2.5, v: { ours: x(1), bytetrack: x(1) } }])).toBe(true);
    expect(harderText(steps)).toBe("Ours held until 1.5 s hidden, ByteTrack held until 0.5 s hidden.");
  });
  it("flags a tracker that keeps the number again after breaking", () => {
    const steps: HarderStep[] = [
      { hide_s: 0.5, v: { ours: k(1), bytetrack: k(1) } },
      { hide_s: 1, v: { ours: x(1), bytetrack: x(1) } },
      { hide_s: 1.5, v: { ours: k(1), bytetrack: x(1) } },
    ];
    expect(harderText(steps)).toBe("Ours held until 0.5 s hidden, ByteTrack held until 0.5 s hidden. Not monotonic: ours kept it again at 1.5 s.");
  });
  it("broke at the first step / never broke", () => {
    const steps: HarderStep[] = [{ hide_s: 0.5, v: { ours: k(1), bytetrack: x(1) } }, { hide_s: 1, v: { ours: k(1), bytetrack: x(1) } }];
    expect(harderText(steps)).toBe("Ours held through 1 s (the longest tried), ByteTrack broke already at 0.5 s.");
  });
});
