import { describe, expect, it } from "vitest";
import { DEFAULT_SPEC, STRIP_STEPS, settingsRow, stripRows, type HarderStep } from "../src/stress/breakit";
import { cropAt, cropFollow, cropWalkingArea, lostLine, personHeight, replaySentence, wholeSceneText, type SceneData, type Verdict } from "../src/stress/scene";
import { flagOn } from "../src/lib/flags";

const kept = (id: number): Verdict => ({ kind: "kept", id, frame: 10, text: `Kept #${id}` });
const changed = (a: number, b: number): Verdict => ({ kind: "changed", from: a, to: b, frame: 10, text: `#${a} → #${b}` });

function scene(points: [number, number][], pillars: [number, number, number, number][] = []): SceneData {
  const truth = points.map(([x, y], i) => ({ f: i + 1, p: [[1, x, y, 1]] as [number, number, number, number][], hidden: i >= 2 && i <= 3 ? [1] : [] }));
  const pane = { label: "", config: "", tracks: [], matched: [], events: [], crossings: [], count: 0, true_count: 0 };
  return { seed: 1, settings: {}, width: 960, height: 540, gate_y: 300, fps: 30, truth, true_crossings: [], panes: { bytetrack: pane, ours: pane }, pillars };
}

describe("replay sentence", () => {
  it("says which tracker gave the hidden person a new number", () => {
    expect(replaySentence(10, 1, 1, { bytetrack: changed(1, 7), ours: kept(1) })).toBe("10 people, person 1 hidden 1 s. ByteTrack gave them a new number, ours didn't.");
    expect(replaySentence(5, 3, 2.5, { bytetrack: kept(3), ours: changed(3, 9) })).toBe("5 people, person 3 hidden 2.5 s. Ours gave them a new number, ByteTrack didn't.");
    expect(replaySentence(5, 3, 4, { bytetrack: changed(3, 8), ours: changed(3, 9) })).toBe("5 people, person 3 hidden 4 s. Both trackers gave them a new number.");
    expect(replaySentence(5, 3, 0.5, { bytetrack: kept(3), ours: kept(3) })).toBe("5 people, person 3 hidden 0.5 s. Both trackers kept their number.");
    expect(replaySentence(20, 4, null, {}, "ours")).toBe("20 people, following person 4. Ours changed their number first.");
  });
});

describe("pane crops", () => {
  it("follow: 2x zoom centred on the person, clamped inside the scene", () => {
    const d = scene([[480, 300], [480, 300], [480, 300], [480, 300], [900, 520]]);
    const c = cropFollow(d, 1, 1, 2);
    expect([c.w, c.h]).toEqual([480, 270]);
    expect(c.x + c.w / 2).toBeCloseTo(480);
    expect(c.y + c.h / 2).toBeCloseTo(300 - personHeight(300) / 2);
    const edge = cropFollow(d, 1, 5, 2);
    expect(edge.x + edge.w).toBeLessThanOrEqual(960);
    expect(edge.y + edge.h).toBeLessThanOrEqual(540);
  });
  it("walking area: every path (whole bodies) and pillar inside, plus 10 % padding", () => {
    const d = scene([[150, 40], [330, 200], [480, 300], [620, 420], [830, 240]], [[227, 320, 46, 150]]);
    const c = cropWalkingArea(d);
    expect(c.w / c.h).toBeCloseTo(960 / 540);
    for (const t of d.truth) for (const [, x, y] of t.p) {
      expect(x).toBeGreaterThan(c.x); expect(x).toBeLessThan(c.x + c.w);
      expect(Math.max(0, y - personHeight(y))).toBeGreaterThanOrEqual(c.y); expect(y).toBeLessThanOrEqual(c.y + c.h);
    }
    expect(c.x).toBeLessThan(227); expect(c.y + c.h).toBeGreaterThan(470);
    // paths over the whole frame: the crop is the whole scene, never bigger
    const big = cropWalkingArea(scene([[10, 20], [950, 535], [500, 300], [500, 300], [500, 300]]));
    expect(big).toEqual({ x: 0, y: 0, w: 960, h: 540 });
    // a small walking area zooms in
    const small = cropWalkingArea(scene([[400, 280], [430, 300], [460, 320], [480, 330], [500, 340]]));
    expect(small.w).toBeLessThan(960);
  });
  it("pause crop: 1.5x on the person, not clamped (padding past the scene edge)", () => {
    const d = scene([[30, 520], [30, 520], [30, 520], [30, 520], [30, 520]]);
    const c = cropAt(d, 1, 3, 1.5);
    expect([c.w, c.h]).toEqual([640, 360]);
    expect(c.x + c.w / 2).toBeCloseTo(30);
    expect(c.x).toBeLessThan(0);
    expect(c.y + c.h).toBeGreaterThan(540);
  });
});

describe("whole-scene counts", () => {
  const base = scene([[480, 300], [480, 300], [480, 300], [480, 300], [480, 300]]);
  const withPane = (missed: [number, number, number][], extra = 0): SceneData => ({
    ...base,
    true_crossings_by: [[50, 2, 1], [120, 1, 1], [300, 3, -1]],
    panes: { ...base.panes, bytetrack: { ...base.panes.bytetrack, missed, extra, count: 3 - missed.length + extra, true_count: 3 } },
  });
  it("label and computed line with the focus crossing's place among the lost ones", () => {
    expect(wholeSceneText({ count: 2, true_count: 10 })).toBe("Whole scene: counted 2 of 10");
    expect(lostLine(withPane([[120, 1, 1], [300, 3, -1]]), "bytetrack", 1, 118)).toBe("ByteTrack lost 2 crossings across the scene; this is the first.");
    expect(lostLine(withPane([[50, 2, 1], [120, 1, 1]]), "bytetrack", 1, 118)).toBe("ByteTrack lost 2 crossings across the scene; this is the second.");
    expect(lostLine(withPane([[300, 3, -1]]), "bytetrack", 1, 118)).toBe("ByteTrack lost 1 crossing across the scene; it counted this one.");
    expect(lostLine(withPane([], 1), "bytetrack", 1, 118)).toBe("ByteTrack counted every real crossing in the scene. 1 of its counts matches no one's real crossing.");
    expect(lostLine(base, "bytetrack", 1, 118)).toBeNull(); // no per-person data: say nothing
  });
});

describe("make it harder strip and flag", () => {
  it("two rows, one cell per hidden time 0.5-2.5 s: kept / changed / pending", () => {
    const steps: HarderStep[] = [
      { hide_s: 0.5, v: { bytetrack: kept(1), ours: kept(1) } },
      { hide_s: 1, v: { bytetrack: changed(1, 5), ours: kept(1) } },
      { hide_s: 1.5, v: { bytetrack: { kind: "lost", from: 1, frame: 9, text: "#1 → no number" }, ours: null } },
    ];
    const r = stripRows(steps);
    expect(STRIP_STEPS).toEqual([0.5, 1, 1.5, 2, 2.5]);
    expect(r.bytetrack.map((c) => c.state)).toEqual(["kept", "changed", "changed", "pending", "pending"]);
    expect(r.ours.map((c) => c.state)).toEqual(["kept", "kept", "none", "pending", "pending"]);
  });
  it("Make it harder is off by default", () => {
    expect(flagOn("harder", undefined, "")).toBe(false);
    expect(flagOn("harder", "", "?seed=7")).toBe(false);
    expect(flagOn("harder", "harder", "")).toBe(true);
    expect(flagOn("harder", undefined, "?ff=x,harder")).toBe(true);
  });
  it("settings are one plain row", () => {
    expect(settingsRow(DEFAULT_SPEC)).toBe("Seed 7 · person 1 hidden 1.5 s at the gate · 1 pillar · 1 group · 0 passers-by · ours on default settings");
  });
});

describe("cell replay paused frame: other people dimmed, caption says so", () => {
  it("counts people other than the focus inside the crop", async () => {
    const { othersInView, pausedNote } = await import("../src/stress/scene");
    const d = { truth: [{ p: [[1, 100, 300, 1], [2, 120, 300, 1], [3, 900, 300, 1]], hidden: [] }] } as never;
    expect(othersInView(d, 1, { x: 0, y: 100, w: 300, h: 300 }, 1)).toBe(1);
    expect(othersInView(d, 1, { x: 0, y: 100, w: 960, h: 300 }, 1)).toBe(2);
    expect(pausedNote(1, 1)).toBe("Paused: person 1 ringed; the other 1 person in view is dimmed.");
    expect(pausedNote(4, 3)).toBe("Paused: person 4 ringed; the other 3 people in view are dimmed.");
    expect(pausedNote(4, 0)).toBe("Paused: person 4 ringed; nobody else is in this view.");
  });
});
