import { describe, expect, it } from "vitest";
import { DEFAULT_SPEC, STRIP_STEPS, settingsRow, stripRows, type HarderStep } from "../src/stress/breakit";
import { cropFollow, cropWalkingArea, personHeight, replaySentence, type SceneData, type Verdict } from "../src/stress/scene";
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
  it("walking area: covers path and pillars, and a person on the gate line is >= 90 CSS px", () => {
    const d = scene([[150, 40], [330, 200], [480, 300], [620, 420], [760, 560]], [[227, 320, 46, 150]]);
    const paneW = 600;
    const c = cropWalkingArea(d, 1, paneW);
    expect((personHeight(300) * paneW) / c.w).toBeGreaterThanOrEqual(90 - 1e-9);
    expect(c.w / c.h).toBeCloseTo(960 / 540);
    // the hide point (person at the gate, x 480) and the gate line are inside the crop
    expect(c.x).toBeLessThan(480);
    expect(c.x + c.w).toBeGreaterThan(480);
    expect(c.y).toBeLessThan(300);
    expect(c.y + c.h).toBeGreaterThan(300);
    // a wide pane does not need to cut the walking area down as much
    expect(cropWalkingArea(d, 1, 2000).w).toBeGreaterThan(c.w);
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
