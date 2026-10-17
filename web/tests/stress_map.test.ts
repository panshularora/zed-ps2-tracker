import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { AMBER, BLUE, NEUTRAL, borders, cellFill, colourStep, fmtGap, headline, isHatched, type MapCell, type StressMapData } from "../src/stress/model";

const load = (n: string) => JSON.parse(readFileSync(resolve(__dirname, "../public/offline", n), "utf8")) as StressMapData;
const base = load("stress__map__default.json");
const mem6 = load("stress__map__mem6.json");
const c = (diff: number, verdict: MapCell["verdict"]) => ({ diff_mean: diff, verdict });

function synth(v: (n: number, h: number) => MapCell["verdict"]): StressMapData {
  const hs = [0.5, 1, 2, 3], ps = [5, 10];
  return {
    ...base,
    axes: { people: ps, hide_s: hs },
    ours_memory_s: 3,
    cells: ps.flatMap((n) => hs.map((h) => ({ ...base.cells[0], people: n, hide_s: h, verdict: v(n, h), diff_mean: v(n, h) === "ours_better" ? -2 : 1 }))),
  };
}

describe("stress map colour and hatch", () => {
  it("steps: 0 for |gap| = 0, capped at 5 per side", () => {
    expect(colourStep(0)).toBe(0);
    expect(colourStep(0.4)).toBe(1);
    expect(colourStep(-2)).toBe(2);
    expect(colourStep(50)).toBe(5);
    expect(colourStep(-50)).toBe(5);
  });
  it("tie is neutral, blue when ours has fewer ID changes, amber when more", () => {
    expect(cellFill(c(-0.1, "tie")).fill).toBe(NEUTRAL);
    expect(cellFill(c(-23.8, "ours_better"))).toEqual({ fill: BLUE.toUpperCase(), side: "ours", step: 5 });
    expect(cellFill(c(30, "ours_worse"))).toEqual({ fill: AMBER.toUpperCase(), side: "baseline", step: 5 });
    const mid = cellFill(c(1.1, "ours_worse"));
    expect(mid.side).toBe("baseline");
    expect(mid.fill).not.toBe(NEUTRAL);
    expect(mid.fill).not.toBe(AMBER.toUpperCase());
  });
  it("only statistically unclear cells are hatched", () => {
    expect(isHatched({ verdict: "unclear" })).toBe(true);
    for (const v of ["ours_better", "ours_worse", "tie"] as const) expect(isHatched({ verdict: v })).toBe(false);
  });
  it("gap text uses a real minus sign and one decimal", () => {
    expect(fmtGap(-3.45)).toBe("\u22123.5");
    expect(fmtGap(1.05)).toBe("+1.1");
    expect(fmtGap(0.01)).toBe("0.0");
  });
});

describe("borders and headline", () => {
  it("clean borders on a synthetic map", () => {
    const d = synth((_n, h) => (h === 1 || h === 2 ? "ours_better" : h === 3 ? "ours_worse" : "tie"));
    const b = borders(d);
    expect(b.winStart).toEqual([1, 1]);
    expect(b.winEnd).toEqual([3, 3]);
    expect(b.startClean && b.endClean).toBe(true);
    expect(b.startLabel).toBe("Ours starts winning at 1 s hidden, at every crowd size");
    expect(b.endLabel).toBe("Ours stops winning at 3 s hidden, at every crowd size: past it ours is worse in 2 of 2 cells");
    expect(headline(d)).toBe("Ours keeps IDs better when people are hidden 1–2 s. Past 3 s its memory runs out.");
  });
  it("says so in words when there is no single border", () => {
    const d = synth((n, h) => (h >= 1 && (n === 5 ? h <= 1 : h <= 2) ? "ours_better" : "unclear"));
    const b = borders(d);
    expect(b.endClean).toBe(false);
    expect(b.endLabel.startsWith("No single border: ours stops winning between 2 s and 3 s hidden")).toBe(true);
    const never = borders(synth(() => "tie"));
    expect(never.startLabel).toBe("Ours is not clearly better at any hidden time");
  });
  it("default config: computed from the committed grid", () => {
    const b = borders(base);
    expect(headline(base)).toBe("Ours keeps IDs better when people are hidden 1–2.5 s. Past 3 s its memory runs out.");
    expect(b.startLabel).toBe("Ours starts winning at 1 s hidden, at every crowd size");
    expect(b.endLabel).toBe("Ours stops winning at 3 s hidden, at every crowd size: past it ours is worse in 13 of 18 cells");
  });
  it("mem6: computed from the committed grid", () => {
    const b = borders(mem6);
    expect(b.startClean).toBe(true);
    expect(headline(mem6)).toMatch(/^Ours keeps IDs better when people are hidden 1–3 s\./);
    expect(b.endClean).toBe(false);
  });
});
