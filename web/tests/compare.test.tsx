import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { boxesAt, layoutChips, liveCaption, liveRate, momentAt, runStats, hiddenNow } from "../src/compare/model";
import { CompareTable } from "../src/compare/CompareTable";
import { DatasetTable } from "../src/components/DatasetTable";
import { datasetRows, loadSummary } from "../src/lib/summary";
import type { ReplayData } from "../src/guided/model";

const frames = { "1": [[1, 0, 0, 10, 10]], "2": [[1, 10, 0, 10, 10]], "3": [[2, 20, 0, 10, 10]] } as never;
function replay(over: Partial<ReplayData> = {}): ReplayData {
  return {
    id: "x", title: "X", kind: "illustrative", width: 100, height: 50, fps: 10, frames: 60, duration: 6, videoUrl: null,
    runs: { baseline: { label: "ByteTrack", short: "ByteTrack", frames }, ours: { label: "BoT-SORT + visibility (ours)", short: "ours", frames } },
    gate: null,
    events: { baseline: [{ t: "cross", f: 2, id: 1, dir: "in" }, { t: "renumber", f: 3, from: 1, to: 2 }], ours: [{ t: "cross", f: 2, id: 1, dir: "in" }] },
    moments: [{ f: 40, kind: "occlusion", changed: { baseline: true, ours: false }, counted_twice: { baseline: true, ours: false }, focus: [0, 0, 1, 1],
      people: [{ box: [0, 0, 1, 1], ids: { baseline: { before: 2, after: 10 }, ours: { before: 2, after: 2 } } }] }],
    ...over,
  };
}

describe("compare model", () => {
  it("interpolates boxes between tracked frames by id", () => {
    expect(boxesAt(frames, 10, 3, 0.05).boxes[0][1]).toBeCloseTo(5);
    expect(boxesAt(frames, 10, 3, 0).boxes[0][1]).toBe(0);
    expect(boxesAt(frames, 10, 3, 5).f).toBeNull();
  });
  it("chips never overlap", () => {
    const chips = layoutChips([0, 1, 2, 3].map((i) => ({ id: i, text: `#${i}`, cx: 50 + i, top: 100, w: 30 })), 20, 400);
    for (let i = 0; i < chips.length; i++) for (let j = i + 1; j < chips.length; j++) {
      const a = chips[i], b = chips[j];
      expect(a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y).toBe(false);
    }
  });
  it("narrates the moment in plain words, before and after, with honest framing", () => {
    const d = replay();
    expect(momentAt(d, 0)).toBeNull();
    expect(liveCaption(d, 0, 3)).toMatch(/^3 people in view/);
    expect(liveCaption(d, 3.0, 3)).toBe("Person 2 walks behind other people and goes out of sight…");
    const after = liveCaption(d, 4.2, 3);
    expect(after).toContain("ByteTrack calls them #10 (a new person)");
    expect(after).toContain("ours keeps #2");
    expect(after).toContain("the gate would count them twice");
    expect(after).toContain("No ground truth");
    expect(liveCaption(replay({ kind: "measured", moments: [{ ...replay().moments[0], verified: true }] }), 4.2, 1)).toContain("Checked against human labels");
  });
  it("slows to 0.5x only near a moment (never stops)", () => {
    expect(liveRate(replay(), 0)).toBe(1);
    expect(liveRate(replay(), 3.9)).toBe(0.5);
  });
  it("counts per run and lists hidden focus people", () => {
    expect(runStats(replay(), "baseline")).toEqual({ idChanges: 1, crossIn: 1, crossOut: 0, tracks: 2 });
    expect(runStats(replay(), "baseline", 2).idChanges).toBe(0);
    expect(hiddenNow({ "1": [[5, 0, 0, 1, 1]], "2": [] } as never, [], new Set([5]), 2, 10).map((h) => h.id)).toEqual([5]);
  });
  it("per-clip table: no ground truth stays unlabelled; measured shows true count", () => {
    const { unmount } = render(<CompareTable data={replay()} />);
    expect(screen.getByTestId("row-baseline").textContent).toContain("not labelled");
    expect(screen.getByTestId("row-baseline").textContent).toContain("needs ground truth");
    unmount();
    render(<CompareTable data={replay({ kind: "measured" })} truth={{ label: "gt", events: [{ t: "cross", f: 2, id: 9, dir: "in" }, { t: "cross", f: 3, id: 8, dir: "out" }] }} />);
    expect(screen.getByTestId("row-ours").textContent).toMatch(/1 \(1 in · 0 out\)2 \(-1\)/);
  });
});

describe("overall dataset table (summary.json only)", () => {
  it("uses the committed summary: ByteTrack MOT17 reproduction provisional, DanceTrack not run, Pexels no ground truth", () => {
    const rows = datasetRows(loadSummary());
    const by = Object.fromEntries(rows.map((r) => [r.id, r]));
    expect(rows.map((r) => r.id)).toEqual(["mot17", "mot20", "dancetrack", "pexels", "synthetic"]);
    expect(by.mot17.cells.bytetrack.tag).toBe("provisional");
    expect(by.mot17.cells.bytetrack.values).toEqual([{ label: "MOTA", value: "66.4" }, { label: "IDF1", value: "73.8" }, { label: "IDSW", value: "165" }]);
    expect(by.mot17.cells.ours.tag).toBe("ran, no metric in summary.json");
    expect(by.mot20.status).toContain("1 leaked, excluded");
    expect(by.dancetrack.cells.botsort.tag).toBe("not run yet");
    expect(by.pexels.cells.ours.tag).toBe("no ground truth");
    expect(by.synthetic.cells.ours.tag).toBe("not run yet");
  });
  it("never shows a number that is not in the file", () => {
    const rows = datasetRows({ ran: [{ id: "mot17_val_half", cache: "c", saw_eval_half: false }] });
    expect(rows.every((r) => Object.values(r.cells).every((c) => !c.values))).toBe(true);
    render(<DatasetTable summary={null} />);
    expect(screen.getByTestId("ds-mot17").textContent).toContain("not run yet");
  });
});
