import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { boxesAt, people, pickFocus, placeChips, storySteps, summaryText, tracksOf, watchLine } from "../src/identity/story";
import { PersonTimeline } from "../src/identity/PersonTimeline";
import { CountBar } from "../src/identity/IdentityPlayer";
import { orderScenarios } from "../src/pages/Scenarios";
import type { ReplayData } from "../src/guided/model";
import type { Reappearance } from "../src/identity/verify";

const rp = (id: number, last: number, back: number, verdict: Reappearance["verdict"]): Reappearance => ({
  id, run: "ours", last, back, gap_frames: back - last - 1, gap_s: (back - last - 1) / 10, verdict,
  app: verdict === "kept" ? 0.93 : 0.6, motion_err: 0.1, motion_tol: 1, size_ratio: 1.1, gt_before: null, gt_after: null,
  reasons: verdict === "kept" ? [] : ["appearance differs (0.60 < 0.8)"], occluder: null,
});
const keptV = (id: number, back: number) => (id === 4 && back === 16 ? rp(4, 10, 16, "kept") : undefined);
const swapV = (id: number, back: number) => (id === 4 && back === 16 ? rp(4, 10, 16, "swap") : undefined);

// #4: frames 1-10, hidden 11-15, back 16-20. #7: 1-5 then renumbered to #8 at 6-20. #9: one-frame flicker.
function replay(): ReplayData {
  const frames: Record<string, [number, number, number, number, number][]> = {};
  for (let f = 1; f <= 20; f++) {
    const bs: [number, number, number, number, number][] = [];
    if (f <= 10 || f >= 16) bs.push([4, f * 10, 100, 40, 80]);
    bs.push([f <= 5 ? 7 : 8, 500, 100, 40, 80]);
    if (f === 3) bs.push([9, 900, 100, 40, 80]);
    frames[String(f)] = bs;
  }
  return {
    id: "t", title: "t", kind: "illustrative", width: 1000, height: 500, fps: 10, frames: 20, duration: 2, videoUrl: null,
    runs: { baseline: { label: "ByteTrack", short: "ByteTrack", frames: {} }, ours: { label: "ours", short: "ours", frames } },
    gate: null, moments: [],
    events: { baseline: [], ours: [{ t: "cross", f: 18, id: 4, dir: "in" }, { t: "renumber", f: 6, from: 7, to: 8 }] },
  };
}

describe("identity story", () => {
  it("people are ID chains from real tracks; summary counts kept vs changed; flicker excluded", () => {
    const { people: ps, brief } = people(replay());
    expect(ps.map((p) => p.ids)).toEqual([[4], [7, 8]]);
    expect(brief).toBe(1);
    expect(ps[0].hidden).toEqual([{ f0: 11, f1: 15, id: 4 }]);
    expect(ps[1].changes).toEqual([{ f: 6, from: 7, to: 8 }]);
    expect(summaryText(ps)).toBe("2 people · 1 kept one number · 1 changed");
  });
  it("focus = hidden-and-kept person; storyline steps in order with real frame counts", () => {
    const d = replay();
    const p = pickFocus(people(d).people)!;
    expect(p.key).toBe(4);
    const st = storySteps(d, p, keptV);
    expect(st.map((s) => s.kind)).toEqual(["enter", "hidden", "back", "cross", "exit"]);
    expect(st[1].text).toContain("disappears for 5 frames");
    expect(st[2].text).toContain("still #4 (appearance and path match)");
    expect(storySteps(d, p)[2].text).toContain("Not checked");
    expect(watchLine(p, st)).toBe("Watch #4 disappear from view and come back with the same number.");
  });
  it("an ID change on the focus person is shown honestly", () => {
    const d = replay();
    const p = people(d).people[1];
    expect(storySteps(d, p).some((s) => s.kind === "change" && s.text.includes("#7 → #8"))).toBe(true);
  });
  it("boxes interpolate, hidden gets a dashed ghost, return flashes 'kept'", () => {
    const tr = tracksOf(replay().runs.ours.frames);
    const mid = boxesAt(tr, 2.5, 10).find((b) => b.id === 4)!;
    expect(mid.box[0]).toBeCloseTo(25);
    expect(boxesAt(tr, 13, 10).find((b) => b.id === 4)?.state).toBe("hidden");
    expect(boxesAt(tr, 17, 10).find((b) => b.id === 4)?.state).toBe("back"); // no verdict -> never "kept"
    expect(boxesAt(tr, 17, 10, keptV).find((b) => b.id === 4)?.state).toBe("kept");
    expect(boxesAt(tr, 17, 10, swapV).find((b) => b.id === 4)?.state).toBe("swap");
    expect(boxesAt(tr, 13, 10, swapV).find((b) => b.id === 4)).toBeUndefined(); // no "still #4" ghost over a swap
    expect(boxesAt(tr, 8, 10).find((b) => b.id === 4)?.state).toBe("seen");
  });
  it("chips sit on the box top edge away from the neighbour; a collision moves the lower one below", () => {
    const c = placeChips([
      { id: 1, box: [100, 100, 50, 100], text: "#1", w: 30, h: 20 },
      { id: 2, box: [110, 105, 50, 100], text: "#2", w: 30, h: 20 },
    ]);
    const a = c.find((x) => x.id === 1)!, b = c.find((x) => x.id === 2)!;
    expect(a.below).toBe(false);
    expect(a.y).toBe(90);
    expect(b.x).toBe(110 + 50 - 30); // neighbour on its left -> right side
    const far = placeChips([{ id: 1, box: [0, 0, 50, 100], text: "#1", w: 30, h: 20 }, { id: 2, box: [500, 0, 50, 100], text: "#2", w: 30, h: 20 }]);
    expect(far.every((x) => !x.below)).toBe(true);
  });
  it("timeline: past 12 people, a 'Show all people' toggle instead of a cutoff line", () => {
    const many = Array.from({ length: 34 }, (_, k) => ({ ...people(replay()).people[0], key: 100 + k, f0: k }));
    render(<PersonTimeline people={many} frames={40} f={5} focus={100} onSeek={() => {}} />);
    expect(screen.getAllByTestId("ptl-row")).toHaveLength(12);
    const tog = screen.getByTestId("ptl-show-all");
    expect(tog.textContent).toBe("Show all people (22 more)");
    expect(document.body.textContent).not.toMatch(/not drawn/);
    fireEvent.click(tog);
    expect(screen.getAllByTestId("ptl-row")).toHaveLength(34);
    expect(tog.textContent).toBe("Show the first 12 only");
  });
  it("timeline: one bar per person, dashed hidden, change tag, seek on click", () => {
    const onSeek = vi.fn();
    render(<PersonTimeline people={people(replay()).people} frames={20} f={5} focus={4} onSeek={onSeek} />);
    expect(screen.getByTestId("timeline-summary").textContent).toBe("2 people · 1 kept one number · 1 changed");
    expect(screen.getAllByTestId("ptl-row")).toHaveLength(2);
    expect(screen.getAllByTestId("ptl-hidden")).toHaveLength(1);
    const ch = screen.getByTestId("ptl-change");
    expect(ch.textContent).toBe("#7 → #8");
    fireEvent.click(ch);
    expect(onSeek).toHaveBeenCalledWith(6);
  });
  it("count bar is a single row 'IN n · OUT n' with no gate-side prose", () => {
    render(<CountBar inN={0} outN={1} />);
    expect(screen.getByTestId("count-bar").textContent).toBe("IN 0 · OUT 1");
  });
  it("scenario order: 'What tracking is' first", () => {
    const o = orderScenarios([{ id: "shibuya_crossing" }, { id: "subway_turnstile" }, { id: "man_entering_metro" }] as never);
    expect(o.map((m) => m.id)).toEqual(["man_entering_metro", "subway_turnstile", "shibuya_crossing"]);
  });
});
