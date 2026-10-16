import { describe, expect, it } from "vitest";
import {
  activeMoment, answerFor, askText, chipText, countsAt, eventLog, extraCountsAt, lookBox, pauseWhy, rateAt, resultText, schedule,
  SLOW_RATE, visibleLog, fallbackText, videoSources, MISSING_CLIP_TEXT, NO_EXPORT_TEXT, stepMoment, stripTicks, tally, tallyText, MOMENT_RULE, spotlightIds, stopCrossed, stopKey, timeOfFrame, trailOf, viewRect, zoomAt, type GEvent, type Moment,
} from "../src/guided/model";

const fps = 10;
const moment = (over: Partial<Moment> = {}): Moment => ({
  f: 41, kind: "overlap", changed: { baseline: true, ours: false }, counted_twice: { baseline: true, ours: false }, focus: [100, 100, 50, 100],
  people: [
    { box: [100, 100, 50, 100], ids: { baseline: { before: 4, after: 11 }, ours: { before: 4, after: 4 } } },
    { box: [120, 100, 50, 100], ids: { baseline: { before: 7, after: 7 }, ours: { before: 8, after: 8 } } },
  ],
  ...over,
});
const d = { fps, frames: 100, moments: [moment(), moment({ f: 81, counted_twice: { baseline: false, ours: false } })] };

describe("key-moment scheduling", () => {
  const stops = schedule(d);
  it("asks just before and answers just after each moment, in order", () => {
    expect(stops.map((s) => s.kind)).toEqual(["ask", "answer", "ask", "answer"]);
    expect(stops[0].t).toBeCloseTo(4 - 0.35);
    expect(stops[1].t).toBeCloseTo(4 + 0.6);
    expect(stops[2].index).toBe(1);
  });
  it("slows to 0.25x on the approach, normal speed elsewhere and in free play", () => {
    expect(rateAt(stops, 1, true)).toBe(1);
    expect(rateAt(stops, 3.0, true)).toBe(SLOW_RATE);
    expect(rateAt(stops, 4.3, true)).toBe(SLOW_RATE);
    expect(rateAt(stops, 5.5, true)).toBe(1);
    expect(rateAt(stops, 3.0, false)).toBe(1);
  });
  it("pauses once per stop while playing; seeks never trigger a pause", () => {
    const done = new Set<string>();
    const s = stopCrossed(stops, 3.6, 3.7, done)!;
    expect(s.kind).toBe("ask");
    done.add(stopKey(s));
    expect(stopCrossed(stops, 3.6, 3.7, done)).toBeNull();
    expect(stopCrossed(stops, 0, 3.7, new Set())).toBeNull(); // jump forward = seek
    expect(stopCrossed(stops, 5, 3.7, new Set())).toBeNull(); // backwards
  });
  it("zooms in over 0.6 s, holds through the pause, eases out", () => {
    expect(zoomAt(stops, 2).k).toBe(0);
    const mid = zoomAt(stops, 3.65 - 0.3).k;
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);
    expect(zoomAt(stops, 4.0)).toEqual({ k: 1, index: 0 });
    expect(zoomAt(stops, 4.6 + 0.7).k).toBe(0);
    expect(activeMoment(stops, 4)).toBe(0);
    expect(activeMoment(stops, 6.3)).toBeNull();
    expect(activeMoment(stops, 7)).toBe(1);
  });
  it("2x view rect stays inside the frame", () => {
    expect(viewRect(1920, 1080, [1900, 1000, 20, 80], 1)).toEqual([960, 540, 960, 540]);
    expect(viewRect(1920, 1080, null, 1)).toEqual([0, 0, 1920, 1080]);
  });
});

describe("counters", () => {
  const ev: GEvent[] = [
    { t: "cross", f: 5, id: 1, dir: "in" }, { t: "cross", f: 9, id: 2, dir: "out" },
    { t: "cross", f: 12, id: 3, dir: "in" }, { t: "lost", f: 6, id: 2 },
  ];
  it("counts crossings up to the frame on screen", () => {
    expect(countsAt(ev, null)).toEqual({ in: 0, out: 0 });
    expect(countsAt(ev, 4)).toEqual({ in: 0, out: 0 });
    expect(countsAt(ev, 9)).toEqual({ in: 1, out: 1 });
    expect(countsAt(ev, 50)).toEqual({ in: 2, out: 1 });
  });
  it("+1 extra only for the tracker that changed the number, after the answer", () => {
    expect(extraCountsAt(d, "baseline", 4.0)).toBe(0);
    expect(extraCountsAt(d, "baseline", 4.7)).toBe(1);
    expect(extraCountsAt(d, "baseline", 99)).toBe(1); // second moment not counted twice
    expect(extraCountsAt(d, "ours", 99)).toBe(0);
  });
});

describe("plain-language text", () => {
  const events: GEvent[] = [
    { t: "cross", f: 11, id: 4, dir: "in" },
    { t: "lost", f: 20, id: 7 },
    { t: "back", f: 28, id: 7, gap_s: 0.8 },
    { t: "renumber", f: 41, from: 4, to: 11 },
  ];
  it("event log lines, synced to time, lost folded into back", () => {
    const log = eventLog({ fps, events: { baseline: events, ours: [] }, moments: d.moments }, "baseline");
    expect(log.map((l) => l.text)).toEqual([
      "Person 4 crossed the line going IN.",
      "Person 7 was out of sight for 0.8 s and came back with the same number.",
      "Person 4 was given a new number: Person 11.",
    ]);
    expect(log[0].t).toBeCloseTo(timeOfFrame({ fps }, 11));
    expect(log[2].moment).toBe(0);
  });
  it("one question per moment, answers per tracker, no right/wrong without ground truth", () => {
    const m = moment();
    expect(askText(m)).toMatch(/keep their number\?$/);
    expect(answerFor(m, "baseline")).toBe("#4 became #11");
    expect(answerFor(m, "ours")).toBe("kept #4");
    const r = resultText(m, "illustrative");
    expect(r).toBe("The two trackers disagree here: ByteTrack #4 became #11, ours kept #4.");
    expect(r).not.toMatch(/wrong|correct|error|right/i);
    expect(resultText({ ...m, verified: true }, "measured")).toMatch(/^Checked against human labels/);
    expect(resultText(m, "measured")).toMatch(/disagree/); // measured clip but unverified moment
  });
  it("pause copy names the person and says where to look", () => {
    const m = moment();
    const ask = pauseWhy(m, "ours", "ask", "illustrative");
    expect(ask.title).toMatch(/person #4/);
    expect(ask.body).toMatch(/overlap/i);
    expect(ask.tag).toMatch(/#4/);
    const ans = pauseWhy(m, "baseline", "answer", "illustrative");
    expect(ans.body).toMatch(/#4 became #11/);
  });
  it("look box is the spotlight person, not the whole frame", () => {
    const frames = { "41": [[4, 100, 100, 200, 400], [7, 400, 200, 40, 80]] as [number, number, number, number, number][] };
    expect(lookBox(frames, 41, new Set([4]), [100, 100, 200, 400], 1920, 1080)).toEqual([100, 100, 200, 400]);
  });
  it("pads a right-edge sliver so the look ring stays on screen", () => {
    const r = lookBox({ "1": [[4, 1845, 203, 74, 450]] }, 1, new Set([4]), [1845, 203, 74, 450], 1920, 1080)!;
    expect(r[2]).toBeGreaterThanOrEqual(120);
    expect(r[0] + r[2]).toBeLessThanOrEqual(1920);
  });
  it("ID chip shows the change for 1 s with a short pulse", () => {
    const ev: GEvent[] = [{ t: "renumber", f: 41, from: 4, to: 11 }];
    expect(chipText(11, 41, ev, fps)).toEqual({ text: "#4 → #11", pulse: 1 });
    expect(chipText(11, 46, ev, fps).text).toBe("#4 → #11");
    expect(chipText(11, 46, ev, fps).pulse).toBe(0);
    expect(chipText(11, 52, ev, fps).text).toBe("Person 11");
  });
  it("spotlight holds both numbers of the changed person and the other person", () => {
    expect([...spotlightIds(moment(), "baseline")].sort((a, b) => a - b)).toEqual([4, 7, 11]);
  });
  it("trail covers the last 2 s of one ID only", () => {
    const frames: Record<string, [number, number, number, number, number][]> = {};
    for (let f = 1; f <= 40; f++) frames[String(f)] = [[f < 30 ? 4 : 11, f, 0, 10, 10]];
    expect(trailOf(frames, 4, 29, fps).length).toBe(21);
    expect(trailOf(frames, 11, 32, fps).length).toBe(3); // restarts with the new number
  });
});

describe("moment tally, strip and stepping", () => {
  const ch = (baseline: boolean, ours: boolean) => moment({ changed: { baseline, ours } });
  const ms = [ch(true, false), { ...ch(false, true), f: 61 }, { ...ch(true, true), f: 91 }];
  it("tally counts each tracker (a 'both' moment counts for each) and the overlap", () => {
    const t = tally(ms);
    expect(t).toEqual({ baseline: 2, ours: 2, both: 1, total: 3 });
    expect(tallyText(t)).toBe("ByteTrack changed 2 · ours changed 2 · both: 1");
    expect(tally([])).toEqual({ baseline: 0, ours: 0, both: 0, total: 0 });
    expect(MOMENT_RULE).toBe("every moment where at least one tracker changed a person's number.");
  });
  it("strip: one tick per moment, top = ByteTrack, bottom = ours, placed by time", () => {
    const ticks = stripTicks({ fps, duration: 10, moments: ms });
    expect(ticks.map((k) => [k.top, k.bottom])).toEqual([[true, false], [false, true], [true, true]]);
    expect(ticks.map((k) => k.x)).toEqual([0.4, 0.6, 0.9]);
    expect(ticks[2].label).toBe("Moment 3: both trackers changed the number");
  });
  it("arrow keys step to the next / previous moment", () => {
    const d2 = { fps, moments: ms }; // moments at 4 s, 6 s, 9 s
    expect(stepMoment(d2, 0, 1)).toBe(0);
    expect(stepMoment(d2, 4 - 1.5, 1)).toBe(1); // at moment 1's lead-in -> next is 2
    expect(stepMoment(d2, 6 - 1.5, -1)).toBe(0);
    expect(stepMoment(d2, 9.5, 1)).toBeNull();
    expect(stepMoment(d2, 0, -1)).toBeNull();
    expect(stepMoment({ fps, moments: [] }, 0, 1)).toBeNull();
  });
});

describe("missing video and the log", () => {
  it("log shows only events at or before the frame on screen", () => {
    const log = [{ f: 1, t: 0, text: "a", kind: "cross" as const }, { f: 130, t: 12.9, text: "b", kind: "cross" as const }];
    expect(visibleLog(log, null)).toEqual([]);
    expect(visibleLog(log, 1).map((l) => l.text)).toEqual(["a"]);
    expect(visibleLog(log, 130).map((l) => l.text)).toEqual(["a", "b"]);
  });
  it("sources: full clip, then demo cut; banner text per case", () => {
    expect(videoSources({ videoUrl: "/api/x", demoUrl: "/demo/x.mp4" })).toEqual(["/api/x", "/demo/x.mp4"]);
    expect(fallbackText({ kind: "illustrative", videoUrl: "/api/x", demoUrl: "/demo/x.mp4" }, false)).toBeNull();
    expect(fallbackText({ kind: "illustrative", videoUrl: "/api/x", demoUrl: null }, true)).toBe(MISSING_CLIP_TEXT);
    expect(fallbackText({ kind: "measured", videoUrl: null, demoUrl: null }, false)).toBe(NO_EXPORT_TEXT);
    expect(MISSING_CLIP_TEXT).toContain("make clips blur");
  });
});
