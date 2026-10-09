import {
  activeRings, describe as describeEvent, fmtClock, gateCounts, gatePulse, indexTracks, lastAtOrBefore,
  seekTimeForTracked, tickPosition, timeOfTracked, trackedFrame, trail, visibleEvents, TRAIL_FRAMES,
} from "../src/lib/playback";
import { EVENTS, result } from "./fixtures";

describe("frame mapping", () => {
  it("maps video frames to tracked frames with a stride", () => {
    expect(trackedFrame(1, 1, 20)).toBe(1);
    expect(trackedFrame(20, 1, 20)).toBe(20);
    expect(trackedFrame(21, 1, 20)).toBeNull(); // beyond the tracked range
    expect(trackedFrame(0, 1, 20)).toBeNull();
    expect(trackedFrame(1, 2, 20)).toBe(1);
    expect(trackedFrame(2, 2, 20)).toBe(1); // held between tracked frames
    expect(trackedFrame(3, 2, 20)).toBe(2);
  });
  it("round-trips tracked frame -> seek time -> tracked frame", () => {
    for (const stride of [1, 2, 3])
      for (let k = 1; k <= 20; k++) {
        const t = seekTimeForTracked(k, stride, 30);
        const v = Math.floor(t * 30 + 1e-6) + 1;
        expect(trackedFrame(v, stride, 20)).toBe(k);
      }
    expect(timeOfTracked(1, 2, 10)).toBe(0);
    expect(timeOfTracked(3, 2, 10)).toBeCloseTo(0.4);
  });
  it("formats the clock", () => {
    expect(fmtClock(0)).toBe("00:00.00");
    expect(fmtClock(65.5)).toBe("01:05.50");
  });
});

describe("tracks", () => {
  const idx = indexTracks(result().tracks);
  const rows = idx.tracks[0].rows;
  it("indexes by frame and binary-searches", () => {
    expect(idx.byFrame.get(1)?.[0].id).toBe(7);
    expect(idx.byFrame.get(9)).toBeUndefined(); // lost frames 8-11
    expect(rows[lastAtOrBefore(rows, 9)][0]).toBe(7);
    expect(lastAtOrBefore(rows, 0)).toBe(-1);
  });
  it("draws trails at the bottom-centre, fading over 30 frames, only for visible tracks", () => {
    const pts = trail(rows, 6);
    expect(pts).toHaveLength(6);
    expect(pts.at(-1)).toEqual({ x: 20, y: 10 + 30 + 50, alpha: 1 });
    expect(pts[0].alpha).toBeCloseTo(1 - 5 / TRAIL_FRAMES);
    expect(trail(rows, 9)).toEqual([]); // not on screen -> no trail
    expect(trail(rows, 20, 5).length).toBeLessThanOrEqual(5);
  });
});

describe("events", () => {
  it("neutral clips show only crossings and recoveries", () => {
    const n = visibleEvents(EVENTS, "neutral");
    expect(n.map((e) => e.type)).toEqual(["crossing", "recovery"]);
    expect(visibleEvents(EVENTS, "alert")).toHaveLength(EVENTS.length);
  });
  it("counts crossings (and held crossings) up to a frame", () => {
    expect(gateCounts(EVENTS, 5)).toEqual({ in: 0, out: 0 });
    expect(gateCounts(EVENTS, 6)).toEqual({ in: 1, out: 0 });
    expect(gateCounts(EVENTS, 20)).toEqual({ in: 1, out: 1 }); // tailgate is not a 2nd crossing
  });
  it("rings a recovered box once for 300 ms of video time", () => {
    const rec = [{ frame: 12, track_id: 7 }];
    const t0 = timeOfTracked(12, 1, 10);
    expect(activeRings(rec, t0 - 0.01, 1, 10).size).toBe(0);
    expect(activeRings(rec, t0, 1, 10).get(7)).toBeCloseTo(1);
    expect(activeRings(rec, t0 + 0.15, 1, 10).get(7)).toBeCloseTo(0.5);
    expect(activeRings(rec, t0 + 0.3, 1, 10).size).toBe(0);
    expect(gatePulse([6], timeOfTracked(6, 1, 10) + 0.1, 1, 10)).toBeGreaterThan(0);
    expect(gatePulse([6], 0, 1, 10)).toBe(0);
  });
  it("describes events neutrally", () => {
    expect(describeEvent(EVENTS[0])).toBe("#7 crossed IN");
    expect(describeEvent(EVENTS[1])).toBe("#7 kept its ID after lost 0.40 s");
    expect(describeEvent(EVENTS[4])).toContain("heuristic");
  });
  it("places scrubber ticks by time", () => {
    const r = result([], 2);
    expect(tickPosition(1, r, 4)).toBe(0);
    expect(tickPosition(11, r, 4)).toBeCloseTo(0.5);
    expect(tickPosition(500, r, 4)).toBe(1);
  });
});
