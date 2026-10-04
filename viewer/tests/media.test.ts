import { describe, expect, it } from "vitest";
import {
  VideoClock,
  fadeProgress,
  framePath,
  frameFromCurrentTime,
  frameFromMediaTime,
  mix,
  seekTime,
  type VideoLike,
} from "../src/media";

describe("frame mapping", () => {
  for (const fps of [14, 25, 29.97, 30, 60]) {
    it(`seek → currentTime → frame round-trips at ${fps} fps`, () => {
      for (let f = 1; f <= 2000; f++) expect(frameFromCurrentTime(seekTime(f, fps), fps)).toBe(f);
    });
    it(`rVFC mediaTime (frame start, ±30 % jitter) → frame at ${fps} fps`, () => {
      for (let f = 1; f <= 2000; f++) {
        const start = (f - 1) / fps;
        for (const j of [-0.3, 0, 0.3]) expect(frameFromMediaTime(start + j / fps, fps)).toBe(f);
      }
    });
  }
  it("currentTime exactly on a frame boundary belongs to that frame", () => {
    expect(frameFromCurrentTime(0, 25)).toBe(1);
    expect(frameFromCurrentTime(1 / 25, 25)).toBe(2);
    expect(frameFromCurrentTime(0.0399, 25)).toBe(1);
  });
  it("frame paths", () => {
    expect(framePath("frames/MOT17-04/%06d.jpg", 527)).toBe("frames/MOT17-04/000527.jpg");
    expect(framePath("f_%d.jpg", 7)).toBe("f_7.jpg");
  });
});

class FakeVideo implements VideoLike {
  currentTime = 0;
  paused = true;
  duration?: number;
  cbs: ((now: number, meta: { mediaTime: number }) => void)[] = [];
  listeners: Record<string, (() => void)[]> = {};
  requestVideoFrameCallback?: VideoLike["requestVideoFrameCallback"];
  constructor(rvfc: boolean) {
    if (rvfc) {
      this.requestVideoFrameCallback = (cb) => {
        this.cbs.push(cb);
        return this.cbs.length;
      };
    }
  }
  play() {
    this.paused = false;
  }
  pause() {
    this.paused = true;
    (this.listeners.pause ?? []).forEach((l) => l());
  }
  addEventListener(t: string, cb: () => void) {
    (this.listeners[t] ??= []).push(cb);
  }
  removeEventListener(t: string, cb: () => void) {
    this.listeners[t] = (this.listeners[t] ?? []).filter((l) => l !== cb);
  }
  present(mediaTime: number) {
    const cb = this.cbs.shift();
    cb?.(0, { mediaTime });
  }
  seeked() {
    (this.listeners.seeked ?? []).forEach((l) => l());
  }
}

describe("VideoClock", () => {
  it("drives frames from rVFC while playing, with the clip offset", () => {
    const v = new FakeVideo(true);
    const seen: number[] = [];
    const clock = new VideoClock(v, 30, 526, (f) => seen.push(f)); // val-half: clip 1 = video 527
    expect(clock.play()).toBe(true);
    v.present((527 - 1) / 30);
    v.present((528 - 1) / 30 + 0.004);
    expect(seen).toEqual([1, 2]);
    clock.pause();
    expect(v.paused).toBe(true);
    // pausing snaps to the centre of the frame on screen (clip 2 = video 528)
    expect(v.currentTime).toBeCloseTo((528 - 0.5) / 30, 9);
    v.seeked();
    expect(seen).toEqual([1, 2, 2]);
  });
  it("seeks to the frame centre and reports the frame after `seeked`", () => {
    const v = new FakeVideo(true);
    const seen: number[] = [];
    const clock = new VideoClock(v, 25, 10, (f) => seen.push(f));
    clock.seek(73);
    expect(v.currentTime).toBeCloseTo((83 - 0.5) / 25, 9);
    v.seeked();
    expect(seen).toEqual([73]);
  });
  it("at the very end (currentTime == duration) reports the last frame", () => {
    const v = new FakeVideo(true);
    const seen: number[] = [];
    const clock = new VideoClock(v, 30, 0, (f) => seen.push(f));
    v.duration = 4;
    v.currentTime = 4;
    clock.emitFromCurrentTime();
    expect(seen).toEqual([120]);
  });
  it("without rVFC (Firefox) the app steps frames itself", () => {
    const v = new FakeVideo(false);
    const clock = new VideoClock(v, 25, 0, () => {});
    expect(clock.hasRvfc).toBe(false);
    expect(clock.play()).toBe(false);
    expect(v.paused).toBe(true);
  });
});

describe("decision fade", () => {
  it("lasts 150 ms of video time after the window closes", () => {
    expect(fadeProgress(15, 15, 25, 150)).toBe(0); // still held
    expect(fadeProgress(16, 15, 25, 150)).toBeCloseTo(40 / 150);
    expect(fadeProgress(18, 15, 25, 150)).toBeCloseTo(120 / 150);
    expect(fadeProgress(19, 15, 25, 150)).toBe(1);
    expect(fadeProgress(16, 15, 60, 150)).toBeCloseTo(16.6667 / 150, 3);
  });
  it("mixes colours", () => {
    expect(mix("#FFFFFF", "#000000", 0)).toBe("rgba(255, 255, 255, 1)");
    expect(mix("#FFFFFF", "#000000", 1, 0.5)).toBe("rgba(0, 0, 0, 0.5)");
  });
});
