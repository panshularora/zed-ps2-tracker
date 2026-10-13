import { describe, expect, it } from "vitest";
import { parseClip, type Clip } from "../src/data";
import {
  boxIou,
  compareVerdicts,
  defaultFollowId,
  identityChain,
  matchAcross,
  overlappingPairs,
  pickBoxAt,
  switchesForChain,
} from "../src/identity";

const clip: Clip = {
  clip: "MOT17-04",
  tracker: "bytetrack",
  fps: 30,
  frames: 100,
  width: 1920,
  height: 1080,
  gate: null,
  tracks: {
    "14": [[1, 10, 10, 40, 80, 0.9], [2, 12, 10, 40, 80, 0.9]],
    "41": [[50, 20, 10, 40, 80, 0.9]],
    "7": [[1, 200, 10, 30, 60, 0.8], [2, 200, 10, 30, 60, 0.8]],
  },
  pending: [],
  events: [
    { f: 50, type: "id_switch", gt: 3, from: 14, to: 41 },
    { f: 70, type: "id_switch", gt: 3, from: 41, to: 99 },
  ],
};

describe("overlap", () => {
  it("is 1 for identical boxes and 0 when they miss", () => {
    const a = { id: 1, x: 0, y: 0, w: 10, h: 10, conf: 1 };
    expect(boxIou(a, { ...a, id: 2 })).toBe(1);
    expect(boxIou(a, { id: 2, x: 20, y: 20, w: 5, h: 5, conf: 1 })).toBe(0);
  });
  it("flags a pair above the IoU floor", () => {
    const boxes = [
      { id: 1, x: 0, y: 0, w: 10, h: 10, conf: 1 },
      { id: 2, x: 5, y: 0, w: 10, h: 10, conf: 1 },
      { id: 3, x: 80, y: 80, w: 10, h: 10, conf: 1 },
    ];
    const pairs = overlappingPairs(boxes, 0.15);
    expect(pairs).toHaveLength(1);
    expect(pairs[0].a.id).toBe(1);
    expect(pairs[0].b.id).toBe(2);
    expect(pairs[0].iou).toBeCloseTo(1 / 3, 5);
  });
});

describe("identity", () => {
  it("walks id_switch both ways so 14, 41 and 99 are one person", () => {
    const c = parseClip(clip);
    expect(identityChain(c.events, 14)).toEqual([14, 41, 99]);
    expect(identityChain(c.events, 41)).toEqual([14, 41, 99]);
    expect(identityChain(c.events, 7)).toEqual([7]);
  });
  it("follows the longest person visible on frame 1", () => {
    const c = parseClip(clip);
    expect(defaultFollowId(c, 1)).toBe(7);
  });
  it("matches the same standing person across two tracker frames", () => {
    const a = { id: 22, x: 10, y: 10, w: 40, h: 80, conf: 1 };
    const others = [
      { id: 9, x: 200, y: 10, w: 40, h: 80, conf: 1 },
      { id: 3, x: 12, y: 12, w: 40, h: 80, conf: 1 },
    ];
    expect(matchAcross(a, others)?.id).toBe(3);
  });
  it("lists the switches that retarget this person", () => {
    const ids = new Set(identityChain(clip.events, 14));
    expect(switchesForChain(clip.events, ids)).toHaveLength(2);
  });
});

describe("compareVerdicts", () => {
  const a = [{ f: 70, type: "id_switch" as const, gt: 8, from: 2, to: 10 }];
  const bSame = [{ f: 70, type: "id_switch" as const, gt: 8, from: 2, to: 10 }];
  const bKeep: typeof a = [];
  const bLater = [{ f: 243, type: "id_switch" as const, gt: 8, from: 7, to: 20 }];

  it("marks both red when they switch the same GT person", () => {
    const v = compareVerdicts(a, new Set([2, 10]), bSame, new Set([2, 10]), 70);
    expect(v).toEqual({ a: "switch", b: "switch", gt: 8 });
  });

  it("marks green recovery only on the tracker that kept the GT id", () => {
    const v = compareVerdicts(a, new Set([2, 10]), bKeep, new Set([2]), 70);
    expect(v.a).toBe("switch");
    expect(v.b).toBe("recovery");
    expect(v.gt).toBe(8);
  });

  it("does not paint green on a frame with no switch", () => {
    const v = compareVerdicts(a, new Set([2, 10]), bLater, new Set([7, 20]), 80);
    expect(v).toEqual({ a: "hold", b: "hold", gt: null });
  });

  it("does not claim recovery when that side has no matched person", () => {
    const v = compareVerdicts(a, new Set([2, 10]), bKeep, new Set(), 70);
    expect(v.b).toBe("hold");
  });

  it("treats a same-gt switch one frame apart as a switch, not recovery", () => {
    const early = [{ f: 69, type: "id_switch" as const, gt: 8, from: 2, to: 10 }];
    const v = compareVerdicts(early, new Set([2, 10]), a, new Set([2, 10]), 70);
    expect(v.a).toBe("switch");
    expect(v.b).toBe("switch");
  });
});

describe("pick", () => {
  it("hits the smallest containing box", () => {
    const boxes = [
      { id: 1, x: 0, y: 0, w: 100, h: 100, conf: 1 },
      { id: 2, x: 10, y: 10, w: 20, h: 20, conf: 1 },
    ];
    expect(pickBoxAt(boxes, 15, 15)?.id).toBe(2);
    expect(pickBoxAt(boxes, 90, 90)?.id).toBe(1);
    expect(pickBoxAt(boxes, 200, 200)).toBeNull();
  });
});
