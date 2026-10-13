import type { ClipEvent, FrameBox, IndexedClip } from "./data";

/** Intersection-over-union of two axis-aligned boxes. */
export function boxIou(a: FrameBox, b: FrameBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export interface OverlapPair {
  a: FrameBox;
  b: FrameBox;
  iou: number;
}

/** Pairs of boxes on the same frame whose IoU is at least `minIou`. */
export function overlappingPairs(boxes: FrameBox[], minIou = 0.15): OverlapPair[] {
  const out: OverlapPair[] = [];
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const iou = boxIou(boxes[i], boxes[j]);
      if (iou >= minIou) out.push({ a: boxes[i], b: boxes[j], iou });
    }
  }
  return out.sort((x, y) => y.iou - x.iou);
}

/**
 * Tracker IDs that belong to the same person as `seed`, walking `id_switch`
 * events in both directions (14 → 41 still highlights both numbers).
 */
export function identityChain(events: ClipEvent[], seed: number): number[] {
  const adj = new Map<number, number[]>();
  const link = (a: number, b: number) => {
    const list = adj.get(a);
    if (list) list.push(b);
    else adj.set(a, [b]);
  };
  for (const e of events) {
    if (e.type === "id_switch") {
      link(e.from, e.to);
      link(e.to, e.from);
    }
  }
  const seen = new Set<number>([seed]);
  const q = [seed];
  while (q.length) {
    const n = q.pop()!;
    for (const m of adj.get(n) ?? []) {
      if (!seen.has(m)) {
        seen.add(m);
        q.push(m);
      }
    }
  }
  return [...seen].sort((a, b) => a - b);
}

/** Person currently on screen (longest track at `frame`), else the longest track in the clip. */
export function defaultFollowId(clip: IndexedClip, frame = 1): number | null {
  const boxes = clip.byFrame.get(frame) ?? [];
  if (boxes.length) {
    let best = boxes[0].id;
    let len = -1;
    for (const b of boxes) {
      const n = clip.tracks[String(b.id)]?.length ?? 0;
      if (n > len || (n === len && b.id < best)) {
        len = n;
        best = b.id;
      }
    }
    return best;
  }
  let best: number | null = null;
  let len = -1;
  for (const [id, rows] of Object.entries(clip.tracks)) {
    if (rows.length > len) {
      len = rows.length;
      best = Number(id);
    }
  }
  return best;
}

/** Same physical person in another tracker this frame: highest IoU with `box`. */
export function matchAcross(
  box: FrameBox | undefined,
  others: FrameBox[],
  minIou = 0.2,
): FrameBox | null {
  if (!box || !others.length) return null;
  let best: FrameBox | null = null;
  let bestIou = minIou;
  for (const o of others) {
    const v = boxIou(box, o);
    if (v >= bestIou) {
      bestIou = v;
      best = o;
    }
  }
  return best;
}

/** Centres of the followed person over the last `length` frames (for a motion trail). */
export function trailCenters(
  clip: IndexedClip,
  ids: ReadonlySet<number>,
  frame: number,
  length = 18,
): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  const f0 = Math.max(1, frame - length + 1);
  for (let f = f0; f <= frame; f++) {
    const hit = (clip.byFrame.get(f) ?? []).find((b) => ids.has(b.id));
    if (hit) out.push({ x: hit.x + hit.w / 2, y: hit.y + hit.h / 2 });
  }
  return out;
}

/** Smallest box that contains (x, y) in clip pixel space. */
export function pickBoxAt(boxes: FrameBox[], x: number, y: number): FrameBox | null {
  let best: FrameBox | null = null;
  let area = Infinity;
  for (const b of boxes) {
    if (x >= b.x && y >= b.y && x <= b.x + b.w && y <= b.y + b.h) {
      const a = b.w * b.h;
      if (a < area) {
        area = a;
        best = b;
      }
    }
  }
  return best;
}

export function canvasToClip(
  cv: HTMLCanvasElement,
  clipWidth: number,
  clientX: number,
  clientY: number,
): { x: number; y: number } {
  const rect = cv.getBoundingClientRect();
  const scale = (cv.clientWidth || 1) / clipWidth;
  return { x: (clientX - rect.left) / scale, y: (clientY - rect.top) / scale };
}

/** id_switch events that change a number inside this identity chain. */
export function switchesForChain(events: ClipEvent[], ids: ReadonlySet<number>): ClipEvent[] {
  return events.filter(
    (e) => e.type === "id_switch" && (ids.has(e.from) || ids.has(e.to)),
  );
}

export type IdentityVerdict = "switch" | "recovery" | "hold";

/** CLEAR id_switch on this exact frame, optionally for a followed chain or a GT id. */
export function switchOnFrame(
  events: ClipEvent[],
  frame: number,
  ids?: ReadonlySet<number>,
  gt?: number,
): ClipEvent | null {
  for (const e of events) {
    if (e.type !== "id_switch" || e.f !== frame) continue;
    if (gt != null && e.gt === gt) return e;
    if (ids && (ids.has(e.from) || ids.has(e.to))) return e;
  }
  return null;
}

function switchNearGt(events: ClipEvent[], frame: number, gt: number, window: number): ClipEvent | null {
  for (const e of events) {
    if (e.type !== "id_switch" || e.gt !== gt) continue;
    if (Math.abs(e.f - frame) <= window) return e;
  }
  return null;
}

/**
 * Side-by-side identity call. Green recovery only when the other tracker
 * switched this ground-truth person on this frame and this tracker did not
 * switch them on this frame or the adjacent ones (same-gt near-miss).
 */
export function compareVerdicts(
  aEvents: ClipEvent[],
  aIds: ReadonlySet<number>,
  bEvents: ClipEvent[],
  bIds: ReadonlySet<number>,
  frame: number,
  near = 2,
): { a: IdentityVerdict; b: IdentityVerdict; gt: number | null } {
  const aExact = switchOnFrame(aEvents, frame, aIds);
  const bExact = switchOnFrame(bEvents, frame, bIds);
  const gt =
    (aExact?.type === "id_switch" ? aExact.gt : undefined) ??
    (bExact?.type === "id_switch" ? bExact.gt : undefined) ??
    null;
  const aHit = Boolean(aExact || (gt != null && switchNearGt(aEvents, frame, gt, near)));
  const bHit = Boolean(bExact || (gt != null && switchNearGt(bEvents, frame, gt, near)));
  const a: IdentityVerdict =
    aHit ? "switch" : bExact && gt != null && aIds.size > 0 ? "recovery" : "hold";
  const b: IdentityVerdict =
    bHit ? "switch" : aExact && gt != null && bIds.size > 0 ? "recovery" : "hold";
  return { a, b, gt };
}
