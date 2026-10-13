import type { ClipEvent, FrameBox, IndexedClip } from "@viewer/data";
import { overlappingPairs, switchesForChain } from "@viewer/identity";
import type { PickedMedia } from "@viewer/useMedia";

export interface Story {
  kind: "need-video" | "switch" | "overlap" | "follow";
  title: string;
  body: string;
}

export function personWords(ids: number[], current?: number | null): string {
  if (!ids.length) return "nobody yet";
  const labels = ids.map((n) => `#${n}`).join(" → ");
  if (current != null && ids.includes(current)) {
    return `this person (now #${current}; tracker labels ${labels})`;
  }
  return `this person (tracker labels ${labels})`;
}

/** Plain-English line for the current frame. */
export function frameStory(opts: {
  clip: IndexedClip;
  frame: number;
  followIds: number[];
  boxes: FrameBox[];
  mediaKind: "video" | "frames" | "blank";
  picked: PickedMedia;
}): Story {
  const { clip, frame, followIds, boxes, mediaKind, picked } = opts;
  const hasMedia = mediaKind !== "blank" || Boolean(picked.video || picked.frames?.size);
  const idSet = new Set(followIds);
  const recent = clip.events.filter(
    (e) =>
      e.type === "id_switch" &&
      e.f <= frame &&
      frame - e.f < clip.fps &&
      (idSet.has(e.from) || idSet.has(e.to)),
  );
  const sw = recent[recent.length - 1];
  if (sw && sw.type === "id_switch") {
    return {
      kind: "switch",
      title: `The tracker changed this person's number from #${sw.from} to #${sw.to}`,
      body: "Same person. New label. The colour on the box stays put so you can keep watching them.",
    };
  }
  const pairs = overlappingPairs(boxes, 0.15);
  if (pairs.length) {
    const top = pairs[0];
    return {
      kind: "overlap",
      title: `Two people overlap (#${top.a.id} and #${top.b.id})`,
      body: `Their boxes share ${(top.iou * 100).toFixed(0)}% of the same pixels. This is when numbers often swap. Amber fill marks every overlap on this frame.`,
    };
  }
  if (!hasMedia) {
    return {
      kind: "need-video",
      title: "Drop a video to see the people",
      body: "Boxes are saved tracker results. This CPU demo does not detect people from an MP4. Open your clip, then play.",
    };
  }
  const current = boxes.find((b) => idSet.has(b.id))?.id ?? followIds[0] ?? null;
  return {
    kind: "follow",
    title: `Following ${personWords(followIds, current)}`,
    body: "Click a box to follow someone else. The number on the box is the tracker's ID — it can change; the colour will not.",
  };
}

export function nextSwitch(events: ClipEvent[], ids: ReadonlySet<number>, frame: number): ClipEvent | null {
  return switchesForChain(events, ids).find((e) => e.f > frame) ?? null;
}

export function prevSwitch(events: ClipEvent[], ids: ReadonlySet<number>, frame: number): ClipEvent | null {
  const list = switchesForChain(events, ids).filter((e) => e.f < frame);
  return list.length ? list[list.length - 1] : null;
}
