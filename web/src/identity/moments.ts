// Guided story moments: people who really were hidden or briefly lost and came back with the same
// number. Each one is listed by (clip, id, frame back) and is only shown if the kept-vs-swap check
// (web/public/verify/<clip>.json) says "kept"; every number in the caption comes from that file.
// Picked by hand from the check's "kept" events (longer gaps, someone in front), then the before/after
// crops were looked at; the check is a heuristic on Pexels clips (see docs/keep_check.md).
import type { Reappearance, VerifyDoc } from "./verify";

export interface StoryRef { clip: string; id: number; back: number }
export const STORY_MOMENTS: StoryRef[] = [
  { clip: "subway_turnstile", id: 10, back: 86 },
  { clip: "subway_turnstile", id: 13, back: 118 },
  { clip: "mexico_city_entrance", id: 1, back: 424 },
  { clip: "rabat_stadium", id: 9, back: 100 },
  { clip: "MOT17-09", id: 13, back: 235 },
];

export interface StoryMoment { ref: StoryRef; ev: Reappearance; caption: string }

export function storyCaption(e: Reappearance): string {
  const hide = e.occluder != null ? `walks behind #${e.occluder}` : "is missed by the detector";
  const why = e.gt_before != null ? `checked against human labels: same labelled person` : "appearance and path match";
  return `Person #${e.id} ${hide} for ${e.gap_s.toFixed(1)} s (${e.gap_frames} frames) … comes back still #${e.id} (${why}).`;
}

/** Story moments for a clip that the check confirmed; unconfirmed refs are dropped, never relabelled. */
export function storyMoments(clip: string, doc: VerifyDoc | null): StoryMoment[] {
  if (!doc) return [];
  return STORY_MOMENTS.filter((r) => r.clip === clip).flatMap((ref) => {
    const ev = doc.events.find((e) => e.run === "ours" && e.id === ref.id && e.back === ref.back);
    return ev && ev.verdict === "kept" ? [{ ref, ev, caption: storyCaption(ev) }] : [];
  });
}
