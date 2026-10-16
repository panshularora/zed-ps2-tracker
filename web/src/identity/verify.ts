// Kept-vs-swap verdicts for "came back with the same number" (scripts/demo/verify_keeps.py writes
// web/public/verify/<clip>.json). A reappearance is only called "kept" when the check says it is the
// same person: on Pexels clips a HEURISTIC (torso colour + path + size), on MOT clips ground truth.
import type { RunKey } from "../lib/scenarios";

export type Verdict = "kept" | "swap" | "unknown";
export interface Reappearance {
  id: number; run: RunKey; last: number; back: number; gap_frames: number; gap_s: number; verdict: Verdict;
  app: number | null; motion_err: number | null; motion_tol: number | null; size_ratio: number | null;
  gt_before: number | null; gt_after: number | null; reasons: string[]; occluder: number | null;
}
export interface VerifyDoc {
  clip: string;
  method: "pexels" | "gt";
  method_text: string;
  thresholds: { app_min: number; motion_base: number; motion_per_s: number; size_max: number; k_frames: number; gt_iou: number };
  summary: Partial<Record<RunKey, Record<Verdict, number>>>;
  events: Reappearance[];
  /** MOT clips only: per labelled person, same denominator for both trackers */
  people?: Partial<Record<RunKey, { labelled: number; kept: number; changed: number; missed: number }>>;
}

/** "N labelled people · ours kept K1 · ByteTrack kept K2"; null when not computed (no win claimed). */
export function peopleLine(doc: VerifyDoc | null): string | null {
  const o = doc?.people?.ours, b = doc?.people?.baseline;
  if (!o || !b || o.labelled !== b.labelled) return null;
  return `${o.labelled} labelled people · ours kept ${o.kept} · ByteTrack kept ${b.kept}`;
}

/** Secondary line: reappearance events, clearly labelled as returns after a gap. */
export function returnsLine(doc: VerifyDoc | null): string | null {
  const o = doc?.summary?.ours, b = doc?.summary?.baseline;
  if (!o || !b) return null;
  return `Returns after a gap (events, not people): ours ${o.kept} same person / ${o.swap} swapped · ByteTrack ${b.kept} / ${b.swap}`;
}
/** (id, first frame back) -> verdict; undefined = not checked */
export type VerdictOf = (id: number, back: number) => Reappearance | undefined;

export const verifyUrl = (clip: string) => `${import.meta.env?.BASE_URL ?? "/"}verify/${clip}.json`;

export async function fetchVerify(clip: string): Promise<VerifyDoc | null> {
  try {
    const r = await fetch(verifyUrl(clip));
    if (!r.ok) return null;
    return (await r.json()) as VerifyDoc;
  } catch {
    return null;
  }
}

export function verdictOf(doc: VerifyDoc | null, run: RunKey = "ours"): VerdictOf {
  const m = new Map<string, Reappearance>();
  for (const e of doc?.events ?? []) if (e.run === run) m.set(`${e.id}:${e.back}`, e);
  return (id, back) => m.get(`${id}:${back}`);
}

export function counts(doc: VerifyDoc | null, run: RunKey): Record<Verdict, number> {
  const out: Record<Verdict, number> = { kept: 0, swap: 0, unknown: 0 };
  for (const e of doc?.events ?? []) if (e.run === run) out[e.verdict]++;
  return out;
}

export const swapLabel = (id: number) => `possible swap: #${id} given to someone else`;
export const keptLabel = (id: number) => `kept #${id}`;
export const backLabel = (id: number) => `#${id} back (not checked)`;

/** Plain tooltip text: what the check does and that it is a heuristic where there is no ground truth. */
export function methodTooltip(doc: VerifyDoc | null): string {
  if (!doc) return "Reappearances on this clip were not checked, so we do not claim 'kept'.";
  const t = doc.thresholds;
  if (doc.method === "gt")
    return `Checked against human labels: the box before and after the gap is matched to a labelled person (IoU ≥ ${t.gt_iou}). Same person = kept; different person = swap.`;
  return `Heuristic check, not ground truth. 'kept' only if (1) torso colour before vs after the gap matches (histogram similarity ≥ ${t.app_min}, ${t.k_frames} frames each side), `
    + `(2) the box reappears where its earlier motion predicts (within ${t.motion_base} + ${t.motion_per_s}×gap-seconds box heights) and (3) its height changes by ≤ ×${t.size_max}. `
    + "Otherwise it is shown as a possible swap.";
}

export function reasonText(e: Reappearance): string {
  if (e.verdict === "kept") {
    if (e.gt_before != null) return `same labelled person (${e.gt_before})`;
    if (e.reasons.length) return e.reasons.join("; ");
    return "appearance and path match";
  }
  return e.reasons.join("; ") || "not checked";
}
