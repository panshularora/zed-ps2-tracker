// Identity story: pure logic for the single-tracker (ours only) player in Studio and Scenarios.
// Everything is derived from the tracker's own output (ReplayData.runs.ours + its events); no
// numbers are invented. A "person" here is one tracker identity chain: IDs joined by the
// tracker's own renumber events (a real ID change is kept visible, never merged away silently).
import type { Box } from "../lib/scenarios";
import type { GEvent, ReplayData } from "../guided/model";
import { backLabel, keptLabel, reasonText, swapLabel, type VerdictOf } from "./verify";

export type XYWH = [number, number, number, number];
export interface Span { f0: number; f1: number; id: number }
export interface Person {
  key: number; // the ID shown longest in the chain
  ids: number[]; // in order of appearance
  visible: Span[]; // stretches where a box was drawn
  hidden: Span[]; // gaps inside one ID where the box disappeared and the same ID came back
  changes: { f: number; from: number; to: number }[];
  /** reappearances the kept-vs-swap check flagged: the number may now be on someone else */
  swaps: { f: number; last: number; id: number; why: string }[];
  frames: number; // frames with a box
  f0: number;
  f1: number;
}

export const MIN_PERSON_S = 0.5; // shorter chains are flicker, counted separately

export function tracksOf(frames: Record<string, Box[]>): Map<number, Map<number, XYWH>> {
  const out = new Map<number, Map<number, XYWH>>();
  for (const [f, bs] of Object.entries(frames))
    for (const [id, x, y, w, h] of bs) {
      let m = out.get(id);
      if (!m) out.set(id, (m = new Map()));
      m.set(Number(f), [x, y, w, h]);
    }
  return out;
}

export function people(d: Pick<ReplayData, "runs" | "events" | "fps">, run: "ours" | "baseline" = "ours", verdict?: VerdictOf): { people: Person[]; brief: number } {
  const tracks = tracksOf(d.runs[run].frames);
  const ren = d.events[run].filter((e): e is Extract<GEvent, { t: "renumber" }> => e.t === "renumber");
  // union-find over renumber edges
  const parent = new Map<number, number>();
  const find = (a: number): number => { const p = parent.get(a) ?? a; if (p === a) return a; const r = find(p); parent.set(a, r); return r; };
  for (const e of ren) { const a = find(e.from), b = find(e.to); if (a !== b) parent.set(b, a); }
  const groups = new Map<number, number[]>();
  for (const id of tracks.keys()) { const r = find(id); groups.set(r, [...(groups.get(r) ?? []), id]); }
  const all: Person[] = [];
  for (const ids of groups.values()) {
    const visible: Span[] = [], hidden: Span[] = [];
    let n = 0;
    for (const id of ids) {
      const fs = [...tracks.get(id)!.keys()].sort((a, b) => a - b);
      n += fs.length;
      let s = fs[0];
      for (let i = 1; i <= fs.length; i++) {
        if (i === fs.length || fs[i] !== fs[i - 1] + 1) {
          visible.push({ f0: s, f1: fs[i - 1], id });
          if (i < fs.length) { hidden.push({ f0: fs[i - 1] + 1, f1: fs[i] - 1, id }); s = fs[i]; }
        }
      }
    }
    visible.sort((a, b) => a.f0 - b.f0);
    const order = [...ids].sort((a, b) => Math.min(...tracks.get(a)!.keys()) - Math.min(...tracks.get(b)!.keys()));
    const set = new Set(ids);
    const changes = ren.filter((e) => set.has(e.to)).map((e) => ({ f: e.f, from: e.from, to: e.to })).sort((a, b) => a.f - b.f);
    const dominant = [...ids].sort((a, b) => tracks.get(b)!.size - tracks.get(a)!.size)[0];
    const swaps: Person["swaps"] = [];
    const held = hidden.filter((h) => {
      const v = verdict?.(h.id, h.f1 + 1);
      if (v?.verdict !== "swap") return true;
      swaps.push({ f: h.f1 + 1, last: h.f0 - 1, id: h.id, why: reasonText(v) });
      return false;
    });
    all.push({ key: dominant, ids: order, visible, hidden: held.sort((a, b) => a.f0 - b.f0), changes, swaps: swaps.sort((a, b) => a.f - b.f), frames: n, f0: visible[0].f0, f1: Math.max(...visible.map((v) => v.f1)) });
  }
  const minF = Math.max(2, Math.round(MIN_PERSON_S * d.fps));
  const kept = all.filter((p) => p.frames >= minF).sort((a, b) => a.f0 - b.f0 || a.key - b.key);
  return { people: kept, brief: all.length - kept.length };
}

export function summaryText(ps: Person[]): string {
  const changed = ps.filter((p) => p.changes.length > 0).length;
  const unsure = ps.filter((p) => p.changes.length === 0 && p.swaps.length > 0).length;
  const n = ps.length;
  const base = `${n} ${n === 1 ? "person" : "people"} · ${n - changed - unsure} kept one number · ${changed} changed`;
  return unsure ? `${base} · ${unsure} uncertain (possible swap)` : base;
}

/** Story focus: the longest person who was hidden and came back with the same number; else the longest. */
export function pickFocus(ps: Person[]): Person | null {
  if (!ps.length) return null;
  const byLen = [...ps].sort((a, b) => b.frames - a.frames);
  return byLen.find((p) => p.changes.length === 0 && p.swaps.length === 0 && p.hidden.some((h) => h.f1 - h.f0 + 1 >= 3)) ?? byLen[0];
}

const firstOf = (m: Map<number, XYWH>) => { let lo = Infinity; for (const k of m.keys()) if (k < lo) lo = k; return lo; };
const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
export const lerpBox = (a: XYWH, b: XYWH, k: number): XYWH => [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k), lerp(a[3], b[3], k)];

export interface Drawn { id: number; box: XYWH; state: "seen" | "hidden" | "kept" | "swap" | "back"; }
export const KEPT_S = 1.0;

/** Boxes at fractional tracked frame `ff` (1-based): smooth interpolation, hidden ghosts, and a 1 s
 * chip on reappearance: "kept" only when the check confirmed the same person, "swap" when it flagged a
 * possible identity transfer, "back" (not checked) when there is no verdict. */
export function boxesAt(tracks: Map<number, Map<number, XYWH>>, ff: number, fps: number, verdict?: VerdictOf): Drawn[] {
  const f = Math.floor(ff), k = ff - f;
  const out: Drawn[] = [];
  const keptF = Math.round(KEPT_S * fps);
  for (const [id, m] of tracks) {
    const a = m.get(f);
    if (a) {
      const b = m.get(f + 1);
      // came back after a gap recently?
      let state: Drawn["state"] = "seen";
      let j = f;
      while (j > 1 && m.has(j - 1) && f - j < keptF) j--;
      if (f - j < keptF && !m.has(j - 1) && firstOf(m) < j - 1) {
        const v = verdict?.(id, j)?.verdict;
        state = v === "kept" ? "kept" : v === "swap" ? "swap" : "back";
      }
      out.push({ id, box: b ? lerpBox(a, b, k) : a, state });
      continue;
    }
    // inside a gap of this same id: dashed ghost moving from last-seen to where it comes back
    let p = -1, q = -1;
    for (let j = f - 1; j >= Math.max(1, f - 4 * fps); j--) if (m.has(j)) { p = j; break; }
    if (p < 0) continue;
    for (let j = f + 1; j <= f + 4 * fps; j++) if (m.has(j)) { q = j; break; }
    if (q < 0) continue;
    if (verdict?.(id, q)?.verdict === "swap") continue; // not "still #N": the number may have moved to someone else
    out.push({ id, box: lerpBox(m.get(p)!, m.get(q)!, (ff - p) / (q - p)), state: "hidden" });
  }
  return out;
}

export interface Chip { id: number; x: number; y: number; w: number; h: number; below: boolean; text: string }
/** Chip placement: on the box top edge, on the side away from the nearest neighbour; on collision the lower box's chip moves below it. */
export function placeChips(items: { id: number; box: XYWH; text: string; w: number; h: number }[]): Chip[] {
  const out: Chip[] = [];
  const sorted = [...items].sort((a, b) => a.box[1] - b.box[1]);
  for (const it of sorted) {
    const [x, y, w, h] = it.box;
    const cx = x + w / 2;
    let near: number | null = null, nd = Infinity;
    for (const o of items) { if (o === it) continue; const d = Math.abs(o.box[0] + o.box[2] / 2 - cx); if (d < nd) { nd = d; near = o.box[0] + o.box[2] / 2; } }
    const right = near != null && near < cx; // neighbour on the left -> chip on the right
    const chip: Chip = { id: it.id, x: right ? x + w - it.w : x, y: y - it.h / 2, w: it.w, h: it.h, below: false, text: it.text };
    if (out.some((c) => hit(c, chip))) { chip.y = y + h + 2; chip.below = true; }
    out.push(chip);
  }
  return out;
}
const hit = (a: Chip, b: Chip) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

// ---- guided storyline -------------------------------------------------------------------
export interface Step { f: number; text: string; kind: "enter" | "hidden" | "back" | "swap" | "cross" | "change" | "exit" }

export function chipText(b: Drawn, emphasised: boolean): string {
  if (b.state === "hidden") return `still #${b.id} (hidden)`;
  if (b.state === "swap") return swapLabel(b.id);
  if (!emphasised) return `#${b.id}`;
  return b.state === "kept" ? keptLabel(b.id) : b.state === "back" ? backLabel(b.id) : `#${b.id}`;
}

export function storySteps(d: Pick<ReplayData, "events" | "fps" | "gate">, p: Person, verdict?: VerdictOf): Step[] {
  const n = `#${p.key}`;
  const steps: Step[] = [{ f: p.f0, kind: "enter", text: `Person ${n} comes into view. The box and number ${n} are the tracker's way of saying "this is one person".` }];
  for (const h of p.hidden) {
    const len = h.f1 - h.f0 + 1;
    if (len < 3) continue;
    steps.push({ f: h.f0, kind: "hidden", text: `#${h.id} is out of sight (behind someone or missed by the detector). The box disappears for ${len} frames; the dashed box means "still #${h.id}, hidden".` });
    const v = verdict?.(h.id, h.f1 + 1);
    steps.push({ f: h.f1 + 1, kind: "back", text: v?.verdict === "kept"
      ? `#${h.id} comes back still #${h.id} (${reasonText(v)}).`
      : `#${h.id} is drawn again with the same number. Not checked, so we do not claim it is the same person.` });
  }
  for (const s of p.swaps) steps.push({ f: s.f, kind: "swap", text: `Possible swap: #${s.id} reappears on someone who does not match (${s.why}). We count this as uncertain, not kept.` });
  for (const c of p.changes) steps.push({ f: c.f, kind: "change", text: `Honest note: here our tracker changed the number #${c.from} → #${c.to}. This is the kind of mistake we measure against.` });
  const ids = new Set(p.ids);
  for (const e of d.events.ours) if (e.t === "cross" && ids.has(e.id)) steps.push({ f: e.f, kind: "cross", text: `#${e.id} crosses the line going ${e.dir.toUpperCase()}: counted once.` });
  const last = p.ids[p.ids.length - 1];
  const changed = p.changes.length > 0;
  if (!changed && p.swaps.length) { steps.push({ f: p.f1, kind: "exit", text: `Last box for #${p.key}. Its number may have passed to someone else along the way, so this one is uncertain.` }); return steps.sort((a, b) => a.f - b.f); }
  steps.push({ f: p.f1, kind: "exit", text: changed ? `Last box for this person, now #${last}. One person, ${p.ids.length} numbers: this one was not kept.` : `Last box for #${p.key}: one person, one number from approach to exit.` });
  return steps.sort((a, b) => a.f - b.f);
}

export function watchLine(p: Person | null, steps: Step[]): string {
  if (!p) return "Watch the boxes: each one is a person the tracker follows.";
  const c = p.changes[0];
  if (c) return `Watch #${p.key}: our tracker changed this person's number (#${c.from} → #${c.to}), and we show it.`;
  if (p.swaps.length) return `Watch #${p.key}: when it comes back the box may be on someone else. We flag it as a possible swap.`;
  if (steps.some((s) => s.kind === "hidden")) return `Watch #${p.key} disappear from view and come back with the same number.`;
  return `Watch #${p.key} keep the same number from start to finish.`;
}

export function countsAt(events: GEvent[], f: number): { in: number; out: number } {
  let i = 0, o = 0;
  for (const e of events) { if (e.t !== "cross" || e.f > f) continue; if (e.dir === "in") i++; else o++; }
  return { in: i, out: o };
}
