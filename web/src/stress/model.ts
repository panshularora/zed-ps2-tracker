// Stress Map: pure logic (colour, hatch, borders, headline) over the committed stress grid.
// Every word and number the map shows is computed here from /api/stress/map/{config}.

export type Verdict = "ours_better" | "ours_worse" | "tie" | "unclear";
export interface MapCell {
  people: number;
  hide_s: number;
  diff_mean: number;
  ci95: [number, number];
  p: number;
  verdict: Verdict;
  ours_mean: number;
  bytetrack_mean: number;
  seeds: number[];
  replay_seed: number;
  replay_diff: number;
  first_disagreement: { seed: number; frame: number; pid: number; changed: string } | null;
}
export interface StressMapData {
  config: "default" | "mem6";
  ours_config: string;
  baseline_config: string;
  seeds: number[];
  git: string;
  axes: { people: number[]; hide_s: number[] };
  ours_memory_s: number;
  baseline_memory_s: number;
  cells: MapCell[];
}

export const NEUTRAL = "#262B33";
export const BLUE = "#56B4E9";
export const AMBER = "#F5A524";
/** |mean gap| thresholds: a gap above the k-th one is colour step k+1 (max 5 per side). */
export const STEPS = [0, 1, 3, 6, 12] as const;

function mixHex(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ch = (s: number) => Math.round(((pa >> s) & 255) + (((pb >> s) & 255) - ((pa >> s) & 255)) * t);
  return `#${((1 << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0)).toString(16).slice(1).toUpperCase()}`;
}

export function colourStep(diff: number): number {
  const a = Math.abs(diff);
  return STEPS.filter((t) => a > t).length;
}

/** Ours-worse fills: three flat steps (|gap| ≤ 1, ≤ 3, > 3 ID changes), not a dimmed orange. */
export const AMBER_STEPS = ["#5A4320", "#8A6420", "#F5A524"] as const;

/** Diverging fill: tie = neutral; blue (5 mixed steps) when ours changed fewer IDs, amber (3 flat
 * steps) when more. `ink` is the text colour that reads on the fill. */
export function cellFill(c: Pick<MapCell, "diff_mean" | "verdict">): { fill: string; side: "ours" | "baseline" | "tie"; step: number; ink: string } {
  const step = colourStep(c.diff_mean);
  if (c.verdict === "tie" || step === 0) return { fill: NEUTRAL, side: "tie", step: 0, ink: LIGHT_INK };
  if (c.diff_mean < 0) return { fill: mixHex(NEUTRAL, BLUE, step / 5), side: "ours", step, ink: step >= 4 ? DARK_INK : LIGHT_INK };
  const a = Math.min(step, 3) - 1;
  return { fill: AMBER_STEPS[a], side: "baseline", step, ink: a === 2 ? DARK_INK : LIGHT_INK };
}
export const DARK_INK = "#0E1013";
export const LIGHT_INK = "#E6E8EB";

export const isHatched = (c: Pick<MapCell, "verdict">) => c.verdict === "unclear";

/** '−3.5', '+1.1', '0.0' (true minus sign, one decimal). */
export function fmtGap(x: number): string {
  const r = (Math.sign(x) * Math.round(Math.abs(x) * 10)) / 10;
  if (r === 0) return "0.0";
  return `${r < 0 ? "\u2212" : "+"}${Math.abs(r).toFixed(1)}`;
}

const fmtS = (s: number) => `${Number.isInteger(s) ? s : s.toFixed(1)} s`;

export function cellAt(d: StressMapData, people: number, hide: number): MapCell | undefined {
  return d.cells.find((c) => c.people === people && c.hide_s === hide);
}

export interface Borders {
  /** per density row (axes.people order): column index where ours starts winning, null = never */
  winStart: (number | null)[];
  /** per row: first column after the winning run (= hide_s.length when it runs to the edge) */
  winEnd: (number | null)[];
  startLabel: string;
  endLabel: string;
  startClean: boolean;
  endClean: boolean;
}

export function borders(d: StressMapData): Borders {
  const hs = d.axes.hide_s;
  const winStart: (number | null)[] = [];
  const winEnd: (number | null)[] = [];
  for (const n of d.axes.people) {
    const v = hs.map((h) => cellAt(d, n, h)?.verdict);
    const s = v.indexOf("ours_better");
    if (s < 0) { winStart.push(null); winEnd.push(null); continue; }
    let e = s;
    while (e < hs.length && v[e] === "ours_better") e++;
    winStart.push(s);
    winEnd.push(e);
  }
  const rows = d.axes.people.length;
  const starts = winStart.filter((x): x is number => x != null);
  const ends = winEnd.filter((x): x is number => x != null);
  const never = rows - starts.length;
  let startLabel: string;
  const startClean = starts.length === rows && new Set(starts).size === 1;
  if (starts.length === 0) startLabel = "Ours is not clearly better at any hidden time";
  else if (startClean) startLabel = `Ours starts winning at ${fmtS(hs[starts[0]])} hidden, at every crowd size`;
  else
    startLabel = `No single border: ours starts winning between ${fmtS(hs[Math.min(...starts)])} and ${fmtS(hs[Math.max(...starts)])} hidden, depending on crowd size${never ? ` (never in ${never} of ${rows} rows)` : ""}`;
  // beyond the winning run: what happens there, counted over the cells right of each row's border
  let worse = 0, better = 0, total = 0;
  d.axes.people.forEach((n, r) => {
    const e = winEnd[r];
    if (e == null) return;
    for (let k = e; k < hs.length; k++) {
      const v = cellAt(d, n, hs[k])?.verdict;
      total++;
      if (v === "ours_worse") worse++;
      if (v === "ours_better") better++;
    }
  });
  const endClean = ends.length === rows && new Set(ends).size === 1;
  const inside = ends.filter((e) => e < hs.length);
  let endLabel: string;
  const after = !total
    ? ""
    : worse === 0 && better === 0
      ? `: past it no clear winner in ${total} of ${total} cells`
      : `: past it ours is worse in ${worse} of ${total} cells${better ? `, better in ${better}` : ""}`;
  if (ends.length === 0) endLabel = "";
  else if (inside.length === 0) endLabel = "Ours is still winning at the longest hidden time tested";
  else if (endClean) endLabel = `Ours stops winning at ${fmtS(hs[ends[0]])} hidden, at every crowd size${after}`;
  else
    endLabel = `No single border: ours stops winning between ${fmtS(hs[Math.min(...inside)])} and ${fmtS(hs[Math.max(...inside)])} hidden${inside.length < ends.length ? ` (still winning at ${fmtS(hs[hs.length - 1])} in ${ends.length - inside.length} of ${rows} rows)` : ""}${after}`;
  return { winStart, winEnd, startLabel, endLabel, startClean, endClean };
}

/** The 28px headline: the hidden-time band where ours wins at every crowd size, then what follows. */
export function headline(d: StressMapData): string {
  const hs = d.axes.hide_s;
  const allBetter = hs.map((h) => d.axes.people.every((n) => cellAt(d, n, h)?.verdict === "ours_better"));
  const a = allBetter.indexOf(true);
  if (a < 0) return "Ours is not better than ByteTrack at every crowd size for any hidden time.";
  let b = a;
  while (b + 1 < hs.length && allBetter[b + 1]) b++;
  const band = a === b ? `for ${fmtS(hs[a])}` : `${hs[a]}–${fmtS(hs[b])}`;
  const first = `Ours keeps IDs better when people are hidden ${band}.`;
  if (b + 1 >= hs.length) return `${first} That holds up to the longest hidden time tested.`;
  const rest = d.cells.filter((c) => c.hide_s > hs[b]);
  const worse = rest.filter((c) => c.verdict === "ours_worse").length;
  const better = rest.filter((c) => c.verdict === "ours_better").length;
  const next = hs[b + 1];
  if (worse * 2 > rest.length && d.ours_memory_s <= next)
    return `${first} Past ${fmtS(next)} its memory runs out.`;
  if (worse * 2 > rest.length) return `${first} From ${fmtS(next)} it does worse than ByteTrack in ${worse} of ${rest.length} cells.`;
  return `${first} From ${fmtS(next)} it depends on crowd size: better in ${better}, worse in ${worse} of ${rest.length} cells.`;
}

/** The one line under the toggle. */
export function cellSentence(d: Pick<StressMapData, "cells">): string {
  const n = d.cells[0]?.seeds.length ?? 0;
  return `Each cell: ${n} seeded runs. Number = ours minus ByteTrack in ID changes.`;
}

/** The note next to the toggle: the opt-in config carries its provenance, the default does not. */
export function configNote(cfg: "default" | "mem6", d: Pick<StressMapData, "cells">): string {
  return cfg === "mem6" ? "Tuned on separate seeds · tested on MOT17 only (HOTA 52.5 either way)" : `Default settings · ${d.cells[0]?.seeds.length ?? 0} seeds per cell`;
}
