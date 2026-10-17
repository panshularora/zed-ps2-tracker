// Break it yourself: the scene spec, its shareable URL, and the "Make it harder" summary.
import type { PaneKey, Verdict } from "./scene";

export interface BreakSpec {
  seed: number;
  hide_s: number;
  hide_pid: number | null;
  crowd: number;
  pillars: [number, number][];
  groups: [number, number][];
  path: [number, number][];
  config: "default" | "mem6";
}
export interface SceneState { spec: BreakSpec; frame: number }

export const DEFAULT_SPEC: BreakSpec = {
  seed: 7,
  hide_s: 1.5,
  hide_pid: 1,
  crowd: 0,
  pillars: [[250, 470]],
  groups: [[830, 240]],
  path: [[150, 40], [330, 200], [480, 300], [620, 420], [760, 560]],
  config: "default",
};

const pts = (xs: [number, number][]) => xs.map(([x, y]) => `${Math.round(x)},${Math.round(y)}`).join(";");
function parsePts(s: string | null, max: number): [number, number][] {
  if (!s) return [];
  return s.split(";").slice(0, max).map((p) => p.split(",").map(Number) as [number, number]).filter((p) => p.length === 2 && p.every(Number.isFinite));
}

/** Query string that reproduces the scene exactly (the simulator is a pure function of the spec). */
export function toQuery(st: SceneState): string {
  const s = st.spec;
  const q = new URLSearchParams();
  q.set("seed", String(s.seed));
  q.set("hide", String(s.hide_s));
  q.set("pid", s.hide_pid == null ? "none" : String(s.hide_pid));
  q.set("crowd", String(s.crowd));
  if (s.pillars.length) q.set("pillars", pts(s.pillars));
  if (s.groups.length) q.set("groups", pts(s.groups));
  q.set("path", pts(s.path));
  if (s.config !== "default") q.set("cfg", s.config);
  q.set("f", String(st.frame));
  return q.toString();
}

/** Parse a shared query string; anything missing or malformed falls back to the default scene. */
export function fromQuery(qs: string): SceneState | null {
  const q = new URLSearchParams(qs.startsWith("?") ? qs.slice(1) : qs);
  if (!q.has("seed") || !q.has("path")) return null;
  const num = (k: string, d: number) => { const v = Number(q.get(k)); return q.has(k) && Number.isFinite(v) ? v : d; };
  const path = parsePts(q.get("path"), 400);
  const pid = q.get("pid");
  return {
    spec: {
      seed: Math.trunc(num("seed", DEFAULT_SPEC.seed)),
      hide_s: Math.min(8, Math.max(0, num("hide", DEFAULT_SPEC.hide_s))),
      hide_pid: pid === "none" ? null : Math.trunc(num("pid", 1)),
      crowd: Math.min(12, Math.max(0, Math.trunc(num("crowd", DEFAULT_SPEC.crowd)))),
      pillars: parsePts(q.get("pillars"), 8),
      groups: parsePts(q.get("groups"), 4),
      path: path.length >= 2 ? path : DEFAULT_SPEC.path,
      config: q.get("cfg") === "mem6" ? "mem6" : "default",
    },
    frame: Math.max(1, Math.trunc(num("f", 1))),
  };
}

/** Thin a freehand stroke: keep a point every `minStep` px (and always the last one). */
export function simplifyPath(raw: [number, number][], minStep = 18): [number, number][] {
  const out: [number, number][] = [];
  for (const p of raw) {
    const l = out[out.length - 1];
    if (!l || Math.hypot(p[0] - l[0], p[1] - l[1]) >= minStep) out.push([Math.round(p[0]), Math.round(p[1])]);
  }
  const last = raw[raw.length - 1];
  if (last && out.length && (out[out.length - 1][0] !== Math.round(last[0]) || out[out.length - 1][1] !== Math.round(last[1]))) out.push([Math.round(last[0]), Math.round(last[1])]);
  return out;
}

export const HARDER_STEPS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6];
export interface HarderStep { hide_s: number; v: Record<PaneKey, Verdict | null> }

/** Longest hidden time each tracker kept the number before its first break (null = broke at the first step). */
export function heldUntil(steps: HarderStep[], pane: PaneKey): { held: number | null; brokeAt: number | null } {
  let held: number | null = null;
  for (const s of steps) {
    const v = s.v[pane];
    if (v && v.kind === "kept") held = s.hide_s;
    else return { held, brokeAt: s.hide_s };
  }
  return { held, brokeAt: null };
}

export function harderText(steps: HarderStep[]): string {
  if (!steps.length) return "";
  const max = steps[steps.length - 1].hide_s;
  const part = (name: string, pane: PaneKey) => {
    const { held, brokeAt } = heldUntil(steps, pane);
    if (brokeAt == null) return `${name} held through ${max} s (the longest tried)`;
    if (held == null) return `${name} broke already at ${brokeAt} s`;
    return `${name} held until ${held} s hidden`;
  };
  const again = (["ours", "bytetrack"] as PaneKey[])
    .map((k) => [k, keptAgain(steps, k)] as const)
    .filter(([, xs]) => xs.length)
    .map(([k, xs]) => `${k === "ours" ? "ours" : "ByteTrack"} kept it again at ${xs.join(", ")} s`);
  const note = again.length ? ` Not monotonic: ${again.join("; ")}.` : "";
  return `${part("Ours", "ours")}, ${part("ByteTrack", "bytetrack")}.${note}`;
}

/** Hidden times at which a tracker kept the number after it had already broken once. */
export function keptAgain(steps: HarderStep[], pane: PaneKey): number[] {
  const { brokeAt } = heldUntil(steps, pane);
  if (brokeAt == null) return [];
  return steps.filter((s) => s.hide_s > brokeAt && s.v[pane]?.kind === "kept").map((s) => s.hide_s);
}

/** Stop once both trackers have broken AND both stayed broken for the last two steps (breaks are not
 * always monotonic in hidden time, so one break alone does not end the search). */
export function harderDone(steps: HarderStep[]): boolean {
  if (steps.length < 2) return false;
  const broken = (s: HarderStep) => s.v.ours?.kind !== "kept" && s.v.bytetrack?.kind !== "kept";
  return broken(steps[steps.length - 1]) && broken(steps[steps.length - 2]);
}

/** Hidden times shown in the Make-it-harder strip (one cell each). */
export const STRIP_STEPS = [0.5, 1, 1.5, 2, 2.5];
export type StripCell = { hide_s: number; state: "kept" | "changed" | "pending" | "none" };

/** Two-row strip (ByteTrack on top, ours below): blue = kept, amber = changed or lost. */
export function stripRows(steps: HarderStep[]): Record<PaneKey, StripCell[]> {
  const row = (k: PaneKey): StripCell[] =>
    STRIP_STEPS.map((h) => {
      const s = steps.find((x) => x.hide_s === h);
      if (!s) return { hide_s: h, state: "pending" };
      const v = s.v[k];
      return { hide_s: h, state: v == null ? "none" : v.kind === "kept" ? "kept" : "changed" };
    });
  return { bytetrack: row("bytetrack"), ours: row("ours") };
}

/** One small row of plain settings text (no monospace sentences). */
export function settingsRow(s: BreakSpec): string {
  const n = (k: number, one: string, many: string) => `${k} ${k === 1 ? one : many}`;
  return [
    `Seed ${s.seed}`,
    s.hide_pid == null ? "nobody hidden" : `person ${s.hide_pid} hidden ${s.hide_s} s at the gate`,
    n(s.pillars.length, "pillar", "pillars"),
    n(s.groups.length, "group", "groups"),
    n(s.crowd, "passer-by", "passers-by"),
    s.config === "mem6" ? "ours with longer memory (opt-in)" : "ours on default settings",
  ].join(" · ");
}
