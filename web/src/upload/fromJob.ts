// "Your video" (Studio step 4): turn a finished upload job into the Studio player's ReplayData,
// plus the list of unsure moments. Pure, unit-tested. Wording rule for uploads: there is no ground
// truth, so we only report counts, IDs and events; never accuracy, never "kept".
import type { JobEvent, JobResult } from "../api";
import type { GEvent, ReplayData } from "../guided/model";
import type { Box } from "../lib/scenarios";

export const NO_GT_LABEL = "Your clip · no ground truth";
export const REID_OFF_LABEL = "Re-identification off on CPU";

export interface UnsureMoment {
  f: number; // tracked frame
  t: number; // seconds into the video
  id: number;
  dir: "in" | "out";
  reasons: string[];
  text: string;
}

type GateEv = Extract<JobEvent, { kind: "gate" }>;

const REASON_TEXT: Record<string, string> = {
  occlusion: "partly hidden",
  low_confidence: "weak detections",
};

/** Crossings the tracker was unsure of: ambiguous_hold events (consented clips) or crossings
 * flagged unsure by the same rule (all other clips). */
export function unsureMoments(r: Pick<JobResult, "events" | "video">): UnsureMoment[] {
  const out: UnsureMoment[] = [];
  for (const e of r.events) {
    if (e.kind !== "gate") continue;
    const g = e as GateEv & { unsure?: boolean; reasons?: string[] };
    const held = g.type === "ambiguous_hold";
    if (!held && !(g.type === "crossing" && g.unsure)) continue;
    const reasons = g.reasons ?? [];
    const t = ((g.frame - 1) * r.video.stride) / r.video.fps;
    const why = reasons.map((x) => REASON_TEXT[x] ?? x).join(", ") || "unsure";
    out.push({ f: g.frame, t, id: g.track_id, dir: g.direction, reasons, text: `#${g.track_id} crossed ${g.direction.toUpperCase()} at ${t.toFixed(1)} s, tracker unsure (${why})` });
  }
  return out.sort((a, b) => a.f - b.f);
}

/** Counts and IDs only (no accuracy): what the upload result page is allowed to say. */
export function uploadSummary(r: Pick<JobResult, "events" | "tracks">) {
  let inN = 0, outN = 0, unsure = 0, breaks = 0;
  for (const e of r.events) {
    if (e.kind === "gate" && (e.type === "crossing" || e.type === "ambiguous_hold")) {
      if (e.direction === "in") inN++; else outN++;
      if (e.type === "ambiguous_hold" || (e as { unsure?: boolean }).unsure) unsure++;
    }
    if (e.kind === "identity") breaks++;
  }
  return { ids: Object.keys(r.tracks).length, in: inN, out: outN, unsure, possibleBreaks: breaks };
}

export function fromJobResult(r: JobResult, videoUrl: string | null): ReplayData {
  const frames: Record<string, Box[]> = {};
  for (const [id, rows] of Object.entries(r.tracks)) {
    for (const [f, x, y, w, h] of rows) (frames[String(f)] ??= []).push([Number(id), x, y, w, h]);
  }
  const cross: GEvent[] = [];
  for (const e of r.events) {
    if (e.kind === "gate" && (e.type === "crossing" || e.type === "ambiguous_hold"))
      cross.push({ t: "cross", f: e.frame, id: e.track_id, dir: e.direction });
  }
  const fps = r.video.fps / r.video.stride;
  return {
    id: r.clip.id,
    title: r.clip.title,
    kind: "illustrative",
    width: r.video.width,
    height: r.video.height,
    fps,
    frames: r.frames_tracked,
    duration: r.frames_tracked / fps,
    videoUrl,
    runs: {
      ours: { label: "Ours · BoT-SORT + visibility (CPU, no re-ID)", short: "ours", frames },
      baseline: { label: "not run", short: "ByteTrack", frames: {} },
    },
    gate: { a: r.gate.a, b: r.gate.b },
    events: { ours: cross.sort((a, b) => a.f - b.f), baseline: [] },
    moments: [],
    caption: `${NO_GT_LABEL} · ${REID_OFF_LABEL}`,
  };
}

/** Normalised gate [x1, y1, x2, y2] from two clicks on an image of size w x h. */
export function gateFromClicks(a: [number, number], b: [number, number], w: number, h: number): [number, number, number, number] {
  const n = (v: number, d: number) => Math.round(Math.min(1, Math.max(0, v / d)) * 1e4) / 1e4;
  return [n(a[0], w), n(a[1], h), n(b[0], w), n(b[1], h)];
}

// ---- possible missed crossings: suggestions an operator confirms or dismisses -----------------
// A recovered_crossing event (one number ended at the line, a new one began just across it) is
// never counted automatically. IN/OUT stays the tracker's plain count; only crossings an operator
// confirms go into a separate, labelled "operator-adjusted" count. State lives on the client.

export type Decision = "confirmed" | "dismissed";
export type Decisions = Record<string, Decision>;
export interface Suggestion { key: string; f: number; t: number; from: number; to: number; dir: "in" | "out"; text: string }

type RecEv = Extract<JobEvent, { type: "recovered_crossing" }>;

/** mm:ss of a time in seconds. */
export const mmss = (t: number) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export function suggestions(r: Pick<JobResult, "events" | "video">): Suggestion[] {
  return r.events
    .filter((e): e is RecEv => e.kind === "gate" && e.type === "recovered_crossing")
    .map((e) => {
      const t = ((e.frame - 1) * r.video.stride) / r.video.fps;
      return { key: `${e.frame}:${e.from_track_id}:${e.track_id}`, f: e.frame, t, from: e.from_track_id, to: e.track_id, dir: e.direction, text: `Possible missed crossing at ${mmss(t)} · #${e.from_track_id} → #${e.track_id}` };
    })
    .sort((a, b) => a.f - b.f);
}

/** Suggestions still on screen: dismissed ones disappear. */
export const visibleSuggestions = (s: Suggestion[], d: Decisions) => s.filter((x) => d[x.key] !== "dismissed");

/** Plain IN/OUT plus confirmed suggestions. The plain count itself is never changed. */
export function adjustedCounts(plain: { in: number; out: number }, s: Suggestion[], d: Decisions) {
  const ok = s.filter((x) => d[x.key] === "confirmed");
  return { in: plain.in + ok.filter((x) => x.dir === "in").length, out: plain.out + ok.filter((x) => x.dir === "out").length, confirmed: ok.length };
}

const csvCell = (v: string | number) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

/** Crossings CSV: every plain crossing and every suggestion, with the operator's decision. */
export function crossingsCsv(r: Pick<JobResult, "events" | "video">, d: Decisions): string {
  const head = ["frame", "time_s", "time", "kind", "track_id", "from_track_id", "direction", "operator_decision", "in_plain_count", "in_operator_adjusted_count"];
  const rows: (string | number)[][] = [];
  for (const e of r.events) {
    if (e.kind !== "gate" || (e.type !== "crossing" && e.type !== "ambiguous_hold")) continue;
    const t = ((e.frame - 1) * r.video.stride) / r.video.fps;
    rows.push([e.frame, t.toFixed(2), mmss(t), e.type, e.track_id, "", e.direction, "", "yes", "yes"]);
  }
  for (const s of suggestions(r)) {
    const dec = d[s.key] ?? "pending";
    rows.push([s.f, s.t.toFixed(2), mmss(s.t), "possible_missed_crossing", s.to, s.from, s.dir, dec, "no", dec === "confirmed" ? "yes" : "no"]);
  }
  rows.sort((a, b) => Number(a[0]) - Number(b[0]));
  return [head, ...rows].map((r) => r.map(csvCell).join(",")).join("\n") + "\n";
}
