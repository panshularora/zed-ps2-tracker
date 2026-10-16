// The ONE source of the landing "What we found" checks and the Results page header:
// docs/results/summary.json (written by `make phase3`, eval/phase3.py), read at build time.
// Every number shown comes from that file; a missing key renders as "pending". Nothing here is a
// hard-coded result, so a win or a NO-GO both render without code changes.
const files = import.meta.glob<SummaryFile>("../../../docs/results/summary.json", { eager: true, import: "default" });

export interface RanRow {
  id: string;
  cache: string;
  saw_eval_half: boolean;
  laptop_detector?: boolean;
}

export interface MetricSet {
  MOTA?: number;
  IDF1?: number;
  IDSW?: number;
  FP?: number;
  FN?: number;
}

export interface SummaryFile {
  plan?: string;
  ran?: RanRow[];
  waiting?: string[];
  excluded?: string[];
  reproduction?: {
    reference?: MetricSet;
    ours?: MetricSet;
    within_tolerance?: Record<string, boolean>;
    verdict?: string;
  };
  go_no_go_v2?: string;
  /** optional flat metric map, e.g. { "mot17.p95_ms": 3.2, "delayed.idsw_k0": 183 } */
  metrics?: Record<string, number | null | undefined>;
}

export type Pill = "PASS" | "NOT REPRODUCED" | "NO-GO" | "PROVISIONAL" | "PENDING";

export interface Check {
  id: "leak" | "reproduction" | "go_no_go" | "latency";
  title: string;
  pill: Pill;
  sentence: string;
  /** evidence lines; `struck` lines are shown struck through with their note */
  evidence: { text: string; struck?: string }[];
}

export const PENDING = "pending";
export const LEAK_NOTE = "excluded: saw the evaluation half";
/** p95 tracker latency budget per frame (the go/no-go v2 rule, configs/phase3/plan.yaml) */
export const LATENCY_BUDGET_MS = 20;

export function loadSummary(): SummaryFile | null {
  const v = Object.values(files)[0];
  return v ?? null;
}

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const fmt = (v: number | undefined, d: number): string =>
  fin(v) ? (d === 0 ? Math.round(v).toLocaleString("en-US") : v.toFixed(d)) : PENDING;

/** Rows that may appear in any reported table: never a detector that saw the evaluation half. */
export function cleanRuns(s: SummaryFile | null): RanRow[] {
  return (s?.ran ?? []).filter((r) => !r.saw_eval_half);
}

export function reproduced(s: SummaryFile | null): boolean {
  return s?.reproduction?.verdict === "REPRODUCED";
}

/** The muted "Provisional" banner shows unless the ByteTrack reproduction passed. */
export function showProvisionalBanner(s: SummaryFile | null): boolean {
  return !reproduced(s);
}

export function checks(s: SummaryFile | null): Check[] {
  const out: Check[] = [];
  // 1. leak check
  const ran = s?.ran ?? [];
  const leaked = ran.filter((r) => r.saw_eval_half);
  out.push(
    !s?.ran
      ? { id: "leak", title: "Leak check", pill: "PENDING", sentence: "No results file yet.", evidence: [] }
      : {
          id: "leak",
          title: "Leak check",
          pill: "PASS",
          sentence:
            leaked.length === 0
              ? "No detector was trained on the frames it is scored on."
              : `${leaked.length} run${leaked.length > 1 ? "s" : ""} used a detector trained on the scored frames, so ${leaked.length > 1 ? "they are" : "it is"} left out of every result.`,
          evidence: ran.map((r) => (r.saw_eval_half ? { text: r.id, struck: LEAK_NOTE } : { text: r.id })),
        },
  );
  // 2. reproduction
  const rp = s?.reproduction;
  if (!rp?.verdict) {
    out.push({ id: "reproduction", title: "Baseline reproduction", pill: "PENDING", sentence: "ByteTrack has not been re-run yet.", evidence: [] });
  } else {
    const keys: [keyof MetricSet, number][] = [["MOTA", 1], ["IDF1", 1], ["IDSW", 0], ["FP", 0], ["FN", 0]];
    const ok = rp.verdict === "REPRODUCED";
    out.push({
      id: "reproduction",
      title: "Baseline reproduction",
      pill: ok ? "PASS" : "NOT REPRODUCED",
      sentence: ok
        ? "Our ByteTrack run matches the published MOT17 numbers, so the test setup is sound."
        : "Our ByteTrack run does not match the published MOT17 numbers yet, so the detector setup is being fixed and re-run.",
      evidence: keys
        .filter(([k]) => fin(rp.ours?.[k]) || fin(rp.reference?.[k]))
        .map(([k, d]) => ({ text: `${k} ${fmt(rp.ours?.[k], d)} vs ${fmt(rp.reference?.[k], d)} published` })),
    });
  }
  // 3. go / no-go
  const g = s?.go_no_go_v2;
  out.push(
    !g
      ? { id: "go_no_go", title: "Go / no-go (delayed assignment)", pill: "PENDING", sentence: "Not decided yet.", evidence: [] }
      : {
          id: "go_no_go",
          title: "Go / no-go (delayed assignment)",
          pill: g.startsWith("GO") ? "PASS" : "NO-GO",
          sentence: g.startsWith("GO")
            ? "The rule we wrote down before the run says the delayed assignment helps."
            : "The rule we wrote down before the run says the delayed assignment does not help, so it stays off.",
          evidence: [{ text: `go_no_go_v2: ${g}` }, { text: "rule: docs/preregistration/go_no_go_v2.md" }],
        },
  );
  // 4. latency
  const p95 = s?.metrics?.["mot17.p95_ms"];
  out.push(
    !fin(p95)
      ? { id: "latency", title: "Speed", pill: "PENDING", sentence: "Per-frame timing is not in the results file yet.", evidence: [] }
      : {
          id: "latency",
          title: "Speed",
          pill: !reproduced(s) ? "PROVISIONAL" : p95 <= LATENCY_BUDGET_MS ? "PASS" : "NO-GO",
          sentence: `The tracker needs at most ${p95.toFixed(1)} ms for 95% of frames.`,
          evidence: [{ text: `p95 ${p95.toFixed(1)} ms · budget ${LATENCY_BUDGET_MS} ms` }],
        },
  );
  return out;
}

export interface MethodResult {
  before: string;
  after: string;
  held: string;
  pending: boolean;
}

/** Delayed-assignment before/after (IDSW K=0 vs K=5, share of frames with held boxes). */
export function methodResult(s: SummaryFile | null): MethodResult {
  const m = s?.metrics ?? {};
  const k0 = m["delayed.idsw_k0"];
  const k5 = m["delayed.idsw_k5"];
  const held = m["delayed.held_frames_pct"];
  return {
    before: fin(k0) ? Math.round(k0).toString() : PENDING,
    after: fin(k5) ? Math.round(k5).toString() : PENDING,
    held: fin(held) ? `${held.toFixed(0)}%` : PENDING,
    pending: !fin(k0) || !fin(k5),
  };
}

// ---- overall comparison across datasets (summary.json only) ---------------------------------
export type TrackerKey = "bytetrack" | "botsort" | "ours";
export const TRACKERS: { key: TrackerKey; label: string }[] = [
  { key: "bytetrack", label: "ByteTrack" },
  { key: "botsort", label: "BoT-SORT" },
  { key: "ours", label: "BoT-SORT + visibility (ours)" },
];
export type CellTag = "provisional" | "not run yet" | "ran, no metric in summary.json" | "no ground truth" | "excluded";
export interface Cell { tag: CellTag; values?: { label: string; value: string }[] }
export interface DatasetRow { id: string; dataset: string; status: string; cells: Record<TrackerKey, Cell> }

const all = (c: Cell): Record<TrackerKey, Cell> => ({ bytetrack: c, botsort: c, ours: c });

/** Rows for MOT17, MOT20, DanceTrack, Pexels and the synthetic scene. Every value is read from
 *  summary.json (reproduction block or the flat `metrics` map "<dataset>.<tracker>.<METRIC>");
 *  anything absent is tagged, never filled in. `synthetic` is an optional seeded backend run. */
export function datasetRows(s: SummaryFile | null, synthetic?: { seed: number; rows: Record<TrackerKey, { idChanges: number; counted: number; truth: number } | undefined> } | null): DatasetRow[] {
  const ran = s?.ran ?? [];
  const waiting = s?.waiting ?? [];
  const metric = (ds: string, t: TrackerKey): Cell | null => {
    const vals = ["HOTA", "IDF1", "MOTA", "IDSW"].flatMap((m) => {
      const v = s?.metrics?.[`${ds}.${t}.${m}`];
      return fin(v) ? [{ label: m, value: m === "IDSW" ? String(Math.round(v)) : v.toFixed(1) }] : [];
    });
    return vals.length ? { tag: "provisional", values: vals } : null;
  };
  const row = (id: string, dataset: string, prefix: string): DatasetRow => {
    const runs = ran.filter((r) => r.id.toLowerCase().startsWith(prefix));
    const clean = runs.filter((r) => !r.saw_eval_half);
    const isWaiting = waiting.some((w) => w.toLowerCase().startsWith(prefix));
    let status: string;
    if (clean.length) status = `ran (${clean.length} run${clean.length > 1 ? "s" : ""}${runs.length > clean.length ? `, ${runs.length - clean.length} leaked, excluded` : ""})`;
    else if (runs.length) status = "leaked, excluded";
    else status = isWaiting ? `not run yet: ${waiting.find((w) => w.toLowerCase().startsWith(prefix))?.split(": ").slice(1).join(": ")}` : "not run yet";
    const base: Cell = clean.length ? { tag: "ran, no metric in summary.json" } : { tag: runs.length ? "excluded" : "not run yet" };
    const cells = all(base);
    for (const t of TRACKERS) cells[t.key] = metric(id, t.key) ?? base;
    return { id, dataset, status, cells };
  };
  const mot17 = row("mot17", "MOT17 val-half", "mot17");
  const rep = s?.reproduction?.ours;
  if (rep && mot17.cells.bytetrack.tag !== "provisional") {
    const v = [
      ...(fin(rep.MOTA) ? [{ label: "MOTA", value: rep.MOTA.toFixed(1) }] : []),
      ...(fin(rep.IDF1) ? [{ label: "IDF1", value: rep.IDF1.toFixed(1) }] : []),
      ...(fin(rep.IDSW) ? [{ label: "IDSW", value: String(rep.IDSW) }] : []),
    ];
    if (v.length) mot17.cells.bytetrack = { tag: "provisional", values: v };
  }
  const syn: DatasetRow = { id: "synthetic", dataset: "Synthetic gate scene", status: "not run yet", cells: all({ tag: "not run yet" }) };
  if (synthetic) {
    syn.status = `one seeded backend run, seed ${synthetic.seed} (exact ground truth)`;
    for (const t of TRACKERS) {
      const r = synthetic.rows[t.key];
      if (r) syn.cells[t.key] = { tag: "provisional", values: [{ label: "ID changes", value: String(r.idChanges) }, { label: "counted / true", value: `${r.counted} / ${r.truth}` }] };
    }
  }
  return [
    mot17,
    row("mot20", "MOT20 val-half", "mot20"),
    row("dancetrack", "DanceTrack val", "dancetrack"),
    row("sompt22", "SOMPT22", "sompt22"),
    { id: "pexels", dataset: "Pexels demo clips", status: "illustrative only", cells: all({ tag: "no ground truth" }) },
    syn,
  ];
}

// Recorded seeded run of the synthetic gate scene (scripts/web/export_gate_sim.py), if committed.
const synFiles = import.meta.glob<{ params: { seed: number }; summary: Record<string, { id_changes: number; counted: number; truth: number }> }>(
  "../../../docs/results/synthetic_gate/seed*.json", { eager: true, import: "default" },
);
export function loadSynthetic(): Parameters<typeof datasetRows>[1] {
  const f = Object.values(synFiles)[0];
  if (!f) return null;
  const pick = (k: string) => (f.summary[k] ? { idChanges: f.summary[k].id_changes, counted: f.summary[k].counted, truth: f.summary[k].truth } : undefined);
  return { seed: f.params.seed, rows: { bytetrack: pick("bytetrack"), botsort: pick("botsort"), ours: pick("ours") } };
}
