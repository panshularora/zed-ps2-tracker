// The ONE source of every number on the landing page: docs/results/summary.json, read at build
// time. The file is written by the GPU run; until it exists (or while a key is missing) the card
// shows "pending". Nothing here is a hard-coded result.
const files = import.meta.glob<SummaryFile>("../../../docs/results/summary.json", { eager: true, import: "default" });

export interface SummaryFile {
  /** flat metric map, e.g. { "mot17.idsw_reduction_pct": 12.3 } */
  metrics?: Record<string, number | null | undefined>;
  source?: string;
}

export interface CardSpec {
  key: string;
  unit: string;
  sentence: string;
  metric: string;
  decimals: number;
  /** true for the "where we didn't win" card */
  loss?: boolean;
}

export const CARDS: CardSpec[] = [
  { key: "mot17.idsw_reduction_pct", unit: "%", decimals: 0, sentence: "fewer times a person is given a new ID by mistake", metric: "IDSW vs best baseline · MOT17 val" },
  { key: "mot20.idsw_reduction_pct", unit: "%", decimals: 0, sentence: "fewer ID mix-ups in very dense crowds", metric: "IDSW vs best baseline · MOT20 val" },
  { key: "mot17.idf1_gain_pts", unit: " pts", decimals: 1, sentence: "more of each person's path kept under one ID", metric: "IDF1 gain · MOT17 val" },
  { key: "mot17.p95_ms", unit: " ms", decimals: 1, sentence: "worst-case time to track one frame, fast enough for a live gate", metric: "tracker p95 latency per frame" },
  { key: "mot20.deta_delta_pts", unit: " pts", decimals: 1, sentence: "Where we didn't win: finding people in the first place is no better, because every tracker shares the same detections", metric: "DetA difference · MOT20 val", loss: true },
];

export const PENDING = "pending";

export function loadSummary(): SummaryFile | null {
  const v = Object.values(files)[0];
  return v ?? null;
}

/** Value text for a card, or "pending" when the summary or the key is missing / not finite. */
export function cardValue(spec: CardSpec, summary: SummaryFile | null): string {
  const v = summary?.metrics?.[spec.key];
  if (typeof v !== "number" || !Number.isFinite(v)) return PENDING;
  const sign = spec.key.includes("delta") && v > 0 ? "+" : "";
  return `${sign}${v.toFixed(spec.decimals)}${spec.unit}`;
}
