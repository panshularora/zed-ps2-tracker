// MOT20 status, read at build time from the repo's own result files (never typed in by hand):
// * docs/results/summary.json "ran": which MOT20 detector caches the laptop GPU run produced (no metrics)
// * docs/results/go_no_go.md: the one MOT20 tracker score that exists (CPU, corrupted-gt detections, K = 0)
// * docs/evaluation.md: the MOT20 oracle ceiling (gt boxes as a tracker; not a tracker result)
import { loadSummary } from "./summary";

const raw = import.meta.glob<string>(["../../../docs/results/go_no_go.md", "../../../docs/evaluation.md"], { eager: true, query: "?raw", import: "default" });
const text = (name: string) => Object.entries(raw).find(([k]) => k.endsWith(name))?.[1] ?? "";

/** values kept as the exact strings written in the source file (e.g. "63.0") */
export interface Mot20Score { HOTA: string; IDF1: string; MOTA: string; IDSW?: string; FN?: string }
export interface Mot20Status {
  runs: { id: string; leaked: boolean }[];
  corruptedGtK0: (Mot20Score & { seconds: string }) | null;
  oracle: Mot20Score | null;
  viewerExport: boolean;
}

export function parseGoNoGo(md: string): Mot20Status["corruptedGtK0"] {
  const m = /MOT20[^\n]*\n+K = 0 ran in (\d+) s: HOTA ([\d.]+), IDF1 ([\d.]+), MOTA ([\d.]+), IDSW (\d+)/.exec(md);
  return m ? { seconds: m[1], HOTA: m[2], IDF1: m[3], MOTA: m[4], IDSW: m[5] } : null;
}
export function parseOracle(md: string): Mot20Score | null {
  const m = /\| MOT20 val-half, τ = 0 \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d,]+) \|/.exec(md);
  return m ? { HOTA: m[1], IDF1: m[2], MOTA: m[3], FN: m[4] } : null;
}

export function mot20Status(viewerClips: string[] = []): Mot20Status {
  const s = loadSummary() as { ran?: { id: string; saw_eval_half?: boolean }[] } | null;
  return {
    runs: (s?.ran ?? []).filter((r) => /mot20/i.test(r.id)).map((r) => ({ id: r.id, leaked: !!r.saw_eval_half })),
    corruptedGtK0: parseGoNoGo(text("go_no_go.md")),
    oracle: parseOracle(text("evaluation.md")),
    viewerExport: viewerClips.some((c) => c.startsWith("MOT20")),
  };
}
