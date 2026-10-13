import type { ReactNode } from "react";

// One-line plain-language definitions for the dashboard's technical terms.
export const TERMS: Record<string, string> = {
  HOTA: "Overall tracking score: balances finding people and keeping their IDs (higher is better).",
  IDF1: "Share of each person's path that kept the right ID (higher is better).",
  IDSW: "ID switches: times a tracked person was given a different ID (lower is better).",
  MOTA: "Classic accuracy: penalises misses, false boxes and ID switches (higher is better).",
  AssA: "Association accuracy: how well IDs stay with the same person (higher is better).",
  DetA: "Detection accuracy: how well people are found in each frame (higher is better).",
  "p95 ms": "Per-frame time that 95 % of frames finish within.",
  MAE: "Mean absolute error of the estimate.",
  "Spearman ρ": "Rank correlation between estimate and truth (1 = same order).",
  AUROC: "How well the score separates the two classes (1 = perfectly).",
  AP: "Average precision of the score for the class (higher is better).",
};

export function Term({ name, children }: { name: string; children?: ReactNode }) {
  const tip = TERMS[name];
  if (!tip) return <>{children ?? name}</>;
  return (
    <span className="term" tabIndex={0} data-tip={tip} aria-label={`${name}: ${tip}`}>
      {children ?? name}
    </span>
  );
}
