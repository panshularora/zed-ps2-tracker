export const DASH = "—";

/** A number with fixed decimals, or an em dash when not measured. */
export function num(v: number | null | undefined, digits = 1): string {
  return v == null || Number.isNaN(v) ? DASH : v.toFixed(digits);
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v == null || Number.isNaN(v) ? DASH : `${(v * 100).toFixed(digits)} %`;
}

export function int(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? DASH : Math.round(v).toLocaleString("en-US");
}

export function seconds(v: number | null | undefined): string {
  if (v == null) return DASH;
  return v < 60 ? `${v.toFixed(1)} s` : `${Math.floor(v / 60)} min ${Math.round(v % 60)} s`;
}
