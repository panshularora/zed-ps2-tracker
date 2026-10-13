export function pct(n: number, digits = 2): string {
  return `${n.toFixed(digits)}%`;
}

export function num(n: number, digits = 2): string {
  return n.toFixed(digits);
}

export function ms(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `${n.toFixed(digits)} ms`;
}

export function padFrame(f: number, width = 4): string {
  return String(f).padStart(width, "0");
}

export function timecode(frame: number, fps: number): string {
  const total = frame / Math.max(fps, 1);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${String(mins).padStart(2, "0")}:${secs.toFixed(3).padStart(6, "0")}`;
}

export function delta(a: number, b: number, digits = 2, invert = false): {
  text: string;
  good: boolean;
  zero: boolean;
} {
  const d = a - b;
  const zero = Math.abs(d) < 1e-9;
  const good = invert ? d < 0 : d > 0;
  const sign = d > 0 ? "+" : "";
  return { text: `${sign}${d.toFixed(digits)}`, good, zero };
}
