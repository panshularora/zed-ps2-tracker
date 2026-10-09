import type { AblationRow } from "../api";
import { theme } from "../theme";

interface Props {
  rows: AblationRow[];
  /** which ablation rows to draw (names) */
  names: string[];
  metric: "idsw_per_1k_gt" | "id_recall";
  title: string;
}

/** Small SVG line chart over the visibility bins of docs/results/ablation/ablation.json. */
export default function StressChart({ rows, names, metric, title }: Props) {
  const series = names.map((n) => rows.find((r) => r.name === n)).filter((r): r is AblationRow => !!r);
  if (!series.length || !series[0].stress_visibility.length) return null;
  const bins = series[0].stress_visibility.map((b) => b.bin);
  const vals = series.flatMap((s) => s.stress_visibility.map((b) => b[metric]));
  const max = Math.max(...vals, 1e-9) * 1.15;
  const W = 520;
  const H = 220;
  const pad = { l: 44, r: 12, t: 12, b: 36 };
  const x = (i: number) => pad.l + ((W - pad.l - pad.r) * (i + 0.5)) / bins.length;
  const y = (v: number) => H - pad.b - ((H - pad.t - pad.b) * v) / max;
  const ticks = [0, max / 2, max].map((v) => Number(v.toFixed(metric === "id_recall" ? 2 : 1)));
  return (
    <figure className="panel chart" style={{ margin: 0 }}>
      <figcaption className="label" style={{ marginBottom: 8 }}>
        {title}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} style={{ width: "100%", height: "auto" }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke={theme.panelBorder} strokeWidth={1} />
            <text x={pad.l - 6} y={y(t) + 4} fill={theme.textSecondary} fontSize={10} textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        {bins.map((b, i) => (
          <text key={b} x={x(i)} y={H - pad.b + 16} fill={theme.textSecondary} fontSize={10} textAnchor="middle">
            {b}
          </text>
        ))}
        <text x={(W + pad.l) / 2} y={H - 4} fill={theme.textSecondary} fontSize={10} textAnchor="middle">
          ground-truth visibility bin
        </text>
        {series.map((s) => {
          const color = s.group === "ours" ? theme.ours : theme.baseline;
          const pts = s.stress_visibility.map((b, i) => `${x(i)},${y(b[metric])}`).join(" ");
          return (
            <g key={s.name}>
              <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} />
              {s.stress_visibility.map((b, i) => (
                <circle key={b.bin} cx={x(i)} cy={y(b[metric])} r={2.5} fill={color} />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.group === "ours" ? theme.ours : theme.baseline }} />
            {s.label}
          </span>
        ))}
      </div>
    </figure>
  );
}
