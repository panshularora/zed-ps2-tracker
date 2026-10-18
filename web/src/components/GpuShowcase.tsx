// Results tab, GPU campaign: scale strip, the re-ID rule headline, a leaderboard per evaluation,
// ours vs our own ByteTrack run, the paper reproduction (grey reference), the go/no-go that was
// dropped on purpose and a quiet "where we lose" row. Every number comes from GET /api/benchmarks
// (serve/benchmarks.py reading docs/results/phase3 and docs/results/stress); nothing is typed in.
import { useState } from "react";
import type { GpuRow, GpuRun as Gpu, GpuScale, ReidPair } from "../api";
import { offlinePath } from "../lib/offline";
import { DASH, int, num } from "../lib/format";
import { Term } from "./Term";

export const DATASET_LABEL: Record<string, string> = {
  mot17_val_half: "MOT17 val-half",
  mot20_val_half__sparsetrack_ablation: "MOT20 val-half · SparseTrack det",
  mot20_val_half__mot17full_xdomain: "MOT20 val-half · MOT17-trained det",
  dancetrack_val: "DanceTrack val",
};
export const dsLabel = (id: string) => DATASET_LABEL[id] ?? id.replace(/_/g, " ");

export type MetricKey = "HOTA" | "IDF1" | "AssA" | "MOTA" | "IDSW" | "FP" | "FN" | "p95_ms";
export const METRICS: { key: MetricKey; label: string; lower: boolean; nd: number }[] = [
  { key: "HOTA", label: "HOTA", lower: false, nd: 1 },
  { key: "IDF1", label: "IDF1", lower: false, nd: 1 },
  { key: "AssA", label: "AssA", lower: false, nd: 1 },
  { key: "MOTA", label: "MOTA", lower: false, nd: 1 },
  { key: "IDSW", label: "IDSW", lower: true, nd: 0 },
  { key: "FP", label: "FP", lower: true, nd: 0 },
  { key: "FN", label: "FN", lower: true, nd: 0 },
  { key: "p95_ms", label: "p95 ms", lower: true, nd: 2 },
];
/** The tracker the write-up calls "ours" (BoT-SORT + re-ID + visibility rule). */
export const OURS = "botsort_reid_vis_gated";
const BASE = "bytetrack";
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

const val = (r: GpuRow, k: MetricKey): number | null => {
  const v = r[k];
  return typeof v === "number" ? v : null;
};
const fmt = (v: number | null | undefined, nd: number) => (v == null ? DASH : nd === 0 ? int(v) : num(v, nd));
const signed = (v: number, nd: number) => `${v > 0 ? "+" : v < 0 ? "−" : "±"}${nd === 0 ? int(Math.abs(v)) : Math.abs(v).toFixed(nd)}`;

/** better / worse / same of ``a`` vs ``b`` for a metric, judged by its direction. */
export function judge(a: number | null, b: number | null, lower: boolean): "better" | "worse" | "same" | null {
  if (a == null || b == null) return null;
  if (a === b) return "same";
  return (lower ? a < b : a > b) ? "better" : "worse";
}

/** The best value of each column (direction-aware), for bolding. */
export function bestOf(rows: GpuRow[]): Partial<Record<MetricKey, number>> {
  const out: Partial<Record<MetricKey, number>> = {};
  for (const m of METRICS) {
    const vs = rows.map((r) => val(r, m.key)).filter((v): v is number => v != null);
    if (vs.length) out[m.key] = m.lower ? Math.min(...vs) : Math.max(...vs);
  }
  return out;
}

function Card({ n, label, sub, testid }: { n: string; label: string; sub?: string; testid: string }) {
  return (
    <div className="stat-card" data-testid={testid}>
      <div className="stat-n">{n}</div>
      <div className="stat-l">{label}</div>
      {sub && <div className="stat-s">{sub}</div>}
    </div>
  );
}

export function ScaleStrip({ s, gpu, datasets }: { s: GpuScale; gpu: string | null; datasets: Gpu["datasets"] }) {
  const configs = new Set(datasets.flatMap((d) => d.rows.map((r) => r.name))).size;
  return (
    <section className="section" data-testid="gpu-scale">
      <h2>Scale of the GPU campaign</h2>
      <div className="stat-strip">
        <div className="stat-group" data-testid="scale-real">
          <Card testid="scale-evaluations" n={int(s.evaluations)} label="clean evaluations" sub={datasets.map((d) => dsLabel(d.id)).join(" · ")} />
          <Card testid="scale-runs" n={int(s.tracker_runs)} label="tracker runs scored" sub={`${configs} tracker configs · ${datasets.map((d) => d.rows.length).join(" + ")}`} />
          <Card testid="scale-frames" n={int(s.frames)} label="frames scored" sub={`${int(s.sequences)} sequences · ${int(s.tracked_frames)} frames tracked over all runs`} />
          {s.go_no_go && <Card testid="scale-gng" n="1" label="pre-registered go/no-go" sub={`${s.go_no_go.verdict ?? DASH}: failed ${s.go_no_go.failed}/${s.go_no_go.total}, dropped`} />}
          <Card testid="scale-excluded" n={int(s.excluded)} label="run excluded for leakage" sub="detector saw the eval half; shown openly, in no table" />
        </div>
        {s.stress && (
          <div className="stat-group sim" data-testid="scale-sim">
            <Card testid="scale-stress" n={int(s.stress.runs)} label="simulated stress scenes" sub={`${s.stress.cells} cells × ${s.stress.seeds_per_cell ?? DASH} seeds (${s.stress.seeds[0]}–${s.stress.seeds[1]}) × ${s.stress.trackers} trackers · synthetic, not a benchmark`} />
          </div>
        )}
      </div>
      <p className="caption" data-testid="scale-hw">
        {gpu ?? DASH} · YOLOX-X {s.precision.join("/")} ({s.detectors.length} detector weights: {s.detectors.join(", ")}) · re-ID {s.reid.join(", ")} · TrackEval. Counted from docs/results/phase3{s.stress ? ` and ${s.stress.source}` : ""}.
      </p>
    </section>
  );
}

/** Title only claims what the rows show. */
export function reidTitle(p: ReidPair): string {
  const n = p.rows.length;
  const w = WORDS[n] ?? String(n);
  if (p.fewer_idsw_everywhere && p.hota_never_lower) return `One rule, ${w} datasets: fewer ID switches every time`;
  const fewer = p.rows.filter((r) => r.d_IDSW < 0).length;
  return `One rule, ${w} datasets: fewer ID switches on ${fewer} of ${n}`;
}

function Bar({ v, max, good }: { v: number; max: number; good: boolean }) {
  const w = max > 0 ? Math.round((40 * Math.abs(v)) / max) : 0;
  return <span className={`dbar ${good ? "good" : "bad"}`} style={{ width: w }} aria-hidden="true" />;
}

export function ReidRuleCard({ pair, other }: { pair: ReidPair; other?: ReidPair }) {
  const maxH = Math.max(...pair.rows.map((r) => Math.abs(r.d_HOTA)), 0);
  const pctI = (r: ReidPair["rows"][number]) => (r.base.IDSW ? (100 * r.d_IDSW) / r.base.IDSW : 0);
  const maxI = Math.max(...pair.rows.map((r) => Math.abs(pctI(r))), 0);
  return (
    <section className="section" data-testid="reid-headline">
      <h2 data-testid="reid-title">{reidTitle(pair)}</h2>
      <p className="sub">Re-ID alone adds ID switches; the visibility rule on top of it takes them back out. Same detections and same re-ID model in each pair; only the rule differs.</p>
      <div className="table-wrap">
        <table className="lb">
          <thead>
            <tr>
              <th>Evaluation</th>
              <th>{pair.family} + re-ID: <Term name="HOTA" /> / <Term name="IDSW" /></th>
              <th className="ours-h">+ visibility rule (ours): HOTA / IDSW</th>
              <th>Δ HOTA</th>
              <th>Δ IDSW</th>
            </tr>
          </thead>
          <tbody>
            {pair.rows.map((r) => (
              <tr key={r.dataset} data-testid={`reid-${r.dataset}`}>
                <td>{dsLabel(r.dataset)}</td>
                <td>{num(r.base.HOTA)} / {int(r.base.IDSW)}</td>
                <td>{num(r.ours.HOTA)} / {int(r.ours.IDSW)}</td>
                <td className={r.d_HOTA > 0 ? "better" : r.d_HOTA < 0 ? "worse" : ""} data-testid={`reid-dh-${r.dataset}`}>
                  {signed(r.d_HOTA, 2)} <Bar v={r.d_HOTA} max={maxH} good={r.d_HOTA >= 0} />
                </td>
                <td className={r.d_IDSW < 0 ? "better" : r.d_IDSW > 0 ? "worse" : ""} data-testid={`reid-di-${r.dataset}`}>
                  {signed(r.d_IDSW, 0)} ({signed(pctI(r), 0)} %) <Bar v={pctI(r)} max={maxI} good={r.d_IDSW <= 0} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Rows: {pair.rows[0]?.base.label} vs {pair.rows[0]?.ours.label}, read from each evaluation's ablation.json.
        {other && ` The same rule on BoT-SORT + re-ID: fewer ID switches on ${other.rows.filter((r) => r.d_IDSW < 0).length} of ${other.rows.length}, HOTA ${other.hota_never_lower ? "never lower" : "lower on some"}.`}
      </p>
    </section>
  );
}

export function Leaderboard({ rows }: { rows: GpuRow[] }) {
  const sorted = [...rows].sort((a, b) => (b.HOTA ?? -Infinity) - (a.HOTA ?? -Infinity));
  const best = bestOf(rows);
  return (
    <div className="table-wrap">
      <table className="lb" data-testid="leaderboard">
        <thead>
          <tr>
            <th>#</th>
            <th>Tracker</th>
            {METRICS.map((m) => <th key={m.key}><Term name={m.label} />{m.lower ? " ↓" : ""}</th>)}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.name} className={r.group === "ours" ? "bar-ours lb-ours" : "bar-baseline"} data-testid={`gpu-row-${r.name}`}>
              <td className="muted">{i + 1}</td>
              <td>
                {r.label} {r.group === "control" && <span className="tag">control</span>}
                {r.group === "ours" && <span className="tag ours">ours</span>}
              </td>
              {METRICS.map((m) => {
                const v = val(r, m.key);
                return <td key={m.key} className={v != null && v === best[m.key] ? "best" : ""}>{fmt(v, m.nd)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function OwnBaseline({ rows }: { rows: GpuRow[] }) {
  const a = rows.find((r) => r.name === OURS);
  const b = rows.find((r) => r.name === BASE);
  if (!a || !b) return null;
  return (
    <div className="panel check own-baseline" data-testid="own-baseline">
      <h3>Ours vs our own ByteTrack run (same detections)</h3>
      <div className="table-wrap">
      <table className="lb">
        <thead>
          <tr><th>Metric</th><th>{b.label}</th><th className="ours-h">{a.label}</th><th>Δ ours − ByteTrack</th></tr>
        </thead>
        <tbody>
          {METRICS.map((m) => {
            const x = val(a, m.key);
            const y = val(b, m.key);
            const j = judge(x, y, m.lower);
            return (
              <tr key={m.key} data-testid={`own-${m.key}`}>
                <td>{m.label}{m.lower ? " (lower is better)" : ""}</td>
                <td>{fmt(y, m.nd)}</td>
                <td>{fmt(x, m.nd)}</td>
                <td className={j === "better" ? "better" : j === "worse" ? "worse" : ""} data-state={j ?? ""}>
                  {x == null || y == null ? DASH : `${signed(x - y, m.nd)} · ${j}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      <p className="caption">The fair comparison: both rows track the same cached YOLOX-X boxes. Blue = better, amber = worse, judged by direction.</p>
    </div>
  );
}

export function PaperReference({ rep }: { rep: NonNullable<Gpu["reproduction"]> }) {
  const r = (m: string) => rep.rows.find((x) => x.metric === m);
  const d = (m: string) => r(m)?.delta;
  return (
    <div className="panel check paper-ref" data-testid="gpu-reproduction">
      <div className="check-head">
        <h3>ByteTrack paper vs our ByteTrack rerun (reference only)</h3>
        <span className="pill fail" data-testid="gpu-pill" data-state={rep.verdict}><span aria-hidden="true">✕ </span>{rep.verdict}</span>
      </div>
      <p data-testid="rep-line">
        Not reproduced: our rerun is {d("MOTA") == null ? DASH : Math.abs(d("MOTA") as number).toFixed(1)} MOTA below the paper, mostly from detection
        ({d("FP") == null ? DASH : signed(d("FP") as number, 0)} FP, {d("FN") == null ? DASH : signed(d("FN") as number, 0)} FN). All GPU scores are provisional; this table only shows the reproduction gap.
      </p>
      <table>
        <thead>
          <tr><th>Metric</th><th>Paper</th><th>Our ByteTrack rerun</th><th>Δ</th><th>Tolerance</th></tr>
        </thead>
        <tbody>
          {rep.rows.map((x) => (
            <tr key={x.metric} data-testid={`rep-${x.metric}`}>
              <td>{x.metric}</td>
              <td>{x.paper ?? DASH}</td>
              <td>{x.ours ?? DASH}</td>
              <td>{x.delta == null ? DASH : `${x.delta > 0 ? "+" : ""}${x.delta}`}</td>
              <td>{x.tolerance == null ? <span className="muted">reported only</span> : <>±{x.tolerance} {x.within ? "" : <span className="tag">outside</span>}</>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">Source: {rep.source} ({rep.reference}, MOT17 val-half). Paper values only where the paper reports this split.</p>
    </div>
  );
}

export function NoGoCard({ gng }: { gng: NonNullable<Gpu["go_no_go_v2"]> }) {
  const failed = gng.criteria.filter((c) => !c.passed).length;
  return (
    <div className="panel check" data-testid="gpu-go-no-go">
      <div className="check-head">
        <h3>Tested an idea, failed {failed}/{gng.criteria.length}, dropped</h3>
        <span className="pill fail" data-testid="gpu-pill" data-state={gng.verdict}><span aria-hidden="true">✕ </span>{gng.verdict}</span>
      </div>
      <p>{gng.what} Rules written down before the run; the idea is not in the app.</p>
      <table>
        <thead>
          <tr><th>Criterion</th><th>Needed</th><th>Got (K=0 → K=5)</th><th>Pass</th></tr>
        </thead>
        <tbody>
          {gng.criteria.map((c) => (
            <tr key={c.id} data-testid={`crit-${c.id}`}>
              <td>{c.what}</td>
              <td style={{ whiteSpace: "normal" }}>{c.rule}</td>
              <td style={{ whiteSpace: "normal" }}>{c.got ?? DASH}</td>
              <td>{c.passed ? "yes" : <strong>no</strong>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="caption">Source: {gng.source}; rule frozen in {gng.rule}.</p>
    </div>
  );
}

/** Evaluations where a non-ours tracker has the best HOTA, and the worst ours-vs-ByteTrack latency. */
export function whereWeLose(datasets: Gpu["datasets"]): string[] {
  const out: string[] = [];
  for (const d of datasets) {
    const top = [...d.rows].sort((a, b) => (b.HOTA ?? -Infinity) - (a.HOTA ?? -Infinity))[0];
    const ours = d.rows.filter((r) => r.group === "ours").sort((a, b) => (b.HOTA ?? -Infinity) - (a.HOTA ?? -Infinity))[0];
    if (top && ours && top.group !== "ours") out.push(`${dsLabel(d.id)}: ${top.label} HOTA ${num(top.HOTA)} vs ours ${num(ours.HOTA)}`);
  }
  let worst: { d: string; o: number[]; b: number } | null = null;
  for (const d of datasets) {
    const b = d.rows.find((r) => r.name === BASE)?.p95_ms;
    const o = d.rows.filter((r) => r.group === "ours").map((r) => r.p95_ms).filter((v): v is number => v != null);
    if (b && o.length && (!worst || Math.max(...o) / b > Math.max(...worst.o) / worst.b)) worst = { d: d.id, o, b };
  }
  if (worst) {
    const lo = Math.min(...worst.o);
    const hi = Math.max(...worst.o);
    out.push(`${dsLabel(worst.d)} speed: ours p95 ${num(lo, 0)}–${num(hi, 0)} ms per frame vs ByteTrack ${num(worst.b, 1)} ms`);
  }
  return out;
}

export function GpuShowcase({ g }: { g: Gpu }) {
  const [sel, setSel] = useState(g.datasets[0]?.id ?? "");
  const ds = g.datasets.find((d) => d.id === sel) ?? g.datasets[0];
  const pairs = g.reid_rule?.pairs ?? [];
  const head = pairs.find((p) => p.family === "ByteTrack") ?? pairs[0];
  const other = pairs.find((p) => p !== head);
  const lose = whereWeLose(g.datasets);
  return (
    <>
      {g.scale && <ScaleStrip s={g.scale} gpu={g.gpu} datasets={g.datasets} />}
      {head && <ReidRuleCard pair={head} other={other} />}

      <section className="section" data-testid="gpu-ablation">
        <h2>Every tracker, every evaluation</h2>
        <p className="sub">All rows of each ablation.json, sorted by HOTA. Ours highlighted; best value per column in bold, whoever has it. ↓ = lower is better.</p>
        <div className="seg" role="tablist" aria-label="Evaluation">
          {g.datasets.map((d) => (
            <button key={d.id} role="tab" aria-selected={d.id === ds?.id} className={`chip-btn${d.id === ds?.id ? " on" : ""}`} onClick={() => setSel(d.id)} data-testid={`gpu-ds-${d.id}`}>
              {dsLabel(d.id)}
            </button>
          ))}
        </div>
        {ds && (
          <>
            <Leaderboard rows={ds.rows} />
            <p className="caption">
              Source: {ds.source}. Detector {ds.detector ?? DASH}{ds.trained_on ? ` (${ds.trained_on})` : ""}{ds.input_size ? `, input ${ds.input_size}` : ""}. FP / FN from the same TrackEval run. p95 = tracker <code>update()</code> on the GPU host.
            </p>
            <OwnBaseline rows={ds.rows} />
            {Object.keys(ds.plots).length > 0 && (
              <div className="charts">
                {Object.entries(ds.plots).map(([k, url]) => (
                  <figure className="panel chart" key={`${ds.id}-${k}`} style={{ margin: 0 }}>
                    <figcaption className="label" style={{ marginBottom: 8 }}>{dsLabel(ds.id)}: {k.replace(/_/g, " ")}</figcaption>
                    <img src={url} alt={`${ds.id} ${k}`} loading="lazy" onError={(e) => { const alt = offlinePath(url); if (alt && e.currentTarget.getAttribute("src") !== alt) e.currentTarget.src = alt; }} />
                  </figure>
                ))}
              </div>
            )}
          </>
        )}
      </section>

      <section className="section" data-testid="gpu-verdicts">
        <h2>Checks we ran on ourselves</h2>
        <div className="charts">
          {g.reproduction && <PaperReference rep={g.reproduction} />}
          {g.go_no_go_v2 && <NoGoCard gng={g.go_no_go_v2} />}
        </div>
        {lose.length > 0 && <p className="lose-row muted" data-testid="where-we-lose"><b>Where we lose:</b> {lose.join(" · ")}.</p>}
      </section>
    </>
  );
}
