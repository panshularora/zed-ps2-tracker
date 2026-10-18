// Laptop/DGX GPU run (docs/results/phase3, served by GET /api/benchmarks "gpu"): per-dataset
// ablation tables, the ByteTrack reproduction check and the pre-registered go/no-go v2 with the
// reason each criterion failed. Every value comes from the API response; nothing is typed here.
import { useState } from "react";
import type { GpuRun as Gpu, GpuRow } from "../api";
import { offlinePath } from "../lib/offline";
import { DASH, int, num, pct } from "../lib/format";
import { Term } from "./Term";

const DATASET_LABEL: Record<string, string> = {
  mot17_val_half: "MOT17 val-half",
  mot20_val_half__sparsetrack_ablation: "MOT20 val-half",
  mot20_val_half__mot17full_xdomain: "MOT20 (MOT17 detector)",
  dancetrack_val: "DanceTrack val",
};
const label = (id: string) => DATASET_LABEL[id] ?? id.replace(/_/g, " ");

function Pill({ verdict }: { verdict: string | null | undefined }) {
  const ok = verdict === "GO" || verdict === "REPRODUCED";
  return (
    <span className={ok ? "pill pass" : "pill fail"} data-testid="gpu-pill" data-state={verdict ?? ""}>
      {!ok && <span aria-hidden="true">✕ </span>}
      {verdict ?? DASH}
    </span>
  );
}

export function GpuAblation({ rows }: { rows: GpuRow[] }) {
  const v = (k: "HOTA" | "IDF1" | "MOTA" | "AssA") => rows.map((r) => r[k] ?? -Infinity);
  const best = (k: "HOTA" | "IDF1" | "MOTA" | "AssA") => Math.max(...v(k));
  const minIdsw = Math.min(...rows.map((r) => r.IDSW ?? Infinity));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Tracker</th>
            <th><Term name="HOTA" /></th>
            <th><Term name="IDF1" /></th>
            <th><Term name="MOTA" /></th>
            <th><Term name="IDSW" /></th>
            <th><Term name="AssA" /></th>
            <th><Term name="DetA" /></th>
            <th><Term name="p95 ms" /></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className={r.group === "ours" ? "bar-ours" : "bar-baseline"} data-testid={`gpu-row-${r.name}`}>
              <td>
                {r.label} {r.group === "control" && <span className="tag">control</span>}
                {r.group === "ours" && <span className="tag ours">ours</span>}
              </td>
              <td className={r.HOTA === best("HOTA") ? "best" : ""}>{num(r.HOTA)}</td>
              <td className={r.IDF1 === best("IDF1") ? "best" : ""}>{num(r.IDF1)}</td>
              <td className={r.MOTA === best("MOTA") ? "best" : ""}>{num(r.MOTA)}</td>
              <td className={r.IDSW === minIdsw ? "best" : ""}>{r.IDSW == null ? DASH : int(r.IDSW)}</td>
              <td className={r.AssA === best("AssA") ? "best" : ""}>{num(r.AssA)}</td>
              <td>{num(r.DetA)}</td>
              <td>{num(r.p95_ms, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function GpuRunSection({ g }: { g: Gpu }) {
  const [sel, setSel] = useState(g.datasets[0]?.id ?? "");
  const ds = g.datasets.find((d) => d.id === sel) ?? g.datasets[0];
  const rep = g.reproduction;
  const gng = g.go_no_go_v2;
  return (
    <>
      <section className="section" data-testid="gpu-verdicts">
        <h2>GPU run: why it says NOT REPRODUCED and NO-GO</h2>
        <p className="sub">
          {g.profile} run on {g.gpu}, read from {g.source}. Two separate checks; neither compares ours with ByteTrack. Plain-English write-up: <code>{g.analysis}</code>.
        </p>
        <div className="charts">
          {rep && (
            <div className="panel check" data-testid="gpu-reproduction">
              <div className="check-head">
                <h3>ByteTrack reproduction</h3>
                <Pill verdict={rep.verdict} />
              </div>
              <p>Our ByteTrack on MOT17 val-half vs the paper. ID switches match; the gap is in the boxes (more false positives and misses), so every GPU score is provisional.</p>
              <table>
                <thead>
                  <tr><th>Metric</th><th>Paper</th><th>Ours</th><th>Δ</th><th>Tolerance</th></tr>
                </thead>
                <tbody>
                  {rep.rows.map((r) => (
                    <tr key={r.metric} data-testid={`rep-${r.metric}`}>
                      <td>{r.metric}</td>
                      <td>{r.paper ?? DASH}</td>
                      <td>{r.ours ?? DASH}</td>
                      <td>{r.delta == null ? DASH : `${r.delta > 0 ? "+" : ""}${r.delta}`}</td>
                      <td>{r.tolerance == null ? <span className="muted">reported only</span> : <>±{r.tolerance} {r.within ? "" : <span className="tag">outside</span>}</>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="caption">Source: {rep.source} ({rep.reference}).</p>
            </div>
          )}
          {gng && (
            <div className="panel check" data-testid="gpu-go-no-go">
              <div className="check-head">
                <h3>Go / no-go v2 (delayed decision)</h3>
                <Pill verdict={gng.verdict} />
              </div>
              <p>{gng.what}</p>
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
          )}
        </div>
      </section>

      <section className="section" data-testid="gpu-ablation">
        <h2>GPU run: tracking accuracy per dataset</h2>
        <p className="sub">
          YOLOX-X detections + OSNet-AIN re-ID, TrackEval. Best per column in blue. Provisional (see the reproduction check).

        </p>
        <div className="seg" role="tablist" aria-label="Dataset">
          {g.datasets.map((d) => (
            <button key={d.id} role="tab" aria-selected={d.id === ds?.id} className={`chip-btn${d.id === ds?.id ? " on" : ""}`} onClick={() => setSel(d.id)} data-testid={`gpu-ds-${d.id}`}>
              {label(d.id)}
            </button>
          ))}
        </div>
        {ds && (
          <>
            <GpuAblation rows={ds.rows} />
            <p className="caption">
              Source: {ds.source}. Detector {ds.detector ?? DASH}{ds.trained_on ? ` (${ds.trained_on})` : ""}{ds.input_size ? `, input ${ds.input_size}` : ""}. p95 = tracker <code>update()</code> on the GPU host.
            </p>
            {Object.keys(ds.plots).length > 0 && (
              <div className="charts">
                {Object.entries(ds.plots).map(([k, url]) => (
                  <figure className="panel chart" key={`${ds.id}-${k}`} style={{ margin: 0 }}>
                    <figcaption className="label" style={{ marginBottom: 8 }}>{label(ds.id)}: {k.replace(/_/g, " ")}</figcaption>
                    <img src={url} alt={`${ds.id} ${k}`} loading="lazy" onError={(e) => { const alt = offlinePath(url); if (alt && e.currentTarget.getAttribute("src") !== alt) e.currentTarget.src = alt; }} />
                  </figure>
                ))}
              </div>
            )}
          </>
        )}
      </section>

      {g.gate_events && (
        <section className="section" data-testid="gpu-gate-events">
          <h2>GPU run: gate events</h2>
          <p className="sub">MOT17-02/04/09 val-half on GPU detections: 25 gt crossings and 9 tailgates, too few to rank trackers.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Tracker</th><th>Crossings gt / trk</th><th>Recall</th><th>Precision</th><th>Attribution error</th><th>Count error</th><th>Tailgate P / R</th></tr>
              </thead>
              <tbody>
                {g.gate_events.rows.map((r) => (
                  <tr key={r.label} className={/vis/.test(r.label) ? "bar-ours" : "bar-baseline"}>
                    <td>{r.label.replace(" (check)", "")} {r.check && <span className="tag">sanity check</span>}</td>
                    <td>{int(r.gt_crossings)} / {int(r.trk_crossings)}</td>
                    <td>{pct(r.recall)}</td>
                    <td>{pct(r.precision)}</td>
                    <td>{pct(r.attribution_error)}</td>
                    <td>{pct(r.count_error)}</td>
                    <td>{pct(r.tailgate_precision, 0)} / {pct(r.tailgate_recall, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="caption">Source: {g.gate_events.source}.</p>
        </section>
      )}
    </>
  );
}
