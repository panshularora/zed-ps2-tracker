// Laptop/DGX GPU run (docs/results/phase3, served by GET /api/benchmarks "gpu"): the plain
// ablation table and the GPU gate events (the showcase tables live in GpuShowcase.tsx). Every value comes from the API response; nothing is typed here.
import type { GpuRun as Gpu, GpuRow } from "../api";
import { DASH, int, num, pct } from "../lib/format";
import { Term } from "./Term";

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

/** GPU gate events; the rest of the GPU run is in GpuShowcase (top of the Results tab). */
export function GpuRunSection({ g }: { g: Gpu }) {
  return (
    <>
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
