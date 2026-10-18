import { mot20Status, type Mot20Status as S } from "../lib/mot20";

/** Exactly what exists for MOT20 in the repo, with provisional labels. */
export function Mot20Status({ s = mot20Status() }: { s?: S }) {
  const k0 = s.corruptedGtK0, o = s.oracle;
  return (
    <div className="panel" style={{ padding: 12 }} data-testid="mot20-status">
      <table className="cmp-table">
        <thead><tr><th>What</th><th>HOTA</th><th>IDF1</th><th>MOTA</th><th>IDSW</th><th>Status</th></tr></thead>
        <tbody>
          <tr data-testid="mot20-k0">
            <td>Our tracker (delayed assignment, K = 0) on MOT20 val half, detections = <b>corrupted ground truth</b> (seed 0), CPU</td>
            {k0 ? <><td className="tabular">{k0.HOTA}</td><td className="tabular">{k0.IDF1}</td><td className="tabular">{k0.MOTA}</td><td className="tabular">{k0.IDSW}</td></> : <td colSpan={4} className="muted">not found</td>}
            <td><span className="tag">provisional</span> scored against ground truth (TrackEval), but no real detector; K &gt; 0 did not finish; no ByteTrack row on the same input</td>
          </tr>
          <tr data-testid="mot20-oracle">
            <td>Oracle ceiling: the ground-truth boxes themselves submitted as a tracker (vis &gt; 0)</td>
            {o ? <><td className="tabular">{o.HOTA}</td><td className="tabular">{o.IDF1}</td><td className="tabular">{o.MOTA}</td><td className="muted">–</td></> : <td colSpan={4} className="muted">not found</td>}
            <td><span className="tag">not a tracker result</span> upper bound only</td>
          </tr>
          {s.runs.map((r) => (
            <tr key={r.id} data-testid="mot20-run">
              <td className="mono">{r.id}</td>
              <td colSpan={4} className="muted">no metrics in summary.json</td>
              <td><span className="tag">{r.leaked ? "leaked, excluded" : "ran, no metric in summary.json"}</span></td>
            </tr>
          ))}
          <tr data-testid="mot20-export">
            <td>MOT20-02 viewer export (ByteTrack vs ours boxes for Scenarios)</td>
            <td colSpan={4} className="muted">{s.viewerExport ? "exported" : "not exported (pending GPU session)"}</td>
            <td><span className="tag">{s.viewerExport ? "exported" : "pending"}</span></td>
          </tr>
        </tbody>
      </table>
      <p className="credit" style={{ marginTop: 8 }}>
        Sources: docs/results/go_no_go.md (MOT20 CPU corrupted-gt row), docs/evaluation.md (oracle), docs/results/summary.json (laptop GPU ablation). MOT20 numbers in the table above this panel are from the leak-free SparseTrack ablation cache; the MOT17→MOT20 transfer cache is a second clean run. The leftover SAW_EVAL_HALF folder on disk is excluded.
      </p>
    </div>
  );
}
