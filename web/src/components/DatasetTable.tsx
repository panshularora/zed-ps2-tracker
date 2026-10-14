import { datasetRows, loadSummary, loadSynthetic, TRACKERS, type Cell, type SummaryFile } from "../lib/summary";

function CellView({ c }: { c: Cell }) {
  return (
    <>
      {c.values?.map((v) => (
        <div key={v.label} className="tabular"><small className="muted">{v.label}</small> {v.value}</div>
      ))}
      <span className="tag" data-tag={c.tag}>{c.tag}</span>
    </>
  );
}

/** Overall comparison across datasets, every value from docs/results/summary.json. */
export function DatasetTable({ summary = loadSummary(), synthetic = loadSynthetic() }: { summary?: SummaryFile | null; synthetic?: Parameters<typeof datasetRows>[1] }) {
  const rows = datasetRows(summary, synthetic);
  const ref = summary?.reproduction?.reference;
  return (
    <div className="panel" data-testid="dataset-table">
      <table className="cmp-table">
        <thead>
          <tr><th>Dataset</th><th>Status</th>{TRACKERS.map((t) => <th key={t.key}>{t.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} data-testid={`ds-${r.id}`}>
              <td>{r.dataset}</td>
              <td className="muted">{r.status}</td>
              {TRACKERS.map((t) => <td key={t.key}><CellView c={r.cells[t.key]} /></td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="credit" style={{ marginTop: 8 }}>
        Source: docs/results/summary.json only (laptop GPU run, provisional). MOT17 ByteTrack is our reproduction
        {ref?.MOTA != null ? `; ByteTrack's published numbers are MOTA ${ref.MOTA} / IDF1 ${ref.IDF1}, and the reproduction verdict is ${summary?.reproduction?.verdict ?? "pending"}` : ""}, so no tracker comparison from this run should be quoted. The synthetic row is one seeded run of the live gate scene (docs/results/synthetic_gate), exact ground truth but a toy world: static camera, so BoT-SORT runs without camera-motion correction there.
        Cells without a number say why: the run happened but summary.json carries no per-tracker metric, it has not run, or there is no ground truth.
      </p>
    </div>
  );
}
