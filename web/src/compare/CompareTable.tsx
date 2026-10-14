// Per-scenario comparison table: only numbers present in the replay data; no ground truth = said so.
import type { RunKey } from "../lib/scenarios";
import { countsAt, type ReplayData } from "../guided/model";
import { runStats } from "./model";
import type { Truth } from "./ComparePlayer";

export function CompareTable({ data, truth, suspected }: { data: ReplayData; truth?: Truth | null; suspected?: Record<RunKey, number> | null }) {
  const measured = data.kind === "measured";
  const tc = truth ? countsAt(truth.events, data.frames) : null;
  const trueTotal = tc ? tc.in + tc.out : null;
  return (
    <div className="panel" data-testid="compare-table" style={{ marginTop: 12 }}>
      <div className="label">
        Comparison on this clip · {measured ? <span className="tag">measured · human labels</span> : <span className="tag">illustrative · no ground truth</span>}
      </div>
      <table className="cmp-table">
        <thead>
          <tr>
            <th>Tracker</th>
            <th>{measured ? "ID switches (vs labels)" : "Number changes (tracker-reported)"}</th>
            <th>Gate count</th>
            <th>True count</th>
            <th>IDF1 / HOTA</th>
          </tr>
        </thead>
        <tbody>
          {(["baseline", "ours"] as const).map((r) => {
            const s = runStats(data, r);
            const counted = s.crossIn + s.crossOut;
            return (
              <tr key={r} data-testid={`row-${r}`}>
                <td>{data.runs[r].label}{r === "baseline" ? " (baseline)" : ""}</td>
                <td>{suspected ? suspected[r] : s.idChanges}</td>
                <td>{counted} <small className="muted">({s.crossIn} in · {s.crossOut} out)</small></td>
                <td>{trueTotal != null ? <>{trueTotal}{counted !== trueTotal && <small className="over"> ({counted > trueTotal ? "+" : ""}{counted - trueTotal})</small>}</> : <span className="muted">not labelled</span>}</td>
                <td className="muted">{measured ? "not computed per clip (see Results)" : "needs ground truth"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="credit" style={{ marginTop: 8 }}>
        {measured
          ? `ID switches and the true gate count come from MOTChallenge human labels on the ${data.frames}-frame replay. IDF1/HOTA are only computed over all val-half sequences (Results page), provisional.`
          : "No ground truth: a number change here means one tracker gave a person a new number, not that it was wrong. Gate counts use a geometry-chosen line, so no true count exists."}
      </p>
    </div>
  );
}
