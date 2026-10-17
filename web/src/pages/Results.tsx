import { offlinePath } from "../lib/offline";
import { FoundChecks } from "../components/Checks";
import { Term } from "../components/Term";
import { loadSummary } from "../lib/summary";
import { DatasetTable } from "../components/DatasetTable";
import { Mot20Status } from "../components/Mot20Status";
import { api, type AblationRow, type Benchmarks } from "../api";
import StressChart from "../components/StressChart";
import StressMap from "../stress/StressMap";
import { DASH, int, num, pct } from "../lib/format";
import { useAsync } from "../lib/useAsync";

function rowClass(r: AblationRow): string {
  const bar = r.group === "ours" ? "bar-ours" : "bar-baseline";
  return r.headline ? `${bar} headline` : bar;
}

const PENDING = (
  <td className="pending">
    {DASH} <span className="tag">pending DGX run</span>
  </td>
);

export function ResultsView({ b }: { b: Benchmarks }) {
  const abl = b.ablation;
  const rows = abl?.rows ?? [];
  const best = (k: "HOTA" | "IDF1" | "MOTA" | "AssA") => Math.max(...rows.map((r) => r[k]));
  const minIdsw = Math.min(...rows.map((r) => r.IDSW));
  return (
    <div>
      <h1>Results</h1>
      <StressMap />
      <p className="muted" style={{ marginTop: 24, maxWidth: 820 }}>
        Every number below is read from the repo's own output files (named under each table). Tables first, charts after.
        Nothing here uses the demo clips: they have no ground truth.
      </p>

      <section className="section" data-testid="gpu-run-checks">
        <h2>Latest GPU run: checks</h2>
        <p className="sub">From docs/results/summary.json. Runs whose detector saw the evaluation half are marked “leaked, excluded” and appear in no table.</p>
        <FoundChecks summary={loadSummary()} />
      </section>

      <section className="section" data-testid="overall-comparison">
        <h2>All datasets at a glance</h2>
        <p className="sub">ByteTrack vs BoT-SORT vs ours on every dataset, read only from summary.json. Provisional: see the reproduction check above.</p>
        <DatasetTable />
      </section>
      <section className="section" data-testid="mot20-section">
        <h2>MOT20 (dense crowds): what exists</h2>
        <p className="sub">Provisional. Only what the repo's files contain; every value below is parsed from the named file at build time.</p>
        <Mot20Status />
      </section>
      <section className="section">
        <h2>Tracking accuracy</h2>
        <p className="sub">{abl ? abl.setting : "docs/results/ablation/ablation.json not found."}</p>
        {abl && (
          <>
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
                    <tr key={r.name} className={rowClass(r)} data-testid={`row-${r.name}`}>
                      <td>
                        {r.label} {r.group === "control" && <span className="tag">control</span>}
                        {r.name.includes("delayed") && <span className="tag">no-go</span>}
                      </td>
                      <td className={r.HOTA === best("HOTA") ? "best" : ""}>{num(r.HOTA)}</td>
                      <td className={r.IDF1 === best("IDF1") ? "best" : ""}>{num(r.IDF1)}</td>
                      <td className={r.MOTA === best("MOTA") ? "best" : ""}>{num(r.MOTA)}</td>
                      <td className={r.IDSW === minIdsw ? "best" : ""}>{int(r.IDSW)}</td>
                      <td className={r.AssA === best("AssA") ? "best" : ""}>{num(r.AssA)}</td>
                      <td>{num(r.DetA)}</td>
                      <td>{num(r.p95_ms, 2)}</td>
                    </tr>
                  ))}
                  <tr className="bar-ours">
                    <td>Ours on YOLOX-X detections + re-ID</td>
                    {PENDING}
                    {PENDING}
                    <td className="pending">{DASH}</td>
                    <td className="pending">{DASH}</td>
                    <td className="pending">{DASH}</td>
                    <td className="pending">{DASH}</td>
                    <td className="pending">{DASH}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="caption">
              Source: {abl.source} (reproduce: <code>{abl.reproduce}</code>). Blue bar: our method; grey: baselines and controls.
              Best per column in blue. p95 = single-threaded tracker <code>update()</code> time on the CPU box. The visibility
              estimate matches a 90-frame buffer control, so it has not been shown to beat keeping lost tracks longer.
            </p>
          </>
        )}
      </section>

      {b.gate_events && (
        <section className="section">
          <h2>Gate events</h2>
          <p className="sub">{b.gate_events.note}</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Tracker</th>
                  <th>Crossings gt / trk</th>
                  <th>Recall</th>
                  <th>Precision</th>
                  <th>Attribution error</th>
                  <th>Count error</th>
                  <th>Tailgate P / R</th>
                </tr>
              </thead>
              <tbody>
                {b.gate_events.rows.map((r) => (
                  <tr key={r.label} className={/visibility|delayed/.test(r.label) ? "bar-ours" : "bar-baseline"}>
                    <td>
                      {r.label.replace(" (check)", "")} {r.check && <span className="tag">sanity check</span>}
                    </td>
                    <td>
                      {int(r.gt_crossings as number)} / {int(r.trk_crossings as number)}
                    </td>
                    <td>{pct(r.recall as number)}</td>
                    <td>{pct(r.precision as number)}</td>
                    <td>{pct(r.attribution_error as number)}</td>
                    <td>{pct(r.count_error as number)}</td>
                    <td>
                      {pct(r.tailgate_precision as number, 0)} / {pct(r.tailgate_recall as number, 0)}
                    </td>
                  </tr>
                ))}
                <tr className="bar-ours">
                  <td>Ours on GPU detections</td>
                  {PENDING}
                  <td className="pending">{DASH}</td>
                  <td className="pending">{DASH}</td>
                  <td className="pending">{DASH}</td>
                  <td className="pending">{DASH}</td>
                  <td className="pending">{DASH}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="caption">Source: {b.gate_events.source} (reproduce: <code>make gates</code>).</p>
        </section>
      )}

      {b.visibility && (
        <section className="section">
          <h2>Visibility estimate</h2>
          <p className="sub">How well the box-geometry visibility estimate predicts ground-truth visibility (MOT17 val-half, detector boxes).</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Estimate</th>
                  <th>n</th>
                  <th><Term name="MAE" /></th>
                  <th><Term name="Spearman ρ" /></th>
                  <th>AUROC v&lt;0.5</th>
                  <th>AP v&lt;0.5</th>
                </tr>
              </thead>
              <tbody>
                {b.visibility.rows.map((r) => (
                  <tr key={r.key} className={r.ours ? "bar-ours" : "bar-baseline"}>
                    <td>
                      {r.label} {r.key.startsWith("oracle") && <span className="tag">oracle</span>}
                    </td>
                    <td>{int(r.n)}</td>
                    <td>{num(r.mae, 3)}</td>
                    <td>{num(r.spearman, 3)}</td>
                    <td>{num(r["auroc_vis_lt_0.5"], 3)}</td>
                    <td>{num(r["ap_vis_lt_0.5"], 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="caption">Source: {b.visibility.source}.</p>
        </section>
      )}

      {b.go_no_go && (
        <section className="section">
          <h2>
            Delayed decision <span className="tag">sanity check</span>
          </h2>
          <p className="sub">
            Pre-registered go/no-go for the K-frame look-ahead: verdict <strong>{b.go_no_go.verdict}</strong>. ID switches rise at every K
            &gt; 0, so it stays out of every headline configuration. Not a results table.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Condition</th>
                  <th>K (frames)</th>
                  <th><Term name="HOTA" /></th>
                  <th><Term name="IDF1" /></th>
                  <th><Term name="IDSW" /></th>
                  <th><Term name="p95 ms" /></th>
                </tr>
              </thead>
              <tbody>
                {b.go_no_go.rows.map((r) => (
                  <tr key={`${r.condition}-${r.K}`} className="bar-baseline">
                    <td>{r.condition}</td>
                    <td>{r.K}</td>
                    <td>{num(r.HOTA)}</td>
                    <td>{num(r.IDF1)}</td>
                    <td>{int(r.IDSW_trackeval)}</td>
                    <td>{num(r.p95_ms, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="caption">
            Source: {b.go_no_go.source}; write-up {b.go_no_go.summary}.
          </p>
        </section>
      )}

      <section className="section">
        <h2>Pending the DGX run</h2>
        <p className="sub">These need YOLOX-X detections and re-ID embeddings from the GPU session; they come back through make ingest and make phase3.</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Result</th>
                <th>Value</th>
                <th>Will be written to</th>
              </tr>
            </thead>
            <tbody>
              {b.phase3.items.map((i) => (
                <tr key={i.id}>
                  <td style={{ whiteSpace: "normal" }}>{i.what}</td>
                  {b.phase3.status === "pending" ? PENDING : <td>see files</td>}
                  <td>
                    <code>{i.source}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {abl && (
        <section className="section">
          <h2>Stress curves</h2>
          <p className="sub">The same runs, broken down by how visible each ground-truth person was. Below 50 % visibility the public detector finds almost nobody, so every tracker's ID recall collapses there.</p>
          <div className="charts">
            <StressChart rows={rows} names={["botsort", "ours_vis_botsort"]} metric="id_recall" title="ID recall by visibility (BoT-SORT vs ours)" />
            <StressChart rows={rows} names={["botsort", "ours_vis_botsort"]} metric="idsw_per_1k_gt" title="ID switches per 1k gt boxes by visibility" />
            {Object.entries(b.plots).map(([k, url]) => (
              <figure className="panel chart" key={k} style={{ margin: 0 }}>
                <figcaption className="label" style={{ marginBottom: 8 }}>
                  {k.replace(/_/g, " ")} {k.startsWith("go_no_go") && <span className="tag">sanity check</span>}
                </figcaption>
                <img src={url} alt={k} loading="lazy" onError={(e) => { const alt = offlinePath(url); if (alt && e.currentTarget.getAttribute("src") !== alt) e.currentTarget.src = alt; }} />
              </figure>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export default function Results() {
  const { data, error, loading } = useAsync(api.benchmarks);
  if (loading) return <p className="muted">Loading results…</p>;
  if (error || !data) return <div className="error-box">Could not load results: {error?.message}</div>;
  return <ResultsView b={data} />;
}
