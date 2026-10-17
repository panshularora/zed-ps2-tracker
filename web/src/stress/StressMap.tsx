// Stress Map (top of Results): ours vs ByteTrack over 8 hidden times × 6 crowd sizes, 20 seeds a cell.
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { CELL_SENTENCE, borders, cellAt, cellFill, fmtGap, headline, isHatched, type MapCell, type StressMapData } from "./model";
import type { SceneData } from "./scene";
import { ScenePlayer } from "./ScenePlayer";

const CELL = 56, GAP = 2, LABEL_W = 40;
const HATCH = `url("data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><path d="M-2 10 L10 -2" stroke="#E6E8EB" stroke-opacity="0.35" stroke-width="1"/></svg>')}")`;
type Cfg = "default" | "mem6";

function Borderline({ d, cols, which }: { d: StressMapData; cols: (number | null)[]; which: "start" | "end" }) {
  const n = d.axes.hide_s.length;
  const rows = [...d.axes.people].reverse();
  const order = rows.map((p) => cols[d.axes.people.indexOf(p)]);
  const segs: string[] = [];
  let cur: string[] = [];
  order.forEach((c, r) => {
    if (c == null || c >= n || c <= 0) { if (cur.length) segs.push(cur.join(" ")); cur = []; return; }
    const x = LABEL_W + c * (CELL + GAP) - GAP / 2;
    cur.push(`${x},${r * (CELL + GAP)}`, `${x},${r * (CELL + GAP) + CELL}`);
  });
  if (cur.length) segs.push(cur.join(" "));
  return (
    <>
      {segs.map((s, i) => (
        <polyline key={`${which}${i}`} points={s} fill="none" stroke="#E6E8EB" strokeWidth={2} strokeDasharray={which === "end" ? "5 3" : undefined} data-testid={`border-${which}`} />
      ))}
    </>
  );
}

function Replay({ cell, cfg }: { cell: MapCell; cfg: Cfg }) {
  const [data, setData] = useState<SceneData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setData(null); setErr(null);
    api.stressReplay({ people: cell.people, hide_s: cell.hide_s, seed: cell.replay_seed, config: cfg })
      .then((d) => live && setData(d))
      .catch((e: Error) => live && setErr(e.message));
    return () => { live = false; };
  }, [cell, cfg]);
  const fd = cell.first_disagreement;
  return (
    <div className="stress-replay" data-testid="stress-replay">
      <p className="small muted mono" data-testid="replay-settings">
        seed {cell.replay_seed} (median gap of the cell's {cell.seeds.length} seeds: {fmtGap(cell.replay_diff)}) · {cell.people} people · hidden {cell.hide_s} s · noise 0.15 ·
        ours = {cfg === "mem6" ? "longer memory (opt-in)" : "default"} ·{" "}
        {fd ? `opens paused at frame ${fd.frame}, the first frame where the two trackers disagree (person ${fd.pid}, ${fd.changed === "ours" ? "ours" : "ByteTrack"} changed the number)` : "the two trackers never disagree on this seed"}
      </p>
      {err && <p className="muted" data-testid="replay-error">Replay needs the local service (it re-runs both trackers): {err}</p>}
      {!data && !err && <p className="muted">Re-running both trackers on this seed…</p>}
      {data && <ScenePlayer data={data} startFrame={fd?.frame ?? 1} focus={fd?.pid ?? null} />}
    </div>
  );
}

export default function StressMap() {
  const [cfg, setCfg] = useState<Cfg>("default");
  const [maps, setMaps] = useState<Partial<Record<Cfg, StressMapData>>>({});
  const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<{ people: number; hide_s: number } | null>(null);
  useEffect(() => {
    Promise.all([api.stressMap("default"), api.stressMap("mem6")])
      .then(([a, b]) => setMaps({ default: a, mem6: b }))
      .catch((e: Error) => setErr(e.message));
  }, []);
  const d = maps[cfg];
  const b = useMemo(() => (d ? borders(d) : null), [d]);
  if (err) return <section className="section"><h2>Stress Map</h2><p className="muted">Could not load docs/results/stress/stress_map.json: {err}</p></section>;
  if (!d || !b) return <section className="section"><h2>Stress Map</h2><p className="muted">Loading…</p></section>;
  const rows = [...d.axes.people].reverse(); // most people at the top
  const hs = d.axes.hide_s;
  const gridW = LABEL_W + hs.length * (CELL + GAP) - GAP;
  const gridH = rows.length * (CELL + GAP) - GAP;
  const cell = sel ? cellAt(d, sel.people, sel.hide_s) : undefined;
  return (
    <section className="section stress-map" data-testid="stress-map">
      <h2>Stress Map · one person, one ID</h2>
      <p className="stress-headline" data-testid="stress-headline">{headline(d)}</p>
      <div className="seg" role="group" aria-label="Ours config">
        <button className={`btn${cfg === "default" ? " primary" : ""}`} onClick={() => setCfg("default")} data-testid="cfg-default">Default</button>
        <button className={`btn${cfg === "mem6" ? " primary" : ""}`} onClick={() => setCfg("mem6")} data-testid="cfg-mem6">Longer memory (opt-in)</button>
      </div>
      <p className="small muted">Tuned on separate seeds · tested on MOT17 only (HOTA 52.5 either way)</p>
      <p className="small" style={{ maxWidth: 760 }}>{CELL_SENTENCE}</p>
      <div className="stress-frame">
        <div className="stress-ylabel">People in view</div>
        <div>
          <div className="stress-grid-wrap" style={{ width: gridW, height: gridH }}>
            <div className="stress-grid" style={{ gridTemplateColumns: `${LABEL_W - GAP}px repeat(${hs.length}, ${CELL}px)`, gap: GAP }}>
              {rows.map((n) => [
                <div key={`l${n}`} className="stress-rowlabel">{n}</div>,
                ...hs.map((h) => {
                  const c = cellAt(d, n, h)!;
                  const f = cellFill(c);
                  const on = sel?.people === n && sel?.hide_s === h;
                  return (
                    <button
                      key={`${n}-${h}`}
                      className={`stress-cell${on ? " on" : ""}`}
                      data-testid={`cell-${n}-${h}`}
                      data-verdict={c.verdict}
                      data-step={f.step}
                      data-hatched={isHatched(c) ? "1" : "0"}
                      style={{ backgroundColor: f.fill, backgroundImage: isHatched(c) ? HATCH : "none", color: f.step >= 4 ? "#0E1013" : "#E6E8EB" }}
                      title={`${n} people, hidden ${h} s: ours ${c.ours_mean.toFixed(1)} vs ByteTrack ${c.bytetrack_mean.toFixed(1)} ID changes per clip; 95% CI of the gap ${fmtGap(c.ci95[0])} to ${fmtGap(c.ci95[1])}; ${c.verdict.replace("ours_", "ours ")}`}
                      onClick={() => setSel({ people: n, hide_s: h })}
                    >
                      {fmtGap(c.diff_mean)}
                    </button>
                  );
                }),
              ])}
            </div>
            <svg className="stress-lines" width={gridW} height={gridH} aria-hidden>
              <Borderline d={d} cols={b.winStart} which="start" />
              <Borderline d={d} cols={b.winEnd} which="end" />
            </svg>
          </div>
          <div className="stress-xticks" style={{ paddingLeft: LABEL_W, gridTemplateColumns: `repeat(${hs.length}, ${CELL}px)`, gap: GAP }}>
            {hs.map((h) => <span key={h}>{h}</span>)}
          </div>
          <div className="stress-xlabel" style={{ paddingLeft: LABEL_W }}>Hidden for (seconds)</div>
        </div>
        <div className="stress-legend small">
          <div><span className="sw" style={{ background: "#56B4E9" }} /> ours fewer ID changes</div>
          <div><span className="sw" style={{ background: "#F5A524" }} /> ours more</div>
          <div><span className="sw" style={{ background: "#262B33" }} /> tie</div>
          <div><span className="sw" style={{ backgroundColor: "#262B33", backgroundImage: HATCH }} /> not statistically clear</div>
          <div className="sm-line"><svg width="16" height="16"><line x1="8" y1="0" x2="8" y2="16" stroke="#E6E8EB" strokeWidth="2" /></svg> <span data-testid="label-start">{b.startLabel}</span></div>
          {b.endLabel && <div className="sm-line"><svg width="16" height="16"><line x1="8" y1="0" x2="8" y2="16" stroke="#E6E8EB" strokeWidth="2" strokeDasharray="5 3" /></svg> <span data-testid="label-end">{b.endLabel}</span></div>}
          <div className="muted">Clear = 95% bootstrap CI of the per-seed gap excludes 0 and sign test p &lt; 0.05. Seeds {d.seeds[0]}–{d.seeds[d.seeds.length - 1]}, CPU, git {d.git.slice(0, 7)}.</div>
        </div>
      </div>
      {cell && <Replay cell={cell} cfg={cfg} />}
    </section>
  );
}
