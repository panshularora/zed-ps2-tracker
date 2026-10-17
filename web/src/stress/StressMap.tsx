// Stress Map (top of Results): ours vs ByteTrack over 8 hidden times × 6 crowd sizes, 20 seeds a cell.
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { AMBER_STEPS, BLUE, NEUTRAL, borders, cellAt, cellFill, cellSentence, configNote, fmtGap, headline, isHatched, type MapCell, type StressMapData } from "./model";
import { PANE_ORDER, hideWindow, replaySentence, verdict, type SceneData } from "./scene";
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
  const pid = fd?.pid ?? null;
  const sentence = useMemo(() => {
    if (!data || pid == null) return null;
    const v = Object.fromEntries(PANE_ORDER.map((k) => [k, verdict(data, k, pid)]));
    return replaySentence(cell.people, pid, hideWindow(data, pid) ? cell.hide_s : null, v, fd?.changed);
  }, [data, pid, cell, fd]);
  return (
    <div className="stress-replay" data-testid="stress-replay">
      <p className="replay-sentence" data-testid="replay-sentence">
        {sentence ?? (fd ? `${cell.people} people, hidden ${cell.hide_s} s.` : `${cell.people} people, hidden ${cell.hide_s} s. The two trackers never disagree on this seed.`)}
      </p>
      <details className="stress-how" data-testid="replay-settings">
        <summary>What am I looking at?</summary>
        <p>
          A re-run of one of this cell's {cell.seeds.length} seeds: seed {cell.replay_seed}, the one with the cell's median gap ({fmtGap(cell.replay_diff)} ID changes).
          {" "}{cell.people} people, each hidden {cell.hide_s} s, detector noise 0.15. Ours runs on {cfg === "mem6" ? "the longer-memory (opt-in)" : "default"} settings.
          {" "}{fd ? `It opens paused at frame ${fd.frame}, the first frame where the two trackers disagree (person ${fd.pid}; ${fd.changed === "ours" ? "ours" : "ByteTrack"} changed the number).` : "The two trackers never disagree on this seed."}
          {" "}Both panes are zoomed 2× on that person. Figures drawn grey with a dashed outline are hidden from the camera.
        </p>
      </details>
      {err && <p className="muted" data-testid="replay-error">Replay needs the local service (it re-runs both trackers): {err}</p>}
      {!data && !err && <p className="muted">Re-running both trackers on this seed…</p>}
      {data && <ScenePlayer data={data} startFrame={fd?.frame ?? 1} focus={pid} view="follow" reveal="always" />}
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
  useEffect(() => {
    if (d && typeof window !== "undefined" && window.location.hash === "#stress-map") document.getElementById("stress-map")?.scrollIntoView();
  }, [d]);
  const b = useMemo(() => (d ? borders(d) : null), [d]);
  if (err) return <section className="section" id="stress-map"><h2>Stress Map</h2><p className="muted">Could not load docs/results/stress/stress_map.json: {err}</p></section>;
  if (!d || !b) return <section className="section" id="stress-map"><h2>Stress Map</h2><p className="muted">Loading…</p></section>;
  const rows = [...d.axes.people].reverse(); // most people at the top
  const hs = d.axes.hide_s;
  const gridW = LABEL_W + hs.length * (CELL + GAP) - GAP;
  const gridH = rows.length * (CELL + GAP) - GAP;
  const cell = sel ? cellAt(d, sel.people, sel.hide_s) : undefined;
  return (
    <section className="section stress-map" id="stress-map" data-testid="stress-map">
      <h2>Stress Map · one person, one ID</h2>
      <p className="stress-headline" data-testid="stress-headline">{headline(d)}</p>
      <div className="stress-toggle-row">
        <div className="seg" role="group" aria-label="Ours config">
          <button className={`btn${cfg === "default" ? " primary" : ""}`} onClick={() => setCfg("default")} data-testid="cfg-default">Default</button>
          <button className={`btn${cfg === "mem6" ? " primary" : ""}`} onClick={() => setCfg("mem6")} data-testid="cfg-mem6">Longer memory (opt-in)</button>
        </div>
        <span className="stress-note" data-testid="cfg-note">{configNote(cfg, d)}</span>
      </div>
      <p className="stress-oneline" data-testid="cell-sentence">{cellSentence(d)}</p>
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
                      style={{ backgroundColor: f.fill, backgroundImage: isHatched(c) ? HATCH : "none", color: f.ink }}
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
        <div className="stress-legend small" data-testid="stress-legend">
          <div><span className="sw" style={{ background: BLUE }} /> ours fewer ID changes</div>
          <div><span className="sw" style={{ background: AMBER_STEPS[1] }} /> ours more</div>
          <div><span className="sw" style={{ background: NEUTRAL }} /> tie</div>
          <div><span className="sw" style={{ backgroundColor: NEUTRAL, backgroundImage: HATCH }} /> not clear</div>
          <div className="sm-line"><svg width="16" height="16"><line x1="8" y1="0" x2="8" y2="16" stroke="#E6E8EB" strokeWidth="2" /></svg> <span data-testid="label-start">{b.startLabel}</span></div>
          {b.endLabel && <div className="sm-line"><svg width="16" height="16"><line x1="8" y1="0" x2="8" y2="16" stroke="#E6E8EB" strokeWidth="2" strokeDasharray="5 3" /></svg> <span data-testid="label-end">{b.endLabel}</span></div>}
        </div>
      </div>
      <details className="stress-how" data-testid="how-we-tested">
        <summary>How we tested</summary>
        <p>
          Each cell runs both trackers on the same {d.cells[0]?.seeds.length ?? 0} simulator seeds ({d.seeds[0]}–{d.seeds[d.seeds.length - 1]}), CPU, git {d.git.slice(0, 7)}.
          {" "}A cell is coloured as clear only when the 95% bootstrap CI of the per-seed gap excludes 0 and the sign test gives p &lt; 0.05; otherwise it is hatched.
          {" "}Blue steps at |gap| above 0, 1, 3, 6 and 12 ID changes; amber in three flat steps (up to 1, up to 3, above 3). Hover a cell for its means, CI and verdict.
        </p>
      </details>
      {cell && <Replay cell={cell} cfg={cfg} />}
    </section>
  );
}
