// Compare on real clips: ByteTrack (left) vs ours (right), synced on one video clock, from the real
// exported tracks. Pexels scenario files carry both trackers run on the identical detections
// (scripts/demo/precompute_scenarios.py). Kept/swap per reappearance comes from the check in
// web/public/verify (heuristic on Pexels, ground truth on MOT17).
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { fromScenario, videoSources, type ReplayData } from "../guided/model";
import { benchmarkReplay, loadBenchmarkPairs, type BenchPair } from "../guided/viewer";
import { paintFrame } from "../identity/IdentityPlayer";
import { boxesAt, people as peopleOf, tracksOf, type Person, type XYWH } from "../identity/story";
import { counts, fetchVerify, methodTooltip, verdictOf, type VerifyDoc } from "../identity/verify";
import { fetchIndex, fetchScenario, videoUrl, type RunKey, type ScenarioMeta } from "../lib/scenarios";
import { orderScenarios } from "../pages/Scenarios";

export const RUNS: RunKey[] = ["baseline", "ours"];

export interface PersonRow { key: number; ids: number[]; seconds: number; status: "kept" | "changed" | "uncertain"; detail: string; f0: number; kept: number; flagged: number }

export function personRows(d: Pick<ReplayData, "runs" | "events" | "fps">, run: RunKey, doc: VerifyDoc | null): PersonRow[] {
  const v = verdictOf(doc, run);
  const { people } = peopleOf(d, run, v);
  return people.map((p: Person) => {
    const kept = p.hidden.filter((h) => v(h.id, h.f1 + 1)?.verdict === "kept").length;
    const status: PersonRow["status"] = p.changes.length ? "changed" : p.swaps.length ? "uncertain" : "kept";
    const detail = p.changes.length ? p.changes.map((c) => `#${c.from}→#${c.to}`).join(", ")
      : p.swaps.length ? `possible swap at frame ${p.swaps.map((s) => s.f).join(", ")}` : "one number";
    return { key: p.key, ids: p.ids, seconds: p.frames / d.fps, status, detail, f0: p.f0, kept, flagged: p.swaps.length };
  });
}

export function runSummary(rows: PersonRow[], doc: VerifyDoc | null, run: RunKey) {
  const c = counts(doc, run);
  return {
    people: rows.length,
    kept: rows.filter((r) => r.status === "kept").length,
    changed: rows.filter((r) => r.status === "changed").length,
    uncertain: rows.filter((r) => r.status === "uncertain").length,
    backConfirmed: c.kept, backFlagged: c.swap, backUnknown: c.unknown,
  };
}

const iou = (a: XYWH, b: XYWH) => {
  const iw = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const i = iw * ih, u = a[2] * a[3] + b[2] * b[3] - i;
  return u > 0 ? i / u : 0;
};
/** The other tracker's id on the same body at frame f (best IoU ≥ 0.3), for cross-pane highlighting. */
export function matchOther(tracks: Map<number, Map<number, XYWH>>, f: number, box: XYWH): number | null {
  let best = 0.3, who: number | null = null;
  for (const [id, m] of tracks) { const b = m.get(f); if (b) { const v = iou(b, box); if (v > best) { best = v; who = id; } } }
  return who;
}

type Choice = { kind: "scn"; id: string } | { kind: "bench"; name: string };

export default function RealCompare() {
  const [params, setParams] = useSearchParams();
  const [index, setIndex] = useState<ScenarioMeta[]>([]);
  const [pairs, setPairs] = useState<BenchPair[]>([]);
  const sel = params.get("clip") ?? "subway_turnstile";
  const [data, setData] = useState<ReplayData | null>(null);
  const [doc, setDoc] = useState<VerifyDoc | null>(null);
  useEffect(() => { fetchIndex().then(setIndex).catch(() => setIndex([])); loadBenchmarkPairs().then(setPairs).catch(() => setPairs([])); }, []);
  useEffect(() => {
    let live = true;
    const choice: Choice = sel.startsWith("bench:") ? { kind: "bench", name: sel.slice(6) } : { kind: "scn", id: sel };
    setData(null);
    if (choice.kind === "scn") fetchScenario(choice.id).then((d) => live && setData(fromScenario(d, videoUrl(d.meta.id)))).catch(() => live && setData(null));
    else { const p = pairs.find((x) => x.name === choice.name); if (p) setData(benchmarkReplay(p)); }
    fetchVerify(choice.kind === "scn" ? choice.id : choice.name).then((d) => live && setDoc(d));
    return () => { live = false; };
  }, [sel, pairs]);
  return (
    <div className="real-compare" data-testid="real-compare">
      <div className="g-head" style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>Compare on real clips</h1>
        <label className="muted" style={{ fontSize: 13 }}>Clip{" "}
          <select value={sel} onChange={(e) => setParams({ clip: e.target.value })} data-testid="rc-clip">
            {orderScenarios(index).map((m) => <option key={m.id} value={m.id}>{m.title} · illustrative</option>)}
            {pairs.map((p) => <option key={p.name} value={`bench:${p.name}`}>{p.name} · human labels (boxes only)</option>)}
          </select>
        </label>
        <Link to="/dashboard/scenarios" className="muted" style={{ marginLeft: "auto", fontSize: 13 }}>← Scenarios</Link>
      </div>
      {data ? <ComparePanes key={data.id} data={data} doc={doc} /> : <p className="muted">Loading clip…</p>}
    </div>
  );
}

export function ComparePanes({ data, doc }: { data: ReplayData; doc: VerifyDoc | null }) {
  const video = useRef<HTMLVideoElement>(null);
  const cv = { baseline: useRef<HTMLCanvasElement>(null), ours: useRef<HTMLCanvasElement>(null) };
  const tracks = useMemo(() => ({ baseline: tracksOf(data.runs.baseline.frames), ours: tracksOf(data.runs.ours.frames) }), [data]);
  const verdict = useMemo(() => ({ baseline: verdictOf(doc, "baseline"), ours: verdictOf(doc, "ours") }), [doc]);
  const rows = useMemo(() => ({ baseline: personRows(data, "baseline", doc), ours: personRows(data, "ours", doc) }), [data, doc]);
  const [hl, setHl] = useState<Record<RunKey, number | null>>({ baseline: null, ours: null });
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const sources = useMemo(() => videoSources(data), [data]);
  const [srcIdx, setSrcIdx] = useState(0);
  const src = sources[srcIdx] ?? null;
  const off = data.videoOffsetS ?? 0;
  const clock = useRef(0);
  const hlRef = useRef(hl); hlRef.current = hl;
  const hasBase = Object.keys(data.runs.baseline.frames).length > 0;

  useEffect(() => {
    let raf = 0, last = performance.now(), lastUi = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      const v = video.current;
      let cur: number;
      if (v && src && v.readyState >= 2) {
        if (playing && v.paused) v.play().catch(() => undefined);
        if (!playing && !v.paused) v.pause();
        if (v.currentTime - off > data.duration) v.currentTime = off;
        cur = Math.max(0, v.currentTime - off);
      } else { if (playing) clock.current += dt; if (clock.current > data.duration) clock.current = 0; cur = clock.current; }
      const ff = Math.min(data.frames, Math.max(1, 1 + cur * data.fps));
      for (const run of RUNS) {
        const c = cv[run].current, ctx = c?.getContext("2d");
        if (!c || !ctx || !c.clientWidth) continue;
        const dpr = window.devicePixelRatio || 1;
        if (c.width !== Math.round(c.clientWidth * dpr)) { c.width = Math.round(c.clientWidth * dpr); c.height = Math.round(c.clientHeight * dpr); }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const h = hlRef.current[run];
        const all = boxesAt(tracks[run], ff, data.fps, verdict[run]);
        paintFrame(ctx, c.clientWidth, c.clientHeight, data, v && v.readyState >= 2 ? v : null, [0, 0, data.width, data.height], all, h, h != null ? new Set([h]) : new Set(), ff, tracks[run]);
      }
      if (now - lastUi > 100) { lastUi = now; setT(cur); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, src, off, playing, tracks, verdict]);

  function pick(run: RunKey, r: PersonRow) {
    const box = tracks[run].get(r.key)?.get(r.f0) ?? null;
    const otherRun: RunKey = run === "ours" ? "baseline" : "ours";
    const o = box ? matchOther(tracks[otherRun], r.f0, box) : null;
    setHl({ [run]: r.key, [otherRun]: o } as Record<RunKey, number | null>);
    const s = Math.max(0, (r.f0 - 1) / data.fps);
    clock.current = s;
    if (video.current) video.current.currentTime = s + off;
  }

  const sums = { baseline: runSummary(rows.baseline, doc, "baseline"), ours: runSummary(rows.ours, doc, "ours") };
  return (
    <div>
      <p className="muted" style={{ margin: "0 0 8px", fontSize: 13 }}>
        Same video, same detections, two trackers. {data.kind === "illustrative" ? "No ground truth here: 'kept' and 'possible swap' come from a heuristic check (hover ⓘ)." : "Human-labelled clip: kept vs swap is checked against the labels."}
        <span className="info" title={methodTooltip(doc)} data-testid="rc-tooltip"> ⓘ</span>
      </p>
      {src && <video key={src} ref={video} src={src} muted playsInline autoPlay style={{ display: "none" }}
        onError={() => setSrcIdx((i) => i + 1)} onLoadedMetadata={(e) => { if (!(e.currentTarget.duration > 0)) setSrcIdx((i) => i + 1); }} />}
      <div className="rc-panes">
        {RUNS.map((run) => (
          <figure key={run} style={{ margin: 0 }} data-testid={`rc-pane-${run}`}>
            <figcaption className="cmp-head"><b className={run === "ours" ? "ours" : "base"}>{run === "ours" ? `Ours: ${data.runs.ours.label}` : `Baseline: ${data.runs.baseline.label}`}</b></figcaption>
            <div className="rc-stage" style={{ aspectRatio: `${data.width} / ${data.height}` }}>
              {run === "baseline" && !hasBase ? <div className="cmp-hint">ByteTrack tracks not exported for this clip.</div> : <canvas ref={cv[run]} aria-label={`${data.title}, ${run}`} />}
            </div>
            <div className="cmp-counters tabular" data-testid={`rc-summary-${run}`}>
              <span>People <b>{sums[run].people}</b></span>
              <span>Kept one number <b>{sums[run].kept}</b></span>
              <span>Changed <b>{sums[run].changed}</b></span>
              <span>Uncertain <b>{sums[run].uncertain}</b></span>
              <span title="Reappearances (same number drawn again after a gap): confirmed same person / flagged possible swap">Back: confirmed <b>{sums[run].backConfirmed}</b> · flagged <b>{sums[run].backFlagged}</b>{sums[run].backUnknown ? <> · unknown <b>{sums[run].backUnknown}</b></> : null}</span>
            </div>
          </figure>
        ))}
      </div>
      <div className="idp-controls">
        <button type="button" className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
        <span className="mono muted tabular">{t.toFixed(1)} / {data.duration.toFixed(1)} s</span>
        {(hl.ours != null || hl.baseline != null) && <button type="button" className="chip-btn" onClick={() => setHl({ baseline: null, ours: null })}>Clear highlight</button>}
      </div>
      <div className="rc-tables">
        {RUNS.map((run) => (
          <table key={run} className="rc-table" data-testid={`rc-table-${run}`}>
            <thead><tr><th>{run === "ours" ? "Ours" : "ByteTrack"}: person</th><th>Seen</th><th>Status</th><th>Back (ok/flag)</th></tr></thead>
            <tbody>
              {rows[run].map((r) => (
                <tr key={r.key} onClick={() => pick(run, r)} style={{ cursor: "pointer", fontWeight: hl[run] === r.key ? 600 : 400 }} data-testid="rc-row">
                  <td className="mono">#{r.key}{r.ids.length > 1 ? ` (${r.ids.map((i) => `#${i}`).join("→")})` : ""}</td>
                  <td className="tabular">{r.seconds.toFixed(1)} s</td>
                  <td>{r.status === "kept" ? "kept one number" : r.status === "changed" ? `changed ${r.detail}` : `uncertain: ${r.detail}`}</td>
                  <td className="tabular">{r.kept}/{r.flagged}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ))}
      </div>
    </div>
  );
}
