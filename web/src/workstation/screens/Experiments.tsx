import { useEffect, useState } from "react";
import type { IndexedClip } from "@viewer/data";
import { pickMedia, type PickedMedia } from "@viewer/useMedia";
import type { Jump } from "../Workstation";
import { FAILURE } from "../lib/journey";
import {
  fetchConfig,
  fetchConfigs,
  fetchGateEvents,
  fetchGoNoGo,
  fetchOutputs,
  runTrack,
  type ClipRef,
  type GateRow,
  type GoNoGoRow,
  type OutputRun,
  type TrackResponse,
} from "../api";
import { ms, num, pct } from "../format";
import { Link } from "react-router-dom";

interface Props {
  refs: ClipRef[];
  clips: IndexedClip[];
  picked: PickedMedia;
  onPicked: (p: PickedMedia) => void;
  onOpenStudio: (j: Jump) => void;
  onClip: (c: IndexedClip) => void;
}

export function Experiments({ refs, picked, onPicked, onOpenStudio, onClip }: Props) {
  const [configs, setConfigs] = useState<string[]>([]);
  const [config, setConfig] = useState("baselines/bytetrack");
  const [yaml, setYaml] = useState<Record<string, unknown> | null>(null);
  const [gates, setGates] = useState<Record<string, GateRow>>({});
  const [gono, setGono] = useState<GoNoGoRow[]>([]);
  const [outputs, setOutputs] = useState<OutputRun[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [videoName, setVideoName] = useState<string | null>(null);
  const [gate, setGate] = useState("");
  const [offset, setOffset] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TrackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);

  useEffect(() => {
    fetchConfigs().then((c) => {
      setConfigs(c);
      if (c.length && !c.includes(config)) setConfig(c[0]);
    });
    fetchGateEvents().then(setGates).catch(() => undefined);
    fetchGoNoGo().then(setGono).catch(() => undefined);
    fetchOutputs().then(setOutputs).catch(() => undefined);
  // eslint-disable-next-line react-hooks/exhaustive-deps -- reload only when the clip changes (ported as-is)
  }, []);

  useEffect(() => {
    if (!config) return;
    fetchConfig(config)
      .then((r) => setYaml(r.config))
      .catch(() => setYaml(null));
  }, [config]);

  async function onRun() {
    if (!file) {
      setError("Choose a Stage A detection cache");
      return;
    }
    setBusy(true);
    setError(null);
    setLog((l) => [...l, `[JOB] POST /api/track config=${config} file=${file.name}`]);
    try {
      const { body, clip } = await runTrack(file, config, gate || undefined);
      setResult(body);
      onClip(clip);
      onPicked({ ...picked, frame_offset: offset });
      setLog((l) => [
        ...l,
        `[PASS] ${body.sequence} · ${body.events.length} events · p95 ${ms(body.latency.p95_ms)}`,
      ]);
    } catch (e: unknown) {
      const msg = String(e);
      setError(msg);
      setLog((l) => [...l, `[ERR] ${msg}`]);
    } finally {
      setBusy(false);
    }
  }

  const delayed = yaml && typeof yaml.delayed === "object" && yaml.delayed ? (yaml.delayed as Record<string, unknown>) : null;
  const vis = yaml && typeof yaml.visibility === "object" && yaml.visibility ? (yaml.visibility as Record<string, unknown>) : null;
  function onVideo(files: File[]) {
    const vid = files.find((f) => f.type.startsWith("video/") || /\.(mp4|webm|mov)$/i.test(f.name));
    const next = pickMedia(files);
    setVideoName(vid?.name ?? (next.frames?.size ? `${next.frames.size} frames` : null));
    onPicked({ ...next, frame_offset: offset });
  }

  function watchDemo() {
    onPicked({ ...picked, frame_offset: offset });
    onOpenStudio({
      clip: FAILURE.clip,
      left: FAILURE.left,
      right: FAILURE.right,
      frame: FAILURE.lookFrom,
      trackId: FAILURE.seedId,
    });
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="max-w-3xl rounded-sm border border-line bg-paper p-3 text-[13px]" data-testid="your-video-moved">
        Uploading your own clip lives in the Studio now:{" "}
        <Link to="/dashboard/studio/your-video" className="text-signal font-semibold" data-testid="try-your-video-link">Try your own video (Studio step 4)</Link>.
        This page keeps the research tools.
      </div>
      <details className="bg-raised border border-line rounded-sm p-4">
        <summary className="cursor-pointer text-[12px] text-mute">Advanced: play a video under the saved demo boxes, or run a Stage A detection cache</summary>
      <section className="space-y-3 mt-3">
        <div
          className="rounded-sm border border-dashed border-line bg-paper px-4 py-10 text-center"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            onVideo([...e.dataTransfer.files]);
          }}
        >
          <label className="cursor-pointer text-[13px] text-mute">
            {videoName || (picked.video ? "Video ready" : "Drop an MP4 or numbered JPEGs here")}
            <input
              type="file"
              multiple
              accept="video/*,image/jpeg,image/png"
              className="hidden"
              onChange={(e) => onVideo([...(e.target.files ?? [])])}
            />
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1 text-[12px] text-mute">
            Frame offset
            <input
              type="number"
              value={offset}
              onChange={(e) => {
                const n = Number(e.target.value) || 0;
                setOffset(n);
                onPicked({ ...picked, frame_offset: n });
              }}
              className="w-20 bg-paper border border-line rounded-sm px-1.5 py-0.5 font-mono text-ink"
            />
          </label>
          <button
            type="button"
            onClick={watchDemo}
            className="px-3 py-2 rounded-sm bg-signal text-paper font-semibold text-[12px]"
          >
            Play under the demo boxes
          </button>
        </div>
        <section className="text-[12px] text-mute">
          <h3 className="text-ink font-semibold">Optional: run tracker on a detection cache</h3>
          <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
            <label className="flex flex-col gap-1">
              Stage A detection cache
              <input type="file" accept=".npz" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
            <label className="flex flex-col gap-1">
              Config
              <select
                value={config}
                onChange={(e) => setConfig(e.target.value)}
                className="bg-paper border border-line rounded-sm px-2 py-1 text-ink"
              >
                {configs.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              Gate x1,y1,x2,y2
              <input
                value={gate}
                onChange={(e) => setGate(e.target.value)}
                placeholder="default h60"
                className="bg-paper border border-line rounded-sm px-2 py-1 text-ink"
              />
            </label>
          </div>
          <button
            type="button"
            onClick={onRun}
            disabled={busy}
            className="mt-3 px-3 py-1.5 rounded-sm border border-line text-ink font-semibold disabled:opacity-50"
          >
            {busy ? "Running…" : "Run tracker on this cache"}
          </button>
          {file && <div className="mt-1 font-mono text-[11px]">{file.name}</div>}
        </section>
        {error && <div className="text-fail text-[12px]">{error}</div>}
        {result && (
          <div className="flex flex-wrap items-center gap-3 bg-paper border border-line rounded-sm px-3 py-2 text-[12px]">
            <span>
              {result.sequence} · {result.config}
            </span>
            <span>events {result.events.length}</span>
            <span>p50 {ms(result.latency.p50_ms)}</span>
            <span>p95 {ms(result.latency.p95_ms)}</span>
            <button
              type="button"
              className="ml-auto px-2 py-1 rounded-sm border border-line font-semibold"
              onClick={() =>
                onOpenStudio({
                  clip: result.sequence,
                  left: result.config.replace("/", "-"),
                  frame: 1,
                })
              }
            >
              Watch this run
            </button>
          </div>
        )}
      </section>
      </details>

      <section className="text-[12px] text-mute">
        <h3 className="text-ink font-semibold">Lab notes (config, delayed no-go, local outputs)</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5 mt-3">
        <section className="bg-paper rounded border border-slate-200 p-3.5">
          <h3 className="font-bold text-slate-900 text-[13px] mb-2">Active config</h3>
          {!yaml && <p className="text-slate-500">Load a config from /api/config.</p>}
          {yaml && (
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 font-mono text-[11px]">
              <span className="text-slate-500">name</span>
              <span className="text-slate-800">{String(yaml.name ?? config)}</span>
              <span className="text-slate-500">track_buffer</span>
              <span>{String((yaml.lifecycle as { track_buffer?: number } | undefined)?.track_buffer ?? "—")}</span>
              <span className="text-slate-500">high_thresh</span>
              <span>{String((yaml.detection as { high_thresh?: number } | undefined)?.high_thresh ?? "—")}</span>
              <span className="text-slate-500">visibility</span>
              <span>{String(vis?.estimator ?? vis?.enabled ?? "—")}</span>
              <span className="text-slate-500">delayed.enabled</span>
              <span className={delayed?.enabled ? "text-brand-700 font-bold" : ""}>
                {String(Boolean(delayed?.enabled))}
              </span>
              <span className="text-slate-500">max_delay_frames</span>
              <span>{delayed?.enabled ? String(delayed.max_delay_frames) : "—"}</span>
              <span className="text-slate-500">appearance</span>
              <span>{String((yaml.appearance as { enabled?: boolean } | undefined)?.enabled ?? false)}</span>
            </dl>
          )}
        </section>
        <section className="bg-paper rounded border border-slate-200 p-3.5">
          <h3 className="font-bold text-slate-900 text-[13px] mb-2">Eval log</h3>
          <div className="h-40 overflow-y-auto bg-slate-50 border border-slate-200 rounded p-2 font-mono text-[11px] space-y-1">
            {log.length === 0 && <div className="text-slate-400">Idle. Runs append here.</div>}
            {log.map((line, i) => (
              <div key={i} className={line.startsWith("[ERR]") ? "text-rose-700" : "text-slate-700"}>
                {line}
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="bg-paper rounded border border-slate-200 overflow-hidden">
        <div className="px-3.5 py-2 bg-slate-50 border-b border-slate-200">
          <h2 className="font-bold text-slate-900 text-[13px]">Gate-event metrics (MOT17 val-half, public dets)</h2>
          <p className="text-[11px] text-slate-500 mt-0.5">
            25 gt crossings / 9 tailgates — too few to rank trackers. Numbers show the metric works.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="bg-slate-50 font-mono text-[11px] uppercase text-slate-600 border-b">
                <th className="py-2 px-3">Tracker</th>
                <th className="py-2 px-3">F1</th>
                <th className="py-2 px-3">Recall</th>
                <th className="py-2 px-3">Attr. err</th>
                <th className="py-2 px-3">Tailgate P / R</th>
                <th className="py-2 px-3">Count err</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(gates).map(([name, g]) => (
                <tr key={name} className="hover:bg-slate-50">
                  <td className="py-2 px-3 font-medium">{name}</td>
                  <td className="py-2 px-3 font-mono">{pct(g.f1 * 100, 1)}</td>
                  <td className="py-2 px-3 font-mono">{pct(g.recall * 100, 1)}</td>
                  <td className="py-2 px-3 font-mono">{pct(g.attribution_error * 100, 1)}</td>
                  <td className="py-2 px-3 font-mono">
                    {pct(g.tailgate_precision * 100, 0)} / {pct(g.tailgate_recall * 100, 0)}
                  </td>
                  <td className="py-2 px-3 font-mono">{num(g.count_error, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-paper rounded border border-slate-200 overflow-hidden">
        <div className="px-3.5 py-2 bg-amber-50 border-b border-amber-200">
          <h2 className="font-bold text-slate-900 text-[13px]">Go / no-go: delayed assignment</h2>
          <p className="text-[11px] text-amber-900 mt-0.5">
            Verdict: no-go. Every K&gt;0 row has more ID switches than K=0 on public FRCNN. Pitch leads
            with visibility + gate events; delayed is an ablation.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="bg-slate-50 font-mono text-[11px] uppercase text-slate-600 border-b">
                <th className="py-2 px-3">Condition</th>
                <th className="py-2 px-3">K</th>
                <th className="py-2 px-3">HOTA</th>
                <th className="py-2 px-3">IDF1</th>
                <th className="py-2 px-3">IDSW</th>
                <th className="py-2 px-3">p95</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {gono
                .filter((r) => r.condition.includes("public FRCNN") || r.K === 0 || r.K === 5)
                .slice(0, 24)
                .map((r, i) => (
                  <tr key={`${r.condition}-${r.K}-${i}`} className={r.K === 0 ? "bg-slate-50" : ""}>
                    <td className="py-2 px-3">{r.condition}</td>
                    <td className="py-2 px-3 font-mono">{r.K}</td>
                    <td className="py-2 px-3 font-mono">{num(r.HOTA, 1)}</td>
                    <td className="py-2 px-3 font-mono">{num(r.IDF1, 1)}</td>
                    <td className="py-2 px-3 font-mono">{Math.round(r.IDSW_trackeval)}</td>
                    <td className="py-2 px-3 font-mono">{ms(r.p95_ms, 1)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-paper rounded border border-slate-200 overflow-hidden">
        <div className="px-3.5 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <h2 className="font-bold text-slate-900 text-[13px]">Local eval outputs</h2>
          <span className="font-mono text-[11px] text-slate-500">{outputs.length} runs · {refs.length} clips</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="bg-slate-50 font-mono text-[11px] uppercase text-slate-600 border-b">
                <th className="py-2 px-3">Split</th>
                <th className="py-2 px-3">Tracker</th>
                <th className="py-2 px-3">HOTA</th>
                <th className="py-2 px-3">IDF1</th>
                <th className="py-2 px-3">IDSW</th>
                <th className="py-2 px-3">Sequences</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {outputs.map((r) => (
                <tr key={`${r.split}-${r.tracker}`}>
                  <td className="py-2 px-3 font-mono">{r.split}</td>
                  <td className="py-2 px-3 font-mono">{r.tracker}</td>
                  <td className="py-2 px-3 font-mono">{r.combined ? num(r.combined.HOTA, 2) : "—"}</td>
                  <td className="py-2 px-3 font-mono">{r.combined ? num(r.combined.IDF1, 2) : "—"}</td>
                  <td className="py-2 px-3 font-mono">{r.combined ? Math.round(r.combined.IDSW) : "—"}</td>
                  <td className="py-2 px-3 font-mono text-[11px] text-slate-500">{r.sequences.length}</td>
                </tr>
              ))}
              {outputs.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-slate-500">
                    No outputs/ yet. Run make eval.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
      </section>
    </div>
  );
}
