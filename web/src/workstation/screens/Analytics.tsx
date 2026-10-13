import { useEffect, useMemo, useState } from "react";
import type { IndexedClip } from "@viewer/data";
import type { Jump } from "../Workstation";
import { fetchAblation, type AblationRow } from "../api";
import { FAILURE, GAP_RESUME, GATE_FLOWS, METHOD, PAPERS, prettyTracker } from "../lib/journey";

interface Props {
  clips: IndexedClip[];
  onInspect: (j: Jump) => void;
}

const SHOW = ["bytetrack", "botsort", "ours_vis_botsort"] as const;

export function Analytics({ clips, onInspect }: Props) {
  const [rows, setRows] = useState<AblationRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAblation()
      .then(setRows)
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const shown = SHOW.map((name) => rows.find((r) => r.name === name)).filter(
    (r): r is AblationRow => Boolean(r),
  );
  const clipCounts = useMemo(() => clipSwitches(clips, FAILURE.clip), [clips]);

  const bestIdf1 = bestOf(shown, "IDF1", false);
  const bestIdsw = bestOf(shown, "IDSW", true);
  const clipWinner = clipCounts.length
    ? [...clipCounts].sort((a, b) => a.n - b.n)
    : [];
  const comparable = clipWinner.length >= 2;
  const clipTied = comparable && clipWinner[0].n === clipWinner[clipWinner.length - 1].n;
  const clipBest = comparable && !clipTied ? clipWinner[0] : null;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <header className="max-w-4xl space-y-1">
        <div className="font-display uppercase tracking-[0.18em] text-[11px] text-signal">
          Step 3 · the scores
        </div>
        <h1 className="font-display text-[26px] leading-none text-ink font-semibold">
          How often this happens, and what we tried
        </h1>
        <p className="text-[13px] text-mute max-w-3xl">
          On the clip you just watched, both trackers change LOOK from #{FAILURE.from} to #
          {FAILURE.to} around frames 69–70. The job is to stop that. The tables below are real
          scores. A “best” badge only appears on the table that tracker actually wins.
        </p>
      </header>

      <section className="bg-raised border border-line rounded-sm overflow-hidden" data-testid="gate-flows">
        <div className="px-4 py-2.5 border-b border-line">
          <h2 className="font-semibold text-ink text-[13px]">What the gate reports</h2>
          <p className="text-[12px] text-mute mt-0.5">
            If someone walks through, we log that they went in or out. If two people cross almost
            together, we log that too. If a number jumps on the same person, that is an ID switch.
            We only have 25 crossings on public detections, so this shows the metric works — it is
            not a ranking.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-mute border-b border-line">
                <th className="py-2 px-4">At the gate</th>
                <th className="py-2 px-4">If identity fails</th>
                <th className="py-2 px-4">What we measure</th>
                <th className="py-2 px-4">On public FRCNN</th>
              </tr>
            </thead>
            <tbody>
              {GATE_FLOWS.map((f) => (
                <tr key={f.flow} className="border-b border-line/60 align-top">
                  <td className="py-2.5 px-4 font-medium text-ink">{f.flow}</td>
                  <td className="py-2.5 px-4 text-mute">{f.break}</td>
                  <td className="py-2.5 px-4">{f.metric}</td>
                  <td className="py-2.5 px-4 text-mute">{f.public}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {error && <div className="bg-fail/15 text-fail border border-fail/40 rounded-sm p-3 font-mono">{error}</div>}

      <section className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <Measure
          kicker="This clip · MOT17-09"
          title="ID switches"
          body={
            clipCounts.length
              ? clipCounts.map((c) => `${prettyTracker(c.tracker).title} ${c.n}`).join(" · ")
              : "Loading clip events…"
          }
          best={
            clipBest
              ? `${prettyTracker(clipBest.tracker).title} fewer switches on this clip`
              : "No winner — both switch at frames 69–70"
          }
          awarded={Boolean(clipBest)}
        />
        <Measure
          kicker="MOT17 val-half · public FRCNN"
          title="IDF1"
          body={shown.map((r) => `${shortLabel(r)} ${r.IDF1.toFixed(1)}`).join(" · ")}
          best={bestIdf1 ? `${shortLabel(bestIdf1)} highest IDF1 on this table` : "—"}
          awarded={Boolean(bestIdf1)}
        />
        <Measure
          kicker="MOT17 val-half · public FRCNN"
          title="IDSW"
          body={shown.map((r) => `${shortLabel(r)} ${Math.round(r.IDSW)}`).join(" · ")}
          best={bestIdsw ? `${shortLabel(bestIdsw)} fewest switches on this table` : "—"}
          awarded={Boolean(bestIdsw)}
        />
      </section>

      <section className="bg-raised border border-line rounded-sm overflow-hidden">
        <div className="px-4 py-2.5 border-b border-line">
          <h2 className="font-semibold text-ink text-[13px]">Public FRCNN MOT17 val-half</h2>
          <p className="text-[12px] text-mute mt-0.5">
            This detector barely sees people who are blocked (about 1% of the most occluded ones).
            Do not read this table as “identity through occlusion”. Delayed assignment is not in
            here — we tried it and dropped it.
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-mute border-b border-line">
                <th className="py-2 px-4">Tracker</th>
                <th className="py-2 px-4">HOTA</th>
                <th className="py-2 px-4">IDF1</th>
                <th className="py-2 px-4">IDSW</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.name} className="border-b border-line/60">
                  <td className="py-2.5 px-4 font-medium text-ink">{r.label}</td>
                  <td className="py-2.5 px-4 font-mono">
                    {r.HOTA.toFixed(2)}
                    {bestOf(shown, "HOTA", false)?.name === r.name && <Best />}
                  </td>
                  <td className="py-2.5 px-4 font-mono">
                    {r.IDF1.toFixed(2)}
                    {bestIdf1?.name === r.name && <Best />}
                  </td>
                  <td className="py-2.5 px-4 font-mono">
                    {Math.round(r.IDSW)}
                    {bestIdsw?.name === r.name && <Best />}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-raised border border-line rounded-sm p-4 space-y-3">
        <div>
          <h2 className="font-semibold text-ink text-[13px]">Do they come back with the same number?</h2>
          <p className="text-[11px] text-mute mt-0.5">{GAP_RESUME.source}</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <GapCard bin={GAP_RESUME.short.bin} rows={GAP_RESUME.short.rows} />
          <GapCard bin={GAP_RESUME.long.bin} rows={GAP_RESUME.long.rows} note={GAP_RESUME.long.note} />
        </div>
      </section>

      <section className="bg-raised border border-line rounded-sm p-4 space-y-2" data-testid="method">
        <h2 className="font-semibold text-ink text-[13px]">What we run, what we kept, what we dropped</h2>
        <p className="text-[12px] text-ink">{METHOD.ship}</p>
        <p className="text-[12px] text-mute">{METHOD.keep}</p>
        <ul className="text-[12px] text-mute list-disc pl-4 space-y-1">
          {METHOD.reject.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        <p className="text-[12px] text-mute" data-testid="privacy-note">
          The numbers are tracker IDs, not faces. Nothing is stored as an identity, and the video
          stays on the machine.
        </p>
      </section>

      <section className="bg-raised border border-line rounded-sm overflow-hidden" data-testid="papers">
        <div className="px-4 py-2.5 border-b border-line">
          <h2 className="font-semibold text-ink text-[13px]">What the scores actually mean</h2>
        </div>
        <div className="divide-y divide-line/80">
          {PAPERS.map((p) => (
            <div key={p.title} className="px-4 py-2.5 grid grid-cols-1 sm:grid-cols-[10rem_1fr] gap-1 sm:gap-4">
              <div className="font-semibold text-ink text-[12px]">{p.title}</div>
              <div className="text-[12px] text-mute">{p.used}</div>
            </div>
          ))}
        </div>
      </section>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() =>
            onInspect({
              clip: FAILURE.clip,
              left: FAILURE.left,
              right: FAILURE.right,
              frame: FAILURE.frame,
              trackId: FAILURE.seedId,
            })
          }
          className="px-3 py-2 rounded-sm border border-line text-ink font-semibold text-[12px]"
        >
          ← Back to the overlap
        </button>
      </div>
    </div>
  );
}

function Measure({
  kicker,
  title,
  body,
  best,
  awarded,
}: {
  kicker: string;
  title: string;
  body: string;
  best: string;
  awarded: boolean;
}) {
  return (
    <article className="bg-raised border border-line rounded-sm p-3.5 space-y-1.5">
      <div className="font-display uppercase tracking-[0.16em] text-[10px] text-mute">{kicker}</div>
      <h3 className="font-display text-[20px] text-ink leading-none">{title}</h3>
      <p className="text-[12px] text-mute">{body}</p>
      <p className={`text-[12px] font-semibold ${awarded ? "text-keep" : "text-mute"}`}>{best}</p>
    </article>
  );
}

function GapCard({
  bin,
  rows,
  note,
}: {
  bin: string;
  rows: readonly { name: string; pct: number; best?: boolean }[];
  note?: string;
}) {
  return (
    <div className="border border-line rounded-sm p-3 space-y-2">
      <div className="text-[12px] font-semibold text-ink">{bin}</div>
      {rows.map((r) => (
        <div key={r.name} className="flex items-center justify-between font-mono text-[12px]">
          <span className="text-mute">{r.name}</span>
          <span className={r.best ? "text-keep font-bold" : "text-ink"}>
            {r.pct.toFixed(1)}%{r.best ? " best" : ""}
          </span>
        </div>
      ))}
      {note && <p className="text-[11px] text-mute">{note}</p>}
    </div>
  );
}

function Best() {
  return <span className="ml-2 text-[10px] uppercase tracking-wide text-keep font-bold">best</span>;
}

function shortLabel(r: AblationRow): string {
  if (r.name === "ours_vis_botsort") return "vis-BoT";
  if (r.name === "botsort") return "BoT-SORT";
  return "ByteTrack";
}

function bestOf(rows: AblationRow[], key: "IDF1" | "IDSW" | "HOTA", invert: boolean): AblationRow | null {
  if (!rows.length) return null;
  return rows.reduce((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (invert) return bv < av ? b : a;
    return bv > av ? b : a;
  });
}

function clipSwitches(clips: IndexedClip[], name: string) {
  return clips
    .filter((c) => c.clip === name && (c.tracker === "botsort" || c.tracker === "bytetrack"))
    .map((c) => ({
      tracker: c.tracker,
      n: c.events.filter((e) => e.type === "id_switch").length,
    }));
}
