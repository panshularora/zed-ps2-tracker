import { Term } from "../components/Term";
import { Link } from "react-router-dom";
import { DatasetTable } from "../components/DatasetTable";
import { api, type Benchmarks, type Clip } from "../api";
import { DASH, int, num } from "../lib/format";
import { useAsync } from "../lib/useAsync";

const DATASETS = [
  ["MOT17", "Standard street-camera pedestrians. Our headline numbers use the val-half split."],
  ["MOT20", "Very crowded scenes. Harder counting and more overlap."],
  ["DanceTrack", "People in similar clothes, so appearance matching is weak. A negative control for “just look at the clothes”."],
  ["SOMPT22", "Static elevated cameras on poles — the closest public set to a gate camera. Extracted locally; detector caches not run yet."],
  ["Pexels clips", "Demo videos with faces blurred. No human labels, so they are not a score."],
  ["Synthetic gate", "Toy people with exact ground truth. Used to watch keep vs switch live."],
];
const TRACKERS = [
  ["ByteTrack", "Baseline. Links boxes by overlap from frame to frame. After a hide it often hands out a new number."],
  ["BoT-SORT", "ByteTrack plus camera-motion correction and a Kalman filter. Headline comparison tracker."],
  ["OC-SORT / Hybrid-SORT / Deep OC-SORT", "Other published baselines we re-ran on the same detections."],
  ["Ours: BoT-SORT + visibility", "One tracker everywhere (gate scene, clips, MOT17): a person whose box is estimated as hidden is remembered 3× longer before their number is retired."],
  ["Delayed assignment", "Wait extra frames before committing an ID. Measured and left off: documented no-go."],
];

const STILL_CLIP = "subway_turnstile";

export interface OverviewData {
  bench: Benchmarks | null;
  still: Clip | null;
}

async function load(): Promise<OverviewData> {
  const [bench, clips] = await Promise.allSettled([api.benchmarks(), api.clips()]);
  return {
    bench: bench.status === "fulfilled" ? bench.value : null,
    still: clips.status === "fulfilled" ? (clips.value.find((c) => c.id === STILL_CLIP) ?? null) : null,
  };
}

export function OverviewView({ data }: { data: OverviewData | null }) {
  const head = data?.bench?.headline ?? null;
  const still = data?.still?.poster_url ?? null;
  const tiles = [
    { label: "IDF1", term: "IDF1", value: num(head?.IDF1), unit: head ? "%" : "", src: "identity F1, higher is better" },
    { label: "ID switches", term: "IDSW", value: int(head?.IDSW), unit: "", src: "over the 7 val-half sequences, lower is better" },
    { label: "Tracker p95 latency", term: "p95 ms", value: num(head?.p95_ms, 2), unit: head?.p95_ms != null ? "ms" : "", src: "per frame, one CPU thread, tracker only" },
  ];
  return (
    <div>
      <section className="hero" style={still ? { backgroundImage: `url(${still})` } : undefined} aria-label="Turnstile still, faces blurred">
        {still && <span className="hero-credit">Pexels · faces blurred</span>}
        <div className="hero-text">
          <span className="label">Overview · Zed Digital PS-2</span>
          <h1 style={{ marginTop: 8 }}>
            Every person crossing a crowded gate keeps one ID from approach to exit, even while hidden behind someone else.
          </h1>
        </div>
      </section>

      <div className="tiles">
        {tiles.map((t) => (
          <div className="panel tile" key={t.label} data-testid="tile">
            <div className="label"><Term name={t.term}>{t.label}</Term></div>
            <div className="value">
              {t.value}
              {t.value !== DASH && t.unit && <small>{t.unit}</small>}
            </div>
            <div className="src">{t.value === DASH ? "not measured" : t.src}</div>
          </div>
        ))}
      </div>
      <p className="caption">
        {head
          ? `${head.label} on MOT17 val-half with public detections (no re-ID), from docs/results/ablation. GPU-detector numbers are pending the DGX run.`
          : "Benchmark results unavailable: the service could not read docs/results."}
      </p>

      <div className="cta">
        <Link to="/dashboard/live" className="btn primary">
          Open the gate scene
        </Link>
        <Link to="/dashboard/scenarios" className="btn">
          Scenarios
        </Link>
        <Link to="/dashboard/results#stress-map" className="btn" data-testid="overview-stress-map">
          Stress Map: when does each tracker lose an ID?
        </Link>
      </div>

      <section className="section cols3">
        <div className="panel">
          <h3>Problem</h3>
          <p>
            At a gate, a person who walks behind someone else for a second often comes back with a new ID. Hands-free validation,
            tailgating checks and passenger counts all need one ID per person per crossing.
          </p>
        </div>
        <div className="panel">
          <h3>Approach</h3>
          <p>
            Strictly online BoT-SORT-style tracking plus a visibility estimate from box geometry: tracks that disappear while
            occluded are kept alive longer and re-associated, instead of being replaced.
          </p>
        </div>
        <div className="panel">
          <h3>Status</h3>
          <p>
            CPU phase done on MOT17 with public detections. YOLOX detections and re-ID are pending a GPU run; the delayed-decision idea
            was tested and is a documented no-go.
          </p>
        </div>
      </section>

      <section className="section" data-testid="what-we-use">
        <h2>Datasets and trackers</h2>
        <p className="sub">What each name on this dashboard is. Scores live in the table below; empty cells mean the run is not in summary.json yet.</p>
        <div className="glossary">
          <div>
            <h3>Datasets</h3>
            {DATASETS.map(([h, p]) => (
              <p key={h}><strong>{h}.</strong> {p}</p>
            ))}
          </div>
          <div>
            <h3>Trackers</h3>
            {TRACKERS.map(([h, p]) => (
              <p key={h}><strong>{h}.</strong> {p}</p>
            ))}
          </div>
        </div>
      </section>

      <section className="section" data-testid="overall-comparison">
        <h2>All datasets</h2>
        <p className="sub">ByteTrack vs BoT-SORT vs ours, read only from summary.json.</p>
        <DatasetTable />
      </section>
    </div>
  );
}

export default function Overview() {
  const { data } = useAsync(load);
  return <OverviewView data={data} />;
}
