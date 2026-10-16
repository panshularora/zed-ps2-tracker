import { BASE_NAME, OURS_NAME } from "../lib/ours";
// "What we use", in plain words. Every claim is checked against the repo: docs/licenses.md,
// docs/results/GPU_RUN_2026-10.md, configs/, tracker/visibility.py, tracker/cmc.py,
// third_party/cmc_files. Run status lines come from docs/results/summary.json via datasetRows().
import { Link } from "react-router-dom";
import { datasetRows, loadSummary } from "../lib/summary";

const DATASETS = [
  {
    id: "mot17",
    name: "MOT17",
    cite: "Milan et al. 2016",
    what: "Street and shopping-mall videos where people have been labelled by hand, frame by frame, with one fixed number each.",
    why: "The standard test for \"does each person keep their number\": everyone reports on it, so our numbers can be checked against published ones.",
    licence: "CC BY-NC-SA 3.0 (non-commercial, share-alike; credit MOTChallenge). Source: docs/licenses.md.",
  },
  {
    id: "mot20",
    name: "MOT20",
    cite: "Dendorfer et al. 2020",
    what: "Very dense crowds (stations, squares, night scenes), also hand-labelled.",
    why: "Closest public stand-in for a busy gate: people constantly hide each other.",
    licence: "CC BY-NC-SA 3.0, same as MOT17.",
  },
  {
    id: "dancetrack",
    name: "DanceTrack",
    cite: "Sun et al., CVPR 2022",
    what: "Groups of dancers who look alike and move unpredictably.",
    why: "When clothes and motion don't help, it shows whether a tracker is guessing.",
    licence: "Annotations CC BY 4.0; videos non-commercial.",
  },
  {
    id: "pexels",
    name: "Pexels demo clips",
    cite: "pexels.com",
    what: "Short real gate and station videos (turnstiles, entrances), faces blurred.",
    why: "To show the problem on footage that looks like the product's. Nobody labelled them, so they are illustrative only, never a score.",
    licence: "Pexels License (not Apache-2.0); see web/public/demo/LICENSE_CLIPS.md.",
  },
];

const MODELS = [
  {
    name: "Detector: YOLOX-X",
    cite: "Ge et al. 2021",
    text: "Finds every person in each frame and draws a box with a confidence score. For MOT17 the GPU run uses ByteTrack's own YOLOX-X weights (bytetrack_ablation), so our ByteTrack can be compared with the published one. The committed comparison table on Results uses MOTChallenge's public detections instead, and the Pexels clips use the small YOLOX-S on CPU.",
  },
  {
    name: "ByteTrack (baseline)",
    cite: "Zhang et al., ECCV 2022",
    text: "Links boxes from frame to frame by where each person should be next. Its trick: it keeps the low-confidence boxes for a second matching pass, because a half-hidden person often gets a low score.",
  },
  {
    name: "BoT-SORT",
    cite: "Aharon et al. 2022",
    text: "ByteTrack's matching plus camera-motion correction (CMC): if the camera pans, it shifts every prediction first. Our CMC reads the BoT-SORT authors' pre-computed camera-motion files (third_party/cmc_files); clips without a file run with CMC off.",
  },
  {
    name: "Our visibility extension",
    cite: "configs/ours/visibility_botsort.yaml",
    text: "From the boxes alone it estimates how much of each person is hidden by someone closer to the camera (lower in the picture), multiplied by the detector score. A person who was partly hidden when lost (estimated visibility under 70%) is kept waiting 3x longer for their return before the tracker gives up, so they can come back with the same number.",
  },
  {
    name: "Re-identification (appearance)",
    cite: "OSNet-AIN",
    text: "Supported (configs/baselines/*_reid.yaml) but switched off in every result shown here: all numbers are from position and motion only.",
  },
];

const METRICS = [
  ["HOTA", "One overall score that weighs finding people and keeping their numbers equally. Higher is better."],
  ["IDF1", "How often a person is called by their correct number over the whole video. Higher is better."],
  ["IDSW", "ID switches: how many times a person's number changed. Lower is better."],
  ["MOTA", "Mostly about finding people: misses and false boxes dominate it, number changes count only a little. Higher is better."],
];

const PIPE = [
  ["Camera frames", "video from the gate"],
  ["Detector", "YOLOX: one box per person"],
  ["Tracker", `${OURS_NAME} (left of every comparison: ${BASE_NAME})`],
  ["One number per person", "kept while they are hidden"],
  ["Gate line", "counts IN / OUT once per number"],
];

const STILL: Record<string, string | null> = { mot17: "MOT17-09", mot20: null, dancetrack: null, pexels: "subway_turnstile" };
const MODEL_STATUS: Record<string, string> = {
  "Detector: YOLOX-X": "in use", "ByteTrack (baseline)": "baseline", "BoT-SORT": "base of ours",
  "Our visibility extension": "ours", "Re-identification (appearance)": "off",
};
const REASON: Record<string, string> = {
  "Detector: YOLOX-X": "Finds every person in each frame and gives each a box.",
  "ByteTrack (baseline)": "The published tracker we compare against; left of every comparison.",
  "BoT-SORT": "ByteTrack plus camera-motion correction; the base of ours.",
  "Our visibility extension": "Remembers a half-hidden person 3x longer before dropping their number.",
  "Re-identification (appearance)": "Supported but switched off in every result shown here.",
};

function Still({ id, label }: { id: string | null; label: string }) {
  return id
    ? <img className="wwu-still" src={`${import.meta.env?.BASE_URL ?? "/"}thumbs/${id}.jpg`} alt="" width={48} height={48} />
    : <span className="wwu-still wwu-still-empty" aria-hidden>{label.slice(0, 2)}</span>;
}

export default function WhatWeUse() {
  const rows = Object.fromEntries(datasetRows(loadSummary()).map((r) => [r.id, r]));
  return (
    <div className="wwu" data-testid="what-we-use">
      <h1 className="page-h">What we use</h1>
      <p className="page-lead">Four datasets, one detector and two trackers; the tag says whether each one has run.</p>
      <div className="wwu-rows" data-testid="wwu-datasets">
        {DATASETS.map((d) => (
          <div key={d.id} className="wwu-row" data-testid={`wwu-ds-${d.id}`}>
            <Still id={STILL[d.id]} label={d.name} />
            <span className="wwu-name">{d.name}</span>
            <span className="wwu-why">{d.why}</span>
            <span className="tag wwu-tag">{rows[d.id]?.status ?? "not run yet"}</span>
            <span hidden>{d.licence}</span>
          </div>
        ))}
      </div>
      <div className="wwu-rows" data-testid="wwu-models">
        {MODELS.map((m) => (
          <div key={m.name} className="wwu-row">
            <span className="wwu-still wwu-still-empty" aria-hidden>{m.name.slice(0, 2)}</span>
            <span className="wwu-name">{m.name}</span>
            <span className="wwu-why">{REASON[m.name] ?? m.text}</span>
            <span className="tag wwu-tag">{MODEL_STATUS[m.name] ?? "—"}</span>
            <span hidden>{m.text}</span>
          </div>
        ))}
      </div>
      <details className="about">
        <summary>Licences, scores and the pipeline</summary>
        <div data-testid="wwu-licences">{DATASETS.map((d) => <p key={d.id}>{d.name}: {d.licence}</p>)}</div>
        <dl className="wwu-metrics" data-testid="wwu-metrics">{METRICS.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
        <ol className="pipe" aria-label="Pipeline" data-testid="wwu-pipeline">
          {PIPE.map(([t, x]) => <li key={t} className="pipe-step"><span>{t}</span> <span className="muted">{x}</span></li>)}
        </ol>
        <p className="muted">Comparison table: <Link to="/dashboard/results">Results</Link>.</p>
      </details>
    </div>
  );
}
