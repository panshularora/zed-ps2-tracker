import { type ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ExampleClip } from "../components/ExampleClip";
import { fetchScenario, type ScenarioData } from "../lib/scenarios";
import { CARDS, cardValue, loadSummary, PENDING, type SummaryFile } from "../lib/summary";

export const EXAMPLE_CLIP = "subway_turnstile";
const GH = "https://github.com/panshularora/zed-ps2-tracker/blob/main/";

/** Fade in with a 12 px rise over 320 ms, once, when the step first scrolls into view. */
function Step({ n, title, children, visual }: { n: number; title: string; children: ReactNode; visual: ReactNode }) {
  const ref = useRef<HTMLElement>(null);
  const [shown, setShown] = useState(typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const el = ref.current;
    if (!el || shown) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setShown(true); io.disconnect(); }
    }, { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, [shown]);
  return (
    <section ref={ref} className={`step${shown ? " in" : ""}`} aria-labelledby={`step-${n}`} data-testid="step">
      <div className="step-text">
        <span className="label">Step {n} of 6</span>
        <h2 id={`step-${n}`}>{title}</h2>
        {children}
      </div>
      <div className="step-visual">{visual}</div>
    </section>
  );
}

function FairTestDiagram() {
  const trackers = ["ByteTrack", "BoT-SORT", "OC-SORT", "Hybrid-SORT", "Ours"];
  return (
    <svg viewBox="0 0 640 300" className="diagram" role="img" aria-label="Shared detections feed several trackers, all scored by the same TrackEval scorer on MOT17 and MOT20">
      <rect x="10" y="120" width="140" height="60" rx="4" fill="#161A1F" stroke="#8A929C" />
      <text x="80" y="146" textAnchor="middle" fill="#E6E8EB" fontSize="13">Same detections</text>
      <text x="80" y="164" textAnchor="middle" fill="#8A929C" fontSize="11">MOT17 · MOT20</text>
      {trackers.map((t, i) => {
        const y = 20 + i * 56;
        const ours = t === "Ours";
        return (
          <g key={t}>
            <path d={`M150 150 C 200 150, 200 ${y + 22}, 250 ${y + 22}`} fill="none" stroke="#8A929C" strokeWidth="1.5" />
            <rect x="250" y={y} width="140" height="44" rx="4" fill="#161A1F" stroke={ours ? "#56B4E9" : "#8A929C"} strokeWidth={ours ? 2 : 1} />
            <text x="320" y={y + 27} textAnchor="middle" fill={ours ? "#56B4E9" : "#E6E8EB"} fontSize="13">{t}</text>
            <path d={`M390 ${y + 22} C 440 ${y + 22}, 440 150, 490 150`} fill="none" stroke="#8A929C" strokeWidth="1.5" />
          </g>
        );
      })}
      <rect x="490" y="120" width="140" height="60" rx="4" fill="#161A1F" stroke="#8A929C" />
      <text x="560" y="146" textAnchor="middle" fill="#E6E8EB" fontSize="13">Same scorer</text>
      <text x="560" y="164" textAnchor="middle" fill="#8A929C" fontSize="11">TrackEval</text>
    </svg>
  );
}

export function NumberCards({ summary }: { summary: SummaryFile | null }) {
  return (
    <div className="ncards">
      {CARDS.map((c) => {
        const v = cardValue(c, summary);
        return (
          <div key={c.key} className={`panel ncard${c.loss ? " loss" : ""}`} data-testid="number-card">
            <div className={`nvalue${v === PENDING ? " pending" : ""}`}>{v}</div>
            <p>{c.sentence}</p>
            <div className="nmetric">{c.metric}</div>
          </div>
        );
      })}
    </div>
  );
}

export function LandingView({ example, summary }: { example: ScenarioData | null; summary: SummaryFile | null }) {
  const clip = (run: "baseline" | "ours") =>
    example ? <ExampleClip data={example} run={run} /> : <div className="stage"><div className="empty">Example clip loading…</div></div>;
  return (
    <div className="landing">
      <Step n={1} title="A gate camera mixes people up" visual={clip("baseline")}>
        <p>When two people pass a gate close together, a standard tracker can lose one and give them a new ID. One person then counts twice, or an alert fires for the wrong person.</p>
        <p className="muted small">Real station footage (Pexels, faces blurred). Watch for the moment the box label changes.</p>
      </Step>
      <Step n={2} title="Why this is hard" visual={
        <div className="why">
          {[["Crowds", "Many people inside one small area at once."], ["Occlusion", "People walk behind each other and vanish for a moment."], ["Lookalikes", "Dark coats and similar shapes are easy to confuse."]].map(([h, p]) => (
            <div className="panel" key={h}><h3>{h}</h3><p className="muted">{p}</p></div>
          ))}
        </div>
      }>
        <p>The camera only sees boxes around people. When boxes overlap or someone is hidden, the tracker has to guess who is who.</p>
      </Step>
      <Step n={3} title="What we built" visual={clip("ours")}>
        <p>We took a strong existing tracker (BoT-SORT) and taught it to notice when a person is hidden behind someone else, so it waits for them instead of handing out a new ID.</p>
      </Step>
      <Step n={4} title="How we tested it fairly" visual={<FairTestDiagram />}>
        <p>Every tracker gets exactly the same detected boxes and is scored by the same standard scorer (TrackEval) on two public, human-labelled datasets: MOT17 and MOT20.</p>
        <p className="muted small">Method: <a href={`${GH}docs/evaluation.md`} target="_blank" rel="noreferrer">docs/evaluation.md</a></p>
      </Step>
      <Step n={5} title="What we found" visual={<NumberCards summary={summary} />}>
        <p>Numbers come straight from the results file of the full GPU run. Cards say “pending” until that run is in.</p>
        <p className="muted small">Source: <a href={`${GH}docs/results`} target="_blank" rel="noreferrer">docs/results</a></p>
      </Step>
      <Step n={6} title="Try it" visual={
        <div className="try"><Link to="/dashboard" className="btn primary big">Open the dashboard</Link><Link to="/dashboard/scenarios" className="btn">See the six scenarios</Link></div>
      }>
        <p>Play the clips, switch between the baseline and our tracker, and see the benchmark tables.</p>
      </Step>
    </div>
  );
}

export default function Landing() {
  const [example, setExample] = useState<ScenarioData | null>(null);
  useEffect(() => {
    fetchScenario(EXAMPLE_CLIP).then(setExample, () => setExample(null));
  }, []);
  return <LandingView example={example} summary={loadSummary()} />;
}
