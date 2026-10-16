import { useState } from "react";
import { neutralColor } from "../lib/scenarios";
import type { Person } from "./story";
import { summaryText } from "./story";

export const MAX_ROWS = 12;

/** One 8px bar per person: solid where seen, dashed 50 % where hidden, a 4px break + tag at an ID
 * change, and a break + "swap?" tag where the kept-vs-swap check flagged a possible identity transfer
 * (no dashed bridge there: we do not claim it was still the same person). */
export const TINY_GAP_S = 0.3, SHORT_TRACK_S = 1;

/** Quiet view: short tracks (< 1 s, no ID change) are hidden and tiny gaps (< 0.3 s) drawn as one
 * clean bar. ID changes and flagged swaps are never hidden. */
export function quietView(people: Person[], fps: number, all: boolean): { shown: Person[]; hiddenCount: number } {
  if (all) return { shown: people, hiddenCount: 0 };
  const shown = people
    .filter((p) => p.changes.length > 0 || p.swaps.length > 0 || p.frames / fps >= SHORT_TRACK_S)
    .map((p) => {
      const tiny = p.hidden.filter((h) => (h.f1 - h.f0 + 1) / fps < TINY_GAP_S);
      if (!tiny.length) return p;
      return { ...p, hidden: p.hidden.filter((h) => !tiny.includes(h)), visible: [...p.visible, ...tiny] };
    });
  return { shown, hiddenCount: people.length - shown.length };
}

export function PersonTimeline({ people: everyone, frames, f, focus, onSeek, fps = 30 }: { people: Person[]; frames: number; f: number; focus: number | null; onSeek: (frame: number) => void; fps?: number }) {
  const [all, setAll] = useState(false);
  const { shown: people, hiddenCount } = quietView(everyone, fps, all);
  const rows = [...people].sort((a, b) => (a.key === focus ? -1 : b.key === focus ? 1 : a.f0 - b.f0)).slice(0, MAX_ROWS);
  const pct = (x: number) => `${(((x - 1) / Math.max(1, frames - 1)) * 100).toFixed(3)}%`;
  const width = (a: number, b: number) => `${(((b - a + 1) / Math.max(1, frames)) * 100).toFixed(3)}%`;
  const seekAt = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(Math.max(1, Math.min(frames, Math.round(1 + ((e.clientX - r.left) / Math.max(1, r.width)) * (frames - 1)))));
  };
  return (
    <section className="ptl" aria-label="Per-person timeline" data-testid="person-timeline">
      <div className="ptl-summary" data-testid="timeline-summary">{summaryText(everyone)}
        {(hiddenCount > 0 || all) && (
          <button type="button" className="chip-btn ptl-all" data-testid="ptl-show-all" aria-pressed={all} onClick={() => setAll((x) => !x)}>
            {all ? "Hide short tracks and tiny gaps" : `Show all (${hiddenCount} short tracks hidden)`}
          </button>
        )}</div>
      <div className="ptl-rows">
        {rows.map((p) => (
          <div key={p.key} className={`ptl-row${p.key === focus ? " focus" : ""}`} data-testid="ptl-row">
            <span className="ptl-id mono" style={{ color: neutralColor(p.key) }}>#{p.key}</span>
            <div className="ptl-bar" onClick={seekAt} role="button" tabIndex={-1} aria-label={`Person #${p.key}: seek`}>
              {p.visible.map((s, i) => (
                <span key={`v${i}`} className="ptl-seen" data-testid="ptl-seen" style={{ left: pct(s.f0), width: width(s.f0, s.f1), background: neutralColor(s.id) }} />
              ))}
              {p.hidden.map((s, i) => (
                <span key={`h${i}`} className="ptl-hidden" data-testid="ptl-hidden" style={{ left: pct(s.f0), width: width(s.f0, s.f1), color: neutralColor(s.id) }} />
              ))}
              {p.swaps.map((c, i) => (
                <button key={`s${i}`} type="button" className="ptl-swap" data-testid="ptl-swap" style={{ left: pct(c.f) }}
                  onClick={(e) => { e.stopPropagation(); onSeek(c.last); }} title={`Possible swap: #${c.id} given to someone else (${c.why})`} aria-label={`Possible swap #${c.id}`}>
                </button>
              ))}
              {p.changes.map((c, i) => (
                <button key={`c${i}`} type="button" className="ptl-change" data-testid="ptl-change" style={{ left: pct(c.f) }}
                  onClick={(e) => { e.stopPropagation(); onSeek(c.f); }} title={`ID change #${c.from} → #${c.to}`}>
                  <span className="ptl-tag mono">#{c.from} → #{c.to}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
        <div className="ptl-head" style={{ left: `calc(40px + (100% - 40px) * ${((f - 1) / Math.max(1, frames - 1)).toFixed(4)})` }} aria-hidden data-testid="ptl-playhead" />
      </div>
      {people.length > rows.length && <div className="ptl-more muted">{people.length - rows.length} more people not drawn (summary counts everyone).</div>}
    </section>
  );
}
