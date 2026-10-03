import { seekFrameForEvent, type ClipEvent } from "./data";
import { theme } from "./theme";

interface Props {
  frames: number;
  fps: number;
  frame: number;
  events: { ev: ClipEvent; side: string }[];
  onSeek: (f: number) => void;
}

function markColor(e: ClipEvent): string {
  if (e.type === "id_switch") return theme.error;
  if (e.type === "cross") return e.correct ? theme.gate : theme.error;
  return theme.text;
}

export function Timeline({ frames, fps, frame, events, onSeek }: Props) {
  const pct = (f: number) => `${((f - 1) / Math.max(1, frames - 1)) * 100}%`;
  return (
    <div className="timeline">
      <input
        aria-label="frame"
        type="range"
        min={1}
        max={frames}
        value={frame}
        onChange={(e) => onSeek(Number(e.target.value))}
      />
      <div className="marks">
        {events.map(({ ev, side }, i) => (
          <button
            key={i}
            className={`mark mark-${ev.type}`}
            title={`${side}: ${ev.type} @ ${ev.f}`}
            style={{ left: pct(ev.f), background: markColor(ev) }}
            onClick={() => onSeek(seekFrameForEvent(ev.f, fps, frames))}
          />
        ))}
        <div className="cursor" style={{ left: pct(frame) }} />
      </div>
    </div>
  );
}
