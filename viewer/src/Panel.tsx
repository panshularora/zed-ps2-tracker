import { useEffect, useRef } from "react";
import { gateCounts, openPending, type IndexedClip } from "./data";
import { idColor, theme } from "./theme";

interface Props {
  clip: IndexedClip;
  frame: number;
}

/** One tracker's view: blank canvas in clip pixel space, boxes, held decisions, gate. */
export function Panel({ clip, frame }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const cssW = cv.clientWidth || 640;
    const scale = cssW / clip.width;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(clip.height * scale * dpr);
    ctx.setTransform(scale * dpr, 0, 0, scale * dpr, 0, 0);
    ctx.fillStyle = theme.background;
    ctx.fillRect(0, 0, clip.width, clip.height);
    const px = 1 / scale; // one CSS pixel in clip units

    // gate (pulses once over 300 ms of video time after a crossing)
    if (clip.gate) {
      const pulseFrames = Math.max(1, Math.round((theme.gatePulseMs / 1000) * clip.fps));
      const pulsing = clip.events.some(
        (e) => e.type === "cross" && e.f <= frame && frame - e.f < pulseFrames,
      );
      ctx.strokeStyle = theme.gate;
      ctx.lineWidth = (pulsing ? 2 * theme.gateWidth : theme.gateWidth) * px;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(clip.gate.a[0], clip.gate.a[1]);
      ctx.lineTo(clip.gate.b[0], clip.gate.b[1]);
      ctx.stroke();
    }

    const held = new Map<number, number>(); // id -> frames left
    for (const p of openPending(clip.pending, frame)) {
      for (const id of p.candidates) held.set(id, Math.max(held.get(id) ?? 0, p.f1 - frame));
    }
    const errorIds = new Set(
      clip.events
        .filter((e) => e.type === "id_switch" && e.f <= frame && frame - e.f < clip.fps)
        .map((e) => (e.type === "id_switch" ? e.to : -1)),
    );

    ctx.font = theme.labelFont.replace("11px", `${11 * px}px`);
    ctx.textBaseline = "top";
    for (const b of clip.byFrame.get(frame) ?? []) {
      const color = idColor(b.id);
      const isHeld = held.has(b.id);
      ctx.lineWidth = theme.boxWidth * px;
      if (isHeld) {
        ctx.strokeStyle = theme.held;
        ctx.setLineDash(theme.heldDash.map((d) => d * px));
      } else {
        ctx.strokeStyle = errorIds.has(b.id) ? theme.error : color;
        ctx.setLineDash([]);
      }
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.setLineDash([]);
      const label = isHeld ? `${b.id} · held ${held.get(b.id)}` : `${b.id}`;
      const tw = ctx.measureText(label).width + 6 * px;
      const th = 14 * px;
      ctx.fillStyle = isHeld ? theme.held : color;
      ctx.fillRect(b.x, b.y - th, tw, th);
      ctx.fillStyle = theme.labelText;
      ctx.fillText(label, b.x + 3 * px, b.y - th + 2 * px);
    }
  }, [clip, frame]);

  const counts = gateCounts(clip.events, frame);
  return (
    <section className="panel" data-testid={`panel-${clip.tracker}`}>
      <header>
        <span className="title">{clip.tracker}</span>
        {clip.gate && (
          <span className="counters">
            in <b>{counts.in}</b> · out <b>{counts.out}</b>
          </span>
        )}
      </header>
      <canvas ref={ref} />
    </section>
  );
}
