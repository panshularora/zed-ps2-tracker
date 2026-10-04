import { useLayoutEffect, useRef } from "react";
import { closingPending, gateCounts, openPending, type IndexedClip } from "./data";
import { fadeProgress, mix } from "./media";
import { idColor, theme } from "./theme";

/** A drawable real frame (video element or decoded JPEG), or null for the blank canvas. */
export type FrameImage = HTMLVideoElement | HTMLImageElement | null;

interface Props {
  clip: IndexedClip;
  frame: number;
  image?: FrameImage;
  /** bumped when the image content changes (new video frame presented, JPEG loaded) */
  imageTick?: number;
}

function ready(img: FrameImage): img is HTMLVideoElement | HTMLImageElement {
  if (!img) return false;
  if (img instanceof HTMLVideoElement) return img.readyState >= 2 && img.videoWidth > 0;
  return img.complete && img.naturalWidth > 0;
}

/** One tracker's view in clip pixel space: real frame (if any), boxes, held decisions, gate. */
export function Panel({ clip, frame, image = null, imageTick = 0 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
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
    if (ready(image)) ctx.drawImage(image, 0, 0, clip.width, clip.height);
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
    // 150 ms fade from dashed white to the solid ID colour once a held window closes
    const fadeFrames = Math.ceil((theme.decisionFadeMs / 1000) * clip.fps);
    const fading = new Map<number, number>(); // id -> progress 0..1
    for (const p of closingPending(clip.pending, frame, fadeFrames)) {
      const t = fadeProgress(frame, p.f1, clip.fps, theme.decisionFadeMs);
      for (const id of p.candidates) if (!held.has(id)) fading.set(id, Math.min(fading.get(id) ?? 1, t));
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
      const fade = fading.get(b.id);
      if (fade !== undefined && fade < 1 && !errorIds.has(b.id)) {
        ctx.setLineDash(theme.heldDash.map((d) => d * px));
        ctx.strokeStyle = mix("#FFFFFF", color, fade, 0.8 * (1 - fade));
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.setLineDash([]);
        ctx.strokeStyle = mix("#FFFFFF", color, fade, fade);
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.fillStyle = mix("#FFFFFF", color, fade, 0.8 + 0.2 * fade);
        const label = `${b.id}`;
        const tw = ctx.measureText(label).width + 6 * px;
        const th = 14 * px;
        ctx.fillRect(b.x, b.y - th, tw, th);
        ctx.fillStyle = theme.labelText;
        ctx.fillText(label, b.x + 3 * px, b.y - th + 2 * px);
        continue;
      }
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
  }, [clip, frame, image, imageTick]);

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
