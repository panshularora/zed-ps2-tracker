import { useLayoutEffect, useRef } from "react";
import { closingPending, gateCounts, openPending, type FrameBox, type IndexedClip } from "./data";
import { canvasToClip, pickBoxAt } from "./identity";
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
  /** hide the built-in title/counters chrome (dashboard wraps the canvas itself) */
  chrome?: boolean;
  className?: string;
  showBoxes?: boolean;
  showHeld?: boolean;
  showGate?: boolean;
  /** IDs that belong to the followed person (walked through id_switch from/to). */
  focusIds?: ReadonlySet<number>;
  /** Draw everyone else faint so the followed person is readable. */
  dimOthers?: boolean;
  /** Keep this colour on the followed person even when their tracker number changes. */
  stableColor?: string;
  /** Label drawn on the followed box (e.g. "Person"). Tracker #id is secondary. */
  personLabel?: string;
  /** Motion trail (centres) for the followed person. */
  trail?: { x: number; y: number }[];
  /** Box IDs that overlap another box this frame (amber fill). */
  overlapIds?: ReadonlySet<number>;
  /** Verified identity call on the followed person this frame. */
  verdict?: "switch" | "recovery" | "hold";
  /** Click a box to follow that ID. */
  onPick?: (id: number | null) => void;
}

function ready(img: FrameImage): img is HTMLVideoElement | HTMLImageElement {
  if (!img) return false;
  if (img instanceof HTMLVideoElement) return img.readyState >= 2 && img.videoWidth > 0;
  return img.complete && img.naturalWidth > 0;
}

/** One tracker's view in clip pixel space: real frame (if any), boxes, held decisions, gate. */
export function Panel({
  clip,
  frame,
  image = null,
  imageTick = 0,
  chrome = true,
  className,
  showBoxes = true,
  showHeld = true,
  showGate = true,
  focusIds,
  dimOthers = false,
  stableColor,
  personLabel,
  trail,
  overlapIds,
  verdict = "hold",
  onPick,
}: Props) {
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
    if (showGate && clip.gate) {
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
    if (showHeld) {
      for (const p of openPending(clip.pending, frame)) {
        for (const id of p.candidates) held.set(id, Math.max(held.get(id) ?? 0, p.f1 - frame));
      }
    }
    // 150 ms fade from dashed white to the solid ID colour once a held window closes
    const fadeFrames = Math.ceil((theme.decisionFadeMs / 1000) * clip.fps);
    const fading = new Map<number, number>(); // id -> progress 0..1
    if (showHeld) {
      for (const p of closingPending(clip.pending, frame, fadeFrames)) {
        const t = fadeProgress(frame, p.f1, clip.fps, theme.decisionFadeMs);
        for (const id of p.candidates) if (!held.has(id)) fading.set(id, Math.min(fading.get(id) ?? 1, t));
      }
    }
    if (trail && trail.length > 1) {
      const color = stableColor ?? theme.gate;
      ctx.beginPath();
      ctx.moveTo(trail[0].x, trail[0].y);
      for (let i = 1; i < trail.length; i++) ctx.lineTo(trail[i].x, trail[i].y);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.75;
      ctx.lineWidth = 3 * px;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.globalAlpha = 1;
      const tip = trail[trail.length - 1];
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 4 * px, 0, Math.PI * 2);
      ctx.fill();
    }

    const errorIds = new Set(
      clip.events
        .filter((e) => e.type === "id_switch" && e.f <= frame && frame - e.f < clip.fps)
        .map((e) => (e.type === "id_switch" ? e.to : -1)),
    );

    if (showBoxes) {
      ctx.font = theme.labelFont.replace("11px", `${11 * px}px`);
      ctx.textBaseline = "top";
      const boxes = clip.byFrame.get(frame) ?? [];
      const following = focusIds && focusIds.size > 0;
      const drawOne = (b: FrameBox, dim: boolean) => {
        const focused = Boolean(following && focusIds!.has(b.id));
        const color = focused && stableColor ? stableColor : idColor(b.id);
        const isHeld = held.has(b.id);
        const overlap = overlapIds?.has(b.id) ?? false;
        ctx.save();
        if (dim) ctx.globalAlpha = 0.18;
        ctx.lineWidth = (focused ? theme.boxWidth + 1 : theme.boxWidth) * px;
        if (overlap && !dim) {
          ctx.fillStyle = "rgba(245, 165, 36, 0.22)";
          ctx.fillRect(b.x, b.y, b.w, b.h);
        }
        const fade = fading.get(b.id);
        if (fade !== undefined && fade < 1 && !errorIds.has(b.id) && !focused) {
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
          ctx.restore();
          return;
        }
        const call =
          focused && verdict === "switch"
            ? "switch"
            : focused && verdict === "recovery"
              ? "recovery"
              : errorIds.has(b.id) && focused
                ? "switch"
                : null;
        if (isHeld && !focused) {
          ctx.strokeStyle = theme.held;
          ctx.setLineDash(theme.heldDash.map((d) => d * px));
        } else {
          ctx.strokeStyle =
            call === "switch" ? theme.error : call === "recovery" ? theme.success : overlap && !focused ? theme.gate : color;
          ctx.setLineDash([]);
          if (call) ctx.lineWidth = (theme.boxWidth + 2) * px;
        }
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.setLineDash([]);
        const switched = clip.events.find(
          (e) =>
            e.type === "id_switch" &&
            e.f <= frame &&
            frame - e.f < clip.fps &&
            (e.from === b.id || e.to === b.id),
        );
        let label = isHeld && !focused ? `${b.id} · held ${held.get(b.id)}` : `${b.id}`;
        if (focused && personLabel) {
          label =
            call === "recovery"
              ? `${personLabel}  kept`
              : switched && switched.type === "id_switch"
                ? `${personLabel}  ${switched.from}→${switched.to}`
                : `${personLabel}  #${b.id}`;
        } else if (focused && switched && switched.type === "id_switch") {
          label = `${switched.from} → ${switched.to}`;
        } else if (focused) {
          label = `this · ${b.id}`;
        }
        const tw = ctx.measureText(label).width + 6 * px;
        const th = 14 * px;
        ctx.fillStyle =
          call === "switch" ? theme.error : call === "recovery" ? theme.success : isHeld && !focused ? theme.held : color;
        ctx.fillRect(b.x, b.y - th, tw, th);
        ctx.fillStyle = call ? "#ffffff" : theme.labelText;
        ctx.fillText(label, b.x + 3 * px, b.y - th + 2 * px);
        ctx.restore();
      };
      if (dimOthers && following) {
        for (const b of boxes) if (!focusIds!.has(b.id)) drawOne(b, true);
        for (const b of boxes) if (focusIds!.has(b.id)) drawOne(b, false);
      } else {
        for (const b of boxes) drawOne(b, false);
      }
    }
  }, [
    clip,
    frame,
    image,
    imageTick,
    showBoxes,
    showHeld,
    showGate,
    focusIds,
    dimOthers,
    stableColor,
    personLabel,
    trail,
    overlapIds,
    verdict,
  ]);

  const counts = gateCounts(clip.events, frame);
  const canvas = (
    <canvas
      ref={ref}
      className={chrome ? undefined : "block w-full h-auto"}
      style={onPick ? { cursor: "pointer" } : undefined}
      onClick={
        onPick
          ? (e) => {
              const cv = ref.current;
              if (!cv) return;
              const p = canvasToClip(cv, clip.width, e.clientX, e.clientY);
              const hit = pickBoxAt(clip.byFrame.get(frame) ?? [], p.x, p.y);
              onPick(hit ? hit.id : null);
            }
          : undefined
      }
    />
  );
  if (!chrome) {
    return (
      <div className={className ?? "h-full w-full"} data-testid={`panel-${clip.tracker}`}>
        {canvas}
      </div>
    );
  }
  return (
    <section className={className ?? "panel"} data-testid={`panel-${clip.tracker}`}>
      <header>
        <span className="title">{clip.tracker}</span>
        {clip.gate && (
          <span className="counters">
            in <b>{counts.in}</b> · out <b>{counts.out}</b>
          </span>
        )}
      </header>
      {canvas}
    </section>
  );
}
