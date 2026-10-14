import { useLayoutEffect, useRef, useState } from "react";
import { closingPending, gateCounts, openPending, type FrameBox, type IndexedClip } from "./data";
import { lockedCrowdCrop, pickBoxAt, type OverlapPair } from "./identity";
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
  /** Last place this person stood, drawn dashed while they are missing or just back. */
  ghost?: FrameBox | null;
  ghostLabel?: string;
  /** Zoom the canvas to the people. Crop is locked for the clip so the camera does not pan. */
  cropCrowd?: boolean;
  /** Leave / return pins for the followed person (ID break). */
  idBreak?: {
    leave: FrameBox | null;
    leaveFrame: number;
    back: FrameBox | null;
    backFrame: number;
    fromId: number;
    toId: number;
    phase: "here" | "gone" | "back";
  } | null;
  /** Box IDs that overlap another box this frame (amber fill). */
  overlapIds?: ReadonlySet<number>;
  /** Overlap pairs to hatch the shared pixels of two people. */
  overlaps?: readonly OverlapPair[];
  /** The ID that must survive overlap / leave / return (LOOK's original number). */
  keepId?: number;
  /** Ground-truth ID for that same person, drawn as the keep box when present. */
  gtId?: number;
  keepBox?: FrameBox | null;
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

function hatchRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  px: number,
  color: string,
) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.28;
  ctx.fillRect(x, y, w, h);
  ctx.globalAlpha = 0.95;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6 * px;
  const step = 7 * px;
  for (let t = -h; t < w + h; t += step) {
    ctx.beginPath();
    ctx.moveTo(x + t, y);
    ctx.lineTo(x + t + h, y + h);
    ctx.stroke();
  }
  ctx.restore();
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
  ghost = null,
  ghostLabel,
  cropCrowd = false,
  idBreak = null,
  overlapIds,
  overlaps,
  keepId,
  gtId,
  keepBox = null,
  verdict = "hold",
  onPick,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const cropRef = useRef({ x: 0, y: 0, w: 1, h: 1 });
  const [hostSize, setHostSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host || chrome) return;
    const ro = new ResizeObserver(() => {
      setHostSize({ w: host.clientWidth, h: host.clientHeight });
    });
    ro.observe(host);
    setHostSize({ w: host.clientWidth, h: host.clientHeight });
    return () => ro.disconnect();
  }, [chrome]);

  useLayoutEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const cssW = (!chrome && hostSize.w ? hostSize.w : cv.clientWidth) || 640;
    const cssH = !chrome && hostSize.h ? hostSize.h : 0;
    const crop = cropCrowd
      ? lockedCrowdCrop(clip)
      : { x: 0, y: 0, w: clip.width, h: clip.height };
    cropRef.current = crop;
    const scale = cssH > 0 ? Math.min(cssW / crop.w, cssH / crop.h) : cssW / crop.w;
    const dpr = window.devicePixelRatio || 1;
    const drawW = crop.w * scale;
    const drawH = crop.h * scale;
    if (!chrome) {
      cv.style.width = `${drawW}px`;
      cv.style.height = `${drawH}px`;
    }
    cv.width = Math.round(drawW * dpr);
    cv.height = Math.round(drawH * dpr);
    ctx.setTransform(scale * dpr, 0, 0, scale * dpr, -crop.x * scale * dpr, -crop.y * scale * dpr);
    ctx.fillStyle = theme.background;
    ctx.fillRect(crop.x, crop.y, crop.w, crop.h);
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
      const overlapSet =
        overlapIds ??
        (overlaps?.length ? new Set(overlaps.flatMap((p) => [p.a.id, p.b.id])) : undefined);
      const drawOne = (b: FrameBox, dim: boolean) => {
        const focused = Boolean(following && focusIds!.has(b.id));
        const color = focused && stableColor ? stableColor : idColor(b.id);
        const isHeld = held.has(b.id);
        const overlap = overlapSet?.has(b.id) ?? false;
        ctx.save();
        if (dim) ctx.globalAlpha = 0.18;
        ctx.lineWidth = (focused ? theme.boxWidth + 1.5 : overlap ? theme.boxWidth : 1) * px;
        if (overlap && !dim) {
          ctx.fillStyle = focused ? "rgba(245, 165, 36, 0.16)" : "rgba(245, 165, 36, 0.10)";
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
          focused && verdict === "recovery"
            ? "recovery"
            : focused && (verdict === "switch" || errorIds.has(b.id))
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
        let label = `#${b.id}`;
        if (focused && personLabel) {
          if (call === "recovery") label = `${personLabel} kept #${b.id}`;
          else if (keepId != null && b.id !== keepId)
            label = `${personLabel} got #${b.id}`;
          else if (switched && switched.type === "id_switch")
            label = `${personLabel} ${switched.from}→${switched.to}`;
          else if (keepId != null) label = `${personLabel} keep #${b.id}`;
          else label = `${personLabel} #${b.id}`;
        } else if (overlap && keepId != null && !focused) {
          label = `other #${b.id}`;
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
    if (overlaps?.length) {
      ctx.save();
      ctx.font = theme.labelFont.replace("11px", `${11 * px}px`);
      ctx.textBaseline = "top";
      for (const p of overlaps) {
        const r = p.inter;
        hatchRect(ctx, r.x, r.y, r.w, r.h, px, theme.gate);
        ctx.strokeStyle = theme.gate;
        ctx.lineWidth = 2 * px;
        ctx.setLineDash([]);
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        const lookBox = focusIds?.has(p.a.id) ? p.a : focusIds?.has(p.b.id) ? p.b : null;
        const otherBox = lookBox === p.a ? p.b : lookBox === p.b ? p.a : null;
        const chip =
          lookBox && otherBox && keepId != null && lookBox.id !== keepId
            ? `got #${lookBox.id} overlaps #${otherBox.id}  keep #${keepId}`
            : lookBox && otherBox && keepId != null
              ? `#${lookBox.id} overlaps #${otherBox.id}  keep #${keepId}`
              : `overlap #${p.a.id}+#${p.b.id}`;
        const tw = ctx.measureText(chip).width + 8 * px;
        const th = 15 * px;
        const cx = r.x + r.w / 2 - tw / 2;
        const cy = r.y - th - 3 * px;
        ctx.fillStyle = "rgba(14,16,19,0.88)";
        ctx.fillRect(cx, cy, tw, th);
        ctx.fillStyle = theme.gate;
        ctx.fillText(chip, cx + 4 * px, cy + 2 * px);
      }
      ctx.restore();
    }
    if (keepBox && keepId != null) {
      ctx.save();
      ctx.setLineDash([6 * px, 4 * px]);
      ctx.strokeStyle = theme.success;
      ctx.lineWidth = 2.5 * px;
      ctx.strokeRect(keepBox.x, keepBox.y, keepBox.w, keepBox.h);
      ctx.setLineDash([]);
      const keepText = gtId != null ? `keep #${keepId}  (GT ${gtId})` : `keep #${keepId}`;
      ctx.font = theme.labelFont.replace("11px", `${11 * px}px`);
      ctx.textBaseline = "top";
      const ktw = ctx.measureText(keepText).width + 8 * px;
      const kth = 15 * px;
      const ky = keepBox.y + keepBox.h + 2 * px;
      ctx.fillStyle = theme.success;
      ctx.fillRect(keepBox.x, ky, ktw, kth);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(keepText, keepBox.x + 4 * px, ky + 2 * px);
      ctx.restore();
    }
    if (ghost) {
      const color = stableColor ?? theme.error;
      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.setLineDash([8 * px, 5 * px]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3 * px;
      ctx.strokeRect(ghost.x, ghost.y, ghost.w, ghost.h);
      ctx.setLineDash([]);
      const gl = ghostLabel ?? `left here #${ghost.id}`;
      ctx.font = theme.labelFont.replace("11px", `${11 * px}px`);
      ctx.textBaseline = "top";
      const tw = ctx.measureText(gl).width + 8 * px;
      const th = 16 * px;
      ctx.fillStyle = color;
      ctx.fillRect(ghost.x, Math.max(0, ghost.y - th), tw, th);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(gl, ghost.x + 4 * px, Math.max(0, ghost.y - th) + 2 * px);
      ctx.restore();
    }
    if (idBreak && idBreak.phase !== "here" && idBreak.leave) {
      const leave = idBreak.leave;
      ctx.save();
      ctx.font = theme.labelFont.replace("11px", `${11 * px}px`);
      ctx.textBaseline = "top";
      const th = 16 * px;
      const x0 = leave.x + leave.w / 2;
      const y0 = leave.y + leave.h / 2;
      if (idBreak.phase === "gone") {
        ctx.setLineDash([8 * px, 5 * px]);
        ctx.strokeStyle = theme.error;
        ctx.lineWidth = 2.5 * px;
        ctx.strokeRect(leave.x, leave.y, leave.w, leave.h);
        ctx.setLineDash([]);
        const leftText = `LEFT f${idBreak.leaveFrame}  dropped #${idBreak.fromId}`;
        const ltw = ctx.measureText(leftText).width + 8 * px;
        ctx.fillStyle = theme.error;
        ctx.fillRect(leave.x, Math.max(0, leave.y - th), ltw, th);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(leftText, leave.x + 4 * px, Math.max(0, leave.y - th) + 2 * px);
      } else if (idBreak.back) {
        const back = idBreak.back;
        const x1 = back.x + back.w / 2;
        const y1 = back.y + back.h / 2;
        ctx.fillStyle = theme.error;
        ctx.beginPath();
        ctx.arc(x0, y0, 4 * px, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = theme.error;
        ctx.lineWidth = 1.5 * px;
        ctx.setLineDash([5 * px, 4 * px]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.setLineDash([]);
        const mid = `same person · should still be #${idBreak.fromId}`;
        const mw = ctx.measureText(mid).width + 8 * px;
        const mx = Math.min(
          Math.max(crop.x + 4 * px, back.x + back.w + 6 * px),
          crop.x + crop.w - mw - 4 * px,
        );
        const my = Math.max(crop.y + 4 * px, back.y);
        ctx.fillStyle = "rgba(14,16,19,0.88)";
        ctx.fillRect(mx, my, mw, th);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(mid, mx + 4 * px, my + 2 * px);
      }
      ctx.restore();
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
    ghost,
    ghostLabel,
    overlapIds,
    overlaps,
    keepId,
    gtId,
    keepBox,
    verdict,
    hostSize,
    chrome,
    cropCrowd,
    idBreak,
  ]);

  const counts = gateCounts(clip.events, frame);
  const canvas = (
    <canvas
      ref={ref}
      className={chrome ? undefined : "block max-w-full max-h-full"}
      style={onPick ? { cursor: "pointer" } : undefined}
      onClick={
        onPick
          ? (e) => {
              const cv = ref.current;
              if (!cv) return;
              const crop = cropRef.current;
              const rect = cv.getBoundingClientRect();
              const x = crop.x + ((e.clientX - rect.left) / (rect.width || 1)) * crop.w;
              const y = crop.y + ((e.clientY - rect.top) / (rect.height || 1)) * crop.h;
              const hit = pickBoxAt(clip.byFrame.get(frame) ?? [], x, y);
              onPick(hit ? hit.id : null);
            }
          : undefined
      }
    />
  );
  if (!chrome) {
    return (
      <div
        ref={hostRef}
        className={className ?? "h-full w-full flex items-center justify-center"}
        data-testid={`panel-${clip.tracker}`}
      >
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
