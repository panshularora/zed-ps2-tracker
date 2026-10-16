// Flat human figure for boxes-only clips and the synthetic scenes: head, shoulders and body as one
// filled shape (no 3D), two-frame walk cycle, muted clothing colour derived from the ID colour.
// Hidden people are drawn at 30% with a dashed outline, never removed.
export interface FigureOpts {
  /** centre x and the y of the feet, in canvas units */
  cx: number;
  footY: number;
  /** full height (head top to feet) */
  h: number;
  color: string;
  /** 0 or 1: the two walk frames */
  step: 0 | 1;
  hidden?: boolean;
  /** width/height ratio of the body (front-facing default 0.36) */
  aspect?: number;
}

/** Mute a hex colour toward the theme grey so clothing reads as clothing, not a highlighter. */
export function muted(hex: string, k = 0.62): string {
  const n = parseInt(hex.replace("#", ""), 16);
  const g = [0x4a, 0x52, 0x5c];
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v, i) => Math.round(v * k + g[i] * (1 - k)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export const walkStep = (t: number, fps = 4): 0 | 1 => (Math.floor(t * fps) % 2 === 0 ? 0 : 1);

/** Geometry (for tests and chip placement): head top y and body width. */
export function figureGeom(o: Pick<FigureOpts, "cx" | "footY" | "h" | "aspect">) {
  const w = o.h * (o.aspect ?? 0.36);
  const headR = Math.min(o.h * 0.1, w * 0.32);
  const top = o.footY - o.h;
  return { w, headR, top, headCy: top + headR, shoulderY: top + headR * 2.25, hipY: top + o.h * 0.58 };
}

/** Trace the one-piece silhouette path (head + shoulders + torso + two legs). */
export function figurePath(ctx: CanvasRenderingContext2D, o: FigureOpts) {
  const g = figureGeom(o);
  const { cx } = o;
  const sw = g.w / 2, hw = g.w * 0.36, r = g.headR;
  const spread = o.step === 0 ? g.w * 0.08 : g.w * 0.26;
  const leg = g.w * 0.17;
  ctx.beginPath();
  ctx.arc(cx, g.headCy, r, 0, Math.PI * 2);
  ctx.moveTo(cx - sw + r * 0.6, g.shoulderY);
  ctx.lineTo(cx + sw - r * 0.6, g.shoulderY);
  ctx.quadraticCurveTo(cx + sw, g.shoulderY, cx + sw, g.shoulderY + r * 0.7);
  ctx.lineTo(cx + hw, g.hipY);
  ctx.lineTo(cx + spread + leg, o.footY);
  ctx.lineTo(cx + spread - leg * 0.2, o.footY);
  ctx.lineTo(cx, g.hipY + (o.footY - g.hipY) * 0.18);
  ctx.lineTo(cx - spread + leg * 0.2, o.footY);
  ctx.lineTo(cx - spread - leg, o.footY);
  ctx.lineTo(cx - hw, g.hipY);
  ctx.lineTo(cx - sw, g.shoulderY + r * 0.7);
  ctx.quadraticCurveTo(cx - sw, g.shoulderY, cx - sw + r * 0.6, g.shoulderY);
  ctx.closePath();
}

export function drawFigure(ctx: CanvasRenderingContext2D, o: FigureOpts) {
  const g = figureGeom(o);
  ctx.save();
  // soft ground shadow (raised camera)
  ctx.globalAlpha *= o.hidden ? 0.3 : 1;
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.beginPath();
  ctx.ellipse(o.cx, o.footY, g.w * 0.55, Math.max(1.5, g.w * 0.14), 0, 0, Math.PI * 2);
  ctx.fill();
  figurePath(ctx, o);
  ctx.fillStyle = muted(o.color);
  ctx.fill();
  ctx.lineWidth = Math.max(1, o.h / 90);
  ctx.strokeStyle = o.color;
  if (o.hidden) ctx.setLineDash([Math.max(3, o.h / 22), Math.max(2, o.h / 30)]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

/** Solid ID chip centred above a point: filled with the colour, dark text, 13px semibold. */
export function drawChip(ctx: CanvasRenderingContext2D, cx: number, bottom: number, text: string, fill: string, size = 13) {
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.font = `600 ${size}px Inter, sans-serif`;
  const w = ctx.measureText(text).width + 12, h = size + 8;
  const x = cx - w / 2, y = bottom - h;
  ctx.fillStyle = fill;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, 4); ctx.fill();
  ctx.fillStyle = "#0E1013"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, cx, y + h / 2 + 0.5);
  ctx.restore();
  return { x, y, w, h };
}
