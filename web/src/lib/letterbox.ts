// object-fit: contain for canvas players. The frame keeps its aspect ratio inside the stage, with
// bars (letterbox / pillarbox) instead of cropping; boxes, chips and the gate are drawn in the same
// content rectangle so they stay on the people and the line.

export interface ContentRect { x: number; y: number; w: number; h: number; /** stage px per source px */ s: number }

/** Where a vw x vh source sits inside a cw x ch stage under object-fit: contain (centred). */
export function letterbox(cw: number, ch: number, vw: number, vh: number): ContentRect {
  if (!(cw > 0 && ch > 0 && vw > 0 && vh > 0)) return { x: 0, y: 0, w: Math.max(0, cw), h: Math.max(0, ch), s: 1 };
  const s = Math.min(cw / vw, ch / vh);
  const w = vw * s, h = vh * s;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h, s };
}

/** Stage point (css px) -> source point; null when the point is on a bar. */
export function stageToSource(r: ContentRect, px: number, py: number, vx = 0, vy = 0): [number, number] | null {
  if (px < r.x || py < r.y || px > r.x + r.w || py > r.y + r.h) return null;
  return [vx + (px - r.x) / r.s, vy + (py - r.y) / r.s];
}

/** Source point -> stage point (css px). */
export const sourceToStage = (r: ContentRect, x: number, y: number, vx = 0, vy = 0): [number, number] => [r.x + (x - vx) * r.s, r.y + (y - vy) * r.s];

/** Stage aspect for a clip: its own when landscape (16:9 or wider), else a 16:9 stage with bars, so a
 * portrait clip neither crops nor turns the page into a very tall stage. */
export const stageAspect = (w: number, h: number): number => (w > 0 && h > 0 ? Math.max(w / h, 16 / 9) : 16 / 9);

/** Clip and transform a 2D context so drawing in source coordinates (view = source rect) lands in r. */
export function applyContent(ctx: CanvasRenderingContext2D, r: ContentRect, vx = 0, vy = 0): void {
  ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
  ctx.setTransform(ctx.getTransform().translate(r.x, r.y).scale(r.s, r.s).translate(-vx, -vy));
}
