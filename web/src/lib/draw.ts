// Canvas overlay for the demo stage. Coordinates are video pixels; CSS scales the canvas.
import type { JobResult, Styling } from "../api";
import { idColor, theme } from "../theme";
import { trail, type Indexed } from "./playback";

export interface DrawState {
  k: number | null; // tracked frame on screen
  rings: Map<number, number>;
  pulse: number;
  styling: Styling;
  alertIds: Set<number>; // tracks with a tailgate / hold at this frame (consented clips only)
}

export function gateNormal(a: [number, number], b: [number, number]): [number, number] {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  return [-dy / len, dx / len]; // eval/gate_events.py: side > 0 = "in"
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function drawOverlay(ctx: CanvasRenderingContext2D, r: JobResult, idx: Indexed, s: DrawState): void {
  const { width: W, height: H } = r.video;
  ctx.clearRect(0, 0, W, H);
  const scale = Math.max(W, H) / 1280; // line widths tuned at 1280 px
  const { a, b } = r.gate;
  ctx.save();
  ctx.strokeStyle = theme.gate;
  ctx.globalAlpha = 0.85 + 0.15 * s.pulse;
  ctx.lineWidth = (theme.gateWidth + 3 * s.pulse) * scale;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
  const [nx, ny] = gateNormal(a, b);
  ctx.globalAlpha = 1;
  ctx.font = `600 ${14 * scale}px Inter, sans-serif`;
  ctx.fillStyle = theme.gate;
  const lx = a[0] + (b[0] - a[0]) * 0.96 - 30 * scale;
  const ly = a[1] + (b[1] - a[1]) * 0.96;
  ctx.fillText("IN", lx + nx * 24 * scale, ly + ny * 24 * scale + 5 * scale);
  ctx.fillText("OUT", lx - nx * 24 * scale, ly - ny * 24 * scale + 5 * scale);
  ctx.restore();
  if (s.k == null) return;
  // trails under the boxes: 1.5 px, ID colour, fading over the last 30 tracked frames
  ctx.save();
  ctx.lineWidth = theme.trailWidth * scale;
  ctx.lineCap = "round";
  for (const t of idx.tracks) {
    const pts = trail(t.rows, s.k);
    if (pts.length < 2) continue;
    ctx.strokeStyle = idColor(t.id);
    for (let i = 1; i < pts.length; i++) {
      ctx.globalAlpha = pts[i].alpha;
      ctx.beginPath();
      ctx.moveTo(pts[i - 1].x, pts[i - 1].y);
      ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();
    }
  }
  ctx.restore();
  ctx.save();
  ctx.font = `600 ${12 * scale}px 'JetBrains Mono', monospace`;
  for (const { id, row } of idx.byFrame.get(s.k) ?? []) {
    const [, x, y, w, h, , vis] = row;
    const color = s.styling === "alert" && s.alertIds.has(id) ? theme.error : idColor(id);
    ctx.strokeStyle = color;
    ctx.lineWidth = theme.boxWidth * scale;
    ctx.setLineDash(vis < 0.5 ? [6 * scale, 4 * scale] : []);
    ctx.globalAlpha = vis < 0.5 ? 0.7 : 1;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    const label = String(id);
    const tw = ctx.measureText(label).width + 8 * scale;
    const th = 17 * scale;
    ctx.fillStyle = color;
    ctx.fillRect(x - scale, y - th, tw, th);
    ctx.fillStyle = theme.labelText;
    ctx.fillText(label, x + 3 * scale, y - 4 * scale);
    const ring = s.rings.get(id);
    if (ring !== undefined && ring > 0) {
      const pad = (6 + 10 * (1 - ring)) * scale;
      ctx.strokeStyle = idColor(id);
      ctx.globalAlpha = ring;
      ctx.lineWidth = 3 * scale;
      roundRect(ctx, x - pad, y - pad, w + 2 * pad, h + 2 * pad, 4 * scale);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }
  ctx.restore();
}
