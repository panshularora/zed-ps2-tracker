// Canvas painting for the guided replay (video px coordinates, view rect = zoom).
import type { Box, RunKey } from "../lib/scenarios";
import { neutralColor } from "../lib/scenarios";
import { chipText, lastSeen, trailOf, type ReplayData } from "./model";

export interface Paint {
  run: RunKey;
  f: number | null;
  view: [number, number, number, number];
  spot: Set<number>;
  dim: number; // 0..1 how much to darken non-spotlit frame (1 => 45 % brightness)
  gatePulse: number;
}

const GATE = "#F5A524"; // theme gate amber, not red

export function paint(ctx: CanvasRenderingContext2D, cw: number, ch: number, d: ReplayData, src: CanvasImageSource | null, p: Paint) {
  const [vx, vy, vw, vh] = p.view;
  const s = cw / vw; // css px per video px
  ctx.save();
  ctx.fillStyle = "#0E1013";
  ctx.fillRect(0, 0, cw, ch);
  ctx.setTransform(ctx.getTransform().scale(s, ch / vh).translate(-vx, -vy));
  if (src) ctx.drawImage(src, 0, 0, d.width, d.height);
  const boxes: Box[] = p.f != null ? d.runs[p.run].frames[String(p.f)] ?? [] : [];
  const spotBoxes = boxes.filter((b) => p.spot.has(b[0]));
  // spotlit people hidden right now: dashed "last seen" ghost, so a disappearance is visible
  const ghosts: Box[] = [];
  if (p.f != null) {
    const here = new Set(boxes.map((b) => b[0]));
    for (const id of p.spot) {
      if (here.has(id)) continue;
      // a number that was just replaced is not "hidden": its new number is drawn instead
      if (d.events[p.run].some((e) => e.t === "renumber" && e.from === id && e.f <= p.f!)) continue;
      const ls = lastSeen(d.runs[p.run].frames, id, p.f, d.fps);
      // only if no later number took over this spot (avoid ghosting an id that was renumbered)
      if (ls && !spotBoxes.some((b) => overlap(b, ls.box) > 0.3)) ghosts.push(ls.box);
    }
  }
  if (p.dim > 0) {
    // darken everything except spotlit boxes to ~45 % brightness
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, d.width, d.height);
    for (const [, x, y, w, h] of [...spotBoxes, ...ghosts]) ctx.rect(x + w, y, -w, h); // reverse winding = hole
    ctx.fillStyle = `rgba(0,0,0,${0.55 * p.dim})`;
    ctx.fill("nonzero");
    ctx.restore();
  }
  const px = 1 / s; // one css px in video px
  // gate line
  if (d.gate) {
    const { a, b } = d.gate;
    ctx.strokeStyle = GATE;
    ctx.lineWidth = (3 + 3 * p.gatePulse) * px;
    ctx.setLineDash([12 * px, 6 * px]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.setLineDash([]);
    // IN / OUT side labels (eval/gate_events.py: left normal = "in")
    const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
    const nx = -dy / L, ny = dx / L;
    const mx = a[0] + dx * 0.08, my = a[1] + dy * 0.08;
    ctx.font = `700 ${13 * px}px Inter, sans-serif`;
    ctx.fillStyle = GATE;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const off = (Math.abs(nx) > 0.5 ? 46 : 18) * px; // vertical line: labels sit left / right
    const vmy = Math.max(my, vy + 40 * px);
    ctx.fillText("IN side", Math.max(40 * px, mx + nx * off), vmy + ny * off);
    ctx.fillText("OUT side", Math.max(40 * px, mx - nx * off), vmy - ny * off);
  }
  if (p.f != null) {
    // others: 1 px, 35 % opacity, no labels
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1 * px;
    for (const [id, x, y, w, h] of boxes) {
      if (p.spot.has(id)) continue;
      ctx.strokeStyle = neutralColor(id);
      ctx.strokeRect(x, y, w, h);
    }
    ctx.globalAlpha = 1;
    for (const [id, x, y, w, h] of ghosts) {
      const c = neutralColor(id);
      ctx.strokeStyle = c;
      ctx.lineWidth = 2 * px;
      ctx.setLineDash([8 * px, 6 * px]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      const text = `Person ${id} · hidden`;
      ctx.font = `700 ${15 * px}px Inter, sans-serif`;
      const tw = ctx.measureText(text).width + 18 * px, chH = 28 * px;
      const cx = Math.min(Math.max(vx, x + w / 2 - tw / 2), vx + vw - tw);
      const cy = Math.max(vy + 2 * px, y - chH - 6 * px);
      ctx.fillStyle = "rgba(14,16,19,.9)";
      ctx.beginPath();
      ctx.roundRect(cx, cy, tw, chH, 4 * px);
      ctx.fill();
      ctx.strokeStyle = c;
      ctx.lineWidth = 1.5 * px;
      ctx.stroke();
      ctx.fillStyle = c;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, cx + tw / 2, cy + chH / 2 + px);
    }
    // spotlit: 2 s trail, 3 px box, 28 px chip above head
    for (const [id, x, y, w, h] of spotBoxes) {
      const c = neutralColor(id);
      const tr = trailOf(d.runs[p.run].frames, id, p.f, d.fps);
      ctx.strokeStyle = c;
      ctx.lineWidth = 3 * px;
      ctx.lineCap = "round";
      for (let i = 1; i < tr.length; i++) {
        ctx.globalAlpha = tr[i][2];
        ctx.beginPath();
        ctx.moveTo(tr[i - 1][0], tr[i - 1][1]);
        ctx.lineTo(tr[i][0], tr[i][1]);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.lineWidth = 3 * px;
      ctx.strokeRect(x, y, w, h);
      const chip = chipText(id, p.f, d.events[p.run], d.fps);
      const k = 1 + 0.25 * chip.pulse;
      const chH = 28 * px * k;
      ctx.font = `700 ${15 * px * k}px Inter, sans-serif`;
      const tw = ctx.measureText(chip.text).width + 18 * px * k;
      const cx = Math.min(Math.max(vx, x + w / 2 - tw / 2), vx + vw - tw);
      const cy = Math.max(vy + 2 * px, y - chH - 6 * px);
      ctx.fillStyle = c;
      ctx.beginPath();
      ctx.roundRect(cx, cy, tw, chH, 4 * px);
      ctx.fill();
      ctx.fillStyle = "#0E1013";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(chip.text, cx + tw / 2, cy + chH / 2 + px);
    }
  }
  ctx.restore();
}

/** Small still: crop `focus` (padded) from src and draw `run`'s spotlit boxes with chips. */
export function paintStill(ctx: CanvasRenderingContext2D, cw: number, ch: number, d: ReplayData, src: CanvasImageSource | null, run: RunKey, f: number, focus: number[], spot: Set<number>) {
  const pad = 0.35;
  const w0 = focus[2] * (1 + 2 * pad), h0 = focus[3] * (1 + 2 * pad);
  const ar = cw / ch;
  const w = Math.max(w0, h0 * ar), h = w / ar;
  const x = Math.min(Math.max(0, focus[0] + focus[2] / 2 - w / 2), Math.max(0, d.width - w));
  const y = Math.min(Math.max(0, focus[1] + focus[3] / 2 - h / 2), Math.max(0, d.height - h));
  paint(ctx, cw, ch, d, src, { run, f, view: [x, y, Math.min(w, d.width), Math.min(h, d.height)], spot, dim: 0.6, gatePulse: 0 });
}

function overlap(a: Box, b: Box): number {
  const iw = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const ih = Math.max(0, Math.min(a[2] + a[4], b[2] + b[4]) - Math.max(a[2], b[2]));
  const i = iw * ih, u = a[3] * a[4] + b[3] * b[4] - i;
  return u > 0 ? i / u : 0;
}
