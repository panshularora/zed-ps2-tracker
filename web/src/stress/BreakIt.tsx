// Break it yourself: build a scene (pillars, a group, a drawn path, a hidden person), run BOTH real
// trackers on the deterministic simulator (POST /api/break/run), see who keeps the number.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { AMBER } from "./model";
import { verdict, personHeight, type SceneData } from "./scene";
import { ScenePlayer } from "./ScenePlayer";
import { DEFAULT_SPEC, HARDER_STEPS, fromQuery, harderDone, harderText, simplifyPath, toQuery, type BreakSpec, type HarderStep } from "./breakit";

const W = 960, H = 540, GATE_Y = 300, PILLAR_W = 46, PILLAR_H = 150;
type Tool = "path" | "pillar" | "group" | "hide";

function personAnchors(spec: BreakSpec): { pid: number; x: number; y: number }[] {
  const out = [{ pid: 1, x: spec.path[0][0], y: Math.max(60, spec.path[0][1]) }];
  spec.groups.forEach(([gx, gy], i) => { for (let k = 0; k < 3; k++) out.push({ pid: 2 + i * 3 + k, x: gx + (k - 1) * 26, y: gy }); });
  return out;
}

function drawEditor(c: HTMLCanvasElement, spec: BreakSpec, draft: [number, number][] | null, ghost: { tool: Tool; x: number; y: number } | null) {
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "#161A1F";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = AMBER;
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 6]);
  ctx.beginPath(); ctx.moveTo(0, GATE_Y); ctx.lineTo(W, GATE_Y); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#8A929C";
  ctx.font = "13px Inter, system-ui, sans-serif";
  ctx.fillText("gate line", 8, GATE_Y - 8);
  const path = draft ?? spec.path;
  ctx.strokeStyle = "#56B4E9";
  ctx.lineWidth = 2;
  ctx.beginPath();
  path.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  const pillar = (x: number, y: number, a = 1) => { ctx.globalAlpha = a; ctx.fillStyle = "#262B33"; ctx.fillRect(x - PILLAR_W / 2, y - PILLAR_H, PILLAR_W, PILLAR_H); ctx.strokeStyle = "#3A414C"; ctx.strokeRect(x - PILLAR_W / 2 + 0.5, y - PILLAR_H + 0.5, PILLAR_W - 1, PILLAR_H - 1); ctx.globalAlpha = 1; };
  const fig = (x: number, y: number, hidden: boolean, a = 1) => {
    const h = personHeight(y);
    ctx.globalAlpha = a;
    ctx.strokeStyle = "#C9CED6"; ctx.fillStyle = "#C9CED6"; ctx.lineWidth = 2;
    if (hidden) ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.arc(x, y - h + h * 0.11, h * 0.11, 0, Math.PI * 2); if (hidden) ctx.stroke(); else ctx.fill();
    ctx.beginPath(); ctx.roundRect(x - h * 0.13, y - h * 0.76, h * 0.26, h * 0.34, 3); if (hidden) ctx.stroke(); else ctx.fill();
    ctx.beginPath(); ctx.moveTo(x - 4, y - h * 0.42); ctx.lineTo(x - 6, y); ctx.moveTo(x + 4, y - h * 0.42); ctx.lineTo(x + 6, y); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  };
  spec.pillars.forEach(([x, y]) => pillar(x, y));
  for (const a of personAnchors(spec)) {
    fig(a.x, a.y, spec.hide_pid === a.pid);
    if (spec.hide_pid === a.pid) { ctx.fillStyle = AMBER; ctx.fillText(`hidden ${spec.hide_s} s at the gate`, a.x + 14, a.y - 8); }
  }
  if (ghost?.tool === "pillar") pillar(ghost.x, ghost.y, 0.5);
  if (ghost?.tool === "group") for (let k = 0; k < 3; k++) fig(ghost.x + (k - 1) * 26, ghost.y, false, 0.5);
}

export default function BreakIt() {
  const initial = useMemo(() => (typeof window !== "undefined" ? fromQuery(window.location.search) : null), []);
  const [spec, setSpec] = useState<BreakSpec>(initial?.spec ?? DEFAULT_SPEC);
  const [startFrame, setStartFrame] = useState(initial?.frame ?? 1);
  const [tool, setTool] = useState<Tool>("path");
  const [data, setData] = useState<SceneData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [harder, setHarder] = useState<{ steps: HarderStep[]; running: boolean } | null>(null);
  const frame = useRef(startFrame);
  const editor = useRef<HTMLCanvasElement>(null);
  const draft = useRef<[number, number][] | null>(null);
  const [ghost, setGhost] = useState<{ tool: Tool; x: number; y: number } | null>(null);
  const [drag, setDrag] = useState<Tool | null>(null);
  const [, force] = useState(0);

  const toScene = (cx: number, cy: number): [number, number] | null => {
    const r = editor.current?.getBoundingClientRect();
    if (!r || cx < r.left || cx > r.right || cy < r.top || cy > r.bottom) return null;
    return [((cx - r.left) / r.width) * W, ((cy - r.top) / r.height) * H];
  };

  useEffect(() => { if (editor.current) drawEditor(editor.current, spec, draft.current, ghost); });

  // run both trackers whenever the scene changes
  useEffect(() => {
    let live = true;
    setBusy(true); setErr(null);
    const t = window.setTimeout(() => {
      api.breakRun(spec).then((d) => live && setData(d)).catch((e: Error) => live && setErr(e.message)).finally(() => live && setBusy(false));
    }, 150);
    return () => { live = false; window.clearTimeout(t); };
  }, [spec]);

  const share = useCallback((f: number) => {
    frame.current = f;
    const q = toQuery({ spec, frame: f });
    if (typeof window !== "undefined") window.history.replaceState(null, "", `${window.location.pathname}?${q}${window.location.hash}`);
  }, [spec]);
  useEffect(() => { share(frame.current); }, [share]);

  // tray drag with pointer events
  useEffect(() => {
    if (!drag) return;
    const move = (e: PointerEvent) => { const p = toScene(e.clientX, e.clientY); setGhost(p ? { tool: drag, x: p[0], y: p[1] } : null); };
    const up = (e: PointerEvent) => { const p = toScene(e.clientX, e.clientY); if (p) place(drag, p); setDrag(null); setGhost(null); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  });

  function place(t: Tool, [x, y]: [number, number]) {
    const fy = Math.min(H - 10, Math.max(PILLAR_H + 10, y));
    if (t === "pillar") setSpec((s) => ({ ...s, pillars: [...s.pillars, [Math.round(x), Math.round(fy)]].slice(-8) as [number, number][] }));
    if (t === "group") setSpec((s) => ({ ...s, groups: [...s.groups, [Math.round(x), Math.round(y)]].slice(-4) as [number, number][] }));
    setStartFrame(1);
  }
  function hidePid(pid: number) { setSpec((s) => ({ ...s, hide_pid: pid })); setStartFrame(1); setTool("path"); }

  function down(e: React.PointerEvent<HTMLCanvasElement>) {
    const p = toScene(e.clientX, e.clientY);
    if (!p) return;
    if (tool === "pillar" || tool === "group") { place(tool, p); return; }
    if (tool === "hide") {
      const hit = personAnchors(spec).find((a) => { const h = personHeight(a.y); return Math.abs(p[0] - a.x) < h * 0.35 && p[1] > a.y - h && p[1] < a.y + 6; });
      if (hit) hidePid(hit.pid);
      return;
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    draft.current = [p];
    force((x) => x + 1);
  }
  function moveDraw(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!draft.current) return;
    const r = e.currentTarget.getBoundingClientRect();
    draft.current.push([((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H]);
    force((x) => x + 1);
  }
  function upDraw() {
    const raw = draft.current;
    draft.current = null;
    if (raw && raw.length > 1) {
      const path = simplifyPath(raw);
      if (path.length >= 2) { setSpec((s) => ({ ...s, path })); setStartFrame(1); }
    }
    force((x) => x + 1);
  }

  async function makeHarder() {
    const pid = spec.hide_pid ?? 1;
    const steps: HarderStep[] = [];
    setHarder({ steps, running: true });
    for (const h of HARDER_STEPS) {
      try {
        const d = await api.breakRun({ ...spec, hide_s: h, hide_pid: pid });
        steps.push({ hide_s: h, v: { bytetrack: verdict(d, "bytetrack", pid), ours: verdict(d, "ours", pid) } });
        setHarder({ steps: [...steps], running: true });
        if (harderDone(steps)) break;
      } catch (e) { setErr((e as Error).message); break; }
    }
    setHarder({ steps: [...steps], running: false });
  }

  const v = data && spec.hide_pid != null ? { bt: verdict(data, "bytetrack", spec.hide_pid), ours: verdict(data, "ours", spec.hide_pid) } : null;
  const link = typeof window !== "undefined" ? `${window.location.origin}${window.location.pathname}?${toQuery({ spec, frame: frame.current })}` : "";
  return (
    <section className="section" data-testid="break-it">
      <h2>Break it yourself</h2>
      <p className="small muted" style={{ maxWidth: 820 }}>
        Drag a pillar or a group onto the floor, draw the blue person's path with your pointer, pick who gets hidden. Both real trackers (ByteTrack and ours) run on the same
        deterministic simulator; the same scene always gives the same result, so the link below reproduces it exactly.
      </p>
      <div className="break-tray" role="toolbar" aria-label="Tray">
        {([["pillar", "Pillar"], ["group", "Group of people"], ["hide", "Hide a person (click one)"]] as [Tool, string][]).map(([t, label]) => (
          <button key={t} className={`btn${tool === t ? " active" : ""}`} data-testid={`tray-${t}`}
            onPointerDown={(e) => { if (t !== "hide") { e.preventDefault(); setDrag(t); } }}
            onClick={() => setTool((x) => (x === t ? "path" : t))}>{label}</button>
        ))}
        <button className={`btn${tool === "path" ? " active" : ""}`} onClick={() => setTool("path")} data-testid="tray-path">Draw path</button>
        <button className="btn" onClick={() => { setSpec(DEFAULT_SPEC); setStartFrame(1); }}>Reset scene</button>
        <label className="small muted">seed <input type="number" value={spec.seed} onChange={(e) => setSpec((s) => ({ ...s, seed: Math.trunc(Number(e.target.value) || 0) }))} style={{ width: 72 }} /></label>
        <label className="small muted">hidden for <input type="number" step={0.5} min={0} max={8} value={spec.hide_s} onChange={(e) => setSpec((s) => ({ ...s, hide_s: Math.min(8, Math.max(0, Number(e.target.value) || 0)) }))} style={{ width: 56 }} /> s</label>
        <label className="small muted">passers-by <input type="number" min={0} max={12} value={spec.crowd} onChange={(e) => setSpec((s) => ({ ...s, crowd: Math.min(12, Math.max(0, Math.trunc(Number(e.target.value) || 0))) }))} style={{ width: 48 }} /></label>
        <label className="small muted">
          <input type="checkbox" checked={spec.config === "mem6"} onChange={(e) => setSpec((s) => ({ ...s, config: e.target.checked ? "mem6" : "default" }))} /> ours with longer memory (opt-in)
        </label>
      </div>
      <div className="break-editor">
        <canvas ref={editor} width={W} height={H} data-testid="break-editor" onPointerDown={down} onPointerMove={moveDraw} onPointerUp={upDraw} />
      </div>
      <p className="small muted mono" data-testid="break-settings">
        seed {spec.seed} · hidden person {spec.hide_pid ?? "none"} for {spec.hide_s} s · {spec.pillars.length} pillars · {spec.groups.length} groups · {spec.crowd} passers-by · ours = {spec.config === "mem6" ? "longer memory (opt-in)" : "default"}
        {busy ? " · running both trackers…" : ""}
      </p>
      {err && <p className="muted" data-testid="break-error">Break it yourself runs both trackers on the local service, which is not reachable: {err}</p>}
      {data && (
        <>
          <ScenePlayer data={data} startFrame={startFrame} focus={spec.hide_pid} showVerdict onPersonClick={tool === "hide" ? hidePid : undefined} onFrame={share} />
          <p className="break-counts" data-testid="break-counts">
            End of scene · true crossings {data.panes.ours.true_count} · ByteTrack counted {data.panes.bytetrack.count} · ours counted {data.panes.ours.count}
            {v ? ` · hidden person: ByteTrack ${v.bt?.text ?? "n/a"}, ours ${v.ours?.text ?? "n/a"}` : ""}
          </p>
        </>
      )}
      <div className="scene-controls">
        <button className="btn" onClick={makeHarder} disabled={!!harder?.running || spec.hide_pid == null} data-testid="make-harder">Make it harder</button>
        <button className="btn" onClick={() => void navigator.clipboard?.writeText(link)} data-testid="copy-link">Copy link to this scene</button>
        <span className="small muted mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 560 }} data-testid="share-link">{link}</span>
      </div>
      {harder && (
        <div data-testid="harder-result" style={{ marginTop: 8 }}>
          <p className="small mono muted">
            {harder.steps.map((s) => `${s.hide_s} s: ByteTrack ${s.v.bytetrack?.text ?? "n/a"} / ours ${s.v.ours?.text ?? "n/a"}`).join(" · ")}
            {harder.running ? " · stepping up…" : ""}
          </p>
          {!harder.running && <p style={{ fontSize: 16, fontWeight: 600 }} data-testid="harder-text">{harderText(harder.steps)}</p>}
        </div>
      )}
    </section>
  );
}
