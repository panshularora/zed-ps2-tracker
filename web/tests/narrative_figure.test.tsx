import { readFileSync } from "node:fs";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { headlineOf, outcome, pickStory, storyOf, tally, verdictText, type HideMoment } from "../src/compare/narrative";
import MomentCompare, { cropFor } from "../src/compare/MomentCompare";
import { drawFigure, figureGeom, muted, walkStep } from "../src/lib/figure";
import { quietView } from "../src/identity/PersonTimeline";
import type { SimFrame } from "../src/lib/gateSim";

const lines = readFileSync("public/offline/gate-sim.ndjson", "utf8").trim().split("\n").map((l) => JSON.parse(l));
const frames = lines.filter((l) => l.type === "frame") as SimFrame[];

const pf = (ids: [number, number][]) => ({ ids, extra: [], events: [], counted: 0, id_changes: 0 });
const fr = (f: number, vis: number, b: number | null, o: number | null): SimFrame => ({
  type: "frame", f, dets: [], true_count: 0,
  people: [[1, 100, 100, vis], [2, 104, 120, 1]],
  bytetrack: pf([...(vis && b != null ? [[1, b] as [number, number]] : []), [2, 2]]),
  ours: pf([...(vis && o != null ? [[1, o] as [number, number]] : []), [2, 2]]),
});

describe("narrative is computed from the numbers, so it cannot contradict itself", () => {
  const m: HideMoment = { person: 1, occluder: 2, lastIdx: 0, backIdx: 2 };
  it("ByteTrack changed, ours kept", () => {
    const fs = [fr(1, 1, 4, 4), fr(2, 0, null, null), fr(3, 1, 10, 4)];
    const s = storyOf(fs, m);
    expect(s.out.bytetrack).toEqual({ before: 4, after: 10, verdict: "changed" });
    expect(s.out.ours).toEqual({ before: 4, after: 4, verdict: "kept" });
    expect(s.headline).toBe("Person 1 walks behind person 2. ByteTrack: comes back as #10, a new number (was #4). Ours: comes back as #4 (kept).");
    expect(s.verdicts).toEqual({ bytetrack: "Changed #4 → #10", ours: "Kept #4 ✓" });
  });
  it("both changed to the same number → one 'Both trackers' clause, no 'same number' claim", () => {
    const fs = [fr(1, 1, 2, 2), fr(2, 0, null, null), fr(3, 1, 10, 10)];
    const s = storyOf(fs, m);
    expect(s.headline).toBe("Person 1 walks behind person 2. Both trackers: comes back as #10, a new number (was #2).");
    expect(s.headline).not.toMatch(/kept|same number/);
  });
  it("on every real hide in the recorded run, headline words match each pane's verdict", () => {
    const { all } = pickStory(frames);
    expect(all.length).toBeGreaterThan(0);
    for (const s of all) {
      for (const p of ["bytetrack", "ours"] as const) {
        const o = s.out[p];
        expect(o).toEqual(outcome(frames, s.m, p));
        expect(verdictText(o)).toBe(o.verdict === "kept" ? `Kept #${o.before} ✓` : `Changed #${o.before} → #${o.after}`);
        if (o.verdict === "kept") expect(s.headline).toContain(`as #${o.before} (kept)`);
        else expect(s.headline).toContain(`as #${o.after}, a new number (was #${o.before})`);
      }
      expect(s.headline).toBe(headlineOf(s.m, s.out));
    }
  });
  it("picks the moment where ByteTrack changed and ours kept when there is one; tally counts all", () => {
    const p = pickStory(frames);
    if (p.kind === "ours-better") {
      expect(p.story!.out.bytetrack.verdict).toBe("changed");
      expect(p.story!.out.ours.verdict).toBe("kept");
    }
    const t = tally(p.all);
    expect(t.bytetrack.kept + t.bytetrack.changed).toBe(p.all.length);
    expect(t.ours.kept + t.ours.changed).toBe(p.all.length);
  });
  it("crop stays inside the scene and is 16:10", () => {
    const p = pickStory(frames);
    const meta = lines[0];
    const [x, y, w, h] = cropFor(frames, p.story!, meta);
    expect(x).toBeGreaterThanOrEqual(0); expect(y).toBeGreaterThanOrEqual(0);
    expect(x + w).toBeLessThanOrEqual(meta.width + 1e-6); expect(y + h).toBeLessThanOrEqual(meta.height + 1e-6);
    expect(w / h).toBeCloseTo(1.6, 1);
  });
  it("page shows one headline and left = ByteTrack, right = ours; no 'same number' badge", async () => {
    const body = readFileSync("public/offline/gate-sim.ndjson", "utf8");
    vi.stubGlobal("fetch", vi.fn(async () => new Response(body)));
    render(<MomentCompare />);
    await waitFor(() => expect(screen.getByTestId("mc-headline")).toBeTruthy());
    expect(screen.getByTestId("mc-pane-bytetrack").textContent).toMatch(/^Left · ByteTrack/);
    expect(screen.getByTestId("mc-pane-ours").textContent).toMatch(/^Right · Ours: BoT-SORT \+ visibility/);
    expect(document.body.textContent).not.toMatch(/same number|LOOK/);
    expect(screen.getByTestId("mc-headline").textContent).toBe(pickStory(frames).story!.headline);
    vi.unstubAllGlobals();
  });
});

function mockCtx() {
  const calls: string[] = [];
  const st = { fillStyle: "", strokeStyle: "", globalAlpha: 1, lineWidth: 1 };
  const ctx = new Proxy(st as Record<string, unknown>, {
    get(t, k: string) { if (k in t) return t[k]; return (...a: unknown[]) => { calls.push(`${k}${k === "setLineDash" ? JSON.stringify(a[0]) : ""}`); }; },
    set(t, k: string, v) { t[k] = v; if (k === "globalAlpha" || k === "fillStyle") calls.push(`${k}=${v}`); return true; },
  });
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
}

describe("figure rendering", () => {
  it("one filled silhouette in a muted version of the ID colour, scaled to the box", () => {
    const { ctx, calls } = mockCtx();
    drawFigure(ctx, { cx: 50, footY: 200, h: 100, color: "#56B4E9", step: 0 });
    expect(calls).toContain(`fillStyle=${muted("#56B4E9")}`);
    expect(calls.filter((c) => c === "fill").length).toBe(2); // shadow + body
    expect(calls).toContain("arc"); // head
    const g = figureGeom({ cx: 50, footY: 200, h: 100 });
    expect(g.top).toBe(100);
    expect(g.headCy - g.headR).toBeCloseTo(100);
  });
  it("hidden person: 30% and dashed outline, not removed", () => {
    const { ctx, calls } = mockCtx();
    drawFigure(ctx, { cx: 50, footY: 200, h: 100, color: "#56B4E9", step: 1, hidden: true });
    expect(calls).toContain("globalAlpha=0.3");
    expect(calls.some((c) => /^setLineDash\[\d/.test(c))).toBe(true);
    expect(calls).toContain("fill");
  });
  it("two-frame walk cycle", () => {
    expect(new Set([0, 0.25, 0.5, 0.75].map((t) => walkStep(t)))).toEqual(new Set([0, 1]));
  });
});

describe("quiet timeline", () => {
  const P = (key: number, frames: number, gap: [number, number] | null, changes = 0) => ({
    key, ids: [key], visible: [{ f0: 1, f1: frames, id: key }], hidden: gap ? [{ f0: gap[0], f1: gap[1], id: key }] : [],
    changes: changes ? [{ f: 5, from: key, to: key + 100 }] : [], swaps: [], frames, f0: 1, f1: frames,
  });
  it("hides < 1 s tracks and folds < 0.3 s gaps, never hides an ID change", () => {
    const ps = [P(1, 100, [10, 14]), P(2, 20, null), P(3, 20, null, 1), P(4, 100, [20, 40])];
    const q = quietView(ps, 30, false);
    expect(q.shown.map((p) => p.key)).toEqual([1, 3, 4]);
    expect(q.hiddenCount).toBe(1);
    expect(q.shown[0].hidden).toEqual([]);
    expect(q.shown[2].hidden).toHaveLength(1);
    expect(quietView(ps, 30, true).shown).toHaveLength(4);
  });
});

describe("takeaway numbers come from data", () => {
  it("MOT takeaway sums per-person verify files; gate takeaway matches the tally", async () => {
    const { motTakeaway, gateTakeaway } = await import("../src/components/Takeaway");
    vi.stubGlobal("fetch", vi.fn(async (u: string) => new Response(readFileSync(`public${String(u).replace(/^\//, "/")}`, "utf8"))));
    const a = JSON.parse(readFileSync("public/verify/MOT17-04.json", "utf8")).people, b = JSON.parse(readFileSync("public/verify/MOT17-09.json", "utf8")).people;
    const m = await motTakeaway();
    expect(m!.big).toBe(`${a.ours.kept + b.ours.kept} of ${a.ours.labelled + b.ours.labelled}`);
    const g = await gateTakeaway();
    const p = pickStory(frames), t = tally(p.all);
    expect(g!.big).toBe(`${t.ours.kept} of ${p.all.length}`);
    vi.unstubAllGlobals();
  });
});
