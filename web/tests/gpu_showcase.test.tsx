import { fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { ResultsView } from "../src/pages/Results";
import { bestOf, judge, reidTitle, whereWeLose } from "../src/components/GpuShowcase";
import type { Benchmarks } from "../src/api";

// the committed snapshot of GET /api/benchmarks (scripts/web/export_offline.py over docs/results)
const snap = (): Benchmarks => JSON.parse(readFileSync(resolve(__dirname, "../public/offline/benchmarks.json"), "utf-8"));

describe("results: GPU campaign showcase (numbers from the bundle files)", () => {
  it("scale strip counts come from the files; the simulated card is separate", () => {
    render(<ResultsView b={snap()} />);
    const real = screen.getByTestId("scale-real");
    expect(within(real).getByTestId("scale-evaluations").textContent).toMatch(/^4clean evaluations/);
    expect(within(real).getByTestId("scale-runs").textContent).toMatch(/^44tracker runs scored14 tracker configs · 14 \+ 10 \+ 10 \+ 10/);
    expect(within(real).getByTestId("scale-frames").textContent).toMatch(/^37,086frames scored40 sequences · 381,468/);
    expect(within(real).getByTestId("scale-gng").textContent).toContain("failed 5/5");
    expect(within(real).getByTestId("scale-excluded").textContent).toMatch(/^1run excluded for leakage/);
    expect(within(real).queryByTestId("scale-stress")).toBeNull();
    const sim = screen.getByTestId("scale-sim");
    expect(within(sim).getByTestId("scale-stress").textContent).toBe("1,920simulated stress scenes48 cells × 20 seeds (1000–1019) × 2 trackers · synthetic, not a benchmark");
    expect(screen.getByTestId("scale-hw").textContent).toContain("RTX 3050");
    expect(screen.getByTestId("scale-hw").textContent).toContain("osnet_ain_x1_0");
  });

  it("re-ID headline: title only claims what the rows show; better cells are marked", () => {
    render(<ResultsView b={snap()} />);
    expect(screen.getByTestId("reid-title").textContent).toBe("One rule, four datasets: fewer ID switches every time");
    const row = (id: string) => screen.getByTestId(`reid-${id}`).textContent;
    expect(row("mot17_val_half")).toContain("62.5 / 183");
    expect(row("mot17_val_half")).toContain("63.2 / 167");
    expect(row("mot20_val_half__sparsetrack_ablation")).toContain("67.5 / 1,278");
    expect(row("mot20_val_half__sparsetrack_ablation")).toContain("69.5 / 1,019");
    expect(row("mot20_val_half__mot17full_xdomain")).toContain("60.3 / 1,160");
    expect(row("dancetrack_val")).toContain("53.3 / 1,883");
    expect(row("dancetrack_val")).toContain("53.4 / 1,681");
    for (const id of ["mot17_val_half", "mot20_val_half__sparsetrack_ablation", "mot20_val_half__mot17full_xdomain", "dancetrack_val"]) {
      expect(screen.getByTestId(`reid-di-${id}`).className).toBe("better");
      expect(screen.getByTestId(`reid-dh-${id}`).className).toBe("better");
    }
    const p = snap().gpu!.reid_rule!.pairs[0];
    expect(reidTitle({ ...p, fewer_idsw_everywhere: false, rows: p.rows.map((r, i) => (i === 0 ? { ...r, d_IDSW: 5 } : r)) })).toBe("One rule, four datasets: fewer ID switches on 3 of 4");
  });

  it("leaderboard: every row sorted by HOTA, ours highlighted, best per column bold even when not ours", () => {
    render(<ResultsView b={snap()} />);
    const abl = screen.getByTestId("gpu-ablation");
    const names = () => within(abl).getByTestId("leaderboard").querySelectorAll("tbody tr");
    expect(names()).toHaveLength(14);
    expect(names()[0].getAttribute("data-testid")).toBe("gpu-row-botsort");
    expect(within(abl).getByTestId("gpu-row-botsort_reid_vis_gated").className).toContain("lb-ours");
    expect(within(abl).getByTestId("gpu-row-bytetrack").textContent).toContain("5,379");
    expect(within(abl).getByTestId("gpu-row-bytetrack").textContent).toContain("12,552");
    fireEvent.click(within(abl).getByTestId("gpu-ds-dancetrack_val"));
    expect(names()).toHaveLength(10);
    expect(names()[0].getAttribute("data-testid")).toBe("gpu-row-hybrid_sort");
    expect(within(abl).getByTestId("gpu-row-hybrid_sort").querySelector("td.best")?.textContent).toBe("57.4");
  });

  it("ours vs our own ByteTrack: direction-aware better / worse", () => {
    render(<ResultsView b={snap()} />);
    const own = screen.getByTestId("own-baseline");
    expect(within(own).getByTestId("own-HOTA").textContent).toBe("HOTA63.463.9+0.5 · better");
    expect(within(own).getByTestId("own-IDF1").textContent).toContain("+1.3 · better");
    expect(within(own).getByTestId("own-IDSW").textContent).toContain("−9 · better");
    expect(within(own).getByTestId("own-MOTA").textContent).toContain("66.8");
    expect(within(own).getByTestId("own-p95_ms").querySelector("td:last-child")?.className).toBe("worse");
    expect(judge(1, 2, true)).toBe("better");
    expect(judge(1, 2, false)).toBe("worse");
    expect(judge(2, 2, false)).toBe("same");
    expect(bestOf([{ name: "a", label: "a", group: "x", HOTA: 1, IDF1: 1, MOTA: 1, AssA: 1, DetA: 1, IDSW: 9, p95_ms: 3 }, { name: "b", label: "b", group: "x", HOTA: 2, IDF1: 0, MOTA: 1, AssA: 1, DetA: 1, IDSW: 4, p95_ms: 5 }])).toMatchObject({ HOTA: 2, IDF1: 1, IDSW: 4, p95_ms: 3 });
  });

  it("paper table is grey reference: gap stated as detection, no better-than-paper wording", () => {
    render(<ResultsView b={snap()} />);
    const rep = screen.getByTestId("gpu-reproduction");
    expect(rep.className).toContain("paper-ref");
    expect(within(rep).getByTestId("rep-line").textContent).toMatch(/10\.2 MOTA below the paper, mostly from detection\s*\(\+2,021 FP, \+3,471 FN\)/);
    expect(rep.textContent).not.toMatch(/better/i);
    expect(screen.getByTestId("gpu-go-no-go").textContent).toContain("Tested an idea, failed 5/5, dropped");
    const lose = screen.getByTestId("where-we-lose").textContent ?? "";
    expect(lose).toContain("DanceTrack val: Hybrid-SORT HOTA 57.4 vs ours 54.3");
    expect(lose).toMatch(/MOT20 val-half · SparseTrack det speed: ours p95 87–95 ms per frame vs ByteTrack 13\.7 ms/);
    expect(whereWeLose([])).toEqual([]);
  });
});
