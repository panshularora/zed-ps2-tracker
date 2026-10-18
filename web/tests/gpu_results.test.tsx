import { fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { ResultsView } from "../src/pages/Results";
import type { Benchmarks } from "../src/api";

// the committed snapshot of GET /api/benchmarks (scripts/web/export_offline.py over docs/results)
const snap = (): Benchmarks => JSON.parse(readFileSync(resolve(__dirname, "../public/offline/benchmarks.json"), "utf-8"));

describe("results: laptop GPU run", () => {
  it("shows the reproduction gap and every failed go/no-go criterion with its numbers", () => {
    render(<ResultsView b={snap()} />);
    const rep = screen.getByTestId("gpu-reproduction");
    expect(within(rep).getByTestId("gpu-pill").dataset.state).toBe("NOT REPRODUCED");
    expect(within(rep).getByTestId("rep-MOTA").textContent).toContain("76.6");
    expect(within(rep).getByTestId("rep-MOTA").textContent).toContain("66.4");
    expect(within(rep).getByTestId("rep-MOTA").textContent).toContain("-10.2");
    const gng = screen.getByTestId("gpu-go-no-go");
    expect(within(gng).getByTestId("gpu-pill").dataset.state).toBe("NO-GO");
    for (const id of ["1_idsw", "2_identity", "3_selectivity", "4_compute", "5_replication"]) {
      expect(within(gng).getByTestId(`crit-${id}`).textContent).toMatch(/no$/);
    }
    expect(within(gng).getByTestId("crit-1_idsw").textContent).toContain("183 -> 198");
    expect(within(gng).getByTestId("crit-3_selectivity").textContent).toContain("50.7 %");
  });

  it("shows the per-dataset ablation, switches datasets, and drops the pending placeholders", () => {
    render(<ResultsView b={snap()} />);
    const abl = screen.getByTestId("gpu-ablation");
    expect(within(abl).getByTestId("gpu-row-botsort").textContent).toContain("64.6");
    expect(within(abl).getByTestId("gpu-row-botsort_reid_vis_gated").className).toContain("bar-ours");
    fireEvent.click(within(abl).getByTestId("gpu-ds-dancetrack_val"));
    expect(within(abl).getByTestId("gpu-row-hybrid_sort").textContent).toContain("57.4");
    expect(within(abl).getByTestId("gpu-row-hybrid_sort").querySelector("td.best")).toBeTruthy();
    fireEvent.click(within(abl).getByTestId("gpu-ds-mot20_val_half__sparsetrack_ablation"));
    expect(within(abl).getByTestId("gpu-row-reid_vis_gated").textContent).toContain("69.5");
    expect(within(abl).queryByTestId("gpu-ds-mot20_val_half__mot20full_SAW_EVAL_HALF")).toBeNull();
    expect(abl.textContent).toContain("mot20_val_half__mot20full_SAW_EVAL_HALF");
    expect(within(abl).getAllByRole("img").map((i) => i.getAttribute("src"))).toContain("/api/benchmarks/plots/gpu-mot20_val_half__sparsetrack_ablation-stress_visibility.png");
    expect(screen.queryByText("pending DGX run")).toBeNull();
    expect(screen.getByTestId("gpu-gate-events")).toBeTruthy();
  });
});
