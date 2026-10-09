import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { buildStages, CONTRACT_EXAMPLES, Diagram } from "../src/pages/Architecture";
import { OverviewView } from "../src/pages/Overview";
import { ResultsView } from "../src/pages/Results";
import { useState } from "react";
import { bench, result } from "./fixtures";

describe("overview", () => {
  it("shows the three headline numbers from the benchmarks", () => {
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><OverviewView data={{ bench: bench(), still: null }} /></MemoryRouter>);
    const tiles = screen.getAllByTestId("tile").map((t) => t.textContent);
    expect(tiles[0]).toContain("59.8");
    expect(tiles[1]).toContain("180");
    expect(tiles[2]).toContain("2.27");
    expect(screen.getByRole("link", { name: "Open the demo" })).toHaveProperty("href", expect.stringContaining("/demo"));
  });
  it("shows an em dash when nothing is measured", () => {
    render(<MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><OverviewView data={{ bench: null, still: null }} /></MemoryRouter>);
    for (const t of screen.getAllByTestId("tile")) {
      expect(t.textContent).toContain("—");
      expect(t.textContent).toContain("not measured");
    }
  });
});

describe("results", () => {
  it("marks our row blue and baselines grey, tags sanity checks, and labels GPU rows pending", () => {
    render(<ResultsView b={bench()} />);
    expect(screen.getByTestId("row-ours_vis_botsort").className).toContain("bar-ours");
    expect(screen.getByTestId("row-ours_vis_botsort").className).toContain("headline");
    expect(screen.getByTestId("row-botsort").className).toContain("bar-baseline");
    expect(screen.getAllByText("sanity check").length).toBeGreaterThanOrEqual(2); // gt-as-tracker + go/no-go
    expect(screen.getAllByText("pending DGX run").length).toBeGreaterThanOrEqual(3);
    const tables = document.querySelectorAll("table");
    const firstFigure = document.querySelector("figure");
    // tables come before charts
    if (firstFigure) expect(tables[0].compareDocumentPosition(firstFigure) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(tables[0]).getAllByText("52.5")[0]).toBeTruthy();
  });
});

function DiagramHarness() {
  const stages = buildStages(result([{ kind: "gate", type: "crossing", frame: 6, track_id: 7, direction: "in", timestamp: "t", gate: "gate" }]), "job1");
  const [active, setActive] = useState("camera");
  const sel = stages.find((s) => s.id === active)!;
  return (
    <>
      <Diagram stages={stages} active={active} onSelect={setActive} />
      <pre data-testid="payload">{JSON.stringify(sel.payload)}</pre>
    </>
  );
}

describe("architecture", () => {
  it("has five stages left to right and shows a stage's payload on click", () => {
    render(<DiagramHarness />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.querySelector(".t")?.textContent)).toEqual(["Gate camera", "Person detector", "Tracker", "Gate logic", "Event API"]);
    fireEvent.click(tabs[1]);
    expect(screen.getByTestId("payload").textContent).toContain("boxes_tlwh");
    fireEvent.click(tabs[3]);
    expect(screen.getByTestId("payload").textContent).toContain("first_crossing");
    fireEvent.click(tabs[4]);
    expect(JSON.parse(screen.getByTestId("payload").textContent!)).toEqual(CONTRACT_EXAMPLES);
  });
  it("asks for a run when there is no job", () => {
    const stages = buildStages(null, null);
    expect(stages[1].payload).toMatch(/Run a job/);
    expect(stages[4].payload).toEqual(CONTRACT_EXAMPLES);
  });
});
