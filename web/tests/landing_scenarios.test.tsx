import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { exampleWindow } from "../src/components/ExampleClip";
import { Term } from "../src/components/Term";
import { frameAt, markerLabel, markers, neutralColor, NEUTRAL_IDS, type ScenarioData, seekBefore } from "../src/lib/scenarios";
import { CARDS, cardValue, loadSummary, PENDING } from "../src/lib/summary";
import { LandingView, NumberCards } from "../src/pages/Landing";
import { ScenarioPlayer, ScenarioRail } from "../src/pages/Scenarios";

const R = { v7_startTransition: true, v7_relativeSplatPath: true };

function scenario(over: Partial<ScenarioData["meta"]> = {}): ScenarioData {
  return {
    meta: {
      id: "subway_turnstile", title: "Subway turnstiles", problem: "Counting at a gate", caption: "Watch IDs.",
      source: "https://www.pexels.com/x", licence: "Pexels License", detector: "yolox_s", kind: "illustrative",
      width: 1920, height: 1080, src_fps: 30, stride: 1, frames_tracked: 300, covered_s: 10,
      id_switches: { baseline: 8, ours: 13 }, tracks: { baseline: 20, ours: 22 }, disagreements: 3, ...over,
    },
    runs: {
      baseline: { label: "ByteTrack", config: "b", id_switches: 8, tracks: 20, frames: { "1": [[1, 10, 10, 50, 100]] } },
      ours: { label: "Ours", config: "o", id_switches: 13, tracks: 22, frames: { "1": [[1, 10, 10, 50, 100]] } },
    },
    disagreements: [
      { frame: 90, kept: "ours", ours_id: 4, baseline_from: 7, baseline_to: 9, box: [0, 0, 10, 10] },
      { frame: 95, kept: "baseline", baseline_id: 2, ours_from: 3, ours_to: 5, box: [0, 0, 10, 10] },
      { frame: 200, kept: "ours", ours_id: 6, baseline_from: 1, baseline_to: 2, box: [0, 0, 10, 10] },
    ],
  };
}

describe("summary numbers", () => {
  it("renders 'pending' when the summary file or a key is missing", () => {
    expect(cardValue(CARDS[0], null)).toBe(PENDING);
    expect(cardValue(CARDS[0], { metrics: {} })).toBe(PENDING);
    expect(cardValue(CARDS[0], { metrics: { [CARDS[0].key]: null } })).toBe(PENDING);
    expect(cardValue(CARDS[0], { metrics: { [CARDS[0].key]: Number.NaN } })).toBe(PENDING);
  });
  it("formats present values with the card's unit", () => {
    expect(cardValue(CARDS[0], { metrics: { [CARDS[0].key]: 12.4 } })).toBe("12%");
  });
  it("has no summary.json yet in the repo, so every landing card is pending", () => {
    expect(loadSummary()).toBeNull();
    render(<NumberCards summary={loadSummary()} />);
    const cards = screen.getAllByTestId("number-card");
    expect(cards).toHaveLength(CARDS.length);
    for (const c of cards) expect(c.textContent).toContain(PENDING);
    expect(CARDS.filter((c) => c.loss)).toHaveLength(1);
  });
  it("landing has six steps and a Try it link to the dashboard", () => {
    render(<MemoryRouter future={R}><LandingView example={null} summary={null} /></MemoryRouter>);
    expect(screen.getAllByTestId("step")).toHaveLength(6);
    expect(screen.getByRole("link", { name: "Open the dashboard" })).toHaveProperty("href", expect.stringContaining("/dashboard"));
  });
});

describe("scenarios helpers", () => {
  it("maps time to tracked frames with stride and range", () => {
    const m = scenario({ stride: 2, src_fps: 60 }).meta;
    expect(frameAt(m, 0)).toBe(1);
    expect(frameAt(m, 1)).toBe(31);
    expect(frameAt(m, 100)).toBeNull();
  });
  it("merges close markers and seeks 2 s before", () => {
    const d = scenario();
    expect(markers(d).map((x) => x.frame)).toEqual([90, 200]);
    expect(seekBefore(d.meta, 90)).toBeCloseTo(0.967, 2);
    expect(seekBefore(d.meta, 10)).toBe(0);
  });
  it("labels markers neutrally on illustrative clips and names the swap on benchmarks", () => {
    const d = scenario().disagreements[0];
    expect(markerLabel("illustrative", d)).toBe("Trackers disagree here");
    expect(markerLabel("measured", d)).toBe("ByteTrack swapped 7 → 9 · ours kept 7");
  });
  it("uses no red in the neutral ID palette", () => {
    for (const c of NEUTRAL_IDS) {
      const r = parseInt(c.slice(1, 3), 16), g = parseInt(c.slice(3, 5), 16), b = parseInt(c.slice(5, 7), 16);
      expect(r > 180 && g < 120 && b < 120).toBe(false);
    }
    expect(neutralColor(-1)).toMatch(/^#/);
  });
  it("picks the example window around a swap that ours kept", () => {
    const w = exampleWindow(scenario());
    expect(w.swapFrame).toBe(90);
    expect(w.end - w.start).toBeCloseTo(6);
  });
});

describe("scenarios tab", () => {
  it("rail shows two pending benchmark slots then the clips with tags", () => {
    const sel = vi.fn();
    render(<ScenarioRail items={[scenario().meta]} selected="subway_turnstile" onSelect={sel} />);
    expect(screen.getAllByTestId("bench-slot")).toHaveLength(2);
    expect(screen.getAllByText("Measured · human labels")).toHaveLength(2);
    expect(screen.getAllByText(/non-commercial/)).toHaveLength(2);
    expect(screen.getByText("Illustrative · no ground truth")).toBeTruthy();
    fireEvent.click(screen.getByTestId("scenario-card"));
    expect(sel).toHaveBeenCalledWith("subway_turnstile");
  });
  it("player shows caption, honest counts, markers and a working toggle", () => {
    render(<ScenarioPlayer data={scenario()} />);
    expect(screen.getByText("Watch IDs.")).toBeTruthy();
    expect(screen.getByTestId("switch-counts").textContent).toContain("ByteTrack 8 · ours 13");
    expect(screen.getByText(/Illustrative only/)).toBeTruthy();
    expect(screen.getByText(/detector yolox_s/)).toBeTruthy();
    expect(screen.getAllByTestId("seek-marker")).toHaveLength(2);
    const base = screen.getByRole("button", { name: "Baseline (ByteTrack)" });
    fireEvent.click(base);
    expect(base.getAttribute("aria-pressed")).toBe("true");
  });
  it("tooltips give a one-line definition", () => {
    render(<Term name="IDSW" />);
    expect(screen.getByText("IDSW").getAttribute("data-tip")).toMatch(/ID switches/);
  });
});
