import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Term } from "../src/components/Term";
import { frameAt, markerLabel, markers, neutralColor, NEUTRAL_IDS, type ScenarioData, seekBefore } from "../src/lib/scenarios";
import { FoundChecks, VerdictPill } from "../src/components/Checks";
import { checks, cleanRuns, LEAK_NOTE, loadSummary, methodResult, PENDING, showProvisionalBanner, type SummaryFile } from "../src/lib/summary";
import { LandingView } from "../src/pages/Landing";
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

const WIN: SummaryFile = {
  ran: [{ id: "mot17_val_half", cache: "MOT17/ablation/train", saw_eval_half: false }],
  reproduction: { reference: { MOTA: 76.6, IDF1: 79.3 }, ours: { MOTA: 76.2, IDF1: 79.0 }, verdict: "REPRODUCED" },
  go_no_go_v2: "GO",
  metrics: { "mot17.p95_ms": 3.2, "delayed.idsw_k0": 100, "delayed.idsw_k5": 80, "delayed.held_frames_pct": 4 },
};

describe("what we found (docs/results/summary.json)", () => {
  it("the repo summary: leak excluded, NOT REPRODUCED with numbers, NO-GO, latency + method pending", () => {
    const s = loadSummary();
    expect(s).not.toBeNull();
    const c = checks(s);
    expect(c.map((x) => x.id)).toEqual(["leak", "reproduction", "go_no_go", "latency"]);
    expect(c.map((x) => x.pill)).toEqual(["PASS", "NOT REPRODUCED", "NO-GO", "PENDING"]);
    const leak = c[0].evidence.find((e) => e.struck);
    expect(leak?.text).toBe("mot20_val_half__mot20full_SAW_EVAL_HALF");
    expect(leak?.struck).toBe(LEAK_NOTE);
    expect(c[1].evidence[0].text).toBe("MOTA 66.4 vs 76.6 published");
    expect(c[1].evidence.map((e) => e.text)).toContain("FN 12,552 vs 9,081 published");
    expect(cleanRuns(s).some((r) => r.saw_eval_half)).toBe(false);
    expect(cleanRuns(s)).toHaveLength(3);
    expect(methodResult(s).pending).toBe(true);
    expect(showProvisionalBanner(s)).toBe(true);
  });
  it("renders the banner, struck-through leaked row and ✕ outlined fail pills (no red)", () => {
    render(<FoundChecks summary={loadSummary()} />);
    expect(screen.getByTestId("provisional-banner").textContent).toBe("Provisional · detector setup being fixed");
    const cards = screen.getAllByTestId("check-card");
    expect(cards).toHaveLength(4);
    expect(cards[0].querySelector("s")?.textContent).toBe("mot20_val_half__mot20full_SAW_EVAL_HALF");
    const pills = screen.getAllByTestId("pill");
    expect(pills.map((p) => p.getAttribute("data-state"))).toEqual(["PASS", "NOT REPRODUCED", "NO-GO", "PENDING"]);
    expect(pills[1].textContent).toBe("✕ NOT REPRODUCED");
    expect(pills[1].className).toBe("pill fail");
    expect(pills[3].className).toBe("pill provisional");
    expect(screen.getByTestId("method-result").textContent).toContain(PENDING);
  });
  it("a passing run renders without code changes: PASS pills, no banner, before/after numbers", () => {
    const c = checks(WIN);
    expect(c.map((x) => x.pill)).toEqual(["PASS", "PASS", "PASS", "PASS"]);
    expect(showProvisionalBanner(WIN)).toBe(false);
    render(<FoundChecks summary={WIN} />);
    expect(screen.queryByTestId("provisional-banner")).toBeNull();
    const m = screen.getByTestId("method-result").textContent ?? "";
    expect(m).toContain("100");
    expect(m).toContain("80");
    expect(m).toContain("4%");
  });
  it("no summary: every check pending, banner shown; latency provisional until reproduced", () => {
    expect(checks(null).every((x) => x.pill === "PENDING")).toBe(true);
    expect(showProvisionalBanner(null)).toBe(true);
    const notRepro = { ...WIN, reproduction: { verdict: "NOT REPRODUCED" } };
    expect(checks(notRepro)[3].pill).toBe("PROVISIONAL");
    render(<VerdictPill pill="PASS" />);
    expect(screen.getByTestId("pill").className).toBe("pill pass");
  });
  it("landing has six steps and a Try it link to the dashboard", () => {
    render(<MemoryRouter future={R}><LandingView summary={null} /></MemoryRouter>);
    expect(screen.getAllByTestId("step")).toHaveLength(6);
    expect(screen.getByTestId("id-mixup")).toBeTruthy();
    expect(screen.getByText("What is an ID mix-up?")).toBeTruthy();
    // neutral explainer: no tracker is named as winning or losing in step 1
    expect(screen.getByTestId("id-mixup").textContent).not.toMatch(/ByteTrack|ours kept|BoT-SORT/i);
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
});

describe("scenarios tab", () => {
  it("rail: playable clips first ('What tracking is' leads), MOT17 benchmarks after", () => {
    const sel = vi.fn();
    const first = { ...scenario().meta, id: "man_entering_metro", problem: "What tracking is" };
    render(<ScenarioRail items={[scenario().meta, first]} bench={["MOT17-09", "MOT17-04"]} selected="subway_turnstile" onSelect={sel} />);
    const nav = screen.getByRole("navigation");
    const cards = [...nav.children];
    expect(cards[0].textContent).toMatch(/^What tracking is/);
    expect(cards[1].getAttribute("data-testid")).toBe("scenario-card");
    expect(cards[1].className).toContain("active");
    const img = cards[1].querySelector("img")!;
    expect(img.getAttribute("src")).toBe("/thumbs/subway_turnstile.jpg");
    fireEvent.error(img);
    expect(cards[1].querySelector(".thumb-ph")).toBeTruthy();
    expect(cards[2].textContent).toMatch(/^MOT17-09: real labelled street video \(boxes only\)/);
    const bench = screen.getAllByTestId("bench-card");
    expect(bench[1].textContent).toBe("MOT17-04 · Measured · video not exported yet (boxes only)");
    expect((screen.getByTestId("bench-slot") as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(bench[1]);
    expect(sel).toHaveBeenCalledWith("bench:MOT17-04");
  });
  it("player: ours only, intro, watch line, counters in a bar outside the frame, no comparison", () => {
    const guide = { gate: { a: [0, 540], b: [1920, 540] }, events: { baseline: [], ours: [{ t: "cross", f: 1, id: 1, dir: "in" }] }, moments: [] };
    render(<MemoryRouter><ScenarioPlayer data={{ ...scenario(), guide } as never} /></MemoryRouter>);
    expect(screen.getByTestId("identity-player")).toBeTruthy();
    expect(screen.queryByTestId("compare-player")).toBeNull();
    expect(screen.queryByTestId("pane-baseline")).toBeNull();
    expect(screen.getByTestId("scenario-intro").textContent).toContain("What you'll see");
    expect(screen.getByTestId("watch-line").textContent).toMatch(/^Watch/);
    const bar = screen.getByTestId("count-bar");
    expect(bar.textContent).toContain("IN 1 · OUT 0");
    expect(screen.getByTestId("identity-stage").contains(bar)).toBe(false);
    expect(screen.getByTestId("compare-link").getAttribute("href")).toBe("/dashboard/live");
    expect(screen.getByText(/Illustrative only/)).toBeTruthy();
  });
  it("strip ticks and arrow keys jump between moments", async () => {
    const guide = {
      gate: { a: [0, 540], b: [1920, 540] }, events: { baseline: [], ours: [] },
      moments: [100, 400].map((f) => ({ f, kind: "overlap", changed: { baseline: true, ours: false }, counted_twice: { baseline: false, ours: false },
        focus: [0, 0, 10, 10], people: [{ box: [0, 0, 10, 10], ids: { baseline: { before: 2, after: 9 }, ours: { before: 2, after: 2 } } }] })),
    };
    const d = { ...scenario(), guide };
    const { fromScenario } = await import("../src/guided/model");
    const { GuidedPlayer } = await import("../src/guided/GuidedPlayer");
    const replay = { ...fromScenario(d as never, null), duration: 20 };
    render(<GuidedPlayer data={replay} />);
    const seek = screen.getByLabelText("Seek position") as HTMLInputElement;
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(Number(seek.value)).toBeCloseTo((100 - 1) / replay.fps - 1.5, 1);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(Number(seek.value)).toBeCloseTo((400 - 1) / replay.fps - 1.5, 1);
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(Number(seek.value)).toBeCloseTo((100 - 1) / replay.fps - 1.5, 1);
    fireEvent.click(screen.getAllByTestId("moment-tick")[1]);
    expect(Number(seek.value)).toBeCloseTo((400 - 1) / replay.fps - 1.5, 1);
  });
  it("video 404s: falls back to the demo cut, then to the schematic with a hint; log hides future events", async () => {
    const { fromScenario, MISSING_CLIP_TEXT } = await import("../src/guided/model");
    const { GuidedPlayer } = await import("../src/guided/GuidedPlayer");
    const guide = { gate: { a: [0, 540], b: [1920, 540] }, moments: [],
      events: { baseline: [], ours: [{ t: "cross", f: 1, id: 1, dir: "in" }, { t: "cross", f: 280, id: 2, dir: "out" }] } };
    const replay = fromScenario({ ...scenario(), guide } as never, "/api/clips/subway_turnstile/video");
    render(<GuidedPlayer data={replay} />);
    const v = () => document.querySelector("video");
    expect(v()?.getAttribute("src")).toBe("/api/clips/subway_turnstile/video");
    expect(screen.queryByTestId("video-hint")).toBeNull();
    fireEvent.error(v()!);
    expect(v()?.getAttribute("src")).toBe("/demo/subway_turnstile.mp4");
    fireEvent.error(v()!);
    expect(v()).toBeNull();
    expect(screen.getByTestId("video-hint").textContent).toBe(MISSING_CLIP_TEXT);
    expect(document.querySelector("canvas")).toBeTruthy(); // boxes still drawn on the schematic
    const log = screen.getByTestId("event-log").textContent ?? "";
    expect(log).toContain("Person 1 crossed the line going IN.");
    expect(log).not.toContain("Person 2"); // 9.3 s event is not shown at the start
  });
  it("tooltips give a one-line definition", () => {
    render(<Term name="IDSW" />);
    expect(screen.getByText("IDSW").getAttribute("data-tip")).toMatch(/ID switches/);
  });
});
