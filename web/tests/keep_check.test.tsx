// Kept-vs-swap verifier in the UI + comparison summaries (gate scene, real clips).
import { readFileSync } from "fs";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { chipText, people, summaryText, tracksOf, boxesAt } from "../src/identity/story";
import { counts, methodTooltip, swapLabel, verdictOf, type VerifyDoc } from "../src/identity/verify";
import { storyCaption, storyMoments, STORY_MOMENTS } from "../src/identity/moments";
import { PersonTimeline } from "../src/identity/PersonTimeline";
import { differences, paneStats, transfersOf } from "../src/compare/gateDiff";
import { personRows, runSummary } from "../src/compare/RealCompare";
import { fromScenario } from "../src/guided/model";
import type { SimFrame } from "../src/lib/gateSim";

const json = (p: string) => JSON.parse(readFileSync(p, "utf8"));
const turnDoc = json("public/verify/subway_turnstile.json") as VerifyDoc;
const turn = fromScenario(json("public/scenarios/subway_turnstile.json"), null, null);

// re-pinned after the overlays were recomputed on the raw clips (see tests/test_verify_keeps.py)
describe("kept-vs-swap verdicts (regression: subway turnstile #14)", () => {
  it("#14 reappearing at frame 318 (last seen 230) on a different person is a possible swap, never 'kept'", () => {
    const v = verdictOf(turnDoc, "ours");
    expect(v(14, 318)?.verdict).toBe("swap");
    const tr = tracksOf(turn.runs.ours.frames);
    const b = boxesAt(tr, 318, turn.fps, v).find((x) => x.id === 14)!; // #14 is drawn for one frame only
    expect(b.state).toBe("swap");
    expect(chipText(b, true)).toBe(swapLabel(14));
    expect(chipText(b, true)).toBe("possible swap: #14 given to someone else");
    // the earlier gap (110 -> 149) passed the check
    expect(boxesAt(tr, 150, turn.fps, v).find((x) => x.id === 14)?.state).toBe("kept");
  });
  it("flagged swaps break the timeline and are counted as uncertain in the summary", () => {
    const { people: ps } = people(turn, "ours", verdictOf(turnDoc, "ours"));
    const four = ps.find((p) => p.ids.includes(14))!;
    expect(four.swaps.map((s) => [s.last, s.f])).toEqual([[230, 318]]);
    expect(four.hidden.some((h) => h.f1 + 1 === 318)).toBe(false);
    expect(summaryText(ps)).toMatch(/uncertain \(possible swap\)/);
    render(<PersonTimeline people={[four]} frames={turn.frames} f={1} focus={14} onSeek={() => undefined} />);
    expect(screen.getByTestId("ptl-swap").textContent).toContain("swap? #14");
  });
  it("counts per run match the file; tooltip says heuristic, not ground truth", () => {
    expect(counts(turnDoc, "ours")).toEqual({ kept: turnDoc.summary.ours!.kept, swap: turnDoc.summary.ours!.swap, unknown: 0 });
    expect(methodTooltip(turnDoc)).toMatch(/Heuristic check, not ground truth/);
    expect(methodTooltip(json("public/verify/MOT17-09.json"))).toMatch(/human labels/);
    expect(methodTooltip(null)).toMatch(/not checked/);
  });
});

describe("story moments", () => {
  it("every listed moment is confirmed kept by its check file, with a caption built from it", () => {
    expect(STORY_MOMENTS.length).toBeGreaterThanOrEqual(3);
    for (const r of STORY_MOMENTS) {
      const ms = storyMoments(r.clip, json(`public/verify/${r.clip}.json`));
      const m = ms.find((x) => x.ref === r)!;
      expect(m, `${r.clip} #${r.id}`).toBeTruthy();
      expect(m.caption).toContain(`comes back still #${r.id}`);
    }
    const t14 = storyMoments("subway_turnstile", turnDoc)[0];
    expect(t14.caption).toBe(storyCaption(t14.ev));
    expect(t14.caption).toMatch(/^Person #14 walks behind #10 for 1\.3 s \(38 frames\)/);
  });
  it("a moment whose check is not 'kept' is dropped, not relabelled", () => {
    expect(storyMoments("subway_turnstile", turnDoc).map((m) => m.ev.id)).toEqual([14]);
    const bad: VerifyDoc = { ...turnDoc, events: turnDoc.events.map((e) => (e.id === 14 ? { ...e, verdict: "swap" as const } : e)) };
    expect(storyMoments("subway_turnstile", bad).map((m) => m.ev.id)).toEqual([]);
  });
});

function frame(f: number, people: [number, number][], bt: [number, number][], ours: [number, number][], ev: { bytetrack?: [number, number, number, number, number][]; ours?: [number, number, number, number, number][] } = {}, counted = [0, 0]): SimFrame {
  return {
    type: "frame", f, people: people.map(([p, x]) => [p, x, 0, 1]), dets: [], true_count: 2,
    bytetrack: { ids: bt, extra: [], events: ev.bytetrack ?? [], counted: counted[0], id_changes: (ev.bytetrack ?? []).length },
    ours: { ids: ours, extra: [], events: ev.ours ?? [], counted: counted[1], id_changes: (ev.ours ?? []).length },
  };
}

describe("gate comparison summary", () => {
  it("transfers = a tracker number moving from one simulated person to another", () => {
    const fs = [frame(1, [[1, 0], [2, 9]], [[1, 5], [2, 6]], [[1, 5], [2, 6]]), frame(2, [[1, 0], [2, 9]], [[2, 5]], [[1, 5], [2, 6]], { bytetrack: [[2, 6, 5, 9, 0]] }, [1, 2])];
    expect(transfersOf(fs, "bytetrack")).toEqual([{ f: 2, idx: 1, tid: 5, from: 1, to: 2 }]);
    expect(transfersOf(fs, "ours")).toEqual([]);
    expect(paneStats(fs, "bytetrack")).toEqual({ idChanges: 1, transfers: 1, counted: 1, truth: 2 });
    const d = differences(fs, 30);
    expect(d.map((x) => [x.pane, x.kind, x.people])).toEqual([["bytetrack", "change", [2]], ["bytetrack", "transfer", [2, 1]]]);
  });
  it("on the recorded seed the strip numbers equal the tracker's own end summary", () => {
    const lines = readFileSync("public/offline/gate-sim.ndjson", "utf8").trim().split("\n").map((l) => JSON.parse(l));
    const fs = lines.filter((l) => l.type === "frame") as SimFrame[];
    const end = lines[lines.length - 1];
    for (const p of ["bytetrack", "ours"] as const) {
      const s = paneStats(fs, p);
      expect([s.idChanges, s.counted, s.truth]).toEqual([end.summary[p].id_changes, end.summary[p].counted, end.summary[p].truth]);
    }
    expect(differences(fs, 30).length).toBeGreaterThan(0);
  });
});

describe("real-clip comparison table", () => {
  it("per-person rows for both trackers; kept never exceeds people; reappearances are the listed people's", () => {
    for (const run of ["baseline", "ours"] as const) {
      const rows = personRows(turn, run, turnDoc);
      const s = runSummary(rows);
      expect(s.kept + s.changed + s.uncertain).toBe(rows.length);
      expect(s.kept).toBeLessThanOrEqual(s.people);
      // reappearance events are a different unit (one person can come back several times) and
      // only those of listed people are counted, so they can be fewer than the whole check file
      expect(s.backConfirmed).toBe(rows.reduce((a, r) => a + r.kept, 0));
      expect(s.backConfirmed).toBeLessThanOrEqual(turnDoc.summary[run]!.kept);
      expect(s.backFlagged).toBeLessThanOrEqual(turnDoc.summary[run]!.swap);
    }
    expect(personRows(turn, "ours", turnDoc).find((r) => r.ids.includes(4))?.status).toBe("uncertain");
  });
});

import { mot20Status, parseGoNoGo, parseOracle } from "../src/lib/mot20";
describe("MOT20 status in Results", () => {
  it("parses only what the files say", () => {
    const md = readFileSync("../docs/results/go_no_go.md", "utf8");
    const k0 = parseGoNoGo(md)!;
    expect(md).toContain(`HOTA ${k0.HOTA}, IDF1 ${k0.IDF1}, MOTA ${k0.MOTA}, IDSW ${k0.IDSW}`);
    expect(Number(parseOracle(readFileSync("../docs/evaluation.md", "utf8"))?.HOTA)).toBeGreaterThan(90);
    const s = mot20Status();
    expect(s.runs.map((r) => r.leaked)).toEqual([false, true, false]);
    expect(s.viewerExport).toBe(false);
  });
});
