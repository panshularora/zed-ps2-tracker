import { readFileSync } from "fs";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { fromScenario } from "../src/guided/model";
import { IdentityPlayer } from "../src/identity/IdentityPlayer";
import { boxesAt, chipText, people, tracksOf } from "../src/identity/story";
import { collapseRenumbers, keepIdAssociate, keepIdVerdict } from "../src/identity/keepId";
import type { Box } from "../src/lib/scenarios";

function span(frames: Record<string, Box[]>, id: number, f0: number, f1: number, x = 100) {
  for (let f = f0; f <= f1; f++) {
    const k = String(f);
    frames[k] = [...(frames[k] ?? []), [id, x + f, 40, 50, 80]];
  }
}

describe("keepIdAssociate (isolated bodies)", () => {
  it("a 40-frame gap keeps the same id", () => {
    const frames: Record<string, Box[]> = {};
    span(frames, 1, 1, 10);
    for (let f = 51; f <= 60; f++) frames[String(f)] = [[2, 100 + f, 40, 50, 80]];
    const { frames: out } = keepIdAssociate(frames, { fps: 30, width: 640 });
    expect(out["10"][0][0]).toBe(1);
    expect(out["51"][0][0]).toBe(1);
    expect(out["60"][0][0]).toBe(1);
  });
  it("a far new box mints id 2", () => {
    const frames: Record<string, Box[]> = { "1": [[1, 10, 10, 40, 80]], "2": [[1, 12, 10, 40, 80], [9, 500, 10, 40, 80]] };
    const { frames: out } = keepIdAssociate(frames, { fps: 30, width: 640, maxDist: 96 });
    const ids = out["2"].map((b) => b[0]).sort((a, b) => a - b);
    expect(ids).toEqual([1, 9]); // unused original id is kept; not reminted as 2
  });
});

describe("collapseRenumbers (Studio / Scenarios)", () => {
  it("rewrites the minted id onto the first number; unrelated ids stay", () => {
    const frames: Record<string, Box[]> = {
      "1": [[7, 10, 10, 40, 80], [4, 200, 10, 40, 80]],
      "2": [[21, 12, 12, 40, 80], [4, 202, 10, 40, 80]],
    };
    const { frames: out, idOf } = collapseRenumbers(frames, [{ t: "renumber", f: 2, from: 7, to: 21 }]);
    expect(out["1"].map((b) => b[0])).toEqual([7, 4]);
    expect(out["2"].map((b) => b[0])).toEqual([7, 4]);
    expect(idOf(21)).toBe(7);
    expect(idOf(4)).toBe(4);
  });
});

describe("subway turnstile Studio Keep-ID", () => {
  const raw = JSON.parse(readFileSync("public/scenarios/subway_turnstile.json", "utf8"));
  const d = fromScenario(raw, null, null); // illustrative default: keepId on
  const v = keepIdVerdict();

  it("is labelled Keep-ID and does not claim vis-botsort kept the number", () => {
    expect(d.keepId).toBe(true);
    expect(d.runs.ours.label).toBe("Keep-ID");
    expect(d.events.ours.some((e) => e.t === "renumber")).toBe(false);
  });
  it("#4 is the same number at both returns (75 and 128), shown as kept not a swap", () => {
    const tr = tracksOf(d.runs.ours.frames);
    const idAt = (f: number) => (d.runs.ours.frames[String(f)] ?? []).some((b) => b[0] === 4);
    expect(idAt(1)).toBe(true);
    expect(idAt(75)).toBe(true);
    expect(idAt(95)).toBe(true);
    expect(idAt(128)).toBe(true);
    expect(boxesAt(tr, 76, d.fps, v).find((b) => b.id === 4)?.state).toBe("kept");
    expect(boxesAt(tr, 129, d.fps, v).find((b) => b.id === 4)?.state).toBe("kept");
    expect(chipText(boxesAt(tr, 129, d.fps, v).find((b) => b.id === 4)!, true)).toBe("kept #4");
    const four = people(d, "ours", v).people.find((p) => p.ids.includes(4))!;
    expect(four.ids).toEqual([4]);
    expect(four.changes).toEqual([]);
    expect(four.swaps).toEqual([]);
  });
  it("vis-botsort mint 7→21 is collapsed to #7", () => {
    expect((d.runs.ours.frames["52"] ?? []).some((b) => b[0] === 7)).toBe(true);
    expect((d.runs.ours.frames["65"] ?? []).some((b) => b[0] === 7)).toBe(true);
    expect((d.runs.ours.frames["65"] ?? []).some((b) => b[0] === 21)).toBe(false);
    expect((d.runs.ours.frames["173"] ?? []).some((b) => b[0] === 10)).toBe(true);
    expect((d.runs.ours.frames["173"] ?? []).some((b) => b[0] === 38)).toBe(false);
  });
  it("IdentityPlayer shows Keep-ID, no swap list, and #4 as the person to watch", () => {
    render(<IdentityPlayer data={{ ...d, videoUrl: null, demoUrl: null }} />);
    expect(screen.getByTestId("identity-player").textContent).toContain("Keep-ID");
    expect(screen.queryByTestId("swap-list")).toBeNull();
    expect(screen.getByTestId("watch-line").textContent).toMatch(/come back with the same number/);
    expect(screen.getByTestId("watch-line").textContent).not.toMatch(/possible swap/i);
    expect(screen.getByTestId("kept-tooltip").textContent).toMatch(/Came back with the same number/);
  });
});
