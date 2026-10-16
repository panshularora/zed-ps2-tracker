import { readFileSync } from "fs";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { fromScenario } from "../src/guided/model";
import { IdentityPlayer } from "../src/identity/IdentityPlayer";
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

describe("subway turnstile Studio: never relabelled (team decision A)", () => {
  const raw = JSON.parse(readFileSync("public/scenarios/subway_turnstile.json", "utf8"));
  const d = fromScenario(raw, null, null); // default: the tracker's own numbers
  void keepIdVerdict;

  it("is not Keep-ID and keeps the run's own label", () => {
    expect(d.keepId).toBe(false);
    expect(d.runs.ours.label).toBe(raw.runs.ours.label);
    expect(d.runs.ours.frames).toBe(raw.runs.ours.frames);
  });
  it("the real 7 → 21 renumber stays visible (no collapse)", () => {
    expect((d.runs.ours.frames["65"] ?? []).some((b) => b[0] === 21)).toBe(true);
    expect((d.runs.ours.frames["65"] ?? []).some((b) => b[0] === 7)).toBe(false);
  });
  it("IdentityPlayer lists #4's flagged swap honestly", () => {
    render(<IdentityPlayer data={{ ...d, videoUrl: null, demoUrl: null }} verify={JSON.parse(readFileSync("public/verify/subway_turnstile.json", "utf8"))} />);
    expect(screen.queryByText(/Keep-ID/)).toBeNull();
    expect(screen.getByTestId("swap-list").textContent).toMatch(/#4/);
  });
});
