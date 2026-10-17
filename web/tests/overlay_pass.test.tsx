// Scenarios overlay pass: one event tag on the frame (focus person only), hidden = dashed box at the
// last-seen spot with a dimmed chip, other events as timeline ticks + caption-bar lines, 16:9 rail thumbs.
import { readFileSync } from "fs";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { boxesAt, eventLines, frameTag, people, tracksOf, type Drawn } from "../src/identity/story";
import { verdictOf, type VerifyDoc } from "../src/identity/verify";
import { chipAlpha, paintFrame } from "../src/identity/IdentityPlayer";
import { PersonTimeline } from "../src/identity/PersonTimeline";
import { ScenarioRail, THUMB_CROP } from "../src/pages/Scenarios";
import { fromScenario, type ReplayData } from "../src/guided/model";

const json = (p: string) => JSON.parse(readFileSync(p, "utf8"));
const turnDoc = json("public/verify/subway_turnstile.json") as VerifyDoc;
const turnScn = json("public/scenarios/subway_turnstile.json");

/** Canvas stand-in that records every fillText. */
function recorder() {
  const texts: string[] = [];
  const m: Record<string, unknown> = {};
  const tf = { translate: () => tf, scale: () => tf };
  const ctx = new Proxy(m, {
    get(_t, k: string) {
      if (k === "fillText") return (s: string) => texts.push(s);
      if (k === "measureText") return (s: string) => ({ width: s.length * 7 });
      if (k === "getTransform") return () => tf;
      if (k in m) return m[k];
      return () => undefined;
    },
    set(_t, k: string, v) { m[k] = v; return true; },
  }) as unknown as CanvasRenderingContext2D;
  return { ctx, texts };
}

const d = { width: 1000, height: 600, fps: 30, gate: null } as unknown as ReplayData;
const all: Drawn[] = [
  { id: 14, box: [100, 100, 50, 120], state: "kept" },
  { id: 7, box: [300, 100, 50, 120], state: "swap" },
  { id: 20, box: [500, 100, 50, 120], state: "back" },
  { id: 3, box: [700, 100, 50, 120], state: "hidden" },
  { id: 5, box: [850, 100, 50, 120], state: "seen" },
];

describe("only the focus person's event is text on the frame", () => {
  it("frameTag: a short tag for the focus person, nothing for anyone else", () => {
    expect(frameTag(all[0], true)).toBe("kept #14");
    expect(frameTag(all[1], true)).toBe("swap? #7");
    expect(frameTag(all[2], true)).toBe("back #20");
    expect(frameTag(all[3], true)).toBeNull();
    expect(frameTag(all[4], true)).toBeNull();
    for (const b of all) expect(frameTag(b, false)).toBeNull();
  });

  it("painted text = one '#N' chip per person + exactly one tag (focus #14); no 'still … hidden', no swap sentence", () => {
    const { ctx, texts } = recorder();
    paintFrame(ctx, 1000, 600, d, null, [0, 0, 1000, 600], all, null, new Set([14]), 1, new Map());
    expect(texts.filter((t) => /^#\d+$/.test(t)).sort()).toEqual(["#14", "#20", "#3", "#5", "#7"]);
    expect(texts.filter((t) => !/^#\d+$/.test(t))).toEqual(["kept #14"]);
    expect(texts.join(" ")).not.toMatch(/still|hidden|possible swap|given to|not checked/);
  });

  it("following someone moves the single tag to them", () => {
    const { ctx, texts } = recorder();
    paintFrame(ctx, 1000, 600, d, null, [0, 0, 1000, 600], all, 7, new Set([14]), 1, new Map());
    expect(texts.filter((t) => !/^#\d+$/.test(t))).toEqual(["swap? #7"]);
  });

  it("other people's events go to the caption bar instead", () => {
    expect(eventLines(all, new Set([14]))).toEqual([
      "possible swap: #7 given to someone else",
      "#20 back (not checked)",
    ]);
    expect(eventLines(all, new Set([14, 7, 20]))).toEqual([]);
  });
});

describe("a hidden person is a dashed box parked at the last-seen spot, with a dimmed chip", () => {
  it("subway #14 hidden 111–148: the box stays on its frame-110 position the whole gap", () => {
    const tr = tracksOf(turnScn.runs.ours.frames);
    const v = verdictOf(turnDoc, "ours");
    const last = tr.get(14)!.get(110)!;
    for (const f of [111, 125, 148]) {
      const b = boxesAt(tr, f, turnScn.meta.fps ?? 30, v).find((x) => x.id === 14)!;
      expect(b.state).toBe("hidden");
      expect(b.box).toEqual(last);
    }
    expect(chipAlpha("hidden")).toBe(0.5);
    expect(chipAlpha("seen")).toBe(1);
  });
});

describe("timeline: every reappearance is a small tick", () => {
  it("one ptl-back tick per hidden span, with no text", () => {
    const data = fromScenario(turnScn, null);
    const ps = people(data, "ours", verdictOf(turnDoc, "ours")).people;
    const p = ps.find((x) => x.ids.includes(14))!;
    render(<PersonTimeline people={[p]} frames={data.frames} f={1} focus={null} onSeek={() => undefined} />);
    const ticks = screen.getAllByTestId("ptl-back");
    expect(ticks).toHaveLength(p.hidden.length);
    for (const t of ticks) expect(t.textContent).toBe("");
  });
});

describe("rail thumbnails", () => {
  it("fill a 16:9 frame (cover crop), so the portrait Fast Lane clip is not a thin strip", () => {
    render(<ScenarioRail items={[{ id: "rabat_stadium", problem: "Event entrance (Fast Lane)", kind: "illustrative" } as never]} selected={null} onSelect={() => undefined} />);
    const img = screen.getAllByTestId("rail-thumb").find((i) => i.getAttribute("src")?.includes("rabat_stadium")) as HTMLImageElement;
    expect(img.parentElement!.className).toContain("thumb-frame");
    expect(img.style.objectFit).toBe("cover");
    expect(THUMB_CROP.objectFit).toBe("cover");
  });
});
