import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { parseClip } from "@viewer/data";
import { resultText } from "../src/guided/model";
import { fromViewerClips } from "../src/guided/viewer";
import { benchmarkPairs } from "../src/workstation/screens/GuidedStudio";

const load = (f: string) => parseClip(JSON.parse(gunzipSync(readFileSync(resolve(__dirname, "../../viewer/data", f))).toString("utf8")));

describe("benchmark clips in the guided player", () => {
  const clips = ["MOT17-09.bytetrack", "MOT17-09.botsort", "MOT17-09.gt"].map((n) => load(`${n}.json.gz`));
  it("pairs ByteTrack with another tracker, never with gt", () => {
    const [p] = benchmarkPairs(clips);
    expect(p.name).toBe("MOT17-09");
    expect(p.base.tracker).toBe("bytetrack");
    expect(p.ours.tracker).toBe("botsort");
  });
  it("moments come from measured id switches and say so", () => {
    const d = fromViewerClips(clips[0], clips[1], { baseline: "ByteTrack", ours: "BoT-SORT" });
    expect(d.kind).toBe("measured");
    expect(d.moments.length).toBeGreaterThan(0);
    expect(d.moments.every((m) => m.verified)).toBe(true);
    expect(resultText(d.moments[0], d.kind, { baseline: "ByteTrack", ours: "BoT-SORT" })).toMatch(/^Checked against human labels: ByteTrack #\d+ became #\d+/);
  });
});
