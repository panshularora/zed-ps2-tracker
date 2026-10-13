// Vitest port of Panshul's dashboard/test_journey.mjs, test_sidebar.mjs and test_agent.mjs
// (Playwright scripts against a live server with a Windows Chrome path). The workstation is
// rendered in jsdom against the real repo data: viewer/data/*.json.gz and docs/results/*.json are
// served through a mocked fetch, so the API shape matches serve/dashboard_api.py.
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { STEPS, FAILURE, prettyTracker, storyBeat } from "../src/workstation/lib/journey";
import { nextSwitch, personWords, prevSwitch } from "../src/workstation/lib/story";
import Workstation, { jumpFromSearch, jumpToSearch, SCREEN_PATH, storyJump } from "../src/workstation/Workstation";
import { delta, ms, timecode } from "../src/workstation/format";

// jsdom has no DecompressionStream: decode the real clip gzips with node:zlib instead.
vi.mock("../src/workstation/api", async (orig) => {
  const mod = await orig<typeof import("../src/workstation/api")>();
  const { gunzipSync } = await import("node:zlib");
  const { parseClip } = await import("@viewer/data");
  return {
    ...mod,
    fetchClip: async (file: string) => {
      const r = await fetch(`/api/viewer-clips/${encodeURIComponent(file)}`);
      return parseClip(JSON.parse(gunzipSync(Buffer.from(await r.arrayBuffer())).toString("utf8")));
    },
  };
});

const REPO = resolve(__dirname, "../..") + "/";
const CLIPS = readdirSync(`${REPO}viewer/data`).filter((f) => f.endsWith(".json.gz"));

function json(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

function fakeApi(url: string): Response {
  if (url === "/api/workstation") return json({ status: "ok", version: "0.2.0", configs: 3, clips: CLIPS.length, results: {} });
  if (url === "/api/viewer-clips") {
    return json({ clips: CLIPS.map((f) => { const [clip, tracker] = f.replace(".json.gz", "").split("."); return { file: f, clip, tracker }; }) });
  }
  if (url.startsWith("/api/viewer-clips/")) {
    return new Response(readFileSync(`${REPO}viewer/data/${decodeURIComponent(url.split("/").pop()!)}`), { status: 200 });
  }
  if (url === "/api/results/ablation") return json(JSON.parse(readFileSync(`${REPO}docs/results/ablation/ablation.json`, "utf8")));
  if (url === "/api/configs") return json(["baselines/bytetrack", "ours/vis_botsort"]);
  if (url === "/api/outputs") return json({ runs: [] });
  return new Response("not found", { status: 404 });
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async (u: RequestInfo | URL) => fakeApi(String(u))));
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as never;
});
afterEach(() => vi.unstubAllGlobals());

function mount(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        {(Object.entries(SCREEN_PATH) as [keyof typeof SCREEN_PATH, string][]).map(([s, p]) => (
          <Route key={s} path={p} element={<Workstation screen={s} />} />
        ))}
      </Routes>
    </MemoryRouter>,
  );
}

describe("journey copy (lib/journey.ts)", () => {
  it("story beats follow the preselected failure", () => {
    expect(storyBeat(FAILURE.lookFrom).phase).toBe("here");
    expect(storyBeat(FAILURE.goneFrom).phase).toBe("gone");
    expect(storyBeat(FAILURE.backFrom).phase).toBe("back");
    expect(storyBeat(FAILURE.backFrom).now).toContain(String(FAILURE.to));
  });
  it("four steps in order and plain tracker names", () => {
    expect(STEPS.map((s) => s.id)).toEqual(["watch", "compare", "evidence", "try"]);
    expect(prettyTracker("bytetrack").title).toBe("ByteTrack");
    expect(prettyTracker("ours_vis_botsort").title).toBe("BoT-SORT + vis-gate");
  });
});

describe("story + routing helpers", () => {
  it("finds next / previous switch on a chain", () => {
    const ev = [{ type: "id_switch", f: 10, from: 2, to: 5 }, { type: "id_switch", f: 40, from: 5, to: 9 }] as never;
    const ids = new Set([2, 5, 9]);
    expect(nextSwitch(ev, ids, 0)?.f).toBe(10);
    expect(nextSwitch(ev, ids, 10)?.f).toBe(40);
    expect(prevSwitch(ev, ids, 40)?.f).toBe(10);
    expect(personWords([2, 5], 5)).toContain("now #5");
  });
  it("jump round-trips through the query string and defaults to the failure", () => {
    const j = { clip: "MOT17-04", left: "botsort", right: "bytetrack", frame: 12, trackId: 3 };
    expect(jumpFromSearch(jumpToSearch(j))).toEqual(j);
    expect(storyJump("watch").clip).toBe(FAILURE.clip);
    expect(storyJump("compare").frame).toBe(FAILURE.compareFrom);
    expect(storyJump("evidence")).toEqual({});
  });
  it("formatters", () => {
    expect(ms(null)).toBe("—");
    expect(timecode(30, 30)).toBe("00:01.000");
    expect(delta(1, 2, 1, true)).toEqual({ text: "-1.0", good: true, zero: false });
  });
});

describe("workstation screens (was test_journey.mjs)", () => {
  it("watch: heading, live beat, privacy note, gate ticker", async () => {
    mount("/dashboard/studio");
    expect(screen.getByTestId("step-watch")).toBeTruthy();
    await screen.findByText(/They're still in the picture/, undefined, { timeout: 5000 });
    expect(screen.getAllByTestId("privacy-note").length).toBeGreaterThan(0);
    expect(screen.getByTestId("gate-ticker")).toBeTruthy();
    expect(document.querySelectorAll("canvas").length).toBeGreaterThanOrEqual(1);
  });
  it("compare: two canvases and the jump to the switch", async () => {
    mount("/dashboard/compare");
    await waitFor(() => expect(document.querySelectorAll("canvas").length).toBeGreaterThanOrEqual(2), { timeout: 5000 });
    await act(async () => { fireEvent.click(screen.getByTestId("jump-switch")); });
    await screen.findAllByText(/ID switch/, undefined, { timeout: 5000 });
  });
  it("evidence: gate flows, method and papers", async () => {
    mount("/dashboard/evidence");
    expect(await screen.findByTestId("gate-flows")).toBeTruthy();
    expect(screen.getByText("Same person, same number")).toBeTruthy();
    expect(screen.getByTestId("method").textContent).toContain("Delayed assignment");
    expect(screen.getByTestId("papers")).toBeTruthy();
  });
  it("experiments: drop-your-video panel", async () => {
    mount("/dashboard/experiments");
    await screen.findByText(/Drop your video/, undefined, { timeout: 5000 });
  });
  it("step header navigates between screens (was test_sidebar.mjs)", async () => {
    mount("/dashboard/studio");
    await act(async () => { fireEvent.click(screen.getByTestId("step-evidence")); });
    expect(await screen.findByTestId("gate-flows")).toBeTruthy();
    await act(async () => { fireEvent.click(screen.getByTestId("step-try")); });
    await screen.findByText(/Drop your video/);
  });
});
