import { render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { MISSING_CLIP_TEXT } from "../src/guided/model";
import { pickOpener } from "../src/pages/Scenarios";

vi.mock("../src/pages/Scenarios", async (orig) => {
  const real = await orig<typeof import("../src/pages/Scenarios")>();
  return { ...real, default: () => <p>scenarios page</p> };
});
vi.mock("../src/pages/Overview", () => ({ default: () => <p>overview page</p> }));
vi.mock("../src/pages/Landing", () => ({ default: () => <p>landing</p> }));

const R = { v7_startTransition: true, v7_relativeSplatPath: true };

describe("navigation", () => {
  it("flat Dashboard nav, no Story / Try it / Evidence groups, no Demo tab", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/dashboard"]}><App /></MemoryRouter>);
    const labels = [...document.querySelectorAll("nav[aria-label='Pages'] a")].map((a) => a.textContent);
    expect(labels).toEqual(["Dashboard", "Overview", "What we use", "Gate scene", "Scenarios", "Studio", "Results", "Experiments", "Analytics", "Architecture"]);
    expect(screen.getByText("overview page")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Overview" }).className).toContain("active");
    expect(screen.getByRole("link", { name: "Dashboard" }).className).not.toContain("active");
    expect(screen.queryByRole("link", { name: "Demo" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Try it" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Evidence" })).toBeNull();
  });
  it("keeps / as the Dashboard landing and names the inner page Overview", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/"]}><App /></MemoryRouter>);
    expect(screen.getByText("landing")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Dashboard" }).className).toContain("active");
    expect(screen.getByRole("link", { name: "Overview" }).getAttribute("href")).toBe("/dashboard");
  });
  it("redirects the old Demo route to Scenarios", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/dashboard/demo"]}><App /></MemoryRouter>);
    expect(screen.getByText("scenarios page")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Scenarios" }).className).toContain("active");
  });
  it("footer: provisional laptop-GPU summary, Pexels clips not Apache-2.0, nothing pending on a DGX", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/dashboard"]}><App /></MemoryRouter>);
    const f = document.querySelector("footer")!.textContent!;
    expect(f).toContain("provisional");
    expect(f).toContain("YOLOX-X detectors");
    expect(f).not.toContain("public detections");
    expect(f).toContain("Pexels License (not Apache-2.0)");
    expect(f).not.toMatch(/DGX/);
  });
});

describe("opener and wording", () => {
  const items = [{ id: "subway_turnstile" }, { id: "milan_turnstiles" }] as never;
  it("opens on 'What tracking is' (scenario order), MOT17-09 only when no clips; waits for the list", () => {
    expect(pickOpener(items, null)).toBeNull();
    expect(pickOpener([{ id: "milan_turnstiles" }, { id: "man_entering_metro" }] as never, [{ name: "MOT17-09" }] as never)).toBe("man_entering_metro");
    expect(pickOpener(items, [])).toBe("subway_turnstile");
    expect(pickOpener([] as never, [{ name: "MOT17-09" }] as never)).toBe("bench:MOT17-09");
  });
  it("says 'in the repo folder', never 'on the server'", () => {
    expect(MISSING_CLIP_TEXT).toContain("In the repo folder run make clips blur");
    expect(MISSING_CLIP_TEXT).not.toContain("server");
  });
  it("lists every Pexels clip source in the clip licence file", () => {
    const md = readFileSync("public/demo/LICENSE_CLIPS.md", "utf8").replace(/\r\n/g, "\n");
    const idx = JSON.parse(readFileSync("public/scenarios/index.json", "utf8")) as { id: string; source: string }[];
    expect(md).toContain("not** covered by\nthis repository's Apache-2.0");
    expect(md).toMatch(/Do not reuse them as stock/);
    for (const s of idx) { expect(md).toContain(`${s.id}.mp4`); expect(md).toContain(s.source); }
  });
});

