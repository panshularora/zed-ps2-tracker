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
  it("four items: Story, What we use, Try it, Results; Try it has Gate scene first", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/dashboard/live"]}><App /></MemoryRouter>);
    const labels = [...document.querySelectorAll("nav[aria-label='Pages'] a")].map((a) => a.textContent);
    expect(labels).toEqual(["Story", "What we use", "Try it", "Results"]);
    const subs = [...document.querySelectorAll("nav[aria-label='Sections'] a")].map((a) => a.textContent);
    expect(subs).toEqual(["Gate scene", "Same person, two trackers", "Studio", "Scenarios"]);
    expect(screen.getByRole("link", { name: "Try it" }).className).toContain("active");
    expect(screen.getByTestId("next-btn").getAttribute("href")).toBe("/dashboard/compare");
  });
  it("Story at / is the 'What is an ID mix-up?' page, Next goes to What we use", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/"]}><App /></MemoryRouter>);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("What is an ID mix-up?");
    expect(screen.getByTestId("story-status").textContent).toContain("Pending GPU rerun");
    expect(screen.getByRole("link", { name: "Story" }).className).toContain("active");
    expect(screen.getByTestId("next-btn").getAttribute("href")).toBe("/dashboard/what-we-use");
  });
  it("redirects the old Demo route to Scenarios", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/dashboard/demo"]}><App /></MemoryRouter>);
    expect(screen.getByText("scenarios page")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Scenarios" }).className).toContain("active");
    expect(screen.getByRole("link", { name: "Try it" }).className).toContain("active");
  });
  it("footer: provisional laptop-GPU summary, Pexels clips not Apache-2.0, nothing pending on a DGX", async () => {
    const { default: App } = await import("../src/App");
    render(<MemoryRouter future={R} initialEntries={["/dashboard"]}><App /></MemoryRouter>);
    const f = document.querySelector("footer")!.textContent!;
    expect(f).toContain("provisional");
    expect(f).toContain("summary.json (laptop GPU run)");
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

