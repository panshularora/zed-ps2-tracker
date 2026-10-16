import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import App, { PATH, nextOf } from "../src/App";

describe("nav", () => {
  it("Results carries Experiments, Analytics, Architecture as sub-tabs", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    render(<MemoryRouter initialEntries={["/dashboard/results"]}><App /></MemoryRouter>);
    const subs = [...document.querySelectorAll("nav[aria-label='Sections'] a")].map((a) => a.textContent);
    expect(subs).toEqual(["Comparison", "Experiments", "Analytics", "Architecture"]);
    expect(screen.getByRole("link", { name: "Results" }).className).toContain("active");
    vi.unstubAllGlobals();
  });
  it("every page but the last has a Next on the one path", () => {
    for (let i = 0; i < PATH.length - 1; i++) expect(nextOf(PATH[i].to)).toEqual(PATH[i + 1]);
    expect(nextOf(PATH[PATH.length - 1].to)).toBeNull();
    expect(nextOf("/dashboard")).toEqual(PATH[1]);
  });
});
