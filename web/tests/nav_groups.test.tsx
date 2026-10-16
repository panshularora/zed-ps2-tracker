import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import App from "../src/App";

describe("nav", () => {
  it("is a flat Dashboard list with no Story / Try it / Evidence groups", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    render(<MemoryRouter initialEntries={["/dashboard/results"]}><App /></MemoryRouter>);
    expect(screen.queryByTestId("nav-group")).toBeNull();
    expect(screen.queryByRole("link", { name: "Try it" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Evidence" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Story" })).toBeNull();
    expect(screen.getByRole("link", { name: "Dashboard" }).getAttribute("href")).toBe("/");
    expect(screen.getByRole("link", { name: "Overview" }).getAttribute("href")).toBe("/dashboard");
    expect(screen.getByRole("link", { name: "Gate scene" }).getAttribute("href")).toBe("/dashboard/live");
    vi.unstubAllGlobals();
  });
});
