import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import App from "../src/App";

describe("nav group headers", () => {
  it("each group header links to its first page", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    render(<MemoryRouter initialEntries={["/dashboard/results"]}><App /></MemoryRouter>);
    const groups = screen.getAllByTestId("nav-group");
    expect(groups.map((g) => [g.textContent, g.getAttribute("href")])).toEqual([
      ["Story", "/dashboard"],
      ["Try it", "/dashboard/live"],
      ["Evidence", "/dashboard/results"],
    ]);
    vi.unstubAllGlobals();
  });
});
