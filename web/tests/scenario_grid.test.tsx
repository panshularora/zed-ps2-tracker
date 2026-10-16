import { readFileSync, existsSync } from "node:fs";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import Scenarios from "../src/pages/Scenarios";

describe("Scenarios grid", () => {
  it("opens on a thumbnail grid; clicking a card opens the large player, back returns", async () => {
    vi.stubGlobal("fetch", vi.fn(async (u: string) => {
      const f = `public${String(u).split("?")[0]}`;
      return existsSync(f) ? new Response(readFileSync(f)) : new Response("nf", { status: 404 });
    }));
    render(<MemoryRouter><Scenarios /></MemoryRouter>);
    await waitFor(() => expect(screen.getAllByTestId("scenario-card").length).toBeGreaterThanOrEqual(6));
    expect(screen.getByTestId("scenario-grid")).toBeTruthy();
    expect(screen.queryByTestId("scenario-player")).toBeNull();
    await act(async () => { fireEvent.click(screen.getAllByTestId("scenario-card")[0]); });
    await waitFor(() => expect(screen.getByTestId("scenario-player")).toBeTruthy());
    await act(async () => { fireEvent.click(screen.getByTestId("back-to-grid")); });
    expect(screen.getByTestId("scenario-grid")).toBeTruthy();
    vi.unstubAllGlobals();
  });
});
