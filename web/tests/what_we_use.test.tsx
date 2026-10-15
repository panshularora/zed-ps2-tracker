import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import WhatWeUse from "../src/pages/WhatWeUse";

describe("What we use", () => {
  it("lists datasets with licence and run status from summary.json, models, metrics, pipeline and the comparison table", () => {
    render(<MemoryRouter><WhatWeUse /></MemoryRouter>);
    for (const id of ["mot17", "mot20", "dancetrack", "pexels"]) expect(screen.getByTestId(`wwu-ds-${id}`)).toBeTruthy();
    expect(screen.getByTestId("wwu-ds-mot17").textContent).toContain("CC BY-NC-SA 3.0");
    expect(screen.getByTestId("wwu-ds-dancetrack").textContent).toContain("not run yet");
    expect(screen.getByTestId("wwu-ds-mot20").textContent).toContain("leaked, excluded");
    expect(screen.getByTestId("wwu-models").textContent).toMatch(/YOLOX-X.*ByteTrack.*BoT-SORT.*visibility/s);
    expect(screen.getByTestId("wwu-models").textContent).toContain("switched off in every result");
    expect(screen.getByTestId("wwu-metrics").textContent).toMatch(/HOTA.*IDF1.*IDSW.*MOTA/s);
    expect(screen.getAllByRole("listitem").length).toBeGreaterThanOrEqual(5);
    expect(screen.getByTestId("dataset-table")).toBeTruthy();
  });
});
