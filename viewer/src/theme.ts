// Viewer theme (3D Art Director spec, 06_phase2_specs.md). Shared with the deck: change here only.
export const ID_COLORS = [
  "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8DD3C7",
  "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#BC80BD",
  "#CCEBC5", "#FFED6F", "#1B9E77", "#A6CEE3",
] as const;

export const theme = {
  background: "#0E1013",
  panel: "#161A1F",
  panelBorder: "#262B33",
  text: "#E6E8EB",
  textSecondary: "#8A929C",
  labelText: "#0E1013",
  held: "rgba(255, 255, 255, 0.8)", // #FFFFFF at 80 %
  heldDash: [6, 4] as number[],
  error: "#E5484D",
  success: "#3dba7c",
  gate: "#F5A524",
  gateWidth: 2,
  boxWidth: 2,
  labelFont: "11px 'JetBrains Mono', ui-monospace, monospace",
  uiFont: "Inter, system-ui, sans-serif",
  radius: 4,
  decisionFadeMs: 150,
  gatePulseMs: 300,
} as const;

export function idColor(id: number): string {
  return ID_COLORS[((id % 20) + 20) % 20];
}
