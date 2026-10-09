// Product theme = the viewer's tokens (viewer/src/theme.ts, also used by the deck) + web additions.
import { ID_COLORS, idColor, theme as viewerTheme } from "@viewer/theme";

export { ID_COLORS, idColor };

export const theme = {
  ...viewerTheme,
  ours: "#56B4E9",
  baseline: "#8A929C",
  trailWidth: 1.5,
  ringMs: 300,
  monoFont: "'JetBrains Mono', ui-monospace, monospace",
} as const;
