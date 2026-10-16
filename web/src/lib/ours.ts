// One name and one line for "ours", used on every page. Confirmed in code: the "ours" run on the
// Pexels clips, the MOT17 benchmark rows and the gate scene is configs/ours/visibility_botsort.yaml
// (BoT-SORT + the depth-order visibility estimate; serve/gate_sim.py, precompute_scenarios.py).
export const OURS_NAME = "Ours: BoT-SORT + visibility";
export const OURS_SHORT = "Ours";
export const OURS_LINE = "BoT-SORT plus a visibility rule: someone last seen half-hidden behind another person is remembered 3x longer before their number is dropped.";
export const BASE_NAME = "ByteTrack (baseline)";
export const BASE_SHORT = "ByteTrack";
export const BASE_LINE = "The published tracker we compare against: forgets a lost person after 30 frames (1 s at 30 fps).";
export const KEPT = "#56B4E9";
export const CHANGED = "#F5A524";
export const GREY = "#8A929C";
export const INK = "#0E1013";
