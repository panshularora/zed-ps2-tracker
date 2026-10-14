/** One preselected failure the demo is built around. Both saved trackers switch here. */
export const FAILURE = {
  clip: "MOT17-09",
  left: "botsort",
  right: "bytetrack",
  /** Reappearance as a new ID. BoT-SORT at 69, ByteTrack at 70. */
  frame: 70,
  /** Last stretch while they are still on screen as #2 (track ends at 33). */
  lookFrom: 24,
  /** Watch starts at the beginning so the crowd is on screen for a while. */
  playFrom: 1,
  /** Hold this many seconds on the first frame of gone and back. */
  dwellSec: 2.5,
  /** Default playback is slow enough to read the captions. */
  playSpeed: 0.25,
  /** First frame after #2 has left. */
  goneFrom: 34,
  /** BoT-SORT puts them back as #10. */
  backFrom: 69,
  /** Compare starts here so you see the empty gap, then the return. */
  compareFrom: 58,
  seedId: 2,
  gt: 8,
  from: 2,
  to: 10,
} as const;

export type StoryPhase = "here" | "gone" | "back";

export function storyBeat(frame: number): {
  phase: StoryPhase;
  step: string;
  title: string;
  now: string;
  why: string;
} {
  if (frame < FAILURE.goneFrom) {
    return {
      phase: "here",
      step: "1 / 3",
      title: `This person is #${FAILURE.from}`,
      now: `They're still in the picture as number ${FAILURE.from}.`,
      why: `LOOK is #${FAILURE.from}. When the hatched region shows two people overlapping, that number must stay #${FAILURE.from}. Overlap is not a new ID.`,
    };
  }
  if (frame < FAILURE.backFrom) {
    return {
      phase: "gone",
      step: "2 / 3",
      title: `#${FAILURE.from} walked away. Tracker forgot them.`,
      now: "They're gone. The tracker has no box for them.",
      why: `Dashed red is the last tracker box for #${FAILURE.from}. Green keep is ground-truth ${FAILURE.gt} — same person, same ID. The tracker should have held #${FAILURE.from}.`,
    };
  }
  return {
    phase: "back",
    step: "3 / 3",
    title: `Same person came back as #${FAILURE.to}`,
    now: `They're back, but the tracker calls them number ${FAILURE.to}.`,
    why: `Preserving identity means they stay #${FAILURE.from} (GT ${FAILURE.gt}). The tracker issued #${FAILURE.to}. That new number is the failure, not the overlap.`,
  };
}

export type StepId = "watch" | "compare" | "evidence" | "try";

export const STEPS: {
  id: StepId;
  n: string;
  label: string;
  short: string;
}[] = [
  { id: "watch", n: "1", label: "Watch the failure", short: "Start here" },
  { id: "compare", n: "2", label: "Compare the trackers", short: "Side by side" },
  { id: "evidence", n: "3", label: "See the evidence", short: "Scores" },
  { id: "try", n: "4", label: "Your video", short: "Last" },
];

export function prettyTracker(name: string): { title: string; why: string } {
  const n = name.toLowerCase();
  if (n.includes("botsort") && (n.includes("vis") || n.includes("reid"))) {
    return { title: "BoT-SORT + vis-gate", why: "re-ID with a visibility brake" };
  }
  if (n.includes("botsort")) {
    return { title: "BoT-SORT", why: "the one we run — also corrects for camera shake" };
  }
  if (n.includes("vis")) {
    return { title: "ByteTrack + vis-gate", why: "re-ID only while visible" };
  }
  if (n.includes("reid")) {
    return { title: "ByteTrack + re-ID", why: "appearance, no visibility brake" };
  }
  if (n.includes("byte")) {
    return { title: "ByteTrack", why: "simpler baseline — overlapping boxes only" };
  }
  return { title: name, why: "saved run" };
}

/** Public-FRCNN MOT17 val-half names as stored in ablation.json. */
export const EVIDENCE_ROWS = ["bytetrack", "botsort", "ours_vis_botsort"] as const;

export const GAP_RESUME = {
  source:
    "On MOT17, after someone disappears for at least 5 frames (YOLOX detections): how often do they come back with the same number?",
  short: {
    bin: "5–14 frames gone",
    rows: [
      { name: "ByteTrack", pct: 68.0 },
      { name: "BoT-SORT", pct: 71.6, best: true },
      { name: "vis-gated", pct: 63.8 },
    ],
  },
  long: {
    bin: "30+ frames gone",
    rows: [
      { name: "ByteTrack", pct: 10.8 },
      { name: "vis-gated", pct: 30.6, best: true },
    ],
    note: "The jump on long gaps is mostly because we keep lost tracks around longer, not because visibility can see through a crowd. BoT-SORT was not scored in this long-gap probe.",
  },
} as const;

/** Why identity at a crowded gate matters (PS-2 scene, not a product name). */
export const GATE_FLOWS = [
  {
    flow: "Same person, same number",
    need: "The same person keeps the same number from approach to exit.",
    break: "After they were hidden, they get a new number, so anything tied to the old number is now on the wrong person.",
    metric: "Wrong person on a matched crossing",
    public: "14–33% on 25 MOT17 gates. Too few crossings to rank trackers.",
  },
  {
    flow: "Two people close together",
    need: "Two bodies crossing together stay two IDs.",
    break: "If the tracker merges them, the second person disappears. If it splits one person, you get a ghost.",
    metric: "Two people crossing within one second",
    public: "Caught 44% of 9 known close pairs; 80% of flags were real. The sample is small.",
  },
  {
    flow: "Counting people",
    need: "One crossing per person per direction.",
    break: "Split tracks count one person twice. Merged tracks miss someone.",
    metric: "How far the count is from the true crossings",
    public: "28–40% off on the same 25 crossings.",
  },
] as const;

export const PAPERS = [
  {
    title: "ByteTrack",
    used: "First it matches the obvious people, then it tries leftover weak detections. People who already walked out of view do not get that second try.",
  },
  {
    title: "BoT-SORT",
    used: "Same matching, plus it corrects when the camera itself moves. This is the tracker we run.",
  },
  {
    title: "HOTA",
    used: "One score for two jobs: finding people, and keeping the right IDs on them.",
  },
  {
    title: "IDF1",
    used: "How much of each person's walk stays under one number. Higher is better.",
  },
  {
    title: "ID switch",
    used: "The tracker changed the number while the person was still in the scene. That is the clip you watched.",
  },
] as const;

export const METHOD = {
  ship: "We run BoT-SORT. It only looks at the current frame, and it takes about 2 milliseconds per frame on one CPU thread.",
  keep: "We use a visibility estimate only to stop appearance matching when someone is hidden. It is not a way of seeing through a crowd. On public detections it behaves like giving lost people a longer memory.",
  reject: [
    "Delayed assignment (waiting a few frames before giving someone an ID): more switches, and slower. We dropped it.",
    "Using appearance on people who were already lost: more switches in a crowd.",
    "Letting lost people match weak detections: failed our occlusion tests, so we turned it off.",
  ],
} as const;
