// Plain-English intros per scenario: what you'll see, what to watch for, the takeaway.
// Prose only; every number on screen comes from the tracker output, not from here.
export interface Intro { see: string; watch: string; takeaway: string }

export const SCENARIO_ORDER = ["man_entering_metro", "subway_turnstile", "milan_turnstiles", "mexico_city_entrance", "rabat_stadium", "shibuya_crossing"];

export const INTROS: Record<string, Intro> = {
  man_entering_metro: {
    see: "One man walking into a metro station. The tracker draws a box around him and gives him a number.",
    watch: "Whether the number stays the same for the whole walk. The timeline below shows it honestly, including any change.",
    takeaway: "Tracking = one box and one number per person, frame after frame. Counting at a gate depends on that number not changing.",
  },
  subway_turnstile: {
    see: "Commuters passing subway turnstiles, often walking behind each other.",
    watch: "The highlighted person: their box goes dashed while they are hidden, then they come back with the same number.",
    takeaway: "Keeping the number through a short hide is what stops one person being counted twice at the gate.",
  },
  milan_turnstiles: {
    see: "People bunching at a barrier, close enough that boxes touch.",
    watch: "Neighbouring boxes: each person should keep their own number when they bunch up.",
    takeaway: "Close neighbours are where numbers get mixed up; the timeline shows any change.",
  },
  mexico_city_entrance: {
    see: "A busy entrance where people pass in front of each other.",
    watch: "The highlighted person disappearing behind others and returning.",
    takeaway: "Short occlusions are normal; the goal is the same number before and after.",
  },
  rabat_stadium: {
    see: "A stadium fast-lane entrance with steady foot traffic.",
    watch: "People crossing the amber gate line and the IN / OUT bar above the video.",
    takeaway: "Each crossing should add exactly one; a changed number could add a second.",
  },
  shibuya_crossing: {
    see: "A very dense crossing with many similar-looking people. This is the hardest clip.",
    watch: "The timeline: dense crowds are where our tracker still changes numbers, and we show every change.",
    takeaway: "This is what we are working to improve; the numbers here are not solved.",
  },
  "MOT17-09": {
    see: "A real street video from the MOT17 benchmark, labelled by humans. The video frames are not licensed for this repo, so you see only the boxes on an empty frame.",
    watch: "Each rectangle is one pedestrian as our tracker saw them. Follow the highlighted person: dashed means hidden, and the timeline shows whether their number changed.",
    takeaway: "Because humans labelled this clip, ID changes here are measured, not guessed. On this clip the right tracker is plain BoT-SORT (our visibility variant was not exported for it).",
  },
};
