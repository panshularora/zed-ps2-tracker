// Studio step 4 · Your video: only the upload flow and its result in the Studio player. No config,
// eval log or research tools here (those stay on the Experiments page).
import { YourVideo } from "./YourVideo";

export function YourVideoScreen() {
  return (
    <div className="h-full overflow-y-auto p-4 space-y-4" data-testid="your-video-screen">
      <header className="max-w-3xl space-y-1">
        <div className="font-display uppercase tracking-[0.18em] text-[11px] text-signal">Step 4 · Your video</div>
        <h1 className="font-display text-[26px] leading-none text-ink font-semibold">Your video</h1>
        <p className="text-[13px] text-mute" data-testid="your-video-intro">Upload a clip, draw your gate, and see what the tracker counts.</p>
        <details className="yv-what text-[13px] text-mute" data-testid="your-video-what">
          <summary className="cursor-pointer text-ink">What am I looking at?</summary>
          <p className="mt-2">
            Draw your gate on the clip's first frame, and our tracker (BoT-SORT + visibility) runs on
            it here on the CPU. You get the same player as the Studio: boxes and numbers, IN/OUT at
            your line, a timeline per number, and the moments the tracker was unsure. Your clip has
            no ground truth, so we show counts, IDs and events, never a score.
          </p>
        </details>
      </header>
      <YourVideo />
    </div>
  );
}
