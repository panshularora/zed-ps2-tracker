// Real frames under the boxes: an MP4 through one shared <video> (the clock), or JPEG frames.
// Frame mapping (clip frames are 1-based; `offset` = video frame of clip frame 1, minus 1):
//   while playing, from requestVideoFrameCallback: frame = round(meta.mediaTime * fps) + 1
//   when paused (or without rVFC, e.g. Firefox):  frame = floor(currentTime * fps) + 1
//   to show frame f:                               currentTime = (f - 1 + 0.5) / fps
// No `timeupdate`, and no rAF loop reading currentTime.

export interface MediaSpec {
  type: "video" | "frames";
  /** video: URL of the MP4; frames: URL pattern with %06d (or %0Nd) for the video frame index */
  src: string;
  /** clip frame f shows video frame f + frame_offset (val-half clips start mid-sequence) */
  frame_offset?: number;
}

/** Video frame index (1-based) presented at mediaTime, from requestVideoFrameCallback. */
export function frameFromMediaTime(mediaTime: number, fps: number): number {
  return Math.round(mediaTime * fps) + 1;
}

/** Video frame index (1-based) shown at currentTime when paused / without rVFC. */
export function frameFromCurrentTime(currentTime: number, fps: number): number {
  return Math.floor(currentTime * fps + 1e-6) + 1;
}

/** currentTime that lands in the middle of video frame f (robust to rounding at either edge). */
export function seekTime(videoFrame: number, fps: number): number {
  return (videoFrame - 1 + 0.5) / fps;
}

export function clipToVideoFrame(f: number, offset = 0): number {
  return f + offset;
}

export function videoToClipFrame(v: number, offset = 0): number {
  return v - offset;
}

/** Fill a %06d-style pattern (and only the first one) with a frame index. */
export function framePath(pattern: string, videoFrame: number): string {
  return pattern.replace(/%0?(\d*)d/, (_m, w: string) =>
    String(videoFrame).padStart(w ? Number(w) : 0, "0"),
  );
}

/** The part of HTMLVideoElement the clock uses (so tests can pass a fake). */
export interface VideoLike {
  currentTime: number;
  paused: boolean;
  duration?: number;
  play(): Promise<void> | void;
  pause(): void;
  addEventListener(type: "seeked" | "pause" | "ended", cb: () => void): void;
  removeEventListener(type: "seeked" | "pause" | "ended", cb: () => void): void;
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
}

/**
 * Drives the app's frame from a video. With rVFC (Chrome, Safari, Edge) playback is the video's
 * own; without it (Firefox) the video stays paused and the app steps frames itself, seeking the
 * video to each one, which is frame-accurate.
 */
export class VideoClock {
  readonly hasRvfc: boolean;
  private handle: number | null = null;
  /** last clip frame presented through rVFC (what is on screen while playing) */
  private last: number | null = null;
  private readonly onSeeked = () => this.emitFromCurrentTime();

  constructor(
    private readonly video: VideoLike,
    private readonly fps: number,
    private readonly offset: number,
    private readonly onFrame: (clipFrame: number) => void,
  ) {
    this.hasRvfc = typeof video.requestVideoFrameCallback === "function";
    video.addEventListener("seeked", this.onSeeked);
  }

  /** Frame for the current paused position (also used after every seek). */
  emitFromCurrentTime(): void {
    let v = frameFromCurrentTime(this.video.currentTime, this.fps);
    const d = this.video.duration;
    if (d !== undefined && Number.isFinite(d) && d > 0) v = Math.min(v, Math.round(d * this.fps));
    this.onFrame(videoToClipFrame(v, this.offset));
  }

  seek(clipFrame: number): void {
    this.video.currentTime = seekTime(clipToVideoFrame(clipFrame, this.offset), this.fps);
  }

  /** Start playback. Returns false when the caller must step frames itself (no rVFC). */
  play(): boolean {
    if (!this.hasRvfc) return false;
    const tick = (_now: number, meta: { mediaTime: number }) => {
      this.last = videoToClipFrame(frameFromMediaTime(meta.mediaTime, this.fps), this.offset);
      this.onFrame(this.last);
      if (!this.video.paused) this.handle = this.video.requestVideoFrameCallback!(tick);
    };
    this.handle = this.video.requestVideoFrameCallback!(tick);
    void this.video.play();
    return true;
  }

  /** Pause, then snap to the centre of the frame last shown, so the counter and the pixels agree
   * (currentTime right after pause() can sit just before that frame's timestamp). */
  pause(): void {
    if (this.handle !== null) this.video.cancelVideoFrameCallback?.(this.handle);
    this.handle = null;
    const wasPlaying = !this.video.paused;
    this.video.pause();
    if (wasPlaying && this.last !== null) this.seek(this.last);
  }

  dispose(): void {
    this.pause();
    this.video.removeEventListener("seeked", this.onSeeked);
  }
}

/**
 * 150 ms decision fade: after a held window closes at f1, the box goes from dashed white to its
 * solid ID colour over `ms` of video time. Returns 0 (still held) .. 1 (fully committed).
 */
export function fadeProgress(frame: number, f1: number, fps: number, ms: number): number {
  if (frame <= f1) return 0;
  return Math.min(1, ((frame - f1) * 1000) / fps / ms);
}

/** Mix two #RRGGBB colours (t = 0 → a, 1 → b) as an rgba() string with the given alpha. */
export function mix(a: string, b: string, t: number, alpha = 1): string {
  const p = (h: string, i: number) => parseInt(h.slice(1 + 2 * i, 3 + 2 * i), 16);
  const c = [0, 1, 2].map((i) => Math.round(p(a, i) + (p(b, i) - p(a, i)) * t));
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}
