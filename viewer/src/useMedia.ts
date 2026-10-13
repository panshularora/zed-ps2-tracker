import { useEffect, useMemo, useRef, useState } from "react";
import type { IndexedClip } from "./data";
import { VideoClock, clipToVideoFrame, framePath, type MediaSpec } from "./media";
import type { FrameImage } from "./Panel";

/** Media picked at runtime: one MP4, or JPEG files named with their video frame number. */
export interface PickedMedia {
  video?: string; // object URL
  frames?: Map<number, string>; // video frame -> object URL
  /** clip frame f shows video frame f + frame_offset (overrides clip.media.frame_offset). */
  frame_offset?: number;
}

export function pickMedia(files: File[]): PickedMedia {
  const vid = files.find((f) => f.type.startsWith("video/") || /\.(mp4|webm|mov)$/i.test(f.name));
  if (vid) return { video: URL.createObjectURL(vid) };
  const frames = new Map<number, string>();
  for (const f of files) {
    const m = /(\d+)\.(jpe?g|png)$/i.exec(f.name);
    if (m) frames.set(Number(m[1]), URL.createObjectURL(f));
  }
  return frames.size ? { frames } : {};
}

interface MediaState {
  image: FrameImage;
  tick: number;
  videoRef: React.RefObject<HTMLVideoElement>;
  videoSrc: string | null;
  clock: VideoClock | null;
  error: string | null;
  kind: "video" | "frames" | "blank";
  onVideoError: () => void;
}

/**
 * Real frames for a clip. Priority: runtime pick, then `clip.media`, else the blank canvas.
 * `onClockFrame` receives frames presented by the video while it plays (rVFC) or after seeks.
 */
export function useMedia(
  clip: IndexedClip | undefined,
  frame: number,
  picked: PickedMedia,
  onClockFrame: (f: number) => void,
): MediaState {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [tick, setTick] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [clock, setClock] = useState<VideoClock | null>(null);
  const spec: MediaSpec | null = clip?.media ?? null;
  const offset = picked.frame_offset ?? spec?.frame_offset ?? 0;
  const videoSrc = picked.video ?? (spec?.type === "video" ? spec.src : null);
  const framesPattern = !picked.video && !picked.frames && spec?.type === "frames" ? spec.src : null;
  const kind = error ? "blank" : videoSrc ? "video" : picked.frames || framesPattern ? "frames" : "blank";

  // video clock: rVFC while playing, `seeked` while paused
  const cb = useRef(onClockFrame);
  cb.current = onClockFrame;
  useEffect(() => {
    const v = videoRef.current;
    if (!clip || !v || kind !== "video") return;
    const c = new VideoClock(v, clip.fps, offset, (f) => {
      cb.current(f);
      setTick((t) => t + 1);
    });
    setClock(c);
    return () => {
      c.dispose();
      setClock(null);
    };
  }, [clip, kind, offset, videoSrc]);

  // JPEG frames: load the current one (and keep a small cache)
  const cache = useRef(new Map<string, HTMLImageElement>());
  const url = useMemo(() => {
    if (kind !== "frames" || !clip) return null;
    const v = clipToVideoFrame(frame, offset);
    return picked.frames ? (picked.frames.get(v) ?? null) : framesPattern && framePath(framesPattern, v);
  }, [kind, clip, frame, offset, picked.frames, framesPattern]);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) {
      setImg(null);
      return;
    }
    let im = cache.current.get(url);
    if (!im) {
      im = new Image();
      im.src = url;
      cache.current.set(url, im);
      if (cache.current.size > 64) cache.current.delete(cache.current.keys().next().value!);
    }
    const target = im;
    if (target.complete) setImg(target);
    else target.onload = () => {
      setImg(target);
      setTick((t) => t + 1);
    };
  }, [url]);

  const image: FrameImage = kind === "video" ? videoRef.current : kind === "frames" ? img : null;
  return {
    image, tick, videoRef, videoSrc: kind === "video" ? videoSrc : null, clock, error, kind,
    onVideoError: () => setError(`could not load video ${videoSrc}; showing the blank canvas`),
  };
}
