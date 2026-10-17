// Small stills for the suggestion rows, grabbed in the browser from the face-blurred clip the
// service serves (the original is never served, so neither are these). One hidden <video>, one
// seek at a time.
import { useEffect, useState } from "react";
import type { XYWH } from "./fromJob";

/** rects are in the job's frame pixels (srcW x srcH); w x h is the CSS size of each still. */
export interface ThumbReq { key: string; t: number; srcW: number; srcH: number; rects: { r: XYWH; w: number; h: number }[] }
/** Returns one data URL per rect, or null if the clip cannot be read. */
export type Grabber = (videoUrl: string, req: ThumbReq) => Promise<string[] | null>;

let shared: { url: string; v: HTMLVideoElement; busy: Promise<unknown> } | null = null;

export const videoGrabber: Grabber = async (videoUrl, req) => {
  if (typeof document === "undefined") return null;
  if (!shared || shared.url !== videoUrl) {
    const v = document.createElement("video");
    v.muted = true; v.preload = "auto"; v.playsInline = true; v.src = videoUrl;
    shared = { url: videoUrl, v, busy: Promise.resolve() };
  }
  const s = shared;
  const run = async () => {
    const v = s.v;
    if (v.readyState < 1) await new Promise((ok, bad) => { v.addEventListener("loadedmetadata", ok, { once: true }); v.addEventListener("error", bad, { once: true }); });
    await new Promise((ok) => { v.addEventListener("seeked", ok, { once: true }); v.currentTime = Math.min(Math.max(0, req.t), Math.max(0, (v.duration || req.t) - 0.05)); });
    return req.rects.map(({ r, w, h }) => {
      const c = document.createElement("canvas");
      c.width = w * 2; c.height = h * 2; // 2x for sharp small images
      const ctx = c.getContext("2d");
      if (!ctx) return "";
      ctx.fillStyle = "#0E1013";
      ctx.fillRect(0, 0, c.width, c.height);
      // fit the rect into w x h, centred
      const sx = (v.videoWidth || req.srcW) / req.srcW, sy = (v.videoHeight || req.srcH) / req.srcH;
      const k = Math.min(c.width / r[2], c.height / r[3]);
      const dw = r[2] * k, dh = r[3] * k;
      ctx.drawImage(v, r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy, (c.width - dw) / 2, (c.height - dh) / 2, dw, dh);
      return c.toDataURL("image/jpeg", 0.8);
    });
  };
  const p = s.busy.then(run, run);
  s.busy = p.catch(() => undefined);
  try { return await p; } catch { return null; }
};

/** Grab every request once (in order); returns key -> data URLs as they arrive. */
export function useThumbs(videoUrl: string | null, reqs: ThumbReq[], grab: Grabber = videoGrabber): Record<string, string[]> {
  const [out, setOut] = useState<Record<string, string[]>>({});
  const sig = reqs.map((r) => r.key).join("|");
  useEffect(() => {
    if (!videoUrl) return;
    let live = true;
    (async () => {
      for (const r of reqs) {
        const urls = await grab(videoUrl, r);
        if (!live) return;
        if (urls) setOut((o) => ({ ...o, [r.key]: urls }));
      }
    })();
    return () => { live = false; };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- requests are identified by their keys
  }, [videoUrl, sig, grab]);
  return out;
}
