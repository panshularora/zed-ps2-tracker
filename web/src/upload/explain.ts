// Turn a backend / network failure into one calm sentence plus the exact command that fixes it.
// The raw text is kept for the "Details" toggle; it is never the headline.
import { ApiError } from "../api";

export interface Explained { cause: string; fix: string | null; raw: string }

export const MODELS_FIX = "python -m serve.detector fetch --out data/models";
export const FFMPEG_FIX = "winget install Gyan.FFmpeg   (or: pip install imageio-ffmpeg)";
export const SERVE_FIX = "python -m uvicorn serve.app:app --host 127.0.0.1 --port 8000";

/** ``e``: an ApiError, an Error, a job's ``error`` object, or a string. */
export function explainError(e: unknown): Explained {
  let raw = "", code = "", fix: string | null = null, cause: string | null = null;
  if (e instanceof ApiError) {
    raw = e.message; code = e.code; fix = e.fix ?? null;
  } else if (e && typeof e === "object" && "message" in e) {
    const o = e as { message?: unknown; code?: unknown; fix?: unknown; cause?: unknown };
    raw = String(o.message ?? ""); code = typeof o.code === "string" ? o.code : "";
    fix = typeof o.fix === "string" ? o.fix : null;
    cause = typeof o.cause === "string" && !(e instanceof Error) ? o.cause : null;
  } else raw = String(e);
  const t = raw.toLowerCase();
  if (/ffmpeg/.test(t)) return { cause: "ffmpeg is not installed, so the blurred copy of your video cannot be written.", fix: fix ?? FFMPEG_FIX, raw };
  if (code === "model_missing" || /onnx|facedetectoryn|yunet|yolox_\w+\.onnx|model .* missing/.test(t))
    return { cause: code === "model_missing" && cause ? cause : "A detector model file is missing on this computer and could not be downloaded.", fix: fix ?? MODELS_FIX, raw };
  if (code === "network" || /failed to fetch|cannot reach the service|networkerror/.test(t))
    return { cause: "The tracker backend is not running.", fix: SERVE_FIX, raw };
  if (code === "too_large" || code === "too_long" || code === "unsupported_type" || code === "not_a_video") return { cause: raw, fix: null, raw };
  return { cause: cause ?? "The tracker could not finish this video.", fix, raw };
}
