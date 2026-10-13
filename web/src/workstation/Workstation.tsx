// Panshul's MOT research workstation (was dashboard/, branch feat/dashboard-workstation), mounted in
// the web app under /dashboard/{studio,compare,evidence,experiments}. Routing moved from the
// #/hash router to react-router; screens, journey/story and identity overlays are his.
import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { IndexedClip } from "@viewer/data";
import type { PickedMedia } from "@viewer/useMedia";
import { fetchClip, fetchClips, fetchWorkstation, type ClipRef, type Screen } from "./api";
import { Analytics } from "./screens/Analytics";
import { Experiments } from "./screens/Experiments";
import { Studio } from "./screens/Studio";
import { Header } from "./components/Header";
import { FAILURE } from "./lib/journey";
import "./workstation.css";

export interface Jump {
  clip?: string;
  left?: string;
  right?: string;
  frame?: number | null;
  trackId?: number;
}

export const SCREEN_PATH: Record<Screen, string> = {
  watch: "/dashboard/studio",
  compare: "/dashboard/compare",
  evidence: "/dashboard/evidence",
  try: "/dashboard/experiments",
};

export function storyJump(screen: Screen, extra?: Jump): Jump {
  if (screen === "watch" || screen === "compare") {
    return {
      clip: FAILURE.clip,
      left: FAILURE.left,
      right: FAILURE.right,
      frame: screen === "watch" ? FAILURE.lookFrom : FAILURE.compareFrom,
      trackId: FAILURE.seedId,
      ...extra,
    };
  }
  return extra ?? {};
}

export function jumpFromSearch(search: string): Jump | undefined {
  const p = new URLSearchParams(search);
  if (!p.get("clip")) return undefined;
  return {
    clip: p.get("clip") ?? undefined,
    left: p.get("left") ?? undefined,
    right: p.get("right") ?? undefined,
    frame: p.get("f") ? Number(p.get("f")) : undefined,
    trackId: p.get("id") ? Number(p.get("id")) : undefined,
  };
}

export function jumpToSearch(j: Jump): string {
  const p = new URLSearchParams();
  if (j.clip) p.set("clip", j.clip);
  if (j.left) p.set("left", j.left);
  if (j.right) p.set("right", j.right);
  if (j.frame) p.set("f", String(j.frame));
  if (j.trackId) p.set("id", String(j.trackId));
  const q = p.toString();
  return q ? `?${q}` : "";
}

export default function Workstation({ screen }: { screen: Screen }) {
  const location = useLocation();
  const navigate = useNavigate();
  const jump = useMemo(() => storyJump(screen, jumpFromSearch(location.search)), [screen, location.search]);
  const [refs, setRefs] = useState<ClipRef[]>([]);
  const [clips, setClips] = useState<IndexedClip[]>([]);
  const [picked, setPicked] = useState<PickedMedia>({});
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("connecting");

  useEffect(() => {
    fetchWorkstation()
      .then((d) => setStatus(`v${d.version} · ${d.clips} clips`))
      .catch(() => setStatus("API offline — start `make serve`"));
    fetchClips()
      .then(async (list) => {
        setRefs(list);
        setClips(await Promise.all(list.map((c) => fetchClip(c.file))));
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const go = useCallback(
    (next: Screen, extra?: Jump) => navigate(SCREEN_PATH[next] + jumpToSearch(storyJump(next, extra))),
    [navigate],
  );

  const addClip = useCallback((clip: IndexedClip) => {
    setClips((cs) => [...cs.filter((c) => !(c.clip === clip.clip && c.tracker === clip.tracker)), clip]);
  }, []);

  return (
    <div className="ws h-full flex flex-col bg-page text-ink font-sans text-xs antialiased" data-testid="workstation">
      <Header screen={screen} onScreen={(s) => go(s)} status={status} />
      {error && (
        <div className="px-4 py-2 bg-fail/15 text-fail border-b border-fail/40 font-mono">{error}</div>
      )}
      <div className="flex-1 min-h-0">
        {(screen === "watch" || screen === "compare") && (
          <Studio
            clips={clips}
            jump={jump}
            picked={picked}
            onPicked={setPicked}
            mode={screen}
            onNext={() => go(screen === "watch" ? "compare" : "evidence")}
          />
        )}
        {screen === "evidence" && <Analytics clips={clips} onInspect={(j) => go("compare", j)} />}
        {screen === "try" && (
          <Experiments
            refs={refs}
            clips={clips}
            picked={picked}
            onPicked={setPicked}
            onOpenStudio={(j) => go("watch", j)}
            onClip={addClip}
          />
        )}
      </div>
    </div>
  );
}
