import { useCallback, useEffect, useState } from "react";
import type { IndexedClip } from "@viewer/data";
import type { PickedMedia } from "@viewer/useMedia";
import {
  fetchClip,
  fetchClips,
  fetchDashboard,
  parseHash,
  writeHash,
  type ClipRef,
  type Screen,
} from "./api";
import { Analytics } from "./screens/Analytics";
import { Experiments } from "./screens/Experiments";
import { Studio } from "./screens/Studio";
import { Header } from "./components/Header";
import { FAILURE } from "./lib/journey";

export interface Jump {
  clip?: string;
  left?: string;
  right?: string;
  frame?: number | null;
  trackId?: number;
}

function storyJump(screen: Screen, extra?: Jump): Jump {
  if (screen === "watch") {
    return {
      clip: FAILURE.clip,
      left: FAILURE.left,
      right: FAILURE.right,
      frame: FAILURE.playFrom,
      trackId: FAILURE.seedId,
      ...extra,
    };
  }
  if (screen === "compare") {
    return {
      clip: FAILURE.clip,
      left: FAILURE.left,
      right: FAILURE.right,
      frame: FAILURE.compareFrom,
      trackId: FAILURE.seedId,
      ...extra,
    };
  }
  return extra ?? {};
}

export function App() {
  const initial = parseHash();
  const [screen, setScreen] = useState<Screen>(initial.screen);
  const [jump, setJump] = useState<Jump>(() => {
    const fromHash: Jump = {
      clip: initial.params.get("clip") ?? undefined,
      left: initial.params.get("left") ?? undefined,
      right: initial.params.get("right") ?? undefined,
      frame: initial.params.get("f") ? Number(initial.params.get("f")) : undefined,
      trackId: initial.params.get("id") ? Number(initial.params.get("id")) : undefined,
    };
    return storyJump(initial.screen, fromHash.clip ? fromHash : undefined);
  });
  const [refs, setRefs] = useState<ClipRef[]>([]);
  const [clips, setClips] = useState<IndexedClip[]>([]);
  const [picked, setPicked] = useState<PickedMedia>({});
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("connecting");

  useEffect(() => {
    fetchDashboard()
      .then((d) => setStatus(`v${d.version} · ${d.clips} clips`))
      .catch(() => setStatus("API offline — start `make serve`"));
    fetchClips()
      .then(async (list) => {
        setRefs(list);
        const loaded = await Promise.all(list.map((c) => fetchClip(c.file)));
        setClips(loaded);
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  useEffect(() => {
    const onHash = () => {
      const { screen: s, params } = parseHash();
      setScreen(s);
      const fromHash: Jump = {
        clip: params.get("clip") ?? undefined,
        left: params.get("left") ?? undefined,
        right: params.get("right") ?? undefined,
        frame: params.get("f") ? Number(params.get("f")) : undefined,
        trackId: params.get("id") ? Number(params.get("id")) : undefined,
      };
      setJump(storyJump(s, fromHash.clip ? fromHash : undefined));
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((next: Screen, extra?: Jump) => {
    const j = storyJump(next, extra);
    const params = new URLSearchParams();
    if (j.clip) params.set("clip", j.clip);
    if (j.left) params.set("left", j.left);
    if (j.right) params.set("right", j.right);
    if (j.frame) params.set("f", String(j.frame));
    if (j.trackId) params.set("id", String(j.trackId));
    setJump(j);
    setScreen(next);
    writeHash(next, params);
  }, []);

  const addClip = useCallback((clip: IndexedClip) => {
    setClips((cs) => {
      const rest = cs.filter((c) => !(c.clip === clip.clip && c.tracker === clip.tracker));
      return [...rest, clip];
    });
  }, []);

  return (
    <div className="h-full flex flex-col bg-page text-ink font-sans text-xs antialiased">
      <Header screen={screen} onScreen={(s) => go(s)} status={status} />
      {error && (
        <div className="px-4 py-2 bg-fail/15 text-fail border-b border-fail/40 font-mono">
          {error}
        </div>
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
        {screen === "evidence" && (
          <Analytics clips={clips} onInspect={(j) => go("compare", j)} />
        )}
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
