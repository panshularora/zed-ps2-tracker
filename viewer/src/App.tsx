import { useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { base64ToBytes, clampFrame, loadClipGz, type IndexedClip } from "./data";
import { Panel } from "./Panel";
import { Timeline } from "./Timeline";
import { pickMedia, useMedia, type PickedMedia } from "./useMedia";
import { EMBEDDED } from "./generated/clips";

export function App() {
  const [clips, setClips] = useState<IndexedClip[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [clipName, setClipName] = useState<string>("");
  const [left, setLeft] = useState<string>("");
  const [right, setRight] = useState<string>("");
  const [frame, setFrame] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [picked, setPicked] = useState<PickedMedia>({});

  useEffect(() => {
    Promise.all(EMBEDDED.map((c) => loadClipGz(base64ToBytes(c.b64))))
      .then((cs) => setClips(cs))
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const names = useMemo(() => [...new Set(clips.map((c) => c.clip))], [clips]);
  const current = clips.filter((c) => c.clip === (clipName || names[0]));
  const trackers = current.map((c) => c.tracker);
  const L = current.find((c) => c.tracker === left) ?? current[0];
  const R = current.find((c) => c.tracker === right) ?? current[1] ?? current[0];

  // initial frame from the URL (#f=123) so screenshots and links can deep-link
  useEffect(() => {
    const read = () => {
      const m = /f=(\d+)/.exec(window.location.hash);
      if (m) setFrame(Number(m[1]));
    };
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, [clips.length]);

  const lastClock = useRef<number | null>(null); // last frame reported by the video itself
  const media = useMedia(L, clampFrame(frame, L?.frames ?? 1), picked, (f) => {
    lastClock.current = f;
    // commit synchronously so the panels draw inside the rVFC callback, i.e. with the frame
    // that was just presented (a deferred render could draw the video one frame early or late)
    flushSync(() => setFrame(f));
  });
  const videoDriven = playing && media.clock !== null && media.clock.hasRvfc;

  // play / pause: the video drives frames through rVFC when it can; otherwise we step frames
  useEffect(() => {
    const c = media.clock;
    if (!c) return;
    if (playing && c.play()) return () => c.pause();
    c.pause();
  }, [playing, media.clock]);
  useEffect(() => {
    if (!playing || !L || videoDriven) return;
    const id = window.setInterval(() => {
      setFrame((f) => (f >= L.frames ? (setPlaying(false), f) : f + 1));
    }, 1000 / L.fps);
    return () => window.clearInterval(id);
  }, [playing, L, videoDriven]);
  // user seeks (timeline, #f=, stepping without rVFC) move the video to the frame centre
  useEffect(() => {
    if (frame === lastClock.current) return; // the video is already showing it
    if (media.clock && !videoDriven) media.clock.seek(frame);
  }, [frame, media.clock, videoDriven]);

  async function onFiles(files: FileList | null) {
    if (!files) return;
    try {
      const loaded = await Promise.all(
        [...files].map(async (f) => loadClipGz(new Uint8Array(await f.arrayBuffer()))),
      );
      setClips((cs) => [...cs, ...loaded]);
    } catch (e) {
      setError(String(e));
    }
  }

  if (error) return <div className="error">{error}</div>;
  if (!L) {
    return (
      <div className="empty">
        <p>No embedded clips. Open one or more .json.gz clip files:</p>
        <input type="file" multiple accept=".gz" onChange={(e) => onFiles(e.target.files)} />
      </div>
    );
  }
  const f = clampFrame(frame, L.frames);
  const events = [
    ...L.events.map((ev) => ({ ev, side: L.tracker })),
    ...(R !== L ? R.events.map((ev) => ({ ev, side: R.tracker })) : []),
  ];
  return (
    <div className="app">
      <div className="toolbar">
        <select value={clipName || names[0]} onChange={(e) => setClipName(e.target.value)}>
          {names.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        <select aria-label="left" value={L.tracker} onChange={(e) => setLeft(e.target.value)}>
          {trackers.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <select aria-label="right" value={R.tracker} onChange={(e) => setRight(e.target.value)}>
          {trackers.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
        <span className="frame">
          frame <b>{f}</b> / {L.frames} · {(f / L.fps).toFixed(1)} s
        </span>
        <label className="open">
          open video / frames
          <input
            type="file"
            multiple
            accept="video/*,image/jpeg,image/png"
            onChange={(e) => setPicked(pickMedia([...(e.target.files ?? [])]))}
          />
        </label>
        <span className="frame">{media.kind === "blank" ? "blank canvas" : media.kind}</span>
        <label className="open">
          open clips
          <input type="file" multiple accept=".gz" onChange={(e) => onFiles(e.target.files)} />
        </label>
      </div>
      {media.error && <div className="note">{media.error}</div>}
      {media.videoSrc && (
        <video
          ref={media.videoRef}
          src={media.videoSrc}
          muted
          playsInline
          preload="auto"
          style={{ display: "none" }}
          onError={media.onVideoError}
          onLoadedData={() => media.clock?.seek(f)}
        />
      )}
      <div className="panels">
        <Panel clip={L} frame={f} image={media.image} imageTick={media.tick} />
        <Panel clip={R} frame={f} image={media.image} imageTick={media.tick} />
      </div>
      <Timeline frames={L.frames} fps={L.fps} frame={f} events={events} onSeek={setFrame} />
    </div>
  );
}
