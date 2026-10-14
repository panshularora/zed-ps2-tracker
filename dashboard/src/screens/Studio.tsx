import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  clampFrame,
  gateCounts,
  type ClipEvent,
  type FrameBox,
  type IndexedClip,
} from "@viewer/data";
import {
  compareVerdicts,
  defaultFollowId,
  identityChain,
  firstTrackBox,
  lastKnownBox,
  matchAcross,
  overlappingPairs,
  OVERLAP_MIN_SCORE,
  switchOnFrame,
  switchesForChain,
  type IdentityVerdict,
  type OverlapPair,
} from "@viewer/identity";
import { Panel } from "@viewer/Panel";
import { pickMedia, useMedia, type PickedMedia } from "@viewer/useMedia";
import { idColor } from "@viewer/theme";
import type { Jump } from "../App";
import { padFrame, timecode } from "../format";
import { FAILURE, prettyTracker, storyBeat } from "../lib/journey";

interface Props {
  clips: IndexedClip[];
  jump: Jump;
  picked: PickedMedia;
  onPicked: (p: PickedMedia) => void;
  mode: "watch" | "compare";
  onNext: () => void;
}

function trackSpan(clip: IndexedClip, id: number): { f0: number; f1: number } | null {
  const rows = clip.tracks[String(id)];
  if (!rows?.length) return null;
  return { f0: rows[0][0], f1: rows[rows.length - 1][0] };
}

function boxesAt(clip: IndexedClip | undefined, frame: number): FrameBox[] {
  return clip?.byFrame.get(frame) ?? [];
}

export function Studio({ clips, jump, picked, onPicked, mode, onNext }: Props) {
  const storyClips = clips.filter((c) => c.clip === FAILURE.clip);
  const pool = storyClips.length ? storyClips : clips;
  const gtClip = pool.find((c) => c.tracker === "gt");
  const trackers = pool.filter((c) => c.tracker !== "gt").map((c) => c.tracker);
  const leftName = trackers.find((t) => t === FAILURE.left) ?? trackers[0] ?? "";
  const rightName =
    trackers.find((t) => t === FAILURE.right && t !== leftName) ??
    trackers.find((t) => t !== leftName) ??
    leftName;
  const L = pool.find((c) => c.tracker === leftName) ?? pool[0];
  const R = pool.find((c) => c.tracker === rightName) ?? pool[1] ?? pool[0];
  const compare = mode === "compare";

  const [frame, setFrame] = useState(jump.frame ?? FAILURE.playFrom);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(FAILURE.playSpeed);
  const [followId, setFollowId] = useState<number | null>(jump.trackId ?? FAILURE.seedId);
  const [rightSeed, setRightSeed] = useState<number | null>(null);
  const holdLeft = useRef(0);

  useEffect(() => {
    if (!L) return;
    setFollowId(jump.trackId ?? FAILURE.seedId ?? defaultFollowId(L, 1));
    setRightSeed(FAILURE.seedId);
    setFrame(
      clampFrame(jump.frame ?? (mode === "compare" ? FAILURE.compareFrom : FAILURE.playFrom), L.frames),
    );
    setSpeed(FAILURE.playSpeed);
    holdLeft.current = 0;
    setPlaying(mode === "compare");
  }, [L, jump.trackId, jump.frame, mode]);

  const lastClock = useRef<number | null>(null);
  const media = useMedia(L, clampFrame(frame, L?.frames ?? 1), picked, (f) => {
    lastClock.current = f;
    flushSync(() => setFrame(f));
  });
  const videoDriven = playing && media.clock !== null && media.clock.hasRvfc;

  useEffect(() => {
    const c = media.clock;
    if (!c) return;
    if (playing && speed === 1 && c.play()) return () => c.pause();
    c.pause();
  }, [playing, media.clock, speed]);

  useEffect(() => {
    if (!playing || !L || (videoDriven && speed === 1)) return;
    const tickMs = 1000 / (L.fps * speed);
    const dwellTicks = Math.max(1, Math.round((FAILURE.dwellSec * 1000) / tickMs));
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (holdLeft.current > 0) {
          holdLeft.current -= 1;
          return f;
        }
        if (f >= L.frames) {
          setPlaying(false);
          return f;
        }
        const next = f + 1;
        if (storyBeat(next).phase !== storyBeat(f).phase) holdLeft.current = dwellTicks;
        return next;
      });
    }, tickMs);
    return () => window.clearInterval(id);
  }, [playing, L, videoDriven, speed, mode]);

  useEffect(() => {
    if (frame === lastClock.current) return;
    if (media.clock && !videoDriven) media.clock.seek(frame);
  }, [frame, media.clock, videoDriven]);

  const f = clampFrame(frame, L?.frames ?? 1);
  const followIds = L && followId != null ? identityChain(L.events, followId) : [];
  const focusSet = new Set(followIds);
  const leftBoxes = boxesAt(L, f);
  const rightBoxes = boxesAt(R, f);
  const currentBox = leftBoxes.find((b) => focusSet.has(b.id));
  const rightHit = matchAcross(currentBox, rightBoxes, 0.4);
  useEffect(() => {
    if (rightHit) setRightSeed(rightHit.id);
  }, [rightHit?.id]);

  if (!L) {
    return (
      <div className="h-full flex items-center justify-center text-mute">Loading MOT17-09…</div>
    );
  }

  const rightFollow = rightHit?.id ?? rightSeed;
  const rightIds = rightFollow != null ? identityChain(R.events, rightFollow) : [];
  const rightFocus = new Set(rightIds);
  const overlaps = overlappingPairs(leftBoxes, OVERLAP_MIN_SCORE).filter(
    (p) => focusSet.has(p.a.id) || focusSet.has(p.b.id),
  );
  const overlapIds = new Set(overlaps.flatMap((p) => [p.a.id, p.b.id]));
  const rightOverlaps = overlappingPairs(rightBoxes, OVERLAP_MIN_SCORE).filter(
    (p) => rightFocus.has(p.a.id) || rightFocus.has(p.b.id),
  );
  const keepGtBox = gtClip?.byFrame.get(f)?.find((b) => b.id === FAILURE.gt) ?? null;
  const showKeepBox = Boolean(keepGtBox && (currentBox == null || currentBox.id !== FAILURE.from));
  const stableColor = followId != null ? idColor(followId) : undefined;
  const beat = storyBeat(f);
  const leftBorn = firstTrackBox(L, FAILURE.to);
  const rightBorn = firstTrackBox(R, FAILURE.to);
  const leftBreak = {
    leave: lastKnownBox(L, new Set([FAILURE.from]), FAILURE.goneFrom - 1),
    leaveFrame: FAILURE.goneFrom - 1,
    back: currentBox ?? leftBorn?.box ?? null,
    backFrame: leftBorn?.frame ?? FAILURE.backFrom,
    fromId: FAILURE.from,
    toId: FAILURE.to,
    phase: beat.phase,
  };
  const rightBreak = {
    leave: lastKnownBox(R, new Set([FAILURE.from]), FAILURE.goneFrom - 1),
    leaveFrame: FAILURE.goneFrom - 1,
    back: rightHit ?? rightBorn?.box ?? null,
    backFrame: rightBorn?.frame ?? FAILURE.backFrom,
    fromId: FAILURE.from,
    toId: FAILURE.to,
    phase: beat.phase,
  };
  const leftPretty = prettyTracker(L.tracker);
  const rightPretty = prettyTracker(R.tracker);
  const leftSwitch = switchOnFrame(L.events, f, focusSet);
  const compared = compare
    ? compareVerdicts(L.events, focusSet, R.events, rightFocus, f)
    : {
        a: leftSwitch ? ("switch" as const) : ("hold" as const),
        b: "hold" as const,
        gt: leftSwitch?.type === "id_switch" ? leftSwitch.gt : null,
      };

  function togglePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    holdLeft.current = 0;
    setSpeed(FAILURE.playSpeed);
    setFrame((n) => (n >= L.frames ? FAILURE.playFrom : n));
    setPlaying(true);
  }

  function applyFiles(files: File[]) {
    const next = pickMedia(files);
    onPicked({ ...next, frame_offset: picked.frame_offset });
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      <Brief
        mode={mode}
        frame={f}
        overlap={overlaps[0]}
        left={compared.a}
        right={compared.b}
        onPlay={togglePlay}
        playing={playing}
        onNext={onNext}
      />
      <NowBanner frame={f} events={L.events} />

      {media.error && (
        <div className="px-4 py-1.5 bg-signal/10 text-signal border-b border-line text-[12px]">
          {media.error} Drop a matching video in step 4 to see pixels under these boxes.
        </div>
      )}
      {media.videoSrc && (
        <video
          ref={media.videoRef}
          src={media.videoSrc}
          muted
          playsInline
          preload="auto"
          className="hidden"
          onError={media.onVideoError}
          onLoadedData={() => media.clock?.seek(f)}
        />
      )}

      <div
        className={`flex-1 p-2 min-h-0 grid gap-2 ${compare ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          applyFiles([...e.dataTransfer.files]);
        }}
      >
        <Viewport
          title={leftPretty.title}
          hint={
            compare
              ? leftPretty.why
              : "Camera is locked. Only people move. Hatched orange is two bodies sharing pixels. LOOK must keep #2."
          }
          clip={L}
          frame={f}
          image={media.image}
          tick={media.tick}
          counts={gateCounts(L.events, f)}
          focusIds={focusSet}
          dimOthers={false}
          missing={
            !currentBox && beat.phase === "gone"
              ? `LEFT at frame ${FAILURE.goneFrom - 1}. Tracker dropped #${FAILURE.from}. Green keep is still this person.`
              : null
          }
          overlapIds={overlapIds}
          overlaps={overlaps}
          keepId={FAILURE.from}
          gtId={FAILURE.gt}
          keepBox={showKeepBox ? keepGtBox : null}
          stableColor={stableColor}
          personLabel="LOOK"
          idBreak={leftBreak}
          verdict={compared.a}
          currentId={currentBox?.id}
          onPick={(id) => {
            if (id != null) {
              setFollowId(id);
              setRightSeed(null);
            }
          }}
        />
        {compare && (
          <Viewport
            title={rightPretty.title}
            hint="Camera is locked. Same person, lined up by where the box is."
            clip={R}
            frame={f}
            image={media.image}
            tick={media.tick}
            counts={gateCounts(R.events, f)}
            focusIds={rightFocus}
            dimOthers={false}
            missing={
              !rightHit && beat.phase === "gone"
                ? `LEFT at frame ${FAILURE.goneFrom - 1}. Tracker dropped #${FAILURE.from}. Green keep is still this person.`
                : null
            }
            overlapIds={new Set(rightOverlaps.flatMap((p) => [p.a.id, p.b.id]))}
            overlaps={rightOverlaps}
            keepId={FAILURE.from}
            gtId={FAILURE.gt}
            keepBox={
              keepGtBox && (rightHit == null || rightHit.id !== FAILURE.from) ? keepGtBox : null
            }
            stableColor={stableColor}
            personLabel="LOOK"
            idBreak={rightBreak}
            currentId={rightHit?.id}
            verdict={compared.b}
            onPick={(id) => {
              if (id != null) {
                const hit = matchAcross(
                  rightBoxes.find((b) => b.id === id),
                  leftBoxes,
                  0.4,
                );
                setFollowId(hit?.id ?? id);
                setRightSeed(id);
              }
            }}
          />
        )}
      </div>

      <div className="bg-paper border-t border-line px-4 py-1.5 flex flex-wrap items-center gap-2 shrink-0 text-[12px]">
        <span
          className="inline-block w-2.5 h-2.5 rounded-sm"
          style={{ background: stableColor ?? "#94a3b8" }}
        />
        <span className="text-ink font-medium">
          {currentBox
            ? currentBox.id === FAILURE.from
              ? `LOOK is #${currentBox.id} — this number must survive overlap`
              : `LOOK is the same person. Tracker now says #${currentBox.id}. Keep #${FAILURE.from} (GT ${FAILURE.gt}).`
            : `LOOK has no tracker box. Keep #${FAILURE.from} (GT ${FAILURE.gt}) is the ID that should still be theirs.`}
        </span>
        {compare && (
          <span className="text-mute">Other tracker: #{rightFollow ?? "—"}</span>
        )}
        {overlaps.length > 0 && (
          <span className="px-1.5 py-0.5 rounded bg-signal/15 text-signal">
            overlap #{overlaps[0].a.id}+#{overlaps[0].b.id} · keep #{FAILURE.from}
          </span>
        )}
        <span className="ml-auto text-mute hidden md:flex items-center gap-3 text-[11px]">
          <span>
            <span className="font-semibold text-ink">LOOK</span> = one body in the crowd
          </span>
          <span className="text-signal">hatched = two people sharing pixels</span>
          <span className="text-keep">green = keep #{FAILURE.from}</span>
          <span className="text-fail">red = tracker issued a new number</span>
        </span>
      </div>

      <PlaybackBar
        frame={f}
        frames={L.frames}
        fps={L.fps}
        playing={playing}
        speed={speed}
        mark={FAILURE.frame}
        onPlay={() => setPlaying((p) => !p)}
        onFrame={(n) => {
          setPlaying(false);
          setFrame(n);
        }}
        onSpeed={setSpeed}
      />
      <PersonTimeline clip={L} frame={f} followIds={followIds} onSeek={setFrame} />
    </div>
  );
}

function Brief({
  mode,
  frame,
  overlap,
  left,
  right,
  onPlay,
  playing,
  onNext,
}: {
  mode: "watch" | "compare";
  frame: number;
  overlap?: { iou: number };
  left: IdentityVerdict;
  right: IdentityVerdict;
  onPlay: () => void;
  playing: boolean;
  onNext: () => void;
}) {
  const beat = storyBeat(frame);
  return (
    <div className="px-4 py-2 bg-paper border-b border-line shrink-0">
      {mode === "watch" ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 justify-between">
          <div className="min-w-0">
            <div className="font-display uppercase tracking-[0.18em] text-[11px] text-signal">
              What's happening here
            </div>
            <h1 className="font-display text-[18px] sm:text-[20px] leading-tight text-ink font-semibold">
              They walk out of view. They come back with a new number.
            </h1>
            <p className="text-[12px] text-mute">
              Press play for the full clip at ¼ speed. Step 1: #{FAILURE.from} is here. Step 2: they
              leave. Step 3: they come back as #{FAILURE.to}.
            </p>
            <p className="text-[11px] text-mute" data-testid="privacy-note">
              Numbers are invented per run. We do not store faces, and it cannot look at future frames.
            </p>
            {overlap && beat.phase === "here" && (
              <p className="text-[11px] text-signal">
                Two people share pixels (hatched). LOOK must stay #{FAILURE.from}. Overlap is not a
                new ID.
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2 shrink-0">
            <button
              type="button"
              onClick={onPlay}
              className="px-3 py-1.5 rounded-sm bg-signal text-paper font-semibold text-[12px]"
            >
              {playing ? "Pause" : "Play the full clip (slow)"}
            </button>
            <button
              type="button"
              onClick={onNext}
              className="px-3 py-1.5 rounded-sm border border-line text-ink font-semibold text-[12px]"
            >
              Compare both trackers →
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 justify-between">
          <div className="min-w-0">
            <div className="font-display uppercase tracking-[0.18em] text-[11px] text-signal">
              Same person, two trackers
            </div>
            <h1 className="font-display text-[18px] sm:text-[20px] leading-tight text-ink font-semibold">
              Watch them come back on both sides
            </h1>
            <p className="text-[12px] text-mute">
              Left BoT-SORT, right ByteTrack. Playback starts while LOOK is still missing, so you
              see the return. Green keep is #{FAILURE.from} (GT {FAILURE.gt}). Red is a new number.
              Neither side keeps #{FAILURE.from}.
            </p>
            <p className="text-[12px] text-mute">
              BoT-SORT is {verdictWords(left)}; ByteTrack is {verdictWords(right)}.
            </p>
          </div>
          <button
            type="button"
            onClick={onNext}
            className="px-3 py-1.5 rounded-sm bg-signal text-paper font-semibold text-[12px] shrink-0"
          >
            See the evidence →
          </button>
        </div>
      )}
    </div>
  );
}

function Viewport({
  title,
  hint,
  clip,
  frame,
  image,
  tick,
  counts,
  focusIds,
  dimOthers = false,
  overlapIds,
  overlaps,
  keepId,
  gtId,
  keepBox,
  stableColor,
  personLabel,
  idBreak,
  currentId,
  verdict,
  missing,
  onPick,
}: {
  title: string;
  hint: string;
  clip: IndexedClip;
  frame: number;
  image: HTMLVideoElement | HTMLImageElement | null;
  tick: number;
  counts: { in: number; out: number };
  focusIds: Set<number>;
  dimOthers?: boolean;
  overlapIds: Set<number>;
  overlaps?: OverlapPair[];
  keepId?: number;
  gtId?: number;
  keepBox?: FrameBox | null;
  stableColor?: string;
  personLabel?: string;
  idBreak?: {
    leave: FrameBox | null;
    leaveFrame: number;
    back: FrameBox | null;
    backFrame: number;
    fromId: number;
    toId: number;
    phase: "here" | "gone" | "back";
  } | null;
  currentId?: number;
  verdict: IdentityVerdict;
  missing?: string | null;
  onPick: (id: number | null) => void;
}) {
  const people = (clip.byFrame.get(frame) ?? []).map((b) => b.id);
  const beat = storyBeat(frame);
  return (
    <div className="flex flex-col bg-raised rounded-sm border border-line overflow-hidden min-h-0">
      <div className="px-3 py-1 border-b border-line shrink-0 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="font-semibold text-ink text-[12px] truncate">{title}</div>
          <div className="text-[11px] text-mute truncate">{hint}</div>
        </div>
        <VerdictChip verdict={verdict} />
      </div>
      <div className="relative flex-1 bg-[#090d16] min-h-0 overflow-hidden flex items-center justify-center">
        <Panel
          clip={clip}
          frame={frame}
          image={image}
          imageTick={tick}
          chrome={false}
          className="h-full w-full flex items-center justify-center"
          focusIds={focusIds}
          dimOthers={dimOthers}
          overlapIds={overlapIds}
          overlaps={overlaps}
          keepId={keepId}
          gtId={gtId}
          keepBox={keepBox}
          stableColor={stableColor}
          personLabel={personLabel}
          cropCrowd
          showHeld={false}
          idBreak={idBreak}
          verdict={verdict}
          onPick={onPick}
        />
        <div className="pointer-events-none absolute left-2 top-2 max-w-[min(100%-1rem,28rem)]">
          <div
            className={`rounded-sm px-2.5 py-1.5 text-white shadow-sm ${
              beat.phase === "back" ? "bg-fail/95" : "bg-black/80"
            }`}
          >
            <div className="font-display text-[15px] sm:text-[17px] font-semibold leading-tight">
              {beat.step} · {beat.title}
            </div>
            {currentId != null && beat.phase !== "gone" && (
              <div className="text-[11px] font-mono mt-0.5">
                {currentId === FAILURE.from
                  ? `now #${currentId} — keep this number`
                  : `tracker #${currentId} · should still be #${FAILURE.from} (GT ${FAILURE.gt})`}
              </div>
            )}
            {missing && <div className="text-[11px] mt-0.5">{missing}</div>}
          </div>
        </div>
      </div>
      <div className="px-3 py-1 text-[11px] text-mute flex flex-wrap items-center gap-x-3 gap-y-1">
        <span>
          {people.length} people · {counts.in} in · {counts.out} out
        </span>
        <span className="font-mono text-[10px] flex flex-wrap gap-1">
          {people.map((id) => (
            <span
              key={id}
              className={
                focusIds.has(id)
                  ? "px-1 rounded-sm bg-fail text-white"
                  : "px-1 rounded-sm bg-raised text-ink border border-line"
              }
            >
              #{id}
            </span>
          ))}
        </span>
      </div>
    </div>
  );
}

function verdictWords(verdict: IdentityVerdict): string {
  if (verdict === "switch") return "changing the number";
  if (verdict === "recovery") return "keeping the old number";
  return "holding the same number";
}

function VerdictChip({ verdict }: { verdict: IdentityVerdict }) {
  if (verdict === "switch") {
    return (
      <span className="shrink-0 px-1.5 py-0.5 rounded-sm bg-fail text-white font-bold text-[10px] uppercase tracking-wide">
        ID switch
      </span>
    );
  }
  if (verdict === "recovery") {
    return (
      <span className="shrink-0 px-1.5 py-0.5 rounded-sm bg-keep text-white font-bold text-[10px] uppercase tracking-wide">
        Kept ID
      </span>
    );
  }
  return (
    <span className="shrink-0 px-1.5 py-0.5 rounded-sm border border-line text-mute text-[10px]">
      same number
    </span>
  );
}

function PlaybackBar({
  frame,
  frames,
  fps,
  playing,
  speed,
  mark,
  onPlay,
  onFrame,
  onSpeed,
}: {
  frame: number;
  frames: number;
  fps: number;
  playing: boolean;
  speed: number;
  mark: number;
  onPlay: () => void;
  onFrame: (f: number) => void;
  onSpeed: (s: number) => void;
}) {
  return (
    <div className="bg-paper border-t border-line px-4 py-2 flex flex-wrap items-center justify-between gap-3 shrink-0">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 bg-raised px-2.5 py-1 rounded-sm border border-line">
          <span className="text-mute uppercase text-[10px] font-bold">Frame</span>
          <span className="font-mono font-bold text-ink text-[13px]">{padFrame(frame)}</span>
          <span className="text-mute font-mono text-[11px]">/ {frames}</span>
        </div>
        <div className="font-mono text-mute text-[12px]">{timecode(frame, fps)}</div>
      </div>
      <div className="flex items-center gap-1">
        <StepBtn onClick={() => onFrame(Math.max(1, frame - 10))} icon="fast_rewind" />
        <StepBtn onClick={() => onFrame(Math.max(1, frame - 1))} icon="skip_previous" />
        <button
          type="button"
          onClick={onPlay}
          className="w-8 h-8 rounded-sm bg-signal text-paper hover:brightness-110 flex items-center justify-center"
        >
          <span className="material-symbols-outlined text-[17px]">{playing ? "pause" : "play_arrow"}</span>
        </button>
        <StepBtn onClick={() => onFrame(Math.min(frames, frame + 1))} icon="skip_next" />
        <StepBtn onClick={() => onFrame(Math.min(frames, frame + 10))} icon="fast_forward" />
        <button
          type="button"
          data-testid="jump-switch"
          onClick={() => onFrame(mark)}
          className="ml-1 px-2 py-1 rounded-sm border border-fail/50 text-fail text-[11px] font-semibold"
        >
          Jump to switch
        </button>
      </div>
      <div className="flex-1 max-w-sm hidden lg:flex items-center relative">
        <input
          type="range"
          min={1}
          max={frames}
          value={frame}
          onChange={(e) => onFrame(Number(e.target.value))}
          className="w-full accent-[var(--color-signal)]"
        />
      </div>
      <label className="flex items-center gap-1 text-[11px] text-mute">
        Speed
        <select
          value={speed}
          onChange={(e) => onSpeed(Number(e.target.value))}
          className="bg-raised border border-line rounded-sm px-1.5 py-0.5 text-ink"
        >
          <option value={0.25}>0.25x</option>
          <option value={0.5}>0.5x</option>
          <option value={1}>1.0x</option>
          <option value={2}>2.0x</option>
        </select>
      </label>
    </div>
  );
}

function StepBtn({ onClick, icon }: { onClick: () => void; icon: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-7 h-7 rounded-sm border border-line bg-raised hover:bg-paper flex items-center justify-center text-mute"
    >
      <span className="material-symbols-outlined text-[15px]">{icon}</span>
    </button>
  );
}

function PersonTimeline({
  clip,
  frame,
  followIds,
  onSeek,
}: {
  clip: IndexedClip;
  frame: number;
  followIds: number[];
  onSeek: (f: number) => void;
}) {
  const pct = (fr: number) => `${((fr - 1) / Math.max(1, clip.frames - 1)) * 100}%`;
  const switches = switchesForChain(clip.events, new Set(followIds));
  const spans = followIds
    .map((id) => {
      const span = trackSpan(clip, id);
      return span ? { id, ...span, color: idColor(id) } : null;
    })
    .filter((s): s is { id: number; f0: number; f1: number; color: string } => s != null);
  return (
    <div className="h-14 bg-paper flex flex-col shrink-0 border-t border-line">
      <div className="px-4 py-1 text-[10px] uppercase tracking-wider text-mute font-bold flex flex-wrap gap-x-4 gap-y-0.5">
        <span>This person over time</span>
        <span>Here as #{FAILURE.from}</span>
        <span>Gone after frame {FAILURE.goneFrom - 1}</span>
        <span>Back as #{FAILURE.to}</span>
      </div>
      <div className="flex-1 relative px-4 pb-2">
        <div className="h-7 bg-raised rounded-sm border border-line relative overflow-hidden">
          {spans.map((s) => (
            <div
              key={s.id}
              title={`#${s.id} frames ${s.f0}–${s.f1}`}
              className="absolute top-1 bottom-1 opacity-80"
              style={{
                left: pct(s.f0),
                width: `calc(${pct(s.f1)} - ${pct(s.f0)})`,
                background: s.color,
              }}
            />
          ))}
          {switches.map((ev, i) => (
            <button
              key={i}
              type="button"
              title={labelEvent(ev)}
              onClick={() => onSeek(ev.f)}
              className="absolute top-0 bottom-0 w-0.5 bg-fail"
              style={{ left: pct(ev.f) }}
            />
          ))}
          <div className="absolute top-0 bottom-0 w-0.5 bg-ink z-10" style={{ left: pct(frame) }} />
        </div>
      </div>
    </div>
  );
}

function labelEvent(ev: ClipEvent): string {
  if (ev.type === "id_switch") return `number ${ev.from} → ${ev.to} at frame ${ev.f}`;
  if (ev.type === "cross") return `gate ${ev.dir} id ${ev.id} at frame ${ev.f}`;
  return `tailgate ${ev.ids.join(", ")}`;
}

function NowBanner({ events, frame }: { events: ClipEvent[]; frame: number }) {
  const beat = storyBeat(frame);
  const tone =
    beat.phase === "back"
      ? "bg-fail/10 text-fail"
      : beat.phase === "gone"
        ? "bg-signal/10 text-signal"
        : "bg-keep/10 text-keep";
  const label = beat.phase === "here" ? "Here" : beat.phase === "gone" ? "Gone" : "Back";
  const switchEv = events.find(
    (e) => e.type === "id_switch" && e.from === FAILURE.from && Math.abs(e.f - frame) <= 2,
  );
  return (
    <div
      className="px-4 py-3 bg-raised border-b border-line text-[13px]"
      data-testid="gate-ticker"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className={`shrink-0 px-1.5 py-0.5 rounded-sm font-semibold text-[11px] uppercase ${tone}`}>
          {beat.step} · {label}
        </span>
        <span className="font-display text-[20px] sm:text-[24px] font-semibold text-ink leading-none">
          {beat.title}
        </span>
        {switchEv && switchEv.type === "id_switch" && (
          <span className="text-fail font-semibold">
            #{switchEv.from} became #{switchEv.to}
          </span>
        )}
      </div>
      <p className="text-ink mt-1">{beat.now}</p>
      <p className="text-mute">{beat.why}</p>
    </div>
  );
}
