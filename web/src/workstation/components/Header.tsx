import type { Screen } from "../api";
import { STEPS } from "../lib/journey";

interface Props {
  screen: Screen;
  onScreen: (s: Screen) => void;
  status: string;
}

export function Header({ screen, onScreen, status }: Props) {
  const idx = Math.max(0, STEPS.findIndex((s) => s.id === screen));
  return (
    <header className="shrink-0 border-b border-line bg-paper">
      <div className="flex items-center justify-between gap-3 px-4 py-2.5">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-7 h-7 rounded-sm bg-signal text-paper font-display font-bold text-[13px] flex items-center justify-center tracking-tight">
            PS
          </div>
          <div className="min-w-0 leading-tight">
            <div className="font-display font-semibold text-[15px] text-ink tracking-wide uppercase">
              Gate ID
            </div>
            <div className="text-[11px] text-mute truncate">
              Zed Digital PS-2 · keep one person as one number, even when they get blocked
            </div>
          </div>
        </div>
        <div className="hidden sm:block font-mono text-[10px] text-mute">{status}</div>
      </div>
      <nav className="flex items-stretch border-t border-line overflow-x-auto" aria-label="Demo steps">
        {STEPS.map((s, i) => {
          const on = s.id === screen;
          const done = i < idx;
          return (
            <button
              key={s.id}
              type="button"
              data-testid={`step-${s.id}`}
              onClick={() => onScreen(s.id)}
              className={`relative flex-1 min-w-[7.5rem] sm:min-w-[9.5rem] px-3 py-2.5 text-left border-r border-line last:border-r-0 transition-colors ${
                on ? "bg-raised" : "hover:bg-raised/60"
              }`}
            >
              <div className="flex items-baseline gap-2">
                <span
                  className={`font-display text-[18px] leading-none ${
                    on ? "text-signal" : done ? "text-keep" : "text-mute"
                  }`}
                >
                  {s.n}
                </span>
                <span className={`text-[12px] font-semibold ${on ? "text-ink" : "text-mute"}`}>
                  {s.label}
                </span>
              </div>
              <div className={`text-[10px] mt-0.5 ${on ? "text-signal" : "text-mute/80"}`}>
                {i === 0 ? "Start here" : s.short}
              </div>
              {on && <span className="absolute inset-x-0 bottom-0 h-0.5 bg-signal" />}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
