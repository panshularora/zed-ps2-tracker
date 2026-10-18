// Calm amber error card for the upload flow: one-line cause, the exact fix command, Retry, and the
// raw error behind a "Details" toggle (closed by default). Never a stack trace as the headline.
import type { Explained } from "./explain";

export function ErrorCard({ err, onRetry }: { err: Explained; onRetry?: () => void }) {
  return (
    <div className="yv-err rounded-sm p-3 space-y-2 text-[12px]" role="alert" data-testid="error-card">
      <div className="text-ink" data-testid="error-cause">{err.cause}</div>
      {err.fix && (
        <div className="text-mute">
          Fix: <code className="font-mono text-ink" data-testid="error-fix">{err.fix}</code>
        </div>
      )}
      <div className="flex items-center gap-2">
        {onRetry && <button type="button" className="yv-btn" onClick={onRetry} data-testid="error-retry">Retry</button>}
        {err.raw && err.raw !== err.cause && (
          <details className="text-mute" data-testid="error-details">
            <summary className="cursor-pointer">Details</summary>
            <pre className="font-mono text-[11px] whitespace-pre-wrap break-all mt-1">{err.raw}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
