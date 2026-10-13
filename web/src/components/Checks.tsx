import { checks, type Check, methodResult, type Pill, showProvisionalBanner, type SummaryFile } from "../lib/summary";

const PILL_CLASS: Record<Pill, string> = {
  PASS: "pill pass",
  "NOT REPRODUCED": "pill fail",
  "NO-GO": "pill fail",
  PROVISIONAL: "pill provisional",
  PENDING: "pill provisional",
};

export function VerdictPill({ pill }: { pill: Pill }) {
  const fail = pill === "NOT REPRODUCED" || pill === "NO-GO";
  return (
    <span className={PILL_CLASS[pill]} data-testid="pill" data-state={pill}>
      {fail && <span aria-hidden="true">✕ </span>}
      {pill}
    </span>
  );
}

function CheckCard({ c }: { c: Check }) {
  return (
    <div className="panel check" data-testid="check-card" data-check={c.id}>
      <div className="check-head">
        <h3>{c.title}</h3>
        <VerdictPill pill={c.pill} />
      </div>
      <p>{c.sentence}</p>
      {c.evidence.length > 0 && (
        <ul className="evidence">
          {c.evidence.map((e) => (
            <li key={e.text}>
              {e.struck ? (
                <>
                  <s>{e.text}</s> <span className="struck-note">{e.struck}</span>
                </>
              ) : (
                e.text
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ProvisionalBanner({ summary }: { summary: SummaryFile | null }) {
  if (!showProvisionalBanner(summary)) return null;
  return (
    <div className="prov-banner" data-testid="provisional-banner">
      Provisional · detector setup being fixed
    </div>
  );
}

/** Evaluation-setup checks first, then the method result (before/after), all from summary.json. */
export function FoundChecks({ summary }: { summary: SummaryFile | null }) {
  const m = methodResult(summary);
  return (
    <div className="found">
      <ProvisionalBanner summary={summary} />
      <div className="checks">
        {checks(summary).map((c) => (
          <CheckCard key={c.id} c={c} />
        ))}
      </div>
      <div className="panel method" data-testid="method-result">
        <h3>Method result: delayed assignment, ID switches</h3>
        <div className="pair">
          <div className="half">
            <div className="label">before (no delay)</div>
            <div className={`nvalue${m.before === "pending" ? " pending" : ""}`}>{m.before}</div>
          </div>
          <div className="half">
            <div className="label">after (5-frame delay)</div>
            <div className={`nvalue${m.after === "pending" ? " pending" : ""}`}>{m.after}</div>
          </div>
        </div>
        <div className="evidence">frames with held boxes: {m.held}</div>
      </div>
    </div>
  );
}
