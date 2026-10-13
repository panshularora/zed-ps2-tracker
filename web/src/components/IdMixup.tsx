// Neutral explainer of an ID mix-up (no tracker is named and no result is claimed).
// Replaces the landing's earlier running example, a hand-picked clip moment that favoured ours.

const A = "#E6E8EB";
const B = "#F5A524";
const MUTED = "#8A929C";

function Person({ x, label, color, hidden = false }: { x: number; label: string; color: string; hidden?: boolean }) {
  return (
    <g opacity={hidden ? 0.35 : 1}>
      <rect x={x} y={52} width={34} height={70} rx={4} fill="none" stroke={color} strokeWidth={2} strokeDasharray={hidden ? "5 4" : undefined} />
      <rect x={x} y={36} width={34} height={16} rx={2} fill={color} />
      <text x={x + 17} y={48} textAnchor="middle" fontSize={11} fontWeight={600} fill="#0E1013">{label}</text>
    </g>
  );
}

function Frame({ x, title, children }: { x: number; title: string; children: React.ReactNode }) {
  return (
    <g transform={`translate(${x},0)`}>
      <rect x={0} y={20} width={190} height={120} rx={4} fill="#161A1F" stroke="#262B33" />
      <line x1={0} x2={190} y1={108} y2={108} stroke="#F5A524" strokeWidth={1} strokeDasharray="4 4" opacity={0.6} />
      {children}
      <text x={95} y={160} textAnchor="middle" fontSize={12} fill={MUTED}>{title}</text>
    </g>
  );
}

export function IdMixupExplainer() {
  return (
    <figure className="mixup" data-testid="id-mixup">
      <svg viewBox="0 0 640 260" className="diagram" role="img"
        aria-label="Two people, ID 3 and ID 7, pass close together. One is hidden for a moment. Afterwards the tracker should still call them 3 and 7; an ID mix-up is when the labels swap or one person gets a new number.">
        <Frame x={10} title="1 · two people, two IDs">
          <Person x={40} label="ID 3" color={A} />
          <Person x={116} label="ID 7" color={B} />
        </Frame>
        <Frame x={225} title="2 · one walks behind the other">
          <Person x={86} label="ID 7" color={B} hidden />
          <Person x={74} label="ID 3" color={A} />
        </Frame>
        <Frame x={440} title="3 · they separate again">
          <Person x={40} label="?" color={B} />
          <Person x={116} label="?" color={A} />
        </Frame>
        <g fontSize={12}>
          <rect x={10} y={186} width={300} height={62} rx={4} fill="#161A1F" stroke="#262B33" />
          <text x={22} y={208} fill="#E6E8EB" fontWeight={600}>Kept</text>
          <text x={22} y={228} fill={MUTED}>Each person still has the ID they had in</text>
          <text x={22} y={242} fill={MUTED}>frame 1: the one who was 7 is still 7.</text>
          <rect x={330} y={186} width={300} height={62} rx={4} fill="#161A1F" stroke="#262B33" />
          <text x={342} y={208} fill="#E6E8EB" fontWeight={600}>ID mix-up (an “ID switch”)</text>
          <text x={342} y={228} fill={MUTED}>The labels swap, or the hidden person comes</text>
          <text x={342} y={242} fill={MUTED}>back with a new number, such as ID 12.</text>
        </g>
      </svg>
      <figcaption className="muted small">
        Illustration, not footage. Every tracker, ours included, makes some of these mistakes; the benchmark counts how often.
      </figcaption>
    </figure>
  );
}

export function HiddenPersonDiagram() {
  const steps = [
    ["Seen", "Box and ID as usual."],
    ["Hidden", "Notice the person is covered, not gone, and keep their ID on hold."],
    ["Back", "When a box reappears nearby, try the held ID before handing out a new one."],
  ];
  return (
    <div className="why" data-testid="hidden-diagram">
      {steps.map(([h, p]) => (
        <div className="panel" key={h}><h3>{h}</h3><p className="muted">{p}</p></div>
      ))}
    </div>
  );
}
