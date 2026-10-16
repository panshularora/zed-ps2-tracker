// Story: the "What is an ID mix-up?" template — one headline, one short paragraph, one diagram.
// The takeaway number and Next come from App (components/Takeaway.tsx).
import { IdMixupExplainer } from "../components/IdMixup";
import { OURS_NAME } from "../lib/ours";

export default function Story() {
  return (
    <div className="story" data-testid="story">
      <h1 className="page-h">What is an ID mix-up?</h1>
      <p className="page-lead">
        A tracker gives every person a number and should keep it until they leave. When someone walks behind another
        person, the tracker can lose them and hand out a new number when they reappear, so one person is counted twice.
        We compare ByteTrack with {OURS_NAME}, which remembers a half-hidden person longer.
      </p>
      <div className="story-visual"><IdMixupExplainer /></div>
      <p className="page-lead" data-testid="story-status">Benchmark numbers are provisional: Pending GPU rerun.</p>
    </div>
  );
}
