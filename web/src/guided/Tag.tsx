export function Tag({ kind }: { kind: "illustrative" | "measured" }) {
  return kind === "measured"
    ? <span className="tag tag-measured" data-testid="clip-tag">Measured · human labels</span>
    : <span className="tag tag-illustrative" data-testid="clip-tag">Illustrative · no ground truth</span>;
}
