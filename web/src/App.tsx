import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import Architecture from "./pages/Architecture";
import Story from "./pages/Story";
import Results from "./pages/Results";
import Scenarios from "./pages/Scenarios";
import RealCompare from "./compare/RealCompare";
import MomentCompare from "./compare/MomentCompare";
import { Takeaway } from "./components/Takeaway";
import GateScene from "./pages/GateScene";
import WhatWeUse from "./pages/WhatWeUse";
import Workstation from "./workstation/Workstation";
import { useEffect, useState } from "react";
import { isOffline, onOffline } from "./lib/offline";

export function OfflineBanner() {
  const [off, setOff] = useState(isOffline());
  useEffect(() => onOffline(setOff), []);
  if (!off) return null;
  return (
    <span className="offline-banner" role="status" data-testid="offline-banner" title="Showing the committed snapshot (docs/results, viewer/data, web/public/demo). Live tracking runs need make serve.">
      Offline snapshot
    </span>
  );
}

export interface NavItem { to: string; label: string; match: string[]; subs?: { to: string; label: string }[] }

/** Four top-level items; Try it and Results carry sub-tabs. PATH is the guided order for "Next →". */
export const NAV: NavItem[] = [
  { to: "/", label: "Story", match: ["/", "/dashboard"] },
  { to: "/dashboard/what-we-use", label: "What we use", match: ["/dashboard/what-we-use"] },
  {
    to: "/dashboard/live", label: "Try it", match: ["/dashboard/live", "/dashboard/compare", "/dashboard/studio", "/dashboard/scenarios", "/dashboard/scenarios/compare"],
    subs: [
      { to: "/dashboard/live", label: "Gate scene" },
      { to: "/dashboard/compare", label: "Same person, two trackers" },
      { to: "/dashboard/studio", label: "Studio" },
      { to: "/dashboard/scenarios", label: "Scenarios" },
    ],
  },
  {
    to: "/dashboard/results", label: "Results", match: ["/dashboard/results", "/dashboard/experiments", "/dashboard/evidence", "/dashboard/architecture"],
    subs: [
      { to: "/dashboard/results", label: "Comparison" },
      { to: "/dashboard/experiments", label: "Experiments" },
      { to: "/dashboard/evidence", label: "Analytics" },
      { to: "/dashboard/architecture", label: "Architecture" },
    ],
  },
];
export const NAV_PAGES = NAV.map(({ to, label }) => ({ to, label }));
export const PATH: { to: string; label: string }[] = [
  { to: "/", label: "Story" },
  { to: "/dashboard/what-we-use", label: "What we use" },
  ...NAV[2].subs!,
  ...NAV[3].subs!,
];
export function nextOf(path: string): { to: string; label: string } | null {
  const p = path === "/dashboard" ? "/" : path === "/dashboard/scenarios/compare" ? "/dashboard/scenarios" : path;
  const i = PATH.findIndex((x) => x.to === p);
  return i >= 0 && i < PATH.length - 1 ? PATH[i + 1] : null;
}


export default function App() {
  const path = useLocation().pathname;
  const group = NAV.find((g) => g.match.includes(path));
  const next = nextOf(path);
  return (
    <div className="shell">
      <header className="topbar">
        <NavLink to="/" className="brand" style={{ color: "inherit", textDecoration: "none" }} end>
          <span className="brand-mark" aria-hidden />
          <span>
            Gate tracker <small>· Zed Digital PS-2</small>
          </span>
        </NavLink>
        <OfflineBanner />
        <nav className="nav" aria-label="Pages">
          {NAV.map((p) => (
            <NavLink key={p.to} to={p.to} end className={() => (p.match.includes(path) ? "active" : "")}>
              {p.label}
            </NavLink>
          ))}
        </nav>
      </header>
      {group?.subs && (
        <nav className="subnav" aria-label="Sections" data-testid="subnav">
          {group.subs.map((x) => (
            <NavLink key={x.to} to={x.to} end className={() => (x.to === path || (x.to === "/dashboard/scenarios" && path === "/dashboard/scenarios/compare") ? "active" : "")}>{x.label}</NavLink>
          ))}
        </nav>
      )}
      <main>
        <Routes>
          <Route path="/" element={<Story />} />
          <Route path="/dashboard" element={<Story />} />
          <Route path="/dashboard/what-we-use" element={<WhatWeUse />} />
          <Route path="/dashboard/live" element={<GateScene />} />
          <Route path="/dashboard/scenarios" element={<Scenarios />} />
          <Route path="/dashboard/scenarios/compare" element={<RealCompare />} />
          <Route path="/dashboard/studio" element={<Workstation screen="watch" />} />
          <Route path="/dashboard/compare" element={<MomentCompare />} />
          <Route path="/dashboard/evidence" element={<Workstation screen="evidence" />} />
          <Route path="/dashboard/experiments" element={<Workstation screen="try" />} />
          <Route path="/dashboard/demo" element={<Navigate to="/dashboard/scenarios" replace />} />
          <Route path="/dashboard/results" element={<Results />} />
          <Route path="/dashboard/architecture" element={<Architecture />} />
          <Route path="*" element={<p className="muted">Page not found.</p>} />
        </Routes>
        <div className="page-foot" data-testid="page-foot">
          <Takeaway path={path} />
          {next && <NavLink to={next.to} className="btn primary next-btn" data-testid="next-btn">Next: {next.label} →</NavLink>}
        </div>
      </main>
      <footer className="footer">
        <span>Benchmark numbers are provisional: MOT17 val-half, public detections, read from docs/results/summary.json (laptop GPU run).</span>
        <span>
          Code: Apache-2.0. Demo clips: <a href="https://www.pexels.com/license/" target="_blank" rel="noreferrer">Pexels License</a> (not Apache-2.0), faces blurred; see web/public/demo/LICENSE_CLIPS.md.
        </span>
      </footer>
    </div>
  );
}
