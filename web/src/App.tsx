import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import Architecture from "./pages/Architecture";
import Landing from "./pages/Landing";
import Overview from "./pages/Overview";
import Results from "./pages/Results";
import Scenarios from "./pages/Scenarios";
import RealCompare from "./compare/RealCompare";
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
    <div className="offline-banner" role="status" data-testid="offline-banner">
      Backend offline: showing the committed snapshot (docs/results, viewer/data, web/public/demo). Live tracking runs need <code>make serve</code>.
    </div>
  );
}

export const NAV_GROUPS: { label: string; pages: { to: string; label: string; sub?: boolean }[] }[] = [
  { label: "Story", pages: [{ to: "/dashboard", label: "Overview" }, { to: "/dashboard/what-we-use", label: "What we use" }] },
  { label: "Try it", pages: [{ to: "/dashboard/live", label: "Gate scene" }, { to: "/dashboard/scenarios", label: "Scenarios" }, { to: "/dashboard/studio", label: "Studio" }] },
  {
    label: "Evidence",
    pages: [
      { to: "/dashboard/results", label: "Results" },
      { to: "/dashboard/experiments", label: "Experiments" },
      { to: "/dashboard/evidence", label: "Analytics" },
      { to: "/dashboard/architecture", label: "Architecture" },
    ],
  },
];

export default function App() {
  const path = useLocation().pathname;
  const landing = path === "/";
  return (
    <div className="shell">
      <header className="topbar">
        <NavLink to="/" className="brand" style={{ color: "inherit", textDecoration: "none" }}>
          <span className="brand-mark" aria-hidden />
          <span>
            Gate tracker <small>· Zed Digital PS-2</small>
          </span>
        </NavLink>
        <nav className="nav" aria-label="Pages">
          {landing ? (
            <NavLink to="/dashboard">Dashboard</NavLink>
          ) : (
            NAV_GROUPS.map((g) => (
              <div key={g.label} className="nav-group" role="group" aria-label={g.label}>
                <NavLink to={g.pages[0].to} end className="nav-group-label" data-testid="nav-group" title={`${g.label}: open ${g.pages[0].label}`}>{g.label}</NavLink>
                {g.pages.map((p) => (
                  <NavLink key={p.to} to={p.to} end={p.to === "/dashboard"} className={({ isActive }) => `${p.sub ? "sub " : ""}${isActive || (p.to === "/dashboard/studio" && path === "/dashboard/compare") ? "active" : ""}`}>
                    {p.label}
                  </NavLink>
                ))}
              </div>
            ))
          )}
        </nav>
      </header>
      <OfflineBanner />
      <main>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Overview />} />
          <Route path="/dashboard/what-we-use" element={<WhatWeUse />} />
          <Route path="/dashboard/live" element={<GateScene />} />
          <Route path="/dashboard/scenarios" element={<Scenarios />} />
          <Route path="/dashboard/scenarios/compare" element={<RealCompare />} />
          <Route path="/dashboard/studio" element={<Workstation screen="watch" />} />
          <Route path="/dashboard/compare" element={<Workstation screen="compare" />} />
          <Route path="/dashboard/evidence" element={<Workstation screen="evidence" />} />
          <Route path="/dashboard/experiments" element={<Workstation screen="try" />} />
          <Route path="/dashboard/demo" element={<Navigate to="/dashboard/scenarios" replace />} />
          <Route path="/dashboard/results" element={<Results />} />
          <Route path="/dashboard/architecture" element={<Architecture />} />
          <Route path="*" element={<p className="muted">Page not found.</p>} />
        </Routes>
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
