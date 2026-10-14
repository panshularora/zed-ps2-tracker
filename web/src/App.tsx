import { Navigate, NavLink, Route, Routes, useLocation } from "react-router-dom";
import Architecture from "./pages/Architecture";
import Landing from "./pages/Landing";
import Overview from "./pages/Overview";
import Results from "./pages/Results";
import Scenarios from "./pages/Scenarios";
import Workstation from "./workstation/Workstation";

export const NAV_GROUPS: { label: string; pages: { to: string; label: string; sub?: boolean }[] }[] = [
  { label: "Story", pages: [{ to: "/dashboard", label: "Overview" }] },
  { label: "Try it", pages: [{ to: "/dashboard/scenarios", label: "Scenarios" }, { to: "/dashboard/studio", label: "Studio" }] },
  {
    label: "Evidence",
    pages: [
      { to: "/dashboard/results", label: "Results" },
      { to: "/dashboard/experiments", label: "Experiments" },
      { to: "/dashboard/evidence", label: "Analytics" },
      { to: "/dashboard/architecture", label: "Architecture", sub: true },
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
                <span className="nav-group-label">{g.label}</span>
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
      <main>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Overview />} />
          <Route path="/dashboard/scenarios" element={<Scenarios />} />
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
