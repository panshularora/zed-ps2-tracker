import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import Architecture from "./pages/Architecture";
import Demo from "./pages/Demo";
import Landing from "./pages/Landing";
import Overview from "./pages/Overview";
import Results from "./pages/Results";
import Scenarios from "./pages/Scenarios";

const PAGES = [
  { to: "/dashboard", label: "Overview" },
  { to: "/dashboard/scenarios", label: "Scenarios" },
  { to: "/dashboard/demo", label: "Demo" },
  { to: "/dashboard/results", label: "Results" },
  { to: "/dashboard/architecture", label: "Architecture" },
];

export default function App() {
  const landing = useLocation().pathname === "/";
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
            PAGES.map((p) => (
              <NavLink key={p.to} to={p.to} end={p.to === "/dashboard"} className={({ isActive }) => (isActive ? "active" : "")}>
                {p.label}
              </NavLink>
            ))
          )}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Overview />} />
          <Route path="/dashboard/scenarios" element={<Scenarios />} />
          <Route path="/dashboard/demo" element={<Demo />} />
          <Route path="/dashboard/results" element={<Results />} />
          <Route path="/dashboard/architecture" element={<Architecture />} />
          <Route path="*" element={<p className="muted">Page not found.</p>} />
        </Routes>
      </main>
      <footer className="footer">
        <span>Benchmark numbers: MOT17 val-half, public detections, from docs/results. GPU rows pending the DGX run.</span>
        <span>
          Demo footage: <a href="https://www.pexels.com" target="_blank" rel="noreferrer">Pexels</a> (Pexels License), faces blurred. Apache-2.0.
        </span>
      </footer>
    </div>
  );
}
