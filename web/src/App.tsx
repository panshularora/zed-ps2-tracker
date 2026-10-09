import { NavLink, Route, Routes } from "react-router-dom";
import Architecture from "./pages/Architecture";
import Demo from "./pages/Demo";
import Overview from "./pages/Overview";
import Results from "./pages/Results";

const PAGES = [
  { to: "/", label: "Overview" },
  { to: "/demo", label: "Demo" },
  { to: "/results", label: "Results" },
  { to: "/architecture", label: "Architecture" },
];

export default function App() {
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
          {PAGES.map((p) => (
            <NavLink key={p.to} to={p.to} end={p.to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>
              {p.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/results" element={<Results />} />
          <Route path="/architecture" element={<Architecture />} />
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
