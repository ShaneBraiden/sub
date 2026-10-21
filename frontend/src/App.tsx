import { useEffect } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { IconMonitor, IconMoon, IconSun } from "./components/Icons";
import { useStoredState, useSystem } from "./hooks/useApi";
import JobPage from "./pages/JobPage";
import JobsPage from "./pages/JobsPage";
import NewJobPage from "./pages/NewJobPage";

type Theme = "system" | "light" | "dark";

function ThemeSwitch() {
  const [theme, setTheme] = useStoredState<Theme>("subfix.theme", "system");
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") delete root.dataset.theme;
    else root.dataset.theme = theme;
  }, [theme]);
  const next: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
  const Icon = theme === "light" ? IconSun : theme === "dark" ? IconMoon : IconMonitor;
  return (
    <button
      className="btn ghost icon"
      onClick={() => setTheme(next[theme])}
      title={`Theme: ${theme} (click to change)`}
      aria-label={`Theme: ${theme}`}
    >
      <Icon />
    </button>
  );
}

function SystemPill() {
  const { system, offline } = useSystem();
  if (offline) {
    return (
      <div className="sys" title="The backend is not reachable">
        <span className="dot bad" /> Server offline
      </div>
    );
  }
  if (!system) return <div className="sys"><span className="dot" /> Connecting…</div>;
  const d = system.device;
  const device = d.device === "cuda" ? d.name ?? "GPU" : "CPU";
  const busy = system.running ? `processing${system.queued ? `, ${system.queued} queued` : ""}` : "idle";
  const mem =
    d.memory_total_mb != null && d.memory_free_mb != null
      ? ` · ${((d.memory_total_mb - d.memory_free_mb) / 1024).toFixed(1)}/${(d.memory_total_mb / 1024).toFixed(1)} GB`
      : "";
  return (
    <div className="sys" title={system.models_loaded.length ? `Loaded: ${system.models_loaded.join(", ")}` : "No models loaded yet"}>
      <span className={`dot ${system.running ? "busy" : "ok"}`} />
      {device}
      {mem} · {busy}
    </div>
  );
}

function NotFound() {
  return (
    <main className="page narrow">
      <div className="card empty">
        <h2>Page not found</h2>
        <Link to="/">Start a new job</Link>
      </div>
    </main>
  );
}

export default function App() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);     // a block body: scrollTo returns a Promise in newer browsers, not a clean-up
  }, [pathname]);
  return (
    <>
      <header className="topbar">
        <Link to="/" className="brand">
          <img src="/favicon.svg" alt="" />
          Subfix <span>audio-aware subtitles</span>
        </Link>
        <nav className="nav">
          <NavLink to="/" end>
            New job
          </NavLink>
          <NavLink to="/jobs">History</NavLink>
        </nav>
        <div className="spacer" />
        <SystemPill />
        <ThemeSwitch />
      </header>
      <Routes>
        <Route path="/" element={<NewJobPage />} />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="/jobs/:id" element={<JobPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
