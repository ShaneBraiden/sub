import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { Job } from "../api/types";
import { IconCaptions, IconFilm, IconInbox, IconTrash, IconWave } from "../components/Icons";
import StatusBadge from "../components/StatusBadge";
import { useToast } from "../components/Toasts";
import { isTerminal } from "../hooks/useApi";
import { isVideo } from "../lib/cues";
import { formatDuration, formatPercent, formatRelative, plural } from "../lib/format";

function changesOf(job: Job): string {
  const c = job.summary?.counts;
  if (!c) return "–";
  const words = c.sub + c.ins + c.del;
  const base = job.mode === "A" ? `${plural(words, "word")} · ${plural(c.moved_cues, "cue")} re-timed` : `${plural(job.summary?.cues ?? 0, "cue")}`;
  const ch = job.summary?.chapters ?? 0;
  return ch >= 2 ? `${base} · ${plural(ch, "chapter")}` : base;
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const toast = useToast();

  const load = useCallback(async () => {
    try {
      const r = await api.jobs();
      setJobs(r.jobs);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const active = jobs?.some((j) => !isTerminal(j.status)) ?? false;
  useEffect(() => {
    if (!active) return;
    const t = window.setInterval(load, 2000);
    return () => window.clearInterval(t);
  }, [active, load]);

  const remove = async (job: Job) => {
    if (!window.confirm(`Delete “${job.media_name}” and its files?`)) return;
    try {
      if (job.status === "running") await api.cancel(job.id);
      else await api.remove(job.id);
      toast(job.status === "running" ? "Cancelling…" : "Job deleted");
      load();
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>History</h1>
          <p>Every file you processed on this server, newest first.</p>
        </div>
        <Link className="btn primary" to="/">
          New job
        </Link>
      </div>

      {error && <div className="alert error" style={{ marginBottom: 16 }}>{error}</div>}

      <div className="card">
        {jobs === null ? (
          <div className="empty">Loading…</div>
        ) : jobs.length === 0 ? (
          <div className="empty">
            <IconInbox />
            <h2 style={{ color: "var(--fg)" }}>No jobs yet</h2>
            <p>Upload a video or audio file to get started.</p>
            <Link className="btn primary" to="/">
              New job
            </Link>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Status</th>
                  <th className="hide-sm">Length</th>
                  <th className="hide-sm">Changes</th>
                  <th className="hide-sm">Started</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id} className="link" onClick={() => navigate(`/jobs/${j.id}`)}>
                    <td>
                      <div className="file-cell">
                        <div className="f-ico">{isVideo(j.media_name) ? <IconFilm /> : <IconWave />}</div>
                        <div style={{ minWidth: 0 }}>
                          <div className="f-name" title={j.media_name}>
                            {j.media_name}
                          </div>
                          <div className="small muted row" style={{ gap: 6 }}>
                            <span className="chip" title={j.mode === "A" ? "Audio + subtitles" : "Audio only"}>
                              Mode {j.mode}
                            </span>
                            {j.subs_name && (
                              <span className="row" style={{ gap: 4 }}>
                                <IconCaptions width={13} height={13} /> {j.subs_name}
                              </span>
                            )}
                            {j.edited && <span className="chip accent">edited</span>}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <div className="row" style={{ gap: 8 }}>
                        <StatusBadge status={j.status} />
                        {j.status === "running" && <span className="small muted num">{formatPercent(j.progress)}</span>}
                        {j.status === "queued" && j.queue_position != null && (
                          <span className="small muted">#{j.queue_position} in line</span>
                        )}
                      </div>
                    </td>
                    <td className="hide-sm num">{formatDuration(j.summary?.audio_duration)}</td>
                    <td className="hide-sm small">{changesOf(j)}</td>
                    <td className="hide-sm small muted nowrap">{formatRelative(j.created)}</td>
                    <td style={{ textAlign: "right" }}>
                      <button
                        className="btn ghost icon sm"
                        title={j.status === "running" ? "Cancel" : "Delete"}
                        aria-label={`Delete ${j.media_name}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          remove(j);
                        }}
                      >
                        <IconTrash />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
