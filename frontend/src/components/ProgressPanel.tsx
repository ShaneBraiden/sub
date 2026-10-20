import { useEffect, useState } from "react";
import type { Job } from "../api/types";
import { formatClock, formatPercent } from "../lib/format";
import { plannedStages } from "../lib/labels";
import { IconCheck } from "./Icons";

interface Props {
  job: Job;
  stages: { key: string; label: string }[];
  onCancel: () => void;
}

export default function ProgressPanel({ job, stages, onCancel }: Props) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), 500);
    return () => window.clearInterval(t);
  }, []);

  const planned = plannedStages(stages, job.mode, job.options, job.stages.map((s) => s.key));
  const seen = new Map(job.stages.map((s, i) => [s.key, { ...s, order: i }]));
  const lastSeen = job.stages.length ? planned.findIndex((p) => p.key === job.stages[job.stages.length - 1].key) : -1;
  const elapsed = job.started ? now - job.started : 0;
  const queued = job.status === "queued";

  return (
    <div className="card">
      <div className="card-pad stack" style={{ gap: 14 }}>
        <div className="row between">
          <div className="row" style={{ gap: 12 }}>
            {queued ? <span className="pending-dot" style={{ width: 12, height: 12 }} /> : <span className="spinner lg" />}
            <div>
              <h2>{queued ? "Waiting in the queue" : job.cancelling ? "Cancelling…" : job.stage ?? "Starting"}</h2>
              <p className="small muted">
                {queued
                  ? job.queue_position
                    ? `${job.queue_position} job${job.queue_position > 1 ? "s" : ""} ahead. Files run one at a time on this machine.`
                    : "Starting soon."
                  : job.cancelling
                    ? "The job stops when the current step finishes."
                    : "The first job after the server starts also loads the models, which takes longer."}
              </p>
            </div>
          </div>
          <div className="row">
            {!queued && <span className="num muted">{formatClock(elapsed)}</span>}
            <button className="btn danger sm" onClick={onCancel} disabled={job.cancelling}>
              {queued ? "Remove from queue" : "Cancel"}
            </button>
          </div>
        </div>
        <div className={`bar${queued ? " indet" : ""}`} aria-label="Progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
          <div style={{ width: `${Math.max(job.progress, 0.02) * 100}%` }} />
        </div>
        {!queued && <div className="small muted num" style={{ textAlign: "right", marginTop: -8 }}>{formatPercent(job.progress)}</div>}
      </div>
      <div style={{ borderTop: "1px solid var(--border)", padding: "10px 14px" }}>
        <ol className="stages">
          {planned.map((p, i) => {
            const s = seen.get(p.key);
            const current = !queued && job.stage_key === p.key && s?.seconds == null;
            const done = s != null && s.seconds != null;
            const skipped = !s && i < lastSeen;
            const state = current ? "current" : done ? "done" : skipped ? "skipped" : "pending";
            return (
              <li key={p.key} className={state}>
                <span className="s-ico">
                  {current ? <span className="spinner" /> : done || skipped ? <IconCheck /> : <span className="pending-dot" />}
                </span>
                <span>
                  {p.label}
                  {skipped && <span className="small muted"> · reused from an earlier run</span>}
                </span>
                <span className="small muted num">
                  {done ? `${s!.seconds!.toFixed(1)} s` : current && s ? `${Math.max(0, now - s.started).toFixed(0)} s` : ""}
                </span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
