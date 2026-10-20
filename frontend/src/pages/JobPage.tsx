import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { Job, JobOptions, Result } from "../api/types";
import DownloadMenu from "../components/DownloadMenu";
import { IconAlert, IconBack, IconCaptions, IconRefresh, IconTrash } from "../components/Icons";
import Modal from "../components/Modal";
import OptionsForm, { changedOptions } from "../components/OptionsForm";
import ProgressPanel from "../components/ProgressPanel";
import ResultView from "../components/ResultView";
import StatusBadge from "../components/StatusBadge";
import { useToast } from "../components/Toasts";
import { useJob, useOptionsInfo } from "../hooks/useApi";
import { formatDuration, formatRelative } from "../lib/format";

function RerunDialog({ job, onClose }: { job: Job; onClose: () => void }) {
  const { info } = useOptionsInfo();
  const navigate = useNavigate();
  const toast = useToast();
  const [value, setValue] = useState<JobOptions>(() => {
    const rest = { ...job.options };
    delete rest.terms;
    return rest;
  });
  const [terms, setTerms] = useState(job.options.terms ?? "");
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!info) return;
    setBusy(true);
    try {
      const next = await api.rerun(job.id, { ...changedOptions(value, info.defaults), terms });
      onClose();
      navigate(`/jobs/${next.id}`);
    } catch (e) {
      toast((e as Error).message, "error");
      setBusy(false);
    }
  };

  return (
    <Modal
      title="Run again with different settings"
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn primary" onClick={run} disabled={!info || busy}>
            {busy ? "Starting…" : "Run again"}
          </button>
        </>
      }
    >
      <div className="card-pad" style={{ paddingBottom: 0 }}>
        <label className="field">
          <span>Names and terms</span>
          <textarea className="input" rows={2} value={terms} onChange={(e) => setTerms(e.target.value)} />
        </label>
      </div>
      {info ? <OptionsForm info={info} value={value} onChange={setValue} mode={job.mode} /> : <div className="card-pad muted">Loading…</div>}
      <p className="small muted" style={{ padding: "0 20px 14px" }}>
        The same files are used. Unchanged steps are reused from the earlier run, so this is usually quick.
      </p>
    </Modal>
  );
}

function LogViewer({ jobId }: { jobId: string }) {
  const [log, setLog] = useState<string | null>(null);
  useEffect(() => {
    api.log(jobId).then(setLog).catch(() => setLog(""));
  }, [jobId]);
  if (!log) return null;
  return (
    <details className="card disclosure">
      <summary>Run log</summary>
      <pre className="code" style={{ margin: 12 }}>
        {log}
      </pre>
    </details>
  );
}

export default function JobPage() {
  const { id = "" } = useParams();
  const { job, error, refresh } = useJob(id);
  const { info } = useOptionsInfo();
  const toast = useToast();
  const navigate = useNavigate();
  const [result, setResult] = useState<Result | null>(null);
  const [resultError, setResultError] = useState<string | null>(null);
  const [rerun, setRerun] = useState(false);

  const loadResult = useCallback(() => {
    api
      .result(id)
      .then((r) => {
        setResult(r);
        setResultError(null);
      })
      .catch((e: Error) => setResultError(e.message));
  }, [id]);

  useEffect(() => {
    setResult(null);
  }, [id]);

  useEffect(() => {
    if (job?.status === "done") loadResult();
  }, [job?.status, loadResult]);

  useEffect(() => {
    if (job) document.title = `${job.status === "done" ? "" : `${job.status} · `}${job.media_name} · Subfix`;
    return () => {
      document.title = "Subfix";
    };
  }, [job]);

  const cancel = async () => {
    try {
      await api.cancel(id);
      refresh();
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  const remove = async () => {
    if (!job || !window.confirm(`Delete “${job.media_name}” and all its files?`)) return;
    try {
      await api.remove(id);
      toast("Job deleted");
      navigate("/jobs");
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  if (error) {
    return (
      <main className="page narrow">
        <div className="card empty">
          <IconAlert />
          <h2 style={{ color: "var(--fg)" }}>{error.status === 404 ? "Job not found" : "Cannot load this job"}</h2>
          <p>{error.message}</p>
          <Link className="btn" to="/jobs">
            Back to history
          </Link>
        </div>
      </main>
    );
  }
  if (!job) {
    return (
      <main className="page">
        <div className="empty">
          <span className="spinner lg" />
        </div>
      </main>
    );
  }

  const active = job.status === "queued" || job.status === "running";

  return (
    <main className={`page${active || job.status !== "done" ? " narrow" : ""}`}>
      <div className="job-head">
        <div className="title">
          <div className="crumb">
            <Link to="/jobs">
              <IconBack width={13} height={13} style={{ verticalAlign: -2 }} /> History
            </Link>
          </div>
          <h1>{job.media_name}</h1>
          <div className="row small muted" style={{ gap: 8 }}>
            <StatusBadge status={job.status} />
            <span className="chip">{job.mode === "A" ? "Mode A · audio + subtitles" : "Mode B · audio only"}</span>
            {job.subs_name && (
              <span className="row" style={{ gap: 4 }}>
                <IconCaptions width={14} height={14} /> {job.subs_name}
              </span>
            )}
            <span>started {formatRelative(job.created)}</span>
            {job.status === "done" && job.elapsed != null && <span>· took {formatDuration(job.elapsed)}</span>}
            {job.options.terms && <span title={job.options.terms}>· {job.options.terms.split(",").length} term(s)</span>}
          </div>
        </div>
        <div className="row">
          {!active && (
            <button className="btn" onClick={() => setRerun(true)}>
              <IconRefresh /> Run again
            </button>
          )}
          {!active && (
            <button className="btn danger icon" onClick={remove} title="Delete job" aria-label="Delete job">
              <IconTrash />
            </button>
          )}
          {job.status === "done" && result && <DownloadMenu jobId={job.id} result={result} />}
        </div>
      </div>

      {active && info && <ProgressPanel job={job} stages={info.stages} onCancel={cancel} />}

      {job.status === "failed" && (
        <div className="stack">
          <div className="alert error">
            <IconAlert width={20} height={20} />
            <div>
              <b>The job failed.</b>
              <pre>{job.error}</pre>
            </div>
          </div>
          <LogViewer jobId={job.id} />
        </div>
      )}

      {job.status === "cancelled" && (
        <div className="alert">
          <div>
            <b>Cancelled.</b> Use <i>Run again</i> to process the same files.
          </div>
        </div>
      )}

      {job.status === "done" &&
        (result ? (
          <ResultView job={job} result={result} onSaved={loadResult} />
        ) : resultError ? (
          <div className="alert error">{resultError}</div>
        ) : (
          <div className="empty">
            <span className="spinner lg" />
            Loading the results…
          </div>
        ))}

      {rerun && <RerunDialog job={job} onClose={() => setRerun(false)} />}
    </main>
  );
}
