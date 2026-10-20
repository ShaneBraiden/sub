import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { JobOptions } from "../api/types";
import FileDrop from "../components/FileDrop";
import { IconCaptions, IconChevron, IconFilm, IconSliders } from "../components/Icons";
import OptionsForm, { changedOptions } from "../components/OptionsForm";
import { useToast } from "../components/Toasts";
import { useOptionsInfo, useStoredState } from "../hooks/useApi";
import { formatPercent } from "../lib/format";

export default function NewJobPage() {
  const { info, error } = useOptionsInfo();
  const toast = useToast();
  const navigate = useNavigate();
  const [media, setMedia] = useState<File | null>(null);
  const [subs, setSubs] = useState<File | null>(null);
  const [terms, setTerms] = useStoredState("subfix.terms", "");
  const [saved, setSaved] = useStoredState<JobOptions>("subfix.options", {});
  const [upload, setUpload] = useState<number | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const mode = subs ? "A" : "B";

  useEffect(() => () => abortRef.current?.(), []);

  const changed = info ? Object.keys(changedOptions(saved, info.defaults)).length : 0;

  const submit = async () => {
    if (!media || !info) return;
    const options: JobOptions = { ...changedOptions(saved, info.defaults), terms };
    if (mode === "B" && options.segmentation === "keep") delete options.segmentation;
    setUpload(0);
    const req = api.createJob(media, subs, options, (f) => setUpload(f));
    abortRef.current = req.abort;
    try {
      const job = await req.promise;
      navigate(`/jobs/${job.id}`);
    } catch (e) {
      toast((e as Error).message, "error");
      setUpload(null);
    } finally {
      abortRef.current = null;
    }
  };

  return (
    <main className="page narrow">
      <div className="page-head">
        <div>
          <h1>Correct subtitles using the audio</h1>
          <p>
            Upload a video or audio file, with or without its subtitles. The audio is transcribed and used as
            evidence to fix wrong words, punctuation, timing and line breaks.
          </p>
        </div>
      </div>

      {error && (
        <div className="alert error" style={{ marginBottom: 16 }}>
          <div>
            <b>Cannot load the settings.</b> {error}
          </div>
        </div>
      )}

      <div className="stack">
        <div className="drops">
          <FileDrop
            title="Video or audio"
            hint="Drop a file or click to browse · mp4, mkv, mov, mp3, wav, …"
            accept={info?.media_exts ?? [".mp4", ".mkv", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac"]}
            file={media}
            onFile={setMedia}
            onReject={(m) => toast(m, "error")}
            icon={<IconFilm />}
            maxMb={info?.max_upload_mb}
          />
          <FileDrop
            title="Subtitles"
            hint="An .srt or .vtt file to correct"
            accept={info?.subs_exts ?? [".srt", ".vtt"]}
            file={subs}
            onFile={setSubs}
            onReject={(m) => toast(m, "error")}
            icon={<IconCaptions />}
            optional
          />
        </div>

        <div className="mode-note">
          <span className="mode-tag">{mode}</span>
          {mode === "A" ? (
            <div>
              <b>Correct existing subtitles.</b> Words are compared with what was said, timing is re-synced, and cues
              are re-split to fit the reading-speed and line-length limits.
            </div>
          ) : (
            <div>
              <b>Generate subtitles from the audio.</b> Add a subtitle file above to correct an existing one instead.
            </div>
          )}
        </div>

        <div className="card card-pad">
          <label className="field">
            <span>Names and terms in this video</span>
            <textarea
              className="input"
              rows={2}
              placeholder="e.g. Kubernetes, Chennai, Dr. Ramanathan"
              value={terms}
              onChange={(e) => setTerms(e.target.value)}
              maxLength={4000}
            />
            <span className="hint">Comma separated. They guide speech recognition and are preferred when correcting.</span>
          </label>
        </div>

        <details className="card disclosure">
          <summary>
            <IconChevron className="chev" />
            <IconSliders />
            Settings
            {changed > 0 && <span className="chip accent">{changed} changed</span>}
            <span style={{ flex: 1 }} />
            {changed > 0 && (
              <button
                type="button"
                className="btn ghost sm"
                onClick={(e) => {
                  e.preventDefault();
                  setSaved({});
                }}
              >
                Reset to defaults
              </button>
            )}
          </summary>
          {info ? <OptionsForm info={info} value={saved} onChange={setSaved} mode={mode} /> : <div className="card-pad muted">Loading…</div>}
        </details>

        <div className="submit-bar">
          <button className="btn primary lg" disabled={!media || !info || upload !== null} onClick={submit}>
            {mode === "A" ? "Correct subtitles" : "Generate subtitles"}
          </button>
          {upload !== null ? (
            <div className="upload-progress">
              <div className="row between small">
                <span>{upload < 1 ? "Uploading…" : "Starting the job…"}</span>
                <span className="num muted">{formatPercent(upload)}</span>
              </div>
              <div className="bar">
                <div style={{ width: `${upload * 100}%` }} />
              </div>
            </div>
          ) : (
            <span className="muted small">
              {media ? "Files are processed one at a time on this machine." : "Choose a video or audio file to start."}{" "}
              <Link to="/jobs">See earlier jobs</Link>
            </span>
          )}
          {upload !== null && upload < 1 && (
            <button className="btn ghost" onClick={() => abortRef.current?.()}>
              Cancel upload
            </button>
          )}
        </div>
      </div>
    </main>
  );
}
