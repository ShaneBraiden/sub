import type { FormatReport, Result } from "../api/types";
import { formatDuration, formatPercent, formatShift } from "../lib/format";
import { STAGE_TIME_LABEL } from "../lib/labels";

function Compare({ label, before, after, beforeLabel }: { label: string; before?: number; after?: number; beforeLabel: string }) {
  if (after == null) return null;
  return (
    <div className="metric">
      <div className="m-head">
        <span>{label}</span>
      </div>
      <div className="m-bars">
        {before != null && (
          <div className="mbar">
            <span>{beforeLabel}</span>
            <div className="track">
              <div className="fill" style={{ width: `${before * 100}%` }} />
            </div>
            <span className="val">{formatPercent(before)}</span>
          </div>
        )}
        <div className="mbar after">
          <span>After</span>
          <div className="track">
            <div className="fill" style={{ width: `${after * 100}%` }} />
          </div>
          <span className="val">{formatPercent(after)}</span>
        </div>
      </div>
    </div>
  );
}

export default function QualityPanel({ result }: { result: Result }) {
  const L = result.limits;
  const b: FormatReport = result.format_before;
  const a: FormatReport = result.format_after;
  const beforeLabel = result.mode === "A" ? "Before" : "Whisper";
  const times = Object.entries(result.stage_times).filter(([, s]) => s > 0.005);
  const maxTime = Math.max(...times.map(([, s]) => s), 0.001);
  const s = result.sync;
  const timing = result.timing as Record<string, number | string | undefined>;

  return (
    <div className="quality-grid">
      <section>
        <h3>Subtitle guidelines met (share of cues)</h3>
        <div className="metric-list">
          <Compare label={`Reading speed ≤ ${L.max_cps} characters/s`} before={b.fmt_cps17_ok} after={a.fmt_cps17_ok} beforeLabel={beforeLabel} />
          <Compare label={`Reading speed ≤ ${L.lenient_cps} characters/s (lenient)`} before={b.fmt_cps20_ok} after={a.fmt_cps20_ok} beforeLabel={beforeLabel} />
          <Compare label={`Lines ≤ ${L.max_chars} characters`} before={b.fmt_lines_ok} after={a.fmt_lines_ok} beforeLabel={beforeLabel} />
          <Compare label={`At most ${L.max_lines} line${L.max_lines > 1 ? "s" : ""}`} before={b.fmt_nlines_ok} after={a.fmt_nlines_ok} beforeLabel={beforeLabel} />
          <Compare label={`On screen ${L.min_dur.toFixed(2)}–${L.max_dur} s`} before={b.fmt_dur_ok} after={a.fmt_dur_ok} beforeLabel={beforeLabel} />
          <Compare label={`Gap between cues ≥ ${Math.round(L.min_gap * 1000)} ms`} before={b.fmt_gap_ok} after={a.fmt_gap_ok} beforeLabel={beforeLabel} />
          <Compare label="All rules at once" before={b.fmt_all_ok} after={a.fmt_all_ok} beforeLabel={beforeLabel} />
        </div>
      </section>

      <section>
        <h3>Processing time</h3>
        <dl className="kv">
          <dt>Audio length</dt>
          <dd>{formatDuration(result.audio_duration)}</dd>
          <dt>Processing</dt>
          <dd>{formatDuration(result.processing_time)}</dd>
          <dt>Real-time factor</dt>
          <dd>
            {result.rtf.toFixed(3)} <span className="muted">({(1 / Math.max(result.rtf, 1e-6)).toFixed(0)}× faster than real time)</span>
          </dd>
        </dl>
        <div className="time-bars">
          {times.map(([k, v]) => (
            <div className="tbar" key={k}>
              <span>{STAGE_TIME_LABEL[k] ?? k}</span>
              <div className="track">
                <div className="fill" style={{ width: `${(v / maxTime) * 100}%` }} />
              </div>
              <span className="val">{v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(1)} s`}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3>Timing</h3>
        <dl className="kv">
          {s ? (
            <>
              <dt>Subtitle sync</dt>
              <dd>{s.note ?? s.method}</dd>
              <dt>Offset</dt>
              <dd>{formatShift(s.b)}</dd>
              <dt>Speed</dt>
              <dd>
                ×{s.a.toFixed(5)} {Math.abs(s.a - 1) > 5e-4 && <span className="chip warn">frame-rate drift</span>}
              </dd>
              {s.pieces.length > 1 && (
                <>
                  <dt>Offset jumps</dt>
                  <dd>{s.pieces.length - 1}</dd>
                </>
              )}
              <dt>Anchors</dt>
              <dd>
                {s.n_anchors} matched phrases, {formatPercent(s.inlier_ratio)} consistent
              </dd>
            </>
          ) : result.mode === "A" ? (
            <>
              <dt>Subtitle sync</dt>
              <dd>off</dd>
            </>
          ) : null}
          <dt>Word timing</dt>
          <dd>{timing.backend === "whisper" ? "Whisper timestamps" : `Forced alignment (${String(timing.backend ?? "–")})`}</dd>
          {typeof timing.aligned_words === "number" && (
            <>
              <dt>Aligned words</dt>
              <dd>
                {timing.aligned_words.toLocaleString()} in {String(timing.chunks)} chunk(s)
                {Number(timing.failed_chunks) > 0 && <span className="chip warn" style={{ marginLeft: 6 }}>{String(timing.failed_chunks)} failed</span>}
              </dd>
            </>
          )}
        </dl>
      </section>

      <section>
        <h3>Recognition</h3>
        <dl className="kv">
          <dt>Whisper model</dt>
          <dd>{String(result.asr.model ?? "–")}</dd>
          <dt>Words heard</dt>
          <dd>{Number(result.asr.words ?? 0).toLocaleString()}</dd>
          {result.asr.language != null && (
            <>
              <dt>Language</dt>
              <dd>{String(result.asr.language)}</dd>
            </>
          )}
        </dl>
        <details>
          <summary className="small muted" style={{ cursor: "pointer" }}>
            Full configuration used
          </summary>
          <pre className="code" style={{ marginTop: 8 }}>
            {JSON.stringify(result.config, null, 2)}
          </pre>
        </details>
      </section>
    </div>
  );
}
