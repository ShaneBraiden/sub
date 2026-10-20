import { useMemo, useState } from "react";
import type { Result, SpeakerReport } from "../api/types";
import { formatClock, formatDuration, formatPercent, formatTime } from "../lib/format";
import { speakerColors, speakerName } from "../lib/speakers";

interface Props {
  result: Result;
  time: number;
  onSeek: (t: number) => void;
}

interface Tip {
  x: number;
  y: number;
  lines: string[];
}

function Spk({ name, color }: { name: string; color?: string }) {
  return (
    <span className="spk" title={name}>
      <span className="spk-dot" style={{ background: color }} />
      {speakerName(name)}
    </span>
  );
}

function seekFromEvent(e: React.MouseEvent<HTMLElement>, total: number, onSeek: (t: number) => void) {
  const r = e.currentTarget.getBoundingClientRect();
  onSeek(Math.max(0, Math.min(total, ((e.clientX - r.left) / Math.max(r.width, 1)) * total)));
}

/** Who spoke when: one lane per speaker (the subtitle's speaker turns) + a lane of overlapping speech. */
function Lanes({ report, colors, total, time, onSeek }: { report: SpeakerReport; colors: Map<string, string>; total: number; time: number; onSeek: (t: number) => void }) {
  const [tip, setTip] = useState<Tip | null>(null);
  const pct = (t: number) => `${(Math.max(0, Math.min(total, t)) / total) * 100}%`;
  const show = (e: React.MouseEvent, lines: string[]) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTip({ x: Math.min(r.left, window.innerWidth - 360), y: r.bottom + 8, lines });
  };
  const head = <span className="spk-head" style={{ left: pct(time) }} />;
  return (
    <div className="spk-lanes" onMouseLeave={() => setTip(null)}>
      {report.speakers.map((s) => (
        <div className="spk-lane" key={s.speaker}>
          <Spk name={s.speaker} color={colors.get(s.speaker)} />
          <div className="spk-track" role="img" aria-label={`${speakerName(s.speaker)} speaks for ${formatDuration(s.talk_time)}`} onClick={(e) => seekFromEvent(e, total, onSeek)}>
            {report.turns
              .filter((t) => t.speaker === s.speaker && t.start != null && t.end != null)
              .map((t, i) => (
                <span
                  key={i}
                  className="spk-turn"
                  style={{ left: pct(t.start!), width: `calc(${pct(t.end!)} - ${pct(t.start!)})`, background: colors.get(s.speaker) }}
                  onMouseEnter={(e) =>
                    show(e, [
                      `${speakerName(s.speaker)} · ${formatTime(t.start)}–${formatTime(t.end)}`,
                      `${t.words} word${t.words === 1 ? "" : "s"}${t.overlap ? " · overlapping speech" : ""}`,
                      t.text.length > 140 ? `${t.text.slice(0, 140)}…` : t.text,
                    ])
                  }
                />
              ))}
            {head}
          </div>
        </div>
      ))}
      {report.overlaps.length > 0 && (
        <div className="spk-lane">
          <span className="spk">Overlap</span>
          <div className="spk-track" role="img" aria-label={`${report.overlaps.length} stretches of overlapping speech`} onClick={(e) => seekFromEvent(e, total, onSeek)}>
            {report.overlaps.map((o, i) => (
              <span
                key={i}
                className="spk-ovl"
                style={{ left: pct(o.start), width: `calc(${pct(o.end)} - ${pct(o.start)})` }}
                onMouseEnter={(e) =>
                  show(e, [`Overlapping speech · ${formatTime(o.start)}–${formatTime(o.end)}`, o.speakers.map(speakerName).join(" + ")])
                }
              />
            ))}
            {head}
          </div>
        </div>
      )}
      <div className="spk-axis" aria-hidden>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <span key={f}>{formatClock(f * total)}</span>
        ))}
      </div>
      {tip && (
        <div className="tooltip" style={{ left: Math.max(8, tip.x), top: tip.y }}>
          {tip.lines.map((l, i) => (i === 0 ? <div key={i}><b>{l}</b></div> : <div key={i}>{l}</div>))}
        </div>
      )}
    </div>
  );
}

export default function SpeakersPanel({ result, time, onSeek }: Props) {
  const report = result.speakers;
  const colors = useMemo(() => speakerColors(result), [result]);
  if (!report) return <div className="empty">This job was run without speaker identification.</div>;
  const T = report.totals;
  const total = Math.max(result.audio_duration, T.audio_duration, 0.1);

  return (
    <div className="spk-panel">
      <div className="spk-totals">
        <span>
          <b>{T.speakers}</b> speaker{T.speakers === 1 ? "" : "s"}
        </span>
        <span>
          <b>{formatDuration(T.speech_time)}</b> of speech
        </span>
        <span>
          <b>{formatDuration(T.overlap_time)}</b> overlapping ({formatPercent(T.overlap_share, 1)})
        </span>
        <span>
          <b>{T.speaker_changes.toLocaleString()}</b> speaker changes
        </span>
        <span>
          average turn <b>{formatDuration(T.mean_turn)}</b>
        </span>
        <span>
          <b>{T.dialogue_cues}</b> two-speaker cue{T.dialogue_cues === 1 ? "" : "s"}
        </span>
      </div>

      <section>
        <h3 style={{ marginBottom: 10 }}>Who spoke when</h3>
        <Lanes report={report} colors={colors} total={total} time={time} onSeek={onSeek} />
      </section>

      <section>
        <h3 style={{ marginBottom: 8 }}>Per speaker</h3>
        <div style={{ overflowX: "auto" }}>
          <table className="data spk-table">
            <thead>
              <tr>
                <th>Speaker</th>
                <th className="num">Talk time</th>
                <th>Share of speech</th>
                <th className="num">Words</th>
                <th className="num hide-sm">Words/min</th>
                <th className="num">Turns</th>
                <th className="num hide-sm">Avg turn</th>
                <th className="num hide-sm">Longest</th>
                <th className="num" title="Time this speaker talks while someone else talks too">Overlap</th>
                <th className="num" title="Turns that start while another speaker is still talking">Interruptions</th>
              </tr>
            </thead>
            <tbody>
              {report.speakers.map((s) => (
                <tr key={s.speaker} className={s.first != null ? "link" : undefined} onClick={() => s.first != null && onSeek(s.first)} title={s.first != null ? `First heard at ${formatTime(s.first)}` : undefined}>
                  <td>
                    <Spk name={s.speaker} color={colors.get(s.speaker)} />
                  </td>
                  <td className="num">{formatDuration(s.talk_time)}</td>
                  <td>
                    <div className="spk-share">
                      <div className="track">
                        <div className="fill" style={{ width: `${s.share * 100}%`, background: colors.get(s.speaker) }} />
                      </div>
                      <span className="small num">{formatPercent(s.share)}</span>
                    </div>
                  </td>
                  <td className="num">{s.words.toLocaleString()}</td>
                  <td className="num hide-sm">{s.wpm != null ? Math.round(s.wpm) : "–"}</td>
                  <td className="num">{s.turns}</td>
                  <td className="num hide-sm">{formatDuration(s.mean_turn)}</td>
                  <td className="num hide-sm">{formatDuration(s.longest_turn)}</td>
                  <td className="num">{formatDuration(s.overlap_time)}</td>
                  <td className="num">{s.interruptions}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small muted" style={{ marginTop: 8 }}>
          Talk time and overlap come from the diarization; words and turns from the subtitle text. Shares can add up to more than 100% when people talk over each other.
        </p>
      </section>

      <section>
        <h3 style={{ marginBottom: 8 }}>Speaker turns</h3>
        <div className="spk-turns">
          <table className="data">
            <tbody>
              {report.turns.map((t, i) => (
                <tr key={i} className="link" onClick={() => t.start != null && onSeek(t.start)}>
                  <td className="small num" style={{ whiteSpace: "nowrap" }}>
                    {formatTime(t.start)}
                  </td>
                  <td>{t.speaker ? <Spk name={t.speaker} color={colors.get(t.speaker)} /> : <span className="muted">–</span>}</td>
                  <td>
                    {t.text}
                    {t.overlap && (
                      <span className="chip warn" style={{ marginLeft: 6 }}>
                        overlap
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
