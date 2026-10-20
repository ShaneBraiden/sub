import { memo, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { CueData, DiffRow, Result, Token } from "../api/types";
import { cps, lastCueBefore, rowMatches, type RowFilter } from "../lib/cues";
import { formatTime } from "../lib/format";
import { EDIT_STAGE_LABEL } from "../lib/labels";
import { cueSpeakers, speakerColors, speakerName } from "../lib/speakers";
import { IconEdit } from "./Icons";
import { useToast } from "./Toasts";

interface Draft {
  start: number;
  end: number;
  text: string;
}

interface Props {
  jobId: string;
  result: Result;
  time: number;
  playing: boolean;
  onSeek: (t: number) => void;
  onSaved: () => void;
}

function Tokens({ tokens }: { tokens: Token[] }) {
  return (
    <>
      {tokens.map((t, i) => (
        <span key={i}>
          {i > 0 && " "}
          {t.k === "same" ? t.t : <span className={`tk ${t.k}`} data-tip={t.tip || undefined}>{t.t}</span>}
        </span>
      ))}
    </>
  );
}

const TIP_KEY: Record<string, string> = { was: "Subtitle had", ASR: "Audio heard", stages: "Changed by" };

/** "was: x | ASR: y (p=0.93) | score=1.42 | rule=phonetic | stages: fusion > punct" -> labelled lines */
function tipLines(tip: string) {
  return tip.split(" | ").map((part, i) => {
    const m = part.match(/^(\w+)(?::\s|=)(.*)$/);
    if (!m) return <div key={i}>{part}</div>;
    const [, key, raw] = m;
    const value = key === "stages" ? raw.split(" > ").map((s) => EDIT_STAGE_LABEL[s] ?? s).join(" → ") : raw;
    const label = TIP_KEY[key] ?? (key === "score" ? "Fusion score" : key === "rule" ? "Rule" : key);
    return (
      <div key={i}>
        <b>{label}:</b> {value}
      </div>
    );
  });
}

interface RowProps {
  row: DiffRow;
  cue: CueData;
  edited: CueData | undefined;
  active: boolean;
  editing: boolean;
  draft: Draft | undefined;
  maxCps: number;
  countSpaces: boolean;
  onSeek: (t: number) => void;
  onDraft: (i: number, d: Draft | undefined) => void;
  i: number;
  colors: Map<string, string>;
}

function SpeakerChips({ speakers, overlap, colors }: { speakers: string[]; overlap?: boolean; colors: Map<string, string> }) {
  if (!speakers.length && !overlap) return null;
  return (
    <div className="c-spk">
      {speakers.map((s) => (
        <span key={s} className="spk" title={s}>
          <span className="spk-dot" style={{ background: colors.get(s) }} />
          {speakerName(s)}
        </span>
      ))}
      {overlap && (
        <span className="chip warn" title="Part of this cue is spoken while someone else talks">
          overlap
        </span>
      )}
    </div>
  );
}

const Row = memo(function Row({ row, cue, edited, active, editing, draft, maxCps, countSpaces, onSeek, onDraft, i, colors }: RowProps) {
  const shown = draft ?? (edited ? { start: edited.start, end: edited.end, text: edited.text } : { start: cue.start, end: cue.end, text: cue.text });
  const isEdited = !!edited && (edited.text !== cue.text || edited.start !== cue.start || edited.end !== cue.end);
  const labels = (!draft && edited?.labels) || cue.labels;
  const rate = cps(shown.text, shown.start, shown.end, countSpaces, labels);
  const speakers = cueSpeakers((isEdited && edited?.speakers) || row.speakers);
  const invalid = !shown.text.trim() || shown.end <= shown.start;

  return (
    <div
      className={`cue${active ? " active" : ""}${editing ? " editing" : ""}`}
      data-row={i}
      onClick={editing ? undefined : () => onSeek(shown.start)}
    >
      <div className="c-idx">{row.index}</div>
      <div className={`c-time${row.moved || (isEdited && edited!.start !== cue.start) ? " moved" : ""}`}>
        {row.old_start != null && (row.moved || draft) ? <div>{formatTime(row.old_start)}</div> : null}
        <div className="new">
          {row.old_start != null && (row.moved || draft) ? "→ " : ""}
          {formatTime(shown.start)}
        </div>
        <div className="muted">{(shown.end - shown.start).toFixed(2)} s</div>
      </div>
      <div className="c-text c-orig">
        <Tokens tokens={row.orig} />
      </div>
      <div className="c-text c-new">
        {!editing && <SpeakerChips speakers={speakers} overlap={row.overlap} colors={colors} />}
        {editing ? (
          <div className="cue-edit">
            <textarea
              className="input"
              rows={2}
              value={shown.text}
              aria-label={`Text of cue ${row.index}`}
              style={invalid ? { borderColor: "var(--danger)" } : undefined}
              onChange={(e) => onDraft(i, { ...shown, text: e.target.value })}
            />
            <div className="times">
              <input
                className="input sm num"
                type="number"
                step={0.01}
                min={0}
                value={shown.start}
                aria-label="Start (s)"
                title="Start (seconds)"
                onChange={(e) => onDraft(i, { ...shown, start: parseFloat(e.target.value) || 0 })}
              />
              <input
                className="input sm num"
                type="number"
                step={0.01}
                min={0}
                value={shown.end}
                aria-label="End (s)"
                title="End (seconds)"
                onChange={(e) => onDraft(i, { ...shown, end: parseFloat(e.target.value) || 0 })}
              />
            </div>
            {draft && (
              <button className="btn ghost sm" style={{ justifySelf: "start" }} onClick={() => onDraft(i, undefined)}>
                Undo changes to this cue
              </button>
            )}
          </div>
        ) : isEdited ? (
          <>
            <span className="edited-text">{edited!.text}</span>
            <span className="edited-note">
              <span className="chip accent">edited</span> automatic: {cue.text.replace(/\n/g, " ")}
            </span>
          </>
        ) : (
          <Tokens tokens={row.new} />
        )}
      </div>
      <div className={`c-cps${rate > maxCps ? " bad" : ""}`} title={`${rate.toFixed(1)} characters per second (limit ${maxCps})`}>
        {rate.toFixed(1)}
      </div>
    </div>
  );
});

export default function DiffTable({ jobId, result, time, playing, onSeek, onSaved }: Props) {
  const toast = useToast();
  const [filter, setFilter] = useState<RowFilter>("all");
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const [editing, setEditing] = useState(false);
  const [drafts, setDrafts] = useState<Record<number, Draft>>({});
  const [saving, setSaving] = useState(false);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [speaker, setSpeaker] = useState("");
  const list = useRef<HTMLDivElement>(null);
  const colors = useMemo(() => speakerColors(result), [result]);
  const speakerList = result.speakers?.speakers.map((s) => s.speaker) ?? [];
  const hasOverlap = result.rows.some((r) => r.overlap);

  const cues = result.cues;
  const edited = result.edited_cues && result.edited_cues.length === cues.length ? result.edited_cues : null;
  const timeline = edited ?? cues;
  const active = lastCueBefore(timeline, time);
  const maxCps = result.limits.max_cps;
  const beforeLabel = result.mode === "A" ? "Original" : "Whisper";

  const counts = useMemo(() => {
    const c: Record<RowFilter, number> = { all: 0, text: 0, timing: 0, cps: 0, edited: 0, overlap: 0 };
    result.rows.forEach((r, i) => {
      (Object.keys(c) as RowFilter[]).forEach((f) => {
        if (rowMatches(r, f, "", edited?.[i], cues[i])) c[f] += 1;
      });
    });
    return c;
  }, [result.rows, edited, cues]);

  const visible = useMemo(
    () =>
      result.rows
        .map((_, i) => i)
        .filter((i) => rowMatches(result.rows[i], filter, query, edited?.[i], cues[i]))
        .filter((i) => !speaker || (result.rows[i].speakers ?? []).includes(speaker)),
    [result.rows, filter, query, edited, cues, speaker],
  );

  // keep the playing cue in view
  useEffect(() => {
    if (!follow || !playing || editing || active < 0) return;
    const box = list.current;
    const el = box?.querySelector<HTMLElement>(`[data-row="${active}"]`);
    if (!box || !el || box.scrollHeight <= box.clientHeight) return; // only scroll the list, never the page
    const head = 40;
    if (el.offsetTop < box.scrollTop + head || el.offsetTop + el.offsetHeight > box.scrollTop + box.clientHeight) {
      box.scrollTo({ top: Math.max(0, el.offsetTop - head - box.clientHeight / 4), behavior: "smooth" });
    }
  }, [active, follow, playing, editing]);

  const onDraft = useMemo(
    () => (i: number, d: Draft | undefined) =>
      setDrafts((old) => {
        const next = { ...old };
        if (d) next[i] = d;
        else delete next[i];
        return next;
      }),
    [],
  );

  const dirty = Object.keys(drafts).length;
  const badDraft = Object.values(drafts).some((d) => !d.text.trim() || d.end <= d.start);

  const save = async () => {
    setSaving(true);
    try {
      const base = edited ?? cues;
      const out = base.map((c, i) => drafts[i] ?? { start: c.start, end: c.end, text: c.text });
      await api.saveCues(jobId, out);
      setDrafts({});
      setEditing(false);
      toast("Edits saved. Downloads now include them.");
      onSaved();
    } catch (e) {
      toast((e as Error).message, "error");
    } finally {
      setSaving(false);
    }
  };

  const revertAll = async () => {
    if (!window.confirm("Discard all saved manual edits and go back to the automatic subtitles?")) return;
    try {
      await api.dropEdits(jobId);
      toast("Manual edits removed");
      onSaved();
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  const FILTERS: [RowFilter, string][] = [
    ["all", "All"],
    ["text", "Text changed"],
    ["timing", "Re-timed"],
    ["cps", "Too fast"],
    ...(hasOverlap ? ([["overlap", "Overlap"]] as [RowFilter, string][]) : []),
    ...(edited ? ([["edited", "Edited"]] as [RowFilter, string][]) : []),
  ];

  return (
    <div
      onMouseOver={(e) => {
        const el = (e.target as HTMLElement).closest<HTMLElement>("[data-tip]");
        if (el) {
          const r = el.getBoundingClientRect();
          setTip({ x: Math.min(r.left, window.innerWidth - 360), y: r.bottom + 6, text: el.dataset.tip! });
        } else if (tip) setTip(null);
      }}
      onMouseLeave={() => setTip(null)}
    >
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Filter cues">
          {FILTERS.map(([f, label]) => (
            <button key={f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>
              {label}
              <span className="count">{counts[f]}</span>
            </button>
          ))}
        </div>
        <input className="input sm search" type="search" placeholder="Search text…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search cues" />
        {speakerList.length > 0 && (
          <select className="input sm" style={{ width: "auto" }} value={speaker} onChange={(e) => setSpeaker(e.target.value)} aria-label="Show one speaker">
            <option value="">All speakers</option>
            {speakerList.map((s) => (
              <option key={s} value={s}>
                {speakerName(s)}
              </option>
            ))}
          </select>
        )}
        <span style={{ flex: 1 }} />
        <label className="toggle" style={{ padding: 0 }} title="Scroll to the cue being played">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          <span className="sw" />
          <span className="small">Follow playback</span>
        </label>
        {!editing ? (
          <button className="btn sm" onClick={() => setEditing(true)}>
            <IconEdit /> Edit cues
          </button>
        ) : (
          <button
            className="btn sm"
            onClick={() => {
              if (dirty && !window.confirm("Discard unsaved changes?")) return;
              setDrafts({});
              setEditing(false);
            }}
          >
            Done editing
          </button>
        )}
      </div>
      <div className="legend-row">
        <span><span className="tk sub">replaced</span></span>
        <span><span className="tk ins">inserted</span></span>
        <span><span className="tk del">deleted</span></span>
        <span><span className="tk fmt">punctuation / case</span></span>
        <span style={{ color: "var(--moved)", fontWeight: 600 }}>re-timed</span>
        <span>Hover a highlighted word for the evidence. Click a cue to play it.</span>
        {edited && (
          <button className="btn ghost sm" style={{ marginLeft: "auto" }} onClick={revertAll}>
            Revert manual edits
          </button>
        )}
      </div>
      <div className="cue-list" ref={list}>
        <div className="cue cue-head">
          <div>#</div>
          <div>Time</div>
          <div>{beforeLabel}</div>
          <div>Corrected</div>
          <div style={{ textAlign: "right" }}>CPS</div>
        </div>
        {visible.length === 0 && <div className="empty">No cues match.</div>}
        {visible.map((i) => (
          <Row
            key={i}
            i={i}
            row={result.rows[i]}
            cue={cues[i]}
            edited={edited?.[i]}
            active={i === active}
            editing={editing}
            draft={drafts[i]}
            maxCps={maxCps}
            countSpaces={result.limits.cps_count_spaces}
            onSeek={onSeek}
            onDraft={onDraft}
            colors={colors}
          />
        ))}
        {editing && (
          <div className="save-bar">
            <span className="small">{dirty ? `${dirty} cue${dirty > 1 ? "s" : ""} changed` : "Change the text or times, then save."}</span>
            {badDraft && <span className="small" style={{ color: "var(--danger)" }}>A cue is empty or ends before it starts.</span>}
            <span style={{ flex: 1 }} />
            <button className="btn sm" disabled={!dirty || saving} onClick={() => setDrafts({})}>
              Discard
            </button>
            <button className="btn primary sm" disabled={!dirty || saving || badDraft} onClick={save}>
              {saving ? "Saving…" : "Save edits"}
            </button>
          </div>
        )}
      </div>
      {tip && (
        <div className="tooltip" style={{ left: Math.max(8, tip.x), top: tip.y }}>
          {tipLines(tip.text)}
        </div>
      )}
    </div>
  );
}
