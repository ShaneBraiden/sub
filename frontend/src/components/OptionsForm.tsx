import type { JobOptions, Mode, OptionField, OptionsInfo } from "../api/types";
import { ASR_MODEL_LABEL, CHAPTER_MODE_LABEL, SEGMENTATION_LABEL, SPEAKER_LABEL_STYLE } from "../lib/labels";

interface Props {
  info: OptionsInfo;
  value: JobOptions;
  onChange: (v: JobOptions) => void;
  mode: Mode;
}

type BoolKey = "fusion" | "sync" | "punctuation" | "truecase" | "spelling" | "grammar_rules" | "grammar_model";

const TOGGLES: { key: BoolKey; title: string; text: string; modeA?: boolean }[] = [
  { key: "sync", title: "Fix overall timing", text: "Constant offset, frame-rate drift and ad-break jumps.", modeA: true },
  { key: "fusion", title: "Compare with the audio", text: "Replace, insert or delete words the audio contradicts.", modeA: true },
  { key: "punctuation", title: "Punctuation", text: "Restore missing full stops, commas and question marks." },
  { key: "truecase", title: "Capitalisation", text: "Sentence starts, “I”, and casing taken from the audio." },
  { key: "spelling", title: "Spelling", text: "Dictionary-based fixes for misspelt words." },
  { key: "grammar_rules", title: "Grammar rules", text: "LanguageTool rules such as a/an and agreement." },
  { key: "grammar_model", title: "Grammar model", text: "Sentence-level correction, guarded against over-editing." },
];

/** Options the user changed relative to the server defaults (+ terms, always sent). */
export function changedOptions(value: JobOptions, defaults: JobOptions): JobOptions {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    if (k === "terms" || !(k in defaults)) continue; // options this server does not offer are dropped
    if (v !== undefined && v !== (defaults as Record<string, unknown>)[k]) out[k] = v;
  }
  return out as JobOptions;
}

function field(info: OptionsInfo, key: keyof JobOptions): OptionField | undefined {
  return info.fields.find((f) => f.key === key);
}

function NumberRow({
  info, value, onChange, k, label, step, unit, slider = false,
}: {
  info: OptionsInfo;
  value: JobOptions;
  onChange: (v: JobOptions) => void;
  k: "max_cps" | "max_chars" | "min_duration" | "max_duration" | "min_chapter";
  label: string;
  step: number;
  unit?: string;
  slider?: boolean;
}) {
  const f = field(info, k);
  const current = (value[k] ?? info.defaults[k]) as number | undefined;
  if (current === undefined || !f) return null;
  const set = (n: number) => {
    if (Number.isNaN(n)) return;
    const clamped = Math.min(f.max ?? n, Math.max(f.min ?? n, n));
    onChange({ ...value, [k]: clamped });
  };
  const id = `opt-${k}`;
  return (
    <label className="field" htmlFor={id}>
      <span>
        {label}
        {unit && <span className="muted"> ({unit})</span>}
      </span>
      {slider ? (
        <div className="range-row">
          <input type="range" min={f.min} max={f.max} step={step} value={current} onChange={(e) => set(parseFloat(e.target.value))} aria-label={label} />
          <input id={id} className="input sm num" type="number" min={f.min} max={f.max} step={step} value={current} onChange={(e) => set(parseFloat(e.target.value))} />
        </div>
      ) : (
        <input id={id} className="input sm num" type="number" min={f.min} max={f.max} step={step} value={current} onChange={(e) => set(parseFloat(e.target.value))} />
      )}
    </label>
  );
}

export default function OptionsForm({ info, value, onChange, mode }: Props) {
  const d = info.defaults;
  const v = { ...d, ...value };
  const has = (k: keyof JobOptions) => d[k] !== undefined;
  const segChoices = (field(info, "segmentation")?.choices ?? ["dp", "greedy"]).filter((c) => c !== "keep" || mode === "A");
  const asrChoices = field(info, "asr_model")?.choices ?? ["auto"];

  return (
    <div className="opt-groups">
      <section className="opt-group" aria-label="Subtitle format">
        <h3>Subtitle format</h3>
        <NumberRow info={info} value={value} onChange={onChange} k="max_cps" label="Reading speed limit" unit="characters / s" step={0.5} slider />
        <NumberRow info={info} value={value} onChange={onChange} k="max_chars" label="Characters per line" step={1} slider />
        {has("max_lines") && (
          <label className="field">
            <span>Lines per cue</span>
            <select className="input sm" value={v.max_lines} onChange={(e) => onChange({ ...value, max_lines: parseInt(e.target.value, 10) })}>
              {[1, 2, 3].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
          <NumberRow info={info} value={value} onChange={onChange} k="min_duration" label="Shortest cue" unit="s" step={0.1} />
          <NumberRow info={info} value={value} onChange={onChange} k="max_duration" label="Longest cue" unit="s" step={0.5} />
        </div>
        {has("segmentation") && (
          <label className="field">
            <span>Cue splitting</span>
            <select
              className="input sm"
              value={v.segmentation === "keep" && mode !== "A" ? "dp" : v.segmentation}
              onChange={(e) => onChange({ ...value, segmentation: e.target.value as JobOptions["segmentation"] })}
            >
              {segChoices.map((c) => (
                <option key={c} value={c}>
                  {SEGMENTATION_LABEL[c] ?? c}
                </option>
              ))}
            </select>
          </label>
        )}
      </section>

      <section className="opt-group" aria-label="Text correction">
        <h3>Correction steps</h3>
        <div>
          {TOGGLES.filter((t) => has(t.key)).map((t) => {
            const disabled = !!t.modeA && mode !== "A";
            return (
              <label key={t.key} className={`toggle${disabled ? " disabled" : ""}`} title={disabled ? "Needs a subtitle file" : undefined}>
                <input
                  type="checkbox"
                  checked={!!v[t.key] && !disabled}
                  disabled={disabled}
                  onChange={(e) => onChange({ ...value, [t.key]: e.target.checked })}
                />
                <span className="sw" />
                <span className="t-text">
                  <b>{t.title}</b>
                  <small>{disabled ? "Only with a subtitle file." : t.text}</small>
                </span>
              </label>
            );
          })}
        </div>
      </section>

      <section className="opt-group" aria-label="Audio and extras">
        <h3>Audio & extras</h3>
        {has("asr_model") && (
          <label className="field">
            <span>Speech recognition model</span>
            <select className="input sm" value={v.asr_model} onChange={(e) => onChange({ ...value, asr_model: e.target.value })}>
              {asrChoices.map((c) => (
                <option key={c} value={c}>
                  {ASR_MODEL_LABEL[c] ?? c}
                </option>
              ))}
            </select>
            <span className="hint">Whisper. Automatic picks large-v3-turbo on a GPU and small.en on a CPU.</span>
          </label>
        )}
        {has("timing") && (
          <label className="field">
            <span>Word timing</span>
            <select className="input sm" value={v.timing} onChange={(e) => onChange({ ...value, timing: e.target.value as JobOptions["timing"] })}>
              <option value="align">Forced alignment (most accurate)</option>
              <option value="whisper">Whisper timestamps (faster)</option>
            </select>
            <span className="hint">
              {v.timing === "whisper"
                ? "Uses Whisper's own word times; words that were corrected get interpolated times."
                : "Times every word of the corrected text against the audio."}
            </span>
          </label>
        )}
        {has("diarize") && (
          <div className="opt-sub" aria-label="Speakers">
            <label className="toggle">
              <input type="checkbox" checked={!!v.diarize} onChange={(e) => onChange({ ...value, diarize: e.target.checked })} />
              <span className="sw" />
              <span className="t-text">
                <b>Identify speakers</b>
                <small>Label who is speaking (pyannote). Cues never mix speakers, except two-line dialogue cues.</small>
              </span>
            </label>
            {v.diarize && info.diarization && !(info.diarization.installed && info.diarization.token) && (
              <p className="hint warn-text" role="status">
                {!info.diarization.installed
                  ? "pyannote.audio is not installed on the server (pip install -r requirements-diarize.txt)."
                  : "No Hugging Face token found on the server: set HF_TOKEN after accepting the model's conditions (see the README)."}
              </p>
            )}
            {v.diarize && has("num_speakers") && (
              <label className="field">
                <span>Number of speakers</span>
                <select
                  className="input sm"
                  value={v.num_speakers ?? 0}
                  onChange={(e) => onChange({ ...value, num_speakers: parseInt(e.target.value, 10) })}
                >
                  <option value={0}>Detect automatically</option>
                  {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
                <span className="hint">Setting the exact number, when you know it, avoids merged or split speakers.</span>
              </label>
            )}
            {v.diarize && has("speaker_labels") && (
              <label className="field">
                <span>Speaker names in the subtitles</span>
                <select
                  className="input sm"
                  value={v.speaker_labels}
                  onChange={(e) => onChange({ ...value, speaker_labels: e.target.value as JobOptions["speaker_labels"] })}
                >
                  {(field(info, "speaker_labels")?.choices ?? ["every", "change", "none"]).map((c) => (
                    <option key={c} value={c}>
                      {SPEAKER_LABEL_STYLE[c] ?? c}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
        )}
        {has("chapters") && (
          <div className="opt-sub" aria-label="Chapters">
            <label className="toggle">
              <input type="checkbox" checked={!!v.chapters} onChange={(e) => onChange({ ...value, chapters: e.target.checked })} />
              <span className="sw" />
              <span className="t-text">
                <b>Chapter markers</b>
                <small>Split the recording where the topic changes and name each part (WebVTT chapters, FFmpeg, timestamp list).</small>
              </span>
            </label>
            {v.chapters && has("chapter_mode") && (
              <label className="field">
                <span>Kind of recording</span>
                <select
                  className="input sm"
                  value={v.chapter_mode ?? "talk"}
                  onChange={(e) => onChange({ ...value, chapter_mode: e.target.value as JobOptions["chapter_mode"] })}
                >
                  {(field(info, "chapter_mode")?.choices ?? ["talk", "meeting"]).map((c) => (
                    <option key={c} value={c}>
                      {CHAPTER_MODE_LABEL[c] ?? c}
                    </option>
                  ))}
                </select>
                <span className="hint">
                  {v.chapter_mode === "meeting"
                    ? "Also uses pauses and words like “okay” and “so” that open a new topic, and makes more, shorter chapters."
                    : "Chapters follow changes in vocabulary: best with one main speaker."}
                </span>
              </label>
            )}
            {v.chapters && has("num_chapters") && (
              <label className="field">
                <span>Number of chapters</span>
                <select
                  className="input sm"
                  value={v.num_chapters ?? 0}
                  onChange={(e) => onChange({ ...value, num_chapters: parseInt(e.target.value, 10) })}
                >
                  <option value={0}>Choose automatically</option>
                  {Array.from({ length: 19 }, (_, i) => i + 2).map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
                <span className="hint">Fewer are made if the recording is too short for that many.</span>
              </label>
            )}
            {v.chapters && <NumberRow info={info} value={value} onChange={onChange} k="min_chapter" label="Shortest chapter" unit="s" step={10} />}
          </div>
        )}
      </section>
    </div>
  );
}
