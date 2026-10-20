import { useMemo, useState } from "react";
import type { Edit } from "../api/types";
import { formatTime } from "../lib/format";
import { EDIT_OP_LABEL, EDIT_STAGE_LABEL } from "../lib/labels";

const PAGE = 400;

function decisionOf(e: Edit): "applied" | "rejected" | "blocked" {
  if (e.op === "blocked") return "blocked";
  if (e.decision === "reject" || e.op === "guard") return "rejected";
  return "applied";
}

export default function ChangesTable({ edits, onSeek, mode }: { edits: Edit[]; onSeek: (t: number) => void; mode: "A" | "B" }) {
  const [stage, setStage] = useState<string>("all");
  const [showRejected, setShowRejected] = useState(false);
  const [limit, setLimit] = useState(PAGE);

  const stages = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of edits) if (showRejected || decisionOf(e) === "applied") m.set(e.stage, (m.get(e.stage) ?? 0) + 1);
    return [...m.entries()];
  }, [edits, showRejected]);

  const rows = useMemo(
    () => edits.filter((e) => (stage === "all" || e.stage === stage) && (showRejected || decisionOf(e) === "applied")),
    [edits, stage, showRejected],
  );
  const hidden = edits.filter((e) => decisionOf(e) !== "applied").length;

  return (
    <div>
      <div className="toolbar">
        <div className="seg" role="group" aria-label="Filter by step">
          <button className={stage === "all" ? "on" : ""} onClick={() => setStage("all")}>
            All<span className="count">{stages.reduce((n, [, c]) => n + c, 0)}</span>
          </button>
          {stages.map(([s, c]) => (
            <button key={s} className={stage === s ? "on" : ""} onClick={() => setStage(s)}>
              {EDIT_STAGE_LABEL[s] ?? s}
              <span className="count">{c}</span>
            </button>
          ))}
        </div>
        <span style={{ flex: 1 }} />
        {hidden > 0 && (
          <label className="toggle" style={{ padding: 0 }}>
            <input type="checkbox" checked={showRejected} onChange={(e) => setShowRejected(e.target.checked)} />
            <span className="sw" />
            <span className="small">Show {hidden} rejected / blocked</span>
          </label>
        )}
      </div>
      {rows.length === 0 ? (
        <div className="empty">
          {mode === "A" ? "No changes were needed in this step." : "No changes."}
        </div>
      ) : (
        <div className="table-wrap cue-list">
          <table className="data edit-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Change</th>
                <th>Result</th>
                <th className="hide-sm">Evidence</th>
                <th className="hide-sm">Time</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, limit).map((e, i) => {
                const d = decisionOf(e);
                const evidence = [
                  e.prob != null && `audio p=${Number(e.prob).toFixed(2)}`,
                  e.score != null && `score ${Number(e.score).toFixed(2)}`,
                  e.lm != null && e.lm !== 0 && `LM ${Number(e.lm) > 0 ? "+" : ""}${Number(e.lm).toFixed(2)}`,
                  e.rule && `rule ${e.rule}`,
                  e.label && `label “${e.label}”`,
                  e.rejected_sentences != null && `${e.rejected_sentences} sentence(s) kept unchanged`,
                ].filter(Boolean);
                return (
                  <tr key={i} className={d !== "applied" ? "rejected" : ""}>
                    <td className="nowrap">
                      <span className="chip">{EDIT_STAGE_LABEL[e.stage] ?? e.stage}</span>
                    </td>
                    <td>
                      <span className="small muted">{EDIT_OP_LABEL[e.op] ?? e.op}</span>{" "}
                      {e.old != null && <span className="old-word">{String(e.old)}</span>}
                      {e.old != null && e.new != null && <span className="arrow">→</span>}
                      {e.new != null && <span className="new-word">{String(e.new)}</span>}
                    </td>
                    <td className="nowrap">
                      <span className={`chip ${d === "applied" ? "ok" : d === "blocked" ? "warn" : ""}`}>
                        {d === "applied" ? "applied" : d === "blocked" ? "kept (audio)" : "kept original"}
                      </span>
                    </td>
                    <td className="hide-sm small muted">{evidence.join(" · ") || "–"}</td>
                    <td className="hide-sm nowrap">
                      {typeof e.time === "number" ? (
                        <button className="btn ghost sm mono" onClick={() => onSeek(Math.max(0, (e.time as number) - 0.5))}>
                          {formatTime(e.time)}
                        </button>
                      ) : (
                        <span className="muted">–</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {rows.length > limit && (
            <div className="row" style={{ justifyContent: "center", padding: 12 }}>
              <button className="btn sm" onClick={() => setLimit((l) => l + PAGE)}>
                Show more ({rows.length - limit} left)
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
