import type { Summary, SummarySentence, SummaryVersion } from "../api/types";

export type SummaryKind = "abstractive" | "extractive";

/** The version shown first: the LLM's when the job made one, else the extractive one. */
export function defaultKind(summary: Summary): SummaryKind {
  return summary.abstractive ? "abstractive" : "extractive";
}

export function version(summary: Summary, kind: SummaryKind): SummaryVersion {
  return (kind === "abstractive" && summary.abstractive) || summary.extractive;
}

/** Quoted sentences of an extractive summary part, looked up by index. */
export function quoted(summary: Summary, indices: number[] | undefined): SummarySentence[] {
  if (!indices?.length) return [];
  const byIndex = new Map(summary.sentences.map((s) => [s.index, s]));
  return indices.map((i) => byIndex.get(i)).filter((s): s is SummarySentence => !!s);
}

/** Plain-text version for the clipboard: headline, overview and chapters. */
export function summaryText(summary: Summary, kind: SummaryKind, clock: (t: number) => string): string {
  const v = version(summary, kind);
  const out: string[] = [];
  if (v.headline.text) out.push(v.headline.text, "");
  if (v.overview.text) out.push(v.overview.text, "");
  for (const c of v.chapters) {
    out.push(`${clock(c.start)} ${c.title || `Part ${c.index}`}`);
    if (c.text) out.push(c.text);
    out.push("");
  }
  return out.join("\n").trim() + "\n";
}

/** Method name -> what people call it. */
export const SUMMARY_METHOD_LABEL: Record<string, string> = {
  embed: "sentence embeddings (LSA) + MMR",
  textrank: "TextRank",
  tfidf: "TF-IDF centroid + MMR",
};
