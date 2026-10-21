import type { Sentiment, SentimentLabel, SentimentUnit } from "../api/types";

export const SENTIMENT_LABEL: Record<SentimentLabel, string> = { pos: "positive", neu: "neutral", neg: "negative" };

/** +0.42 -> "+0.42", -0.1 -> "−0.10" (a real minus sign). */
export function formatScore(x: number): string {
  if (!Number.isFinite(x)) return "–";
  const v = Math.abs(x) < 0.005 ? 0 : x;
  return `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(2)}`;
}

/** A word for a mean score: the band edges match what a reader would call it. */
export function describeScore(x: number): string {
  const a = Math.abs(x);
  if (a < 0.05) return "neutral";
  const side = x > 0 ? "positive" : "negative";
  if (a < 0.15) return `slightly ${side}`;
  if (a < 0.35) return side;
  return `strongly ${side}`;
}

/** Index of the unit whose middle is nearest to time t (units are in time order), or -1. */
export function nearestUnit(units: SentimentUnit[], t: number): number {
  if (!units.length) return -1;
  let lo = 0;
  let hi = units.length - 1;
  const mid = (u: SentimentUnit) => (u.start + u.end) / 2;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (mid(units[m]) < t) lo = m + 1;
    else hi = m;
  }
  if (lo > 0 && Math.abs(mid(units[lo - 1]) - t) <= Math.abs(mid(units[lo]) - t)) return lo - 1;
  return lo;
}

/** The smoothed curve's value at time t (linear between grid points), or null where there is no speech. */
export function curveAt(s: Pick<Sentiment, "times" | "curve">, t: number): number | null {
  const { times, curve } = s;
  if (!times.length) return null;
  if (t <= times[0]) return curve[0];
  if (t >= times[times.length - 1]) return curve[curve.length - 1];
  let lo = 0;
  let hi = times.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (times[m] <= t) lo = m;
    else hi = m;
  }
  const a = curve[lo];
  const b = curve[hi];
  if (a == null || b == null) return a ?? b ?? null;
  return a + ((b - a) * (t - times[lo])) / Math.max(times[hi] - times[lo], 1e-9);
}

/**
 * SVG path of a curve with gaps: "M x y L x y ..." with a new "M" after every null.
 * x and y map time and score to pixels.
 */
export function curvePath(times: number[], values: (number | null)[], x: (t: number) => number, y: (v: number) => number): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < times.length; i++) {
    const v = values[i];
    if (v == null) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x(times[i]).toFixed(1)} ${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** Closed areas between the curve and the zero line, one per run without gaps (fill them, clipped to one side). */
export function areaPath(times: number[], values: (number | null)[], x: (t: number) => number, y: (v: number) => number): string {
  let d = "";
  let run: [number, number][] = [];
  const flush = () => {
    if (run.length > 1) {
      d += `M${x(run[0][0]).toFixed(1)} ${y(0).toFixed(1)}`;
      for (const [t, v] of run) d += `L${x(t).toFixed(1)} ${y(v).toFixed(1)}`;
      d += `L${x(run[run.length - 1][0]).toFixed(1)} ${y(0).toFixed(1)}Z`;
    }
    run = [];
  };
  for (let i = 0; i < times.length; i++) {
    const v = values[i];
    if (v == null) flush();
    else run.push([times[i], v]);
  }
  flush();
  return d;
}
