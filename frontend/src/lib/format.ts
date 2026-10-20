/** 75.25 -> "01:15.250" (or "1:01:15.250" past an hour); null -> "–". */
export function formatTime(t: number | null | undefined, ms = true): string {
  if (t == null || !Number.isFinite(t)) return "–";
  const total = Math.max(0, t);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const sec = ms ? s.toFixed(3).padStart(6, "0") : String(Math.floor(s)).padStart(2, "0");
  const mm = String(m).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${sec}` : `${mm}:${sec}`;
}

/** Player clock: 75.2 -> "1:15". */
export function formatClock(t: number): string {
  const total = Math.max(0, Math.floor(t || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

/** 252 -> "4 min 12 s", 8.4 -> "8.4 s". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return s ? `${m} min ${s} s` : `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}

/** Signed seconds with a unit: 0.25 -> "+250 ms", -2.5 -> "−2.50 s". */
export function formatShift(seconds: number): string {
  const sign = seconds < 0 ? "−" : "+";
  const a = Math.abs(seconds);
  return a < 1 ? `${sign}${Math.round(a * 1000)} ms` : `${sign}${a.toFixed(2)} s`;
}

export function formatPercent(x: number | null | undefined, digits = 0): string {
  if (x == null || !Number.isFinite(x)) return "–";
  return `${(x * 100).toFixed(digits)}%`;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[u]}`;
}

/** Epoch seconds -> "just now", "5 min ago", "yesterday 14:02", "12 Oct 14:02". */
export function formatRelative(epoch: number, now = Date.now() / 1000): string {
  const d = now - epoch;
  if (d < 45) return "just now";
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  const date = new Date(epoch * 1000);
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const today = new Date(now * 1000);
  const yesterday = new Date((now - 86400) * 1000);
  if (date.toDateString() === today.toDateString()) return `today ${time}`;
  if (date.toDateString() === yesterday.toDateString()) return `yesterday ${time}`;
  return `${date.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}
