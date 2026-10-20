import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Job, OptionsInfo, SystemInfo } from "../api/types";

const TERMINAL = new Set(["done", "failed", "cancelled"]);
export const isTerminal = (status: string) => TERMINAL.has(status);

let optionsPromise: Promise<OptionsInfo> | null = null;

/** Option defaults/ranges and the pipeline stage list (fetched once per page load). */
export function useOptionsInfo(): { info: OptionsInfo | null; error: string | null } {
  const [info, setInfo] = useState<OptionsInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    optionsPromise ??= api.options();
    optionsPromise
      .then((i) => alive && setInfo(i))
      .catch((e: Error) => {
        optionsPromise = null;
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, []);
  return { info, error };
}

/** Server status, refreshed every few seconds while the tab is visible. */
export function useSystem(intervalMs = 5000): { system: SystemInfo | null; offline: boolean } {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [offline, setOffline] = useState(false);
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const tick = async () => {
      if (document.visibilityState === "visible") {
        try {
          const s = await api.system();
          if (alive) {
            setSystem(s);
            setOffline(false);
          }
        } catch {
          if (alive) setOffline(true);
        }
      }
      if (alive) timer = window.setTimeout(tick, intervalMs);
    };
    tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [intervalMs]);
  return { system, offline };
}

/** Live job snapshot: Server-Sent Events while the job is active, with a polling fallback. */
export function useJob(id: string): { job: Job | null; error: ApiError | null; refresh: () => void } {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [nonce, setNonce] = useState(0);
  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  const jobRef = useRef<Job | null>(null);
  jobRef.current = job;

  useEffect(() => {
    let alive = true;
    let source: EventSource | null = null;
    let poll = 0;
    setJob((j) => (j && j.id !== id ? null : j)); // no stale job while another one loads

    const startPolling = () => {
      const tick = async () => {
        try {
          const j = await api.job(id);
          if (!alive) return;
          setJob(j);
          if (!isTerminal(j.status)) poll = window.setTimeout(tick, 1500);
        } catch (e) {
          if (alive) setError(e as ApiError);
        }
      };
      poll = window.setTimeout(tick, 1500);
    };

    api
      .job(id)
      .then((j) => {
        if (!alive) return;
        setJob(j);
        setError(null);
        if (isTerminal(j.status)) return;
        if (typeof EventSource === "undefined") return startPolling();
        source = new EventSource(api.eventsUrl(id));
        source.onmessage = (ev) => {
          const snap = JSON.parse(ev.data) as Job;
          setJob(snap);
          if (isTerminal(snap.status)) source?.close();
        };
        source.addEventListener("deleted", () => {
          source?.close();
          setError(new ApiError(404, "This job was deleted."));
        });
        source.onerror = () => {
          // the stream closes normally after a terminal event; otherwise fall back to polling
          source?.close();
          if (alive && !(jobRef.current && isTerminal(jobRef.current.status))) startPolling();
        };
      })
      .catch((e: ApiError) => alive && setError(e));

    return () => {
      alive = false;
      source?.close();
      window.clearTimeout(poll);
    };
  }, [id, nonce]);

  return { job, error, refresh };
}

/** localStorage-backed state for small per-browser preferences (never required to work). */
export function useStoredState<T>(key: string, initial: T): [T, (v: T | ((old: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw == null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });
  const set = useCallback(
    (v: T | ((old: T) => T)) => {
      setValue((old) => {
        const nv = typeof v === "function" ? (v as (o: T) => T)(old) : v;
        try {
          localStorage.setItem(key, JSON.stringify(nv));
        } catch {
          /* storage unavailable */
        }
        return nv;
      });
    },
    [key],
  );
  return [value, set];
}
