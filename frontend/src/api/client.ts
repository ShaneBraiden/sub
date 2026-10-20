import type { CueData, Job, JobOptions, OptionsInfo, Result, SystemInfo, Waveform } from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function detail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => (x && typeof x === "object" && "msg" in x ? String(x.msg) : String(x))).join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "Cannot reach the server. Is `python -m app.backend` running?");
  }
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail(body, `${res.status} ${res.statusText}`));
  }
  if (res.status === 204) return undefined as T;
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  system: () => request<SystemInfo>("/api/system"),
  options: () => request<OptionsInfo>("/api/options"),
  jobs: (limit = 200) => request<{ jobs: Job[]; total: number }>(`/api/jobs?limit=${limit}`),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  result: (id: string) => request<Result>(`/api/jobs/${id}/result`),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  rerun: (id: string, options: JobOptions) => request<Job>(`/api/jobs/${id}/rerun`, json("POST", options)),
  remove: (id: string) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),
  saveCues: (id: string, cues: Pick<CueData, "start" | "end" | "text">[]) =>
    request<{ cues: number }>(`/api/jobs/${id}/cues`, json("PUT", { cues })),
  dropEdits: (id: string) => request<void>(`/api/jobs/${id}/cues`, { method: "DELETE" }),
  waveform: (id: string, bins = 2400) => request<Waveform>(`/api/jobs/${id}/waveform?bins=${bins}`),
  log: (id: string) => request<string>(`/api/jobs/${id}/log`),

  fileUrl: (id: string, kind: string, inline = false) => `/api/jobs/${id}/files/${kind}${inline ? "?inline=true" : ""}`,
  mediaUrl: (id: string) => `/api/jobs/${id}/media`,
  audioUrl: (id: string) => `/api/jobs/${id}/audio`,
  eventsUrl: (id: string) => `/api/jobs/${id}/events`,

  /** Multipart upload with progress (fetch has no upload progress events). */
  createJob(
    media: File,
    subs: File | null,
    options: JobOptions,
    onProgress?: (fraction: number) => void,
  ): { promise: Promise<Job>; abort: () => void } {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("media", media);
    if (subs) form.append("subs", subs);
    form.append("options", JSON.stringify(options));
    const promise = new Promise<Job>((resolve, reject) => {
      xhr.open("POST", "/api/jobs");
      xhr.responseType = "json";
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as Job);
        else reject(new ApiError(xhr.status, detail(xhr.response, `Upload failed (${xhr.status})`)));
      };
      xhr.onerror = () => reject(new ApiError(0, "Upload failed: the server could not be reached."));
      xhr.onabort = () => reject(new ApiError(0, "Upload cancelled."));
      xhr.send(form);
    });
    return { promise, abort: () => xhr.abort() };
  },
};
