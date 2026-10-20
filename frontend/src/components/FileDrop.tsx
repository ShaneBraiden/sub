import { useState, type ReactNode } from "react";
import { formatBytes } from "../lib/format";
import { extension } from "../lib/cues";
import { IconX } from "./Icons";

interface Props {
  title: string;
  hint: string;
  accept: string[];
  file: File | null;
  onFile: (f: File | null) => void;
  onReject: (message: string) => void;
  icon: ReactNode;
  optional?: boolean;
  maxMb?: number;
}

export default function FileDrop({ title, hint, accept, file, onFile, onReject, icon, optional, maxMb }: Props) {
  const [over, setOver] = useState(false);

  const take = (f: File | undefined | null) => {
    if (!f) return;
    const ext = extension(f.name);
    if (!accept.includes(ext)) {
      onReject(`${f.name}: unsupported file type. Use ${accept.join(", ")}.`);
      return;
    }
    if (maxMb && f.size > maxMb * 1024 * 1024) {
      onReject(`${f.name} is larger than ${maxMb} MB.`);
      return;
    }
    onFile(f);
  };

  return (
    <div
      className={`drop${over ? " over" : ""}${file ? " filled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        take(e.dataTransfer.files?.[0]);
      }}
    >
      <input
        type="file"
        accept={accept.join(",")}
        aria-label={title}
        onChange={(e) => {
          take(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <div className="ico">{icon}</div>
      <div className="d-text">
        <div className="d-title">
          {title}
          {optional && <span className="badge-opt">optional</span>}
        </div>
        <div className="d-sub">{file ? `${file.name} · ${formatBytes(file.size)}` : hint}</div>
      </div>
      {file && (
        <button
          type="button"
          className="btn ghost icon sm clear"
          onClick={(e) => {
            e.stopPropagation();
            onFile(null);
          }}
          aria-label={`Remove ${file.name}`}
          title="Remove"
        >
          <IconX />
        </button>
      )}
    </div>
  );
}
