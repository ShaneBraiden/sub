import type { JobStatus } from "../api/types";
import { STATUS_LABEL } from "../lib/labels";

export default function StatusBadge({ status }: { status: JobStatus }) {
  return <span className={`status ${status}`}>{STATUS_LABEL[status]}</span>;
}
