import type { PipelineLogEntry } from "../../types/api";

interface LogViewerProps {
  entries: PipelineLogEntry[];
}

function levelClass(level: PipelineLogEntry["level"]): string {
  if (level === "error") {
    return "text-rose-700";
  }
  if (level === "warn") {
    return "text-amber-700";
  }
  return "text-slate-700";
}

export function LogViewer({ entries }: LogViewerProps) {
  if (!entries.length) {
    return <p className="text-sm text-slate-500">No pipeline events yet.</p>;
  }

  return (
    <ul className="max-h-64 space-y-2 overflow-auto pr-1">
      {entries.map((entry) => (
        <li key={entry.id} className="rounded-lg border border-slate-200 bg-slate-50 p-2">
          <p className={`text-sm ${levelClass(entry.level)}`}>{entry.message}</p>
          <p className="mono mt-1 text-xs text-slate-500">
            {new Date(entry.createdAt).toLocaleTimeString()}
          </p>
        </li>
      ))}
    </ul>
  );
}
