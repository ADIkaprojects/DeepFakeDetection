type StatusTone = "online" | "offline" | "degraded" | "idle";

interface StatusPillProps {
  label: string;
  tone: StatusTone;
}

const toneClassMap: Record<StatusTone, string> = {
  online: "bg-emerald-100 text-emerald-800",
  degraded: "bg-amber-100 text-amber-800",
  offline: "bg-rose-100 text-rose-800",
  idle: "bg-slate-100 text-slate-600",
};

export function StatusPill({ label, tone }: StatusPillProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold tracking-wide ${toneClassMap[tone]}`}
    >
      {label}
    </span>
  );
}
