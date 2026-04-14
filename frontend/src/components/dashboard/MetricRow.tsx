interface MetricRowProps {
  label: string;
  value: string;
}

export function MetricRow({ label, value }: MetricRowProps) {
  return (
    <div className="flex items-center justify-between border-b border-slate-100 py-2 text-sm last:border-b-0">
      <span className="text-slate-600">{label}</span>
      <span className="font-mono text-slate-900">{value}</span>
    </div>
  );
}
