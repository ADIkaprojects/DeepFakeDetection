interface ToggleProps {
  enabled: boolean;
  onChange: (next: boolean) => void;
  label: string;
  helpText?: string;
}

export function Toggle({ enabled, onChange, label, helpText }: ToggleProps) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
      <div>
        <p className="text-sm font-medium text-slate-900">{label}</p>
        {helpText ? <p className="text-xs text-slate-600">{helpText}</p> : null}
      </div>
      <button
        type="button"
        aria-label={label}
        title={label}
        onClick={() => onChange(!enabled)}
        className={`relative h-6 w-12 rounded-full transition ${enabled ? "bg-cyan-600" : "bg-slate-300"}`}
      >
        <span className="sr-only">{label}</span>
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition ${enabled ? "left-6" : "left-0.5"}`}
        />
      </button>
    </div>
  );
}
