interface SliderProps {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (value: number) => void;
}

export function Slider({ label, min, max, step = 0.01, value, onChange }: SliderProps) {
  return (
    <label className="flex flex-col gap-2 text-sm text-slate-700">
      <span className="font-medium text-slate-900">{label}</span>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 accent-cyan-600"
        />
        <span className="w-14 rounded-md bg-slate-100 px-2 py-1 text-center font-mono text-xs text-slate-700">
          {value.toFixed(2)}
        </span>
      </div>
    </label>
  );
}
