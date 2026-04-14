import type { PropsWithChildren, ReactNode } from "react";

interface PanelProps extends PropsWithChildren {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  className?: string;
}

export function Panel({ title, subtitle, actions, className = "", children }: PanelProps) {
  return (
    <section
      className={`rounded-2xl border border-slate-200/80 bg-white/80 p-5 shadow-sm backdrop-blur-sm ${className}`}
      aria-label={title}
    >
      <header className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="font-display text-lg font-semibold text-slate-900">{title}</h2>
          {subtitle ? <p className="mt-1 text-sm text-slate-600">{subtitle}</p> : null}
        </div>
        {actions ? <div>{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}
